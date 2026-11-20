from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np
import torch

from core.capture import SA2Capture
from core.controller import SA2Controller
from core.recorder import _default_is_pressed
from models.speed_cnn import SpeedCNN


def load_policy(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[SpeedCNN, dict]:
    """Loads a SpeedCNN checkpoint and returns (model_in_eval_mode, config)."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=target_device, weights_only=True)
    config = ckpt["config"]

    model = SpeedCNN(**config).to(target_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Converts a BGR (H, W, 3) or grayscale (H, W) uint8 frame to float32 in [0.0, 1.0]."""
    if frame.ndim == 3 and frame.shape[-1] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")
    return gray.astype(np.float32) / 255.0


class FrameStackBuffer:
    """Maintains a rolling stack of the last N grayscale frames as (1, N, H, W)."""

    def __init__(self, stack_size: int = 4) -> None:
        if stack_size < 1:
            raise ValueError(f"stack_size must be >= 1, got {stack_size}")
        self.stack_size = stack_size
        self._frames: deque[np.ndarray] = deque(maxlen=stack_size)

    def reset(self) -> None:
        self._frames.clear()

    def push(self, frame: np.ndarray) -> torch.Tensor:
        gray = preprocess_frame(frame)
        if not self._frames:
            # Pad initial stack with the first frame so inference starts immediately
            for _ in range(self.stack_size):
                self._frames.append(gray)
        else:
            self._frames.append(gray)

        stacked = np.stack(list(self._frames), axis=0)  # (N, H, W)
        return torch.from_numpy(stacked).unsqueeze(0)   # (1, N, H, W)


class SA2Player:
    """Closed-loop real-time inference runner for Sonic Adventure 2."""

    def __init__(
        self,
        model: SpeedCNN,
        capture,
        controller,
        stack_size: int = 4,
        fps: float = 20.0,
        button_threshold: float = 0.5,
        pause_key: str = "f9",
        stop_key: str = "esc",
        dry_run: bool = False,
        device: torch.device | None = None,
        is_pressed: Callable[[str], bool] = _default_is_pressed,
    ) -> None:
        self.model = model
        self.capture = capture
        self.controller = controller
        self.buffer = FrameStackBuffer(stack_size=stack_size)
        self.frame_interval = 1.0 / max(fps, 1e-6)
        self.button_threshold = button_threshold
        self.pause_key = pause_key.lower()
        self.stop_key = stop_key.lower()
        self.dry_run = dry_run
        self.device = device or next(model.parameters()).device
        self._is_pressed = is_pressed

        self.paused = False
        self._prev_pause_down = False

    def step(self) -> np.ndarray:
        """Captures one frame, predicts an action vector, and applies it to the controller."""
        frame = self.capture.grab_frame()
        obs = self.buffer.push(frame).to(self.device)
        action = self.model.predict(obs)[0].cpu().numpy()

        lx, ly = float(action[0]), float(action[1])
        jump = bool(action[2] >= self.button_threshold)
        act = bool(action[3] >= self.button_threshold)
        cam_l = bool(action[4] >= self.button_threshold)
        cam_r = bool(action[5] >= self.button_threshold)

        if not self.dry_run:
            self.controller.set_state(
                lx=lx,
                ly=ly,
                jump=jump,
                action=act,
                cam_left=cam_l,
                cam_right=cam_r,
            )
        return action

    def run(self, max_steps: int | None = None, start_delay: float = 3.0) -> int:
        """Runs the live inference loop until stop_key is pressed or max_steps is reached."""
        if start_delay > 0:
            mode = "DRY-RUN (no keys pressed)" if self.dry_run else "LIVE CONTROL"
            print(
                f"Starting {mode} in {start_delay:.1f}s... Focus the SA2 window. "
                f"[{self.pause_key.upper()}] pause/resume | [{self.stop_key.upper()}] quit"
            )
            time.sleep(start_delay)

        steps = 0
        try:
            while max_steps is None or steps < max_steps:
                loop_start = time.perf_counter()

                if self._is_pressed(self.stop_key):
                    print("Stop key pressed. Exiting...")
                    break

                pause_down = self._is_pressed(self.pause_key)
                if pause_down and not self._prev_pause_down:
                    self.paused = not self.paused
                    if self.paused:
                        self.controller.reset()
                        self.buffer.reset()
                        print("Paused. Keys released.")
                    else:
                        print("Resumed.")
                self._prev_pause_down = pause_down

                if not self.paused:
                    action = self.step()
                    steps += 1
                    if steps % 20 == 0 or self.dry_run:
                        print(
                            f"step={steps:4d} | lx={action[0]:+.2f} ly={action[1]:+.2f} | "
                            f"J={action[2]:.2f} A={action[3]:.2f} "
                            f"CL={action[4]:.2f} CR={action[5]:.2f}",
                            end="\r",
                        )

                elapsed = time.perf_counter() - loop_start
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.controller.reset()
            print(f"\nStopped after {steps} steps. All keys released.")
        return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained SpeedCNN agent live in SA2.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/speed_cnn_best.pt",
        help="Path to trained .pt checkpoint",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Target inference FPS")
    parser.add_argument(
        "--button-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for button presses",
    )
    parser.add_argument("--pause-key", default="f9", help="Key to pause/resume control")
    parser.add_argument("--stop-key", default="esc", help="Kill-switch key to stop and exit")
    parser.add_argument("--delay", type=float, default=3.0, help="Startup countdown in seconds")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run live capture and inference without sending keyboard inputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, config = load_policy(args.checkpoint)
    capture = SA2Capture()
    try:
        with SA2Controller() as controller:
            player = SA2Player(
                model=model,
                capture=capture,
                controller=controller,
                stack_size=config["in_channels"],
                fps=args.fps,
                button_threshold=args.button_threshold,
                pause_key=args.pause_key,
                stop_key=args.stop_key,
                dry_run=args.dry_run,
            )
            player.run(start_delay=args.delay)
    finally:
        capture.close()


if __name__ == "__main__":
    main()