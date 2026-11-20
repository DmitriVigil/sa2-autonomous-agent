import argparse
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from core.controller import SA2Controller
from core.recorder import _default_is_pressed


def render_input_overlay(
    frame: np.ndarray,
    action: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    scale: int = 4,
) -> np.ndarray:
    """Upscales a recorded frame and draws the paired action vector as a HUD."""
    h, w = frame.shape[:2]
    canvas = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    lx, ly, jump, act, cam_l, cam_r = [float(v) for v in action]

    # Semi-transparent HUD backdrop at bottom-left
    overlay = canvas.copy()
    cv2.rectangle(overlay, (10, canvas.shape[0] - 95), (305, canvas.shape[0] - 10), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    # Stick indicator (circle + dot)
    center = (55, canvas.shape[0] - 52)
    radius = 26
    cv2.circle(canvas, center, radius, (180, 180, 180), 1)
    dot_x = int(center[0] + lx * (radius - 5))
    dot_y = int(center[1] - ly * (radius - 5))  # Screen Y is inverted vs. ly
    cv2.circle(canvas, (dot_x, dot_y), 6, (0, 255, 255), -1)

    # Button indicator boxes
    buttons = [
        ("JUMP", jump > 0.5, (105, canvas.shape[0] - 80)),
        ("ACT", act > 0.5, (195, canvas.shape[0] - 80)),
        ("CAM L", cam_l > 0.5, (105, canvas.shape[0] - 42)),
        ("CAM R", cam_r > 0.5, (195, canvas.shape[0] - 42)),
    ]
    for label, active, (bx, by) in buttons:
        color = (0, 220, 90) if active else (70, 70, 70)
        text_color = (0, 0, 0) if active else (210, 210, 210)
        cv2.rectangle(canvas, (bx, by), (bx + 80, by + 26), color, -1)
        cv2.putText(
            canvas, label, (bx + 10, by + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA
        )

    # Top-left frame & FPS counter
    status_text = f"Frame {frame_idx + 1}/{total_frames} | {fps:4.1f} FPS"
    cv2.putText(
        canvas, status_text, (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
    )
    return canvas


class SA2Playback:
    """Loads a recorded .npz run and replays it visually or live via SA2Controller."""

    def __init__(
        self,
        dataset_path: str | Path,
        target_fps: int = 20,
        controller: SA2Controller | None = None,
        is_pressed_fn: Callable[[str], bool] | None = None,
        clock_fn: Callable[[], float] = time.perf_counter,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        self.target_fps = target_fps
        self.frame_duration = 1.0 / target_fps
        self._controller = controller
        self._is_pressed = is_pressed_fn if is_pressed_fn is not None else _default_is_pressed
        self._clock = clock_fn
        self._sleep = sleep_fn

        with np.load(self.dataset_path) as data:
            if "frames" not in data or "actions" not in data:
                raise ValueError("Dataset .npz must contain both 'frames' and 'actions' arrays.")
            self.frames = data["frames"]
            self.actions = data["actions"]

        if len(self.frames) != len(self.actions):
            raise ValueError(
                f"Mismatched dataset lengths: {len(self.frames)} frames vs {len(self.actions)} actions."
            )
        if len(self.frames) == 0:
            raise ValueError("Dataset is empty.")

    def _wait_for_tick(self, start_time: float, frame_idx: int) -> None:
        """Drift-compensated frame pacing so long runs don't accumulate lag."""
        target_time = start_time + (frame_idx + 1) * self.frame_duration
        remaining = target_time - self._clock()
        if remaining > 0:
            self._sleep(remaining)

    def play_visual(self, scale: int = 4) -> int:
        """Plays back recorded frames + input HUD in an OpenCV window. Returns frames shown."""
        window_name = f"SA2 Playback - {self.dataset_path.name}"
        total = len(self.frames)
        start_time = self._clock()
        last_tick = start_time
        shown = 0

        try:
            for i in range(total):
                now = self._clock()
                dt = max(now - last_tick, 1e-6)
                inst_fps = 1.0 / dt if i > 0 else float(self.target_fps)
                last_tick = now

                hud_frame = render_input_overlay(
                    self.frames[i], self.actions[i], i, total, inst_fps, scale=scale
                )
                cv2.imshow(window_name, hud_frame)
                shown += 1

                # Poll window events without blocking; ESC or 'q' exits, Space pauses
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord(" "):
                    pause_start = self._clock()
                    while True:
                        k2 = cv2.waitKey(20) & 0xFF
                        if k2 in (ord(" "), 27, ord("q")):
                            if k2 in (27, ord("q")):
                                return shown
                            break
                    # Shift start_time forward by pause duration to keep pacing accurate
                    start_time += self._clock() - pause_start

                self._wait_for_tick(start_time, i)
        finally:
            cv2.destroyAllWindows()

        return shown

    def play_live(self, countdown: int = 5, abort_key: str = "esc") -> int:
        """Replays recorded actions directly into SA2 via SA2Controller. Returns frames sent."""
        ctrl = self._controller if self._controller is not None else SA2Controller()
        total = len(self.frames)
        sent = 0

        print(f"Loaded {total} frames ({total / self.target_fps:.1f}s). Click into SA2!")
        for sec in range(countdown, 0, -1):
            print(f"  Starting in {sec}... (Press {abort_key.upper()} anytime to abort)")
            self._sleep(1.0)

        start_time = self._clock()
        with ctrl:
            for i in range(total):
                if self._is_pressed(abort_key):
                    print(f"Playback aborted at frame {i}/{total}.")
                    break

                lx, ly, jump, act, cam_l, cam_r = self.actions[i]
                ctrl.set_state(
                    lx=float(lx),
                    ly=float(ly),
                    jump=bool(jump > 0.5),
                    action=bool(act > 0.5),
                    cam_left=bool(cam_l > 0.5),
                    cam_right=bool(cam_r > 0.5),
                )
                sent += 1
                self._wait_for_tick(start_time, i)

        elapsed = self._clock() - start_time
        avg_fps = sent / max(elapsed, 1e-6)
        print(f"Live playback finished: {sent}/{total} frames in {elapsed:.2f}s ({avg_fps:.1f} FPS).")
        return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play back a recorded SA2 .npz dataset.")
    parser.add_argument("dataset", help="Path to the .npz file in data/")
    parser.add_argument(
        "--mode",
        choices=["visual", "live"],
        default="visual",
        help="'visual' opens a HUD replay window; 'live' injects inputs into SA2",
    )
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS (default: 20)")
    parser.add_argument("--scale", type=int, default=4, help="Visual window upscale factor")
    parser.add_argument("--countdown", type=int, default=5, help="Seconds before live playback starts")
    args = parser.parse_args()

    player = SA2Playback(args.dataset, target_fps=args.fps)
    if args.mode == "visual":
        player.play_visual(scale=args.scale)
    else:
        player.play_live(countdown=args.countdown)