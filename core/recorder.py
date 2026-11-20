import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable
import numpy as np
import win32api

from core.capture import SA2Capture
from core.controller import DEFAULT_KEYMAP

# Special virtual-key codes beyond standard A-Z / 0-9 ASCII keys
SPECIAL_VK_CODES = {
    "space": 0x20,
    "esc": 0x1B,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}


def key_to_vk(key: str) -> int:
    """Converts a key name (e.g. 'w', 'p', 'f10') to a Windows virtual-key code."""
    k = key.lower()
    if k in SPECIAL_VK_CODES:
        return SPECIAL_VK_CODES[k]
    if len(k) == 1 and k.isalnum():
        return ord(k.upper())
    raise ValueError(f"Unsupported key name: '{key}'")


def _default_is_pressed(key: str) -> bool:
    """Returns True if the key is currently held down anywhere in Windows."""
    return bool(win32api.GetAsyncKeyState(key_to_vk(key)) & 0x8000)


class SA2Recorder:
    """Records synchronized (frame, action) pairs from live SA2 gameplay."""

    def __init__(
        self,
        capture: SA2Capture | None = None,
        keymap: dict[str, str] | None = None,
        pause_key: str = "f9",
        save_key: str = "f10",
        output_dir: str | Path = "data",
        is_pressed_fn: Callable[[str], bool] | None = None,
    ):
        self.capture = capture if capture is not None else SA2Capture(target_fps=20)
        self.keymap = {**DEFAULT_KEYMAP, **(keymap or {})}
        self.pause_key = pause_key.lower()
        self.save_key = save_key.lower()
        if self.pause_key == self.save_key:
            raise ValueError(
                f"pause_key and save_key must be different (both set to '{self.pause_key}')"
            )

        gameplay_keys = {k.lower() for k in self.keymap.values()}
        for label, hotkey in (("pause_key", self.pause_key), ("save_key", self.save_key)):
            if hotkey in gameplay_keys:
                raise ValueError(
                    f"{label} ('{hotkey}') collides with a gameplay key in keymap"
                )
        self.output_dir = Path(output_dir)
        self._is_pressed = is_pressed_fn if is_pressed_fn is not None else _default_is_pressed

        self.paused = True
        self._prev_pause_down = False
        self.frames: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []

    def read_action(self) -> np.ndarray:
        """
        Samples current keyboard state into a 6-float vector matching SA2Controller.set_state:
        [lx, ly, jump, action, cam_left, cam_right]
        """
        up = self._is_pressed(self.keymap["up"])
        down = self._is_pressed(self.keymap["down"])
        left = self._is_pressed(self.keymap["left"])
        right = self._is_pressed(self.keymap["right"])

        lx = float(right) - float(left)
        ly = float(up) - float(down)
        jump = float(self._is_pressed(self.keymap["jump"]))
        action = float(self._is_pressed(self.keymap["action"]))
        cam_left = float(self._is_pressed(self.keymap["cam_left"]))
        cam_right = float(self._is_pressed(self.keymap["cam_right"]))

        return np.array([lx, ly, jump, action, cam_left, cam_right], dtype=np.float32)

    def step(self) -> bool:
        """
        Processes one recorder tick.
        Returns False when save_key is pressed, True otherwise.
        """
        if self._is_pressed(self.save_key):
            return False

        # Rising-edge toggle so holding pause_key doesn't rapidly flip states
        pause_down = self._is_pressed(self.pause_key)
        if pause_down and not self._prev_pause_down:
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RECORDING"
            print(f"[{status}] Recorded frames so far: {len(self.frames)}")
        self._prev_pause_down = pause_down

        # Always call grab_frame() to maintain steady 20 FPS loop pacing
        frame = self.capture.grab_frame()
        if not self.paused:
            self.frames.append(frame)
            self.actions.append(self.read_action())

        return True

    def save(self, stage_name: str = "run") -> Path | None:
        """Saves recorded frames and actions to a compressed .npz dataset."""
        if not self.frames:
            print("No frames recorded; skipping save.")
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{stage_name}_{timestamp}.npz"

        frames_arr = np.asarray(self.frames, dtype=np.uint8)
        actions_arr = np.asarray(self.actions, dtype=np.float32)
        np.savez_compressed(filepath, frames=frames_arr, actions=actions_arr)

        print(
            f"Saved {len(self.frames)} samples to {filepath} | "
            f"frames: {frames_arr.shape}, actions: {actions_arr.shape}"
        )
        return filepath

    def record(self, stage_name: str = "run") -> Path | None:
        """Runs the main recording loop until save_key is pressed."""
        print(
            f"Recorder ready for '{stage_name}' (starts PAUSED). "
            f"Press '{self.pause_key.upper()}' to start/pause, "
            f"'{self.save_key.upper()}' to save and quit."
        )
        try:
            while self.step():
                pass
            return self.save(stage_name)
        finally:
            self.capture.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Sonic Adventure 2 gameplay data.")
    parser.add_argument("--stage", default="city_escape", help="Stage label for output filename")
    parser.add_argument("--pause-key", default="f9", help="Key to toggle pause/resume (default: p)")
    parser.add_argument("--save-key", default="f10", help="Key to save and exit (default: q)")
    parser.add_argument("--output-dir", default="data", help="Directory for saved .npz files")
    args = parser.parse_args()

    recorder = SA2Recorder(
        pause_key=args.pause_key,
        save_key=args.save_key,
        output_dir=args.output_dir,
    )
    recorder.record(stage_name=args.stage)