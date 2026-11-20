import ctypes
import time
import cv2
import mss
import numpy as np
import win32gui

# Prevent Windows DPI scaling (125%, 150%, etc.) from offsetting capture coordinates
ctypes.windll.user32.SetProcessDPIAware()


class SA2Capture:
    def __init__(
        self,
        window_title: str = "SONIC ADVENTURE 2",
        target_size: tuple[int, int] = (160, 90),
        target_fps: int = 20,
    ):
        self.window_title = window_title
        self.target_size = target_size
        self.frame_duration = 1.0 / target_fps
        self.hwnd = self._find_window(window_title)
        self.sct = mss.MSS()
        self._last_frame_time = 0.0

    def _find_window(self, exact_title: str) -> int:
        matched_hwnd = None

        def callback(hwnd, _):
            nonlocal matched_hwnd
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).strip()
                if title.lower() == exact_title.lower():
                    matched_hwnd = hwnd

        win32gui.EnumWindows(callback, None)
        if not matched_hwnd:
            raise RuntimeError(
                f"Could not find a visible window titled '{exact_title}'. "
                "Make sure SA2 is running in Windowed mode."
            )
        return matched_hwnd

    def _get_client_region(self) -> dict:
        """Returns the exact playable area in screen coordinates (no title bar/borders)."""
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            raise RuntimeError("SA2 window appears to be minimized.")

        screen_left, screen_top = win32gui.ClientToScreen(self.hwnd, (left, top))
        return {
            "top": screen_top,
            "left": screen_left,
            "width": width,
            "height": height,
        }

    def grab_frame(self) -> np.ndarray:
        """Captures one frame, resizes to target_size (BGR), and enforces target FPS."""
        now = time.perf_counter()
        elapsed = now - self._last_frame_time
        if elapsed < self.frame_duration:
            time.sleep(self.frame_duration - elapsed)
        self._last_frame_time = time.perf_counter()

        region = self._get_client_region()
        # Grab BGRA and slice off the alpha channel -> BGR
        raw = np.array(self.sct.grab(region))[:, :, :3]

        # INTER_AREA is best for downsampling (reduces aliasing on rings/rails)
        return cv2.resize(raw, self.target_size, interpolation=cv2.INTER_AREA)

    def close(self):
        self.sct.close()


if __name__ == "__main__":
    cap = SA2Capture(target_fps=20)
    print(f"Window: '{win32gui.GetWindowText(cap.hwnd)}' | Region: {cap._get_client_region()}")
    print("Press 'q' in the preview window to quit.")

    frame_count = 0
    fps_timer = time.perf_counter()

    try:
        while True:
            frame = cap.grab_frame()
            frame_count += 1

            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                print(f"Capture FPS: {frame_count} | Mean pixel value: {frame.mean():.1f}")
                frame_count = 0
                fps_timer = now

            # Scale up 3x for human eyes (CNN still gets the 160x90 'frame')
            preview = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("SA2 CNN Input (160x90)", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.close()
        cv2.destroyAllWindows()