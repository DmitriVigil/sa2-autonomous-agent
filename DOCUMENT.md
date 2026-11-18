# Technical Documentation

This document serves as the technical reference and design log for `sa2-autonomous-agent`.

---

## 1. Environment & Game Configuration

*Sonic Adventure 2* must be configured with the following launcher settings before running capture, recording, or inference scripts:

* **Resolution:** `1280 x 720` (exact 8× multiple of the `160x90` model input, preserving the 16:9 aspect ratio without distortion).
* **Refresh Rate:** `60Hz` *(Required — SA2 ties game physics directly to 60 FPS; higher refresh rates cause speed-up bugs).*
* **Display Mode:** Windowed (`Full Screen` unchecked) so `win32gui` can lock onto the client viewport cleanly.
* **Window Visibility:** The game window must remain unminimized and unobstructed on screen, as `mss` captures visible screen coordinates.

---

## 2. Module Reference

### `core/capture.py` — Screen Capture & Preprocessing
Handles window discovery, coordinate locking, frame downsampling, and frame-rate pacing.

* **Class:** `SA2Capture(window_title="SONIC ADVENTURE 2", target_size=(160, 90), target_fps=20)`
* **`grab_frame() -> np.ndarray`:** Captures the active game client area, strips the alpha channel (`BGRA` → `BGR`), resizes to `(90, 160, 3)` `uint8`, and enforces the `target_fps` frame budget.
* **`close() -> None`:** Releases the underlying `mss.MSS` screen-capture handle.

---

## 3. Architectural Decisions

* **Exact Window Title Matching:** `_find_window()` performs a case-insensitive exact match on visible window titles rather than a substring search to avoid accidentally locking onto terminals or IDE windows whose paths contain `sonic adventure 2`.
* **Client-Area Capture + DPI Awareness:** Uses `SetProcessDPIAware()` and `win32gui.GetClientRect()` to strip Windows title bars and borders, ensuring the CNN only receives playable game pixels regardless of display scaling.
* **Downsampling (`160x90` via `cv2.INTER_AREA`):** Reduces each frame from ~921k pixels (720p) to 14.4k pixels (64× smaller) to keep `.npy` datasets lightweight and PyTorch training fast, while `INTER_AREA` minimizes aliasing on thin visual cues like rings and grind rails.

---

## 4. Planned Sections
* **Input Specification (`core/controller.py`):** Action space mapping (analog stick axes and button states).
* **Dataset Format (`core/recorder.py`):** `.npy` array schemas and frame-to-input alignment.
* **Model Architectures (`models/`):** Layer specs, hyperparameters, and OpenCV HUD/radar crop coordinates.