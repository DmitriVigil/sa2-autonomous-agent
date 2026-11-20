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

### `core/controller.py` — Input Injection
Handles DirectInput keyboard injection for *Sonic Adventure 2* via `pydirectinput` (`PAUSE = 0.0`).

* **Class:** `SA2Controller(keymap=None, deadzone=0.3, backend=None)`
* **Default QWERTY Keymap:** `W`/`A`/`S`/`D` (movement), `J` (jump), `K` (action), `Q`/`E` (camera left/right).
* **`set_state(lx, ly, jump, action, cam_left, cam_right) -> None`:** Converts continuous `[-1.0, 1.0]` axes and boolean flags into DirectInput scan-code transitions, only firing `keyDown`/`keyUp` on state changes.
* **`reset() -> None`:** Releases all held keys (also invoked automatically via context manager `__exit__`).
---

## 3. Architectural Decisions

* **Exact Window Title Matching:** `_find_window()` performs a case-insensitive exact match on visible window titles rather than a substring search to avoid accidentally locking onto terminals or IDE windows whose paths contain `sonic adventure 2`.
* **Client-Area Capture + DPI Awareness:** Uses `SetProcessDPIAware()` and `win32gui.GetClientRect()` to strip Windows title bars and borders, ensuring the CNN only receives playable game pixels regardless of display scaling.
* **Downsampling (`160x90` via `cv2.INTER_AREA`):** Reduces each frame from ~921k pixels (720p) to 14.4k pixels (64× smaller) to keep `.npy` datasets lightweight and PyTorch training fast, while `INTER_AREA` minimizes aliasing on thin visual cues like rings and grind rails.
* **DirectInput Keyboard over Virtual Gamepad (`pydirectinput` vs. `vgamepad`):** *SA2*'s 2012 PC port locks onto XInput controllers only at launch and disables physical keyboard navigation when Player 1 is set to Gamepad. Using `pydirectinput` scan codes keeps the real keyboard active for menus and recording while preserving a clean `set_state(lx, ly, ...)` interface for future models.
---

## 4. Planned Sections
* **Input Specification (`core/controller.py`):** Action space mapping (analog stick axes and button states).
* **Dataset Format (`core/recorder.py`):** `.npy` array schemas and frame-to-input alignment.
* **Model Architectures (`models/`):** Layer specs, hyperparameters, and OpenCV HUD/radar crop coordinates.