# Testing & Verification

This document tracks the black-box (live integration) and white-box (unit/branch) test suites for `sa2-autonomous-agent`.

## Running the Test Suite

Run all automated unit tests from the repository root:

```bash
python -m pytest -v
```

*(White-box tests mock Windows OS handles, `mss`, and system clocks, so Sonic Adventure 2 does not need to be running to execute them.)*

---

## Module: Screen Capture (`core/capture.py`)

### Black-Box Testing (Live Integration)
Verified manually against a live instance of *Sonic Adventure 2* running in **1280x720 (60Hz) Windowed Mode**:

* **Window Lock:** Confirmed exact title matching locks onto `SONIC ADVENTURE 2` and ignores terminal windows containing the project path.
* **Throughput:** Sustained steady `20 FPS` capture loop without frame drops.
* **Downsampling:** Confirmed live visual output scales cleanly to `(90, 160, 3)` BGR arrays via `cv2.INTER_AREA`.

### White-Box Testing (`tests/test_capture.py`)
All 9 unit test cases (7 test functions, including parametrized boundary checks) currently **PASS**.

| Test Case | Target Method | Branch / Behavior Verified | Status |
| :--- | :--- | :--- | :--- |
| `test_find_window_ignores_substring_and_invisible` | `_find_window` | Ignores substring matches and hidden windows; performs case-insensitive exact match. | Passing |
| `test_find_window_raises_when_missing` | `_find_window` | Raises `RuntimeError` when no matching visible window exists. | Passing |
| `test_get_client_region_valid_coordinates` | `_get_client_region` | Translates client rect to screen coordinates excluding title bar and borders. | Passing |
| `test_get_client_region_raises_when_minimized` *(3 cases)* | `_get_client_region` | Raises `RuntimeError` when client `width <= 0` or `height <= 0`. | Passing |
| `test_grab_frame_strips_alpha_and_resizes` | `grab_frame` | Drops alpha channel (`BGRA` → `BGR`), resizes to `(90, 160, 3)` `uint8`, skips sleep when elapsed ≥ frame duration. | Passing |
| `test_grab_frame_sleeps_when_ahead_of_fps_target` | `grab_frame` | Sleeps for the remaining frame budget (`~0.04s` at 20 FPS) when capture finishes early. | Passing |
| `test_close_releases_mss` | `close` | Releases the underlying `mss.MSS` resource handle. | Passing |

---

## Planned Test Coverage
* **`tests/test_controller.py`:** Input range clamping, deadzone handling, and virtual gamepad/keyboard state resets.
* **`tests/test_recorder.py`:** Frame-to-input synchronization, pause/resume state transitions, and `.npy` serialization shape checks.