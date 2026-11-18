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

---

## Module: Input Controller (`core/controller.py`)

### Black-Box Testing (Live Integration)
Verified manually in-game with *Sonic Adventure 2* configured for **Keyboard (QWERTY)**:

* **DirectInput Injection:** Confirmed Sonic runs forward (`W`), jumps mid-stride (`J`), releases jump while continuing to run, triggers Action/Spin Dash (`K`), and stops cleanly when the script exits.

### White-Box Testing (`tests/test_controller.py`)
All 11 unit test cases (5 test functions, including parametrized deadzone checks) currently **PASS**.

| Test Case | Target Method | Branch / Behavior Verified | Status |
| :--- | :--- | :--- | :--- |
| `test_set_state_axis_deadzone_mapping` *(7 cases)* | `set_state` | Maps `lx`/`ly` axes to `WASD` keys and ignores values at or inside the deadzone boundary. | Passing |
| `test_set_state_buttons_and_custom_keymap` | `set_state` | Applies button flags (`jump`, `action`, `cam_left`, `cam_right`) and respects custom keymap overrides. | Passing |
| `test_set_state_only_sends_transitions` | `set_state` | Diffs active key sets so identical frames trigger zero OS calls and only changed keys fire `keyDown`/`keyUp`. | Passing |
| `test_reset_releases_all_held_keys` | `reset` | Releases every currently held key and clears `_held_keys`. | Passing |
| `test_context_manager_releases_keys_on_exception` | `__exit__` | Guarantees all held keys are released if an unhandled exception occurs mid-run. | Passing |