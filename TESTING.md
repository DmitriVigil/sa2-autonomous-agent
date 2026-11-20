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

---

## Module: Gameplay Recorder (`core/recorder.py`)

### Black-Box Testing (Live Integration)
Verified manually during a live *City Escape* run in *Sonic Adventure 2*:

* **Hotkey Flow:** Confirmed the recorder boots in a `PAUSED` state, toggles cleanly into `RECORDING` via `P`, and saves/exits via `Q`.
* **Synchronized Capture & Serialization:** Recorded `520` samples (~26 seconds at 20 FPS) to `data/city_escape_20261118_192527.npz` with verified array shapes `frames: (520, 90, 160, 3)` (`uint8`) and `actions: (520, 6)` (`float32`).
* **Input Verification:** Inspected the saved `.npz` file with `numpy` and confirmed live keyboard inputs were captured alongside frames.

### White-Box Testing (`tests/test_recorder.py`)
All 16 unit test cases (12 test functions, including parametrized virtual-key checks) currently **PASS**.

| Test Case | Target Method | Branch / Behavior Verified | Status |
| :--- | :--- | :--- | :--- |
| `test_key_to_vk_valid_keys` *(5 cases)* | `key_to_vk` | Converts alphanumeric keys (case-insensitive) and special keys (`space`, `F10`) to Windows virtual-key codes. | Passing |
| `test_key_to_vk_raises_on_unsupported_key` | `key_to_vk` | Raises `ValueError` for invalid or unsupported key strings. | Passing |
| `test_read_action_maps_axes_and_buttons` | `read_action` | Encodes held movement/button keys into a 6-float `float32` vector `[lx, ly, jump, action, cam_left, cam_right]`. | Passing |
| `test_read_action_cancels_opposing_directions` | `read_action` | Cancels simultaneous opposing directional inputs (`W+S`, `A+D`) to `0.0`. | Passing |
| `test_read_action_respects_custom_keymap` | `read_action` | Reads custom keybind overrides (e.g., `jump: "space"`) correctly. | Passing |
| `test_step_starts_paused_and_does_not_record` | `step` | Maintains frame pacing via `grab_frame()` while paused without appending samples. | Passing |
| `test_step_pause_toggle_is_rising_edge_only` | `step` | Toggles pause only on initial press of `P`; holding `P` across ticks does not flicker states. | Passing |
| `test_step_returns_false_when_save_key_pressed` | `step` | Exits the tick immediately (`False`) without grabbing a frame when `save_key` is pressed. | Passing |
| `test_save_skips_when_empty` | `save` | Returns `None` and writes no file if zero frames were recorded. | Passing |
| `test_save_writes_valid_npz_dataset` | `save` | Writes a compressed `.npz` file with expected `frames` and `actions` shapes, dtypes, and values. | Passing |
| `test_record_loop_saves_and_closes_capture` | `record` | Runs the tick loop until `save_key`, persists the `.npz` dataset, and closes `SA2Capture`. | Passing |
| `test_record_closes_capture_even_on_exception` | `record` | Guarantees `capture.close()` runs in the `finally` block if capture raises mid-recording. | Passing |

---

## Module: Speed Policy Network (`models/speed_cnn.py`)

### Black-Box Testing (Latency & Real-Data Overfit)
Verified on local hardware (`Python 3.13.14`, CPU) using real stacked frames from `data/city_escape_20261119_211011.npz`:

* **Inference Latency:** `1,052,554` parameters averaged **`2.63 ms`** per `(1, 4, 90, 160)` forward pass over 100 iterations (well under the `50.00 ms` budget for 20 FPS real-time play).
* **Single-Batch Overfit:** Trained on 16 real 4-frame grayscale stacks for 80 Adam steps (`lr=1e-3`); total BC loss dropped from `0.8117` to `0.0142` (`axis_loss`: `0.0994` → `0.0109`, `button_loss`: `0.7124` → `0.0033`), with predicted actions matching recorded targets.

### White-Box Testing (`tests/test_speed_cnn.py`)
All 8 unit test cases (6 test functions, including parametrized channel checks) currently **PASS**.

| Test Case | Target Method | Branch / Behavior Verified | Status |
| :--- | :--- | :--- | :--- |
| `test_forward_output_shapes_and_bounds` | `forward` | Verifies `(B, 2)` axes bounded to `[-1, 1]` via `Tanh` and `(B, 4)` raw button logits. | Passing |
| `test_configurable_input_channels` *(3 cases)* | `__init__`, `forward` | Dynamically computes conv flatten size for `1`, `3`, and `4` input channels. | Passing |
| `test_extract_features_shape` | `extract_features` | Returns shared `(B, latent_dim)` visual features for future PPO transfer. | Passing |
| `test_invalid_input_raises_value_error` | `extract_features` | Rejects non-4D tensors and mismatched channel counts with `ValueError`. | Passing |
| `test_predict_returns_action_vector_and_preserves_training_mode` | `predict` | Returns `(B, 6)` vector with sigmoid button probabilities `[0, 1]` and restores `.training` state. | Passing |
| `test_compute_bc_loss_and_backward_pass` | `compute_bc_loss` | Computes combined MSE + weighted `BCEWithLogitsLoss` and verifies gradients flow to all parameters. | Passing |

---

## Module: Stacked Frame Dataset (`core/dataset.py`)

### Black-Box Testing (Real `.npz` Ingestion & Action Audit)
Verified against the Milestone 1 *City Escape* recordings in `data/`:

* **Dataset Loading:** Loaded `1,491` valid 4-frame grayscale samples with verified tensor shape `(4, 90, 160)` and computed button `pos_weight = [1.0, 1.0, 10.0, 1.0]`.
* **Data Quality Diagnostic:** Action-channel audit surfaced `[148, 247, 0, 0, 1, 0]` non-zero frames out of `1,497` total frames (~83% idle frames and zero `jump`/`action` presses in the short Milestone 1 test clips), and flagged the `Q` (`cam_left` vs. `save_key`) hotkey collision in `core/recorder.py` ahead of Milestone 2 dataset collection.

### White-Box Testing (`tests/test_dataset.py`)
All 5 unit test cases currently **PASS**.

| Test Case | Target Method | Branch / Behavior Verified | Status |
| :--- | :--- | :--- | :--- |
| `test_loads_single_npz_and_aligns_target_with_newest_frame` | `__init__`, `__getitem__` | Converts BGR frames to normalized `[0, 1]` grayscale stacks and aligns target action with the newest frame. | Passing |
| `test_multiple_episodes_do_not_bleed_across_boundaries` | `__init__`, `__getitem__` | Ensures 4-frame stacks never cross episode boundaries between separate `.npz` files. | Passing |
| `test_stage_filter_and_missing_files_raise` | `_resolve_paths` | Filters `.npz` files by stage name and raises `FileNotFoundError` when no files match. | Passing |
| `test_skips_short_episode_below_stack_size` | `__init__` | Safely skips recordings shorter than `stack_size` without failing dataset initialization. | Passing |
| `test_compute_button_pos_weight_clamping_and_zero_positives` | `compute_button_pos_weight` | Clamps `neg/pos` ratios to `[1.0, max_weight]` and defaults zero-positive buttons to `1.0` to prevent `NaN`/`Inf`. | Passing |

---

## Module: Behavioral Cloning Trainer (`train.py`)

### Black-Box Testing (End-to-End Smoke Test on Real Data)
Executed a 5-epoch CPU training run (`python train.py --data-dir data --stage city_escape --epochs 5 --batch-size 32`) on `1,491` real samples (`1,267` train / `224` val):

* **Convergence:** Training loss decreased from `0.6486` to `0.1447`; validation loss decreased monotonically every epoch from `0.2654` to `0.1208` (`axis_loss`: `0.2369` → `0.1192`, `button_loss`: `0.0285` → `0.0016`).
* **Artifact Persistence:** Saved loadable `checkpoints/speed_cnn_best.pt` and `checkpoints/speed_cnn_last.pt` with embedded model configuration metadata.

### White-Box Testing (`tests/test_train.py`)
All 2 unit test cases currently **PASS**.

| Test Case | Target Method | Branch / Behavior Verified | Status |
| :--- | :--- | :--- | :--- |
| `test_train_creates_loadable_checkpoints` | `train`, `train_one_epoch`, `evaluate` | Runs full synthetic train/val loop, writes best/last `.pt` checkpoints, and reconstructs `SpeedCNN` cleanly for inference. | Passing |
| `test_train_rejects_invalid_val_split` | `train` | Raises `ValueError` for out-of-bounds `val_split` values (`0.0` and `1.0`). | Passing |

---

## Module: Live Inference Runner (`play.py`)

### Black-Box Testing (Live `SA2` Dry-Run & Closed-Loop Control)
Verified against live `sonic2app.exe` in *City Escape* using `checkpoints/speed_cnn_best.pt`:

* **Integration Catch & Fix:** Initial dry run caught interface mismatches (`grab_frame()` and `reset()`) against real `SA2Capture` and `SA2Controller` classes; patched `play.py` and aligned unit test mocks to enforce the real interfaces.
* **Dry-Run Mode (`--dry-run`):** Streamed live 20 FPS predictions for `1,072` steps (~53 seconds) without key injection and exited cleanly on `ESC`.
* **Live Control Mode:** Ran `920` steps (~46 seconds) with live DirectInput injection and clean key release on exit (predicted near-zero axes as expected from the mostly idle Milestone 1 test clips).

### White-Box Testing (`tests/test_play.py`)
All 5 unit test cases currently **PASS**.

| Test Case | Target Method | Branch / Behavior Verified | Status |
| :--- | :--- | :--- | :--- |
| `test_load_policy_reconstructs_model_from_checkpoint` | `load_policy` | Loads `.pt` checkpoint, rebuilds `SpeedCNN` from saved `config`, and puts model in `.eval()` mode. | Passing |
| `test_load_policy_missing_file_raises` | `load_policy` | Raises `FileNotFoundError` for missing checkpoint paths. | Passing |
| `test_frame_stack_buffer_pads_first_frame_and_rolls` | `FrameStackBuffer.push` | Pads initial stack with the first frame and rolls newest grayscale frames into channel index `-1`. | Passing |
| `test_step_thresholds_buttons_and_respects_dry_run` | `SA2Player.step` | Applies button probability thresholding to `controller.set_state` and suppresses all inputs when `dry_run=True`. | Passing |
| `test_run_stops_on_kill_switch_and_releases_keys` | `SA2Player.run` | Terminates loop immediately on `stop_key` (`ESC`) and guarantees `controller.reset()` runs on exit. | Passing |