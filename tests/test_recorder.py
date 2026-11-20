from unittest.mock import MagicMock
import numpy as np
import pytest
from core.recorder import SA2Recorder, key_to_vk


def _make_recorder(tmp_path, pressed_keys: set[str], keymap=None):
    mock_capture = MagicMock()
    mock_capture.grab_frame.return_value = np.zeros((90, 160, 3), dtype=np.uint8)
    recorder = SA2Recorder(
        capture=mock_capture,
        keymap=keymap,
        pause_key="p",
        save_key="q",
        output_dir=tmp_path,
        is_pressed_fn=lambda k: k.lower() in pressed_keys,
    )
    return recorder, mock_capture


# --- 1. key_to_vk branch tests ---

@pytest.mark.parametrize(
    "key, expected",
    [
        ("w", ord("W")),
        ("W", ord("W")),
        ("5", ord("5")),
        ("space", 0x20),
        ("F10", 0x79),
    ],
)
def test_key_to_vk_valid_keys(key, expected):
    assert key_to_vk(key) == expected


def test_key_to_vk_raises_on_unsupported_key():
    with pytest.raises(ValueError, match="Unsupported key name"):
        key_to_vk("ctrl+alt")


# --- 2. read_action mapping & opposing-axis tests ---

def test_read_action_maps_axes_and_buttons(tmp_path):
    pressed = {"w", "a", "j", "e"}
    rec, _ = _make_recorder(tmp_path, pressed)

    action = rec.read_action()
    assert action.dtype == np.float32
    # [lx, ly, jump, action, cam_left, cam_right]
    np.testing.assert_array_equal(
        action, np.array([-1.0, 1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )


def test_read_action_cancels_opposing_directions(tmp_path):
    pressed = {"w", "s", "a", "d"}
    rec, _ = _make_recorder(tmp_path, pressed)

    action = rec.read_action()
    np.testing.assert_array_equal(action[:2], np.array([0.0, 0.0], dtype=np.float32))


def test_read_action_respects_custom_keymap(tmp_path):
    pressed = {"space"}
    rec, _ = _make_recorder(tmp_path, pressed, keymap={"jump": "space"})

    action = rec.read_action()
    np.testing.assert_array_equal(
        action, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )


# --- 3. step() state transitions & rising-edge pause toggle ---

def test_step_starts_paused_and_does_not_record(tmp_path):
    pressed = {"w"}
    rec, mock_cap = _make_recorder(tmp_path, pressed)

    assert rec.step() is True
    assert rec.paused is True
    mock_cap.grab_frame.assert_called_once()
    assert len(rec.frames) == 0
    assert len(rec.actions) == 0


def test_step_pause_toggle_is_rising_edge_only(tmp_path):
    pressed = set()
    rec, _ = _make_recorder(tmp_path, pressed)

    # Tick 1: Press P -> switches to RECORDING and captures 1 sample
    pressed.add("p")
    assert rec.step() is True
    assert rec.paused is False
    assert len(rec.frames) == 1

    # Tick 2: Still holding P -> stays RECORDING (no flicker), captures 2nd sample
    assert rec.step() is True
    assert rec.paused is False
    assert len(rec.frames) == 2

    # Tick 3: Release P -> still RECORDING, captures 3rd sample
    pressed.remove("p")
    assert rec.step() is True
    assert rec.paused is False
    assert len(rec.frames) == 3

    # Tick 4: Press P again -> switches back to PAUSED, no new sample
    pressed.add("p")
    assert rec.step() is True
    assert rec.paused is True
    assert len(rec.frames) == 3


def test_step_returns_false_when_save_key_pressed(tmp_path):
    pressed = {"q"}
    rec, mock_cap = _make_recorder(tmp_path, pressed)

    assert rec.step() is False
    mock_cap.grab_frame.assert_not_called()


# --- 4. save() & record() lifecycle tests ---

def test_save_skips_when_empty(tmp_path):
    rec, _ = _make_recorder(tmp_path, set())
    assert rec.save("city_escape") is None
    assert list(tmp_path.glob("*.npz")) == []


def test_save_writes_valid_npz_dataset(tmp_path):
    pressed = {"p", "w", "j"}
    rec, _ = _make_recorder(tmp_path, pressed)
    rec.step()  # Unpauses and records 1 sample

    saved_path = rec.save("city_escape")
    assert saved_path is not None and saved_path.exists()

    with np.load(saved_path) as data:
        assert data["frames"].shape == (1, 90, 160, 3)
        assert data["frames"].dtype == np.uint8
        assert data["actions"].shape == (1, 6)
        assert data["actions"].dtype == np.float32
        np.testing.assert_array_equal(
            data["actions"][0],
            np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )


def test_record_loop_saves_and_closes_capture(tmp_path):
    # Tick 1: unpause + hold W, Tick 2: press Q to save and exit
    tick_states = [{"p", "w"}, {"q"}]
    current_keys = set()

    def fake_is_pressed(k: str) -> bool:
        return k.lower() in current_keys

    mock_cap = MagicMock()
    mock_cap.grab_frame.return_value = np.zeros((90, 160, 3), dtype=np.uint8)

    rec = SA2Recorder(
        capture=mock_cap,
        output_dir=tmp_path,
        is_pressed_fn=fake_is_pressed,
    )

    original_step = rec.step

    def stepped():
        current_keys.clear()
        current_keys.update(tick_states.pop(0))
        return original_step()

    rec.step = stepped

    saved_path = rec.record("city_escape")
    assert saved_path is not None and saved_path.exists()
    mock_cap.close.assert_called_once()


def test_record_closes_capture_even_on_exception(tmp_path):
    rec, mock_cap = _make_recorder(tmp_path, set())
    mock_cap.grab_frame.side_effect = RuntimeError("Window minimized")

    with pytest.raises(RuntimeError, match="Window minimized"):
        rec.record("city_escape")

    mock_cap.close.assert_called_once()