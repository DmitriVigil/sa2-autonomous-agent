from unittest.mock import MagicMock
import pytest
from core.controller import SA2Controller


def _make_controller(deadzone=0.3, keymap=None):
    mock_backend = MagicMock()
    ctrl = SA2Controller(keymap=keymap, deadzone=deadzone, backend=mock_backend)
    return ctrl, mock_backend


# --- 1. Deadzone & movement axis tests ---

@pytest.mark.parametrize(
    "lx, ly, expected_keys",
    [
        (0.0, 0.0, set()),
        (0.3, -0.3, set()),             # Exact deadzone boundary -> ignored
        (0.0, 0.8, {"w"}),              # Forward
        (0.0, -0.8, {"s"}),             # Backward
        (-0.8, 0.0, {"a"}),             # Left
        (0.8, 0.0, {"d"}),              # Right
        (0.5, 0.5, {"w", "d"}),         # Diagonal
    ],
)
def test_set_state_axis_deadzone_mapping(lx, ly, expected_keys):
    ctrl, backend = _make_controller(deadzone=0.3)
    ctrl.set_state(lx=lx, ly=ly)

    pressed = {call.args[0] for call in backend.keyDown.call_args_list}
    assert pressed == expected_keys
    assert ctrl._held_keys == expected_keys
    backend.keyUp.assert_not_called()


# --- 2. Button & custom keymap tests ---

def test_set_state_buttons_and_custom_keymap():
    ctrl, backend = _make_controller(keymap={"jump": "space"})
    ctrl.set_state(jump=True, action=True, cam_left=True, cam_right=True)

    pressed = {call.args[0] for call in backend.keyDown.call_args_list}
    assert pressed == {"space", "k", "q", "e"}


# --- 3. State diffing (no redundant keyDown/keyUp spam) ---

def test_set_state_only_sends_transitions():
    ctrl, backend = _make_controller()

    # Frame 1: Run forward + jump
    ctrl.set_state(ly=1.0, jump=True)
    backend.reset_mock()

    # Frame 2: Identical state -> zero OS calls
    ctrl.set_state(ly=1.0, jump=True)
    backend.keyDown.assert_not_called()
    backend.keyUp.assert_not_called()

    # Frame 3: Release jump, add action, keep running forward
    ctrl.set_state(ly=1.0, jump=False, action=True)
    backend.keyUp.assert_called_once_with("j")
    backend.keyDown.assert_called_once_with("k")
    assert ctrl._held_keys == {"w", "k"}


# --- 4. Cleanup & exception safety ---

def test_reset_releases_all_held_keys():
    ctrl, backend = _make_controller()
    ctrl.set_state(ly=1.0, lx=-1.0, jump=True)
    backend.reset_mock()

    ctrl.reset()

    released = {call.args[0] for call in backend.keyUp.call_args_list}
    assert released == {"w", "a", "j"}
    assert ctrl._held_keys == set()


def test_context_manager_releases_keys_on_exception():
    mock_backend = MagicMock()

    with pytest.raises(ValueError):
        with SA2Controller(backend=mock_backend) as ctrl:
            ctrl.set_state(ly=1.0, action=True)
            mock_backend.reset_mock()
            raise ValueError("Simulated crash")

    released = {call.args[0] for call in mock_backend.keyUp.call_args_list}
    assert released == {"w", "k"}