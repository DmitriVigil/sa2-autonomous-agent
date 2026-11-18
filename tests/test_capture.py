from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from core.capture import SA2Capture


def _make_capture(target_size=(160, 90), target_fps=20):
    """Helper to instantiate SA2Capture with mocked OS/MSS dependencies."""
    with patch("core.capture.SA2Capture._find_window", return_value=12345), \
         patch("core.capture.mss.MSS") as mock_mss:
        cap = SA2Capture(target_size=target_size, target_fps=target_fps)
        return cap, mock_mss.return_value


# --- 1. _find_window branch tests ---

@patch("core.capture.win32gui")
@patch("core.capture.mss.MSS")
def test_find_window_ignores_substring_and_invisible(mock_mss, mock_win32):
    """Ensures Git Bash paths and hidden windows are ignored in favor of exact matches."""
    fake_windows = [
        (101, True, "MINGW64:/c/Users/dmitr/sonic adventure 2 AI"),  # Substring trap
        (102, False, "SONIC ADVENTURE 2"),                           # Invisible window
        (103, True, "  sonic adventure 2  "),                        # Valid match (case/whitespace)
    ]

    def fake_enum(callback, _):
        for hwnd, _, _ in fake_windows:
            callback(hwnd, None)

    mock_win32.EnumWindows.side_effect = fake_enum
    mock_win32.IsWindowVisible.side_effect = lambda h: dict((w[0], w[1]) for w in fake_windows)[h]
    mock_win32.GetWindowText.side_effect = lambda h: dict((w[0], w[2]) for w in fake_windows)[h]

    cap = SA2Capture(window_title="SONIC ADVENTURE 2")
    assert cap.hwnd == 103


@patch("core.capture.win32gui")
@patch("core.capture.mss.MSS")
def test_find_window_raises_when_missing(mock_mss, mock_win32):
    mock_win32.EnumWindows.side_effect = lambda cb, _: None

    with pytest.raises(RuntimeError, match="Could not find a visible window"):
        SA2Capture(window_title="SONIC ADVENTURE 2")


# --- 2. _get_client_region branch tests ---

@patch("core.capture.win32gui")
def test_get_client_region_valid_coordinates(mock_win32):
    cap, _ = _make_capture()
    mock_win32.GetClientRect.return_value = (0, 0, 1280, 720)
    mock_win32.ClientToScreen.return_value = (100, 50)

    region = cap._get_client_region()
    assert region == {"top": 50, "left": 100, "width": 1280, "height": 720}


@pytest.mark.parametrize("rect", [(0, 0, 0, 0), (0, 0, -10, 720), (0, 0, 1280, 0)])
@patch("core.capture.win32gui")
def test_get_client_region_raises_when_minimized(mock_win32, rect):
    cap, _ = _make_capture()
    mock_win32.GetClientRect.return_value = rect

    with pytest.raises(RuntimeError, match="minimized"):
        cap._get_client_region()


# --- 3. grab_frame preprocessing & FPS pacing tests ---

@patch("core.capture.time.sleep")
@patch("core.capture.time.perf_counter")
def test_grab_frame_strips_alpha_and_resizes(mock_clock, mock_sleep):
    cap, mock_sct = _make_capture(target_size=(160, 90), target_fps=20)
    cap._get_client_region = MagicMock(return_value={"top": 0, "left": 0, "width": 1280, "height": 720})

    # Simulate a 720p BGRA frame (4 channels)
    fake_bgra = np.full((720, 1280, 4), 255, dtype=np.uint8)
    mock_sct.grab.return_value = fake_bgra
    mock_clock.side_effect = [1.0, 1.0]  # Elapsed >= frame_duration -> no sleep

    frame = cap.grab_frame()

    assert frame.shape == (90, 160, 3)  # OpenCV shape is (height, width, channels)
    assert frame.dtype == np.uint8
    mock_sleep.assert_not_called()


@patch("core.capture.time.sleep")
@patch("core.capture.time.perf_counter")
def test_grab_frame_sleeps_when_ahead_of_fps_target(mock_clock, mock_sleep):
    cap, mock_sct = _make_capture(target_fps=20)  # frame_duration = 0.05s
    cap._last_frame_time = 10.00
    cap._get_client_region = MagicMock(return_value={"top": 0, "left": 0, "width": 160, "height": 90})
    mock_sct.grab.return_value = np.zeros((90, 160, 4), dtype=np.uint8)

    # Only 0.01s elapsed -> should sleep ~0.04s
    mock_clock.side_effect = [10.01, 10.05]

    cap.grab_frame()

    mock_sleep.assert_called_once()
    assert mock_sleep.call_args[0][0] == pytest.approx(0.04)


# --- 4. Resource cleanup ---

def test_close_releases_mss():
    cap, mock_sct = _make_capture()
    cap.close()
    mock_sct.close.assert_called_once()