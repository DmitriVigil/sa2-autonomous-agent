import numpy as np
import pytest
import torch

from core.dataset import SA2Dataset


def _write_fake_npz(path, frames, actions):
    np.savez_compressed(path, frames=frames, actions=actions)


def test_loads_single_npz_and_aligns_target_with_newest_frame(tmp_path):
    n = 10
    frames = np.zeros((n, 90, 160, 3), dtype=np.uint8)
    for i in range(n):
        frames[i] = i * 10  # Distinct intensity per frame

    actions = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        actions[i, 0] = i / 10.0  # Unique lx per frame

    npz_path = tmp_path / "city_escape_001.npz"
    _write_fake_npz(npz_path, frames, actions)

    ds = SA2Dataset(npz_path, stack_size=4)
    assert len(ds) == n - 3

    x0, y0 = ds[0]
    assert x0.shape == (4, 90, 160)
    assert x0.dtype == torch.float32
    assert y0.shape == (6,)
    assert torch.all(x0 >= 0.0) and torch.all(x0 <= 1.0)

    # First stack uses frames [0, 1, 2, 3], so target must be action[3]
    assert pytest.approx(y0[0].item(), abs=1e-5) == 0.3
    assert pytest.approx(x0[0, 0, 0].item(), abs=1e-3) == 0.0
    assert pytest.approx(x0[3, 0, 0].item(), abs=1e-3) == 30.0 / 255.0


def test_multiple_episodes_do_not_bleed_across_boundaries(tmp_path):
    ep0_frames = np.zeros((6, 90, 160, 3), dtype=np.uint8)
    ep1_frames = np.full((6, 90, 160, 3), 255, dtype=np.uint8)
    actions = np.zeros((6, 6), dtype=np.float32)

    _write_fake_npz(tmp_path / "city_escape_ep0.npz", ep0_frames, actions)
    _write_fake_npz(tmp_path / "city_escape_ep1.npz", ep1_frames, actions)

    ds = SA2Dataset(tmp_path, stack_size=4)
    assert len(ds) == 6  # (6 - 3) + (6 - 3)

    # Every stack must be pure black (ep0) or pure white (ep1)
    for i in range(len(ds)):
        stack, _ = ds[i]
        unique_vals = torch.unique(stack)
        assert len(unique_vals) == 1


def test_stage_filter_and_missing_files_raise(tmp_path):
    frames = np.zeros((5, 90, 160, 3), dtype=np.uint8)
    actions = np.zeros((5, 6), dtype=np.float32)
    _write_fake_npz(tmp_path / "city_escape_1.npz", frames, actions)
    _write_fake_npz(tmp_path / "metal_harbor_1.npz", frames, actions)

    ds = SA2Dataset(tmp_path, stack_size=4, stage_filter="city_escape")
    assert len(ds) == 2

    with pytest.raises(FileNotFoundError):
        SA2Dataset(tmp_path, stack_size=4, stage_filter="green_forest")


def test_skips_short_episode_below_stack_size(tmp_path):
    short_frames = np.zeros((2, 90, 160, 3), dtype=np.uint8)
    short_actions = np.zeros((2, 6), dtype=np.float32)
    long_frames = np.zeros((6, 90, 160, 3), dtype=np.uint8)
    long_actions = np.zeros((6, 6), dtype=np.float32)

    _write_fake_npz(tmp_path / "city_escape_short.npz", short_frames, short_actions)
    _write_fake_npz(tmp_path / "city_escape_long.npz", long_frames, long_actions)

    ds = SA2Dataset(tmp_path, stack_size=4)
    assert len(ds) == 3


def test_compute_button_pos_weight_clamping_and_zero_positives(tmp_path):
    # 23 frames -> 20 valid samples with stack_size=4
    frames = np.zeros((23, 90, 160, 3), dtype=np.uint8)
    actions = np.zeros((23, 6), dtype=np.float32)
    # Button 0 (jump): 1 positive out of 20 -> ratio 19.0, clamped to 10.0
    actions[3, 2] = 1.0
    # Button 1 (action): 5 positives out of 20 -> ratio 15/5 = 3.0
    actions[3:8, 3] = 1.0
    # Button 2 (cam_left): 15 positives out of 20 -> ratio 5/15 = 0.33, clamped to 1.0
    actions[3:18, 4] = 1.0
    # Button 3 (cam_right): 0 positives -> safe fallback 1.0

    _write_fake_npz(tmp_path / "city_escape_weights.npz", frames, actions)
    ds = SA2Dataset(tmp_path, stack_size=4)

    weights = ds.compute_button_pos_weight(max_weight=10.0)
    assert weights.shape == (4,)
    assert pytest.approx(weights[0].item(), abs=1e-4) == 10.0
    assert pytest.approx(weights[1].item(), abs=1e-4) == 3.0
    assert pytest.approx(weights[2].item(), abs=1e-4) == 1.0
    assert pytest.approx(weights[3].item(), abs=1e-4) == 1.0