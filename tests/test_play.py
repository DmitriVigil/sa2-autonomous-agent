import numpy as np
import pytest
import torch

from models.speed_cnn import SpeedCNN
from play import FrameStackBuffer, SA2Player, load_policy


class _FakeCapture:
    def __init__(self) -> None:
        self.count = 0

    def grab_frame(self) -> np.ndarray:
        val = min(self.count * 20, 255)
        self.count += 1
        return np.full((90, 160, 3), val, dtype=np.uint8)


class _FakeController:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.released = 0

    def set_state(self, **kwargs) -> None:
        self.calls.append(kwargs)

    def reset(self) -> None:
        self.released += 1


class _StubModel(torch.nn.Module):
    def __init__(self, action_vec: list[float]) -> None:
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        self._vec = torch.tensor([action_vec], dtype=torch.float32)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self._vec


def test_load_policy_reconstructs_model_from_checkpoint(tmp_path):
    config = {
        "in_channels": 4,
        "input_height": 90,
        "input_width": 160,
        "latent_dim": 64,
        "dropout": 0.1,
    }
    model = SpeedCNN(**config)
    ckpt_path = tmp_path / "test_ckpt.pt"
    torch.save({"config": config, "model_state_dict": model.state_dict()}, ckpt_path)

    loaded, loaded_cfg = load_policy(ckpt_path, device=torch.device("cpu"))
    assert loaded_cfg == config
    assert loaded.training is False
    assert loaded.predict(torch.rand(1, 4, 90, 160)).shape == (1, 6)


def test_load_policy_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy(tmp_path / "nonexistent.pt")


def test_frame_stack_buffer_pads_first_frame_and_rolls():
    buf = FrameStackBuffer(stack_size=4)
    f0 = np.full((90, 160, 3), 50, dtype=np.uint8)
    f1 = np.full((90, 160, 3), 200, dtype=np.uint8)

    t0 = buf.push(f0)
    assert t0.shape == (1, 4, 90, 160)
    assert torch.allclose(t0, torch.full((1, 4, 90, 160), 50.0 / 255.0), atol=1e-3)

    t1 = buf.push(f1)
    assert pytest.approx(t1[0, 2, 0, 0].item(), abs=1e-3) == 50.0 / 255.0
    assert pytest.approx(t1[0, 3, 0, 0].item(), abs=1e-3) == 200.0 / 255.0


def test_step_thresholds_buttons_and_respects_dry_run():
    model = _StubModel([0.4, 0.9, 0.75, 0.2, 0.5, 0.49])
    ctrl = _FakeController()
    player = SA2Player(
        model=model,
        capture=_FakeCapture(),
        controller=ctrl,
        button_threshold=0.5,
        dry_run=False,
    )

    player.step()
    assert len(ctrl.calls) == 1
    assert ctrl.calls[0] == {
        "lx": pytest.approx(0.4),
        "ly": pytest.approx(0.9),
        "jump": True,
        "action": False,
        "cam_left": True,
        "cam_right": False,
    }

    # Dry-run must not send inputs
    ctrl_dry = _FakeController()
    player_dry = SA2Player(
        model=model,
        capture=_FakeCapture(),
        controller=ctrl_dry,
        dry_run=True,
    )
    player_dry.step()
    assert ctrl_dry.calls == []


def test_run_stops_on_kill_switch_and_releases_keys():
    model = _StubModel([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    ctrl = _FakeController()
    ticks = {"n": 0}

    def fake_is_pressed(key: str) -> bool:
        if key == "esc":
            ticks["n"] += 1
            return ticks["n"] >= 4  # Stop on 4th iteration
        return False

    player = SA2Player(
        model=model,
        capture=_FakeCapture(),
        controller=ctrl,
        fps=1000.0,
        is_pressed=fake_is_pressed,
    )

    steps = player.run(start_delay=0.0)
    assert steps == 3
    assert len(ctrl.calls) == 3
    assert ctrl.released == 1