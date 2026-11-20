import numpy as np
import pytest
import torch

from models.speed_cnn import SpeedCNN
from train import train


def _make_synthetic_dataset(directory, num_frames: int = 24):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(num_frames, 90, 160, 3), dtype=np.uint8)
    actions = np.zeros((num_frames, 6), dtype=np.float32)
    actions[:, 1] = 1.0  # Forward
    actions[::4, 2] = 1.0  # Occasional jump
    np.savez_compressed(
        directory / "city_escape_synth.npz", frames=frames, actions=actions
    )


def test_train_creates_loadable_checkpoints(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "checkpoints"
    data_dir.mkdir()
    _make_synthetic_dataset(data_dir, num_frames=24)

    result = train(
        data_dir=data_dir,
        stage_filter="city_escape",
        output_dir=out_dir,
        epochs=2,
        batch_size=8,
        val_split=0.2,
        device_name="cpu",
    )

    assert result["best_checkpoint"].exists()
    assert result["last_checkpoint"].exists()
    assert len(result["history"]) == 2
    assert np.isfinite(result["best_val_loss"])

    # Verify checkpoint can reconstruct SpeedCNN and run inference cleanly
    ckpt = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=True)
    assert set(ckpt["config"].keys()) == {
        "in_channels",
        "input_height",
        "input_width",
        "latent_dim",
        "dropout",
    }

    loaded_model = SpeedCNN(**ckpt["config"])
    loaded_model.load_state_dict(ckpt["model_state_dict"])
    pred = loaded_model.predict(torch.rand(1, 4, 90, 160))
    assert pred.shape == (1, 6)


def test_train_rejects_invalid_val_split(tmp_path):
    with pytest.raises(ValueError, match="val_split"):
        train(data_dir=tmp_path, val_split=0.0)

    with pytest.raises(ValueError, match="val_split"):
        train(data_dir=tmp_path, val_split=1.0)