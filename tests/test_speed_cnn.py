import pytest
import torch

from models.speed_cnn import SpeedCNN, compute_bc_loss


def test_forward_output_shapes_and_bounds():
    model = SpeedCNN(in_channels=4)
    x = torch.rand(8, 4, 90, 160)

    axes, button_logits = model(x)

    assert axes.shape == (8, 2)
    assert button_logits.shape == (8, 4)
    assert torch.all(axes >= -1.0) and torch.all(axes <= 1.0)


@pytest.mark.parametrize("in_channels", [1, 3, 4])
def test_configurable_input_channels(in_channels):
    model = SpeedCNN(in_channels=in_channels)
    x = torch.rand(2, in_channels, 90, 160)

    axes, logits = model(x)
    assert axes.shape == (2, 2)
    assert logits.shape == (2, 4)


def test_extract_features_shape():
    model = SpeedCNN(in_channels=4, latent_dim=128)
    x = torch.rand(4, 4, 90, 160)

    feats = model.extract_features(x)
    assert feats.shape == (4, 128)


def test_invalid_input_raises_value_error():
    model = SpeedCNN(in_channels=4)

    with pytest.raises(ValueError, match="4D tensor"):
        model(torch.rand(4, 90, 160))

    with pytest.raises(ValueError, match="4 input channels"):
        model(torch.rand(2, 3, 90, 160))


def test_predict_returns_action_vector_and_preserves_training_mode():
    model = SpeedCNN(in_channels=4)
    model.train()
    x = torch.rand(3, 4, 90, 160)

    actions = model.predict(x)

    assert actions.shape == (3, 6)
    assert torch.all(actions[:, :2] >= -1.0) and torch.all(actions[:, :2] <= 1.0)
    assert torch.all(actions[:, 2:] >= 0.0) and torch.all(actions[:, 2:] <= 1.0)
    assert model.training is True


def test_compute_bc_loss_and_backward_pass():
    model = SpeedCNN(in_channels=4)
    x = torch.rand(4, 4, 90, 160)
    targets = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    pos_weight = torch.tensor([5.0, 5.0, 2.0, 2.0])

    axes, logits = model(x)
    loss, metrics = compute_bc_loss(axes, logits, targets, button_pos_weight=pos_weight)

    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert set(metrics.keys()) == {"total_loss", "axis_loss", "button_loss"}

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)