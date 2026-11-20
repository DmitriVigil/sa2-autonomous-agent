from __future__ import annotations

import torch
import torch.nn as nn


class SpeedCNN(nn.Module):
    """Lightweight CNN policy for Sonic Adventure 2 Speed stages.

    Maps a stack of grayscale frames (B, C, H, W) to:
      - axes: (B, 2) continuous joystick predictions in [-1, 1] -> [lx, ly]
      - button_logits: (B, 4) raw logits -> [jump, action, cam_left, cam_right]
    """

    def __init__(
        self,
        in_channels: int = 4,
        input_height: int = 90,
        input_width: int = 160,
        latent_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Compute flattened conv output size dynamically so input resolution stays flexible
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_height, input_width)
            flat_dim = self.conv(dummy).flatten(1).shape[1]

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Separated heads make transferring `conv` + `encoder` into PPO straightforward
        self.axis_head = nn.Sequential(
            nn.Linear(latent_dim, 2),
            nn.Tanh(),
        )
        self.button_head = nn.Linear(latent_dim, 4)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the shared latent representation (B, latent_dim)."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        return self.encoder(self.conv(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training. Returns (axes, button_logits)."""
        features = self.extract_features(x)
        axes = self.axis_head(features)
        button_logits = self.button_head(features)
        return axes, button_logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper returning a single (B, 6) action vector.

        Axes are in [-1, 1]; button entries are probabilities in [0, 1].
        """
        was_training = self.training
        self.eval()
        try:
            axes, button_logits = self.forward(x)
            button_probs = torch.sigmoid(button_logits)
            return torch.cat([axes, button_probs], dim=1)
        finally:
            self.train(was_training)


def compute_bc_loss(
    axes_pred: torch.Tensor,
    button_logits: torch.Tensor,
    targets: torch.Tensor,
    button_pos_weight: torch.Tensor | None = None,
    button_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined Behavioral Cloning loss: MSE for axes + BCEWithLogits for buttons.

    Args:
        axes_pred: (B, 2) predicted [lx, ly].
        button_logits: (B, 4) predicted raw logits [jump, action, cam_l, cam_r].
        targets: (B, 6) ground-truth action vector from SA2Recorder.
        button_pos_weight: Optional (4,) tensor to upweight rare button presses.
        button_loss_weight: Scalar weight balancing button loss against axis loss.
    """
    axis_targets = targets[:, :2]
    button_targets = targets[:, 2:]

    axis_loss = nn.functional.mse_loss(axes_pred, axis_targets)
    button_loss = nn.functional.binary_cross_entropy_with_logits(
        button_logits, button_targets, pos_weight=button_pos_weight
    )
    total_loss = axis_loss + button_loss_weight * button_loss

    metrics = {
        "total_loss": float(total_loss.item()),
        "axis_loss": float(axis_loss.item()),
        "button_loss": float(button_loss.item()),
    }
    return total_loss, metrics