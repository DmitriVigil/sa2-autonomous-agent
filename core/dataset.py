from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SA2Dataset(Dataset):
    """PyTorch Dataset that builds stacked grayscale frame sequences from SA2 .npz runs.

    Each sample returns:
      - frames: FloatTensor of shape (stack_size, H, W) in [0.0, 1.0]
      - action: FloatTensor of shape (6,) -> [lx, ly, jump, action, cam_left, cam_right]
                aligned with the newest frame in the stack.
    """

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        stack_size: int = 4,
        stage_filter: str | None = None,
    ) -> None:
        if stack_size < 1:
            raise ValueError(f"stack_size must be >= 1, got {stack_size}")

        self.stack_size = stack_size
        self.paths = self._resolve_paths(source, stage_filter)
        if not self.paths:
            raise FileNotFoundError(
                f"No .npz dataset files found for source={source!r} (stage_filter={stage_filter!r})"
            )

        self._episodes_frames: list[np.ndarray] = []
        self._episodes_actions: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []

        for path in self.paths:
            with np.load(path) as data:
                if "frames" not in data or "actions" not in data:
                    raise ValueError(f"{path} is missing 'frames' or 'actions' arrays.")
                raw_frames = data["frames"]
                actions = data["actions"].astype(np.float32)

            if len(raw_frames) != len(actions):
                raise ValueError(
                    f"Mismatched lengths in {path}: {len(raw_frames)} frames vs {len(actions)} actions."
                )
            if len(raw_frames) < self.stack_size:
                continue

            gray_frames = self._to_grayscale(raw_frames)
            ep_idx = len(self._episodes_frames)
            self._episodes_frames.append(gray_frames)
            self._episodes_actions.append(actions)

            # Start at stack_size - 1 so stacks never bleed across separate recordings
            for frame_idx in range(self.stack_size - 1, len(gray_frames)):
                self._index.append((ep_idx, frame_idx))

        if not self._index:
            raise ValueError(
                f"No valid samples found (all recordings had fewer than {self.stack_size} frames)."
            )

    @staticmethod
    def _resolve_paths(
        source: str | Path | Sequence[str | Path],
        stage_filter: str | None,
    ) -> list[Path]:
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                paths = sorted(p.glob("*.npz"))
            elif p.is_file():
                paths = [p]
            else:
                return []
        else:
            paths = sorted(Path(item) for item in source)

        if stage_filter:
            paths = [p for p in paths if stage_filter in p.stem]
        return paths

    @staticmethod
    def _to_grayscale(frames: np.ndarray) -> np.ndarray:
        """Converts (N, H, W, 3) BGR uint8 frames to (N, H, W) uint8 grayscale."""
        if frames.ndim == 3:
            return frames.astype(np.uint8, copy=False)
        if frames.ndim == 4 and frames.shape[-1] == 3:
            return np.stack(
                [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], axis=0
            ).astype(np.uint8)
        raise ValueError(f"Unsupported frames shape: {frames.shape}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep_idx, frame_idx = self._index[idx]
        start = frame_idx - self.stack_size + 1
        stack = self._episodes_frames[ep_idx][start : frame_idx + 1]
        action = self._episodes_actions[ep_idx][frame_idx]

        frames_tensor = torch.from_numpy(stack.copy()).to(torch.float32) / 255.0
        action_tensor = torch.from_numpy(action.copy())
        return frames_tensor, action_tensor

    def compute_button_pos_weight(self, max_weight: float = 10.0) -> torch.Tensor:
        """Computes BCEWithLogitsLoss pos_weight = neg/pos for the 4 button outputs.

        Clamped to [1.0, max_weight]. Buttons with 0 positive presses default to 1.0
        so training never produces NaN/Inf gradients.
        """
        valid_actions = np.concatenate(
            [ep[self.stack_size - 1 :] for ep in self._episodes_actions], axis=0
        )
        buttons = (valid_actions[:, 2:] > 0.5).astype(np.float32)
        total = float(len(buttons))
        pos = buttons.sum(axis=0)
        neg = total - pos

        weights = np.ones(4, dtype=np.float32)
        for i in range(4):
            if pos[i] > 0:
                weights[i] = np.clip(neg[i] / pos[i], 1.0, max_weight)
        return torch.from_numpy(weights)