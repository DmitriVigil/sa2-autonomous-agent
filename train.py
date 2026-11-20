from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from core.dataset import SA2Dataset
from models.speed_cnn import SpeedCNN, compute_bc_loss


def train_one_epoch(
    model: SpeedCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    button_pos_weight: torch.Tensor | None = None,
    button_loss_weight: float = 1.0,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Runs one training epoch and returns mean loss metrics."""
    model.train()
    totals = {"total_loss": 0.0, "axis_loss": 0.0, "button_loss": 0.0}
    num_samples = 0

    for frames, actions in loader:
        frames = frames.to(device)
        actions = actions.to(device)
        batch_size = frames.size(0)

        optimizer.zero_grad(set_to_none=True)
        axes_pred, button_logits = model(frames)
        loss, metrics = compute_bc_loss(
            axes_pred,
            button_logits,
            actions,
            button_pos_weight=button_pos_weight,
            button_loss_weight=button_loss_weight,
        )
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        for k in totals:
            totals[k] += metrics[k] * batch_size
        num_samples += batch_size

    return {k: v / max(num_samples, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: SpeedCNN,
    loader: DataLoader,
    device: torch.device,
    button_pos_weight: torch.Tensor | None = None,
    button_loss_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluates the model on a validation DataLoader."""
    model.eval()
    totals = {"total_loss": 0.0, "axis_loss": 0.0, "button_loss": 0.0}
    num_samples = 0

    for frames, actions in loader:
        frames = frames.to(device)
        actions = actions.to(device)
        batch_size = frames.size(0)

        axes_pred, button_logits = model(frames)
        _, metrics = compute_bc_loss(
            axes_pred,
            button_logits,
            actions,
            button_pos_weight=button_pos_weight,
            button_loss_weight=button_loss_weight,
        )

        for k in totals:
            totals[k] += metrics[k] * batch_size
        num_samples += batch_size

    return {k: v / max(num_samples, 1) for k, v in totals.items()}


def train(
    data_dir: str | Path = "data",
    stage_filter: str | None = "city_escape",
    output_dir: str | Path = "checkpoints",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.15,
    stack_size: int = 4,
    button_loss_weight: float = 1.0,
    max_pos_weight: float = 10.0,
    seed: int = 42,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Full Behavioral Cloning training loop for SpeedCNN."""
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")

    torch.manual_seed(seed)
    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset = SA2Dataset(data_dir, stack_size=stack_size, stage_filter=stage_filter)
    if len(dataset) < 2:
        raise ValueError(f"Need at least 2 samples to split train/val, got {len(dataset)}")

    val_size = max(1, int(round(len(dataset) * val_split)))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    pos_weight = dataset.compute_button_pos_weight(max_weight=max_pos_weight).to(device)

    sample_frames, _ = dataset[0]
    model_config = {
        "in_channels": stack_size,
        "input_height": int(sample_frames.shape[1]),
        "input_width": int(sample_frames.shape[2]),
        "latent_dim": 256,
        "dropout": 0.2,
    }
    model = SpeedCNN(**model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_path = out_path / "speed_cnn_best.pt"
    last_path = out_path / "speed_cnn_last.pt"

    print(
        f"Training SpeedCNN on {device} | total={len(dataset)} "
        f"(train={train_size}, val={val_size}) | pos_weight={pos_weight.cpu().round(decimals=2).tolist()}"
    )

    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            button_pos_weight=pos_weight,
            button_loss_weight=button_loss_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            button_pos_weight=pos_weight,
            button_loss_weight=button_loss_weight,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "button_pos_weight": pos_weight.cpu().tolist(),
        }
        torch.save(checkpoint, last_path)

        saved_best = False
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            torch.save(checkpoint, best_path)
            saved_best = True

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": current_lr})
        tag = " *best*" if saved_best else ""
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | lr={current_lr:.2e} | "
            f"train={train_metrics['total_loss']:.4f} "
            f"(axis={train_metrics['axis_loss']:.4f}, btn={train_metrics['button_loss']:.4f}) | "
            f"val={val_metrics['total_loss']:.4f} "
            f"(axis={val_metrics['axis_loss']:.4f}, btn={val_metrics['button_loss']:.4f}){tag}"
        )

    return {
        "best_val_loss": best_val_loss,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SpeedCNN via Behavioral Cloning.")
    parser.add_argument("--data-dir", default="data", help="Directory containing .npz runs")
    parser.add_argument("--stage", default="city_escape", help="Filename filter (pass '' for all)")
    parser.add_argument("--output-dir", default="checkpoints", help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--stack-size", type=int, default=4)
    parser.add_argument("--button-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(
        data_dir=args.data_dir,
        stage_filter=args.stage or None,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        stack_size=args.stack_size,
        button_loss_weight=args.button_loss_weight,
        seed=args.seed,
    )
    print(f"Done. Best val_loss={result['best_val_loss']:.4f} -> {result['best_checkpoint']}")


if __name__ == "__main__":
    main()