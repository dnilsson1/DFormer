"""
CurvePT Training Script.

Supports:
- Mixed precision training (AMP)
- Gradient clipping
- Cosine LR schedule with warmup
- WandB logging with visualization
- Checkpoint saving/resuming
- Separate LR for LoRA backbone params

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --data.train_dir /path/to/data
"""

import os
import sys
import argparse
import time
import math
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.curvept import build_curvept
from losses.curve_loss import CurvePTLoss
from data.dlo_dataset import DLODataset
from data.transforms import build_train_transforms, build_val_transforms


def load_config(config_path: str, overrides: dict = None) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            d = config
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = value
    return config


def build_optimizer(model, config: dict):
    """Build AdamW optimizer with separate LR for backbone LoRA."""
    tc = config["training"]
    param_groups = model.get_param_groups(
        lr=tc["lr"],
        backbone_lr_factor=tc["backbone_lr_factor"]
    )
    return torch.optim.AdamW(
        param_groups,
        weight_decay=tc["weight_decay"]
    )


def build_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Build cosine scheduler with linear warmup."""
    tc = config["training"]
    total_steps = tc["num_epochs"] * steps_per_epoch
    warmup_steps = tc["warmup_epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(tc["min_lr"] / tc["lr"],
                   0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def prepare_targets(batch: dict, device: torch.device) -> dict:
    """Move ground-truth fields to device (shared by train and eval loops)."""
    return {
        "coords": batch["coords"].to(device, non_blocking=True),
        "visibility": batch["visibility"].to(device, non_blocking=True),
        "radius": batch["radius"].to(device, non_blocking=True),
        "num_curves": batch["num_curves"].to(device, non_blocking=True),
    }


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, epoch, config):
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    tc = config["training"]
    grad_accum = max(1, tc.get("grad_accum_steps", 1))
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Move to device with non_blocking=True
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        normals = batch["normals"].to(device, non_blocking=True)
        targets = prepare_targets(batch, device)

        # Forward pass with AMP
        with autocast("cuda", enabled=tc["amp"]):
            predictions = model(rgb, depth, normals)
            loss_dict = criterion(predictions, targets)
            loss = loss_dict["loss"]
            loss_scaled = loss / grad_accum

        # Backward pass
        scaler.scale(loss_scaled).backward()

        # Step optimizer and scheduler at accumulation boundary
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
            if tc["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            # Only advance scheduler if gradients were not unscaled due to inf/nan
            if scale_after >= scale_before:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        # Progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        # WandB logging
        if batch_idx % tc["log_every"] == 0:
            try:
                import wandb
                if wandb.run is not None:
                    log_data = {"train/loss": loss.item(),
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "train/epoch": epoch}
                    for k, v in loss_dict["breakdown"].items():
                        log_data[f"train/{k}"] = v
                    wandb.log(log_data)
            except (ImportError, Exception):
                pass

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation and return average loss plus physical metrics in mm."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    total_cd_mm = 0.0
    total_mne_mm = 0.0
    total_radius_err_mm = 0.0
    total_correct_5mm = 0
    total_correct_10mm = 0
    total_correct_20mm = 0
    total_correct_50mm = 0
    total_evaluated_nodes = 0
    num_matched_curves = 0

    from losses.hungarian import hungarian_match
    from losses.chamfer import chamfer_distance

    for batch in tqdm(loader, desc="Validating", leave=False):
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        normals = batch["normals"].to(device, non_blocking=True)
        targets = prepare_targets(batch, device)

        predictions = model(rgb, depth, normals)
        loss_dict = criterion(predictions, targets)
        total_loss += loss_dict["loss"].item()
        num_batches += 1

        # Physical metrics on final decoder layer predictions
        pred_coords = predictions["pred_coords"][-1]  # (B, Q, N, 3) in meters
        pred_conf = predictions["pred_conf"][-1]      # (B, Q, 1)
        pred_vis = predictions["pred_vis"][-1]        # (B, Q, N)
        pred_radius = predictions["pred_radius"][-1]  # (B, Q, 1)
        gt_coords = targets["coords"]                 # (B, K, N, 3) in meters
        gt_vis = targets["visibility"]                # (B, K, N)
        gt_radius = targets["radius"]                 # (B, K)
        num_curves = targets["num_curves"]

        matches = hungarian_match(
            pred_coords, pred_conf, pred_vis, gt_coords, gt_vis,
            num_curves=num_curves
        )

        B = pred_coords.shape[0]
        for b in range(B):
            pred_idx, gt_idx = matches[b]
            K = int(num_curves[b].item())
            if len(pred_idx) == 0 or K == 0:
                continue
            valid = gt_idx < K
            pred_idx = pred_idx[valid]
            gt_idx = gt_idx[valid]
            if len(pred_idx) == 0:
                continue

            # Convert meters to mm (* 1000)
            p_mm = pred_coords[b, pred_idx] * 1000.0  # (M, N, 3)
            g_mm = gt_coords[b, gt_idx] * 1000.0      # (M, N, 3)
            v_gt = gt_vis[b, gt_idx]                  # (M, N)

            # Linear 3D Chamfer Distance in mm (chamfer_distance returns squared mm)
            cd_sq = chamfer_distance(p_mm, g_mm, mask=v_gt)
            cd_linear = torch.sqrt(cd_sq.clamp(min=1e-8))
            total_cd_mm += cd_linear.item()

            # Direction-invariant node alignment for Mean Node Error (MNE)
            err_fwd = torch.norm(p_mm - g_mm, dim=-1)  # (M, N)
            err_rev = torch.norm(p_mm - g_mm.flip(dims=[1]), dim=-1)
            err_nodes = torch.where(err_fwd.mean(dim=-1, keepdim=True) < err_rev.mean(dim=-1, keepdim=True),
                                    err_fwd, err_rev)  # (M, N)

            if v_gt.sum() > 0:
                mne = (err_nodes * v_gt).sum() / v_gt.sum().clamp(min=1)
                total_mne_mm += mne.item()
                total_correct_5mm += ((err_nodes < 5.0) * (v_gt > 0.5)).sum().item()
                total_correct_10mm += ((err_nodes < 10.0) * (v_gt > 0.5)).sum().item()
                total_correct_20mm += ((err_nodes < 20.0) * (v_gt > 0.5)).sum().item()
                total_correct_50mm += ((err_nodes < 50.0) * (v_gt > 0.5)).sum().item()
                total_evaluated_nodes += (v_gt > 0.5).sum().item()

            # Radius error in mm
            r_p_mm = pred_radius[b, pred_idx, 0] * 1000.0
            r_g_mm = gt_radius[b, gt_idx] * 1000.0
            total_radius_err_mm += (r_p_mm - r_g_mm).abs().mean().item()

            num_matched_curves += len(pred_idx)

    avg_loss = total_loss / max(num_batches, 1)
    avg_cd_mm = total_cd_mm / max(num_matched_curves, 1)
    avg_mne_mm = total_mne_mm / max(num_matched_curves, 1)
    avg_rad_mm = total_radius_err_mm / max(num_matched_curves, 1)
    pck_5mm = (total_correct_5mm / max(total_evaluated_nodes, 1)) * 100.0
    pck_10mm = (total_correct_10mm / max(total_evaluated_nodes, 1)) * 100.0
    pck_20mm = (total_correct_20mm / max(total_evaluated_nodes, 1)) * 100.0
    pck_50mm = (total_correct_50mm / max(total_evaluated_nodes, 1)) * 100.0

    metrics = {
        "val/loss": avg_loss,
        "val/chamfer_mm": avg_cd_mm,
        "val/mne_mm": avg_mne_mm,
        "val/radius_err_mm": avg_rad_mm,
        "val/pck_5mm": pck_5mm,
        "val/pck_10mm": pck_10mm,
        "val/pck_20mm": pck_20mm,
        "val/pck_50mm": pck_50mm,
    }

    return avg_loss, metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, path):
    """Save training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }, path)
    print(f"  [CHECKPOINT] Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train CurvePT")
    parser.add_argument("--config", type=str,
                        default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size (for smoke tests)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (use 1 for one-sample smoke tests)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override data-loader workers (use 0 for Windows smoke tests)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--wandb_mode", type=str, default=None,
                        choices=["online", "offline", "disabled"],
                        help="Override WandB mode ('online', 'offline', 'disabled')")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--reset_scheduler", action="store_true",
                        help="Reset optimizer and scheduler when resuming for fine-tuning")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.wandb_mode is not None:
        config.setdefault("logging", {})["mode"] = args.wandb_mode

    tc = config["training"]
    dc = config["data"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # WandB
    if not args.no_wandb:
        try:
            import wandb
            lc = config.get("logging", {})
            wandb.init(
                project=lc.get("project", "curvept"),
                entity=lc.get("entity", None),
                name=lc.get("run_name", None),
                mode=lc.get("mode", None),
                config=config,
            )
        except ImportError:
            print("WandB not available — logging disabled.")
        except Exception as e:
            print(f"WandB initialization failed ({e}) — continuing with logging disabled.")

    # Build model
    print("Building CurvePT model...")
    model = build_curvept(config).to(device)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable "
          f"({100 * trainable / total:.1f}%)")

    # Build loss
    criterion = CurvePTLoss(config)

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)

    # Build datasets
    print("Loading datasets...")
    train_transform = build_train_transforms(config)
    val_transform = build_val_transforms(config)

    train_dataset = DLODataset(
        root_dir=dc["train_dir"],
        num_nodes=config["model"]["num_nodes"],
        max_curves=config["model"]["num_queries"],
        image_height=dc["image_height"],
        image_width=dc["image_width"],
        max_depth=dc["max_depth"],
        transform=train_transform,
        split_file=dc.get("train_split_file"),
        use_noisy_depth=dc.get("use_noisy_depth", False),
    )

    if args.max_samples is not None:
        from torch.utils.data import Subset
        indices = list(range(min(args.max_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)

    persistent_workers = dc["num_workers"] > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=dc["batch_size"],
        shuffle=True,
        num_workers=dc["num_workers"],
        pin_memory=dc["pin_memory"],
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    # Validation dataset (optional)
    val_loader = None
    if os.path.exists(dc.get("val_dir", "")):
        val_dataset = DLODataset(
            root_dir=dc["val_dir"],
            num_nodes=config["model"]["num_nodes"],
            max_curves=config["model"]["num_queries"],
            image_height=dc["image_height"],
            image_width=dc["image_width"],
            max_depth=dc["max_depth"],
            transform=val_transform,
            split_file=dc.get("val_split_file"),
            use_noisy_depth=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=dc["batch_size"],
            shuffle=False,
            num_workers=dc["num_workers"],
            pin_memory=dc["pin_memory"],
            persistent_workers=persistent_workers,
        )

    # Scheduler (accounting for gradient accumulation)
    grad_accum = max(1, tc.get("grad_accum_steps", 1))
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)

    # AMP scaler
    scaler = GradScaler("cuda", enabled=tc["amp"])

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if not args.reset_scheduler:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            print("  [RESET] Initializing fresh optimizer and scheduler for fine-tuning")
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch}")

    # Create checkpoint directory
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training CurvePT for {tc['num_epochs']} epochs")
    print(f"  Batch size: {dc['batch_size']} (Effective batch: {dc['batch_size'] * grad_accum})")
    print(f"  Grad accum steps: {grad_accum}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Optimizer steps/epoch: {steps_per_epoch}")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")
    best_pt_path = ckpt_dir / "best.pt"
    if best_pt_path.exists():
        try:
            best_ckpt = torch.load(best_pt_path, map_location="cpu", weights_only=False)
            if "loss" in best_ckpt and best_ckpt["loss"] is not None:
                best_val_loss = float(best_ckpt["loss"])
                print(f"  Loaded prior best validation loss: {best_val_loss:.4f}")
        except Exception:
            pass

    for epoch in range(start_epoch, tc["num_epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, config
        )
        epoch_time = time.time() - t0

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"time={epoch_time:.1f}s")

        # Validation
        if val_loader and (epoch + 1) % tc["eval_every"] == 0:
            val_loss, metrics = validate(model, val_loader, criterion, device)
            print(f"         | val_loss={val_loss:.4f} | CD={metrics['val/chamfer_mm']:.1f}mm | "
                  f"MNE={metrics['val/mne_mm']:.1f}mm | PCK@10mm={metrics['val/pck_10mm']:.1f}% | "
                  f"PCK@20mm={metrics['val/pck_20mm']:.1f}% | PCK@50mm={metrics['val/pck_50mm']:.1f}%")

            try:
                import wandb
                if wandb.run:
                    wandb.log({**metrics, "epoch": epoch})
            except (ImportError, Exception):
                pass

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, val_loss,
                    ckpt_dir / "best.pt"
                )

        # Periodic checkpoint
        if (epoch + 1) % tc["save_every"] == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, train_loss,
                ckpt_dir / f"epoch_{epoch:03d}.pt"
            )

    # Save final
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        tc["num_epochs"] - 1, train_loss,
        ckpt_dir / "final.pt"
    )

    print("\n[SUCCESS] Training complete!")


if __name__ == "__main__":
    main()
