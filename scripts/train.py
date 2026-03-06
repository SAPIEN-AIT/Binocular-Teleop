# scripts/train.py
"""
Training script for imitation-learning policies (BC / ACT).

Usage
-----
    python scripts/train.py                              # defaults
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --policy act --epochs 500    # override via CLI

The script:
  1. Loads all HDF5 demos from ``data/``
  2. Normalises observations & actions (zero-mean, unit-std)
  3. Trains the selected policy (BC or ACT) with AdamW
  4. Saves best + last checkpoints to ``checkpoints/``
  5. (Optionally) logs to Weights & Biases
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import yaml

from src.policy.dataset import load_demos, OBS_DIM, ACT_DIM
from src.policy.bc import BCPolicy
from src.policy.act import ACTPolicy


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an IL policy on LEAP demos")
    p.add_argument("--config", type=str,
                   default=str(_ROOT / "configs" / "train_config.yaml"))
    # Allow quick overrides without editing the YAML
    p.add_argument("--policy",  type=str, choices=["bc", "act"], default=None)
    p.add_argument("--epochs",  type=int, default=None)
    p.add_argument("--lr",      type=float, default=None)
    p.add_argument("--batch",   type=int, default=None)
    p.add_argument("--seed",    type=int, default=None)
    p.add_argument("--wandb",   action="store_true", default=False,
                   help="Enable W&B logging (overrides config)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    # ── Unpack config ────────────────────────────────────────────────────────
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    out_cfg   = cfg["output"]
    policy_kind = cfg["policy"]                    # "bc" or "act"

    seed       = train_cfg["seed"]
    epochs     = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr         = train_cfg["lr"]
    wd         = train_cfg["weight_decay"]
    grad_clip  = train_cfg.get("grad_clip", 0)
    log_every  = out_cfg.get("log_every", 5)
    save_every = out_cfg.get("save_every", 0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"[Train] device = {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    demo_dir = _ROOT / data_cfg["demo_dir"]
    chunk_size = cfg.get("act", {}).get("chunk_size", 1) if policy_kind == "act" else 1

    train_loader, val_loader, norm = load_demos(
        demo_dir,
        chunk_size=chunk_size,
        batch_size=batch_size,
        val_ratio=data_cfg.get("val_ratio", 0.1),
        seed=seed,
        num_workers=data_cfg.get("num_workers", 0),
    )
    # Move norm stats to device
    norm = {k: v.to(device) for k, v in norm.items()}

    # ── Model ────────────────────────────────────────────────────────────────
    if policy_kind == "bc":
        bc_cfg = cfg.get("bc", {})
        model = BCPolicy(
            obs_dim=OBS_DIM, act_dim=ACT_DIM,
            hidden=bc_cfg.get("hidden", [256, 256]),
            dropout=bc_cfg.get("dropout", 0.0),
        ).to(device)
    elif policy_kind == "act":
        act_cfg = cfg.get("act", {})
        model = ACTPolicy(
            obs_dim=OBS_DIM, act_dim=ACT_DIM,
            chunk_size=act_cfg.get("chunk_size", 10),
            d_model=act_cfg.get("d_model", 256),
            n_heads=act_cfg.get("n_heads", 4),
            enc_layers=act_cfg.get("enc_layers", 2),
            dec_layers=act_cfg.get("dec_layers", 4),
            latent_dim=act_cfg.get("latent_dim", 32),
            kl_weight=act_cfg.get("kl_weight", 10.0),
        ).to(device)
    else:
        raise ValueError(f"Unknown policy: {policy_kind}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] policy={policy_kind}  params={n_params:,}")

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    if train_cfg.get("lr_schedule", "none") == "cosine":
        warmup = train_cfg.get("warmup_epochs", 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup, 1)
        )

    # ── W&B ──────────────────────────────────────────────────────────────────
    wb_cfg = cfg.get("wandb", {})
    use_wandb = wb_cfg.get("enabled", False)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wb_cfg.get("project", "binocular-teleop"),
                entity=wb_cfg.get("entity") or None,
                tags=wb_cfg.get("tags", []),
                config=cfg,
                job_type="train",
                reinit=True,
            )
            print(f"[W&B] → {wandb_run.url}")
        except Exception as e:
            print(f"[W&B] init failed: {e} — continuing without")
            use_wandb = False

    # ── Checkpoint dir ───────────────────────────────────────────────────────
    ckpt_dir = _ROOT / out_cfg.get("save_dir", "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Train / val helpers ──────────────────────────────────────────────────

    def _normalise(obs, act):
        obs = (obs - norm["obs_mean"]) / norm["obs_std"]
        act = (act - norm["act_mean"]) / norm["act_std"]
        return obs, act

    def run_epoch(loader, *, train_mode: bool):
        model.train(train_mode)
        total_loss = 0.0
        n = 0
        for obs, act in loader:
            obs, act = obs.to(device), act.to(device)
            obs, act = _normalise(obs, act)

            if policy_kind == "bc":
                pred = model(obs)
                loss = model.loss_fn(pred, act)
            else:  # act
                pred, kl = model(obs, act)
                loss = model.loss_fn(pred, act, kl)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * obs.size(0)
            n += obs.size(0)
        return total_loss / max(n, 1)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val = float("inf")
    t0 = time.time()

    print(f"\n{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>10}  {'Time':>8}")
    print("-" * 52)

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, train_mode=True)
        val_loss   = run_epoch(val_loader,   train_mode=False)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # ── Logging ──────────────────────────────────────────────────────────
        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"{epoch:6d}  {train_loss:10.6f}  {val_loss:10.6f}  "
                  f"{current_lr:10.2e}  {elapsed:7.1f}s")

        if use_wandb and wandb_run is not None:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": current_lr,
            })

        # ── Checkpointing ───────────────────────────────────────────────────
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "norm_stats": {k: v.cpu() for k, v in norm.items()},
            "config": cfg,
        }

        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, ckpt_dir / "best.pt")

        if save_every > 0 and epoch % save_every == 0:
            torch.save(payload, ckpt_dir / f"epoch_{epoch:04d}.pt")

    # ── Final save ───────────────────────────────────────────────────────────
    torch.save(payload, ckpt_dir / "last.pt")
    print(f"\n[Train] Done — best val loss = {best_val:.6f}")
    print(f"[Train] Checkpoints → {ckpt_dir.resolve()}")

    if use_wandb and wandb_run is not None:
        import wandb
        art = wandb.Artifact(f"{policy_kind}-policy", type="model")
        art.add_file(str(ckpt_dir / "best.pt"))
        wandb_run.log_artifact(art)
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.policy is not None:
        cfg["policy"] = args.policy
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.batch is not None:
        cfg["training"]["batch_size"] = args.batch
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True

    train(cfg)


if __name__ == "__main__":
    main()
