#!/usr/bin/env python3
"""Train a TopK Sparse Autoencoder on AlphaGenome trunk embeddings.

Matches the ESMC SAE architecture: 1536-dim input, 16384 features, TopK k=64.

Quick smoke test (chr21 only, no W&B):
    python scripts/train_sae.py \\
        --h5 data/alphagenome_embeddings_chr21.h5 \\
        --steps 500 \\
        --batch-size 2048 \\
        --output checkpoints/sae_test \\
        --no-wandb

Full run:
    python scripts/train_sae.py \\
        --h5 data/alphagenome_embeddings.h5 \\
        --val-chrom chr22 \\
        --steps 200000 \\
        --output checkpoints/sae_hg38 \\
        --wandb
"""

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(
        description="Train TopK SAE on AlphaGenome embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Data
    p.add_argument("--h5", required=True, help="HDF5 embeddings file")
    p.add_argument(
        "--val-chrom",
        default=None,
        help="Chromosome to hold out for validation (e.g. chr22). "
        "Omit to train on all chromosomes without validation.",
    )
    p.add_argument(
        "--chromosomes",
        default=None,
        help="Comma-separated list of chromosomes to use for training "
        "(default: all in file except --val-chrom).",
    )
    # Model
    p.add_argument("--d-model", type=int, default=1536, help="Embedding dim (default: 1536)")
    p.add_argument("--n-features", type=int, default=16384, help="SAE feature count (default: 16384)")
    p.add_argument("--k", type=int, default=64, help="TopK sparsity (default: 64)")
    p.add_argument("--k-aux", type=int, default=512, help="AuxK dead-feature activation count (default: 512)")
    # Training
    p.add_argument("--batch-size", type=int, default=2048, help="Bins per optimizer step (default: 2048)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--chunk-size", type=int, default=2048,
                   help="Consecutive bins per dataset item (default: 2048, = batch-size for 1 chunk/step)")
    p.add_argument("--num-workers", type=int, default=4)
    # Output
    p.add_argument("--output", default="checkpoints/sae", help="Checkpoint directory")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=100)
    # Logging
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", default="genome-feature-atlas")
    p.add_argument("--wandb-run-name", default=None)
    # Misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def build_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


def validate(model, val_loader, device, max_batches=50):
    model.eval()
    total_recon = total_l0 = n = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            x = batch.to(device)
            if x.dim() == 3:
                x = x.view(-1, x.shape[-1])
            x_hat, z, pre_acts = model(x)
            total_recon += F.mse_loss(x_hat, x).item()
            total_l0 += model.l0(z)
            n += 1
    model.train()
    return {"val_recon": total_recon / max(n, 1), "val_l0": total_l0 / max(n, 1)}


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, val_metrics: dict, cfg_dict: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "config": cfg_dict,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_metrics": val_metrics,
        },
        path,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------ setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Deferred import so the script can be imported for testing without torch
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from genome_feature_atlas.sae import TopKSAE, EmbeddingH5Dataset
    from genome_feature_atlas.sae.model import SAEConfig

    # ------------------------------------------------------------------ data
    with __import__("h5py").File(args.h5, "r") as f:
        all_chroms = sorted(f.keys())

    train_chroms = (
        [c.strip() for c in args.chromosomes.split(",")]
        if args.chromosomes
        else [c for c in all_chroms if c != args.val_chrom]
    )

    train_ds = EmbeddingH5Dataset(args.h5, chromosomes=train_chroms, chunk_size=args.chunk_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, args.batch_size // args.chunk_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = None
    if args.val_chrom:
        try:
            val_ds = EmbeddingH5Dataset(args.h5, chromosomes=[args.val_chrom], chunk_size=args.chunk_size)
            val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)
        except ValueError as e:
            print(f"Warning: {e}. Skipping validation.", file=sys.stderr)

    print(f"Train chromosomes: {train_chroms}")
    print(f"Train bins: {train_ds.n_bins:,}")
    if val_loader:
        print(f"Val chromosome: {args.val_chrom} ({val_ds.n_bins:,} bins)")

    # ------------------------------------------------------------------ model
    cfg = SAEConfig(
        d_model=args.d_model,
        n_features=args.n_features,
        k=args.k,
        k_aux=args.k_aux,
    )
    model = TopKSAE(cfg).to(device)

    print(f"\nModel: {args.n_features} features, k={args.k}, d_model={args.d_model}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Initialize b_pre from training data
    print("Initializing b_pre from training data...")
    init_sample = train_ds.sample_for_init(n=16384, device="cpu")
    model.init_from_data(init_sample.to(device))
    print(f"  b_pre mean: {model.b_pre.data.mean():.4f}, std: {model.b_pre.data.std():.4f}")

    # ------------------------------------------------------------------ optimizer / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_steps, args.steps)
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step}")

    # ------------------------------------------------------------------ W&B
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # ------------------------------------------------------------------ training loop
    cfg_dict = {
        "d_model": cfg.d_model,
        "n_features": cfg.n_features,
        "k": cfg.k,
        "k_aux": cfg.k_aux,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
    }

    model.train()
    data_iter = itertools.cycle(train_loader)

    # Accumulation buffer when batch_size > chunk_size
    accum_buf: list[torch.Tensor] = []
    accum_size = 0
    bins_per_step = args.batch_size

    t0 = time.time()
    step = start_step
    log_metrics: dict = {}

    print(f"\nStarting training for {args.steps} steps...")
    print("-" * 70)

    while step < args.steps:
        # Accumulate chunks until we have enough bins
        while accum_size < bins_per_step:
            chunk = next(data_iter)
            # chunk shape: (loader_batch, chunk_size, d_model) or (chunk_size, d_model)
            if chunk.dim() == 3:
                chunk = chunk.view(-1, chunk.shape[-1])
            accum_buf.append(chunk)
            accum_size += chunk.shape[0]

        # Take exactly bins_per_step bins
        x_full = torch.cat(accum_buf, dim=0)
        x = x_full[:bins_per_step].to(device, non_blocking=True)
        # Keep remainder for next step
        leftover = x_full[bins_per_step:]
        accum_buf = [leftover] if leftover.shape[0] > 0 else []
        accum_size = leftover.shape[0]

        # Forward
        x_hat, z, pre_acts = model(x)
        loss, metrics = model.loss(x, x_hat, pre_acts)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Post-step maintenance
        model.normalize_decoder()
        with torch.no_grad():
            model.update_feature_ema(z)

        # L0 metric (should stay near k)
        metrics["l0"] = model.l0(z)
        metrics["lr"] = scheduler.get_last_lr()[0]
        log_metrics = metrics

        # Logging
        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"step {step+1:>7d}/{args.steps}  "
                f"loss={metrics['loss']:.4f}  "
                f"recon={metrics['recon']:.4f}  "
                f"aux={metrics['aux']:.4f}  "
                f"L0={metrics['l0']:.1f}  "
                f"dead={metrics['dead_frac']:.3f}  "
                f"lr={metrics['lr']:.2e}  "
                f"({elapsed:.0f}s)"
            )
            if wandb_run:
                wandb_run.log({f"train/{k}": v for k, v in metrics.items()}, step=step + 1)

        # Checkpointing + validation
        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            val_metrics = {}
            if val_loader:
                val_metrics = validate(model, val_loader, device)
                print(
                    f"  [val] recon={val_metrics['val_recon']:.4f}  "
                    f"L0={val_metrics['val_l0']:.1f}"
                )
                if wandb_run:
                    wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step + 1)

            ckpt_path = output_dir / f"step_{step+1:07d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step, val_metrics, cfg_dict)
            # Keep a stable "latest" symlink
            latest = output_dir / "latest.pt"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(ckpt_path.name)
            print(f"  Saved checkpoint: {ckpt_path}")

        step += 1

    print("-" * 70)
    print(f"Training complete. Final loss: {log_metrics.get('loss', 'N/A'):.4f}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
