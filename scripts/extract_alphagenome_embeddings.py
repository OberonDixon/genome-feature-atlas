#!/usr/bin/env python3
"""Extract AlphaGenome trunk embeddings across the hg38 reference genome.

Tiles across chromosomes using 1,048,576 bp (1 Mb) windows — AlphaGenome's
designed context — and captures the raw post-TransformerTower trunk (1,536-dim)
via a forward hook. This is the model's true bottleneck: spatially at 128 bp/bin
(8,192 bins per 1 Mb window), channel-wise at 1,536 dimensions.

The OutputEmbedder that model.encode() normally returns (3,072-dim) is a 2×
expansion head for prediction tasks — not the right target for SAE training.
We bypass it entirely.

HDF5 schema
-----------
/{chrom}/embeddings  float16  (n_bins, 1536)  chunks=(1024, 1536)
/{chrom}/bin_starts  int32    (n_bins,)
root attrs: model_path, fasta_path, crop_bp, resolution_bp, embedding_dim,
            layer, organism, date

Storage estimate: ~75 GB float16 for full hg38 (chr1-22, chrX).

Examples
--------
# chr21 test run (~47 Mbp, ~10-20 min on A100)
python scripts/extract_alphagenome_embeddings.py \\
    --model model.pth \\
    --fasta hg38.fa \\
    --output data/alphagenome_embeddings_chr21.h5 \\
    --chromosomes chr21

# Full genome, 1 Mb windows, crop 128 kb each edge (default)
python scripts/extract_alphagenome_embeddings.py \\
    --model model.pth \\
    --fasta hg38.fa \\
    --output data/alphagenome_embeddings_hg38.h5

# Fast pass with no cropping (fewer tiles, slight edge risk)
python scripts/extract_alphagenome_embeddings.py \\
    --model model.pth \\
    --fasta hg38.fa \\
    --output data/alphagenome_embeddings_hg38_nocrop.h5 \\
    --crop-bp 0

# 40 GB GPU: increase batch size
python scripts/extract_alphagenome_embeddings.py \\
    --model model.pth \\
    --fasta hg38.fa \\
    --output data/alphagenome_embeddings_hg38.h5 \\
    --batch-size 2
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

EMBEDDING_DIM = 1536  # post-TransformerTower trunk channel dimension


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract AlphaGenome post-transformer trunk embeddings genome-wide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True, help="Path to model weights (.pth)")
    p.add_argument("--fasta", required=True, help="Path to hg38 FASTA (must be indexed)")
    p.add_argument("--output", required=True, help="Output HDF5 file path")
    p.add_argument(
        "--chromosomes",
        default=None,
        help="Comma-separated chromosomes (default: chr1-22,chrX)",
    )
    p.add_argument(
        "--crop-bp",
        type=int,
        default=131072,
        help=(
            "Base pairs to discard from each window edge (default: 131072 = 128 kb). "
            "Keeps the central 786 kb where the full 1 Mb attention context is available. "
            "Use 0 to disable cropping (~30%% faster, minor edge artifact risk)."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1). 1 Mb windows are large; use 2 on ≥40 GB GPUs.",
    )
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for embeddings (default: float16, ~75 GB full genome)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype-policy",
        default="full_float32",
        choices=["full_float32", "mixed_precision"],
        help="Model compute precision (default: mixed_precision / bfloat16)",
    )
    p.add_argument("--window-size", type=int, default=1048576)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _make_tiling_config(args):
    from alphagenome_pytorch.extensions.inference.full_chromosome import TilingConfig
    return TilingConfig(
        window_size=args.window_size,
        crop_bp=args.crop_bp,
        resolution=128,
        batch_size=args.batch_size,
    )


def extract_chromosome(model, genome, chrom, config, organism_index, device, quiet):
    """Capture post-TransformerTower trunk across one chromosome.

    Uses a forward hook on model.tower. TransformerTower.forward() returns
    (trunk, pair_activations); the hook captures trunk at output[0].

    Returns:
        (embeddings, bin_starts) — float32 numpy arrays ready for dtype casting.
        embeddings shape: (n_bins, 1536)
    """
    import numpy as np
    import torch
    from tqdm import tqdm
    from alphagenome_pytorch.extensions.inference.full_chromosome import _generate_tiles

    chrom_length = genome.chrom_sizes[chrom]
    tiles = _generate_tiles(chrom_length, config)

    if not tiles:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), np.empty((0,), dtype=np.int32)

    # Register hook once; it overwrites the capture dict on each forward call.
    # TransformerTower.forward() returns (trunk_nlc, pair_activations).
    # trunk_nlc shape: (B, 8192, 1536) NLC for 1 Mb windows.
    _capture = {}

    def _trunk_hook(module, input, output):
        _capture["trunk"] = output[0].detach()

    hook = model.tower.register_forward_hook(_trunk_hook)

    all_embs = []
    all_starts = []

    iterator = range(0, len(tiles), config.batch_size)
    if not quiet:
        n_batches = (len(tiles) + config.batch_size - 1) // config.batch_size
        iterator = tqdm(iterator, total=n_batches, desc=f"  {chrom}")

    try:
        for batch_idx in iterator:
            batch_tiles = tiles[batch_idx : batch_idx + config.batch_size]

            seqs = [genome.fetch(chrom, ws, we) for ws, we, _, _ in batch_tiles]
            batch_seq = torch.tensor(np.stack(seqs), dtype=torch.float32, device=device)
            batch_org = torch.tensor(
                [organism_index] * len(batch_tiles), dtype=torch.long, device=device
            )

            with torch.no_grad():
                # encode() with resolutions=(128,) skips the decoder — faster.
                # The hook fires when model.tower is called internally.
                model.encode(batch_seq, batch_org, resolutions=(128,))

            # trunk_nlc: (B, 1024, 1536) NLC — seq-first, channel-last
            emb_batch = _capture["trunk"].cpu().to(torch.float32).numpy()
            del batch_seq, batch_org

            for i, (window_start, window_end, keep_start, keep_end) in enumerate(batch_tiles):
                keep_start_bin = keep_start // 128
                genome_keep_start = window_start + keep_start

                chrom_start = max(0, genome_keep_start)
                chrom_end = min(chrom_length, window_start + keep_end)

                if chrom_end <= chrom_start:
                    continue

                offset_bins = (chrom_start - genome_keep_start) // 128
                n_valid_bins = (chrom_end - chrom_start) // 128

                if n_valid_bins <= 0:
                    continue

                pred_start = keep_start_bin + offset_bins
                pred_end = pred_start + n_valid_bins

                all_embs.append(emb_batch[i, pred_start:pred_end])  # (n_valid_bins, 1536)
                all_starts.append(
                    np.arange(chrom_start, chrom_start + n_valid_bins * 128, 128, dtype=np.int32)
                )
    finally:
        hook.remove()

    if not all_embs:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.concatenate(all_embs, axis=0), np.concatenate(all_starts, axis=0)


def main():
    args = parse_args()

    model_path = Path(args.model)
    fasta_path = Path(args.fasta)
    output_path = Path(args.output)

    if not model_path.exists():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not fasta_path.exists():
        print(f"Error: FASTA not found: {fasta_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    import h5py
    import numpy as np
    import torch
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.extensions.inference.full_chromosome import GenomeSequenceProvider

    chromosomes = (
        [c.strip() for c in args.chromosomes.split(",")]
        if args.chromosomes
        else [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    )

    dtype_policy = (
        DtypePolicy.mixed_precision()
        if args.dtype_policy == "mixed_precision"
        else DtypePolicy.full_float32()
    )
    if not args.quiet:
        print(f"Loading model from {model_path}...")
    model = AlphaGenome.from_pretrained(
        str(model_path), device=args.device, dtype_policy=dtype_policy
    )
    model.eval()

    config = _make_tiling_config(args)

    if not args.quiet:
        print(f"\nTiling config:")
        print(f"  Window:    {config.window_size:,} bp")
        print(f"  Crop:      {config.crop_bp:,} bp each edge")
        print(f"  Step:      {config.step_size:,} bp ({config.step_size // 128} bins)")
        print(f"  Batch:     {config.batch_size}")
        print(f"  Layer:     post-TransformerTower trunk ({EMBEDDING_DIM}-dim, {config.window_size // 128} bins/window)")
        print(f"  Storage:   {args.dtype}")

    if not args.quiet:
        print()
    genome = GenomeSequenceProvider(str(fasta_path), chromosomes=set(chromosomes), cache=True)
    chromosomes = [c for c in chromosomes if c in genome.chrom_sizes]
    if not chromosomes:
        print("Error: no requested chromosomes found in FASTA", file=sys.stderr)
        sys.exit(1)

    store_dtype = np.float16 if args.dtype == "float16" else np.float32

    with h5py.File(str(output_path), "w") as hf:
        hf.attrs["model_path"] = str(model_path.resolve())
        hf.attrs["fasta_path"] = str(fasta_path.resolve())
        hf.attrs["crop_bp"] = args.crop_bp
        hf.attrs["window_size"] = args.window_size
        hf.attrs["resolution_bp"] = 128
        hf.attrs["embedding_dim"] = EMBEDDING_DIM
        hf.attrs["layer"] = "post_transformer_tower"
        hf.attrs["organism"] = "human"
        hf.attrs["organism_index"] = 0
        hf.attrs["dtype"] = args.dtype
        hf.attrs["date"] = datetime.utcnow().isoformat()

        for chrom in chromosomes:
            chrom_len = genome.chrom_sizes[chrom]
            if not args.quiet:
                print(f"\n{chrom} ({chrom_len:,} bp)...")

            embs, starts = extract_chromosome(
                model=model,
                genome=genome,
                chrom=chrom,
                config=config,
                organism_index=0,
                device=args.device,
                quiet=args.quiet,
            )

            if len(embs) == 0:
                if not args.quiet:
                    print(f"  Warning: no bins extracted for {chrom}, skipping")
                continue

            embs_stored = embs.astype(store_dtype)

            grp = hf.create_group(chrom)
            grp.create_dataset(
                "embeddings",
                data=embs_stored,
                chunks=(1024, EMBEDDING_DIM),
                compression="lzf",
            )
            grp.create_dataset("bin_starts", data=starts, compression="lzf")
            grp.attrs["chrom_length"] = chrom_len
            grp.attrs["n_bins"] = len(starts)

            mb = embs_stored.nbytes / 1e6
            if not args.quiet:
                print(f"  Wrote {len(starts):,} bins ({mb:.1f} MB as {args.dtype})")

    if not args.quiet:
        print(f"\nDone. Output: {output_path}")


if __name__ == "__main__":
    main()
