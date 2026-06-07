"""Scan genome-wide embeddings through a trained SAE and record the top-N highest-activation
genomic bins for each feature.

Output: a single HDF5 file with shape-(n_features, n_top) arrays for activations, starts, and
chrom_idx, suitable as input for the downstream LLM annotation stage.

Reading results:
    f = h5py.File("feature_top200.h5")
    names  = f["chrom_names"].asstr()[:]
    feat   = 42
    chroms = names[f["chrom_idx"][feat]]       # (n_top,) str
    starts = f["starts"][feat]                 # (n_top,) int32
    ends   = starts + f.attrs["resolution_bp"] # (n_top,) int32
    vals   = f["activations"][feat]            # (n_top,) float32, sorted desc
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

CHROM_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find top-N highest-activation bins per SAE feature across a genome."
    )
    p.add_argument("--checkpoint", required=True, type=Path, help="SAE checkpoint .pt file")
    p.add_argument("--h5", required=True, type=Path, help="HDF5 embeddings file from extract_alphagenome_embeddings.py")
    p.add_argument("--output", required=True, type=Path, help="Output HDF5 file path (e.g. results/feature_top200.h5)")
    p.add_argument("--n-top", type=int, default=200, help="Top-N bins to keep per feature (default: 200)")
    p.add_argument("--batch-size", type=int, default=4096, help="Bins per forward-pass batch (default: 4096)")
    p.add_argument("--chromosomes", type=str, default=None, help="Comma-separated chromosomes to process (default: all in HDF5 in genomic order)")
    p.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda or cpu (default: cuda if available)")
    return p.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from genome_feature_atlas.sae import TopKSAE
    from genome_feature_atlas.sae.model import SAEConfig

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = SAEConfig(
        d_model=cfg_dict["d_model"],
        n_features=cfg_dict["n_features"],
        k=cfg_dict["k"],
        k_aux=cfg_dict.get("k_aux", 512),
    )
    model = TopKSAE(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, cfg, ckpt


def resolve_chromosomes(h5_path: Path, requested: str | None) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        available = [c for c in CHROM_ORDER if c in f]
        # Also catch any chromosomes not in the standard order
        extras = sorted(set(f.keys()) - set(CHROM_ORDER))
        available = available + extras
    if requested:
        keep = set(requested.split(","))
        available = [c for c in available if c in keep]
        missing = keep - set(available)
        if missing:
            raise ValueError(f"Requested chromosomes not found in HDF5: {missing}")
    return available


def merge_topn(
    buf_vals: np.ndarray,    # (n_features, n_top)
    buf_starts: np.ndarray,  # (n_features, n_top)
    buf_chrom: np.ndarray,   # (n_features, n_top) uint8
    cand_vals: np.ndarray,   # (k_cand, n_features)
    cand_starts: np.ndarray, # (k_cand, n_features) int32
    chrom_i: int,
    n_top: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge batch top-k candidates into the running top-N buffer.

    Returns updated (buf_vals, buf_starts, buf_chrom).
    """
    k_cand = cand_vals.shape[0]
    cand_chrom = np.full((k_cand, buf_vals.shape[0]), chrom_i, dtype=np.uint8)

    # Concatenate current buffer with batch candidates along feature axis
    merged_vals   = np.concatenate([buf_vals,   cand_vals.T],   axis=1)  # (n_features, n_top+k_cand)
    merged_starts = np.concatenate([buf_starts, cand_starts.T], axis=1)
    merged_chrom  = np.concatenate([buf_chrom,  cand_chrom.T],  axis=1)

    total = merged_vals.shape[1]
    if total <= n_top:
        # Not enough candidates yet — keep all, pad the rest remains as-is
        return merged_vals[:, :n_top], merged_starts[:, :n_top], merged_chrom[:, :n_top]

    # Keep the top-N per feature
    part_idx = np.argpartition(merged_vals, -n_top, axis=1)[:, -n_top:]  # (n_features, n_top)
    new_vals   = np.take_along_axis(merged_vals,                     part_idx,              axis=1)
    new_starts = np.take_along_axis(merged_starts,                   part_idx,              axis=1)
    new_chrom  = np.take_along_axis(merged_chrom.astype(np.int32),   part_idx.astype(np.int32), axis=1).astype(np.uint8)
    return new_vals, new_starts, new_chrom


def main() -> None:
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ------------------------------------------------------------------ model
    model, cfg, ckpt = load_model(args.checkpoint, device)
    n_features = cfg.n_features
    n_top = args.n_top
    print(f"SAE: {n_features} features, k={cfg.k}, scanning top-{n_top} per feature")

    # ------------------------------------------------------------------ chroms
    chrom_names = resolve_chromosomes(args.h5, args.chromosomes)
    print(f"Chromosomes: {chrom_names}")

    # ------------------------------------------------------------------ buffers
    # top_vals initialized to -inf so any real activation wins immediately
    top_vals   = np.full((n_features, n_top), -np.inf, dtype=np.float32)
    top_starts = np.zeros((n_features, n_top), dtype=np.int32)
    top_chrom  = np.zeros((n_features, n_top), dtype=np.uint8)

    # ------------------------------------------------------------------ scan
    with h5py.File(args.h5, "r") as f:
        resolution_bp = int(f.attrs.get("resolution_bp", 128))

        for chrom_i, chrom in enumerate(tqdm(chrom_names, desc="Chromosomes")):
            grp = f[chrom]
            embeddings_ds = grp["embeddings"]
            bin_starts_np = grp["bin_starts"][:]  # (n_bins,) int32 — load whole chrom, it's small
            n_bins = embeddings_ds.shape[0]

            for b_start in tqdm(range(0, n_bins, args.batch_size), desc=f"  {chrom}", leave=False):
                b_end = min(b_start + args.batch_size, n_bins)

                # Load from HDF5 → float32 → device
                raw = embeddings_ds[b_start:b_end]  # (batch, 1536) float16 or float32
                emb = torch.from_numpy(raw.astype(np.float32)).to(device)

                # SAE encode only (skip decode)
                with torch.no_grad():
                    z, _ = model.encode(emb)  # (batch, n_features), k non-zeros per row

                # Top-k candidates per feature within this batch
                batch_size_actual = b_end - b_start
                k_cand = min(n_top, batch_size_actual)
                topk_vals, topk_idx = torch.topk(z, k=k_cand, dim=0, largest=True, sorted=False)

                topk_vals_np = topk_vals.cpu().float().numpy()    # (k_cand, n_features)
                topk_idx_np  = topk_idx.cpu().numpy()             # (k_cand, n_features)

                # Map local batch indices → genomic starts
                batch_bin_starts = bin_starts_np[b_start:b_end]   # (batch,)
                topk_starts_np   = batch_bin_starts[topk_idx_np]  # (k_cand, n_features)

                top_vals, top_starts, top_chrom = merge_topn(
                    top_vals, top_starts, top_chrom,
                    topk_vals_np, topk_starts_np,
                    chrom_i, n_top,
                )

    # ------------------------------------------------------------------ sort per feature
    sort_idx    = np.argsort(-top_vals, axis=1)                                               # (n_features, n_top)
    top_vals    = np.take_along_axis(top_vals,                        sort_idx,              axis=1)
    top_starts  = np.take_along_axis(top_starts,                      sort_idx,              axis=1)
    top_chrom   = np.take_along_axis(top_chrom.astype(np.int32),      sort_idx.astype(np.int32), axis=1).astype(np.uint8)

    # ------------------------------------------------------------------ write HDF5
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.output, "w") as out:
        # Root metadata
        out.attrs["checkpoint_path"] = str(args.checkpoint.resolve())
        out.attrs["h5_source"]       = str(args.h5.resolve())
        out.attrs["n_features"]      = n_features
        out.attrs["n_top"]           = n_top
        out.attrs["k"]               = cfg.k
        out.attrs["resolution_bp"]   = resolution_bp
        out.attrs["date"]            = datetime.now(timezone.utc).isoformat()
        out.attrs["chromosomes"]     = ",".join(chrom_names)

        # Chromosome name lookup table (variable-length strings)
        str_dt = h5py.string_dtype(encoding="utf-8")
        out.create_dataset("chrom_names", data=np.array(chrom_names, dtype=object), dtype=str_dt)

        # Per-feature top-N arrays — shape (n_features, n_top)
        out.create_dataset("activations", data=top_vals,   compression="lzf", chunks=(256, n_top))
        out.create_dataset("starts",      data=top_starts, compression="lzf", chunks=(256, n_top))
        out.create_dataset("chrom_idx",   data=top_chrom,  compression="lzf", chunks=(256, n_top))

    n_features_with_activations = int((top_vals > 0).any(axis=1).sum())
    print(f"Wrote {args.output}")
    print(f"  {n_features_with_activations}/{n_features} features have at least one positive activation")


if __name__ == "__main__":
    main()
