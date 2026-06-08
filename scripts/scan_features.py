"""Scan genome-wide embeddings through a trained SAE and record the top-N highest-activation
genomic windows for each feature.

Each AlphaGenome extraction window (window_size - 2*crop_bp bp wide) contributes at most ONE
bin per feature: the bin with the highest activation within that window. This ensures the top-N
results represent N distinct genomic loci rather than N adjacent bins at a single active locus.
The window stride is read from the HDF5 root attrs (window_size and crop_bp) and can be
overridden with --window-stride-bp.

Reading results:
    f = h5py.File("feature_top200.h5")
    names  = f["chrom_names"].asstr()[:]
    feat   = 42
    chroms = names[f["chrom_idx"][feat]]        # (n_top,) str
    starts = f["starts"][feat]                  # (n_top,) int32
    ends   = starts + f.attrs["resolution_bp"]  # (n_top,) int32
    vals   = f["activations"][feat]             # (n_top,) float32, sorted desc
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
    p.add_argument("--window-stride-bp", type=int, default=None,
                   help="Non-redundancy distance in bp: at most one bin per this many bp per feature. "
                        "Defaults to window_size - 2*crop_bp from HDF5 metadata.")
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


def _merge_window_max(
    buf_vals: np.ndarray,       # (n_features, n_top) — modified in-place
    buf_starts: np.ndarray,     # (n_features, n_top) — modified in-place
    buf_chrom: np.ndarray,      # (n_features, n_top) uint8 — modified in-place
    win_max_vals: np.ndarray,   # (n_features,) — best activation found in this window
    win_max_starts: np.ndarray, # (n_features,) int32 — genomic start of that bin
    chrom_i: int,
) -> None:
    """Push one candidate per feature (the window maximum) into the global top-N buffer.

    A candidate is only accepted when the feature was activated in this window (val > 0)
    AND it beats the current minimum in the buffer. Operates in-place.
    """
    cur_min_val = buf_vals.min(axis=1)    # (n_features,)
    cur_min_idx = buf_vals.argmin(axis=1) # (n_features,) — slot to overwrite

    should_update = (win_max_vals > 0) & (win_max_vals > cur_min_val)
    feat_idx = np.where(should_update)[0]
    slot_idx = cur_min_idx[feat_idx]

    buf_vals[feat_idx, slot_idx]   = win_max_vals[feat_idx]
    buf_starts[feat_idx, slot_idx] = win_max_starts[feat_idx]
    buf_chrom[feat_idx, slot_idx]  = chrom_i


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

    # ------------------------------------------------------------------ window stride
    with h5py.File(args.h5, "r") as _f:
        resolution_bp  = int(_f.attrs.get("resolution_bp", 128))
        _window_size   = int(_f.attrs.get("window_size", 0))
        _crop_bp       = int(_f.attrs.get("crop_bp", 0))

    if args.window_stride_bp is not None:
        window_stride_bp = args.window_stride_bp
    elif _window_size and _crop_bp:
        window_stride_bp = _window_size - 2 * _crop_bp
    else:
        raise ValueError(
            "HDF5 is missing window_size / crop_bp attrs. "
            "Provide --window-stride-bp explicitly."
        )

    n_bins_per_window = window_stride_bp // resolution_bp
    print(f"Window stride: {window_stride_bp:,} bp = {n_bins_per_window} bins  "
          f"(one candidate per feature per window)")

    # ------------------------------------------------------------------ buffers
    # Initialised to -inf so any positive activation beats the initial minimum.
    top_vals   = np.full((n_features, n_top), -np.inf, dtype=np.float32)
    top_starts = np.zeros((n_features, n_top), dtype=np.int32)
    top_chrom  = np.zeros((n_features, n_top), dtype=np.uint8)

    # ------------------------------------------------------------------ scan
    with h5py.File(args.h5, "r") as f:
        for chrom_i, chrom in enumerate(tqdm(chrom_names, desc="Chromosomes")):
            grp = f[chrom]
            embeddings_ds = grp["embeddings"]
            bin_starts_np = grp["bin_starts"][:]  # (n_bins,) int32
            n_bins = embeddings_ds.shape[0]

            # Per-window running max — flushed whenever a batch crosses a window boundary.
            # Using a flat batch loop (no cap at window boundaries) keeps GPU batches full.
            cur_win_id     = -1
            win_max_vals   = np.zeros(n_features, dtype=np.float32)
            win_max_starts = np.zeros(n_features, dtype=np.int32)

            for b_start in tqdm(range(0, n_bins, args.batch_size), desc=f"  {chrom}", leave=False):
                b_end = min(b_start + args.batch_size, n_bins)

                raw = embeddings_ds[b_start:b_end]  # float16 or float32
                emb = torch.from_numpy(raw.astype(np.float32)).to(device)

                with torch.no_grad():
                    z, _ = model.encode(emb)  # (batch, n_features) on device

                # Split the batch into (at most 2) window-aligned segments using
                # arithmetic — avoids materialising the full batch as a numpy array.
                first_win_id      = b_start // n_bins_per_window
                first_win_end_bin = (first_win_id + 1) * n_bins_per_window
                split = min(first_win_end_bin - b_start, b_end - b_start)

                segments = [(first_win_id, 0, split)]
                if split < (b_end - b_start):
                    segments.append((first_win_id + 1, split, b_end - b_start))

                for w_id, lo, hi in segments:
                    if w_id != cur_win_id:
                        if cur_win_id >= 0:
                            _merge_window_max(top_vals, top_starts, top_chrom,
                                              win_max_vals, win_max_starts, chrom_i)
                        win_max_vals[:]   = 0
                        win_max_starts[:] = 0
                        cur_win_id = w_id

                    # max(dim=0) on a GPU/CPU tensor: one pass, returns (values, indices).
                    # Transfers only two (n_features,) vectors to CPU — not the full batch.
                    sub_max_val_t, sub_max_idx_t = z[lo:hi].max(dim=0)
                    sub_max_val   = sub_max_val_t.cpu().float().numpy()   # (n_features,)
                    sub_max_idx   = sub_max_idx_t.cpu().numpy()           # (n_features,)
                    sub_max_start = bin_starts_np[b_start + lo + sub_max_idx]

                    improved       = sub_max_val > win_max_vals
                    win_max_vals   = np.where(improved, sub_max_val,   win_max_vals)
                    win_max_starts = np.where(improved, sub_max_start, win_max_starts)

            # Flush the last window of this chromosome
            if cur_win_id >= 0:
                _merge_window_max(top_vals, top_starts, top_chrom,
                                  win_max_vals, win_max_starts, chrom_i)

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
        out.attrs["k"]                = cfg.k
        out.attrs["resolution_bp"]   = resolution_bp
        out.attrs["window_stride_bp"] = window_stride_bp
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
