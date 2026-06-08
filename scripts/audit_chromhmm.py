"""Interactive audit tool: step through SAE features and display ChromHMM tissue
activity summaries for their top-activated genomic loci.

Usage:
    python scripts/audit_chromhmm.py \\
        --input results/feature_top200.h5 \\
        --data-dir data/roadmap_chromhmm/hg19

Controls (at the '>' prompt):
    Enter         next feature
    f <id>        jump to a specific feature by index
    q             quit

Batch mode (non-interactive, writes JSON):
    python scripts/audit_chromhmm.py \\
        --input results/feature_top200.h5 \\
        --data-dir data/roadmap_chromhmm/hg19 \\
        --output chromhmm_summaries.json
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from genome_feature_atlas.chromhmm import (
    STATE_NAMES,
    TISSUE_GROUPS,
    ChromHMMLookup,
    summarize_feature,
)

_W = 72  # display width


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit SAE features against ChromHMM tissue annotations.")
    p.add_argument("--input", required=True, type=Path, help="feature_top200.h5 from scan_features.py")
    p.add_argument("--data-dir", required=True, type=Path, help="Directory with E*_15_coreMarks_dense.bed.bgz files")
    p.add_argument("--n-loci", type=int, default=10, help="Top loci per feature to include in summary (default: 10)")
    p.add_argument("--features", type=str, default=None, help="Comma-separated feature IDs to inspect (default: all in order)")
    p.add_argument("--start-from", type=int, default=0, help="Start iteration at this feature index (default: 0)")
    p.add_argument("--output", type=Path, default=None, help="Write all summaries to this JSON file (batch mode)")
    return p.parse_args()


# ── display helpers ───────────────────────────────────────────────────────────

_TISSUE_ORDER = list(TISSUE_GROUPS.keys())


def _print_feature_header(fid: int, n_active: int, n_slots: int, n_features: int) -> None:
    print(f"\n{'═' * _W}")
    print(f"  Feature {fid:<6}  {n_active}/{n_slots} slots with activation > 0   (of {n_features} features total)")
    print(f"{'═' * _W}")


def _print_summary(summary: dict, n_loci: int) -> None:
    matrix = summary["matrix"]

    print(f"\n  ChromHMM summary ({n_loci} loci × 15-state model, hg19)\n")
    print(f"  {'Tissue':<22}  top states")
    print(f"  {'─' * (_W - 2)}")

    for tissue in _TISSUE_ORDER:
        if tissue not in matrix:
            continue
        top_states = matrix[tissue]["top_states"]
        state_str = "  ".join(
            f"{STATE_NAMES.get(s, s)}({f:.0%})"
            for s, f in top_states
        )
        print(f"  {tissue:<22}  {state_str}")


def _print_footer() -> None:
    print(f"\n{'─' * _W}")
    print("  [Enter] next   [f <id>] jump to feature   [q] quit")


# ── core logic ────────────────────────────────────────────────────────────────

def compute_feature_summary(
    fid: int,
    activations: np.ndarray,
    starts: np.ndarray,
    chrom_idx: np.ndarray,
    chrom_names: list[str],
    lookup: ChromHMMLookup,
    n_loci: int,
    resolution_bp: int,
) -> dict | None:
    """Compute ChromHMM tissue summary for one SAE feature. Returns None if no activations."""
    vals  = activations[fid]
    sts   = starts[fid]
    cidxs = chrom_idx[fid]

    n_active = int((vals > 0).sum())
    if n_active == 0:
        return None

    n_use = min(n_loci, n_active)
    loci = [
        (chrom_names[int(cidxs[r])], int(sts[r]))
        for r in range(n_use)
        if vals[r] > 0
    ]
    return summarize_feature(loci, lookup, resolution_bp=resolution_bp)


def display_feature(
    fid: int,
    activations: np.ndarray,
    starts: np.ndarray,
    chrom_idx: np.ndarray,
    chrom_names: list[str],
    lookup: ChromHMMLookup,
    n_loci: int,
    resolution_bp: int,
    n_features: int,
) -> None:
    vals = activations[fid]
    n_active = int((vals > 0).sum())
    n_slots  = len(vals)

    _print_feature_header(fid, n_active, n_slots, n_features)

    if n_active == 0:
        print("  No activations recorded for this feature.")
        _print_footer()
        return

    summary = compute_feature_summary(
        fid, activations, starts, chrom_idx, chrom_names,
        lookup, n_loci, resolution_bp,
    )
    if summary is None:
        print("  (summary unavailable)")
    else:
        _print_summary(summary, min(n_loci, n_active))
    _print_footer()


def main() -> None:
    args = parse_args()

    with h5py.File(args.input, "r") as f:
        activations   = f["activations"][:]
        starts        = f["starts"][:]
        chrom_idx     = f["chrom_idx"][:]
        chrom_names   = list(f["chrom_names"].asstr()[:])
        resolution_bp = int(f.attrs.get("resolution_bp", 128))
        n_features    = activations.shape[0]

    print(f"Loaded: {args.input}")
    print(f"  {n_features} features · {activations.shape[1]} top regions · "
          f"resolution {resolution_bp} bp · chroms: {', '.join(chrom_names)}")
    print(f"Opening ChromHMM data from {args.data_dir} …")

    if args.features:
        feature_ids = [int(x.strip()) for x in args.features.split(",")]
    else:
        feature_ids = list(range(args.start_from, n_features))

    batch = args.output is not None

    with ChromHMMLookup(args.data_dir) as lookup:
        print(f"  {len(lookup.eids)} cell types loaded.")

        if batch:
            # Non-interactive: compute all and write JSON
            print(f"Batch mode: computing summaries for {len(feature_ids)} features → {args.output}")
            results: dict[str, dict] = {}
            for i, fid in enumerate(feature_ids):
                if i % 500 == 0 and i > 0:
                    print(f"  {i}/{len(feature_ids)} features done …")
                summary = compute_feature_summary(
                    fid, activations, starts, chrom_idx, chrom_names,
                    lookup, args.n_loci, resolution_bp,
                )
                if summary is not None:
                    results[str(fid)] = summary
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as fh:
                json.dump(results, fh, indent=2)
            print(f"Wrote {len(results)} summaries to {args.output}")
            return

        # Interactive mode
        print(f"  Showing summaries for {len(feature_ids)} features using top {args.n_loci} loci each.")
        fid_to_pos = {fid: pos for pos, fid in enumerate(feature_ids)}
        i = 0
        while i < len(feature_ids):
            fid = feature_ids[i]
            display_feature(
                fid, activations, starts, chrom_idx, chrom_names,
                lookup, args.n_loci, resolution_bp, n_features,
            )

            try:
                cmd = input("\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if cmd == "q":
                break
            elif cmd.startswith("f "):
                try:
                    target = int(cmd[2:].strip())
                    if not (0 <= target < n_features):
                        print(f"  Feature ID must be 0–{n_features - 1}.")
                    elif target not in fid_to_pos:
                        feature_ids.insert(i + 1, target)
                        fid_to_pos[target] = i + 1
                        i += 1
                    else:
                        i = fid_to_pos[target]
                    continue
                except ValueError:
                    print("  Usage: f <integer id>")
            else:
                i += 1

    print("Done.")


if __name__ == "__main__":
    main()
