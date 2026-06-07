"""Interactive audit tool: step through SAE features one at a time and see
which Ensembl genes / regulatory elements overlap their top-activated regions.

Usage:
    python scripts/audit_features.py --input results/feature_top200.h5

Controls (at the '>' prompt):
    Enter         next feature
    f <id>        jump to a specific feature by index
    q             quit
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from genome_feature_atlas.ensembl import EnsemblFeature, query_region

_W = 68  # display width


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit SAE features against Ensembl annotations.")
    p.add_argument("--input", required=True, type=Path, help="feature_top200.h5 from scan_features.py")
    p.add_argument("--n-show", type=int, default=10, help="Regions to annotate per feature (default: 10)")
    p.add_argument("--features", type=str, default=None, help="Comma-separated feature IDs to inspect (default: all in order)")
    p.add_argument("--start-from", type=int, default=0, help="Start iteration at this feature index (default: 0)")
    return p.parse_args()


# ── display helpers ───────────────────────────────────────────────────────────

def _fmt_coord(chrom: str, start: int, end: int) -> str:
    return f"{chrom}:{start:,}-{end:,}"


def _strand_char(s: int) -> str:
    return "+" if s == 1 else "−" if s == -1 else "·"


def _print_feature_header(fid: int, n_active: int, n_total_slots: int, n_features: int) -> None:
    print(f"\n{'═' * _W}")
    print(f"  Feature {fid:<6}  {n_active}/{n_total_slots} slots with activation > 0   (of {n_features} features total)")
    print(f"{'═' * _W}")


def _print_region(rank: int, chrom: str, start: int, end: int, activation: float,
                  ens_features: list[EnsemblFeature]) -> None:
    coord = _fmt_coord(chrom, start, end)
    print(f"\n  [{rank + 1:>3}]  {coord:<35}  act={activation:.4f}")

    genes = [f for f in ens_features if f.feature_type == "gene"]
    regs  = [f for f in ens_features if f.feature_type == "regulatory"]

    if not genes and not regs:
        print("        (no overlapping Ensembl features)")
        return

    for g in genes:
        sym    = g.name or g.id
        biotype = g.biotype or "?"
        span    = _fmt_coord(g.chrom, g.start, g.end)
        strand  = _strand_char(g.strand)
        print(f"        gene  {sym:<22} {biotype:<20} {strand}  {span}")
        if g.description:
            truncated = g.description[:72]
            if len(g.description) > 72:
                truncated += "…"
            print(f"              {truncated}")

    for r in regs:
        subtype = r.name or r.id
        span    = _fmt_coord(r.chrom, r.start, r.end)
        print(f"        reg   {subtype:<22} {r.id:<20} {span}")


def _print_gene_summary(gene_counts: Counter, n_shown: int) -> None:
    if not gene_counts:
        return
    top = gene_counts.most_common(8)
    genes_str = "  ".join(f"{g} ({c}/{n_shown})" for g, c in top)
    print(f"\n  Genes across {n_shown} regions: {genes_str}")


def _print_footer() -> None:
    print(f"\n{'─' * _W}")
    print("  [Enter] next   [f <id>] jump to feature   [q] quit")


# ── core logic ────────────────────────────────────────────────────────────────

def display_feature(
    fid: int,
    activations: np.ndarray,
    starts: np.ndarray,
    chrom_idx: np.ndarray,
    chrom_names: list[str],
    n_show: int,
    resolution_bp: int,
    n_features: int,
) -> None:
    vals   = activations[fid]          # (n_top,) float32, sorted desc
    sts    = starts[fid]               # (n_top,) int32
    cidxs  = chrom_idx[fid]            # (n_top,) uint8

    active_mask = vals > 0
    n_active = int(active_mask.sum())
    n_slots  = len(vals)

    _print_feature_header(fid, n_active, n_slots, n_features)

    if n_active == 0:
        print("  No activations recorded for this feature.")
        _print_footer()
        return

    n_show_actual = min(n_show, n_active)
    gene_counts: Counter = Counter()

    for rank in range(n_show_actual):
        if vals[rank] <= 0:
            break
        chrom = chrom_names[int(cidxs[rank])]
        start = int(sts[rank])
        end   = start + resolution_bp

        try:
            ens = query_region(chrom, start, end)
        except Exception as exc:
            print(f"  [{rank + 1:>3}]  {_fmt_coord(chrom, start, end)}  act={float(vals[rank]):.4f}")
            print(f"        [Ensembl query failed: {exc}]")
            continue

        _print_region(rank, chrom, start, end, float(vals[rank]), ens)

        for f in ens:
            if f.feature_type == "gene" and f.name:
                gene_counts[f.name] += 1

    _print_gene_summary(gene_counts, n_show_actual)
    _print_footer()


def main() -> None:
    args = parse_args()

    with h5py.File(args.input, "r") as f:
        activations  = f["activations"][:]           # (n_features, n_top)
        starts       = f["starts"][:]
        chrom_idx    = f["chrom_idx"][:]
        chrom_names  = list(f["chrom_names"].asstr()[:])
        resolution_bp = int(f.attrs.get("resolution_bp", 128))
        n_features   = activations.shape[0]

    if args.features:
        feature_ids = [int(x.strip()) for x in args.features.split(",")]
    else:
        feature_ids = list(range(args.start_from, n_features))

    print(f"Loaded: {args.input}")
    print(f"  {n_features} features · {activations.shape[1]} top regions · "
          f"resolution {resolution_bp} bp · chroms: {', '.join(chrom_names)}")
    print(f"  Showing {args.n_show} regions per feature. "
          f"{'All' if not args.features else len(feature_ids)} features queued.")

    # Build a quick-lookup dict for jump commands
    fid_to_pos = {fid: pos for pos, fid in enumerate(feature_ids)}

    i = 0
    while i < len(feature_ids):
        fid = feature_ids[i]
        display_feature(
            fid, activations, starts, chrom_idx,
            chrom_names, args.n_show, resolution_bp, n_features,
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
                    # Not in current list — insert it as a one-off after the current position
                    feature_ids.insert(i + 1, target)
                    fid_to_pos[target] = i + 1
                    i += 1  # jump to it immediately
                else:
                    i = fid_to_pos[target]
                continue
            except ValueError:
                print("  Usage: f <integer id>")
        else:
            # Enter or anything else → advance
            i += 1

    print("Done.")


if __name__ == "__main__":
    main()
