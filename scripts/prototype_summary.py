"""Prototype: print LLM-agent-compatible annotation summaries for SAE features.

Loads top-activated loci from the feature scan HDF5, queries Ensembl and
ChromHMM, and prints a compact summary block for each requested feature.
A background reference section is printed at the end for comparison.

Usage:
    python scripts/prototype_summary.py \\
        --input  results/feature_top200.h5 \\
        --chromhmm-dir data/roadmap_chromhmm/hg19 \\
        --annotation-dir data/ensembl_grch37 \\
        --features 42,100,500 \\
        --n-loci 200 \\
        [--all-features] \\
        [--annotate] \\
        [--background-json results/prev_background.json] \\
        [--output-json summaries.json]

If --features is omitted the first 5 active features are used (quick sanity check).
If --all-features is set, all active features are processed (may be slow with ChromHMM).
If --chromhmm-dir is omitted, ChromHMM tissue data is skipped.
If --annotation-dir is omitted, falls back to the Ensembl REST API (slower).
If --annotate is set, each feature summary is passed to AnnotatorAgent (requires vLLM).
If --background-json is set, the pre-computed background is used as LLM context.

Two-pass workflow for best annotation quality:
  # Step 1: build background from a large batch (no LLM)
  python scripts/prototype_summary.py --input ... --features 0-199 \\
      --output-json results/batch.json
  # → auto-saves results/batch_background.json

  # Step 2: annotate with full background context
  python scripts/prototype_summary.py --input ... --features 45,46,47 \\
      --annotate --background-json results/batch_background.json
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genome_feature_atlas.annotation_summary import FeatureSummary, annotate_loci
from genome_feature_atlas.chromhmm import STATE_NAMES
from genome_feature_atlas.chromhmm import ChromHMMLookup
from genome_feature_atlas.feature_loci import FeatureLociLoader
from genome_feature_atlas.local_ensembl import LocalAnnotationLookup

_DIV  = "─" * 72
_DIV2 = "─" * 72


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print LLM-ready annotation summaries for SAE features.")
    p.add_argument("--input",          required=True, type=Path,
                   help="feature_top200.h5 from scan_features.py")
    p.add_argument("--chromhmm-dir",   type=Path, default=None,
                   help="Directory of E*_15_coreMarks_dense.bed.bgz files (optional)")
    p.add_argument("--annotation-dir", type=Path, default=None,
                   help="Directory of local Ensembl GFF files from setup_local_annotations.sh "
                        "(optional; falls back to REST API if omitted)")
    p.add_argument("--features",       type=str,  default=None,
                   help="Comma-separated feature IDs (mutually exclusive with --all-features)")
    p.add_argument("--all-features",   action="store_true",
                   help="Process every active feature (may take a long time with ChromHMM)")
    p.add_argument("--n-loci",         type=int,  default=200,
                   help="Top loci per feature (default: 200)")
    p.add_argument("--annotate",        action="store_true",
                   help="Run AnnotatorAgent on each feature; requires vLLM at http://localhost:8000/v1")
    p.add_argument("--background-json", type=Path, default=None,
                   help="Pre-computed background JSON saved by a prior --output-json run "
                        "(auto-saved as <stem>_background.json). Provides structure/TSS/chromatin "
                        "context to the annotator; without it, only activation stats are sent.")
    p.add_argument("--output-json",    type=Path, default=None,
                   help="Write batch JSON output to this file; also auto-saves "
                        "<stem>_background.json when ≥5 features are processed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.features and args.all_features:
        print("Error: --features and --all-features are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    loader = FeatureLociLoader(args.input)
    print(f"Loaded: {args.input}")
    print(f"  {loader.n_features} features · resolution {loader.resolution_bp} bp")

    if args.all_features:
        feature_ids = loader.active_feature_ids
        print(f"  --all-features: {len(feature_ids)} active features to process")
    elif args.features:
        feature_ids = _parse_feature_ids(args.features)
    else:
        feature_ids = loader.active_feature_ids[:5]
        print(f"  No --features specified; using first 5 active: {feature_ids}")

    chromhmm_ctx   = ChromHMMLookup(args.chromhmm_dir) if args.chromhmm_dir else None
    annotation_ctx = LocalAnnotationLookup(args.annotation_dir) if args.annotation_dir else None

    if annotation_ctx is not None:
        print(f"  Annotation source: local files in {args.annotation_dir}")
    else:
        print("  Annotation source: Ensembl REST API (rate-limited; use --annotation-dir for speed)")

    # Activation background: fast numpy op covering all active features.
    act_bg = loader.global_activation_background(args.n_loci)

    # Annotated background: load from a prior run if available.
    if args.background_json:
        with open(args.background_json) as _f:
            _saved = json.load(_f)
        loaded_anno_bg = _saved.get("annotated", {})
        print(f"  Background: loaded from {args.background_json} "
              f"({loaded_anno_bg.get('n_features', 0)} features)")
    else:
        loaded_anno_bg = {}
        if args.annotate:
            print("  Background: activation stats only (no --background-json; "
                  "structure/TSS/chromatin context missing)")

    bg_text = _fmt_background_for_agent(act_bg, loaded_anno_bg)

    if args.annotate:
        from genome_feature_atlas.agents import AnnotatorAgent, AgentContext
        annotator_agent: AnnotatorAgent | None = AnnotatorAgent()
        print("  Annotator: AnnotatorAgent (vLLM at http://localhost:8000/v1)")
    else:
        annotator_agent = None

    summaries: list[FeatureSummary] = []
    results:   dict[int, dict]      = {}

    try:
        for fid in feature_ids:
            loci_with_act = loader.get_loci(fid, n_loci=args.n_loci)
            if not loci_with_act:
                print(f"\n{_DIV}\nFeature {fid}: no active loci\n")
                continue

            loci = [(chrom, start, end) for chrom, start, end, _act in loci_with_act]

            summary = annotate_loci(
                loci,
                chromhmm_lookup=chromhmm_ctx,
                annotation_lookup=annotation_ctx,
                activation_stats=loader.activation_stats(fid, n_loci=args.n_loci),
                feature_id=fid,
                resolution_bp=loader.resolution_bp,
            )

            print(f"\n{_DIV}")
            print(summary.to_text())

            if annotator_agent is not None:
                ctx = AgentContext(feature_summary=summary, background_text=bg_text)
                ann = annotator_agent.run(ctx)
                print(json.dumps(ann, indent=2, ensure_ascii=False))
                if args.output_json:
                    results[fid] = {**summary.to_dict(), "annotation": ann}
            elif args.output_json:
                results[fid] = summary.to_dict()

            summaries.append(summary)
    finally:
        if chromhmm_ctx is not None:
            chromhmm_ctx.close()
        if annotation_ctx is not None:
            annotation_ctx.close()

    # ── background reference ───────────────────────────────────────────────────
    anno_bg = _annotated_background(summaries)
    print(f"\n{_DIV2}")
    print(_fmt_background(act_bg, anno_bg))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2, ensure_ascii=False)
        print(f"JSON written to {args.output_json}")

        if anno_bg.get("n_features", 0) >= 5:
            bg_save_path = args.output_json.with_name(args.output_json.stem + "_background.json")
            with open(bg_save_path, "w") as f:
                json.dump({"activation": act_bg, "annotated": anno_bg}, f, indent=2)
            print(f"Background saved to {bg_save_path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_feature_ids(spec: str) -> list[int]:
    """Parse a comma-separated list of IDs and/or N-M ranges, e.g. '0-100,200,500-600'."""
    ids: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = token.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(token))
    return ids


# ── background helpers ─────────────────────────────────────────────────────────

def _annotated_background(summaries: list[FeatureSummary]) -> dict:
    """Aggregate structure, TSS, and chromatin stats from processed summaries."""
    if not summaries:
        return {}

    # Gene structure: pool raw counts, then compute fractions
    struct_pool: Counter = Counter()
    total_loci = 0
    for s in summaries:
        for label, count in s.gene_structure.items():
            struct_pool[label] += count
        total_loci += s.n_loci
    struct_fracs = {k: v / total_loci for k, v in struct_pool.items()} if total_loci else {}

    # TSS distances: median of per-feature medians, p10/p90 from pooled list (capped at 50k values)
    all_tss: list[int] = []
    for s in summaries:
        all_tss.extend(s.tss_distances_bp[:500])  # cap contribution per feature
    tss_stats = None
    if all_tss:
        all_tss.sort()
        n = len(all_tss)
        tss_stats = {
            "median": all_tss[n // 2],
            "p10":    all_tss[max(0, int(n * 0.10))],
            "p90":    all_tss[min(n - 1, int(n * 0.90))],
        }

    # ChromHMM: average fraction of each state per tissue across summaries
    tissue_state_sums:  dict[str, Counter] = defaultdict(Counter)
    tissue_counts:      Counter = Counter()
    for s in summaries:
        for tissue, m in s.tissue_matrix.items():
            for state, frac in m["top_states"]:
                tissue_state_sums[tissue][state] += frac
            tissue_counts[tissue] += 1

    tissue_avg: dict[str, list[tuple[str, float]]] = {}
    for tissue, state_sums in tissue_state_sums.items():
        n = tissue_counts[tissue]
        sorted_states = sorted(state_sums.items(), key=lambda x: -x[1])
        tissue_avg[tissue] = [(s, round(v / n, 3)) for s, v in sorted_states[:3]]

    return {
        "n_features":   len(summaries),
        "struct_fracs": struct_fracs,
        "tss_stats":    tss_stats,
        "tissue_avg":   tissue_avg,
    }


def _fmt_background_for_agent(act_bg: dict, anno_bg: dict) -> str:
    """Compact background block for LLM consumption — no placeholder messages."""
    lines = ["── Background " + "─" * 58]

    n_act = act_bg.get("n_active_features", 0)
    lines.append(f"Activation ({n_act:,} active features):")
    for key, label in (("max_val", "max"), ("flatness", "flatness"), ("windows", "windows")):
        d = act_bg.get(key, {})
        p25, p50, p75 = d.get("p25", 0), d.get("p50", 0), d.get("p75", 0)
        if key == "flatness":
            lines.append(f"  {label:<10} p50={p50:.0%}   [p25={p25:.0%}, p75={p75:.0%}]")
        elif key == "windows":
            lines.append(f"  {label:<10} p50={int(p50):,}  [p25={int(p25):,}, p75={int(p75):,}]")
        else:
            lines.append(f"  {label:<10} p50={p50:.2f}  [p25={p25:.2f}, p75={p75:.2f}]")

    n_anno = anno_bg.get("n_features", 0)
    if n_anno >= 5:
        lines.append(f"Annotated background ({n_anno} features):")
        struct = anno_bg.get("struct_fracs", {})
        _ORDER = ("CDS", "UTR", "exon", "intron", "intergenic")
        struct_parts = [f"{lbl} {struct[lbl]:.0%}" for lbl in _ORDER if lbl in struct]
        if struct_parts:
            lines.append(f"  Structure:  {'  '.join(struct_parts)}")
        tss = anno_bg.get("tss_stats")
        if tss:
            def _fd(bp: int) -> str:
                s = "+" if bp >= 0 else "-"
                a = abs(bp)
                return f"{s}{a/1000:.1f}kb" if a >= 1000 else f"{s}{a}bp"
            lines.append(
                f"  TSS dist:   median {_fd(tss['median'])}  "
                f"[p10 {_fd(tss['p10'])}, p90 {_fd(tss['p90'])}]"
            )
        tissue_avg = anno_bg.get("tissue_avg", {})
        if tissue_avg:
            lines.append("  Chromatin:")
            for tissue, states in tissue_avg.items():
                state_parts = [f"{STATE_NAMES.get(s, s)}({f:.0%})" for s, f in states]
                lines.append(f"    {tissue:<20} {'  '.join(state_parts)}")

    return "\n".join(lines)


def _fmt_background(act_bg: dict, anno_bg: dict) -> str:
    lines = ["── Background " + "─" * 58]

    # Activation background (all active features)
    n_act = act_bg.get("n_active_features", 0)
    lines.append(f"Activation ({n_act:,} active features):")
    for key, label in (("max_val", "max"), ("flatness", "flatness"), ("windows", "windows")):
        d = act_bg.get(key, {})
        p25, p50, p75 = d.get("p25", 0), d.get("p50", 0), d.get("p75", 0)
        if key == "flatness":
            lines.append(f"  {label:<10} p50={p50:.0%}   [p25={p25:.0%}, p75={p75:.0%}]")
        elif key == "windows":
            lines.append(f"  {label:<10} p50={int(p50):,}  [p25={int(p25):,}, p75={int(p75):,}]")
        else:
            lines.append(f"  {label:<10} p50={p50:.2f}  [p25={p25:.2f}, p75={p75:.2f}]")

    # Annotated background (processed features only)
    n_anno = anno_bg.get("n_features", 0)
    if n_anno < 5:
        lines.append(f"Annotated background: only {n_anno} feature(s) processed — run more for meaningful stats.")
        return "\n".join(lines)

    lines.append(f"Annotated background ({n_anno} features):")

    # Structure
    struct = anno_bg.get("struct_fracs", {})
    _ORDER = ("CDS", "UTR", "exon", "intron", "intergenic")
    struct_parts = [f"{lbl} {struct[lbl]:.0%}" for lbl in _ORDER if lbl in struct]
    if struct_parts:
        lines.append(f"  Structure:  {'  '.join(struct_parts)}")

    # TSS
    tss = anno_bg.get("tss_stats")
    if tss:
        def _fd(bp: int) -> str:
            s = "+" if bp >= 0 else "-"
            a = abs(bp)
            return f"{s}{a/1000:.1f}kb" if a >= 1000 else f"{s}{a}bp"
        lines.append(
            f"  TSS dist:   median {_fd(tss['median'])}  "
            f"[p10 {_fd(tss['p10'])}, p90 {_fd(tss['p90'])}]"
        )

    # ChromHMM
    tissue_avg = anno_bg.get("tissue_avg", {})
    if tissue_avg:
        lines.append("  Chromatin:")
        for tissue, states in tissue_avg.items():
            state_parts = [f"{STATE_NAMES.get(s, s)}({f:.0%})" for s, f in states]
            lines.append(f"    {tissue:<20} {'  '.join(state_parts)}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
