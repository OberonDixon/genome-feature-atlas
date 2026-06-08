"""Unified annotation summary combining Ensembl and ChromHMM for LLM agents.

For a list of genomic loci this module:
  1. Queries the Ensembl REST API for genes, regulatory elements, and TF motifs
  2. Aggregates counts / scores across all loci
  3. Queries ChromHMM epigenetic state via ChromHMMLookup (optional)
  4. Returns a FeatureSummary with both a machine-readable dict and a compact
     text block suitable for direct injection into an LLM agent prompt.

Usage:
    from genome_feature_atlas.annotation_summary import annotate_loci
    from genome_feature_atlas.chromhmm import ChromHMMLookup

    loci = [("chr7", 117_548_546, 117_548_674), ...]  # (chrom, start, end)

    with ChromHMMLookup("data/roadmap_chromhmm/hg19") as lookup:
        summary = annotate_loci(loci, chromhmm_lookup=lookup, feature_id=42)

    print(summary.to_text())
    import json; print(json.dumps(summary.to_dict(), indent=2))

Coordinates follow the 0-based half-open convention used throughout this project.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from genome_feature_atlas import chromhmm as _chromhmm
from genome_feature_atlas import ensembl as _ensembl

# Structural labels in display order (highest priority first)
_STRUCT_ORDER = ("CDS", "UTR", "exon", "intron", "intergenic")


@dataclass
class FeatureSummary:
    """Aggregated biological annotation for a set of genomic loci.

    Typically the top-N activated loci for a single SAE feature.

    Fields:
        feature_id:       SAE feature index (None if called with raw loci)
        n_loci:           number of loci that were annotated
        chromosomes:      unique chromosomes present, sorted by frequency desc

        gene_structure:   {label: count} where label ∈ {CDS, UTR, exon, intron, intergenic}
                          Classification priority: CDS > UTR > exon > intron > intergenic.
                          UTR = exon of a protein_coding gene with no CDS overlap.
        tss_distances_bp: signed TSS distances (bp) for loci that overlap ≥1 gene.
                          Positive = downstream of TSS; negative = upstream (promoter-proximal).
                          Nearest-TSS is chosen when multiple genes overlap a locus.

        top_regulatory:   (subtype, count) sorted by count desc
        top_motifs:       (tf_name, avg_score, count) sorted by count desc

        tissue_matrix:    {tissue: {"top_states": [(state_key, fraction), ...]}}
                          from chromhmm.summarize_feature(); empty if no lookup provided
    """

    feature_id: int | None
    n_loci: int
    chromosomes: list[str]

    activation_stats: dict | None   # from FeatureLociLoader.activation_stats()

    gene_structure: dict[str, int]
    tss_distances_bp: list[int]

    top_regulatory: list[tuple[str, int]]
    top_motifs: list[tuple[str, float, int]]

    tissue_matrix: dict

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        tss_stats = _tss_distance_stats(self.tss_distances_bp)
        return {
            "feature_id": self.feature_id,
            "n_loci": self.n_loci,
            "chromosomes": _chrom_dict(self.chromosomes),
            "activation_stats": self.activation_stats,
            "gene_structure": self.gene_structure,
            "tss_distance": tss_stats,
            "top_regulatory": [
                {"subtype": sub, "count": cnt}
                for sub, cnt in self.top_regulatory
            ],
            "top_motifs": [
                {"tf": tf, "avg_score": round(sc, 3), "count": cnt}
                for tf, sc, cnt in self.top_motifs
            ],
            "tissue_matrix": {
                tissue: {
                    "top_states": [
                        {"state": s, "name": _chromhmm.STATE_NAMES.get(s, s), "fraction": f}
                        for s, f in m["top_states"]
                    ]
                }
                for tissue, m in self.tissue_matrix.items()
            },
        }

    def to_text(self) -> str:
        """Return a compact multi-line summary for LLM prompt injection."""
        lines: list[str] = []

        # ── header ────────────────────────────────────────────────────────────
        fid_str = f"Feature {self.feature_id}" if self.feature_id is not None else "Loci"
        chrom_parts = _chrom_summary(self.chromosomes)
        lines.append(f"{fid_str} | {self.n_loci} loci | {_join(chrom_parts)}")

        # ── activation stats ──────────────────────────────────────────────────
        if self.activation_stats:
            st = self.activation_stats
            flat_pct = f"{st['flatness']:.0%}"
            act_str = f"Activation: {st['max_val']:.2f}→{st['min_shown_val']:.2f}  flatness={flat_pct}"
            total = st.get("total_active_windows")
            if total:
                shown_pct = self.n_loci / total
                stride_kb = st.get("window_stride_bp", 0) // 1000
                stride_note = f"  ~{stride_kb}kb each" if stride_kb else ""
                act_str += f"  |  {self.n_loci}/{total} windows ({shown_pct:.0%}{stride_note})"
            lines.append(act_str)

        # ── chromatin ─────────────────────────────────────────────────────────
        if self.tissue_matrix:
            lines.append("Chromatin:")
            for tissue, m in self.tissue_matrix.items():
                state_parts = [
                    f"{_chromhmm.STATE_NAMES.get(s, s)}({f:.0%})"
                    for s, f in m["top_states"]
                ]
                lines.append(f"  {tissue:<20} {_join(state_parts)}")

        # ── gene structure ────────────────────────────────────────────────────
        struct_parts = [
            f"{lbl}(×{self.gene_structure[lbl]})"
            for lbl in _STRUCT_ORDER
            if lbl in self.gene_structure
        ]
        tss_str = ""
        if self.tss_distances_bp:
            stats = _tss_distance_stats(self.tss_distances_bp)
            tss_str = (
                f"  |  TSS med={_fmt_dist(stats['median'])}"
                f"  [{_fmt_dist(stats['p10'])}, {_fmt_dist(stats['p90'])}]"
            )
        lines.append(f"Genes:      {_join(struct_parts)}{tss_str}")

        # ── regulatory ────────────────────────────────────────────────────────
        if self.top_regulatory:
            parts = [f"{sub}(×{cnt})" for sub, cnt in self.top_regulatory]
            lines.append(f"Regulatory: {_join(parts)}")

        # ── TF motifs ─────────────────────────────────────────────────────────
        if self.top_motifs:
            parts = [f"{tf}(avg_sc={sc:.1f},×{cnt})" for tf, sc, cnt in self.top_motifs]
            lines.append(f"Motifs:     {_join(parts)}")

        return "\n".join(lines)


def annotate_loci(
    loci: list[tuple[str, int, int]],
    chromhmm_lookup: _chromhmm.ChromHMMLookup | None = None,
    annotation_lookup=None,  # LocalAnnotationLookup | None
    activation_stats: dict | None = None,
    feature_id: int | None = None,
    resolution_bp: int = 128,
    max_motifs: int = 6,
    max_regulatory: int = 6,
) -> FeatureSummary:
    """Compile a combined Ensembl + ChromHMM annotation summary for a locus set.

    Args:
        loci:               list of (chrom, start, end) 0-based half-open
        chromhmm_lookup:    open ChromHMMLookup; if None, tissue_matrix is empty
        annotation_lookup:  open LocalAnnotationLookup (fast, no rate limit);
                            if None, falls back to the Ensembl REST API
        feature_id:         SAE feature index to embed in the summary (informational)
        resolution_bp:      genomic bin width; passed to chromhmm.summarize_feature()
        max_genes:          top-N genes to retain
        max_motifs:         top-N TF motifs to retain
        max_regulatory:     top-N regulatory subtypes to retain

    Returns:
        FeatureSummary with structured data and text rendering methods.
    """
    gene_structure:   Counter = Counter()
    tss_distances_bp: list[int] = []
    reg_counts:       Counter = Counter()
    motif_counts:     Counter = Counter()
    motif_total_sc:   dict[str, float] = defaultdict(float)

    chroms: list[str] = [chrom for chrom, _, _ in loci]

    if annotation_lookup is not None:
        # Local tabix files: parallel across source files, no rate limit
        ensembl_results = annotation_lookup.query_all_loci(loci)
    else:
        # Ensembl REST API: parallel across loci, rate-limited to 12 req/s
        with ThreadPoolExecutor(max_workers=len(loci)) as pool:
            ensembl_results = list(pool.map(_fetch_locus, loci))

    for (chrom, start, end), features in zip(loci, ensembl_results):
        locus_center = (start + end) // 2

        has_cds = False
        has_exon = False
        has_protein_coding = False
        # Collect both gene and transcript features: transcript features span
        # the full gene body and serve as a reliable fallback when the parent
        # gene record is absent (e.g. pseudogene types not captured by awk filter).
        gene_tx_features = []

        for f in features:
            if f.feature_type == "cds":
                has_cds = True
            elif f.feature_type == "exon":
                has_exon = True
            elif f.feature_type in ("gene", "transcript"):
                gene_tx_features.append(f)
                if f.biotype == "protein_coding":
                    has_protein_coding = True
            elif f.feature_type == "regulatory" and f.name:
                reg_counts[f.name] += 1
            elif f.feature_type == "motif" and f.name:
                motif_counts[f.name] += 1
                motif_total_sc[f.name] += _parse_score(f.description)

        # Structural classification (priority: CDS > UTR > exon > intron > intergenic)
        # UTR = exon of a protein_coding gene without CDS overlap.
        if has_cds:
            struct_label = "CDS"
        elif has_exon and has_protein_coding:
            struct_label = "UTR"
        elif has_exon:
            struct_label = "exon"
        elif gene_tx_features:
            struct_label = "intron"
        else:
            struct_label = "intergenic"
        gene_structure[struct_label] += 1

        # Signed TSS distance for genic loci (positive = downstream of TSS).
        # When multiple transcripts overlap, pick the one with the nearest TSS.
        # f.start / f.end are 1-based GFF coordinates; locus_center is ~0-based;
        # the 1bp offset is negligible at genomic distance scales.
        if gene_tx_features:
            best_dist: int | None = None
            for f in gene_tx_features:
                tss = f.start if f.strand >= 0 else f.end
                raw = locus_center - tss
                dist = raw if f.strand >= 0 else -raw
                if best_dist is None or abs(dist) < abs(best_dist):
                    best_dist = dist
            if best_dist is not None:
                tss_distances_bp.append(best_dist)

    # ── sort and trim ─────────────────────────────────────────────────────────
    top_regulatory = list(reg_counts.most_common(max_regulatory))
    top_motifs = [
        (tf, motif_total_sc[tf] / cnt, cnt)
        for tf, cnt in motif_counts.most_common(max_motifs)
    ]

    # ── chromatin ─────────────────────────────────────────────────────────────
    if chromhmm_lookup is not None:
        chrom_loci = [(chrom, start) for chrom, start, _end in loci]
        chmm = _chromhmm.summarize_feature(
            chrom_loci, chromhmm_lookup, resolution_bp=resolution_bp, n_top_states=3,
        )
        tissue_matrix = chmm["matrix"]
    else:
        tissue_matrix = {}

    return FeatureSummary(
        feature_id=feature_id,
        n_loci=len(loci),
        chromosomes=chroms,  # full per-locus list; Counter computed on demand in to_text/to_dict
        activation_stats=activation_stats,
        gene_structure=dict(gene_structure),
        tss_distances_bp=tss_distances_bp,
        top_regulatory=top_regulatory,
        top_motifs=top_motifs,
        tissue_matrix=tissue_matrix,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _chrom_dict(chromosomes: list[str]) -> dict:
    """Structured chromosome summary for to_dict()."""
    counts = Counter(chromosomes)
    result = {"chrX": counts.get("chrX", 0)}
    top_auto = _top_autosome(counts, len(chromosomes))
    if top_auto:
        result["top_autosome"] = {"chrom": top_auto[0], "count": top_auto[1]}
    return result


def _chrom_summary(chromosomes: list[str]) -> list[str]:
    """Compact chromosome fields for the header line of to_text()."""
    counts = Counter(chromosomes)
    parts = [f"chrX:{counts.get('chrX', 0)}"]
    top_auto = _top_autosome(counts, len(chromosomes))
    if top_auto:
        parts.append(f"top-auto:{top_auto[0]}(×{top_auto[1]})")
    return parts


def _top_autosome(
    counts: Counter,
    n_loci: int,
    _n_autosomes: int = 22,
) -> tuple[str, int] | None:
    """Return (chrom, count) for the most frequent autosome if it's notably over-represented.

    Threshold: 3× the uniform expectation across autosomes, minimum 3 loci.
    This highlights chromosome-specific features (repeats, centromeres, etc.)
    while suppressing noise from small locus sets.
    """
    threshold = max(3, n_loci * 3 / _n_autosomes)
    autosomes = {c: n for c, n in counts.items() if c not in ("chrX", "chrY")}
    if not autosomes:
        return None
    top_chr, top_n = max(autosomes.items(), key=lambda x: x[1])
    return (top_chr, top_n) if top_n >= threshold else None


def _fetch_locus(locus: tuple[str, int, int]) -> list[_ensembl.EnsemblFeature]:
    """Query Ensembl for one locus; returns empty list on any error."""
    chrom, start, end = locus
    try:
        return _ensembl.query_region(chrom, start, end)
    except Exception:
        return []


def _join(parts: list[str], sep: str = "  ") -> str:
    return sep.join(parts)


def _parse_score(description: str) -> float:
    """Extract a numeric score from strings like 'score=12.34'."""
    if "score=" in description:
        try:
            return float(description.split("score=")[1].split()[0])
        except (IndexError, ValueError):
            pass
    return 0.0


def _tss_distance_stats(distances: list[int]) -> dict | None:
    """Return {n, median, p10, p90} for a list of TSS distances in bp."""
    if not distances:
        return None
    n = len(distances)
    s = sorted(distances)
    return {
        "n": n,
        "median": int(statistics.median(s)),
        "p10": s[max(0, int(n * 0.10))],
        "p90": s[min(n - 1, int(n * 0.90))],
    }


def _fmt_dist(bp: int) -> str:
    """Format a signed bp distance as '+2.3kb' or '-500bp'."""
    sign = "+" if bp >= 0 else "-"
    abs_bp = abs(bp)
    if abs_bp >= 1000:
        return f"{sign}{abs_bp / 1000:.1f}kb"
    return f"{sign}{abs_bp}bp"
