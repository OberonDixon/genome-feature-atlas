"""ChromHMM state lookup over Roadmap Epigenomics 15-state core marks (hg19).

Discovers all E*_15_coreMarks_dense.bed.bgz files in a data directory and
provides fast tabix-backed lookups. Cell types are grouped into 12 higher-level
tissues; per-feature summaries aggregate state distributions across tissues for
downstream agentic annotation.

Usage:
    from genome_feature_atlas.chromhmm import ChromHMMLookup, summarize_feature

    with ChromHMMLookup("data/roadmap_chromhmm/hg19") as lookup:
        # Single locus query — returns {EID: state_name} for all 127 cells
        states = lookup.query_region("chr1", 1_000_000, 1_001_000)

        # Feature summary over a list of (chrom, start) loci
        loci = [("chr1", 1_000_000), ("chr7", 5_412_800)]
        summary = summarize_feature(loci, lookup, resolution_bp=128)

Coordinates follow the 0-based half-open convention used throughout this project.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pysam


# ── tissue groupings ─────────────────────────────────────────────────────────
# 127 EIDs present in the Roadmap data (E060 and E064 absent from release).

TISSUE_GROUPS: dict[str, list[str]] = {
    "ESC/iPSC": [
        "E001", "E002", "E003", "E004", "E005", "E006", "E007", "E008",
        "E009", "E010", "E011", "E012", "E013", "E014", "E015", "E016",
        "E018", "E019", "E020", "E021", "E022", "E024",
    ],
    "Blood/Immune": [
        "E029", "E030", "E031", "E032", "E033", "E034", "E035", "E036",
        "E037", "E038", "E039", "E040", "E041", "E042", "E043", "E044",
        "E045", "E046", "E047", "E048", "E050", "E051", "E062",
        "E112", "E113", "E115", "E116", "E123", "E124",
    ],
    "Brain": [
        "E053", "E054", "E067", "E068", "E069", "E070", "E071", "E072",
        "E073", "E074", "E081", "E082", "E125",
    ],
    "Muscle": [
        "E052", "E089", "E090", "E100", "E107", "E108", "E120", "E121",
    ],
    "Heart/Vasculature": [
        "E065", "E083", "E095", "E104", "E105", "E122",
    ],
    "GI_Tract": [
        "E075", "E076", "E077", "E078", "E079", "E084", "E085", "E092",
        "E094", "E101", "E102", "E103", "E106", "E109", "E110", "E111",
    ],
    "Liver/Pancreas": [
        "E066", "E087", "E098", "E118",
    ],
    "Lung": [
        "E017", "E088", "E096", "E114", "E128",
    ],
    "Skin/Fibroblast": [
        "E055", "E056", "E057", "E058", "E059", "E061", "E126", "E127",
    ],
    "Adipose/MSC": [
        "E023", "E025", "E026", "E049", "E063", "E129",
    ],
    "Reproductive/Fetal": [
        "E080", "E086", "E091", "E093", "E097", "E099", "E117",
    ],
    "Breast": [
        "E027", "E028", "E119",
    ],
}

# States that count as quiescent/inactive for active_fraction calculation.
QUIESCENT_STATES: frozenset[str] = frozenset({"15_Quies"})

_EID_RE = re.compile(r"^(E\d{3})_")


# ── lookup class ─────────────────────────────────────────────────────────────

class ChromHMMLookup:
    """Thin wrapper around pysam.TabixFile for all Roadmap ChromHMM cell types.

    Keeps one open file handle per EID for the lifetime of the object (or until
    close() is called). Context manager protocol is supported.

    Args:
        data_dir: directory containing E*_15_coreMarks_dense.bed.bgz files and
                  their .tbi indices.
    """

    def __init__(self, data_dir: Path | str) -> None:
        data_dir = Path(data_dir)
        bgz_files = sorted(data_dir.glob("E*_15_coreMarks_dense.bed.bgz"))
        if not bgz_files:
            raise FileNotFoundError(
                f"No E*_15_coreMarks_dense.bed.bgz files found in {data_dir}"
            )
        self._handles: dict[str, pysam.TabixFile] = {}
        for path in bgz_files:
            m = _EID_RE.match(path.name)
            if m:
                eid = m.group(1)
                self._handles[eid] = pysam.TabixFile(str(path))

    def query_region(self, chrom: str, start: int, end: int) -> dict[str, str]:
        """Return {EID: state_name} for every cell type at this locus.

        Args:
            chrom:  chromosome with "chr" prefix (e.g. "chr1")
            start:  0-based inclusive start
            end:    0-based exclusive end

        Returns:
            Dict mapping each EID to its ChromHMM state string (e.g. "7_Enh").
            Missing overlaps default to "15_Quies".
        """
        result: dict[str, str] = {}
        for eid, tbx in self._handles.items():
            try:
                rows = list(tbx.fetch(chrom, start, end))
            except ValueError:
                # chromosome not in index (e.g. chrM)
                result[eid] = "15_Quies"
                continue
            if rows:
                fields = rows[0].split("\t")
                result[eid] = fields[3]
            else:
                result[eid] = "15_Quies"
        return result

    @property
    def eids(self) -> list[str]:
        return sorted(self._handles)

    def close(self) -> None:
        for tbx in self._handles.values():
            tbx.close()
        self._handles.clear()

    def __enter__(self) -> "ChromHMMLookup":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ── aggregation ───────────────────────────────────────────────────────────────

def summarize_feature(
    loci: list[tuple[str, int]],
    lookup: ChromHMMLookup,
    resolution_bp: int = 128,
    active_threshold: float = 0.1,
    min_state_frac: float = 0.05,
) -> dict:
    """Aggregate ChromHMM states across loci and tissues into a compact summary.

    For each tissue group the function pools states across all
    (locus × cells_in_group) pairs, then computes:
      - dominant state (most common, may be 15_Quies)
      - active_fraction (fraction of pairs not in QUIESCENT_STATES)
      - top_non_quies (state fractions for non-quiescent states ≥ min_state_frac)

    Args:
        loci:             list of (chrom, start) 0-based positions
        lookup:           open ChromHMMLookup
        resolution_bp:    bin width in bp
        active_threshold: active_fraction cutoff for active_in / inactive_in
        min_state_frac:   minimum fraction for a state to appear in top_non_quies

    Returns:
        {
            "active_in":  [tissue, ...],
            "inactive_in": [tissue, ...],
            "matrix": {
                tissue: {
                    "dominant": state,
                    "active_fraction": float,
                    "top_non_quies": {state: fraction, ...}
                },
                ...
            }
        }
    """
    # accumulate {tissue: [state, ...]} across all loci
    tissue_states: dict[str, list[str]] = {t: [] for t in TISSUE_GROUPS}

    for chrom, start in loci:
        eid_states = lookup.query_region(chrom, start, start + resolution_bp)
        for tissue, eids in TISSUE_GROUPS.items():
            for eid in eids:
                tissue_states[tissue].append(eid_states.get(eid, "15_Quies"))

    matrix: dict[str, dict] = {}
    for tissue, states in tissue_states.items():
        n = len(states)
        if n == 0:
            continue
        counts = Counter(states)
        dominant = counts.most_common(1)[0][0]
        active_n = sum(c for s, c in counts.items() if s not in QUIESCENT_STATES)
        active_fraction = active_n / n
        top_non_quies = {
            s: round(c / n, 3)
            for s, c in counts.most_common()
            if s not in QUIESCENT_STATES and c / n >= min_state_frac
        }
        matrix[tissue] = {
            "dominant": dominant,
            "active_fraction": round(active_fraction, 3),
            "top_non_quies": top_non_quies,
        }

    active_in = [t for t in TISSUE_GROUPS if matrix.get(t, {}).get("active_fraction", 0) >= active_threshold]
    inactive_in = [t for t in TISSUE_GROUPS if matrix.get(t, {}).get("active_fraction", 1) < active_threshold]

    return {
        "active_in": active_in,
        "inactive_in": inactive_in,
        "matrix": matrix,
    }
