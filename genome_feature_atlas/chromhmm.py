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
from concurrent.futures import ThreadPoolExecutor
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

# States considered quiescent (only used to define the boundary of the state space).
QUIESCENT_STATES: frozenset[str] = frozenset({"15_Quies"})

STATE_NAMES: dict[str, str] = {
    "1_TssA":     "Active TSS",
    "2_TssAFlnk": "Flanking TSS",
    "3_TxFlnk":   "Tx 5'/3'",
    "4_Tx":        "Strong transcription",
    "5_TxWk":      "Weak transcription",
    "6_EnhG":      "Genic enhancer",
    "7_Enh":       "Distal enhancer",
    "8_ZNF/Rpts":  "ZNF/Repeats",
    "9_Het":       "Heterochromatin",
    "10_TssBiv":   "Bivalent TSS",
    "11_BivFlnk":  "Flanking bivalent",
    "12_EnhBiv":   "Bivalent enhancer",
    "13_ReprPC":   "Polycomb repressed",
    "14_ReprPCWk": "Weak Polycomb",
    "15_Quies":    "Quiescent",
}

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
    n_top_states: int = 2,
) -> dict:
    """Aggregate ChromHMM states across loci and tissues into a compact summary.

    For each tissue group the function pools states across all
    (locus × cells_in_group) pairs, then returns the top-N states by fraction.

    Args:
        loci:          list of (chrom, start) 0-based positions
        lookup:        open ChromHMMLookup
        resolution_bp: bin width in bp
        n_top_states:  how many top states to retain per tissue

    Returns:
        {
            "matrix": {
                tissue: {
                    "top_states": [(state_key, fraction), ...]  # n_top_states entries
                },
                ...
            }
        }
    """
    # Parallel: one thread per EID, each reading all loci from its own TabixFile.
    # Thread-safe because no two threads share a file handle.
    def _fetch_eid(item: tuple) -> tuple[str, list[str]]:
        eid, tbx = item
        states: list[str] = []
        for chrom, start in loci:
            try:
                rows = list(tbx.fetch(chrom, start, start + resolution_bp))
                states.append(rows[0].split("\t")[3] if rows else "15_Quies")
            except ValueError:
                states.append("15_Quies")
        return eid, states

    with ThreadPoolExecutor(max_workers=len(lookup._handles)) as pool:
        eid_state_map: dict[str, list[str]] = dict(pool.map(_fetch_eid, lookup._handles.items()))

    tissue_states: dict[str, list[str]] = {t: [] for t in TISSUE_GROUPS}
    for tissue, eids in TISSUE_GROUPS.items():
        for eid in eids:
            tissue_states[tissue].extend(eid_state_map.get(eid, ["15_Quies"] * len(loci)))

    matrix: dict[str, dict] = {}
    for tissue, states in tissue_states.items():
        n = len(states)
        if n == 0:
            continue
        counts = Counter(states)
        top_states = [
            (s, round(c / n, 3))
            for s, c in counts.most_common(n_top_states)
        ]
        matrix[tissue] = {"top_states": top_states}

    return {"matrix": matrix}
