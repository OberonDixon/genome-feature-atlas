"""Thin wrapper around the Ensembl REST API /overlap/region endpoint.

Usage:
    from genome_feature_atlas.ensembl import query_region
    features = query_region("chr21", 37_700_572, 37_700_700)
    for f in features:
        print(f.feature_type, f.name, f.biotype)

Coordinates follow the same 0-based half-open convention used everywhere else in
this project (bin_start, bin_start + resolution_bp). Conversion to Ensembl's
1-based inclusive coordinates is handled internally.
"""

import threading
import time
from dataclasses import dataclass

import requests

_BASE = "https://grch37.rest.ensembl.org"
_SESSION = requests.Session()
_SESSION.headers["Accept"] = "application/json"

# Ensembl allows up to 15 req/s; we stay comfortably under that.
_MIN_INTERVAL = 1.0 / 12
_last_request_t: float = 0.0
_rate_lock = threading.Lock()

_ALL_FEATURES = ("gene", "transcript", "exon", "cds", "regulatory", "motif")

# Module-level response cache: (chrom, start, end, features) → list[EnsemblFeature].
# Eliminates repeat API calls when the same locus appears across multiple features.
_region_cache: dict[tuple, list["EnsemblFeature"]] = {}


@dataclass
class EnsemblFeature:
    feature_type: str  # "gene" | "transcript" | "exon" | "cds" | "regulatory" | "motif"
    id: str
    chrom: str         # "chr21" style
    start: int         # Ensembl 1-based inclusive
    end: int           # Ensembl 1-based inclusive
    strand: int        # 1, -1, or 0
    name: str          # gene symbol / isoform name / TF name / exon id / protein id / regulatory subtype
    biotype: str       # protein_coding / lncRNA / exon N / CDS / ENSPFM… / enhancer / …
    description: str   # gene description / MANE tags / TF score / phase info


def query_region(
    chrom: str,
    start: int,
    end: int,
    features: tuple[str, ...] = _ALL_FEATURES,
) -> list[EnsemblFeature]:
    """Return Ensembl features overlapping chrom:start-end.

    Args:
        chrom:    chromosome, with or without "chr" prefix (e.g. "chr21" or "21")
        start:    0-based inclusive start (genomic bp)
        end:      0-based exclusive end  (genomic bp)
        features: Ensembl feature types to query

    Returns a list of EnsemblFeature, possibly empty if nothing overlaps.
    Raises requests.HTTPError on unrecoverable API errors.
    """
    key = (chrom, start, end, features)
    if key in _region_cache:
        return _region_cache[key]

    ensembl_chrom = chrom.removeprefix("chr")
    ens_start = start + 1   # 0-based → 1-based inclusive
    ens_end = end            # 0-based exclusive == 1-based inclusive (same integer)
    url = f"{_BASE}/overlap/region/human/{ensembl_chrom}:{ens_start}-{ens_end}"
    params = [("feature", ft) for ft in features]

    _rate_limit()

    for attempt in range(4):
        resp = _SESSION.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2 ** attempt))
            time.sleep(wait)
            _rate_limit()  # re-claim a slot after the back-off
            continue
        resp.raise_for_status()
        # Filter to features that physically overlap the query range.
        # Necessary for "cds", which Ensembl returns all-CDS-of-overlapping-transcripts
        # rather than only CDS whose coordinates intersect the query.
        result = [
            _parse(item) for item in resp.json()
            if item.get("start", 0) <= ens_end and item.get("end", 0) >= ens_start
        ]
        _region_cache[key] = result
        return result

    resp.raise_for_status()  # will always raise after exhausting retries
    return []  # unreachable, satisfies type checkers


def clear_cache() -> None:
    """Clear the in-process response cache."""
    _region_cache.clear()


# ── internals ────────────────────────────────────────────────────────────────

def _rate_limit() -> None:
    """Claim a request slot, sleeping if necessary to respect the rate limit.

    Thread-safe: holds _rate_lock while updating _last_request_t so concurrent
    callers queue up and each gets a distinct 83 ms slot rather than racing to
    send requests simultaneously.
    """
    global _last_request_t
    with _rate_lock:
        elapsed = time.monotonic() - _last_request_t
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_request_t = time.monotonic()


def _parse(item: dict) -> EnsemblFeature:
    ft = item.get("feature_type", "")
    raw_chrom = item.get("seq_region_name", "")
    chrom = f"chr{raw_chrom}" if raw_chrom and not raw_chrom.startswith("chr") else raw_chrom

    if ft == "gene":
        name        = item.get("external_name") or item.get("id", "")
        biotype     = item.get("biotype", "")
        description = item.get("description") or ""
    elif ft == "transcript":
        name        = item.get("external_name") or item.get("id", "")
        biotype     = item.get("biotype", "")
        tags        = item.get("tag") or []
        description = ",".join(tags)
    elif ft == "exon":
        name        = item.get("exon_id") or item.get("id", "")
        rank        = item.get("rank", "?")
        biotype     = f"exon {rank}"
        description = ""
    elif ft == "cds":
        name        = item.get("protein_id") or item.get("id", "")
        biotype     = "CDS"
        phase       = item.get("phase", "?")
        description = f"phase={phase}"
    elif ft == "regulatory":
        # description holds the subtype: "enhancer", "CTCF binding site", …
        subtype     = item.get("description") or item.get("id", "")
        name        = subtype
        biotype     = subtype
        description = subtype
    elif ft == "motif":
        name        = item.get("transcription_factor_complex") or item.get("stable_id", "")
        biotype     = item.get("binding_matrix_stable_id", "")
        score       = item.get("score")
        description = f"score={score:.2f}" if score is not None else ""
    else:
        name        = item.get("id", "")
        biotype     = item.get("biotype", "")
        description = item.get("description") or ""

    return EnsemblFeature(
        feature_type=ft,
        id=item.get("id", ""),
        chrom=chrom,
        start=item.get("start", 0),
        end=item.get("end", 0),
        strand=item.get("strand", 0),
        name=name,
        biotype=biotype,
        description=description,
    )
