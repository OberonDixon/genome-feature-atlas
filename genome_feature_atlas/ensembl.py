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

import time
from dataclasses import dataclass

import requests

_BASE = "https://rest.ensembl.org"
_SESSION = requests.Session()
_SESSION.headers["Accept"] = "application/json"

# Ensembl allows up to 15 req/s; we stay comfortably under that.
_MIN_INTERVAL = 1.0 / 12
_last_request_t: float = 0.0


@dataclass
class EnsemblFeature:
    feature_type: str  # "gene" | "regulatory"
    id: str
    chrom: str         # "chr21" style
    start: int         # Ensembl 1-based inclusive
    end: int           # Ensembl 1-based inclusive
    strand: int        # 1, -1, or 0 (stranded / unstranded)
    name: str          # gene symbol, or regulatory subtype (e.g. "enhancer")
    biotype: str       # protein_coding / lncRNA / enhancer / CTCF_binding_site / …
    description: str   # free-text description from Ensembl


def query_region(
    chrom: str,
    start: int,
    end: int,
    features: tuple[str, ...] = ("gene", "regulatory"),
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
    ensembl_chrom = chrom.removeprefix("chr")
    ens_start = start + 1   # 0-based → 1-based inclusive
    ens_end = end            # 0-based exclusive == 1-based inclusive (same integer)
    url = f"{_BASE}/overlap/region/human/{ensembl_chrom}:{ens_start}-{ens_end}"
    params = [("feature", ft) for ft in features]

    _rate_limit()

    for attempt in range(4):
        resp = _SESSION.get(url, params=params, timeout=30)
        _touch_last()
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2 ** attempt))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return [_parse(item) for item in resp.json()]

    resp.raise_for_status()  # will always raise after exhausting retries
    return []  # unreachable, satisfies type checkers


# ── internals ────────────────────────────────────────────────────────────────

def _rate_limit() -> None:
    global _last_request_t
    elapsed = time.monotonic() - _last_request_t
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)


def _touch_last() -> None:
    global _last_request_t
    _last_request_t = time.monotonic()


def _parse(item: dict) -> EnsemblFeature:
    ft = item.get("feature_type", "")
    raw_chrom = item.get("seq_region_name", "")
    chrom = f"chr{raw_chrom}" if raw_chrom and not raw_chrom.startswith("chr") else raw_chrom

    if ft == "gene":
        name = item.get("external_name") or item.get("id", "")
    elif ft == "regulatory":
        # description field holds the subtype: "enhancer", "CTCF binding site", …
        name = item.get("description") or item.get("id", "")
    else:
        name = item.get("id", "")

    return EnsemblFeature(
        feature_type=ft,
        id=item.get("id", ""),
        chrom=chrom,
        start=item.get("start", 0),
        end=item.get("end", 0),
        strand=item.get("strand", 0),
        name=name,
        biotype=item.get("biotype", ""),
        description=item.get("description") or "",
    )
