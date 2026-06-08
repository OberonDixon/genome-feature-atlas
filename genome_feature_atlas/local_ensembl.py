"""Local tabix-backed replacement for the Ensembl REST API.

Reads pre-processed, bgzip-compressed GFF files instead of making HTTP requests
to grch37.rest.ensembl.org. Per-locus query time: ~1 ms vs ~300-500 ms (REST).

Setup (one-time):
    bash scripts/setup_local_annotations.sh data/ensembl_grch37/

Usage:
    from genome_feature_atlas.local_ensembl import LocalAnnotationLookup

    with LocalAnnotationLookup("data/ensembl_grch37") as lookup:
        # Returns one list[EnsemblFeature] per locus, in input order.
        features_per_locus = lookup.query_all_loci(
            [("chr7", 117_548_546, 117_548_674), ...]
        )

Coordinates follow the 0-based half-open convention used throughout this project.
Returned EnsemblFeature objects use Ensembl 1-based inclusive coordinates, matching
the REST API wrapper in ensembl.py.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import pysam

from genome_feature_atlas.ensembl import EnsemblFeature

_GENE_FILE       = "genes.gff3.gz"
_REGULATORY_FILE = "regulatory.gff.gz"
_MOTIF_FILE      = "motifs.gff.gz"

# GFF3 type values that map to feature_type="transcript"
_TRANSCRIPT_TYPES = frozenset({
    "mRNA", "transcript", "ncRNA", "lnc_RNA",
    "pseudogenic_transcript", "ncRNA_gene",
})


class LocalAnnotationLookup:
    """Tabix-backed local annotation source returning EnsemblFeature objects.

    Reads from files produced by scripts/setup_local_annotations.sh:
      genes.gff3.gz       — genes, transcripts, exons, CDS
      regulatory.gff.gz   — enhancers, CTCF sites, promoters, …
      motifs.gff.gz       — TF binding-site motifs (optional)

    Args:
        data_dir: directory containing the processed GFF files and their .tbi indices.
    """

    def __init__(self, data_dir: Path | str) -> None:
        data_dir = Path(data_dir)
        self._handles: dict[str, pysam.TabixFile] = {}
        for name in (_GENE_FILE, _REGULATORY_FILE, _MOTIF_FILE):
            path = data_dir / name
            if path.exists():
                self._handles[name] = pysam.TabixFile(str(path))
        if not self._handles:
            raise FileNotFoundError(
                f"No annotation GFF files (genes.gff3.gz / regulatory.gff.gz / "
                f"motifs.gff.gz) found in {data_dir}. "
                f"Run scripts/setup_local_annotations.sh first."
            )

    def query_all_loci(
        self,
        loci: list[tuple[str, int, int]],
    ) -> list[list[EnsemblFeature]]:
        """Return Ensembl features for all loci, queried in parallel across files.

        One thread per source file (gene / regulatory / motif), each reading all
        loci sequentially from its own TabixFile handle — thread-safe, no locks.

        Args:
            loci: list of (chrom, start, end) using 0-based half-open coordinates.

        Returns:
            List with one entry per input locus, each a list of EnsemblFeature.
        """
        def _fetch_source(item: tuple) -> tuple[str, list[list[EnsemblFeature]]]:
            name, tbx = item
            per_locus: list[list[EnsemblFeature]] = []
            for chrom, start, end in loci:
                try:
                    rows = list(tbx.fetch(chrom, start, end))
                    features = [f for r in rows if (f := _parse_row(r, name)) is not None]
                    per_locus.append(features)
                except ValueError:
                    per_locus.append([])
            return name, per_locus

        with ThreadPoolExecutor(max_workers=len(self._handles)) as pool:
            source_results = dict(pool.map(_fetch_source, self._handles.items()))

        # Merge per-source results into per-locus lists
        merged: list[list[EnsemblFeature]] = [[] for _ in loci]
        for per_locus in source_results.values():
            for i, features in enumerate(per_locus):
                merged[i].extend(features)
        return merged

    def close(self) -> None:
        for tbx in self._handles.values():
            tbx.close()
        self._handles.clear()

    def __enter__(self) -> "LocalAnnotationLookup":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ── GFF row parsing ───────────────────────────────────────────────────────────

def _parse_row(row: str, source_file: str) -> EnsemblFeature | None:
    cols = row.split("\t")
    if len(cols) < 9:
        return None
    chrom    = cols[0]          # "chrN" thanks to preprocessing
    gff_type = cols[2]
    start    = int(cols[3])     # 1-based inclusive (GFF3 convention)
    end      = int(cols[4])     # 1-based inclusive
    score    = cols[5]
    strand   = 1 if cols[6] == "+" else (-1 if cols[6] == "-" else 0)
    phase    = cols[7]
    attrs    = _parse_attrs(cols[8])

    if source_file == _GENE_FILE:
        return _parse_gene_row(gff_type, chrom, start, end, strand, phase, attrs)
    elif source_file == _REGULATORY_FILE:
        return _parse_regulatory_row(gff_type, chrom, start, end, strand, attrs)
    else:
        return _parse_motif_row(chrom, start, end, strand, score, attrs)


def _parse_gene_row(
    gff_type: str,
    chrom: str, start: int, end: int, strand: int,
    phase: str, attrs: dict[str, str],
) -> EnsemblFeature | None:
    raw_id = attrs.get("ID", "")

    if gff_type == "gene":
        return EnsemblFeature(
            feature_type="gene",
            id=raw_id.removeprefix("gene:"),
            chrom=chrom, start=start, end=end, strand=strand,
            name=attrs.get("Name", ""),
            biotype=attrs.get("biotype", ""),
            description=unquote(attrs.get("description", "")),
        )

    if gff_type in _TRANSCRIPT_TYPES:
        tags = attrs.get("tag", attrs.get("Tag", ""))
        return EnsemblFeature(
            feature_type="transcript",
            id=raw_id.removeprefix("transcript:"),
            chrom=chrom, start=start, end=end, strand=strand,
            name=attrs.get("Name", ""),
            biotype=attrs.get("biotype", ""),
            description=tags,
        )

    if gff_type == "exon":
        rank = attrs.get("rank", "?")
        exon_id = attrs.get("Name", "") or raw_id.removeprefix("exon:")
        return EnsemblFeature(
            feature_type="exon",
            id=raw_id.removeprefix("exon:"),
            chrom=chrom, start=start, end=end, strand=strand,
            name=exon_id,
            biotype=f"exon {rank}",
            description="",
        )

    if gff_type == "CDS":
        protein_id = attrs.get("protein_id", raw_id.removeprefix("CDS:").split(":")[0])
        return EnsemblFeature(
            feature_type="cds",
            id=protein_id,
            chrom=chrom, start=start, end=end, strand=strand,
            name=protein_id,
            biotype="CDS",
            description=f"phase={phase}",
        )

    return None  # UTR, repeat, or other type we don't report


def _parse_regulatory_row(
    gff_type: str,
    chrom: str, start: int, end: int, strand: int,
    attrs: dict[str, str],
) -> EnsemblFeature:
    # col 2 (gff_type) is the regulatory subtype: "enhancer", "CTCF_binding_site", etc.
    subtype = gff_type.replace("_", " ")
    return EnsemblFeature(
        feature_type="regulatory",
        id=attrs.get("ID", ""),
        chrom=chrom, start=start, end=end, strand=strand,
        name=subtype,
        biotype=subtype,
        description=subtype,
    )


def _parse_motif_row(
    chrom: str, start: int, end: int, strand: int,
    score: str, attrs: dict[str, str],
) -> EnsemblFeature:
    # Ensembl v114 uses "transcription_factor"; older files used "transcription_factor_complex"
    tf_complex = unquote(
        attrs.get("transcription_factor_complex", "")
        or attrs.get("transcription_factor", "")
    )
    # v114: "binding_matrix_id"; older: "binding_matrix_stable_id"
    matrix_id  = attrs.get("binding_matrix_id", "") or attrs.get("binding_matrix_stable_id", "")
    try:
        desc = f"score={float(score):.2f}"
    except ValueError:
        desc = ""
    return EnsemblFeature(
        feature_type="motif",
        id=attrs.get("stable_id", attrs.get("ID", "")),
        chrom=chrom, start=start, end=end, strand=strand,
        name=tf_complex,
        biotype=matrix_id,
        description=desc,
    )


def _parse_attrs(attr_str: str) -> dict[str, str]:
    return {
        k: v
        for k, v in (
            pair.split("=", 1)
            for pair in attr_str.rstrip(";").split(";")
            if "=" in pair
        )
    }
