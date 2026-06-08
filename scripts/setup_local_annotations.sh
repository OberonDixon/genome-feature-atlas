#!/usr/bin/env bash
# Download and preprocess Ensembl GRCh37 annotation files for local tabix queries.
# Produces genes.gff3.gz, regulatory.gff.gz, and motifs.gff.gz in the output
# directory, each sorted, bgzip-compressed, and tabix-indexed.
#
# Usage:
#   bash scripts/setup_local_annotations.sh [output_dir]
#
# Default output_dir: data/ensembl_grch37
#
# Requirements: wget, bgzip, tabix, awk, sort
# Expected disk space: ~800 MB (including temporary raw downloads)

set -euo pipefail

DIR="${1:-data/ensembl_grch37}"
mkdir -p "$DIR"

FTP="https://ftp.ensembl.org/pub/grch37/current"
# All regulation files live under annotation/ as of Ensembl v114
REG_BASE="$FTP/regulation/homo_sapiens/annotation"

# ── helper ────────────────────────────────────────────────────────────────────

add_chr_sort_bgzip() {
    # stdin → add "chr" prefix to col 1 → sort by chrom+pos → bgzip → stdout
    awk 'BEGIN{OFS="\t"} !/^#/ {$1="chr"$1; print}' \
        | sort -k1,1 -k4,4n
}

# ── gene annotations ──────────────────────────────────────────────────────────

echo "=== Gene annotations (genes, transcripts, exons, CDS) ==="
RAW="$DIR/raw_genes.gff3.gz"
wget -q --show-progress -O "$RAW" \
    "$FTP/gff3/homo_sapiens/Homo_sapiens.GRCh37.87.gff3.gz"

# Keep only the feature types used by local_ensembl.py; drop UTRs, repeats, etc.
# $3~/gene/ matches: gene, pseudogene, ncRNA_gene, polymorphic_pseudogene, etc.
zcat "$RAW" \
    | awk '$3~/gene/||$3=="mRNA"||$3=="transcript"||$3=="lnc_RNA"||$3=="ncRNA"||$3=="pseudogenic_transcript"||$3=="exon"||$3=="CDS"' \
    | add_chr_sort_bgzip \
    | bgzip > "$DIR/genes.gff3.gz"
tabix -p gff "$DIR/genes.gff3.gz"
rm "$RAW"
echo "  -> $DIR/genes.gff3.gz ($(du -sh "$DIR/genes.gff3.gz" | cut -f1))"

# ── regulatory annotations ────────────────────────────────────────────────────

echo "=== Regulatory build (enhancers, CTCF sites, promoters, …) ==="
RAW="$DIR/raw_regulatory.gff3.gz"
wget -q --show-progress -O "$RAW" \
    "$REG_BASE/Homo_sapiens.GRCh37.regulatory_features.v114.gff3.gz"
zcat "$RAW" \
    | add_chr_sort_bgzip \
    | bgzip > "$DIR/regulatory.gff.gz"
tabix -p gff "$DIR/regulatory.gff.gz"
rm "$RAW"
echo "  -> $DIR/regulatory.gff.gz ($(du -sh "$DIR/regulatory.gff.gz" | cut -f1))"

# ── motif annotations (large) ─────────────────────────────────────────────────

echo "=== TF motif features (~500 MB) ==="
RAW="$DIR/raw_motifs.gff3.gz"
wget -q --show-progress -O "$RAW" \
    "$REG_BASE/Homo_sapiens.GRCh37.motif_features.v114.gff3.gz"
zcat "$RAW" \
    | add_chr_sort_bgzip \
    | bgzip > "$DIR/motifs.gff.gz"
tabix -p gff "$DIR/motifs.gff.gz"
rm "$RAW"
echo "  -> $DIR/motifs.gff.gz ($(du -sh "$DIR/motifs.gff.gz" | cut -f1))"

echo ""
echo "=== Setup complete. Files in $DIR: ==="
ls -lh "$DIR/"
