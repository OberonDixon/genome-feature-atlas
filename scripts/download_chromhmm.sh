#!/bin/bash

BASE="https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final"

mkdir -p ../data/roadmap_chromhmm/hg19
cd ../data/roadmap_chromhmm/hg19

curl -o EIDlegend.txt "https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/imputed12marks/jointModel/final/EIDlegend.txt"

for i in $(seq -w 1 129); do
    EID="E${i}"
    OUT="${EID}_15_coreMarks_dense.bed.bgz"
    [ -f "$OUT" ] && echo "Skipping $EID" && continue
    echo -n "Downloading $EID... "
    curl -f -s -o "$OUT" "${BASE}/${OUT}" \
        && curl -f -s -o "${OUT}.tbi" "${BASE}/${OUT}.tbi" \
        && echo "OK" \
        || { rm -f "$OUT" "${OUT}.tbi"; echo "not found (skipped)"; }
done