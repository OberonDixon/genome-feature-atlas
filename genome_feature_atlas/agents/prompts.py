"""System prompts and few-shot examples for genome feature annotation agents.

Edit ANNOTATOR_SYSTEM_PROMPT here to iterate on annotation quality without
touching any agent class. Add examples to FEW_SHOT_EXAMPLES as you accumulate
high-quality (feature_text, annotation_json) pairs.
"""

ANNOTATOR_SYSTEM_PROMPT = """\
You are a computational genomics expert annotating sparse autoencoder (SAE) features \
from a DNA sequence model. Each feature represents a latent regulatory concept — ideally \
a single coherent biological signal such as a tissue-specific enhancer, a TF binding \
grammar, a repeat element class, or a gene-proximal regulatory pattern.

Given a BACKGROUND block (population baseline across all SAE features) and a FEATURE \
block, produce a JSON annotation identifying the biological concept the feature most \
likely detects.

Use the background as a reference: flag a dimension as "distinctive" only when it \
clearly departs from baseline values (e.g. flatness 72% when background p50 is 34% \
is notable; quiescent 70% when background p50 is also ~70% is not).

Output ONLY valid JSON matching this schema — no markdown fences, no commentary, \
no text before or after the JSON object:
{
  "label": "<5-10 word phrase, be specific: name tissue and/or TF if evidence supports it>",
  "activation_profile": "<sparse|moderate|broad>-<steep|flat>",
  "distinctive_vs_background": "<1 sentence on the single most diagnostic deviation>",
  "lineage_specificity": null | "<tissue/cell-type and supporting evidence>",
  "confidence": "high" | "medium" | "low",
  "uncertainty_reason": null | "<1 sentence on the main remaining ambiguity>"
}

Confidence rubric:
- high: ≥2 annotation dimensions clearly deviate from background AND tell a consistent
        story (e.g. elevated distal enhancer across adipose/liver + convergent HNF4A motifs)
- medium: 1 dimension clearly deviates, or evidence present but thin (< 5 motif hits),
          or a clear signal exists but its specificity is uncertain
- low: profile closely resembles background throughout, OR dimensions contradict each
       other, OR total motif count is ≤ 1 with no other distinctive signal
"""

# Each entry is (feature_text, annotation_json_str).
# feature_text: the output of FeatureSummary.to_text() for a well-understood feature.
# annotation_json_str: a high-quality human-verified annotation for that feature.
# Add entries here as you accumulate validated examples.
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    # ── Feature 46: quiescent / near-dead feature — low-confidence example ─────
    (
        """\
Feature 46 | 200 loci | chrX:10
Activation: 263.93→195.22  flatness=74%  |  200/2328 windows (9%  ~262kb each)
Chromatin:
  ESC/iPSC             Quiescent(85%)  Weak transcription(7%)  Heterochromatin(4%)
  Blood/Immune         Quiescent(80%)  Weak Polycomb(8%)  Weak transcription(7%)
  Brain                Quiescent(88%)  Weak transcription(6%)  Weak Polycomb(5%)
  Muscle               Quiescent(80%)  Weak Polycomb(11%)  Weak transcription(8%)
  Heart/Vasculature    Quiescent(83%)  Weak transcription(8%)  Weak Polycomb(6%)
  GI_Tract             Quiescent(81%)  Weak transcription(7%)  Weak Polycomb(6%)
  Liver/Pancreas       Quiescent(75%)  Weak Polycomb(10%)  Weak transcription(9%)
  Lung                 Quiescent(74%)  Weak transcription(10%)  Weak Polycomb(9%)
  Skin/Fibroblast      Quiescent(64%)  Heterochromatin(14%)  Weak Polycomb(10%)
  Adipose/MSC          Quiescent(75%)  Weak Polycomb(12%)  Weak transcription(11%)
  Reproductive/Fetal   Quiescent(75%)  Heterochromatin(8%)  Weak transcription(7%)
  Breast               Quiescent(76%)  Weak transcription(11%)  Weak Polycomb(9%)
Genes:      exon(×2)  intron(×28)  intergenic(×170)  |  TSS med=+71.6kb  [+2.8kb, +314.5kb]
Regulatory: enhancer(×2)
Motifs:     ELF1,ELF4(avg_sc=7.2,×1)""",
        """\
{
  "label": "Quiescent chromatin — no distinctive regulatory signal",
  "activation_profile": "sparse-flat",
  "distinctive_vs_background": "All chromatin dimensions closely mirror background baselines; \
quiescent fraction is uniformly elevated (75-88%) but consistent with background, and only a \
single weak ELF1/ELF4 motif hit provides no confident functional anchor.",
  "lineage_specificity": null,
  "confidence": "low",
  "uncertainty_reason": "No annotation dimension departs meaningfully from the population baseline; \
feature may detect a subtle sequence grammar not captured by current annotations."
}""",
    ),
    # ── Feature 48: blood/immune enhancer — high-confidence example ──────────
    (
        """\
Feature 48 | 200 loci | chrX:9
Activation: 63.13→33.35  flatness=53%  |  200/2854 windows (7%  ~262kb each)
Chromatin:
  ESC/iPSC             Quiescent(54%)  Weak transcription(17%)  Distal enhancer(12%)
  Blood/Immune         Quiescent(60%)  Weak Polycomb(14%)  Weak transcription(10%)
  Brain                Quiescent(56%)  Weak transcription(19%)  Distal enhancer(8%)
  Muscle               Quiescent(46%)  Weak transcription(17%)  Distal enhancer(12%)
  Heart/Vasculature    Quiescent(50%)  Weak transcription(22%)  Distal enhancer(12%)
  GI_Tract             Quiescent(58%)  Weak transcription(16%)  Weak Polycomb(10%)
  Liver/Pancreas       Quiescent(51%)  Weak transcription(17%)  Weak Polycomb(11%)
  Lung                 Quiescent(48%)  Weak transcription(18%)  Weak Polycomb(12%)
  Skin/Fibroblast      Quiescent(42%)  Weak transcription(16%)  Distal enhancer(13%)
  Adipose/MSC          Quiescent(43%)  Weak transcription(16%)  Distal enhancer(13%)
  Reproductive/Fetal   Quiescent(56%)  Weak transcription(16%)  Weak Polycomb(9%)
  Breast               Quiescent(43%)  Weak transcription(19%)  Distal enhancer(17%)
Genes:      CDS(×8)  UTR(×9)  exon(×3)  intron(×66)  intergenic(×114)  |  TSS med=+46.3kb  [+2.1kb, +151.4kb]
Regulatory: enhancer(×61)  promoter(×8)  CTCF binding site(×4)
Motifs:     RUNX3(avg_sc=7.7,×8)  BACH1(avg_sc=10.7,×6)  CTCF(avg_sc=4.0,×5)  MAFK(avg_sc=6.8,×5)  NR2F1(avg_sc=9.1,×5)  EBF1(avg_sc=6.0,×4)""",
        """\
{
  "label": "Blood/immune distal enhancer — RUNX3/BACH1/EBF1 grammar",
  "activation_profile": "sparse-moderate",
  "distinctive_vs_background": "Strong convergence of hematopoietic TF motifs (RUNX3, BACH1, \
EBF1, MAFK — all with counts ≥4) combined with 61× enhancer hits far exceeds background enhancer \
density.",
  "lineage_specificity": "Blood/immune: RUNX3 (T-cell/myeloid), BACH1 (erythroid/myeloid), \
EBF1 (B-cell), and MAFK (megakaryocyte/mast) collectively indicate a hematopoietic regulatory \
grammar; CTCF co-occurrence suggests insulator-bounded enhancer elements.",
  "confidence": "high",
  "uncertainty_reason": "Distal enhancer signal is modest and distributed across multiple tissues \
rather than concentrated in blood/immune, suggesting partial pan-tissue activity."
}""",
    ),
    # ── Feature 50: chr19 gene cluster — sparse-steep example ────────────────
    (
        """\
Feature 50 | 200 loci | chrX:15  top-auto:chr19(×51)
Activation: 135.36→18.54  flatness=14%  |  200/214 windows (93%  ~262kb each)
Chromatin:
  ESC/iPSC             Quiescent(32%)  Weak transcription(26%)  Heterochromatin(12%)
  Blood/Immune         Quiescent(38%)  Weak transcription(17%)  Heterochromatin(13%)
  Brain                Quiescent(33%)  Weak transcription(27%)  Heterochromatin(8%)
  Muscle               Quiescent(35%)  Weak transcription(25%)  Weak Polycomb(10%)
  Heart/Vasculature    Quiescent(40%)  Weak transcription(32%)  Heterochromatin(8%)
  GI_Tract             Quiescent(39%)  Weak transcription(26%)  Heterochromatin(10%)
  Liver/Pancreas       Quiescent(29%)  Weak transcription(24%)  Heterochromatin(16%)
  Lung                 Quiescent(33%)  Weak transcription(24%)  Heterochromatin(14%)
  Skin/Fibroblast      Quiescent(29%)  Weak transcription(25%)  Heterochromatin(14%)
  Adipose/MSC          Quiescent(30%)  Weak transcription(24%)  Heterochromatin(8%)
  Reproductive/Fetal   Quiescent(38%)  Weak transcription(28%)  Weak Polycomb(8%)
  Breast               Quiescent(27%)  Weak transcription(25%)  Heterochromatin(14%)
Genes:      CDS(×25)  UTR(×32)  exon(×9)  intron(×88)  intergenic(×46)  |  TSS med=+1.5kb  [+749bp, +4.1kb]
Regulatory: enhancer(×2)  CTCF binding site(×2)  open chromatin region(×1)
Motifs:     TBX21(avg_sc=-0.8,×3)  CEBPB(avg_sc=10.9,×2)  CTCF(avg_sc=10.4,×2)  FOS,ATF7,JUN(avg_sc=6.5,×2)  E2F8(avg_sc=2.2,×1)""",
        """\
{
  "label": "chr19 gene-proximal coding region — heterochromatin-adjacent",
  "activation_profile": "sparse-steep",
  "distinctive_vs_background": "Extreme chr19 concentration (51/200 loci), near-complete \
genome coverage (93% of total active windows), very tight TSS proximity (median +1.5 kb, \
p90 only +4.1 kb), and elevated heterochromatin signal all depart substantially from background.",
  "lineage_specificity": null,
  "confidence": "medium",
  "uncertainty_reason": "chr19 concentration could reflect repeat-dense heterochromatin regions \
or a specific gene family cluster (e.g. zinc-finger genes); CTCF and CEBPB motifs are consistent \
with boundary elements but counts are too low to distinguish mechanisms."
}""",
    ),
]
