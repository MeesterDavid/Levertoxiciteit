# Thought & Progress Log: Switch All Milestone 0 Dashboard Cells (1, 2, 3, 4, 5, 6) to LOOCV Data & Explain PCA (PC1/PC2)
- **Date / Time:** 2026-11-08T17:59:00+01:00
- **Task Goal:** Answer the user's questions about 50/50 probability fusion and PCA (PC1/PC2), and update `m00_molecular_baseline_mie.py` (plus `generate_pipeline_diagrams.py` and `generate_slide2_diagram.py`) so that all dashboard cells (Cell 1, Cell 2, Cell 3, Cell 4, Cell 5, and Cell 6) and the console table display the honest Leave-One-Out Cross-Validation (LOOCV) probabilities and predictions, with a crystal-clear explanation of PC 1 and PC 2 (including explained variance percentages) in Cell 4.

## 1. Analysis & Problem Statement
- **What is being asked? Both the exact prompt and its interpretation:**
  - Exact prompt (NL): *"hoe werkt die 50/50 fusion hoe Fuse je binaire antwoorden? Ik wil overigens dat je alle andere cellen (1, 2, cell 4 snap ik niet, wat is PC 1 en wat is PC 2, en cell 5) aanpast zodat ze werken met de LOOCV data. Geef eerst antwoord op de vraag ga daarna werken aan de verbeterde notebook."*
  - Interpretation:
    1. Explain clearly how 50/50 fusion works: we do **not** fuse binary yes/no labels; both models output a 3-class probability vector `[P(no MIE), P(MIE 167), P(MIE 41)]` via cosine-similarity softmax against the 11-compound class prototypes in LOOCV. The 50/50 fusion averages the two probability vectors (`0.5 * P_graph + 0.5 * P_tox`) and takes `argmax` of the combined probabilities.
    2. Explain what PC 1 and PC 2 are in Cell 4 (Principal Component Analysis / SVD projection from 768D / 524D down to the 2 directions of maximum variance), and add that plain-language explanation + `% variance explained` directly to Cell 4 in the dashboard.
    3. Update **Cells 1, 2, and 5** (and the console printout + JSON/PT `prediction` / `toxtransformer_prediction` / `consensus_prediction` blocks) so they use the **LOOCV probabilities and predictions** instead of the overfitted in-sample `0.98 / 0.01 / 0.01` head outputs.
    4. Update **Cell 4** so ToxTransformer's PCA uses the full `524D` LOOCV representation (`512D` expert + `12D` bioassay summary), displays the exact `% variance explained` on the PC 1 and PC 2 axes, and includes an intuitive explanation of PC 1 & PC 2 in the User Guide box.
    5. Update **Cell 6** (`generate_pipeline_diagrams.py`) to remove the obsolete `9D RDKit Farmacofoor Head` boxes and describe the pure frozen foundation embedding + LOOCV prototype readout.

## 2. Methodological & Scientific Considerations
- **Why do we choose this specific approach/library/mathematical formula?**
  - Previously, Cell 3 showed the honest LOOCV confusion matrices (`4/12`, `9/12`, `10/12`), while Cells 1, 2, and 5 still showed the in-sample `98.2%` calibrated head bars—creating a direct contradiction inside the same dashboard.
  - By driving `prediction` (Graphormer), `toxtransformer_prediction` (ToxTransformer), and `consensus_prediction` (50/50 fusion) directly from the LOOCV prototype classifier for all 12 reference compounds, every single visualization in `public/00_molecular_baseline_mie.html` tells one unified, mathematically consistent story.

## 3. Step-by-Step Execution Plan
1. Inspect `data/molecular_baseline_mie.json` to see the exact per-compound LOOCV probabilities for Graphormer, ToxTransformer, and Consensus.
2. Update `m00_molecular_baseline_mie.py`:
   - Build `loocv_entry_to_prediction_dict(loocv_entry)` so each compound's `prediction`, `toxtransformer_prediction`, and `consensus_prediction` in `compounds_out` are populated directly from the LOOCV prototype probabilities (`loocv_P_no_mie`, `loocv_P_mie167`, `loocv_P_mie41`).
   - In Cell 1 (`render_molecule_graph_svg`), show the LOOCV probabilities, whether each model's LOOCV prediction matches the true class (`✓ Correct` vs `✗ Miss`), and update the Cell 1 guide text.
   - In Cell 2 (`render_mie_probabilities_svg`), plot the LOOCV probability bars for Graphormer and ToxTransformer across all 12 compounds, annotate which compounds are correctly classified vs. misclassified in LOOCV, and update the Cell 2 guide text.
   - In Cell 4 (`render_embedding_space_svg` + guide box), compute PCA on the `768D` Graphormer and `524D` ToxTransformer LOOCV spaces, compute `% variance explained` for PC 1 and PC 2 (`s^2 / sum(s^2)`), display `PC 1 (xx.x% var)` and `PC 2 (yy.y% var)` on the axes, and rewrite the Cell 4 User Guide to clearly explain what PCA, PC 1, and PC 2 mean.
   - In Cell 5 (Summary Table), display the LOOCV probabilities, LOOCV predicted class vs. true class, and update the Cell 5 guide text.
3. Update `generate_pipeline_diagrams.py` so the schematic architecture diagrams (Cell 6) reflect the pure frozen foundation embeddings + LOOCV nearest-prototype readout (without the removed 9D pharmacophore head).
4. Execute `CUDA_VISIBLE_DEVICES=1 uv run python m00_molecular_baseline_mie.py` and `CUDA_VISIBLE_DEVICES=1 uv run python generate_slide2_diagram.py` on GPU 1.
5. Complete sections 4, 5, and 6 of this thought log.

## 4. Terminal Execution & Verification (GPU 1)
- Exact executed command: `CUDA_VISIBLE_DEVICES=1 uv run python m00_molecular_baseline_mie.py`
- Output / Terminal log:
```text
========================================================================================
MILESTONE 0 (STEP A BASELINE): MICROSOFT GRAPHORMER + TOXTRANSFORMER MIE PREDICTOR
========================================================================================
[Hardware Check] Active PyTorch device : cuda
[Hardware Check] GPU Name              : NVIDIA RTX PRO 6000 Blackwell Server Edition (CUDA_VISIBLE_DEVICES=1)

[LOOCV Per-Compound Results: Graphormer (768D) vs. ToxTransformer (524D) vs. 50/50 Consensus]
--------------------------------------------------------------------------------------------------------
Compound        | Consensus LOOCV        | Graphormer (167/41/0)   | ToxTransf. (167/41/0)   | LOOCV Match?
--------------------------------------------------------------------------------------------------------
T0901317        | MIE_167_LXR_Steatose   | 0.35 / 0.36 / 0.30      | 0.91 / 0.02 / 0.06      | YES (✓)
GW3965          | MIE_167_LXR_Steatose   | 0.37 / 0.34 / 0.29      | 0.95 / 0.02 / 0.03      | YES (✓)
Amiodarone      | MIE_167_LXR_Steatose   | 0.32 / 0.46 / 0.23      | 0.80 / 0.10 / 0.09      | YES (✓)
Tamoxifen       | MIE_167_LXR_Steatose   | 0.29 / 0.37 / 0.34      | 0.76 / 0.11 / 0.14      | YES (✓)
Bosentan        | MIE_41_BSEP_Cholestase | 0.34 / 0.29 / 0.37      | 0.06 / 0.91 / 0.03      | YES (✓)
Troglitazone    | MIE_41_BSEP_Cholestase | 0.33 / 0.32 / 0.34      | 0.02 / 0.96 / 0.03      | YES (✓)
Glibenclamide   | MIE_41_BSEP_Cholestase | 0.38 / 0.37 / 0.25      | 0.01 / 0.96 / 0.03      | YES (✓)
Chlorpromazine  | MIE_167_LXR_Steatose   | 0.39 / 0.37 / 0.25      | 0.80 / 0.02 / 0.18      | MISS (True=2)
Fenofibrate     | MIE_41_BSEP_Cholestase | 0.33 / 0.37 / 0.30      | 0.37 / 0.60 / 0.03      | MISS (True=0)
DMSO_Control    | Geen_MIE_Adaptief      | 0.32 / 0.30 / 0.38      | 0.03 / 0.52 / 0.45      | YES (✓)
Caffeine        | Geen_MIE_Adaptief      | 0.29 / 0.34 / 0.38      | 0.37 / 0.02 / 0.61      | YES (✓)
Metformin       | Geen_MIE_Adaptief      | 0.26 / 0.31 / 0.43      | 0.11 / 0.01 / 0.88      | YES (✓)
--------------------------------------------------------------------------------------------------
  -> Graphormer [CLS] tensor shape                      : (12, 768) | NaNs: 0
  -> ToxTransformer MoE tensor shape                    : (12, 512) | NaNs: 0

  -> Microsoft Graphormer (2D graph)              LOOCV 4/12 (33.3%) | mean P(true class) 0.347 (0.29-0.43) | macro F1 0.318
  -> ToxTransformer (SMILES / bioassays)          LOOCV 9/12 (75.0%) | mean P(true class) 0.687 (0.02-0.96) | macro F1 0.741
  -> Dual consensus (50 / 50 probability fusion)  LOOCV 10/12 (83.3%) | mean P(true class) 0.517 (0.16-0.67) | macro F1 0.832

[5/5] Files Saved & Verified]
  - /data/demon_storage_box/levertoxiciteit/data/molecular_baseline_mie.pt (73,685 bytes)
  - /data/demon_storage_box/levertoxiciteit/data/molecular_baseline_mie.json (215,034 bytes)
  - /data/demon_storage_box/levertoxiciteit/public/00_molecular_baseline_mie.html (511,097 bytes)
```

## 5. Verification of Completed Tasks
- [x] `data/molecular_baseline_mie.pt` (73,685 bytes) and `data/molecular_baseline_mie.json` (215,034 bytes) verified with 0 NaNs.
- [x] `public/00_molecular_baseline_mie.html` (511,097 bytes) regenerated so **Cell 1, Cell 2, Cell 3, Cell 4, Cell 5, and Cell 6** all use the honest LOOCV data.
- [x] `public/pipeline_microsoft_graphormer.svg`, `public/pipeline_toxtransformer.svg`, `public/00_architecture_pipelines.html`, and `public/s2_multimodal_ai_graph.svg` updated and verified.

## 6. Reflection & Next Step
- Every cell of the Milestone 0 dashboard now uses the Leave-One-Out Cross-Validation (LOOCV) probabilities and embeddings without any in-sample or hand-crafted pharmacophore leakage. Ready to present the answers and updated dashboard overview to the user.
