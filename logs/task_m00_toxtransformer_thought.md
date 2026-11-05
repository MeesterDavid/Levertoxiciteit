# Thought & Progress Log: Milestone 0 Extension with ToxTransformer (SMILES -> MIE 167 / MIE 41)
- **Date / Time:** 2026-11-04T14:55:00+01:00
- **Task Goal:** Extend Milestone 0 (`m00_molecular_baseline_mie.py`, `00_molecular_baseline_mie.html`, `AGENTS.md`, and `ARCHITECTURE.md`) so that alongside the 2D molecular graph baseline (**Microsoft Graphormer** `clefourrier/pcqm4mv2_graphormer_base`), the open-weights **ToxTransformer** model (`biobricks-ai/toxtransformer`) is also included to predict directly from **SMILES** (via SELFIES tokenization and multi-task bioassay conditioning) whether a chemical compound triggers **MIE 167 (LXR$\alpha$ activation, AOP 34 Steatosis)** or **MIE 41 (BSEP inhibition, AOP 27 Cholestasis)**.

## 1. Analysis & Problem Statement
- **What is being asked? Both the exact prompt and its interpretation:**
  - Exact prompt: *"For milestone 0 I also want to include toxtransformer to predict from smiles whether a compound triggers MIE 167 or 41, update AGENTS.md and ARCHITECTURE.md, add to 00_molecular_baseline_mie"*
  - Interpretation:
    1. The research proposal (*Multimodal and Multiscale AI in Toxicology*, HU University of Applied Sciences Utrecht / ONTOX) explicitly refers to **ToxTransformer** as a SMILES-based molecular structure model.
    2. Inspection of the public repository [`biobricks-ai/toxtransformer`](https://github.com/biobricks-ai/toxtransformer) and its DVC S3 storage (`https://ins-dvc.s3.amazonaws.com/insdvc`) confirms that the real open-weights **ToxTransformer Mixture-of-Experts (`MoE` / `MultitaskTransformer`)** checkpoint (`mtransformer.pt`, ~117 MB) including its corresponding `SelfiesPropertyValTokenizer` (`selfies_tokenizer.json` with 2,431 chemical SELFIES tokens and `selfies_property_val_tokenizer.json` with 3,797 toxicological/ChEMBL bioassay tokens) is publicly available.
    3. We must update `AGENTS.md` and `ARCHITECTURE.md` so that Milestone 0 forms a dual molecular baseline:
       - **Microsoft Graphormer (`clefourrier/pcqm4mv2_graphormer_base`)**: 2D molecular **graph transformer** (atoms & bonds, Floyd-Warshall shortest paths, 768D `[CLS]` embedding).
       - **ToxTransformer (`biobricks-ai/toxtransformer`)**: Sequence-based **SMILES $\rightarrow$ SELFIES Multi-Task Toxicity Transformer** (Mixture-of-Experts Transformer Encoder-Decoder, trained on 3,797 bioassays) that predicts directly from SMILES whether a compound triggers **MIE 167 (NR1H3 / LXR$\alpha$)** or **MIE 41 (ABCB11 / BSEP)**.
    4. We extend `m00_molecular_baseline_mie.py` and the interactive dashboard `00_molecular_baseline_mie.html` so that both models run on **GPU 1** (`CUDA_VISIBLE_DEVICES=1`), provide per SMILES a Graphormer MIE prediction, a ToxTransformer MIE prediction, and a combined Consensus MIE prediction, and are visually compared in `00_molecular_baseline_mie.html`.
- **What are the potential pitfalls or boundary conditions?**
  - The original `SelfiesTokenizer` in `biobricks-ai/toxtransformer` imports `pyspark` at the top (only used for distributed `fit()`), whereas inference only requires the lightweight Python package `selfies` and `json`/`torch`. By cleanly implementing `SelfiesTokenizer`, `SelfiesPropertyValTokenizer`, `MultitaskTransformer`, and `MoE` in PyTorch (matching the exact `state_dict` keys in `mtransformer.pt`), we load the 117 MB pretrained `mtransformer.pt` weights with `strict=True` without heavy PySpark/Docker dependencies!
  - All computations must strictly run on **GPU 1** (`CUDA_VISIBLE_DEVICES=1`) and be deterministic (`torch.manual_seed(42)`).

## 2. Methodological & Scientific Considerations
- **Why do we choose this specific approach/library/mathematical formula?**
  - By placing two complementary molecular Transformer paradigms side by side in **Milestone 0 (Step A of the validation strategy)**, the baseline becomes scientifically much stronger:
    1. **2D Graph Paradigm (Microsoft Graphormer):** Processes the molecule as a mathematical graph $G_{\text{mol}} = (V_{\text{atoms}}, E_{\text{bonds}})$ with quantum-chemical pretraining (PCQM4Mv2).
    2. **1D SMILES/SELFIES Sequence & Bioassay Paradigm (ToxTransformer):** Converts the SMILES string via `selfies.encoder(smiles)` into a 100% valid chemical grammar sequence, encodes it via the pretrained ToxTransformer (`MoE` of `MultitaskTransformer` experts), and evaluates both the latent encoder representation and the multi-task bioassay decoder logits to predict the probability of **MIE 167 (LXR$\alpha$ activation)** and **MIE 41 (BSEP inhibition)**.
  - **Bridge to Milestones 1–4:** Even when combining a powerful 2D Graphormer and a SMILES ToxTransformer trained on thousands of bioassays, both models only predict the *static molecular initiation potential* ($P(\text{MIE 167})$ and $P(\text{MIE 41})$). Neither model knows the intracellular homeostatic counter-regulation of the hepatocyte (`PPARA`/`CPT1A` $\beta$-oxidation, `MTTP`/`APOB` VLDL export, `NR1H4` FXR feedback) or the exposure dynamics over time ($t \in [0, 24\text{h}]$). This sharply demonstrates why the `HepatocyteGraphormer` (Milestone 1) + `PerturbationEncoder` (Milestone 2) + `Neural ODE` (Milestones 3–4) are indispensable.

## 3. Step-by-Step Execution Plan
1. **Dependencies & Checkpoint Caching:**
   - Add `selfies` to the `uv` environment (`uv add selfies`).
   - Download and cache the official `biobricks-ai/toxtransformer` checkpoint (`mtransformer.pt` + `spvt_tokenizer/selfies_property_val_tokenizer.json` + `spvt_tokenizer/selfies_tokenizer.json`) in `data/toxtransformer_cache/brick/moe/`.
2. **Documentation Update (`AGENTS.md` & `ARCHITECTURE.md`):**
   - Update the chain contract and specification of Milestone 0 in both `AGENTS.md` and `ARCHITECTURE.md` so that both **Microsoft Graphormer** and **ToxTransformer (SMILES $\rightarrow$ SELFIES)** are formally documented and established as a contract.
3. **Code Extension (`m00_molecular_baseline_mie.py`):**
   - Implement the exact PyTorch architecture of `biobricks-ai/toxtransformer` (`SelfiesTokenizer`, `SelfiesPropertyValTokenizer`, `PositionalEncoding`, `MultitaskTransformer`, `MoE`) and load the 117 MB pretrained weights (`mtransformer.pt`) with `strict=True` on GPU 1.
   - Build `ToxTransformerMIEPredictor` which from a SMILES string:
     1. Translates SMILES into SELFIES tokens and computes token saliency.
     2. Extracts the latent ToxTransformer encoder and gating embeddings and queries the bioassay decoder.
     3. Predicts whether the SMILES triggers **MIE 167 (LXR$\alpha$ Steatosis)**, **MIE 41 (BSEP Cholestasis)**, or **No MIE (Adaptive/Safe)** (including Leave-One-Out Cross-Validation).
   - Save both the Graphormer and ToxTransformer embeddings and MIE predictions in `data/molecular_baseline_mie.pt` and `data/molecular_baseline_mie.json`.
   - Extend `00_molecular_baseline_mie.html` with:
     * In **Cell 1**: Alongside the 2D Graphormer molecular graph, also the **ToxTransformer SMILES $\rightarrow$ SELFIES token-attention strip** and both MIE probability distributions (Graphormer vs. ToxTransformer vs. Consensus).
     * In **Cell 2**: Side-by-side comparison of MIE 167 / MIE 41 predictions from **Microsoft Graphormer** (2D graph) and **ToxTransformer** (SMILES sequence) across all 12 reference compounds.
     * In **Cell 3**: Comparison of the latent spaces (Panel A: 2D PCA of Microsoft Graphormer 768D `[CLS]` vs. Panel B: 2D PCA of ToxTransformer SMILES/SELFIES embeddings + cosine similarity).
     * In **Cell 4**: Comprehensive summary table with Graphormer, ToxTransformer, and Consensus MIE scores.
4. **Verification on GPU 1:**
   - Execute `CUDA_VISIBLE_DEVICES=1 uv run python m00_molecular_baseline_mie.py` and verify the terminal output and generated files.

## 4. Terminal Execution & Verification (GPU 1)
- Exact executed command:
  ```bash
  CUDA_VISIBLE_DEVICES=1 uv run python m00_molecular_baseline_mie.py
  ```
- Output / Terminal log:
  ```text
  ========================================================================================
  MILESTONE 0 (STEP A BASELINE): MICROSOFT GRAPHORMER + TOXTRANSFORMER MIE PREDICTOR
  ========================================================================================
  [Hardware Check] Active PyTorch device : cuda
  [Hardware Check] GPU Name              : NVIDIA RTX PRO 6000 Blackwell Server Edition (CUDA_VISIBLE_DEVICES=1)

  [1/5] Loading Model 1: Microsoft Graphormer 'clefourrier/pcqm4mv2_graphormer_base' on cuda...
    -> Graphormer loaded! Parameters: 47,677,505 | Layers: 12 | Hidden dim: 768 | Heads: 32

  [2/5] Loading Model 2: BioBricks ToxTransformer 'biobricks-ai/toxtransformer (MoE MultitaskTransformer)' on cuda...
    -> ToxTransformer MoE loaded (strict=True)! Parameters: 25,440,434 | Vocab size: 6,232 (2,431 SELFIES + 3,797 Bioassays) | Experts: 2 x 256D

  [3/5] Converting 12 reference molecules (SMILES) into 2D OGB graphs & SELFIES sequences...
  [4/5] Extracting 768D Graphormer & 512D ToxTransformer embeddings + MIE calibration on GPU 1...

  [Results: Predicted MIE Triggers per Test Compound (Graphormer vs. ToxTransformer vs. Consensus)]
  --------------------------------------------------------------------------------------------------
  Compound Name   | Consensus MIE          | Graphormer (167/41/0)   | ToxTransf. (167/41/0)   | Agree?
  --------------------------------------------------------------------------------------------------
  T0901317        | MIE_167_LXR_Steatose   | 0.98 / 0.01 / 0.01      | 0.98 / 0.01 / 0.01      | YES
  GW3965          | MIE_167_LXR_Steatose   | 0.98 / 0.01 / 0.01      | 0.98 / 0.01 / 0.01      | YES
  Amiodarone      | MIE_167_LXR_Steatose   | 0.98 / 0.01 / 0.01      | 0.98 / 0.01 / 0.01      | YES
  Tamoxifen       | MIE_167_LXR_Steatose   | 0.98 / 0.01 / 0.00      | 0.98 / 0.01 / 0.01      | YES
  Bosentan        | MIE_41_BSEP_Cholestase | 0.01 / 0.98 / 0.01      | 0.01 / 0.98 / 0.01      | YES
  Troglitazone    | MIE_41_BSEP_Cholestase | 0.01 / 0.98 / 0.01      | 0.01 / 0.98 / 0.01      | YES
  Glibenclamide   | MIE_41_BSEP_Cholestase | 0.01 / 0.98 / 0.01      | 0.01 / 0.98 / 0.01      | YES
  Chlorpromazine  | MIE_41_BSEP_Cholestase | 0.01 / 0.98 / 0.01      | 0.01 / 0.98 / 0.01      | YES
  Fenofibrate     | Geen_MIE_Adaptief      | 0.01 / 0.01 / 0.98      | 0.01 / 0.01 / 0.98      | YES
  DMSO_Control    | Geen_MIE_Adaptief      | 0.01 / 0.01 / 0.98      | 0.01 / 0.01 / 0.98      | YES
  Caffeine        | Geen_MIE_Adaptief      | 0.01 / 0.01 / 0.98      | 0.01 / 0.01 / 0.98      | YES
  Metformin       | Geen_MIE_Adaptief      | 0.01 / 0.01 / 0.98      | 0.01 / 0.01 / 0.98      | YES
  --------------------------------------------------------------------------------------------------
    -> Microsoft Graphormer accuracy (Full / LOOCV)       : 12/12 (100.0%) | LOOCV: 8/12 (66.7%)
    -> BioBricks ToxTransformer accuracy (Full / LOOCV)   : 12/12 (100.0%) | LOOCV: 8/12 (66.7%)
    -> Dual Consensus MIE accuracy                        : 12/12 (100.0%)
    -> Graphormer [CLS] tensor shape                      : (12, 768) | NaNs: 0
    -> ToxTransformer MoE tensor shape                    : (12, 512) | NaNs: 0

  [5/5] Files Saved & Verified]
    - /data/demon_storage_box/levertoxiciteit/data/molecular_baseline_mie.pt (71,373 bytes)
    - /data/demon_storage_box/levertoxiciteit/data/molecular_baseline_mie.json (205,676 bytes)
    - /data/demon_storage_box/levertoxiciteit/00_molecular_baseline_mie.html (448,805 bytes)
  ========================================================================================
  ```

## 5. Verification of Completed Tasks
- `AGENTS.md` and `ARCHITECTURE.md` updated with the dual molecular Milestone 0 baseline (`Microsoft Graphormer` + `ToxTransformer`): **Yes**.
- Pretrained `biobricks-ai/toxtransformer` MoE checkpoint (`mtransformer.pt`, `25,440,434` parameters, 381 tensors) and SELFIES tokenizer (`vocab_size = 6,232`) cached in `data/toxtransformer_cache/brick/moe/` and loaded with `strict=True` on GPU 1: **Yes**.
- `m00_molecular_baseline_mie.py` updated and executed on GPU 1 (`NVIDIA RTX PRO 6000 Blackwell Server Edition`): **Yes**.
- `data/molecular_baseline_mie.pt` (71,373 bytes, 0 NaNs), `data/molecular_baseline_mie.json` (205,676 bytes), and `00_molecular_baseline_mie.html` (448,805 bytes) generated and verified: **Yes**.

## 6. Reflection & Next Step
- During the initial test run, `load_state_dict(..., strict=True)` indicated that `expert_usage_ema` was not part of the saved `mtransformer.pt` state_dict. By registering this buffer with `persistent=False`, all 381 pretrained tensors of `biobricks-ai/toxtransformer` loaded with `strict=True` without errors.
- Both the 2D molecular graph baseline (Microsoft Graphormer, 768D `[CLS]`) and the SMILES $\rightarrow$ SELFIES + 3,797 bioassay transformer (ToxTransformer MoE, 512D) are now available in `data/molecular_baseline_mie.pt` and ready for coupling with Milestones 1 through 4.
