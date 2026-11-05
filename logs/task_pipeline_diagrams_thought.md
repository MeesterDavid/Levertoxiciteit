# Thought & Progress Log: Scientific Pipeline & Architecture Diagrams in `public/`
- **Date / Time:** 2026-11-05T14:09:15+01:00
- **Task Goal:** Generate two separate scientific schematic pipeline & model-architecture diagrams (Figure 1: Microsoft Graphormer, Figure 2: BioBricks ToxTransformer) in `public/` (`public/00_architecture_pipelines.html`, `public/pipeline_microsoft_graphormer.svg`, `public/pipeline_toxtransformer.svg`, and embedded in `public/00_molecular_baseline_mie.html`).

## 1. Analysis & Problem Statement
- **What is being asked? Both the exact prompt and its interpretation:**
  - Prompt: Create two separate schematic pipeline diagrams (1: Microsoft Graphormer, 2: BioBricks ToxTransformer) showing `Data -> Bewerking -> AI-Modellen -> Output` from **left to right**, where under **Data** (top to bottom) all combined sources/checkpoints are shown, under **Bewerking** (top to bottom) all preprocessing steps are shown, under **AI-Modellen** (top to bottom) every layer with its exact type and tensor dimensions is drawn, and under **Output** (top to bottom) the embeddings, MIE probabilities, and artifacts are shown. Exported to `public/` (Option 2).
- **What are the potential pitfalls or boundary conditions?**
  - External CDN scripts (like Mermaid CDN) can be blocked in offline environments or VS Code HTML Preview; therefore we provide both the declarative Mermaid.js code/diagrams AND pre-rendered, publication-grade vector SVGs (`public/pipeline_microsoft_graphormer.svg` and `public/pipeline_toxtransformer.svg`) inside `public/00_architecture_pipelines.html` and `public/00_molecular_baseline_mie.html`.
  - Every layer dimension must match the exact PyTorch implementation in `m00_molecular_baseline_mie.py`.

## 2. Methodological & Scientific Considerations
- **Why do we choose this specific approach/library/mathematical formula?**
  - Combining declarative Mermaid.js definitions with standalone high-DPI vector SVGs gives the best of both worlds for scientific papers and presentations: direct Markdown/Slidev compatibility via Mermaid, plus 100% deterministic, publication-ready vector figures (`.svg`) with crisp typography, color-coded columns, and exact tensor shapes (`[1, N+1, 768]`, `[1, 32, N+1, N+1]`, `[1, L, 256]`, `[1, 3797]`, `[1, 524]`, `[1, 3]`).
- **How does this align with the research proposal (e.g., AOPs, attractor dynamics, API sources)?**
  - Transparently documents Step A of the validation strategy (Milestone 0): how 2D molecular graphs (Microsoft Graphormer) and SELFIES sequences + 3,797 bioassays (ToxTransformer MoE) are transformed into MIE 167 (AOP 34 Steatosis) and MIE 41 (AOP 27 Cholestasis) predictions.

## 3. Step-by-Step Execution Plan
1. Create `generate_pipeline_diagrams.py` to generate:
   - `public/pipeline_microsoft_graphormer.svg`
   - `public/pipeline_toxtransformer.svg`
   - `public/00_architecture_pipelines.html` (containing both interactive SVG figures and copyable Mermaid.js code blocks).
2. Embed the two architecture pipeline SVGs as Cell 5 in `m00_molecular_baseline_mie.py` (`public/00_molecular_baseline_mie.html`).
3. Run `CUDA_VISIBLE_DEVICES=1 uv run python generate_pipeline_diagrams.py` and `CUDA_VISIBLE_DEVICES=1 uv run python m00_molecular_baseline_mie.py` on GPU 1.
4. Verify all generated files in `public/` and finalize this thought log.

## 4. Terminal Execution & Verification (GPU 1)
- Exact executed command:
  ```bash
  CUDA_VISIBLE_DEVICES=1 uv run python generate_pipeline_diagrams.py && CUDA_VISIBLE_DEVICES=1 uv run python m00_molecular_baseline_mie.py && ls -la public/
  ```
- Output / Terminal log:
  ```text
  ====================================================================================
  SCIENTIFIC PIPELINE & ARCHITECTURE DIAGRAMS GENERATED IN `public/`
  ====================================================================================
    - /data/demon_storage_box/levertoxiciteit/public/pipeline_microsoft_graphormer.svg (23,040 bytes)
    - /data/demon_storage_box/levertoxiciteit/public/pipeline_toxtransformer.svg (21,833 bytes)
    - /data/demon_storage_box/levertoxiciteit/public/00_architecture_pipelines.html (58,563 bytes)
  ====================================================================================
  [5/5] Files Saved & Verified]
    - /data/demon_storage_box/levertoxiciteit/data/molecular_baseline_mie.pt (71,373 bytes)
    - /data/demon_storage_box/levertoxiciteit/data/molecular_baseline_mie.json (205,572 bytes)
    - /data/demon_storage_box/levertoxiciteit/public/00_molecular_baseline_mie.html (491,186 bytes)
  ```

## 5. Verification of Completed Tasks
- Verified that all files exist in `public/` and contain no `NaN` values:
  - `public/pipeline_microsoft_graphormer.svg` (23,040 bytes)
  - `public/pipeline_toxtransformer.svg` (21,833 bytes)
  - `public/00_architecture_pipelines.html` (58,563 bytes)
  - `public/00_molecular_baseline_mie.html` (491,186 bytes, updated with Cell 5 containing both architecture diagrams)

## 6. Reflection & Next Step
- **Is everything working properly?**
  - Yes, both figures clearly depict the left-to-right 4-stage pipeline (`Data -> Bewerking -> AI-Model Architectuur -> Output`) with top-to-bottom internal stacks and exact layer types/tensor shapes.
- **What is the immediate follow-up?**
  - Ready for Milestone 1, Step 1.2 (`m01_graph_embedding.py`).

---

## 7. Iterative Visual Redesign: AlexNet / PlotNeuralNet 3D Isometric Style
- **User Feedback:** *"ik had eigenlijk meer iets verwacht zoals hoe alexnet afgebeeld is, echt met plaatjes. Is dat mogelijk?"*
- **Design Upgrade:**
  - Replace text-heavy cards with **true visual scientific illustrations + AlexNet / PlotNeuralNet 3D isometric tensor volumes**:
    1. **Data Column (Top → Bottom with visual icons & graphics):**
       - Drawn 2D molecular structure of `T0901317` / `Bosentan` (colored atom spheres C, N, O, S, F + aromatic rings),
       - Visual database cylinders (`ChEMBL / PubChem / DILIrank`, `PCQM4Mv2 3.7M`, `ToxCast 3,797 Bioassays`),
       - Visual mini-AOP receptor icons (LXRα MIE 167 & BSEP MIE 41).
    2. **Preprocessing Column (Top → Bottom with visual matrices & graphs):**
       - Visual 2D Graph $G=(V,E)$ with node degree halos + $[N \times 9]$ colored OGB feature matrix grid,
       - Visual $[N \times N]$ Floyd-Warshall shortest-path heatmap + shortest-path bond chain,
       - For ToxTransformer: visual SELFIES token tape + sinusoidal positional encoding wave ($\sin/\cos$) + token ID matrix.
    3. **AI-Model Architecture Column (Top → Bottom in AlexNet 3D Isometric Tensor Cuboid style):**
       - Every layer drawn as a **3D isometric tensor volume (front, top, and side faces)** with perspective frustum/pyramid projection lines connecting layer to layer,
       - Explicit 3D tensor dimensions ($[N\!+\!1 \times 768]$, 32-channel 3D attention tensor $[32 \times (N\!+\!1) \times (N\!+\!1)]$, 12 stacked 3D Transformer slabs, 1D neuron vector strips for `[CLS]` $1 \times 768$, $1 \times 512$, $1 \times 64$, and $1 \times 3$ softmax bars).
    4. **Output Column (Top → Bottom with visual plots):**
       - Visual 1D embedding barcode, mini 2D PCA cluster plot, visual 3-class MIE bar chart, and glowing atom/token saliency molecule.
