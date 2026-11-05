# AGENTS.md — Agent Instructions, Tool Discipline & Milestone Contracts

This document is the central manual for autonomous coding agents (such as Cline and subagents) within the **Liver Toxicity Simulator (Graphormer + Neural ODE)** project. Read this file together with [`.clinerules`](.clinerules) and [`ARCHITECTURE.md`](ARCHITECTURE.md) before every action.

---

## 1. Strict Agent Loop & Tool-Call Discipline (Crucial for Local Models)

To prevent the agent from getting stuck in a passive "writing illusion" (dumping code in the chat without creating files), the following **hard behavioral rules** apply:

1. **NEVER dump full code in the chat:** Never write a complete Python script as plain text in your chat response. **Always directly** use your file tool (`write_to_file` / `replace_in_file`) to physically write the file to disk.
2. **One tool action per step, immediately followed by verification:**
   - **Step A (Log):** First create or update the thought log in `logs/task_<name>_thought.md` via your file tool.
   - **Step B (Code):** Physically write the Python script to disk via your file tool.
   - **Step C (Run on GPU 1):** Immediately execute the script via your terminal tool with:
     ```bash
     CUDA_VISIBLE_DEVICES=1 uv run python <script_name>.py
     ```
   - **Step D (Check & Fix):** Verify with `ls -la` and the terminal output whether all expected output files (`.pt`, `.json`, `.html`) actually exist and contain no `NaN` values. If an error occurs, resolve it immediately in a new iteration.
   - **Step E (Finalize Log):** Complete sections 4, 5, and 6 of the thought log in `logs/` with the actual terminal output.
3. **Hardware Isolation (Strict):**
   - **GPU 0** is occupied by Ollama's local models. **Never** touch GPU 0.
   - **GPU 1** is for PyTorch, Graphormer, and Neural ODE. Use `CUDA_VISIBLE_DEVICES=1` with every Python command.

---

## 2. How Milestone 0 & 1.1 Lay the Foundation for Milestones 1.2 through 4 (Chain Contract)

Each milestone builds upon the exact data structures and tensor shapes of the previous milestone. **Milestone 0** (Molecular Structure Baseline) and **Milestone 1.1** (Biological Hepatocyte Data) together lay the foundation for **Milestones 1.2, 2, 3, and 4**:

```text
[Milestone 0: m00_molecular_baseline_mie.py (Microsoft Graphormer + ToxTransformer)]
  │  Loads two complementary open-weights molecular baseline models on GPU 1:
  │  1) Microsoft Graphormer ('clefourrier/pcqm4mv2_graphormer_base'): converts 2D SMILES into
  │     OGB atom/bond graphs with Floyd-Warshall spatial encoding (768D [CLS] embedding).
  │  2) ToxTransformer ('biobricks-ai/toxtransformer' MoE MultitaskTransformer): converts SMILES
  │     into SELFIES sequences and bioassay readouts (512D expert embedding + 3,797 bioassays).
  │  Predicts from SMILES (per model and in consensus) whether a compound triggers MIE 167
  │  (LXR activation) or MIE 41 (BSEP inhibition) (Step A of the validation strategy).
  │
  ├─► data/molecular_baseline_mie.pt    (768D Graphormer + 512D ToxTransformer embeddings & MIE probabilities)
  └─► 00_molecular_baseline_mie.html    (Interactive dashboard: Graphormer vs. ToxTransformer MIE & PCA)
        │
        ▼
[Milestone 1.1: fetch_real_hepatocyte_data.py & m01_data_explorer.py]
  │  Fetches real healthy (GTEx/HPA) and pathological (NCBI GEO Steatosis & Cholestasis)
  │  expression profiles + MyGene.info structural features + STRING-db interactions.
  │
  ├─► data/hepatocyte_baseline_data.json  (Human-readable + Marimo input + AOP annotations)
  └─► data/hepatocyte_features.pt         (PyTorch dict with X_healthy, X_steatosis, X_cholestasis [18, 8],
                                           in_degrees [18], out_degrees [18], dist_matrix_19_cls [19, 19])
        │
        ▼
[Milestone 1.2: m01_graph_embedding.py (HepatocyteGraphormer)]
  │  Encodes graph G_0 + node features X [18, 8] + dist_matrix [19, 19] into R^64:
  │  - z0 (z_homeo)   = Graphormer(X_healthy)    -> Tensor [1, 64]
  │  - z_steatose     = Graphormer(X_steatosis)  -> Tensor [1, 64] (Empirically anchored attractor!)
  │  - z_cholestase   = Graphormer(X_cholestasis)-> Tensor [1, 64] (Empirically anchored attractor!)
  │  Exports: compute_baseline_z0(device) -> (z0, z_steatose, z_cholestase, model, graph_dict)
        │
        ▼
[Milestone 2: m02_perturbation_encoder.py (PerturbationEncoder)]
  │  Encodes chemical exposure (SMILES + dose_uM) via RDKit Morgan FP (1024-bit -> MLP [32]):
  │  - p = [e_chem (32) || log10(1 + dose_uM) (1)] -> Tensor [B, 33]
  │  Exports: encode_perturbation(smiles, dose_uM, device), REFERENCE_COMPOUNDS
        │
        ▼
[Milestone 3: m03_neural_ode_trajectory.py (ODEVectorField & torchdiffeq)]
  │  Couples z0 [B, 64] from Milestone 1.2 and p [B, 33] from Milestone 2 over t in [0, 24 hours] (T=100):
  │  - Uses the empirical attractors (z_homeo, z_steatose, z_cholestase) from Milestone 1.2!
  │  - Integrates dz/dt = f_theta(z(t), p, t) -> trajectories Tensor [100, B, 64]
  │  Exports: simulate_trajectory(z0, p, t_span, device)
        │
        ▼
[Milestone 4: m04_attractor_simulation.py (Dose-Response & Waddington Landscape)]
     Simulates 3 compounds x 8 doses (24 trajectories of [100, 1, 64]), projects via PCA/SVD
     into 2D/3D (including the 3 attractor anchor points) and plots the bifurcation S-curve (Point of Departure).
```

---

## 3. Detailed Specification for Milestone 1, Step 1.1

When instructed to execute **Milestone 1, Step 1.1**, create exactly two code files (`fetch_real_hepatocyte_data.py` and `m01_data_explorer.py`) and two data files (`data/hepatocyte_baseline_data.json` and `data/hepatocyte_features.pt`).

### 3.1. Part 1: `fetch_real_hepatocyte_data.py`

#### A. The 18 Genes (Fixed order, index `0` through `17`)
Every gene must be documented in the code with its role and OECD AOP reference:
- **Group 1 — `AOP34_Steatosis` (AOP 34: LXR activation leading to Steatosis):**
  - `0: NR1H3` (LXR-alpha, MIE 167 — Initiator of AOP 34)
  - `1: SREBF1` (SREBP-1c, KE 264 — Master lipogenic transcription factor)
  - `2: FASN` (Fatty acid synthase, KE 116 — *De novo* fatty acid synthesis)
  - `3: ACACA` (Acetyl-CoA carboxylase — Rate-limiting lipogenic enzyme)
  - `4: SCD` (Stearoyl-CoA desaturase 1, KE 258 — Fatty acid desaturation)
  - `5: CD36` (FAT/CD36, KE 54 — Hepatic fatty acid uptake / influx)
- **Group 2 — `Homeostasis` (Physiological Map: $\beta$-oxidation & VLDL export):**
  - `6: PPARA` (PPAR-alpha — Regulator of mitochondrial $\beta$-oxidation)
  - `7: CPT1A` (Carnitine palmitoyltransferase 1A — Transport of fatty acids into mitochondria)
  - `8: MTTP` (Microsomal triglyceride transfer protein — VLDL assembly)
  - `9: APOB` (Apolipoprotein B — Structural core of VLDL export particles)
- **Group 3 — `AOP27_Cholestasis` (AOP 27: BSEP inhibition leading to Cholestasis):**
  - `10: NR1H4` (FXR, KE 288 — Bile acid sensor and homeostatic regulator)
  - `11: ABCB11` (BSEP, MIE 41 — Canalicular bile salt export pump, target of AOP 27)
  - `12: CYP7A1` (Cholesterol 7$\alpha$-hydroxylase — Rate-limiting enzyme in bile acid synthesis)
  - `13: SLC10A1` (NTCP — Basolateral bile acid uptake transporter)
- **Group 4 — `Stress_Apoptosis` (Convergent Organ Injury & Adverse Outcome):**
  - `14: NFE2L2` (Nrf2 — Sensor and regulator against oxidative stress / ROS, KE 1115)
  - `15: NFKB1` (NF-$\kappa$B — Pro-inflammatory transcription factor, KE 149)
  - `16: TP53` (p53 — Cellular and genomic stress sensor)
  - `17: CASP3` (Caspase-3 — Executioner of apoptosis / hepatocyte death, AO 345 / AO 357)

#### B. The 20 Directed Interactions (`AOP_DIRECTED_EDGES`)
Include these exact 20 tuples `(source_idx, target_idx, sign, citation)`:
1. `(0, 1, +1, "AOP34 (KE 264): NR1H3 activation transcriptionally upregulates SREBF1 expression")`
2. `(0, 5, +1, "AOP34 (KE 54): LXR activation directly induces CD36 expression, enhancing FA uptake")`
3. `(1, 2, +1, "AOP34 (KE 116): SREBP-1c drives transcription of FASN (De Novo Lipogenesis)")`
4. `(1, 3, +1, "AOP34 (KE 89): SREBP-1c drives transcription of ACACA")`
5. `(1, 4, +1, "AOP34 (KE 258): SREBP-1c drives transcription of SCD")`
6. `(6, 7, +1, "Homeostasis: PPARA activation upregulates CPT1A to promote mitochondrial beta-oxidation")`
7. `(6, 1, -1, "Cross-talk: PPARA activation counter-regulates/inhibits SREBP-1c lipogenic pathway")`
8. `(7, 8, +1, "Homeostasis: Enhanced fatty acid flux stimulates VLDL assembly via MTTP")`
9. `(8, 9, +1, "Homeostasis: MTTP and APOB co-regulate hepatic triglyceride export")`
10. `(2, 6, -1, "Metabolic overload: Excessive lipogenesis suppresses PPARA compensatory signaling")`
11. `(10, 11, +1, "AOP27/FXR (KE 288): NR1H4 transcriptionally maintains ABCB11 (BSEP) expression")`
12. `(10, 12, -1, "AOP27/FXR: FXR activation represses CYP7A1 to prevent bile acid overload")`
13. `(10, 13, -1, "AOP27/FXR: FXR downregulates basolateral uptake transporter SLC10A1 (NTCP)")`
14. `(12, 10, +1, "Feedback: Bile acid synthesis intermediates activate FXR")`
15. `(11, 14, -1, "AOP27 Pathology (KE 214): Loss of BSEP function leads to bile acid accumulation & Nrf2 stress")`
16. `(5, 14, +1, "Steatotic Lipotoxicity: Excess CD36-mediated fatty acid influx triggers oxidative stress (Nrf2)")`
17. `(14, 15, -1, "Cross-talk: Adaptive Nrf2 antioxidant response suppresses pro-inflammatory NF-kB signaling")`
18. `(15, 16, +1, "Pathology (KE 149): Chronic inflammation and lipotoxicity activate TP53 stress response")`
19. `(16, 17, +1, "Adverse Outcome: TP53 activation triggers CASP3-dependent hepatocyte apoptosis")`
20. `(13, 15, +1, "AOP27 Pathology: Continued bile acid influx via NTCP exacerbates NF-kB inflammation")`

#### C. The 8 Features per Gene & The 3 States (Healthy, Steatosis, Cholestasis)
Fetch live data via `requests` from the **MyGene.info API** (`https://mygene.info/v3/query` and `/v3/gene`) and **STRING-db API** (`https://string-db.org/api/json/network?identifiers=...&species=9606`), combined with validated empirical reference values from **GTEx/HPA** (healthy liver TPM) and **NCBI GEO** (e.g., GSE126848 for NAFLD/steatosis and HepaRG cholestasis profiles from Rodrigues et al., 2018):
- **Feature 0 (`expr_state_log2_tpm`):** State-dependent mRNA expression $\log_2(\text{TPM} + 1)$:
  - In `X_healthy`: Healthy basal liver expression (GTEx/HPA).
  - In `X_steatosis`: Pathological steatosis expression (GEO NAFLD: strong upregulation of `SREBF1`, `FASN`, `ACACA`, `SCD`, `CD36`, stress on `NFE2L2`/`NFKB1`/`TP53`/`CASP3`, and suppression of `PPARA`).
  - In `X_cholestasis`: Pathological cholestasis expression (loss of function/downregulation of `ABCB11`, strong induction of `NFE2L2`, `NFKB1`, `TP53`, `CASP3`, and compensatory repression of `CYP7A1`/`SLC10A1`).
- **Feature 1 (`log10_gene_length_bp`):** Genomic span from MyGene.info (`genomic_pos.end - genomic_pos.start`, in $\log_{10}\text{bp}$).
- **Feature 2 (`exon_or_transcript_count`):** Number of exons/transcripts from MyGene.info (`exons` / `refseq`).
- **Feature 3 (`protein_length_norm`):** Protein length / amino acids (from MyGene.info / UniProt, scaled by `/ 1000.0`).
- **Feature 4 (`in_degree_aop`):** In-degree within the 20 directed AOP interactions.
- **Feature 5 (`out_degree_aop`):** Out-degree within the 20 directed AOP interactions.
- **Feature 6 (`mean_string_score`):** Mean experimental STRING-db `score` with connected genes in the network.
- **Feature 7 (`pathway_role_index`):** Numerical role encoding (`0.25` = Nuclear receptor/TF, `0.50` = Metabolic enzyme, `0.75` = Transporter/Export, `1.00` = Stress/Apoptosis effector).

> **Important for Z-score normalization:** Compute the mean $\mu \in \mathbb{R}^8$ and standard deviation $\sigma \in \mathbb{R}^8$ on `X_healthy_raw`, and normalize `X_healthy`, `X_steatosis`, and `X_cholestasis` **relative to the same healthy baseline** ($\mu_{\text{healthy}}, \sigma_{\text{healthy}}$)! This ensures that pathological shifts ($\Delta X$) in Milestone 1.2 and Milestone 3 remain mathematically on the exact same scale.

#### D. The `[19, 19]` Graphormer Shortest-Path Matrix (`dist_matrix_19_cls`)
- Compute shortest-path distances `dist_matrix_18` (`shape [18, 18]`, self-distance = `0`, unreachable = `9`, clipped at `max_dist = 9`) using Floyd-Warshall on the **undirected** version of the 18-gene graph.
- Build `dist_matrix_19_cls` (`shape [19, 19]`, dtype `torch.long`) for the Graphormer `[CLS]` node at index `0`:
  - `dist_matrix_19_cls[0, 0] = 0`
  - `dist_matrix_19_cls[0, 1:] = 10` and `dist_matrix_19_cls[1:, 0] = 10` (special virtual `[CLS]` distance)
  - `dist_matrix_19_cls[1:, 1:] = dist_matrix_18`

#### E. Mandatory Keys in `data/hepatocyte_features.pt` (Contract for Milestone 1.2)
Save a dictionary using `torch.save(..., "data/hepatocyte_features.pt")` with these exact keys:
- `"gene_symbols"`: `list[str]` (length 18)
- `"gene_groups"`: `list[str]` (length 18)
- `"X_healthy_raw"`, `"X_steatosis_raw"`, `"X_cholestasis_raw"`: `FloatTensor [18, 8]`
- `"X_healthy"`, `"X_steatosis"`, `"X_cholestasis"`: `FloatTensor [18, 8]` (normalized relative to healthy baseline)
- `"in_degrees"`, `"out_degrees"`: `LongTensor [18]` (clipped at max `10`)
- `"dist_matrix_18"`: `LongTensor [18, 18]`
- `"dist_matrix_19_cls"`: `LongTensor [19, 19]`
- `"edge_index"`: `LongTensor [2, 20]`
- `"edge_sign"`: `FloatTensor [20]` (`+1.0` or `-1.0`)
- `"edge_string_score"`: `FloatTensor [20]`

---

### 3.2. Part 2: `m01_data_explorer.py` (Interactive Marimo Notebook)

Create a valid Marimo notebook (`import marimo as mo`) that loads `data/hepatocyte_baseline_data.json`:
1. **Interactive Marimo UI Widgets:**
   - `group_filter = mo.ui.multiselect(...)` for the 4 AOP groups.
   - `state_selector = mo.ui.dropdown(options=["Healthy (Homeostasis)", "Steatosis (AOP 34)", "Cholestasis (AOP 27)", "Delta: Steatosis vs Healthy", "Delta: Cholestasis vs Healthy"], ...)` to dynamically switch node size/color on the network between healthy and diseased states!
   - `string_slider = mo.ui.slider(...)` for the minimum STRING-db interaction score.
2. **Three Interactive Plotly Visualizations (`plotly.graph_objects`):**
   - **Visualization 1 (Biological AOP Network Graph):** 2D network where node size scales with the selected expression state (`state_selector`), node color corresponds to the 4 AOP groups, and edges are green (`+1` activation) or red dashed (`-1` repression) with the full **OECD AOP citation** and **STRING-db score** in the hover tooltip.
   - **Visualization 2 (Healthy vs. Pathological Expression & Feature Heatmap `[18, 8]`):** Interactive comparison of the 8 features across the 18 genes and a grouped bar/heatmap of Healthy vs. Steatosis vs. Cholestasis ($\log_2 \text{TPM}$).
   - **Visualization 3 (The `[19, 19]` Graphormer Shortest-Path Matrix):** Heatmap of `dist_matrix_19_cls` including `[CLS]` at index `0`.
3. **Verification:**
   - Verify that the script validates without errors using:
     ```bash
     CUDA_VISIBLE_DEVICES=1 uv run marimo check m01_data_explorer.py
     ```
