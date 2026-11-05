# Master Architecture: Multimodal & Multiscale AI for Liver Toxicity
Project: Graphormer + Neural ODE Simulator (Hepatocyte Attractor Dynamics)
Framework: PhD Research at Research Group Innovative Testing (HU University of Applied Sciences Utrecht)

---

## 0. Simplified Explanation of AOP34 and AOP27

  **AOP 34, Steatosis, fatty liver:** When certain chemicals flip a specific switch inside the liver cell, the cell begins uncontrollably synthesizing extra fats while also absorbing fats from the blood. Initially, the cell can safely store or burn this excess fat, but under prolonged overload, the clearance system becomes clogged. The cell swells, becomes damaged by toxic lipid accumulation, and eventually dies, leading to severe liver inflammation and scar tissue.

  **AOP 27, Cholestasis, bile accumulation:** When a chemical blocks the liver's export pump, bile salts can no longer leave the cell and accumulate to dangerous levels. The cell first tries to defend itself by immediately shutting down the production and uptake of new bile salts. However, if the chemical blockade is too severe, the accumulated corrosive bile salts corrode the cell from the inside out via oxidative damage and inflammation, causing liver cells to die and bile flow to come to a complete halt.

## 1. Scientific & Biological Context

This system models how human liver cells (hepatocytes, such as HepaRG) respond to chemical exposure by combining an Adverse Outcome Pathway (AOP) with their underlying Physiological Maps:
1. **AOP 34 (LXR activation leading to Steatosis) & Physiological Maps:** 
  Chemical activation of the nuclear receptor LXR-alpha (NR1H3, MIE 167) disrupts hepatic lipid homeostasis through a coordinated transcriptional cascade, supported by positive feed-forward autoregulation of LXR. Whereas the classical AOP 34 depicts progression linearly toward steatosis (Liver Steatosis, AO 345), embedding it within broader Physiological Maps (Staumont et al., 2025) demonstrates that cellular outcomes are governed by non-linear feedback loops between de novo lipogenesis, fatty acid flux, and compensatory mechanisms (Noble, 2012; Maertens, 2025).

   - **Lipogenic cascade & Increased lipid flux:** 
     LXR activation drives parallel transcriptional pathways via both direct binding to LXREs and interactions with master transcription factors:
     1. Transcriptional upregulation of SREBP-1c (*SREBF1*, KE 264) and ChREBP (KE 66).
     2. Increased expression of key lipogenic enzymes: *FASN* (KE 116), *ACACA*, and *SCD* (*SCD1*, KE 258), leading to accelerated de novo fatty acid synthesis (*De Novo FA Synthesis*, KE 89).
     3. Direct and indirect upregulation of the fatty acid transporter FAT/*CD36* (KE 54), resulting in increased influx of free fatty acids from the peripheral circulation (*FA influx*, KE 115).

   - **Adaptive homeostatic attractor ($z_{\text{homeo}} = z_0$):** 
     In the early phase, the resulting triglyceride accumulation (*Triglyceride accumulation*, KE 291) forms a reversible physiological buffer. Within the physiological map, the hepatocyte network adaptively absorbs this load by activating $\beta$-oxidation (*PPARA* / *CPT1A*) and accelerating VLDL assembly and secretion (*MTTP* / *APOB*). As long as the perturbation force $p$ remains within this dynamic capacity, the cellular trajectory returns to the homeostatic attractor.

   - **Deteriorative tipping point toward the pathological attractor ($z_{\text{steatose}}$):** 
     When chronic exposure or excessive chemical activation overwhelms the homeostatic buffers, adaptive compensation fails:
     1. Excessive triglyceride accumulation (KE 291) induces intracellular crowding, nuclear deformation, and mitochondrial dysfunction (*Mitochondrial dysfunction*, KE 177).
     2. Lipotoxic stress exceeds endogenous defense (*NFE2L2*), leading to activation of stress and inflammatory pathways (*NFKB1*, *TP53*) and Caspase-3-dependent cell death (*CASP3*, apoptosis and necrosis).
     3. This cellular injury drives the transition from simple steatosis (AO 345) to an irreversible pathological state characterized by steatohepatitis (NASH) and tissue fibrosis via Kupffer and stellate cell activation.

2. **AOP 27 (BSEP inhibition leading to Cholestasis) & Physiological Maps:** 
   Chemical inhibition of the canalicular bile acid transporter BSEP (ABCB11, MIE 41) leads to pathological intrahepatic bile acid accumulation (Bile accumulation, KE 214). Whereas a classical AOP approaches this process linearly, embedding it within broader Physiological Maps (Staumont et al., 2025) shows that toxicity does not occur binarily, but is governed by a dynamic balance between multi-directional feedback loops (Noble, 2012; Maertens, 2025) within human hepatocytes (HepaRG; Rodrigues et al., 2018).

   - **Adaptive compensatory response (Homeostatic attractor):** 
     As a physiological counter-response, bile salt accumulation induces transcriptional activation of specific nuclear receptors (Activation of nuclear receptors, KE 288), particularly the farnesoid X receptor (FXR / NR1H4). This initiates a coordinated feedback network to protect the homeostatic attractor state ($z_0$):
     1. Transcriptional repression of de novo bile acid synthesis via downregulation of *CYP7A1*.
     2. Inhibition of further basolateral uptake via *SLC10A1* (NTCP).
     3. Activation of endogenous antioxidant defense via the Nrf2 pathway (*NFE2L2*).
     As long as the exposure context (dose and duration) remains within the physiological bandwidth, this network neutralizes the perturbation force $p$ and the cell returns to homeostasis.

   - **Deteriorative transition (Tipping point toward pathological attractor):** 
     When chronic exposure or high doses overwhelm the adaptive compensatory capacity, the system tips via non-linear amplification into a pathological attractor state ($z_{\text{cholestase}}$):
     1. Cytotoxic bile salts cause mitochondrial dysfunction and excessive reactive oxygen species (*ROS*, KE 1115), resulting in tissue and protein damage (*Peptide oxidation*, KE 209).
     2. Bile acid retention triggers chemotaxis and secretion of inflammatory mediators (*Cytokine release*, KE 87), leading to amplified tissue inflammation (*Inflammation*, KE 149 via NF-κB / *NFKB1*).
     3. Combined lipotoxic/oxidative stress induces a genomic stress response via *TP53*, after which activation of *CASP3* (caspase-3) forces irreversible hepatocyte death (apoptosis and necrosis) and functional cholestatic liver injury (*Cholestasis*, AO 357).

### The Waddington Attractor Concept
A healthy hepatocyte resides in a stable, homeostatic attractor state ($z_{\text{homeo}} = z_0$). Chemical exposure exerts a perturbation force $p$ on this system:
- **Adaptive response (Homeostasis):** At a low dose or adaptive stimulus (such as Fenofibrate via PPARA/CPT1A, VLDL export via MTTP/APOB, or FXR-mediated repression via NR1H4), the network absorbs the perturbation and the trajectory $z(t)$ returns to $z_0$.
- **Adverse Outcome (Pathological tipping):** Upon exceeding a critical threshold, the cell crosses a bifurcation point and $z(t)$ irreversibly converges toward a pathological attractor ($z_{\text{steatose}}$ under overloaded lipogenesis, or $z_{\text{cholestase}}$ under toxic bile acid retention).

### Validation Strategy: Structure Baseline vs. Biological Enrichment
To demonstrate the added value of multimodal and multiscale modeling, two complementary open-weights molecular deep learning models — **Microsoft Graphormer** (2D molecular Graph Transformer pretrained on 3.7M molecules from PCQM4Mv2 via Hugging Face `clefourrier/pcqm4mv2_graphormer_base`) and **ToxTransformer** (`biobricks-ai/toxtransformer`, SMILES $\rightarrow$ SELFIES Sequence & Multi-Task Bioassay Mixture-of-Experts Transformer pretrained on 3,797 toxicological/pharmacological bioassays) — jointly serve as a state-of-the-art structure and SMILES baseline:

* **Step A (Milestone 0 — Dual Molecular Baseline: Graphormer + ToxTransformer):** A panel of test compounds (known steatosis and cholestasis inducers + controls) is fed from SMILES both as a 2D atom and bond graph into **Microsoft Graphormer** (`GraphormerForGraphClassification`) and in SMILES/SELFIES sequence form into **ToxTransformer** (`MoE MultitaskTransformer`). Both models predict (individually and in consensus) whether the compound triggers potent **Molecular Initiating Events (MIEs)**: **MIE 167: LXR$\alpha$ activation** for AOP 34 Steatosis or **MIE 41: BSEP inhibition** for AOP 27 Cholestasis.
* **Step B (Milestones 1 through 3 — Hepatocyte Enrichment):** The same compounds are fed via the Perturbation Encoder (Milestone 2, enriched with the molecular baseline embeddings from Milestone 0) as a dose- and time-dependent perturbation force $p$ onto the `HepatocyteGraphormer` (Milestone 1) + `Neural ODE` (Milestone 3).
* **Step C (Milestone 4 — Evaluation / WP4):** The comparison demonstrates that while a pure molecular structure/SMILES baseline (both Graphormer and ToxTransformer) recognizes static MIE potential, it is blind to *why* and *at what dose/duration* a biological system does or does not tip, because it lacks the exposome and homeostatic context ($\beta$-oxidation, VLDL export, FXR repression). The `HepatocyteGraphormer` + `Neural ODE` successfully predicts this tipping point (Point of Departure) thanks to its integration with *Physiological Maps* and attractor dynamics.

---

## 2. System Architecture & File Names

See [`AGENTS.md`](AGENTS.md) for the strict chain contracts and PyTorch dictionary keys between milestones:
- **Milestone 0 (Dual Molecular Structure & SMILES Baseline & MIE Predictor):**
  - `m00_molecular_baseline_mie.py` (Loads both **Microsoft Graphormer** `clefourrier/pcqm4mv2_graphormer_base` and **ToxTransformer** `biobricks-ai/toxtransformer` on GPU 1, converts SMILES into 2D OGB atom/bond graphs and SELFIES sequences, predicts per model and in consensus whether a compound triggers **MIE 167 (LXR$\alpha$)** or **MIE 41 (BSEP)**, saves `data/molecular_baseline_mie.pt`, and generates `00_molecular_baseline_mie.html`).
- **Milestone 1 (Biological Network Graph, Real Data & HepatocyteGraphormer):**
  - *Step 1.1:* `fetch_real_hepatocyte_data.py` & `m01_data_explorer.py` (Fetches healthy GTEx/HPA and pathological GEO Steatosis/Cholestasis data + MyGene.info + STRING-db with formal AOP documentation, saves `data/hepatocyte_features.pt` and `data/hepatocyte_baseline_data.json`, and provides an interactive Marimo explorer + `01_1_data_explorer.html`).
  - *Step 1.2:* `m01_graph_embedding.py` (Builds the `HepatocyteGraphormer` based on the real data, computes the empirically anchored attractors `z0`, `z_steatose`, `z_cholestase`, and generates `01_network_and_embedding.html`).
- **Milestone 2:** `m02_perturbation_encoder.py` (Generates `02_perturbation_profiles.html`).
- **Milestone 3:** `m03_neural_ode_trajectory.py` (Generates `03_trajectory_dynamics.html`).
- **Milestone 4:** `m04_attractor_simulation.py` (Generates `04_attractor_landscape.html`).

---

## 3. Detailed Specifications per Milestone

### MILESTONE 0: The Dual Molecular Baseline & MIE Predictor (`m00_molecular_baseline_mie.py`)

* **Goal:** Implement **Step A** of the validation strategy with a dual molecular baseline combining both 2D graph topology and SMILES sequence and bioassay transformation:
  1. **Microsoft Graphormer** (`GraphormerForGraphClassification` from Hugging Face `transformers`, checkpoint `clefourrier/pcqm4mv2_graphormer_base`).
  2. **ToxTransformer** (`MoE` / `MultitaskTransformer` from `biobricks-ai/toxtransformer`, open-weights checkpoint `brick/moe/mtransformer.pt` with 2 Transformer experts and gating network, pretrained on SELFIES sequences and 3,797 PubChem/ToxCast/ICE bioassays).
* **Model & Input Pipeline (SMILES $\rightarrow$ 2D Molecular Graph + SELFIES $\rightarrow$ Graphormer & ToxTransformer):**
  1. **Pipeline A — RDKit Molecular Graph & Pretrained Microsoft Graphormer (`clefourrier/pcqm4mv2_graphormer_base`):**
     - Each chemical compound (SMILES) is converted by RDKit into a molecular graph $G_{\text{mol}} = (V_{\text{atoms}}, E_{\text{bonds}})$ with the exact 9 categorical OGB atom features (`input_nodes`), 3 OGB bond features (`input_edges` / `attn_edge_type`), and Floyd-Warshall shortest-path distances (`spatial_pos`).
     - Loaded onto **GPU 1** (12 Transformer layers, 768 hidden dimensions, 32 attention heads, ~47M parameters) and extracts the 768-dimensional molecular `[CLS]` graph embedding $h_{\text{graph}} \in \mathbb{R}^{768}$, the PCQM4Mv2 HOMO-LUMO gap, and per-atom gradient saliency.
  2. **Pipeline B — SMILES $\rightarrow$ SELFIES Tokenization & Pretrained ToxTransformer (`biobricks-ai/toxtransformer`):**
     - Each SMILES string is canonically converted into a 100% robust molecular string representation (**SELFIES**, `selfies.encoder(smiles)`) and tokenized via the `SelfiesPropertyValTokenizer` (`vocab_size = 6232`: 2,431 chemical SELFIES symbols + 3,797 bioassay tokens + property tokens).
     - The Mixture-of-Experts (`MoE`) Transformer model (2 experts of 4 encoder and 4 decoder layers each with $d_{\text{model}} = 256$, plus a 4-layer Transformer gating network) is loaded onto **GPU 1** from `data/toxtransformer_cache/brick/moe/mtransformer.pt`.
     - Extracts the 512-dimensional multi-expert SMILES/SELFIES embedding $h_{\text{tox}} \in \mathbb{R}^{512}$, the expert gating weights $(w_{\text{exp0}}, w_{\text{exp1}})$, the autoregressive decoder bioassay log-probabilities, and per-SELFIES-token gradient saliency.
  3. **MIE Trigger Classification (AOP 34 vs. AOP 27 vs. No MIE) — Graphormer, ToxTransformer & Consensus:**
     - Computes from both the Graphormer `[CLS]` representation and the ToxTransformer SMILES/SELFIES representation (and their combined consensus) for the panel of 12 reference compounds (and arbitrary new SMILES via `predict_mie_for_smiles`) the probability that a compound triggers a **Molecular Initiating Event (MIE)**:
       * **Class 0 — `Geen_MIE_Adaptief` (No MIE / Adaptive):** Compounds that do not trigger a pathological MIE (e.g., *DMSO*, *Glucose*, *Caffeine*, and adaptive PPAR$\alpha$ agonist *Fenofibrate*).
       * **Class 1 — `MIE_167_LXR_Steatose` (AOP 34):** Compounds that trigger LXR$\alpha$ (*NR1H3*, MIE 167) or pro-steatotic lipogenesis (e.g., *T0901317*, *GW3965*, *Amiodarone*, *Valproic Acid*).
       * **Class 2 — `MIE_41_BSEP_Cholestase` (AOP 27):** Compounds that inhibit the bile salt export pump BSEP (*ABCB11*, MIE 41) (e.g., *Bosentan*, *Troglitazone*, *Cyclosporine A*, *Chlorpromazine*).
  4. **Output & Contract for Milestones 2 and 4:**
     - Saves `data/molecular_baseline_mie.pt` and `data/molecular_baseline_mie.json` with the 768D Graphormer embeddings, 512D ToxTransformer embeddings, 2D PCA projections for both models, and the MIE probability distributions (`mie_probabilities`, `toxtransformer_mie_probabilities`, `consensus_mie_probabilities`).
     - Generates the standalone interactive dashboard `00_molecular_baseline_mie.html` (featuring 2D molecular graph + SELFIES token-saliency strip, model comparison Graphormer vs. ToxTransformer vs. Consensus, dual 2D PCA latent space, and summary table).

---

### MILESTONE 1: The Biological Network Graph, Real Data & Graphormer

#### Milestone 1, Step 1.1: Real Healthy & Pathological Data Acquisition, AOP Annotation & Marimo Explorer
* **Goal:** Fetch real human data from validated public databases for both the healthy baseline and pathological (steatosis/NAFLD) states, linked to official OECD AOP literature.
* **Exact Public Data Sources:**
  1. **Healthy Baseline:** 
     - **GTEx Portal & Human Protein Atlas (HPA):** Normal basal liver tissue expression (TPM) for the 18 genes.
     - **MyGene.info / Ensembl API:** Gene structure, genomic length, exons, and protein length.
  2. **Pathological States (Steatosis / NAFLD & Toxic):** 
     - **NCBI Gene Expression Omnibus (GEO):** Human liver transcriptome datasets from NAFLD / steatosis patients (e.g., reference series such as GSE126848 or comparable human steatosis/DILI datasets) to empirically anchor the upregulation of lipogenic genes ($SREBF1$, $FASN$, $CD36$) vs. homeostatic compensation in the feature matrix.
  3. **Protein Interactions:** 
     - **STRING-db API:** Experimentally validated interaction scores (`combined_score`) for the 20 directed AOP edges.
* **Exact 18 genes (fixed order, index 0 through 17) with formal annotations:**
  * *Group 1 - AOP 34 (LXR activation & Steatosis):* 
    `0: NR1H3` (Liver X Receptor alpha - Initiator AOP 34), 
    `1: SREBF1` (SREBP-1c - Upstream lipogenic transcription factor), 
    `2: FASN` (Fatty acid synthase - De novo lipogenesis), 
    `3: ACACA` (Acetyl-CoA carboxylase - Rate-limiting enzyme), 
    `4: SCD` (Stearoyl-CoA desaturase - Fatty acid desaturation), 
    `5: CD36` (Fatty acid translocase - Hepatic fatty acid uptake).
  * *Group 2 - Homeostatic Compensation & Oxidation:* 
    `6: PPARA` (PPAR-alpha - $\beta$-oxidation), 
    `7: CPT1A` (Carnitine palmitoyltransferase 1A - Mitochondrial import), 
    `8: MTTP` (Microsomal triglyceride transfer protein - VLDL assembly), 
    `9: APOB` (Apolipoprotein B - VLDL/LDL scaffold).
  * *Group 3 - AOP 27 (BSEP inhibition & Cholestasis):* 
    `10: NR1H4` (FXR - Bile acid regulator), 
    `11: ABCB11` (BSEP - Molecular target AOP 27), 
    `12: CYP7A1` (Cholesterol 7 alpha-hydroxylase - Bile acid synthesis), 
    `13: SLC10A1` (NTCP - Basolateral bile acid uptake).
  * *Group 4 - Cellular Stress, Inflammation & Apoptosis:* 
    `14: NFE2L2` (Nrf2 - Oxidative stress sensor), 
    `15: NFKB1` (NF-$\kappa$B - Inflammation), 
    `16: TP53` (p53 - Genome guardian/stress), 
    `17: CASP3` (Caspase 3 - Apoptosis / Adverse Outcome).
* **Directed interactions with formal AOP literature references:**
  * `(0, 1, +1, "AOP34: NR1H3 activation transcriptionally upregulates SREBF1 expression")`
  * `(0, 5, +1, "AOP34: LXR activation directly induces CD36 expression, enhancing FA uptake")`
  * `(1, 2, +1, "AOP34: SREBP-1c drives transcription of FASN (De Novo Lipogenesis)")`
  * `(1, 3, +1, "AOP34: SREBP-1c drives transcription of ACACA")`
  * `(1, 4, +1, "AOP34: SREBP-1c drives transcription of SCD")`
  * `(6, 7, +1, "Homeostasis: PPARA activation upregulates CPT1A to promote mitochondrial beta-oxidation")`
  * `(6, 1, -1, "Cross-talk: PPARA activation counter-regulates/inhibits SREBP-1c lipogenic pathway")`
  * `(7, 8, +1, "Homeostasis: Enhanced fatty acid flux stimulates VLDL assembly via MTTP")`
  * `(8, 9, +1, "Homeostasis: MTTP and APOB co-regulate hepatic triglyceride export")`
  * `(2, 6, -1, "Metabolic overload: Excessive lipogenesis suppresses PPARa compensatory signaling")`
  * `(10, 11, +1, "AOP27/FXR: NR1H4 transcriptionally maintains ABCB11 (BSEP) expression")`
  * `(10, 12, -1, "AOP27/FXR: FXR activation represses CYP7A1 to prevent bile acid overload")`
  * `(10, 13, -1, "AOP27/FXR: FXR downregulates basolateral uptake transporter SLC10A1 (NTCP)")`
  * `(12, 10, +1, "Feedback: Bile acid intermediates modulate FXR activity")`
  * `(11, 14, -1, "AOP27 Pathology: Loss of BSEP function leads to intracellular bile acid accumulation & Nrf2 oxidative stress response")`
  * `(5, 14, +1, "Steatotic Lipotoxicity: Excess CD36-mediated fatty acid influx triggers oxidative stress (Nrf2)")`
  * `(14, 15, -1, "Cross-talk: Chronic oxidative stress overwhelms Nrf2, promoting pro-inflammatory NF-kB signaling")`
  * `(15, 16, +1, "Pathology: Inflammation and lipotoxicity activate TP53 DNA damage/stress response")`
  * `(16, 17, +1, "Adverse Outcome: p53 activation triggers Caspase-3 dependent hepatocyte apoptosis")`
  * `(13, 15, -1, "Bile acid retention via NTCP exacerbates inflammatory signaling")`
* **Deliverables Step 1.1:** `fetch_real_hepatocyte_data.py` fetches healthy (GTEx/HPA) and pathological (GEO NAFLD) expression profiles, computes the `[19, 19]` distance matrix, and saves everything in `data/hepatocyte_features.pt` and `data/hepatocyte_baseline_data.json`. `m01_data_explorer.py` launches the Marimo notebook with Plotly visualizations to compare the healthy vs. pathological profiles.

#### Milestone 1, Step 1.2: Graphormer Encoder (`m01_graph_embedding.py`)
* **Input:** Loads `data/hepatocyte_features.pt` from Step 1.1.
* **Architecture (`HepatocyteGraphormer`):** Input projection (`8 -> 64`), Centrality embeddings, `[CLS]` token prepending, Spatial attention bias based on the `[19, 19]` Floyd-Warshall distance matrix, 2 Graphormer transformer layers, and a LayerNorm+Tanh readout to $z_0 \in \mathbb{R}^{[1, 64]}$.
* **Output:** Generates standalone Plotly dashboard `01_network_and_embedding.html`.

---

### MILESTONE 2: The Chemical Perturbation Encoder (`m02_perturbation_encoder.py`)
### MILESTONE 3: The Dynamic Vector Field & Neural ODE (`m03_neural_ode_trajectory.py`)
### MILESTONE 4: Attractor Analysis & Tipping Point Simulation (`m04_attractor_simulation.py`)

---

## 4. Hard Execution Rules
1. GPU Isolation: Always via `CUDA_VISIBLE_DEVICES=1 uv run python <script>.py`.
2. No Matplotlib: Exclusively Plotly.
3. Determinism: `torch.manual_seed(42)` and `np.random.seed(42)`.
4. Mandatory Logging: For every task, first create a log file in `logs/` according to the template in `.clinerules`.
