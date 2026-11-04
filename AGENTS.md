# AGENTS.md — Agent Instructies, Tool-Discipline & Mijlpaal-Contracten

Dit document is de centrale handleiding voor autonome codeer-agents (zoals Cline en subagents) binnen het project **Levertoxiciteit Simulator (Graphormer + Neural ODE)**. Lees dit bestand samen met [`.clinerules`](.clinerules) en [`ARCHITECTURE.md`](ARCHITECTURE.md) vóór elke actie.

---

## 1. Strikte Agent Loop & Tool-Call Discipline (Cruciaal voor Lokale Modellen)

Om te voorkomen dat de agent blijft hangen in een passieve "schrijf-illusie" (code uitschrijven in de chat zonder bestanden aan te maken), gelden de volgende **harde gedragsregels**:

1. **NOOIT volledige code in de chat dumpen:** Schrijf nooit een compleet Python-script als platte tekst in je chat-antwoord. Gebruik **altijd direct** je bestands-tool (`write_to_file` / `replace_in_file`) om het bestand fysiek op de schijf te zetten.
2. **Eén tool-actie per stap, direct gevolgd door verificatie:**
   - **Stap A (Log):** Maak of update eerst het denklogboek in `logs/task_<naam>_thought.md` via je bestands-tool.
   - **Stap B (Code):** Schrijf het Python-script fysiek naar schijf via je bestands-tool.
   - **Stap C (Run op GPU 1):** Voer het script direct uit via je terminal-tool met:
     ```bash
     CUDA_VISIBLE_DEVICES=1 uv run python <scriptnaam>.py
     ```
   - **Stap D (Check & Fix):** Controleer met `ls -la` en de terminal-output of alle verwachte output-bestanden (`.pt`, `.json`, `.html`) daadwerkelijk bestaan en geen `NaN`-waarden bevatten. Als er een fout optreedt, los deze direct op in een nieuwe iteratie.
   - **Stap E (Log afronden):** Vul sectie 4, 5 en 6 van het denklogboek in `logs/` aan met de echte terminal-output.
3. **Hardware-isolatie (Strikt):**
   - **GPU 0** is bezet door Ollama's lokale modellen'. Raak GPU 0 **nooit** aan.
   - **GPU 1** is voor PyTorch, Graphormer en Neural ODE. Gebruik bij élk Python-commando `CUDA_VISIBLE_DEVICES=1`.

---

## 2. Hoe Mijlpaal 0 & 1.1 de Fundering leggen voor Mijlpaal 1.2 t/m 4 (Ketencontract)

Elke mijlpaal bouwt voort op de exacte datastructuren en tensor-shapes van de vorige mijlpaal. **Mijlpaal 0** (Moleculaire Structuur-Baseline) en **Mijlpaal 1.1** (Biologische Hepatocyt-Data) leggen samen het fundament voor **Mijlpaal 1.2, 2, 3 en 4**:

```text
[Mijlpaal 0: m00_molecular_baseline_mie.py (Microsoft Graphormer PCQM4Mv2)]
  │  Laadt open-weights GraphormerForGraphClassification ('clefourrier/pcqm4mv2_graphormer_base'),
  │  zet 2D SMILES om naar OGB atoom-/bindingsgrafen en voorspelt of een stof MIE 167 (LXR-activatie)
  │  of MIE 41 (BSEP-remming) triggert (Stap A van de validatiestrategie).
  │
  ├─► data/molecular_baseline_mie.pt    (768D Graphormer [CLS] embeddings + MIE-kansen)
  └─► 00_molecular_baseline_mie.html    (Interactief dashboard: MIE-voorspellingen & 2D PCA ruimte)
        │
        ▼
[Mijlpaal 1.1: fetch_real_hepatocyte_data.py & m01_data_explorer.py]
  │  Haalt echte gezonde (GTEx/HPA) én pathologische (NCBI GEO Steatose & Cholestase)
  │  expressieprofielen + MyGene.info structuurfeatures + STRING-db interacties op.
  │
  ├─► data/hepatocyte_baseline_data.json  (Mens-leesbaar + Marimo input + AOP-annotaties)
  └─► data/hepatocyte_features.pt         (PyTorch dict met X_healthy, X_steatosis, X_cholestasis [18, 8],
                                           in_degrees [18], out_degrees [18], dist_matrix_19_cls [19, 19])
        │
        ▼
[Mijlpaal 1.2: m01_graph_embedding.py (HepatocyteGraphormer)]
  │  Encodeert de graaf G_0 + knoopfeatures X [18, 8] + dist_matrix [19, 19] naar R^64:
  │  - z0 (z_homeo)   = Graphormer(X_healthy)    -> Tensor [1, 64]
  │  - z_steatose     = Graphormer(X_steatosis)  -> Tensor [1, 64] (Empirisch verankerde attractor!)
  │  - z_cholestase   = Graphormer(X_cholestasis)-> Tensor [1, 64] (Empirisch verankerde attractor!)
  │  Exporteert: compute_baseline_z0(device) -> (z0, z_steatose, z_cholestase, model, graph_dict)
        │
        ▼
[Mijlpaal 2: m02_perturbation_encoder.py (PerturbationEncoder)]
  │  Encodeert chemische blootstelling (SMILES + dose_uM) via RDKit Morgan FP (1024-bit -> MLP [32]):
  │  - p = [e_chem (32) || log10(1 + dose_uM) (1)] -> Tensor [B, 33]
  │  Exporteert: encode_perturbation(smiles, dose_uM, device), REFERENCE_COMPOUNDS
        │
        ▼
[Mijlpaal 3: m03_neural_ode_trajectory.py (ODEVectorField & torchdiffeq)]
  │  Koppelt z0 [B, 64] uit Mijlpaal 1.2 en p [B, 33] uit Mijlpaal 2 over t in [0, 24 uur] (T=100):
  │  - Gebruikt de empirische attractoren (z_homeo, z_steatose, z_cholestase) uit Mijlpaal 1.2!
  │  - Integreert dz/dt = f_theta(z(t), p, t) -> trajectories Tensor [100, B, 64]
  │  Exporteert: simulate_trajectory(z0, p, t_span, device)
        │
        ▼
[Mijlpaal 4: m04_attractor_simulation.py (Dosis-Respons & Waddington Landschap)]
     Simuleert 3 stoffen x 8 doseringen (24 trajecten van [100, 1, 64]), projecteert via PCA/SVD
     naar 2D/3D (inclusief de 3 attractor-ankerpunten) en plot de bifurcatie S-curve (Point of Departure).
```

---

## 3. Gedetailleerde Specificatie voor Mijlpaal 1, Stap 1.1

Wanneer je opdracht krijgt om **Mijlpaal 1, Stap 1.1** uit te voeren, maak je exact twee codebestanden (`fetch_real_hepatocyte_data.py` en `m01_data_explorer.py`) en twee databestanden (`data/hepatocyte_baseline_data.json` en `data/hepatocyte_features.pt`).

### 3.1. Deel 1: `fetch_real_hepatocyte_data.py`

#### A. De 18 Genen (Vaste volgorde, index `0` t/m `17`)
Elk gen moet in de code gedocumenteerd zijn met zijn rol en OECD AOP-verwijzing:
- **Groep 1 — `AOP34_Steatosis` (AOP 34: LXR-activatie richting Steatose):**
  - `0: NR1H3` (LXR-alpha, MIE 167 — Initiator van AOP 34)
  - `1: SREBF1` (SREBP-1c, KE 264 — Meester-transcriptiefactor lipogenese)
  - `2: FASN` (Fatty acid synthase, KE 116 — *De novo* vetzuursynthese)
  - `3: ACACA` (Acetyl-CoA carboxylase — Snelheidsbepalend enzym lipogenese)
  - `4: SCD` (Stearoyl-CoA desaturase 1, KE 258 — Vetzuur-desaturatie)
  - `5: CD36` (FAT/CD36, KE 54 — Hepatische vetzuur-opname / influx)
- **Groep 2 — `Homeostasis` (Physiological Map: $\beta$-oxidatie & VLDL-export):**
  - `6: PPARA` (PPAR-alpha — Regulator mitochondriële $\beta$-oxidatie)
  - `7: CPT1A` (Carnitine palmitoyltransferase 1A — Transport vetzuren naar mitochondriën)
  - `8: MTTP` (Microsomal triglyceride transfer protein — VLDL-assemblage)
  - `9: APOB` (Apolipoprotein B — Structurele kern van VLDL-exportdeeltjes)
- **Groep 3 — `AOP27_Cholestasis` (AOP 27: BSEP-remming richting Cholestase):**
  - `10: NR1H4` (FXR, KE 288 — Galzuursensor en homeostatische regulator)
  - `11: ABCB11` (BSEP, MIE 41 — Canaliculaire galzout-exportpomp, doelwit AOP 27)
  - `12: CYP7A1` (Cholesterol 7$\alpha$-hydroxylase — Snelheidsbepalend enzym galzuursynthese)
  - `13: SLC10A1` (NTCP — Basolaterale galzuur-opnametransporter)
- **Groep 4 — `Stress_Apoptosis` (Convergente Orgaanschade & Adverse Outcome):**
  - `14: NFE2L2` (Nrf2 — Sensor en regulator tegen oxidatieve stress / ROS, KE 1115)
  - `15: NFKB1` (NF-$\kappa$B — Pro-inflammatoire transcriptiefactor, KE 149)
  - `16: TP53` (p53 — Cellulaire en genomische stress-sensor)
  - `17: CASP3` (Caspase-3 — Uitvoerder van apoptose / hepatocytendood, AO 345 / AO 357)

#### B. De 20 Gerichte Interacties (`AOP_DIRECTED_EDGES`)
Exact deze 20 tuples `(source_idx, target_idx, sign, citation)` opnemen:
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

#### C. De 8 Features per Gen & De 3 Toestanden (Gezond, Steatose, Cholestase)
Haal via `requests` live data op bij **MyGene.info API** (`https://mygene.info/v3/query` en `/v3/gene`) en **STRING-db API** (`https://string-db.org/api/json/network?identifiers=...&species=9606`), gecombineerd met gevalideerde empirische referentiewaarden uit **GTEx/HPA** (gezonde lever TPM) en **NCBI GEO** (bijv. GSE126848 voor NAFLD/steatose en HepaRG cholestase-profielen uit Rodrigues et al., 2018):
- **Feature 0 (`expr_state_log2_tpm`):** Toestandsafhankelijke mRNA-expressie $\log_2(\text{TPM} + 1)$:
  - In `X_healthy`: Gezonde basale leverexpressie (GTEx/HPA).
  - In `X_steatosis`: Pathologische steatose-expressie (GEO NAFLD: sterke opregulatie van `SREBF1`, `FASN`, `ACACA`, `SCD`, `CD36`, stress op `NFE2L2`/`NFKB1`/`TP53`/`CASP3`, en onderdrukking van `PPARA`).
  - In `X_cholestasis`: Pathologische cholestase-expressie (functieverlies/downregulatie van `ABCB11`, sterke inductie van `NFE2L2`, `NFKB1`, `TP53`, `CASP3`, en compensatoire repressie van `CYP7A1`/`SLC10A1`).
- **Feature 1 (`log10_gene_length_bp`):** Genomische span uit MyGene.info (`genomic_pos.end - genomic_pos.start`, in $\log_{10}\text{bp}$).
- **Feature 2 (`exon_or_transcript_count`):** Aantal exonen/transcripten uit MyGene.info (`exons` / `refseq`).
- **Feature 3 (`protein_length_norm`):** Eiwitlengte / aminozuren (uit MyGene.info / UniProt, geschaald door `/ 1000.0`).
- **Feature 4 (`in_degree_aop`):** In-degree binnen de 20 gerichte AOP-interacties.
- **Feature 5 (`out_degree_aop`):** Out-degree binnen de 20 gerichte AOP-interacties.
- **Feature 6 (`mean_string_score`):** Gemiddelde experimentele STRING-db `score` met verbonden genen in het netwerk.
- **Feature 7 (`pathway_role_index`):** Numerieke rol-codering (`0.25` = Nucleaire receptor/TF, `0.50` = Metabool enzym, `0.75` = Transporter/Export, `1.00` = Stress/Apoptose effector).

> **Belangrijk voor Z-score normalisatie:** Bereken de mean $\mu \in \mathbb{R}^8$ en standaarddeviatie $\sigma \in \mathbb{R}^8$ op `X_healthy_raw`, en normaliseer `X_healthy`, `X_steatosis` en `X_cholestasis` **ten opzichte van dezelfde gezonde baseline** ($\mu_{\text{healthy}}, \sigma_{\text{healthy}}$)! Zo blijven de pathologische verschuivingen ($\Delta X$) in Mijlpaal 1.2 en Mijlpaal 3 wiskundig op exact dezelfde schaal.

#### D. De `[19, 19]` Graphormer Shortest-Path Matrix (`dist_matrix_19_cls`)
- Bereken met Floyd-Warshall op de **ongerichte** versie van de 18-genen graaf de kortste-padafstanden `dist_matrix_18` (`shape [18, 18]`, zelf-afstand = `0`, onbereikbaar = `9`, geclipt op `max_dist = 9`).
- Bouw `dist_matrix_19_cls` (`shape [19, 19]`, dtype `torch.long`) voor de Graphormer `[CLS]`-knoop op index `0`:
  - `dist_matrix_19_cls[0, 0] = 0`
  - `dist_matrix_19_cls[0, 1:] = 10` en `dist_matrix_19_cls[1:, 0] = 10` (speciale virtuele `[CLS]`-afstand)
  - `dist_matrix_19_cls[1:, 1:] = dist_matrix_18`

#### E. Verplichte Sleutels in `data/hepatocyte_features.pt` (Contract voor Mijlpaal 1.2)
Sla met `torch.save(..., "data/hepatocyte_features.pt")` een dictionary op met exact deze sleutels:
- `"gene_symbols"`: `list[str]` (lengte 18)
- `"gene_groups"`: `list[str]` (lengte 18)
- `"X_healthy_raw"`, `"X_steatosis_raw"`, `"X_cholestasis_raw"`: `FloatTensor [18, 8]`
- `"X_healthy"`, `"X_steatosis"`, `"X_cholestasis"`: `FloatTensor [18, 8]` (genormaliseerd t.o.v. gezonde baseline)
- `"in_degrees"`, `"out_degrees"`: `LongTensor [18]` (geclipt op max `10`)
- `"dist_matrix_18"`: `LongTensor [18, 18]`
- `"dist_matrix_19_cls"`: `LongTensor [19, 19]`
- `"edge_index"`: `LongTensor [2, 20]`
- `"edge_sign"`: `FloatTensor [20]` (`+1.0` of `-1.0`)
- `"edge_string_score"`: `FloatTensor [20]`

---

### 3.2. Deel 2: `m01_data_explorer.py` (Interactief Marimo Notebook)

Maak een geldig Marimo-notebook (`import marimo as mo`) dat `data/hepatocyte_baseline_data.json` inlaadt:
1. **Interactieve Marimo UI Widgets:**
   - `group_filter = mo.ui.multiselect(...)` voor de 4 AOP-groepen.
   - `state_selector = mo.ui.dropdown(options=["Healthy (Homeostasis)", "Steatosis (AOP 34)", "Cholestasis (AOP 27)", "Delta: Steatosis vs Healthy", "Delta: Cholestasis vs Healthy"], ...)` om de knoopgrootte/kleur op het netwerk dynamisch te schakelen tussen gezond en ziek!
   - `string_slider = mo.ui.slider(...)` voor de minimale STRING-db interactiescore.
2. **Drie Interactieve Plotly Visualisaties (`plotly.graph_objects`):**
   - **Visualisatie 1 (Biologische AOP-Netwerkgraaf):** 2D netwerk waarbij knoopgrootte schaalt met de gekozen expressietoestand (`state_selector`), knoopkleur correspondeert met de 4 AOP-groepen, en randen groen (`+1` activatie) of rood gestippeld (`-1` repressie) zijn met in de hover-tooltip de volledige **OECD AOP-citatie** én de **STRING-db score**.
   - **Visualisatie 2 (Gezond vs. Pathologisch Expressie & Feature Heatmap `[18, 8]`):** Interactieve vergelijking van de 8 features over de 18 genen én een gegroepeerde bar/heatmap van Gezond vs. Steatose vs. Cholestase ($\log_2 \text{TPM}$).
   - **Visualisatie 3 (De `[19, 19]` Graphormer Shortest-Path Matrix):** Heatmap van `dist_matrix_19_cls` inclusief `[CLS]` op index `0`.
3. **Verificatie:**
   - Controleer of het script foutloos valideert met:
     ```bash
     CUDA_VISIBLE_DEVICES=1 uv run marimo check m01_data_explorer.py
     ```
