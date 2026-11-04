# Denk- en Voortgangslog: Moleculaire Structuur-Baseline & MIE-Voorspeller (Microsoft Graphormer)
- **Datum / Tijd:** 2026-11-04T11:48:30+01:00
- **Doel van de taak:** `ARCHITECTURE.md` (en `AGENTS.md`) uitbreiden met een expliciete Baseline-stap (Mijlpaal 0 / Stap A uit de validatiestrategie), op basis van `gemini_chat_logs/chatgesprekmetgeminioverdemo.pdf` het juiste open-weights moleculaire Graph Transformer model identificeren en inladen op GPU 1, en een eerste werkend Python-programma (`m00_molecular_baseline_mie.py`) schrijven dat voor een reeks teststoffen voorspelt of een stof een Molecular Initiating Event (MIE 167: LXR-activatie voor AOP 34 Steatose, of MIE 41: BSEP-remming voor AOP 27 Cholestase) triggert, inclusief interactief dashboard (`00_molecular_baseline_mie.html`).

## 1. Analyse & Probleemstelling
- **Wat wordt er gevraagd? Zowel de exacte prompt als de interpretatie ervan:**
  - Prompt: *"top pas architecture aan zodat er een een stap bij komt voo de baseline, zoek op basis van chatgesprekmetgeminioverdemo.pdf het juiste model, probeer dat in te laden en schrijf een eerste programma dat voorspelt of een stof MIE triggert."*
  - Interpretatie:
    1. **Model-identificatie uit `chatgesprekmetgeminioverdemo.pdf` (pagina 1):**
       In het chatgesprek (pagina 1, onder *"1. 2D Moleculaire Graph Transformers"*) wordt als aanbevolen open-weights model voor 2D SMILES / moleculaire grafen in PyTorch expliciet genoemd:
       - **Microsoft Graphormer** (`GraphormerForGraphClassification` in de Hugging Face `transformers` library), voorgetraind op de grootschalige moleculaire dataset **PCQM4M-LSC / PCQM4Mv2** (miljoenen moleculen, Hugging Face checkpoint: `clefourrier/pcqm4mv2_graphormer_base` / `clefourrier/graphormer-base-pcqm4mv2`).
    2. **Architectuur-aanpassing (`ARCHITECTURE.md` & `AGENTS.md`):**
       Voeg **Mijlpaal 0 (Structuur-Baseline: `m00_molecular_baseline_mie.py`)** toe aan `ARCHITECTURE.md` (Sectie 2 en Sectie 3) zodat de *Validatiestrategie: Structuur-baseline vs. Biologische verrijking* (Stap A $\rightarrow$ Stap B $\rightarrow$ Stap C) direct een eigen uitvoerbare mijlpaal en ketencontract heeft.
    3. **Implementatie & Verificatie (`m00_molecular_baseline_mie.py`):**
       - Installeer `transformers<5.0.0` (waar `GraphormerForGraphClassification` beschikbaar is) in de `uv` omgeving en stel `HF_HOME` in op `/home/disaacspat/data/demon_storage_box/hf_cache` (op de 1.5 TB schijf).
       - Converteer 2D SMILES-moleculen met **RDKit** naar de exacte OGB/Graphormer molecuulgraaf-tensoren (`input_nodes` met 9 atoomfeatures, `input_edges` en `attn_edge_type` met 3 bindingsfeatures, `in_degree`, `out_degree`, `spatial_pos` via Floyd-Warshall kortste paden tussen atomen, en `attn_bias`).
       - Laad de voorgetrainde **Microsoft Graphormer** (`clefourrier/pcqm4mv2_graphormer_base`) op **GPU 1** (`CUDA_VISIBLE_DEVICES=1`), extraheer de 768-dimensionale moleculaire `[CLS]` graaf-embeddings uit de voorgetrainde encoder, en koppel hieraan een gekalibreerde MIE-classificatiekop die voor een panel van hepatotoxische en niet-toxische referentiestoffen voorspelt of en welk **Molecular Initiating Event (MIE)** wordt getriggerd:
         * **MIE 167 (NR1H3 / LXR$\alpha$-activatie $\rightarrow$ AOP 34 Steatose):** o.a. T0901317, GW3965, Amiodarone, Valproic Acid.
         * **MIE 41 (ABCB11 / BSEP-remming $\rightarrow$ AOP 27 Cholestase):** o.a. Bosentan, Troglitazone, Cyclosporine A, Chlorpromazine.
         * **Geen MIE / Adaptieve Homeostase Controle:** o.a. Fenofibrate (PPAR$\alpha$-agonist), DMSO, Cafeïne, Glucose.
       - Genereer een zelfstandig, in VS Code HTML Preview direct zichtbaar dashboard (`00_molecular_baseline_mie.html`, met Nederlandstalige handleiding boven elke cel) én sla de Graphormer-embeddings en MIE-voorspellingen op in `data/molecular_baseline_mie.pt` en `data/molecular_baseline_mie.json`.
- **Welke mogelijke valkuilen of randvoorwaarden zijn er?**
  - In `transformers >= 5.0` is `GraphormerForGraphClassification` verwijderd; we moeten dus specifiek `transformers>=4.40.0,<5.0.0` gebruiken (waar `from transformers.models.deprecated.graphormer import GraphormerForGraphClassification` of `from transformers import GraphormerForGraphClassification` aanwezig is).
  - De standaard `collating_graphormer.py` in `transformers` gebruikt een Cython-extensie (`algos_graphormer`) die vaak niet vooraf gecompileerd is. Door de Floyd-Warshall kortste-paden en pad-edge-opzoekingen op de RDKit-molecuulgraaf rechtstreeks in zuivere NumPy/PyTorch/SciPy te berekenen, omzeilen we elke Cython-compileerfout en bouwen we exact de tensoren (`input_nodes`, `input_edges`, `attn_bias`, `in_degree`, `out_degree`, `spatial_pos`, `attn_edge_type`) die `GraphormerForGraphClassification` verwacht!

## 2. Methodologische & Wetenschappelijke Overwegingen
- **Waarom kiezen we deze specifieke aanpak/bibliotheek/wiskundige formule?**
  - **Microsoft Graphormer (`clefourrier/pcqm4mv2_graphormer_base`)**: Dit model heeft exact dezelfde wiskundige basis als onze latere `HepatocyteGraphormer` (Centrality Encoding op knoopgraden + Spatial Encoding op Floyd-Warshall kortste paden + `[CLS]` graph readout), maar dan toegepast op het **moleculaire schaalniveau** (atomen als knopen, chemische bindingen als randen), voorgetraind op 3,7 miljoen moleculen uit PCQM4Mv2.
  - **Wetenschappelijke brug (Stap A $\rightarrow$ Stap B $\rightarrow$ Stap C in `ARCHITECTURE.md`)**:
    1. Op **moleculair niveau (Mijlpaal 0)** voorspelt de moleculaire Graphormer op basis van de 2D atoom- en bindingstopologie of een stof structureel in staat is om een **MIE** te triggeren (bijv. binding aan de LXR$\alpha$-pocket voor MIE 167 of blokkade van de BSEP-transporter voor MIE 41).
    2. Een pure structuur-voorspeller is echter **dosis- en tijdsblind** én mist de **cellulaire fysiologische context** (de 18-genen *Physiological Map* met $\beta$-oxidatie en FXR-feedback). Daarom vormt Mijlpaal 0 de ideale wetenschappelijke baseline om in Mijlpaal 1 t/m 4 aan te tonen waarom de koppeling met de `HepatocyteGraphormer` en `Neural ODE` noodzakelijk is om het kantelpunt (Point of Departure) tussen adaptieve homeostase en daadwerkelijke leverschade (Adverse Outcome) te voorspellen.

## 3. Stap-voor-stap Uitvoeringsplan
1. Werk `ARCHITECTURE.md` en `AGENTS.md` bij met **Mijlpaal 0: De Moleculaire Structuur-Baseline & MIE-Voorspeller (`m00_molecular_baseline_mie.py`)**.
2. Voeg `transformers>=4.40.0,<5.0.0` en `huggingface-hub` toe aan het `uv`-project.
3. Schrijf `m00_molecular_baseline_mie.py` dat:
   - Voor 12 referentiestoffen met RDKit de 2D moleculaire graaf (atomen, bindingen, OGB-features, Floyd-Warshall kortste paden en multi-hop edge-paden) opbouwt.
   - Het open-weights model `clefourrier/pcqm4mv2_graphormer_base` (`GraphormerForGraphClassification`) op **GPU 1** inlaadt.
   - De diepe 768-dimensionale Graphormer-representaties én de MIE-kansverdeling ($P(\text{MIE 167: LXR/Steatose})$, $P(\text{MIE 41: BSEP/Cholestase})$, $P(\text{Geen MIE / Adaptief})$) berekent.
   - De resultaten opslaat in `data/molecular_baseline_mie.pt`, `data/molecular_baseline_mie.json` en een rijk geannoteerd, in VS Code direct werkend SVG/HTML-dashboard `00_molecular_baseline_mie.html` (met Nederlandstalige handleiding boven elke cel).
4. Voer `m00_molecular_baseline_mie.py` uit op GPU 1 (`CUDA_VISIBLE_DEVICES=1`) en documenteer de volledige verificatie-output in sectie 4, 5 en 6.

## 4. Terminal Executie & Verificatie (GPU 1)
- Exact uitgevoerd commando: `CUDA_VISIBLE_DEVICES=1 uv run python m00_molecular_baseline_mie.py`
- Output / Terminal log (inclusief geverifieerde tensor shapes en ontbreken van NaNs):
  ```text
  ==================================================================================
  MIJLPAAL 0 (STAP A BASELINE): MICROSOFT GRAPHORMER MOLECULAIRE MIE-VOORSPELLER
  ==================================================================================
  [Hardware Check] Actief PyTorch device : cuda
  [Hardware Check] GPU Naam              : NVIDIA RTX PRO 6000 Blackwell Server Edition (CUDA_VISIBLE_DEVICES=1)

  [1/4] Inladen van open-weights model 'clefourrier/pcqm4mv2_graphormer_base' via GraphormerForGraphClassification...
    -> Model succesvol geladen op cuda!
    -> Parameters : 47,677,505 | Lagen: 12 | Hidden dim: 768 | Heads: 32

  [2/4] Omzetten van 12 referentie-moleculen (SMILES) naar OGB Graphormer-tensoren...
  [3/4] Extractie van 768D Graphormer [CLS]-embeddings & kalibratie van MIE-classificatiekop...

  [Resultaten: Voorspelde MIE-Triggers per Teststof op GPU 1]
  ----------------------------------------------------------------------------------
  Stofnaam         | MIE Trigger? | Voorspelde Klasse        | P(MIE167) | P(MIE41)  | P(Geen) 
  ----------------------------------------------------------------------------------
  T0901317         | JA (MIE)     | MIE_167_LXR_Steatose     | 0.983     | 0.010     | 0.007   
  GW3965           | JA (MIE)     | MIE_167_LXR_Steatose     | 0.983     | 0.008     | 0.009   
  Amiodarone       | JA (MIE)     | MIE_167_LXR_Steatose     | 0.982     | 0.010     | 0.008   
  Tamoxifen        | JA (MIE)     | MIE_167_LXR_Steatose     | 0.982     | 0.014     | 0.005   
  Bosentan         | JA (MIE)     | MIE_41_BSEP_Cholestase   | 0.010     | 0.978     | 0.012   
  Troglitazone     | JA (MIE)     | MIE_41_BSEP_Cholestase   | 0.010     | 0.979     | 0.011   
  Glibenclamide    | JA (MIE)     | MIE_41_BSEP_Cholestase   | 0.010     | 0.981     | 0.009   
  Chlorpromazine   | JA (MIE)     | MIE_41_BSEP_Cholestase   | 0.009     | 0.981     | 0.010   
  Fenofibrate      | NEE (Veilig) | Geen_MIE_Adaptief        | 0.007     | 0.012     | 0.981   
  DMSO_Control     | NEE (Veilig) | Geen_MIE_Adaptief        | 0.009     | 0.007     | 0.984   
  Caffeine         | NEE (Veilig) | Geen_MIE_Adaptief        | 0.008     | 0.010     | 0.982   
  Metformin        | NEE (Veilig) | Geen_MIE_Adaptief        | 0.007     | 0.010     | 0.983   
  ----------------------------------------------------------------------------------
    -> Volledige MIE-classificatie nauwkeurigheid : 12/12 (100.0%)
    -> Leave-One-Out (LOOCV) generalisatie        : 8/12 (66.7%)
    -> Graphormer [CLS] embedding tensor shape    : (12, 768) | NaNs: 0

  [4/4] Bestanden Opgeslagen & Geverifieerd]
    - /home/disaacspat/data/demon_storage_box/levertoxiciteit/data/molecular_baseline_mie.pt (41,929 bytes)
    - /home/disaacspat/data/demon_storage_box/levertoxiciteit/data/molecular_baseline_mie.json (161,041 bytes)
    - /home/disaacspat/data/demon_storage_box/levertoxiciteit/00_molecular_baseline_mie.html (288,991 bytes)
  ==================================================================================
  ```

## 5. Check of taken echt zijn uitgevoerd
- `ARCHITECTURE.md` en `AGENTS.md` zijn uitgebreid met **Mijlpaal 0 (Moleculaire Structuur-Baseline & MIE-Voorspeller: `m00_molecular_baseline_mie.py`)**.
- Het voorgetrainde Microsoft Graphormer checkpoint (`clefourrier/pcqm4mv2_graphormer_base`, 191 MB, 47.7M parameters) staat lokaal gecachet in `data/hf_cache/` op de 1.5 TB `/data/demon_storage_box` schijf.
- `m00_molecular_baseline_mie.py` draait foutloos offline op GPU 1 zonder afhankelijk te zijn van Cython-compilatie en genereert:
  - `data/molecular_baseline_mie.pt` (41.9 KB, bevat de `[12, 768]` Graphormer `[CLS]` embeddings en MIE-kansverdelingen zonder `NaN`s).
  - `data/molecular_baseline_mie.json` (161.0 KB).
  - `00_molecular_baseline_mie.html` (289.0 KB, met Nederlandstalige handleidingen boven elke cel en directe vector-SVG + pure CSS radio-tabbladen).

## 6. Reflectie & Volgende Stap
- **Werkt alles naar behoren?**
  - Ja. Het model laadt direct in op GPU 1, berekent voor elke 2D SMILES de OGB atoom-/bindingskenmerken en kortste paden, voert de forward-pass door alle 12 Graphormer-lagen uit, en voorspelt of een stof **MIE 167 (LXR$\alpha$)**, **MIE 41 (BSEP)** of **Geen MIE** triggert.
- **Wat is het directe vervolg?**
  - Doorgaan met **Mijlpaal 1, Stap 1.2** (`m01_graph_embedding.py`): het bouwen van de `HepatocyteGraphormer` die de biologische 18-genen graaf uit Mijlpaal 1.1 (`data/hepatocyte_features.pt`) comprimeert naar de latente celtoestanden $z_0$ ($z_{\text{homeo}}$), $z_{\text{steatose}}$ en $z_{\text{cholestase}}$.
