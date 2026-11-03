# Master Architecture: Multimodal & Multiscale AI voor Levertoxiciteit
Project: Graphormer + Neural ODE Simulator (Hepatocyte Attractor Dynamics)
Kader: Promotieonderzoek Lectoraat Innovative Testing (Hogeschool Utrecht)

---

## 0. Versimpelde uitleg van AOP34 en AOP27

  **AOP 34, leververvetting:** Wanneer bepaalde chemische stoffen een specifieke schakelaar in de levercel aanzetten, gaat de cel ongecontroleerd extra vetten aanmaken én vetten uit het bloed opzuigen. In het begin kan de cel dit extra vet nog veilig opslaan of verbranden, maar bij langdurige overbelasting raakt het opruimsysteem verstopt. De cel zwelt op, raakt beschadigd door giftige vetstapeling en sterft uiteindelijk af, wat leidt tot ernstige leverontsteking en littekenweefsel.

  **AOP 27, Galstuwing**: Als een chemische stof de afvoerpomp van de lever blokkeert, kunnen galzouten de cel niet meer verlaten en hopen ze zich gevaarlijk op. De cel probeert zich eerst te verdedigen door direct de aanmaak en opname van nieuwe galzouten stil te leggen. Als de chemische blokkade echter te zwaar is, vreten de opgestapelde bijtende galzouten de cel van binnenuit aan met zuurstofschade en ontstekingen, waardoor levercellen afsterven en de galstroom volledig vastloopt.

## 1. Wetenschappelijke & Biologische Context

Dit systeem modelleert hoe menselijke levercellen (hepatocyten, zoals HepaRG) reageren op chemische blootstelling door een Adverse Outcome Pathway (AOP) en hun onderliggende Physiological maps te combineren:
1. **AOP 34 (LXR-activatie richting Steatose) & Physiological Maps:** 
  Chemische activatie van de nucleaire receptor LXR-alpha (NR1H3, MIE 167) verstoort de hepatische lipidenhomeostase door een gecoördineerde transcriptionele cascade, ondersteund door een positieve feed-forward autoregulatie van LXR. Waar de klassieke AOP 34 de progressie lineair schetst richting steatose (Liver Steatosis, AO 345), toont inbedding in bredere Physiological maps (Staumont et al., 2025) dat cellulaire uitkomsten worden bepaald door niet-lineaire terugkoppelingen tussen de novo lipogenese, vetzuurflux en compensatiemechanismen (Noble, 2012; Maertens, 2025).

   - **Lipogene cascade & Verhoogde vetflux:** 
     LXR-activatie stuurt parallelle transcriptieroutes aan via zowel directe binding aan LXREs als interacties met meester-transcriptiefactoren:
     1. Transcriptionele opregulatie van SREBP-1c (*SREBF1*, KE 264) en ChREBP (KE 66).
     2. Verhoogde expressie van lipogene sleutelenzymen: *FASN* (KE 116), *ACACA* en *SCD* (*SCD1*, KE 258), wat leidt tot versnelde de novo vetzuursynthese (*De Novo FA Synthesis*, KE 89).
     3. Directe en indirecte opregulatie van de vetzuurtransporter FAT/*CD36* (KE 54), resulterend in verhoogde influx van vrije vetzuren vanuit de perifere circulatie (*FA influx*, KE 115).

   - **Adaptieve homeostatische attractor ($z_{\text{homeo}} = z_0$):** 
     In de vroege fase vormt de resulterende triglyceride-stapeling (*Triglyceride accumulation*, KE 291) een reversibele, fysiologische buffer. Binnen de fysiologische map vangt het hepatocytennetwerk deze belasting adaptief op door activering van $\beta$-oxidatie (*PPARA* / *CPT1A*) en versnelde VLDL-assemblage en -secretie (*MTTP* / *APOB*). Zolang de perturbatiekracht $p$ binnen deze dynamische capaciteit blijft, keert het cellulaire traject terug naar de homeostatische attractor.

   - **Deterioratief kantelpunt naar de pathologische attractor ($z_{\text{steatose}}$):** 
     Wanneer chronische blootstelling of overmatige chemische activatie de homeostatische buffers overbelast, faalt de adaptieve compensatie:
     1. Excessieve triglyceride-accumulatie (KE 291) induceert intracellulaire verdringing, kernvervorming en mitochondriële dysfunctie (*Mitochondrial dysfunction*, KE 177).
     2. Lipotoxische stress overstijgt de endogene afweer (*NFE2L2*), wat leidt tot activatie van stress- en ontstekingsroutes (*NFKB1*, *TP53*) en Caspase-3-afhankelijke celdood (*CASP3*, apoptose en necrose).
     3. Deze celschade drijft de overgang van eenvoudige steatose (AO 345) naar een irreversibele pathologische staat gekenmerkt door steatohepatitis (NASH) en weefselfibrose via Kupffer- en stellaatcelactivatie.

2. **AOP 27 (BSEP-remming richting Cholestase) & Physiological Maps:** 
   Chemische remming van de canaliculaire galzuurtransporter BSEP (ABCB11, MIE 41) leidt tot pathologische intrahepatische galzuuraccumulatie (Bile accumulation, KE 214). Waar een klassieke AOP dit proces lineair benadert, toont inbedding in bredere Physiological maps (Staumont et al., 2025) dat toxiciteit niet binair optreedt, maar gestuurd wordt door een dynamische balans tussen multi-directionele terugkoppelingen (Noble, 2012; Maertens, 2025) binnen menselijke hepatocyten (HepaRG; Rodrigues et al., 2018).

   - **Adaptieve compensatierespons (Homeostatische attractor):** 
     Als fysiologische tegenreactie induceert de galzoutstapeling transcriptionele activatie van specifieke nucleaire receptoren (Activation of nuclear receptors, KE 288), met name de farnesoid X-receptor (FXR / NR1H4). Dit initieert een gecoördineerd feedbacknetwerk ter bescherming van de homeostatische attractor state ($z_0$):
     1. Transcriptionele repressie van de galzuurnieuwvorming via downregulatie van *CYP7A1*.
     2. Remming van verdere basolaterale opname via *SLC10A1* (NTCP).
     3. Activering van de endogene antioxidant-afweer via de Nrf2-route (*NFE2L2*).
     Zolang de blootstellingscontext (dosis en tijdsduur) binnen de fysiologische bandbreedte blijft, neutraliseert dit netwerk de perturbatiekracht $p$ en keert de cel terug naar homeostase.

   - **Deterioratieve transitie (Kantelpunt naar pathologische attractor):** 
     Wanneer chronische blootstelling of hoge doseringen de adaptieve compensatiecapaciteit overbelasten, kantelt het systeem door niet-lineaire versterking naar een pathologische attractor state ($z_{\text{cholestase}}$):
     1. Cytotoxische galzouten veroorzaken mitochondriële dysfunctie en excessieve reactieve zuurstofspecies (*ROS*, KE 1115), wat resulteert in weefsel- en eiwitschade (*Peptide oxidation*, KE 209).
     2. Galzuurretentie triggert chemotaxis en secretie van ontstekingsmediatoren (*Cytokine release*, KE 87), wat leidt tot versterkte weefselinflammatie (*Inflammation*, KE 149 via NF-κB / *NFKB1*).
     3. Gecombineerde lipotoxische/oxidatieve stress induceert genomische stressrespons via *TP53*, waarna activatie van *CASP3* (caspase-3) onomkeerbare hepatocyten-dood (apoptose en necrose) en functionele cholestatische leverschade (*Cholestasis*, AO 357) forceert.

### Het Waddington Attractor Concept
Een gezonde hepatocyt bevindt zich in een stabiele, homeostatische attractor state ($z_{\text{homeo}} = z_0$). Chemische blootstelling oefent een perturbatiekracht $p$ uit op dit systeem:
- **Adaptieve respons (Homeostase):** Bij een lage dosis of adaptieve prikkel (zoals Fenofibrate via PPARA/CPT1A, VLDL-export via MTTP/APOB, of FXR-gemedieerde repressie via NR1H4) vangt het netwerk de verstoring op en keert het traject $z(t)$ terug naar $z_0$.
- **Adverse Outcome (Pathologische kanteling):** Bij overschrijding van een kritische drempelwaarde passeert de cel een bifurcatiepunt en convergeert $z(t)$ onomkeerbaar naar een pathologische attractor ($z_{\text{steatose}}$ bij overbelaste lipogenese, of $z_{\text{cholestase}}$ bij toxische galzuurretentie).

---

## 2. Systeemarchitectuur & Bestandsnamen

- **Mijlpaal 1:** 
  - *Stap 1.1:* Haal de AOP op
  - *Stap 1.1:* `fetch_real_hepatocyte_data.py` & `m01_data_explorer.py` (Echte gezonde & pathologische data ophalen via MyGene.info, GTEx/HPA en NCBI GEO, AOP WIKI opslaan in `data/hepatocyte_features.pt` en Marimo verkenner starten).
  - *Stap 1.2:* `m01_graph_embedding.py` (Bouwt de HepatocyteGraphormer op basis van de echte data en genereert `01_network_and_embedding.html`).
- **Mijlpaal 2:** `m02_perturbation_encoder.py` (Genereert `02_perturbation_profiles.html`).
- **Mijlpaal 3:** `m03_neural_ode_trajectory.py` (Genereert `03_trajectory_dynamics.html`).
- **Mijlpaal 4:** `m04_attractor_simulation.py` (Genereert `04_attractor_landscape.html`).

---

## 3. Gedetailleerde Specificaties per Mijlpaal

### MIJLPAAL 1: De Biologische Netwerkgraaf, Echte Data & Graphormer

#### Mijlpaal 1, Stap 1.1: Echte Gezonde & Pathologische Data-acquisitie, AOP-annotatie & Marimo Verkenner
* **Doel:** Echte humane data ophalen uit gevalideerde openbare databases voor zowel de gezonde baseline als pathologische (steatose/NAFLD) toestanden, gekoppeld aan officiële OECD AOP-literatuur.
* **Exacte Openbare Databronnen:**
  1. **Gezonde Baseline:** 
     - **GTEx Portal & Human Protein Atlas (HPA):** Normale basale leverweefsel expressie (TPM) voor de 18 genen.
     - **MyGene.info / Ensembl API:** Genstructuur, genomische lengte, exonen en eiwitlengte.
  2. **Pathologische Toestanden (Steatose / NAFLD & Toxisch):** 
     - **NCBI Gene Expression Omnibus (GEO):** Humane levertranscriptoomdatasets van NAFLD / steatose patiënten (bijv. referentie-series zoals GSE126848 of vergelijkbare humane steatose/DILI datasets) om de opregulatie van lipogene genen ($SREBF1$, $FASN$, $CD36$) vs. homeostatische compensatie empirisch te verankeren in de feature matrix.
  3. **Eiwit-Interacties:** 
     - **STRING-db API:** Experimenteel gevalideerde interactiescores (`combined_score`) voor de 20 gerichte AOP-randen.
* **Exact 18 genen (vaste volgorde, index 0 t/m 17) met formele annotaties:**
  * *Groep 1 - AOP 34 (LXR-activatie & Steatose):* 
    `0: NR1H3` (Liver X Receptor alpha - Initiator AOP 34), 
    `1: SREBF1` (SREBP-1c - Upstream lipogene transcriptiefactor), 
    `2: FASN` (Fatty acid synthase - De novo lipogenese), 
    `3: ACACA` (Acetyl-CoA carboxylase - Snelheidsbepalend enzym), 
    `4: SCD` (Stearoyl-CoA desaturase - Vetzuur desaturatie), 
    `5: CD36` (Fatty acid translocase - Leveropname van vetzuren).
  * *Groep 2 - Homeostatische Compensatie & Oxidatie:* 
    `6: PPARA` (PPAR-alpha - $\beta$-oxidatie), 
    `7: CPT1A` (Carnitine palmitoyltransferase 1A - Mitochondriële import), 
    `8: MTTP` (Microsomal triglyceride transfer protein - VLDL assemblage), 
    `9: APOB` (Apolipoprotein B - VLDL/LDL scaffold).
  * *Groep 3 - AOP 27 (BSEP-remming & Cholestase):* 
    `10: NR1H4` (FXR - Galzuurregulator), 
    `11: ABCB11` (BSEP - Moleculair doelwit AOP 27), 
    `12: CYP7A1` (Cholesterol 7 alpha-hydroxylase - Galzuursynthese), 
    `13: SLC10A1` (NTCP - Basolaterale galzuuruptake).
  * *Groep 4 - Cellulaire Stress, Ontsteking & Apoptose:* 
    `14: NFE2L2` (Nrf2 - Oxidatieve stress sensor), 
    `15: NFKB1` (NF-$\kappa$B - Ontsteking), 
    `16: TP53` (p53 - Genoomwachter/stress), 
    `17: CASP3` (Caspase 3 - Apoptose / Adverse Outcome).
* **Gerichte interacties met formele AOP-literatuurverwijzingen:**
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
* **Uitvoeribles Stap 1.1:** `fetch_real_hepatocyte_data.py` haalt gezonde (GTEx/HPA) en pathologische (GEO NAFLD) expressieprofielen op, berekent de `[19, 19]` afstandsmatrix en slaat alles op in `data/hepatocyte_features.pt` en `data/hepatocyte_baseline_data.json`. `m01_data_explorer.py` start het Marimo notebook met Plotly visualisaties om de gezonde vs pathologische profielen te vergelijken.

#### Mijlpaal 1, Stap 1.2: Graphormer Encoder (`m01_graph_embedding.py`)
* **Invoer:** Laadt `data/hepatocyte_features.pt` uit Stap 1.1.
* **Architectuur (`HepatocyteGraphormer`):** Input projectie (`8 -> 64`), Centrality embeddings, `[CLS]` token prependen, Spatial attention bias op basis van de `[19, 19]` Floyd-Warshall afstandsmatrix, 2 Graphormer transformer lagen, en een LayerNorm+Tanh readout naar $z_0 \in \mathbb{R}^{[1, 64]}$.
* **Uitvoer:** Genereert standalone Plotly dashboard `01_network_and_embedding.html`.

---

### MIJLPAAL 2: De Chemische Perturbatie-encoder (`m02_perturbation_encoder.py`)
### MIJLPAAL 3: Het Dynamische Vectorveld & Neural ODE (`m03_neural_ode_trajectory.py`)
### MIJLPAAL 4: Attractor-analyse & Kantelpunt Simulatie (`m04_attractor_simulation.py`)

---

## 4. Harde Uitvoeringsregels
1. GPU Isolatie: Altijd via `CUDA_VISIBLE_DEVICES=1 uv run python <script>.py`.
2. Geen Matplotlib: Uitsluitend Plotly.
3. Determinisme: `torch.manual_seed(42)` en `np.random.seed(42)`.
4. Verplichte Logging: Maak voor elke taak eerst een logbestand aan in `logs/` volgens het sjabloon in `.clinerules`.
