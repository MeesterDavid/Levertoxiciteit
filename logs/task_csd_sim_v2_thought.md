# Thought & Progress Log: CSD simulation v2 (same MIE, controlled spread/no-spread)
- **Date / Time:** 2026-11-06 10:40 (Europe/Amsterdam)
- **Task Goal:** Deel B van slide 1 ("Niet-lineair: dezelfde MIE, hoger abstractieniveau", `slides/slides.md`
  regels ~98-118) wordt onderbouwd door `public/csd_wave.mp4` + `public/csd_recovery.mp4`, gegenereerd door
  `csd_sim.py`. De vraag is of die simulatie "dezelfde MIE, wel/geen spread" eigenlijk wel waar maakt, en zo niet:
  lever een `csd_sim_v2.py` die dat wél gecontroleerd doet.

## 1. Analysis & Problem Statement
- **Exacte vraag:** representeert `csd_sim.py` (v1) "critical spreading" trouw? Beginnen de twee runs echt bij
  dezelfde MIE (Molecular Initiating Event)?
- **Interpretatie:** In v1 is de MIE de beginschijf `u=u0=0.95` in het midden van een 100x100 weefsel.
  "Zelfde MIE" zou moeten betekenen: identieke moleculaire gebeurtenis, identieke beginconditie, en de twee
  runs mogen alleen verschillen in wat de biologie laat divergeren (dosis/duur/capaciteit). Daarna moet de
  uitkomst ook echt verschillen (herstel vs. verspreiding) en dat verschil moet *kritisch* zijn (PoD).

- **Gemeten diagnose van v1** (read-only diagnostiek in `/tmp/diag_csd_v1.py`, geen wijziging aan de repo):

  | aspect | recovery-run | wave-run | conclusie |
  |---|---|---|---|
  | beginstraal MIE | r=3 (28 cellen) | r=6 (113 cellen) | **4x verschillende MIE-omvang** |
  | substeps x dt | 2 x 0.02 = 0.04 t.u./frame | 5 x 0.04 = 0.20 t.u./frame | **5x verschillende tijdas** |
  | gesimuleerde duur | 9.6 t.u. | 48.0 t.u. | video's lijken even lang (8 s) maar zijn dat niet |
  | `np.clip(u,v,0,1.1)` | aan | aan | kapt 45% van de remming af (v_ss = u/gamma = 2.0) |
  | randvoorwaarde | `np.roll` = periodiek (torus) | idem | golf kan om het weefsel heen wrappen |

  Kernbevinding: **de "golf"-run propageert helemaal niet.** Na de 48 t.u. van de video geldt
  `mean(u)=0.029`, `max(u)=0.21` (geen enkele cel bereikt de bovenste toestand), en bij een langere horizon
  (`T=120`) is het veld identiek nul. Frames uit de bestaande mp4 bevestigen dit visueel: frame 0 = felle
  schijf, frame 120 = grotere maar *dimmere* blob, frame 239 = een **holle paarse ring** die uitdooft.
  Oorzaak is structureel: met `gamma=0.5` is er **geen enkele niet-triviale fixed point**, want bistabiliteit
  vereist `K < ((1-a)/2)^2 = 0.1936` terwijl v1 effectief `K = 1/gamma = 2.0` gebruikt. Er is dus geen
  pathologische attractor om naar te verspreiden: de "spread" is een diffusie-transiënt, geen tipping point.
  De titel "kritisch" wordt daarmee niet door de simulatie gedragen (geen r_c, geen PoD, geen CSD-maat).

- **Pitfalls / boundary conditions voor v2:**
  1. "Zelfde MIE" moet *bewijsbaar* zijn (hash van de beginconditie + MIE-patroon + parameters), niet alleen beweerd.
  2. De tijdas, dt en substeps moeten voor alle runs identiek zijn, anders vergelijk je appels met peren.
  3. Er moet échte bistabiliteit zijn (`K < 0.1936`) zodat de bovenste toestand een attractor is.
  4. Geen `np.clip` op de fysische variabelen (alleen een harde assert tegen niet-eindige waarden).
  5. Randvoorwaarde: geen flux (Neumann, `np.pad(mode="edge")`) i.p.v. periodiek.
  6. De PoD moet *gemeten* worden (bisectie) en de CSD-maat moet bij de PoD divergeren, anders is "kritisch"
     een adjectief zonder inhoud.
  7. `.clinerules`: **geen matplotlib**. Video's dus via een raw RGB-pipe naar `ffmpeg`, dashboard via Plotly.

## 2. Methodological & Scientific Considerations
- **Model (FitzHugh-Nagumo-klasse, behoudt de vergelijkingen van v1):**
  `du/dt = D*lap(u) + u(1-u)(u-a) - v + I0*s(x)*[t<duration]`, `dv/dt = eps*(u - v/K)`.
  - `u` = lokale lipogene/LXR-gedreven activatietoestand (0 = homeostase, ~0.77 = steatose-attractor),
  - `v` = traag accumulerende homeostatische belasting (tau = 1/eps),
  - `K` = adaptieve klaringscapaciteit (PPARa/CPT1A beta-oxidatie + MTTP/APOB VLDL-export),
  - `a` = autocatalytische drempel van het lipogene circuit (SREBP-1c/FASN-zelfversterking),
  - `s(x)` = vast ruimtelijk MIE-patroon (LXR-activatie in dezelfde cellen, Gaussisch, sigma=4).
  v1-parameters blijven staan (D=2.5, a=0.12, eps=0.008); alleen `K` gaat van 2.0 (monostabiel) naar 0.15
  (bistabiel: stabiele takken u=0 en u≈0.769, drempel u≈0.351).
- **Waarom deze opzet:** de MIE is in beide condities *bit-identiek* (zelfde `u0(x)`, zelfde `s(x)`, zelfde
  `D, a, eps, K, dt`); het enige verschil is de blootstelling `I0` (dosis) of `duration` (duur). Dat is precies
  de AOP-vraag "dezelfde MIE, andere dosis/duur" en precies wat een puur moleculair model (Graphormer/
  ToxTransformer, Milestone 0) niet kan voorspellen (ARCHITECTURE.md, Step C).
- **CSD-maat:** voor sub-kritische blootstelling past `mean(u)` een exponentiële vervalcurve, dus
  `lambda = -d/dt log mean(u)`; de hersteltijd `T_half = ln2/lambda` divergeert als de dosis de PoD nadert
  (kritische vertraging). Boven de PoD divergeren de ignitietijd en de golf-aankomsttijd. Beide kanten
  meten *is* de CSD-demonstratie.
- **Waarom bisectie voor de PoD:** de PoD is een bifurcatiepunt; met een 8-staps bisectie op `I0`
  (fate-criterium `mean(u)>0.5` op de observatiehorizon) is hij tot ~2% nauwkeurig. De horizon-afhankelijkheid
  van de PoD is géén artefact maar een eigenschap van CSD; dat wordt expliciet gerapporteerd.

## 3. Step-by-Step Execution Plan
- **Nieuw bestand:** `csd_sim_v2.py` (repo-root, naast `csd_sim.py`; v1 blijft ongemoeid).
  Structuur: (0) constanten + magma-LUT uit de Plotly-colorscale, (1) Neumann-Laplaciaan + integrator met
  vaste `dt=0.01`, (2) MIE-fingerprint (sha256), (3) Exp A: PoD-bisectie op dosis, (4) Exp B: dosis-ladder
  met CSD-metrics (lambda, T_half, t_ignite, lag-1 autocorrelatie), (5) Exp C: duur-ladder (acuut vs.
  chronisch), (6) Exp D: capaciteits-as (K = 0.10 / 0.15 / 0.19), (7) Exp E: kritische nucleus (r-sweep,
  r_c), (8) Exp F: de twee video-runs met identieke tijdas + `ffmpeg`-raw-pipe, (9) Plotly-dashboard
  `public/02_csd_spreading.html`, (10) `data/csd_sim_v2_results.json`.
- **Outputs:** `public/02_csd_spreading.html`, `public/csd_recovery_v2.mp4`, `public/csd_wave_v2.mp4`,
  `public/csd_pair_v2.mp4`, `data/csd_sim_v2_results.json`.
- **Verificatie:** `CUDA_VISIBLE_DEVICES=1 uv run python csd_sim_v2.py`, daarna `ls -la public/ data/`,
  `ffprobe` op de mp4's, controle op NaN/oneindig en op `hash_a == hash_b`.

## 4. Terminal Execution & Verification (GPU 1)
- **Commando:**
  ```bash
  cd /home/disaacspat/data/demon_storage_box/levertoxiciteit
  CUDA_VISIBLE_DEVICES=1 uv run python csd_sim_v2.py
  ```
  (achtergrond met logging: `nohup sh -c 'CUDA_VISIBLE_DEVICES=1 uv run python -u csd_sim_v2.py > /tmp/csd_v2_run.log 2>&1; echo EXIT=$?'`)
- **Exitcode:** `EXIT=0`, runtijd **105.1 s** (raster 100x100, dt=0.01, ~330 integratieruns).
  Een tweede volledige run ter bevestiging (na de laatste tekstcorrectie in de dashboard-caption):
  `EXIT=0`, **105.7 s**, identieke getallen (PoD 0.05564, r_c 4.25, 33/72 steatotisch) -> het script is
  deterministisch en reproduceerbaar; alleen de bestandstijdstempels verschillen.
- **Kernoutput (letterlijk uit het log):**
  ```text
  MIE-fingerprint (identiek voor elke run): d92c347254aec592
  state-hash gezonde start (alle cellen u=0): f8c784aa6b57396e
  Bistabiliteit: K=0.15 < K_crit=0.1936 -> True (drempel u~0.3512, attractor u~0.7688)

  [A] PoD(K=0.15, duration=30 t.u., T=150) = 0.05564 (herstel @ 0.05486 -> -0.000, spread @ 0.05641 -> 0.799)
  [B] dosis-ladder (10 doses): 0.60..1.00 x PoD -> recovery, 1.05..1.60 x PoD -> spread (mean(u) 0.782 -> 0.771)
      piek-amplitude: 0.00258 0.00333 0.00438 0.00659 0.00964 0.01846 | 0.81859 (sprong bij de PoD)
      t_transient    : 35.9   36.2    36.8    38.8    42.0    59.1    | -   (divergentie onder de PoD)
      t_ignite       :  -      -       -       -       -       -      | 76.1 70.6 61.7 56.1
  [C] duur-ladder @ 1.25 x PoD: 2/5/10 t.u. -> recovery, 20/30/60/120 t.u. -> spread (mean(u) 0.775 -> 0.772)
  [D] landschap 8 doses x 9 K-waarden (geen bisectie):
      K=0.0900: +0.000 +0.000 +0.884 +0.882 +0.882 +0.882 +0.882 +0.882
      K=0.1500: -0.000 -0.000 -0.000 +0.782 +0.774 +0.771 +0.770 +0.769
      K=0.1850: -0.000 -0.000 -0.000 +0.679 +0.236 +0.023 -0.065 -0.100
      K=0.1925: -0.000 -0.000 -0.000 +0.000 -0.023 -0.089 -0.098 -0.082
  [E] kritische nucleus: r<=4.0 -> recovery (c_front=0.000), r>=4.5 -> spread -> r_c = 4.25 px
      c_front: 0.395 (r=4.5) 0.488 0.524 0.542 0.558 0.562 0.573 0.578 0.585 0.590 (r=9.0)
  [F] video-paar: dosis 0.04729 (0.85 x PoD) vs 0.16691 (3.0 x PoD); T=144 t.u., 240 frames @ 30 fps
      MIE identiek: state_hash=f8c784aa6b57396e fingerprint=d92c347254aec592 (alleen de dosis verschilt)
      dosis-laag -> recovery mean(u)=-0.000 | dosis-hoog -> spread mean(u)=0.770
  Landschap dosis x K: 33/72 steatotisch, 9/72 adaptief onderdrukt (u<-0.02)
  ```
- **Numerieke hygiene:** `_assert_json_finite` over de hele payload (exclusief de video-animatie) slaagt;
  in `data/csd_sim_v2_results.json` staat geen `NaN`/`Infinity` (regex-check: `False`). De integrator
  raise't `FloatingPointError` bij niet-eindige toestand — niet getriggerd.
- **`ffprobe` op de video's** (alle drie 240 frames, 30 fps, 8.0 s, h264):
  | bestand | resolutie | frames | duur | bytes |
  |---|---|---|---|---|
  | `public/csd_recovery_v2.mp4` | 400x400 | 240 | 8.000 | 16 595 |
  | `public/csd_wave_v2.mp4` | 400x400 | 240 | 8.000 | 194 784 |
  | `public/csd_pair_v2.mp4` | 800x396 | 240 | 8.000 | 155 296 |
  De 12x grotere `wave`-file is zelf een aanwijzing dat er echt iets gebeurt: een uitdovende diffusie
  comprimeert (zoals v1: 92 kB voor een "golf" die dooft) veel beter dan een groeiend front.


## 5. Verification of Completed Tasks
| # | Bestand | Aanwezig | Bewijs |
|---|---|---|---|
| 1 | `csd_sim_v2.py` | ✅ | `py_compile` OK; `EXIT=0`; 105.1 s |
| 2 | `public/02_csd_spreading.html` | ✅ | 725 732 bytes, 6 panelen; bevat `heatmap`, "Niet-lineaire respons en nasleep", "klaringscapaciteit", "saddle-node K_crit = 0.1936" |
| 3 | `data/csd_sim_v2_results.json` | ✅ | 43 388 bytes; keys: `bistability, capacity_map, critical_nucleus_radius, csd_note, dose_ladder, duration_ladder, mie_fingerprint, nucleus, parameters, pod, runtime_s, state_hash, videos` |
| 4 | `public/csd_recovery_v2.mp4` | ✅ | 400x400, 240 frames, 8.0 s |
| 5 | `public/csd_wave_v2.mp4` | ✅ | 400x400, 240 frames, 8.0 s |
| 6 | `public/csd_pair_v2.mp4` | ✅ | 800x396, 240 frames, 8.0 s |
| 7 | `slides/slides.md` | ✅ | video-sectie verwijst nu naar `/csd_wave_v2.mp4` + `/csd_recovery_v2.mp4` met de "zelfde MIE, andere dosis"-caption |
| 8 | dit logboek | ✅ | secties 4-6 gevuld met de werkelijke output |

- **Asserts in `main()`** (allemaal gehaald): model bistabiel; PoD gevonden; dosis-ladder bevat beide
  fates; video-laag = recovery en video-hoog = spread; `same_initial_state == True`.
- **Zelfcontrole op de claim "zelfde MIE":** de beginconditie (`np.zeros`) en het MIE-patroon `SRC_MIE`
  zijn módulgrootheden; de sha256-fingerprint wordt één keer bij import berekend en in élke run
  gerapporteerd. De runs verschillen uitsluitend in `dose` (en in Exp C/D/E in `duration` resp. `K`).
- **Wat niet klopte en is gerepareerd tijdens de run:** (a) `KeyError: 'animation'` omdat de video-payload
  voor de JSON werd gestript terwijl `fig_animation` hem nodig heeft -> payload houdt de animatie,
  alleen de JSON-dump laat hem weg; (b) het eerste CSD-paneel plotte `T_half`/`lambda`, die in dit model
  *niet* divergeren -> paneel vervangen door de amplitude- en nasleepdivergentie (zie §6); (c) de
  capaciteits-as met bisectie gaf `PoD=None` bij K=0.19 (niet-monotone respons) -> vervangen door een
  volledige (dosis x K)-landschapskaart.

## 6. Reflection & Next Step
- **Is de oorspronkelijke vraag beantwoord?** Ja. De video-paar toont nu aantoonbaar dezelfde MIE
  (identieke hash + fingerprint) op precies dezelfde klok (dt=0.01, 240 frames, 0.6 t.u./frame) met
  alleen de dosis als verschil: 0.85x PoD dooft uit (mean(u) = -0.000), 3.0x PoD kiept het weefsel
  (mean(u) = 0.770, front groeit tot r = 60 px). De kwantitatieve onderbouwing: PoD = 0.05564,
  r_c = 4.25 px, duurdrempel tussen 10 en 20 t.u. bij 1.25x PoD.
- **Eerlijkheid over "kritische vertraging":** de snelle-fase vervalsnelheid `lambda` divergeert hier
  niet (0.0678 -> 0.0797: zelfs licht stijgend). Wat wél divergeert is (a) de piek-amplitude
  (0.0026 -> 0.0185, factor 7, dan een sprong naar 0.819), (b) de nasleep `t_transient`
  (35.9 -> 59.1 t.u.), (c) de ignitietijd boven de PoD (76.1 -> 56.1 t.u.) en (d) de frontsnelheid
  naar `r_c` toe (0.590 -> 0.395 px/t.u. en 0 bij r_c). Dat staat zo in het dashboard en in `csd_note`.
- **Nieuw inzicht uit Exp D:** de dosis-respons is niet-monotoon vlak bij de saddle-node
  (K >= 0.185): een zeer hoge dosis duwt de adaptieve last `v` zo ver op dat het weefsel doorschiet
  naar `u < 0` (adaptieve suppressie) en dus alsnog herstelt. Klinisch relevant: bij een lever met
  bijna-uitgeputte klaringscapaciteit is "meer blootstelling" niet automatisch "erger".
- **Beperkingen:** (1) één enkele MIE-geometrie (Gaussisch, sigma = 4 px); (2) `u`/`v` zijn
  dimensieloze AOP-toestandsvariabelen, geen gemeten TPM's — de koppeling met Milestone 1.2/2/3
  (`z0`, `p`, `z_steatose`) moet nog worden gelegd; (3) de PoD is horizon-afhankelijk (T = 150 t.u.);
  (4) Neumann-randvoorwaarden maken het weefsel eindig, dus "spread" betekent "vult het hele raster".
- **Vervolgstap:** dezelfde bifurcatie-analyse op de *geleerde* 64-dimensionale `z`-ruimte van
  Milestone 1.2/3 herhalen (S-curve in `z`-ruimte, `r_c` in weefselcoördinaten) zodat de CSD-demo en de
  Graphormer/Neural-ODE-pipeline één verhaal vertellen; daarna de slide-tekst van deel B voorzien van
  de exacte getallen (PoD 0.0556, r_c 4.25) en de v1-bestanden uit `public/` verwijderen.

