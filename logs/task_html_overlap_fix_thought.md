# Thought & Progress Log: HTML-dashboard overlap fix (public/*.html)

- **Date / Time:** 2025-11-05 17:5x (UTC+1)
- **Task Goal:** Repareren van overlappende content in de gegenereerde standalone HTML-dashboards in
  `public/` (tekst over tekst, molecuulgraph over de titel, etc.). Dit raakt de deliverables van
  Milestone 0 (`public/00_molecular_baseline_mie.html`, `public/00_architecture_pipelines.html`),
  Milestone 1.1 (`public/01_1_data_explorer.html`) en de losse slide-1-preview
  (`public/slide1_pitch_preview.html`).

## 1. Analysis & Problem Statement
- **Letterlijke melding van de gebruiker:** "some of the stuff in the html pages is overlapping.
  sometimes text is on top of other text, or the molecule graph is on top of the title etc".
- **Interpretatie:** de HTML-pagina's in `public/` zijn handmatig opgebouwde SVG-layouts met hardgecodeerde
  coördinaten. Overlap ontstaat daar door (a) tekst die niet meegroeit met zijn box, (b) inhoud die in een
  gebied wordt getekend dat al door een titel/label wordt gebruikt, (c) `position:absolute`-overlays in de
  oude hub/preview-pagina, (d) geëxporteerde Marimo/Plotly-labels die over elkaar heen vallen.
- **Boundary conditions / valkuilen:**
  - Er is **geen headless browser** (playwright/pyppeteer/kaleido ontbreken; `/usr/bin/firefox` is de snap
    die door de lopende gebruikersinstantie geblokkeerd is). Visueel screenshot-verificatie is dus niet
    mogelijk -> ik moet de overlap *geometrisch* aantonen en na de fix opnieuw meten.
  - De SVG's komen uit generators (`m00_molecular_baseline_mie.py`, `generate_pipeline_diagrams.py`,
    `generate_slide1_assets.py`) of uit Marimo (`m01_data_explorer.py` -> `marimo export html`).
    Alleen de HTML patchen is niet genoeg: de bron moet mee, anders komt de bug terug.
  - `m00_molecular_baseline_mie.py` laadt twee HF-modellen op GPU 1; voor een **layout-only** re-render
    mag dat niet nodig zijn. `data/molecular_baseline_mie.json` bevat alle render-inputs
    (`compounds`, `pca_coords_2d`, `toxtransformer_pca_coords_2d`, `cosine_similarity_matrix`), dus de
    HTML kan uit de cache opnieuw worden opgebouwd via `generate_standalone_html(...)`.

## 2. Methodological & Scientific Considerations
- **Metrologie boven gokken:** in plaats van "op het oog" hopen dat het goed is, schrijf ik een
  statische auditor (`tools/check_html_overlaps.py`) die elke inline `<svg>` parseert, bounding boxes
  reconstrueert (inclusief `transform="translate(...)"` / `scale(...)` op `<g>`) en drie klassen
  fouten rapporteert:
  1. **text↔text** botsingen (letterlijke "tekst over tekst"),
  2. **shape↔text** waarbij de shape *na* de tekst is getekend (opaque rect/circle dekt het label af),
  3. **buiten de viewBox** (afgekapte content).
  Tekstbreedte wordt geschat met een Helvetica/Inter-achtige advance-width tabel (±5%), wat ruim
  voldoende is om echte botsingen van ruis te onderscheiden (drempel: >40% overlap in *beide* assen).
- **Consistentie met het onderzoeksvoorstel:** de dashboards zijn de visualisatie-laag van de AOP/NAMS-
  pipeline (Milestone 0/1.1). Een onleesbaar diagram ondermijnt de communicatie van de MIE-baseline en de
  AOP-netwerken; de fix is dus puur presentationeel (geen data- of tensorwijziging). Tensor-contracten
  (`X [18,8]`, `z0 [B,64]`, `p [B,33]`, `trajectories [T,B,64]`) blijven onaangeroerd.
- **Plotly-only regel:** de auditor en alle fixes gebruiken geen matplotlib; de HTML's blijven
  standalone Plotly/SVG-bestanden in `public/`.

## 3. Step-by-Step Execution Plan
1. `tools/check_html_overlaps.py` schrijven (pure stdlib; `xml.etree` + regex).
2. Draaien op alle vier de HTML-bestanden -> objectieve lijst met botsingen (`/tmp/overlaps_*.log`).
3. Fixes in de **generators**, niet in de artifacts:
   - `m00_molecular_baseline_mie.py`: molecuul-canvas (`pad_t/pad_b`) vrijhouden van de
     "Model 1 — Microsoft Graphormer…"-kop; idem voor de SELFIES-strip en de MIE-balken.
   - `generate_pipeline_diagrams.py`: box-/tekstafmetingen zodat labels binnen hun kaders blijven.
   - `generate_slide1_assets.py`: `create_standalone_slide1_preview()` herbouwen naar de nieuwe
     tweedelige slide-1-layout (geen `position:absolute`-hub meer).
   - `m01_data_explorer.py` + `marimo export html` als de AOP-netwerk-labels overlappen.
4. Opnieuw genereren (m00 via de JSON-cache, dus **zonder** model-load op GPU 1) en de auditor herhalen
   tot 0 botsingen.
5. Secties 4-6 van dit log vullen met de echte terminal-output.

## 4. Terminal Execution & Verification (GPU 1)

### 4.1 Auditor gebouwd en gekalibreerd (`tools/check_html_overlaps.py`)

De auditor reconstrueert per `<svg>` alle tekst-/vorm-boxen. Twee iteraties waren nodig om hem
*waarheidgetrouw* te maken:

- **`rotate(a cx cy)`-support** toegevoegd (`_rotation_of` + `_rotate_bbox`). Zonder dit werden
  geroteerde labels (heatmap-kolomkoppen, `PC 2`-aslabels) als horizontale rechthoeken gemeten ->
  valse "TEXT-OUTSIDE"-meldingen.
- **Marimo-payload-decoder** toegevoegd (`_marimo_embedded_html`). `marimo export html` (v0.25.2)
  schrijft `mo.Html`-outputs als JSON-strings met `\u003C`-escapes in `__MARIMO_MOUNT_CONFIG__`,
  dus een naïeve `<svg`-regex vindt **0** SVG's in de nieuwe export. Zonder deze decoder zou
  `01_1_data_explorer.html` "schoon" lijken terwijl er niets gemeten was.

Uitgangssituatie (voor de fixes):

```bash
CUDA_VISIBLE_DEVICES=1 uv run python tools/check_html_overlaps.py
# === 00_architecture_pipelines.html === 2 SVG('s), 3 bevinding(en)
# === 00_molecular_baseline_mie.html   === 16 SVG('s), 30 bevinding(en)
# === 01_1_data_explorer.html          === 11 SVG('s), 20 bevinding(en)
# === slide1_pitch_preview.html        === 1 SVG('s), 0 bevinding(en)
# TOTAAL: 53 bevinding(en)   (8x TEXT-COVERED, 12x TEXT-OUTSIDE, 33x TEXT-TEXT)
```

Representatieve echte fouten die dit rapporteerde:

```
SVG #1  TEXT-TEXT     "Model 1 - Microsoft Graphormer (2D Graaf) - [CLS] node 0" [22,71 -> 244,83]
                      botsen met "C" [176,117 -> 184,127]            <- molecuulgraph over de titel
SVG #1  TEXT-OUTSIDE  "Both Graphormer and ToxTransformer see: ... 0.1 uM, PPARA/CPT1A still ..."
                      [693,503 -> 1193,516] valt buiten viewBox 1140x470   <- afgekapte zin
SVG #14 TEXT-COVERED  "Tamoxifen" [198,361 -> 247,371] wordt bedekt door <circle r=8>
PIPE    TEXT-TEXT     "1D. Farmacofoor & Fysicochemische Regels" botsen met "SMARTS + RDKit"
M01     TEXT-OUTSIDE  "Repression / Injury (-1 arrow)" [1113,597 -> 1262,609] buiten viewBox 1160
M01     TEXT-TEXT     "Panel A: [19 x 19] Floyd-Warshall ..." botsen met "03:FASN" ... "14:SLC10A1"
M01     TEXT-TEXT     "1" [818,578 -> 824,588] botsen met "In-Degree (z_deg-)" [777,582 -> 879,593]
```

### 4.2 Uitgevoerde fixes (allemaal in de **generators**, niet in de artifacts)

| Bestand | Fix |
|---|---|
| `m00_molecular_baseline_mie.py` | `pad_t` 82 -> 102 / `pad_b` 378 -> 386: molecuultekening blijft onder de kop (71-83). Atoomstraal + elementsymbool geklemd op de kortste binding (`r = min(10.5+7.0*sal, 0.34*min_bond_px)`, `sym_fs = min(10.5, 0.95*min_bond_px)`) zodat atomen hun buurlabels niet meer afdekken. Nieuwe helpers `svg_text_px()` / `wrap_svg_text()` / `_boxes_overlap()`; de afgekapte "Why Milestones 1-4"-noot wordt nu pixel-nauwkeurig gewrapt i.p.v. `note[:92]` / `note[92:188]`. |
| `m00_molecular_baseline_mie.py` | Embedding-space: markers en labels in twee passes (labels altijd bovenop), labels krijgen een collision-aware offset (-11.5 / +17 / -24 / +27.5 / -37 / +40), heatmap-kolomkoppen van -42deg/9.5px naar -60deg/8.6px met kortere namen. |
| `generate_pipeline_diagrams.py` | Kolombreedtes 360/410/560/360 -> 380/420/590/420 (`svg_w` 1860 -> 1960) en een width-aware kopregel: de badge wordt nu gemeten met `_text_px()`, de kop krimpt (max 12.2px, min 9.0px) en wordt pas daarna ge-ellipseerd -> geen kop meer onder de badge. Badge-font 10.2 -> 9.4 zodat alle koppen volledig passen. |
| `m01_data_explorer.py` | Legenda's van netwerk- en expressieplot vloeien nu naar een tweede regel (`item_w`-meting) i.p.v. buiten de viewBox te lopen. `[19x19]`-paneel: `H` 600 -> 640, `my0/by0` 95 -> 128, `mh/bh` -> 460 (titels vrij van de -58deg gekantelde kolomkoppen) en de degree-ticklabels (604) staan nu los van de In/Out-Degree-legenda (631). |
| `tools/rerender_m00_html.py` (nieuw) | Layout-only re-render van `public/00_molecular_baseline_mie.html` uit `data/molecular_baseline_mie.json` -> **geen** HF-model-load/calibratie nodig; de wetenschap (embeddings, MIE-kansen, LOOCV, PCA) komt byte-voor-byte uit de cache. |

### 4.3 Hergeneratie + verificatie

```bash
CUDA_VISIBLE_DEVICES=1 uv run python tools/rerender_m00_html.py
# [cache] 12 compounds | PC shapes (12, 2), (12, 2) | cos (12, 12)
# - public/pipeline_microsoft_graphormer.svg (23,1 kB) - public/pipeline_toxtransformer.svg (21,8 kB)
# - public/00_architecture_pipelines.html (58,6 kB)
# [ok] re-rendered public/00_molecular_baseline_mie.html (491,643 bytes)

CUDA_VISIBLE_DEVICES=1 uv run marimo check m01_data_explorer.py        # CHECK_EXIT=0
CUDA_VISIBLE_DEVICES=1 uv run marimo export html m01_data_explorer.py -o public/01_1_data_explorer.html   # EXIT=0

CUDA_VISIBLE_DEVICES=1 uv run python tools/check_html_overlaps.py --out /tmp/overlaps_final.md
# === 00_architecture_pipelines.html === 2 SVG('s), 0 bevinding(en)
# === 00_molecular_baseline_mie.html   === 16 SVG('s), 0 bevinding(en)
# === 01_1_data_explorer.html          === 11 SVG('s), 0 bevinding(en)
# === slide1_pitch_preview.html        === 1 SVG('s), 0 bevinding(en)
# TOTAAL: 0 bevinding(en)
```

Regressietest van de auditor zelf (oude, gecommitte versie van de Marimo-export):

```bash
git show HEAD:public/01_1_data_explorer.html > /tmp/old_explorer.html
CUDA_VISIBLE_DEVICES=1 uv run python tools/check_html_overlaps.py /tmp/old_explorer.html
# === old_explorer.html === 11 SVG('s), 20 bevinding(en)   <- detector werkt, 20 -> 0 na de fix
```

Extra metingen (geen `NaN`/lege output, alleen layout):

```
# /tmp/measure_notes.py: alle gewrapte notitie-regels met de AFM-tabel gemeten
note lines measured: 32                     (12 compound-kaarten)
right-most pixel of any note line: 1042.3   (paneelrand = 1105, viewBox = 1140)  -> OK

## 5. Verification of Completed Tasks

- [x] `tools/check_html_overlaps.py` bestaat, draait en is uitgebreid met rotate- en
      Marimo-payload-support (startte op 53 meldingen, eindigt op 0).
- [x] `tools/rerender_m00_html.py` bestaat en schrijft `public/00_molecular_baseline_mie.html`
      (491.643 bytes) + beide pipeline-SVG's + `public/00_architecture_pipelines.html`.
- [x] `m00_molecular_baseline_mie.py` gewijzigd: `pad_t`/`pad_b`, atoom-radius-clamp,
      `svg_text_px` / `wrap_svg_text` / `_boxes_overlap`, twee-pass PCA-labels, heatmap-rotatie.
- [x] `generate_pipeline_diagrams.py` gewijzigd: `_text_px`, kolombreedtes, width-aware kopregel,
      badge-font 9.4.
- [x] `m01_data_explorer.py` gewijzigd: twee legenda-rijen (netwerk + expressie), `[19x19]`-paneel
      `H=640` met verschoven matrix-/centrality-panelen. `marimo check` = exit 0, export = exit 0,
      geexporteerde HTML bevat 11 SVG's (via de JSON-payload) met 0 bevindingen.
- [x] `public/slide1_pitch_preview.html`: geen fix nodig, wel gecontroleerd. `.slide-frame` is een
      vaste 1120x630 canvas; de absoluut gepositioneerde kaarten (`spoke-aop` 14-219 px,
      `center-csd-card` 262-410 px, `spoke-rd` 453-658 px) houden >= 40 px marge. De auditor
      rapporteert 0 bevindingen (1 SVG = pijlenlaag zonder tekst).
- [x] Alle vier de `public/*.html`-bestanden zijn opnieuw gegenereerd (mtime nieuwer dan de
      codewijzigingen) en meten 0 bevindingen.

## 6. Reflection & Next Step

- **Werkt het?** Ja. De drie gemelde symptomen zijn structureel in de bron opgelost:
  (1) molecuulgraph over de titel -> canvas-padding, (2) tekst over tekst -> gemeten
  legenda-/labelplaatsing, wrapping en collision-aware offsets, (3) afgekapte tekst ->
  `wrap_svg_text` i.p.v. vaste character-slices. De auditor is de garantie: 53 -> 0.
- **Beperking:** er is geen headless browser in deze omgeving (playwright/kaleido ontbreken,
  de firefox-snap is door de lopende gebruikersinstantie geblokkeerd), dus verificatie is
  geometrisch i.p.v. visueel. De AFM-tabel schat breedtes op ±5%; de gebruikte drempel
  (35% overlap in beide assen) filtert kerning-ruis.
- **Follow-up (aanbevolen):** (a) `tools/check_html_overlaps.py` in de workflow opnemen na elke
  `marimo export html` en na elke `m00`-run, (b) bij een volgende echte m00-run op GPU 1
  controleert de auditor automatisch de dan geldende data, (c) `public/slide1_pitch_preview.html`
  visueel nalopen in een browser bij het maken van de definitieve slide (de placeholders
  `nam_overview.png` / `csd_center.png` / `reaction_diffusion.gif` zijn nog niet aangeleverd).

# atoomstralen in de molecuulgraph na clamping: 9.4 en 13.4 px (was 10.5 - 17.5)
# pipeline-koppen: 0x ge-ellipseerd, kleinste kop-font 9.4 px
```
