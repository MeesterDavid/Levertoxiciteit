# Thought & Progress Log: Slide 2 — Multimodale & multischaalse AI als centrale these
- **Date / Time:** 2026-11-06
- **Task Goal:** Slide 2 ("Aanpak") van de 3-minutenpitch vullen met de centrale these uit
  `ARCHITECTURE.md` §1: een levercel kiest tussen de homeostatische attractor $z_{\text{homeo}}$ en de
  pathologische attractor $z_{\text{adversity}}$ ($z_{\text{steatose}}$, $z_{\text{cholestase}}$).
  De oplossing is één multi-modaal/multi-schaal AI-model dat eiwit-, gen- en chemische databronnen
  integreert; de figuur toont dat model als knooppunt met twee uitgangen:
  (1) **doorgetrokken pijl** → $P(\text{MIE} \mid \text{structuur})$ — vandaag gebouwd en gevalideerd,
  (2) **stippellijn** → $P(\text{adversity} \mid \text{context}, \text{exposure}, \text{dosis})$ —
  de volgende stap (M1.2 → M4).

## 1. Analysis & Problem Statement
- **Exacte opdracht:** titel die het "centrale dogma" verwoordt (homeostasis vs. adversity);
  als oplossing "AI die verschillende biologische databronnen integreert"; één figuur met
  eiwitinteractie + genexpressie + geninteractie + chemische informatie → knooppunt
  "Multi-modal and Multi-scale AI" → knooppunt "kans dat een stof een specifiek MIE induceert"
  (doorgetrokken pijl) én → knooppunt $P(\text{adversity} \mid \text{context}, \text{exposure}, \text{dosis})$
  (gestippelde pijl, omdat dit nog niet gebouwd is).
- **Interpretatie / boundary conditions:**
  * De repo stond na de revert terug op commit `8780b77`; slide 2 was weer de stub
    (`# Slide 2 — Aanpak, CV & Motivatie`). De eerdere grafiek-assets (`public/s2_*.svg`) en
    `generate_slide2_assets.py` bestaan niet meer; alleen `slides/styles/index.css` is blijven staan.
  * De figuur is één SVG die de hele slide draagt → geen twee losse grafieken meer, maar de
    validatiecijfers moeten **in** de figuur staan (anders verliest de slide het "echte data"-karakter).
  * Een gestippelde pijl mag niet als "kapot" of "optioneel" lezen: met een expliciete legenda
    (doorgetrokken = gebouwd/gevalideerd, gestippeld = gepland) wordt het verschil semantisch.
  * Geen overflow: het canvas is 1120 × 630 (canvasWidth 1120, 16/9). De figuur moet flexibel
    meeschalen (`object-fit: contain`) in plaats van een vaste hoogte krijgen.

## 2. Methodological & Scientific Considerations
- **Waarom één fusieknooppunt?** In `ARCHITECTURE.md` is de meerwaarde van het voorstel precies dat
  de modalities (STRUCTUUR: SMILES; NETWERK: STRING-db + AOP-edges; TOESTAND: genexpressie) niet naast
  elkaar worden gemodelleerd maar in één representatie $z \in \mathbb{R}^{64}$ samenkomen. De figuur
  maakt dat contract zichtbaar: 4 pijlen in, 2 pijlen uit.
- **Waarom de twee uitgangen asymmetrisch?** $P(\text{MIE} \mid \text{structuur})$ is in Mijlpaal 0/1.1
  empirisch gevalideerd (12 referentiestoffen, in-sample 12/12 met $\bar P \approx 0.98$,
  leave-one-out 8/12 met $\bar P \approx 0.39$). $P(\text{adversity} \mid \text{context}, \text{exposure},
  \text{dosis})$ vereist de attractor-dynamiek (M1.2 Graphormer-encoder → M2 perturbatie-encoder →
  M3 Neural ODE → M4 bifurcatie/PoD) en is dus nog een hypothese. Een stippellijn is de enige eerlijke
  weergave; een doorgetrokken pijl zou een resultaat suggereren dat er niet is.
- **Waarom de attractor-namen in de figuur?** $z_{\text{homeo}}$, $z_{\text{steatose}}$ en
  $z_{\text{cholestase}}$ zijn de exacte variabelenamen uit `ARCHITECTURE.md` (§1, adaptieve vs.
  pathologische attractor) en de tensorshapes uit `.clinerules` ($z_0$: `[B, 64]`). De knoppen in de
  figuur gebruiken die namen letterlijk, zodat de slide en de code hetzelfde vocabulaire houden.
- **Tooling:** geen matplotlib (`.clinerules`); de figuur is een handgeschreven SVG via Python
  (zelfde aanpak als `generate_slide1_assets.py` en `public/generate_aop34_svg.py`), zodat tekst,
  pijlen en marker-definities exact te controleren zijn en er geen extra runtime-afhankelijkheid
  (kaleido/plotly-export) nodig is.

## 3. Step-by-Step Execution Plan
1. `generate_slide2_diagram.py` (root, naast `generate_slide1_assets.py`):
   - leest `data/molecular_baseline_mie.json` (12 stoffen; in-sample en LOOCV-kansen per stof) en
     `data/hepatocyte_baseline_data.json` (18 genen, 20 edges, STRING-scores, 3 toestanden);
   - berekent de cijfers die in de figuur komen (in-sample/LOOCV-nauwkeurigheid en gemiddelde
     kansen, aantal edges per teken, gemiddelde STRING-score, aantal bronnen) en print die;
   - schrijft `public/s2_multimodal_ai_graph.svg`: 4 invoerkaarten (eiwitinteractie, genexpressie,
     geninteractie, chemische informatie) → trechter van pijlen → hub "Multi-modale & multi-schaalse
     AI" → doorgetrokken pijl naar uitvoer 1 ($P(\text{MIE} \mid \text{structuur})$) en stippellijn
     naar uitvoer 2 ($P(\text{adversity} \mid \text{context}, \text{exposure}, \text{dosis})$ met
     $z_{\text{homeo}}$ / $z_{\text{steatose}}$ / $z_{\text{cholestase}}$), plus legenda en bronregel.
2. `slides/slides.md`: stub vervangen door de volledige slide (header met de these-titel,
   één regel "de oplossing", de figuur, claim-banner met vandaag/volgende stap) + Nederlandse
   spreektekst in een HTML-commentaar.
3. `slides/styles/index.css`: de slide-2-specifieke klassen (`.sub-claim`, `.figure-card`,
   `.pipeline-figure`) toevoegen; bestaande primitieven blijven staan.
4. Verifiëren op GPU 1: `CUDA_VISIBLE_DEVICES=1 uv run python generate_slide2_diagram.py`,
   daarna `pnpm dev` + Firefox/Selenium-screenshot van beide slides (overflow-meting), en `pnpm build`.

## 4. Terminal Execution & Verification (GPU 1)
- **Generator:**
  `CUDA_VISIBLE_DEVICES=1 uv run python generate_slide2_diagram.py`
  ```text
  == Slide 2 - multimodal & multiscale AI graph ==
     figure           : public/s2_multimodal_ai_graph.svg (14,891 bytes)
     topology         : 4 modalities -> 1 model node -> 2 answers (solid / dotted)
     MIE in-sample    : 12/12 correct, mean P 0.982
     MIE leave-one-out: 8/12, mean P 0.388 (0.184-0.513)
     parameters       : 47,677,505 + 25,440,434 = 73,117,939
     hepatocyte       : 18 genes x 8 features x 3 states, 20 edges (14+ / 6-)
     STRING score     : mean 0.814 (0.18-1.00), 7 open sources
     dist matrix      : [19, 19] ([CLS] prepended)
  ```
  `EXIT=0`; XML-validatie: `XML OK, root {http://www.w3.org/2000/svg}svg children 92`;
  geen dubbele escaping meer (`grep -c '&amp;amp;' → 0`).
- **Echte browser-render (Firefox headless + geckodriver, 1400×820, dev-server poort 3037):**
  ```text
  visiblePageClass: slidev-page slidev-page-2
  pageScrollHeight: 630   pageClientHeight: 630   verticalOverflowPx: 0
  subClaim: 93–129 | figure: 143–640 | img: 143–640 (1249×498) | banner: 662–713
  imgNatural: 1040×424
  ```
  Slide 1 (regressiecheck met de nieuwe globale CSS): `verticalOverflowPx: 0`.
  Tweede resolutie 1280×760: `verticalOverflowPx: 0`.
- **Productiebuild + gebouwde bundle:**
  `cd slides && pnpm build` → `EXIT=0`, `✓ built in 2.21s`;
  `dist/s2_multimodal_ai_graph.svg` (14,952 bytes) aanwezig, `index.html` verwijst ernaar,
  slide-2-tekst zit in `dist/assets/slidev/md-C8EbykZh.js`;
  de gebouwde deck gerenderd via `python3 -m http.server 8099` + ArrowRight →
  `slidev-page-2`, `verticalOverflowPx: 0`, `figure: 143–640`, `banner: 662–713`
  (identiek aan de dev-render).

## 5. Verification of Completed Tasks
| Claim | Verified? | Evidence |
|---|---|---|
| Titel verwoordt het centrale dogma (homeostasis vs. adversity) | ✅ | `<h1 class="main-title">Terug naar z<sub>homeo</sub>, of kantelen naar z<sub>adversity</sub>?</h1>` in `slides/slides.md` |
| "Oplossing = AI die biologische databronnen integreert" staat op de slide | ✅ | `.sub-claim` met tag "DE OPLOSSING" (zichtbaar in `~/ffshot/deck_s2_dist.png`) |
| 4 databronnen → 1 knooppunt "Multi-modal and Multi-scale AI" → 2 antwoorden | ✅ | `public/s2_multimodal_ai_graph.svg` (92 root-children), screenshot `~/ffshot/s2_fig.png` |
| Antwoord 1 met **doorgetrokken** pijl: kans op een specifiek MIE | ✅ | `curve(..., "arr-solid", width=2.3)` + kaart "P(MIE \| structuur)" met 12/12 · 0.98 en 8/12 · 0.39 |
| Antwoord 2 met **stippellijn**: P(adversity \| context, exposure, dosis) | ✅ | `curve(..., dash="7 6")` + kaart "P(adversity \| context, exposure, dosis)" met z_homeo/z_steatose/z_cholestase |
| Geen overflow / geen clipping van de figuur | ✅ | `verticalOverflowPx: 0` op 1400×820, 1280×760 en in de gebouwde bundle |
| Slide 1 niet geregressed door de globale CSS | ✅ | `verticalOverflowPx: 0` op `/1` |
| Alle getallen komen uit de artefacten (niet met de hand) | ✅ | `generate_slide2_diagram.py` leest `data/molecular_baseline_mie.json` en `data/hepatocyte_baseline_data.json` |
| Nederlandse spreektekst toegevoegd | ✅ | HTML-commentaar "SPREEKTEKST SLIDE 2 (± 90–100 seconden)" onderaan `slides.md` |

Aangemaakte/aangepaste bestanden (gecontroleerd met `git status` / `ls -la`):
`generate_slide2_diagram.py` (nieuw), `public/s2_multimodal_ai_graph.svg` (nieuw),
`slides/slides.md` (stub → volledige slide 2), `slides/styles/index.css` (slide-2-klassen),
`logs/task_slide2_multimodal_graph_thought.md` (dit log).

## 6. Reflection & Next Step
- **Werkt het?** Ja. De figuur is volledig datagedreven, de slide heeft 0 px overflow in dev én in de
  productiebuild, en de semantiek van de pijlen (doorgetrokken = gebouwd, stippel = gepland) staat in
  de figuur zelf, in de kaartlabels én in de legenda — dus ook zonder spreektekst correct leesbaar.
- **Bewuste keuzes / beperkingen:** (1) De figuur is een gegenereerd artefact: niet handmatig aanpassen,
  maar `generate_slide2_diagram.py` opnieuw draaien. (2) De LOOCV-cijfers (8/12, gem. P 0.39) staan
  bewust naast de in-sample cijfers (12/12, gem. P 0.98): dat contrast ís de boodschap van
  probabilistische risk assessment. (3) De kleine lettertjes in de figuur (8–9 px in viewBox-eenheden)
  zijn op een projector nog leesbaar, maar dit is het eerste dat je zou opvoeren bij een groter scherm.
- **Directe vervolgstap:** de nieuwe bestanden committen (`generate_slide2_diagram.py`,
  `public/s2_multimodal_ai_graph.svg`, `slides/slides.md`, `slides/styles/index.css`,
  `logs/task_slide2_multimodal_graph_thought.md`) zodat de host opnieuw bouwt; daarna in Mijlpaal 1.2
  de Graphormer-encoder bouwen die de stippellijn daadwerkelijk gaat invullen (z0, z_steatose,
  z_cholestase als `[1, 64]`-ankers).

