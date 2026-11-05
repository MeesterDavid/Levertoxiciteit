# Thought & Progress Log: Pitch Presentation — Slide 1 (Context & De Kwestie)
- **Date / Time:** 2026-11-05T15:13:00+01:00
- **Task Goal:** Meedenken over en opzetten van Slide 1 (en een lege placeholder voor Slide 2) in Slidev (`slides/slides.md`) voor de 3-minuten pitch (max. 2 slides, gevolgd door 7 minuten Q&A) over het onderzoeksvoorstel *Multimodal en Multiscale AI in de Toxicologie*.

## 1. Analysis & Problem Statement
- **What is being asked? Both the exact prompt and its interpretation:**
  - De gebruiker bereidt een **pitch van maximaal 3 minuten met maximaal 2 slides** voor (daarna 7 minuten vragen van de commissie; totaal 10 minuten, strikt gehandhaafd).
  - **Focus van Slide 1:** *De Context* en *De Kwestie* (met zeer weinig tekst op de slide zelf, en een vloeiende, inhoudelijk sterke uitgeschreven spreektekst in de Slidev presenter notes `<!-- ... -->`).
  - **Deel 1 van Slide 1 — De Context:**
    - *Probabilistic Chemical Risk Assessment (PCRA)* & *New Approach Methodologies (NAMs)*: *in vitro* en *in silico* modellen voor dierproefvrije toxicologie.
    - Twee belangrijke pijlers om *adverse outcomes* vast te leggen en mechanistisch te verklaren: **(quantitative) Adverse Outcome Pathways ((q)AOP's)** en **Physiological Maps**.
    - Visueel: Een beeld dat tekenend is voor **NAMs** (`nam_overview.png`).
  - **Deel 2 van Slide 1 — De Kwestie (Waarom AI nodig is):**
    - **AOP's** zijn sterk voor een reductionistische aanpak ($MIE \rightarrow KE \rightarrow AO$), maar zijn **lineair** en het kwantificeren (*qAOP*) is nog volop in ontwikkeling.
    - Biologische werkelijkheid is **non-lineair** en **multiscale**:
      1. *Homeostatische feedbackloops* en cross-talk (gedekt door **Physiological Maps**).
      2. *Initiating events op weefselniveau* (zoals fysische verstoring door **asbest**).
      3. *Tissue-level emergence*: Soms zegt de MIE alleen onvoldoende over of de Adverse Outcome daadwerkelijk optreedt. Voorbeeld: **Cortical Spreading Depression / Disease (CSD)** — er is wel een MIE (*ion channel disruption*), maar of de AO optreedt hangt af van emergente weefseldynamiek (*reaction-diffusion* golven, gemeten met o.a. **MEA-data** / Multi-Electrode Arrays over een heel gebied).
    - **De Claim:** Om een Adverse Outcome probabilistisch te voorspellen moeten we heterogene, multimodale en multiscale data integreren (AOP's, Physiological Maps, MEA-data, imaging, transcriptomics). **Multimodal & Multiscale AI** is bij uitstek geschikt om dit samen te brengen.
    - Visueel (Rechterhelft / Organisatie-diagram): Een hub-and-spoke / organisatie-diagram waarin:
      1. Een klein maar direct herkenbaar plaatje van een **AOP** (`aop_mini.svg` / `aop_mini.png`),
      2. Een klein maar direct herkenbaar plaatje van een **Physiological Map** (`physiological_map_mini.svg` / `physiological_map.png`),
      3. Een **GIF van een reaction-diffusion effect** (`reaction_diffusion.gif`),
      *(plus optioneel MEA-signaal)* allemaal met pijlen wijzen naar het midden: een plaatje van **CSD** (`csd_center.png`) met de term **CSD** eronder, en hoe AI dit samenbrengt.
- **What are the potential pitfalls or boundary conditions?**
  - Het bestand `Onderzoeksvoorstel Multimodal en Multiscale AI in de Toxicologie.gdoc` op de server is `0 bytes` (leeg). We moeten dit eerlijk melden aan de gebruiker, maar we hebben alle inhoudelijke bouwstenen uit de prompt van de gebruiker én `ARCHITECTURE.md`.
  - Slidev laadt statische bestanden vanuit `slides/public/`. Door `slides/public` als symlink te koppelen aan `/home/disaacspat/data/demon_storage_box/levertoxiciteit/public`, werken bestanden die de gebruiker in `/public` plaatst direct in Slidev.
  - Voor een pitch van 3 minuten over 2 slides heb je voor **Slide 1 ongeveer 90 tot 100 seconden (~200-230 woorden)** spreektijd. De spreektekst in de notities moet dus precies op dat tempo geschreven zijn!

## 2. Methodological & Scientific Considerations
- **Slide Design (Cognitive Load Theory):**
  - Tijdens een korte commissie-pitch van 3 minuten moet de commissie naar de kandidaat luisteren, niet lange zinnen lezen. Op de slide zelf gebruiken we daarom alleen **kernbegrippen / korte tags** en een **sterke visuele tweedeling**:
    - **Links (De Context — Dierproefvrije Risicobeoordeling):** Visueel kader voor **NAMs** (*In Vitro* + *In Silico* $\rightarrow$ *Probabilistic Risk Assessment*) met een duidelijke placeholder/visual.
    - **Rechts (De Kwestie — Multiscale Emergentie & Multimodaliteit):** Het integratiediagram waarin **(q)AOP** (lineair/reductionistisch), **Physiological Map** (homeostatische feedbackloops) en **Reaction-Diffusion + MEA** (weefsel-emergentie) met pijlen convergeren naar **CSD** in het midden, omsloten door de **Multimodal & Multiscale AI** integratielaag.
- **Visual Assets & Placeholders in `/public`:**
  - We maken `slides/public` een symlink naar `../public` zodat `/home/disaacspat/data/demon_storage_box/levertoxiciteit/public/` direct als asset-map dient.
  - We maken strakke, direct herkenbare vector-miniaturen voor **AOP** (`public/aop_mini.svg`) en **Physiological Map** (`public/physiological_map_mini.svg`), én we genereren op GPU 1 / Python een echt **reaction-diffusion golf-GIFje** (`public/reaction_diffusion.gif`), plus duidelijke visuele placeholders voor `public/nam_overview.png` en `public/csd_center.png` die in de slide automatisch het eigen plaatje van de gebruiker tonen zodra die het bestand in `/public` plaatst.

## 3. Step-by-Step Execution Plan
1. Koppel `slides/public` via een symlink aan `/home/disaacspat/data/demon_storage_box/levertoxiciteit/public`.
2. Schrijf een hulpscript `generate_slide1_assets.py` en voer dit uit met `CUDA_VISIBLE_DEVICES=1 uv run python generate_slide1_assets.py` om:
   - `public/aop_mini.svg` (helder MIE $\rightarrow$ KE1 $\rightarrow$ KE2 $\rightarrow$ AO keten-icoon) te genereren,
   - `public/physiological_map_mini.svg` (niet-lineair netwerk met feedbackloops) te genereren,
   - `public/reaction_diffusion.gif` (geanimeerde 2D reaction-diffusion / traveling wave simulatie die CSD-uitbreiding over weefsel illustreert) te genereren,
   - `public/nam_overview_placeholder.svg` en `public/csd_center_placeholder.svg` als nette visuele placeholders klaar te zetten.
3. Bouw `slides/slides.md` op met Slide 1 (Context & Kwestie + uitgebreide spreektekst in de presenter notes) en een voorlopige Slide 2 placeholder.

## 4. Terminal Execution & Verification (GPU 1)
- Exact executed command: `CUDA_VISIBLE_DEVICES=1 uv run python generate_slide1_assets.py`
- Output / Terminal log:
  - Generated `public/aop_mini.svg` (2,078 bytes)
  - Generated `public/physiological_map_mini.svg` (2,547 bytes)
  - Generated `public/reaction_diffusion.gif` (128,823 bytes, 36 frames animated 2D wave)
  - Generated `public/nam_overview_placeholder.svg` (2,707 bytes)
  - Generated `public/csd_center_placeholder.svg` (2,772 bytes)
  - Verified `npx slidev build` in `slides/` -> `slides/dist/index.html` built cleanly with exit code 0.

## 5. Verification of Completed Tasks
- Verified with `ls -la public/ slides/public` that all assets and symlinks exist and `slides/slides.md` is updated and compiles cleanly.

## 6. Reflection & Next Step
- Slide 1 is ready for review and discussion with the user. Next step: refine Slide 1 based on user feedback, replace any placeholders with custom images in `/public`, and design Slide 2 (CV, Motivatie & Aanpak).
