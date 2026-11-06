# Thought & Progress Log: Slide 2 — English translation & figure minimalism
- **Date / Time:** 2026-11-06 14:15 (Europe/Amsterdam)
- **Task Goal:** Present slide 2 of the 3-minute pitch (the central dogma `z_homeo` vs. `z_adversity`
  and the one multimodal/multiscale model that answers both) **in English**, with a much quieter
  figure: only the sources that `fetch_real_hepatocyte_data.py` really uses, no editorial headings,
  no milestone badges, no legend, no explanatory sub-captions.

## 1. Analysis & Problem Statement
- **Exact request:** translate the slide to English; shorten the title to just "z_homeo or
  z_adversity"; remove the eyebrow "Aanpak · centrale these" and the pill
  "Slide 2 / 2 · 4 databronnen → 2 antwoorden"; in the first (input) column list only sources that are
  actually used by `fetch_real_hepatocyte_data.py`; in the middle hub remove "ÉÉN MODEL · ÉÉN
  REPRESENTATIE" and the italic "alle modaliteiten in één vectorruimte", keeping **Graphormer +
  ToxFormer** as the only explanation; remove "TWEE ANTWOORDEN" above the output column; remove the
  badges "GEBOUWD · M0" and "VOLGENDE STAP · M1.2 → M4"; in the rose card remove "Neural ODE in één
  attractorlandschap" and "PoD = de bifurcatie → dosisrespons (M4)"; remove the arrow legend; drop
  MyGene.info from the provenance line.
- **Interpretation:** keep the *semantics* of the figure (4 real data sources → 1 model → 2 answers,
  solid = built, dotted = next step) but delete every label that does not carry information for the
  audience. The figure must still be generated from data, not hand-drawn.
- **Boundary conditions:** (1) `slides/styles/index.css` is global, so removing the eyebrow/pill must
  not break slide 1; (2) the SVG is embedded with `object-fit: contain`, so shrinking the canvas
  (424 → 406 px) slightly enlarges the text; (3) the two arrows must still land on the vertical
  centres of the two output cards after those cards shrink; (4) the source list has to stay
  *truthful* — it is derived from `metadata.sources` in `data/hepatocyte_baseline_data.json`, minus
  MyGene.info, so the "open sources" count drops from 7 to 6.

## 2. Methodological & Scientific Considerations
- **Which sources are real?** `fetch_real_hepatocyte_data.py` calls exactly two live APIs
  (`https://mygene.info/v3/query`, `https://string-db.org/api/json/network`) and combines them with
  validated reference profiles: GTEx v8 / Human Protein Atlas (healthy), NCBI GEO GSE126848
  (steatosis) and HepaRG / Rodrigues et al. 2018 (cholestasis), plus OECD AOP 34, OECD AOP 27 and the
  hepatocyte Physiological Maps (Staumont et al. 2025). The JSON therefore carries 7 sources; the
  slide shows 6 (MyGene.info hidden on request). The AOP cards get "OECD AOP 34 & 27 + Physiological
  Maps" — previously it wrongly implied the Physiological Map itself is an OECD product.
- **Why remove the labels?** In a 3-minute pitch the figure is read in ~20 seconds. Headings such as
  "ÉÉN MODEL · ÉÉN REPRESENTATIE", badges ("GEBOUWD · M0") and a legend ("doorgetrokken = gebouwd")
  are internal project language. The colour of the arrow plus the wording of the two cards already
  communicate built vs. planned, so the claim banner can carry that message instead.
- **Why keep `Graphormer 47.7M + ToxFormer 25.4M`?** One concrete, verifiable sentence about the model
  is enough to show that this is not hand-waving; it stays data-driven (parameters read from
  `data/molecular_baseline_mie.json`).
- **Data contracts untouched:** tensor shapes, file keys and milestone boundaries are not affected; only
  the presentation layer changes.

## 3. Step-by-Step Execution Plan
1. `generate_slide2_diagram.py`:
   - English docstring / comments / `print()` summary / `aria-label`;
   - `draw_hub()`: only "Multi-modal & Multi-scale AI" + "Graphormer 47.7M + ToxFormer 25.4M";
   - `draw_output_mie()` / `draw_output_adversity()`: badges and the two bottom lines removed,
     English wording, baselines shifted, rose card shrinks from 162 → 124 px;
   - `build_svg()`: canvas 1040×406, no legend row, no "TWEE ANTWOORDEN", "FOUR DATA SOURCES",
     English input cards, provenance line built from the JSON source list with MyGene.info filtered
     out; delete the now-unused `sup()`, `pill()`, `pill_width()` helpers and the `MONO` constant.
2. `slides/slides.md` (slide 2 only): title `z<sub>homeo</sub> or z<sub>adversity</sub>`, eyebrow and
   header pill removed, English sub-claim, English claim banner, English alt text, English speaker
   notes.
3. Verify on GPU 1: run the generator, validate the XML, screenshot the SVG in Firefox, then
   `pnpm dev` + `.slidev-page` overflow measurement on slides 1 and 2, then `pnpm build`.

## 4. Terminal Execution & Verification (GPU 1)
- Exact executed commands:
  ```bash
  CUDA_VISIBLE_DEVICES=1 uv run python generate_slide2_diagram.py
  python3 -c "import xml.etree.ElementTree as ET;ET.parse('public/s2_multimodal_ai_graph.svg')"
  CUDA_VISIBLE_DEVICES=1 uv run --with selenium --python 3.12 python /tmp/measure.py http://localhost:3037/2 ... 1400 820
  cd slides && pnpm build
  ```
- Generator output (`EXIT=0`):
  ```text
  == Slide 2 - multimodal & multiscale AI graph ==
     figure           : public/s2_multimodal_ai_graph.svg (12,144 chars / 12,181 bytes)
     topology         : 4 data sources -> 1 model node -> 2 answers (solid / dotted)
     MIE in-sample    : 12/12 correct, mean P 0.982
     MIE leave-one-out: 8/12, mean P 0.388 (0.184-0.513)
     parameters       : 47,677,505 + 25,440,434 = 73,117,939
     hepatocyte       : 18 genes x 8 features x 3 states, 20 edges (14+ / 6-)
     STRING score     : mean 0.814 (0.18-1.00)
     sources shown    : 6 of 7 (OECD AOP 34 · OECD AOP 27 · STRING-db v12 · GTEx v8 / HPA ·
                        GEO GSE126848 · HepaRG (Rodrigues 2018))
     dist matrix      : [19, 19] ([CLS] prepended)
  ```
- XML validation: `XML OK children 76` (was 92 before the cleanup: no legend, no badges, no
  editorial headings). Token grep on the SVG: **zero** hits for `genen|stoffen|kans|Bronnen|TWEE
  ANTWOORDEN|VIER DATABRONNEN|GEBOUWD|VOLGENDE STAP|ÉÉN MODEL|modaliteit|doorgetrokken|stippellijn|MyGene`.
- Firefox render of the standalone SVG (1040×406) checked visually: four English source cards, hub
  with only `Multi-modal & Multi-scale AI` + `Graphormer 47.7M + ToxFormer 25.4M`, green card without
  badge, rose card ending at the three attractor chips, provenance footer `Sources: OECD AOP 34 ·
  OECD AOP 27 · STRING-db v12 · GTEx v8 / HPA · GEO GSE126848 · HepaRG (Rodrigues 2018)` and
  `6 open sources · 18 genes · 12 compounds · 73.1M parameters`.
- Slidev dev + built bundle, measured with headless Firefox (`verticalOverflowPx` = page
  `scrollHeight - clientHeight`):
  | view | viewport | verticalOverflowPx | figure box | img natural |
  |---|---|---|---|---|
  | slide 2 (dev `/2`) | 1400×734 | **0** | 1249×505 | 1040×406 |
  | slide 2 (dev `/2`) | 1280×674 | **0** | 1147×464 | 1040×406 |
  | slide 2 (dist, ArrowRight) | 1400×734 | **0** | 1249×505 | 1040×406 |
  | slide 1 (regression, dev `/1`) | 1400×734 | **0** | n/a | n/a |
- Title probe (computed style of `h1.dogma-title`): `innerText = "zhomeo or zadversity"`,
  `fontSize 32px`, `textTransform none`, `sub` at `0.5em` / cyan → no stray capitalisation.
- `pnpm build` → `EXIT=0`, `dist/s2_multimodal_ai_graph.svg` = 12,181 bytes and `cmp` identical to
  `public/`, i.e. the served bundle shows exactly the newly generated figure.
- `python3 -m py_compile generate_slide2_diagram.py` → OK; no dangling references to the removed
  helpers (`pill`, `pill_width`, `sup`, `MONO`).

## 5. Verification of Completed Tasks
- [x] `generate_slide2_diagram.py` — English, minimal figure (canvas 1040×406), regenerated artifact.
- [x] `public/s2_multimodal_ai_graph.svg` — regenerated (never hand-edited), 12,181 bytes.
- [x] `slides/slides.md` — slide 2 in English: title `z_homeo or z_adversity`, no eyebrow, no header
      pill, English sub-claim (now one line), English alt text, English claim banner, English speaker
      notes. Slide 1 untouched (verified: header 60px, banner 662–713, overflow 0).
- [x] `slides/styles/index.css` — added `.dogma-title` (+ `sub`) for the short title; `slide 1` styles
      unchanged.
- [x] `logs/task_slide2_english_thought.md` — this log, sections 1–6.
- All claims above were verified by the terminal output, the XML parse, the Selenium measurements and
  the screenshots (`/home/disaacspat/ffshot/deck_s2_final.png`, `s2_fig_en_full.png`).

## 6. Reflection & Next Step
- Everything renders; de figure is now quiet enough for a 3-second read: 4 real sources → 1 model →
  2 answers, with the solid/dotted distinction carried by colour alone (the claim banner still
  explains built vs. next step in words).
- Deliberate choices to flag: `FOUR DATA SOURCES` stayed as a column header (only the right-hand
  `TWO ANSWERS` label was removed, as requested); the "6 open sources" counter now excludes
  MyGene.info because the label is hidden.
- Side effect worth knowing: `uv run --with selenium --python 3.12` re-created `.venv` in the repo
  root once (torch et al. re-downloaded); the generator still runs with `EXIT=0` afterwards.
- Immediate follow-up (unchanged from the previous task): commit the five files (mind the untracked
  `uv.lock` artifact), let the host rebuild, and replace `public/csd_experiment.webm` (still the v1
  simulation). Then start milestone 1.2 (Graphormer encoder) — that fills the dotted arrow.
