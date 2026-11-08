# Thought & Progress Log: Math styling in the m00 dashboard + slide-2 pipeline figure rework
- **Date / Time:** 2026-11-08 18:45 (local)
- **Task Goal:** (1) All mathematics in `public/00_molecular_baseline_mie.html` must read as mathematics (no monospace code chip, no different background). (2) Rework the slide-2 figure `public/s2_multimodal_ai_graph.svg`: add a fifth data-source node for physiological measurements (e.g. MEA), add a legend (green = done, purple dashed = future work), make every arrow head sit exactly on the node border, remove the overlapping `10/12` / `correct` number pair, correct the misleading "chance that a compound induces a specific MIE" claim, and label the purple arrow from the model to the bottom node with "(Neural) ODEs".

## 1. Analysis & Problem Statement
- **Prompt (NL, summarised):** "Let op alle wiskunde in de m00 html - die ziet er niet uit als wiskunde, met een ander lettertype/achtergrond. En de slide-2 figuur: voeg een extra node toe in de eerste kolom voor fysiologische metingen (bijv. MEA), maak een legenda (groene pijl = done, paarse dashed pijl = future work), let op de pijlen (de lijn loopt niet mooi naar het midden van de pijl), check dat er overal genoeg ruimte tussen de tekst zit en dat er nergens overlappende getallen zijn zoals in de bovenste resultaten-node, de tekst daar klopt niet (het is niet de kans dat een stof een MIE triggert), en zet de (Neural) ODE's op de paarse pijl van het model naar de onderste node."
- **Interpretation:** two independent deliverables. (A) Typography fix inside the generated Milestone-0 HTML dashboard. (B) Geometry + copy rework of the hand-written slide-2 SVG generator.
- **Measured root cause of (A):** the formulas are wrapped in `<code>` and the document has **no** CSS rule for `code` at all. Chromium computed style of the `P_consensus` element: `backgroundColor = rgba(0,0,0,0)`, `fontFamily = monospace`, parent `Inter`. The "different background" is a visual artefact of the browser's default monospace font - the fix is semantic markup (`<span class="math">` with serif math + italic variables) plus an explicit `code { background: none !important }` rule so that any preview that injects a code-chip style cannot reintroduce a box.
- **Boundary conditions / pitfalls:**
  - `public/00_molecular_baseline_mie.html` is *generated* by `m00_molecular_baseline_mie.py` (f-string HTML with doubled `{{ }}` braces) - never hand-edit the artifact. A full m00 run reloads Graphormer + ToxTransformer on GPU 1 (minutes), but `tools/rerender_m00_html.py` rebuilds the same HTML from `data/molecular_baseline_mie.json` without touching the checkpoints -> use that.
  - The slide figure is width-limited on the slide (`.pipeline-figure` = `width/height:100%` + `object-fit:contain`; container ~932x430 px). Current SVG is 1040x406 (aspect 2.56) -> scale 932/1040 = 0.896 with ~66 px unused height. Growing the canvas to 1040x460 (aspect 2.26) therefore costs **zero** font size on the slide and buys room for the fifth node.
  - Extra `<code>` must not disappear silently: real identifiers (HF model ids, `public/`) must stay `<code>`.
  - The `.math`/`code` CSS lives inside an f-string -> every brace must be doubled.
  - The fifth card's arrow must be dashed purple (future work) like the other three biological modalities, not solid green.
  - Arrow heads: the marker was `markerUnits="strokeWidth"` + `markerWidth="7"` -> a 16 px wide triangle for the 2.3 px strokes (and a different size per stroke width) which visually detaches from the line. Fixed by `markerUnits="userSpaceOnUse"` with `refX` exactly on the tip.
  - Overlap `10/12` / `correct`: "10/12" at 15.5 px bold starts at `x+14` and is ~44 px wide -> ends at ~`x+58`, while "correct" started at `x+56` -> 2 px collision. Moved to `x+68`.

## 2. Methodological & Scientific Considerations
- **Why serif math in a sans-serif dashboard:** the deck itself typesets variables cursively (`z_homeo`, `z_adversity`) and the goal is "reads as mathematics". A LaTeX-style serif (`Cambria Math` / STIX / Georgia fallback) with italic variables (`<i>P</i>`, `<i>c</i>`, `<i>n</i>`) and upright multi-letter subscripts (`consensus`, `Graph`, `Tox`) follows the LaTeX convention (variables italic, labels upright) and makes formulas unmistakable at a glance. No external KaTeX/MathJax dependency is introduced: the dashboard must stay a single standalone HTML file.
- **Why the claim had to change:** the node reports a 3-class discriminative output (`no MIE` / `MIE 167` / `MIE 41`) produced by cosine similarity to class prototypes built from the **other 11 of 12 reference compounds** in a frozen foundation embedding space. That is a *prediction of the MIE class from structure alone*, not a mechanistic probability that a compound triggers an MIE in a hepatocyte. The title therefore becomes `Predicted MIE from 12 compounds` and the subtitle `structure-only classifier - held-out LOOCV`; the numbers (`10/12`, mean P 0.52, 3x3 matrix) stay unchanged because they are correct LOOCV statistics.
- **Why a legend is scientifically honest:** the figure encodes model status (built/validated vs. future work) purely by line style. Encoding without a key is not self-documenting, especially for a mixed audience (toxicologists + AI panel). The legend makes the claim boundary of the proposal explicit: only Milestone 0 / 1.1 are built.
- **Why MEA as the fifth source:** the proposal (ARCHITECTURE.md / pitch) uses Multi-Electrode Arrays for tissue-level emergent behaviour (Cortical Spreading Depression), i.e. physiological measurements that a purely molecular/gene-based model cannot see. Adding it as a violet, `future work` node keeps the figure aligned with the research proposal without overclaiming.
- **Consistency rule:** all numbers in the figure keep being derived from `data/molecular_baseline_mie.json` and `data/hepatocyte_baseline_data.json` (no hard-coded metrics), and `assert all(isfinite(...))` stays in place.


## 3. Step-by-Step Execution Plan
- **Files created:** `logs/task_fix_math_and_slide2_diagram_thought.md` (this file).
- **Files modified:**
  1. `m00_molecular_baseline_mie.py` - add `.math` + `code` CSS to the report `<style>` block; convert every formula/quantity to `<span class="math">` with italic variables; demote model/compound/method names and figure column titles from `<code>` to `<b>`; keep `<code>` only for the two HF model ids and `public/`.
  2. `public/00_molecular_baseline_mie.html` - regenerated via `tools/rerender_m00_html.py`.
  3. `generate_slide2_diagram.py` - canvas `1040x460`; `IN_H 72 / IN_STEP 80`; `HUB_Y 157`; `OUT1_Y 58 / OUT2_Y 262`; new `marker()` (userSpaceOnUse, tip on the path end); new `elbow()` rounded orthogonal connector for the two answer arrows; fifth input card (`Physiological measurements`, MEA, violet, `future work - not yet integrated`) with a new `signal` icon and a dashed purple arrow into the hub; `FIVE DATA SOURCES` label; `draw_legend()` under the hub (green solid = done, purple dashed = future work); two-line `(Neural)`/`ODEs` chip on the purple answer arrow inside the 72 px gutter; `Predicted MIE from 12 compounds` + new subtitle; `correct` shifted to `x+68`; missing comma in `P(z_homeo, z_adversity | context, ...)` restored; `aria-label`, module docstring and console topology line updated.
  4. `public/s2_multimodal_ai_graph.svg` - regenerated.
  5. `slides/slides.md` - alt text, sub-claim and the slide-2 speaker notes (four -> five sources, new MIE formulation, dashed instead of dotted).
- **Verification:** `getBBox()` audit in headless Chromium over every `<text>` element of the SVG (all text/text overlaps + text/card overlaps reported), plus cropped screenshots of the formula, the hub, the gutter chip, the legend and the result node.

## 4. Terminal Execution & Verification (GPU 1)
### 4.1 Dashboard HTML regenerated from the cache (no GPU checkpoints reloaded)
```
$ CUDA_VISIBLE_DEVICES=1 uv run python tools/rerender_m00_html.py
  -> Graphormer (2D OGB graph)                    LOOCV 4/12 (33.3%) | mean P(true class) 0.333 | macro F1 0.292
  -> ToxTransformer (SMILES / bioassays)          LOOCV 9/12 (75.0%) | mean P(true class) 0.687 | macro F1 0.741
  -> Dual consensus (50 / 50 probability fusion)  LOOCV 10/12 (83.3%) | mean P(true class) 0.517 | macro F1 0.832
[cache] 12 compounds | PC shapes (12, 2), (12, 2) | cos (12, 12)
[ok] re-rendered public/00_molecular_baseline_mie.html (512,614 bytes)
```
### 4.2 Slide-2 figure regenerated
```
$ CUDA_VISIBLE_DEVICES=1 uv run python generate_slide2_diagram.py
== Slide 2 - multimodal & multiscale AI graph ==
   figure           : public/s2_multimodal_ai_graph.svg (20,492 bytes)
   topology         : 5 data sources -> 1 model node -> 2 answers (solid / dashed)
   MIE leave-one-out: 10/12 correct, mean P 0.517 (0.164-0.665) [graphormer]
   parameters       : 47,677,505 + 25,440,434 = 73,117,939
   hepatocyte       : 18 genes x 8 features x 3 states, 20 edges (14+ / 6-)
   dist matrix      : [19, 19] ([CLS] prepended)
```
### 4.3 Rendered geometry audit (Chromium via Playwright - the engine the deck uses)
```
$ node /tmp/pw_s2.mjs public/s2_multimodal_ai_graph.svg /tmp/pws2
{ "n_text": 63, "viewBox": "0 0 1040 460",
  "overlaps": [ { "a": "P(zhomeo, zadversity | context,",
                  "b": "exposure, dose)", "rw": 1, "rh": 0.33 } ],
  "outside": [] }
```
The single hit is the two lines of the *same* two-line formula: `getBoundingClientRect()` of a
`<text>` that contains subscript `<tspan dy>` nodes includes the subscript descent, so the boxes of
line 1 and line 2 touch even though the ink is 15 px apart (checked in the `out2` crop). Same result
in Firefox headless (`/tmp/s2_audit.py`): 63 text nodes, same single artefact, `outside: []`.

### 4.4 Arrow geometry (parsed from the generated SVG)
```
curves end at: 372.0 178.0 / 372.0 200.0 / 372.0 222.0 / 372.0 244.0 / 372.0 266.0  (hub border x=372)
elbows end at: L 744.0 156.0 and L 744.0 327.0                                     (card border x=744)
markers      : 3x markerUnits="userSpaceOnUse", markerWidth="10.4", refX="11.4"
```
So every head has the same size and its tip sits exactly on the border it points at; the two answer
connectors arrive horizontally, i.e. straight through the axis of their own head.

### 4.5 Math typography in the dashboard (computed style, Chromium + Firefox)
```
n_math = 13 (spans), n_code = 3 (only the two HF model ids + public/)
.math  : family "Cambria Math" | style normal | bg rgba(0,0,0,0) | border 0px | padding 0px
.math i: style italic, family "Cambria Math"
code   : family ui-monospace | bg rgba(0,0,0,0) | border 0px | padding 0px
```
Visual crops: `/tmp/m00ch_formula.png` (`P_consensus(c) = 0.5 · P_Graph(c) + 0.5 · P_Tox(c)` in
serif with italic *P* and upright subscripts) and `/tmp/m00ch_loocv.png` (`[3 × 3]`, `n = 12`).

### 4.6 Overlap regression check on the dashboard
```
$ uv run python tools/check_html_overlaps.py 00_molecular_baseline_mie.html
=== 00_molecular_baseline_mie.html === 17 SVG('s), 2 bevinding(en)
  SVG #3  TEXT-OUTSIDE  "Target: NR1H3 / SREBF1 & Mitochondrial beta-oxidation ..." (1148 > 1140)
  SVG #15 TEXT-TEXT     "PC 2 - 2nd Variation Axis (768D)" vs "DMSO_Contr"
TOTAAL: 2 bevinding(en)
$ git show HEAD:public/00_molecular_baseline_mie.html > /tmp/old_m00.html
$ uv run python tools/check_html_overlaps.py /tmp/old_m00.html
TOTAAL: 2 bevinding(en)        # identical -> both pre-existing, nothing new introduced
```
### 4.7 In-situ slide measurement (980 x 552 Slidev canvas, the real `.pipeline-figure` CSS)
```
img: { w: 932.0, h: 453.0 }   card: { w: 932.0, h: 453.0 }   scale: 0.896
```
The scale is identical to the old 1040x406 figure (932/1040 = 0.896): the extra 54 px of canvas
height is margin that was previously unused, so no glyph became smaller. Screenshots:
`/tmp/slide2_inplace.png`, `/tmp/pws2_full.png`, crops `/tmp/pws2_hub.png`, `/tmp/pws2_gutter.png`,
`/tmp/pws2_legend.png`, `/tmp/pws2_out1.png`, `/tmp/pws2_out2.png`.

## 5. Verification of Completed Tasks
| Claim | Verified how |
|---|---|
| this log exists | editor output |
| `m00_...py` has `.math`/`code` CSS + 13 math spans | `grep -o 'class="math"' ...html \| wc -l` = 13 + computed styles in 4.5 |
| `public/00_molecular_baseline_mie.html` regenerated (512,614 bytes) | `[ok] re-rendered ...` + `ls -la` |
| only real identifiers still in `<code>` | `left_over_code` = the 2 HF ids + `public/` |
| `generate_slide2_diagram.py`: 5 sources, legend, elbow, chip, new title | `topology: 5 data sources ...`, SVG grep, screenshots |
| `public/s2_multimodal_ai_graph.svg` regenerated (20,536 bytes, viewBox 1040x460) | `ls -la` + `viewBox 0 0 1040 460` |
| no text/text overlap introduced | 4.3 + 4.6 (2 pre-existing findings, unchanged) |
| `slides/slides.md`: four -> five, dotted -> dashed, MIE claim | `git diff --numstat` = 7 added / 7 removed lines (526/528/530/540/541/547/548) |
| nothing outside the viewBox | `outside: []` in both browsers |

All files claimed above were really written; nothing was left as a chat dump.

## 6. Reflection & Next Step
- Everything works: the dashboard formulas are now serif mathematics with italic single-letter
  variables and no code chip, and the figure has five sources, a legend, flush arrow heads, a
  labelled purple answer arrow and no colliding numbers.
- The bbox "overlap" in 4.3/4.4 is a measurement artefact of `<tspan dy>` subscripts, not visible
  ink. Reducing it would mean squeezing the line spacing of a two-line formula that already reads
  cleanly, so it is deliberately left alone.
- Two pre-existing dashboard findings (a rotated PCA axis label vs. a tick label, and one pipeline
  tag that is 8 px wider than its viewBox) are untouched by this task; fixing them means another
  layout pass over the PCA / pipeline SVGs.
- Follow-up, out of scope here: `slides/dist` and `slides/slides-export.pdf` are still the 6 Nov
  builds. Run `cd slides && npm run build` (and `npm run export`) to refresh the deck and the PDF
  with the new figure and speaker notes.
