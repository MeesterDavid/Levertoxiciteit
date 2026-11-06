# Thought & Progress Log: Slide 1 — legible AOP figure, banner removal, header clean-up
- **Date / Time:** 2026-11-06
- **Task Goal:** Slide 1 of `slides/slides.md` (NAM + linear-vs-dynamic AOP); feeds the
  Milestone 1.2/2/3 chain by visualising what "dynamic, quantitative" has to add to a linear AOP.

## 1. Analysis & Problem Statement
Exact prompt (verbatim): *"text is not neatly fixed in the colored rectangles, also the text
'from qualitative linear chains etc.' on the first slide doesnt need to be in a coloured
rectangle, we also dont need A and B for the two parts of the kwestie, also create a bit more
space the the AOP. its too small now. nicrease font size and if needed the containers. the
purple container on the bottom is not visible on the slide, remove it"*

Interpretation, item by item:

1. **"text is not neatly fixed in the colored rectangles"** → the labels *inside* the AOP-34
   figure do not fit their boxes. Confirmed by parsing `public/aop_linear_before.svg`:
   `viewBox="0 0 1480 490"` with 10.5–14 px fonts, e.g. `Endoplasmic reticulum stress` (30 chars)
   is drawn at 11 px inside a 150 px wide `<rect>` — an estimated 165 px of text in a 150 px box.
   Rendering the 1480-wide canvas inside a ~716 px wide panel gives an effective 5.2 canvas px per
   label (~9 px on a 1920 projector): unreadable and visibly spilling over the box borders.
2. **The "from qualitative linear chains…" text must not sit in a coloured rectangle** → the
   formula chip `.math-badge` (background + border) in the right panel header goes; the sentence
   becomes plain text with the formula set in a math serif face only.
3. **No A / B step badges** → drop `.step-num.step-a` / `.step-num.step-b` from the two
   `.part-label` rows of the right panel.
4. **More room for the AOP, larger fonts, larger containers if needed** → after deleting the
   bottom claim banner (~55 canvas px) the right panel can give the figure ~270 canvas px instead
   of 190, and the figure itself must be redesigned on a narrower canvas.
5. **"the purple container on the bottom is not visible on the slide, remove it"** → the bottom
   `.claim-banner` on slide 1 (cyan→purple gradient box). It is confirmed *rendered* in headless
   Firefox at 1400×734, but not on the author's screen (projector/viewport where the flex column
   pushes it below the fold). Rather than fight the viewport: delete it together with its CSS.

Pitfalls / boundary conditions:
- The two AOP figures (`.aop-fig-img` + `.aop-fig-overlay`) are absolutely stacked; they must share
  one canvas and one layout, otherwise the click-overlay jumps.
- Deleting the banner frees vertical space, but every enlargement (figure, video, fonts) is bounded
  by the panel box: any overflow silently hides `justify-content: space-between` content again.
- `slides/public` is a symlink to `../public`; only the real `public/` directory must be written.

## 2. Methodological & Scientific Considerations
- **Why redesign instead of enlarging the existing SVG?** The effective on-slide font size is
  `font_size × rendered_width / viewBox_width`. The old canvas is 1480 units wide and is rendered
  at ~716 px, so every label loses 52 % of its size. Scaling the picture up is impossible (the
  panel is width-bound at a 3.0:1 aspect ratio), therefore the canvas must shrink: the new figures
  use `viewBox="0 0 1180 420"` with 12.5–16 px labels → a 15 px label ≈ 9.1 canvas px ≈ 15.6 px on
  a 1920 projector, and every box is sized *from its own text* (see next point).
- **Fit-by-construction:** a `label()` helper measures each line with a conservative per-glyph
  width (0.60 em bold, 0.55 em regular), pads it by 20 px and **raises** when a label does not fit
  its box. The generator therefore cannot emit the "text wider than its rectangle" defect again.
- **AOP fidelity:** the figure keeps the OECD AOP 34 chain that the slide text and the speaker
  notes claim: MIE 167 (LXR activation) → intermediate effects (ChREBP, SREBP-1c, FAS, SCD1, CD36
  up-regulation) → *de novo* FA synthesis + increased FA influx → KE 291 (liver TG accumulation) →
  organelle damage → fatty liver cells → steatosis (AO, > 5–10 % liver weight). The "after" state
  adds exactly what a linear AOP cannot express — the adaptive homeostatic buffer
  (β-oxidation via PPARα/CPT1A, VLDL export via MTTP/APOB) and the bifurcation: within capacity →
  return to z_homeo, beyond capacity → tipping point. This is the visual argument for the
  attractor-to-ODE chain (M1.2 → M4).
- **Deleting the claim banner is not a scientific loss:** the same sentence lives on in the speaker
  notes ("Kortom: om een Adverse Outcome betrouwbaar te voorspellen…"), which is where the pitch
  actually says it.

## 3. Step-by-Step Execution Plan
1. Rewrite `public/generate_aop34_svg.py` so it emits **both** files from one parametrised layout
   (`variant="before" | "after"`), canvas 1180×420, with the fit assertion, a left gutter
   ("Biological organization level", "AOP from LXR"), the molecular/intermediate/key-event chain,
   three downstream columns (organelle / cellular / tissue) and a bottom band that holds either the
   "no feedback" note (before) or the blue homeostatic-buffer module (after). Write both SVGs into
   `public/` (which `slides/public` symlinks to).
2. `slides/slides.md`:
   - delete the bottom `.claim-banner` markup + its CSS;
   - fold the stray "1." into the left panel badge (symmetric with "2. …");
   - remove the `A`/`B` `.step-num` spans (and their now-dead CSS);
   - `.math-badge`: drop background/border/padding — plain inline formula, slightly larger;
   - `.aop-fig-stack`: fixed `aspect-ratio: 1180 / 420` with both layers absolutely positioned so
     the click-overlay stays pixel-aligned;
   - enlarge typography: `.part-title`, `.part-note`, `.video-card figcaption`, `.vn-body`,
     `.kw-title`, `.kw-desc`, `.qr-caption`; video 252×126 → 300×150; QR 140 → 150.
3. `slides/styles/index.css`: drop the now-unused `.claim-banner/.claim-icon/.claim-text` block.
4. Regenerate figures on GPU 1, `pnpm build`, then measure `verticalOverflowPx` and panel clipping
   with headless Firefox on both the dev server and the built bundle; screenshot slide 1.
5. Fill sections 4–6 with the real output.

## 4. Terminal Execution & Verification (GPU 1)
Generator (both figures from one layout, fit-by-construction):

```
$ CUDA_VISIBLE_DEVICES=1 uv run python public/generate_aop34_svg.py
== AOP 34 figures for slide 1 ==
  aop_linear_before.svg            9998 chars  viewBox 0 0 1180 420
  aop_homeostatic_after.svg       10496 chars  viewBox 0 0 1180 420
  every label was validated against its own rectangle (fit-by-construction)
EXIT=0
```

XML validity + node counts: `aop_linear_before 92 nodes, valid XML`,
`aop_homeostatic_after 95 nodes, valid XML`.

Browser-side proof that no label overflows its rectangle (inline the SVG, compare every
`getBBox()` with the smallest `<rect>` that contains its centre):

```
$ CUDA_VISIBLE_DEVICES=1 uv run --with selenium python /tmp/probe_fit.py http://localhost:3037/
[ {"file": "/aop_linear_before.svg",    "labelsInBoxes": 37, "overflow": []},
  {"file": "/aop_homeostatic_after.svg", "labelsInBoxes": 38, "overflow": []} ]
```

Layout probe (`/tmp/probe_s1.py`, dev server, 1400x734):

```
verticalOverflowPx: 0          banner: "removed"
grid: 82..630 (h 548)          gridScroll: scroll 532 / client 532
panel context-panel : clipped 0, lastBottom 617  (panel bottom 630)
panel kwestie-panel : clipped 0, lastBottom 617
aopStack: 162..415  w 713 h 253 (= 1180/420)
aopImgs: base    163..414 711x251 natural 1180x420 complete true
         overlay 163..414 711x251          -> pixel aligned
video: 441..589 296x148        mathBg: rgba(0, 0, 0, 0)   stepNums: 0
partTitles: ["AOP 34", "CORTICAL SPREADING DEPRESSION"]
```

Click state (one SPACE on slide 1):

```
{'cls': 'aop-fig-img aop-fig-overlay slidev-vclick-target slidev-vclick-current',
 'opacity': '1', 'w': 711, 'h': 251}
```

`pnpm build` -> `✓ built in 2.18s` (EXIT 0); `diff -q public/aop_*.svg slides/dist/aop_*.svg`
-> `dist in sync`. The built bundle served with an SPA fallback (`/tmp/spa_serve.py`) and probed at
1400x734, 1400x820, 1280x674 and 1920x1080: `verticalOverflowPx: 0` and both panels `clipped: 0`
at every size. Screenshots: `~/ffshot/s1_final_dev.png`, `s1_clicked.png`, `dist_s1_v2.png`,
`dist_s2_v2.png`.

## 5. Verification of Completed Tasks
- [x] `public/generate_aop34_svg.py` rewritten: emits **both** figures from one 1180x420 layout;
      `label()` / `items()` / `text_block()` raise instead of emitting a label that does not fit.
- [x] `public/aop_linear_before.svg` + `public/aop_homeostatic_after.svg` regenerated on the same
      canvas, so the click overlay stays pixel aligned (163..414 / 711x251 for both layers).
- [x] `slides/slides.md`: bottom `.claim-banner` deleted (`banner: "removed"`); stray `1.` folded
      into `1. New Approach Methodologies`; both A/B `.step-num` chips gone (`stepNums: 0`);
      `.math-badge` is plain inline text (`mathBg: rgba(0,0,0,0)`); `.aop-fig-stack` now uses
      `aspect-ratio: 1180 / 420` with both layers absolutely positioned (figure 190 -> 253 px tall,
      +33 %); typography enlarged (`.part-title` 0.66->0.82 rem, `.part-note` 0.56->0.70,
      `.vn-body` 0.52->0.62, `.vn-title` 0.58->0.68, `.vn-foot` 0.46->0.54, `.kw-title` 0.76->0.82,
      `.kw-desc` 0.66->0.72, figcaption 0.52->0.60) and the video card 252x126 -> 288x144.
- [x] Dead CSS removed from `slides/slides.md` (`.step-num`, `.step-a`, `.step-b`,
      `.claim-banner`, `.claim-icon`, `.claim-text`) and from `slides/styles/index.css`
      (`.claim-banner` block — no other slide used it).
- [x] No overflow anywhere: `verticalOverflowPx: 0`, `clipped: 0` on dev and on the built bundle at
      1280x674 / 1400x734 / 1400x820 / 1920x1080. Slide 2 re-checked in the bundle: unchanged.
- [x] `python -m py_compile public/generate_aop34_svg.py` -> OK.

## 6. Reflection & Next Step
- All five requests are implemented and *measured*, not assumed: the "text spills out of its
  rectangle" defect cannot come back because the generator refuses to emit a label wider than its
  box, and an independent browser probe confirms 0 overflow in both figures (37/38 labels checked).
- Deleting the banner freed ~55 canvas px; they went into the figure (+63 px tall, +23 % wide) and
  into larger typography. Nothing else moved and both panels still end 13 px above their edge.
- Side effects to know about: `public/slide1_pitch_preview.html` (legacy standalone preview, not
  linked from the deck) still contains the old banner markup; `uv.lock` and `slides/dist/` remain
  untracked artifacts; nothing has been committed yet.
- Immediate follow-up (unchanged): `public/csd_experiment.webm` still shows the v1 simulation,
  parts of slide 1 are still Dutch where untouched, and Milestone 1.2 (the Graphormer encoder) is
  the next real modelling step.


---

## 7. Revision (feedback: "de vorige versie van de AOP graaf was beter")
- **Decision:** the flat redesign of the AOP figure is rolled back; the *original* drawing (3D drop
  shadows, yellow/green/red table, dashed "no feedback" box, blue homeostatic column in the
  "after" state) is restored. `public/generate_aop34_svg.py`,
  `public/aop_linear_before.svg` and `public/aop_homeostatic_after.svg` are back at their `HEAD`
  content, so only the *canvas* and a few font sizes differ from the committed version.
- **Kept from the redesign:** everything around the figure (no banner, no A/B chips, bigger
  typography, `.aop-fig-stack` with a fixed aspect ratio + absolutely positioned layers, video and
  QR card enlarged) — the user confirmed the rest of the slides is good.
- **Overlay alignment:** the old pair did *not* share a canvas (`1480x490` vs `1490x490`), so the
  click-overlay was always ~0.7 % off. `aop_linear_before.svg` now uses `viewBox 0 0 1490 490`
  (its rightmost geometry sits at x = 1470, so the extra 10 units are pure margin and nothing
  shifts) and `.aop-fig-stack` uses `aspect-ratio: 1490 / 490`; both layers measure
  `163..395 / 711x232`, i.e. pixel aligned.
- **Text-inside-box fix, typography only (no design change).** Browser `getBBox()` vs containing
  rect on the restored files:
  - before: `Cytoplasm displacement` 152.3 in 150, `Mitochondrial disruption` 153.1 in 150,
    `Endoplasmic reticulum stress` **182.9 in 150** (16.5 px over) — reproduced in `after` too,
    plus `➔ Return to Homeostasis (AOP Prevented)` 200.2 in a 195-wide badge.
  - fix: those four organelle labels go from 11 px to 10.5 px and `Endoplasmic reticulum stress`
    is split over two baselines (`Endoplasmic` y=311 / `reticulum stress` y=331, box is 160..340);
    the badge sub-line goes from 9.5 px to 9 px (189.6 in 195). Same boxes, colours, shadows and
    wording, so the look is unchanged — the text no longer crosses the border.
  - re-measured: `aop_linear_before` 32 labels, **0 overflow**; the three left-overs reported for
    `aop_homeostatic_after` (`Organization Level`, `AOP from`, the IF badge) are heuristic false
    positives — those labels sit in the yellow/green *arrow polygons*, not in the 195x34 /
    185x30 rect the probe compares against (the IF badge text is 189.6 in a 195 rect = 2.7 px
    margin per side).
- **Verification:** generator EXIT 0 (`aop_homeostatic_after.svg` regenerated, byte-identical to
  `HEAD` except the three text edits); dev probe 1400x734 -> `verticalOverflowPx 0`,
  `clipped [0, 0]`, `lastBottom 598 < 630`, stack `713x234`, layers identical;
  `pnpm build` `✓ built in 2.19s`, `dist` SVGs in sync, dist probe `verticalOverflowPx 0`,
  `clipped [0, 0]` at 1280x674 / 1400x734 / 1920x1080. Screenshots:
  `~/ffshot/s1_revert_dev2.png`, `s1_revert_zoom2.png`, `s1_revert_clicked2.png`,
  `s1_revert_clicked_zoom.png`, `dist_s1_revert.png`.
- **Net effect vs the committed version:** the figure is 232 px tall instead of ~190 px (+22 %)
  because the deleted banner freed the vertical space, and no label crosses its border anymore.
- **Follow-up:** nothing is committed yet (`uv.lock` question still open); `public/slide1_pitch_preview.html`
  is still the stale legacy preview; `public/csd_experiment.webm` still shows the v1 simulation.

