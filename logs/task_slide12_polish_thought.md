# Thought & Progress Log: Slide 1–2 polish (LaTeX, claim banner, z-notation)

- **Date / Time:** 2026-11-06 (session 3)
- **Task Goal:** Fix the visual defects reported on slide 1 (unrendered LaTeX + missing/overlapped
  bottom claim) and apply four copy/layout changes on slide 2 (title, demo-case sub-claim,
  `z_adversity` notation with cursive subscripts, remove the bottom claim banner).

## 1. Analysis & Problem Statement

### Slide 1
1. **LaTeX does not render.** Line 70 of `slides/slides.md` contains
   `<span class="math-badge">$P(Z_{\text{homeo}}, ... \mid ...)$</span>`. Slidev's KaTeX plugin is a
   *markdown* plugin: the formula sits inside a raw HTML block (`html_block` token), which markdown-it
   copies verbatim, so no inline math is parsed. There is also no `.math-badge` rule anywhere, and no
   `katex:` key in the headmatter, so the user sees the raw `$P(Z_{\text{homeo}}...` string.
   *Boundary condition:* a pitch may be given offline → do **not** introduce a KaTeX CDN dependency;
   build the formula from HTML (`<em>` + `<sub>`) and style it like math.
2. **The bottom claim banner of slide 1 is pushed out / overlapped.** Reproduced with Selenium:
   `.claim-banner` = y 584–630 (height 46) but `.main-grid` content continues underneath it (the
   keyword cards and the CSD caption bleed through). Root cause: `.main-grid` is `flex: 1` inside a
   definite-height column, but its single implicit grid row is `auto`, so the row is sized to the
   *content* and simply overflows the grid box (the panels never shrink). The unrendered LaTeX made
   the panel header 3 lines high, which triggered it. `verticalOverflowPx` stayed 0 because the spill
   is painted *behind* the opaque banner.

### Slide 2
3. Title `z_homeo or z_adversity` should read **"Homeostasis or Adversity?"**.
4. The sub-claim pill said "The solution"; it must become a **demo case**: "A demo case: a multimodal
   & multiscale model that fuses protein, gene, and chemical data into one representation — and turns
   it into probabilities, not a label."
5. `P(adversity | context, exposure, dose)` must become `P(z_adversity | …)` and, in general, the
   subscript of the z-notation (`homeo`, `adversity`, `steatosis`, `cholestasis`) must be **cursive /
   italic**, while the `z` itself stays upright.
6. The whole bottom claim banner ("Today a validated P(MIE | structure) …") must be **deleted**.

## 2. Methodological & Scientific Considerations
- The LaTeX is replaced by semantic HTML that mirrors KaTeX's look (`<em>` variable + `<sub>` index, a
  serif math font stack), because the figure and slide text already use the same `z<sub>…</sub>`
  convention as the SVG figure (Milestone 0/1.1 contract: `z_homeo`, `z_steatosis`, `z_cholestasis` as
  `[1, 64]` anchors). One notation, three renderers (SVG, Slidev, notes) — no drift.
- Italic subscripts are the typographic convention for indices (`z_homeo` = the *homeo* attractor),
  which also makes `z_adversity` unambiguous next to the word "adversity" in running text.
- The overflow fix is structural (`grid-template-rows: minmax(0, 1fr)`), not cosmetic padding surgery:
  a definite row lets the existing `min-height: 0` flex children shrink, so the banner can never be
  covered again regardless of how long the subtitle becomes.
- Deleting the slide-2 banner returns the freed vertical space to the figure (`.figure-card` is
  `flex: 1`), which makes the SVG larger on the projector — the intended effect.

## 3. Step-by-Step Execution Plan
1. `generate_slide2_diagram.py`: `sub()` gains `font-style="italic"`; rose card title becomes
   `P(z_adversity | context, exposure, dose)`; body text uses `z_adversity` / `z_homeo`; docstrings and
   the `aria-label` follow. Regenerate `public/s2_multimodal_ai_graph.svg`.
2. `slides/slides.md` slide 1: replace the raw `$…$` with the HTML formula chip; add
   `grid-template-rows: minmax(0, 1fr)` to `.main-grid`, `min-height: 0` to `.panel`, `flex: 0 0 auto;
   white-space: nowrap` to `.badge`, and a new `.math-badge` rule.
3. `slides/slides.md` slide 2: new title, new sub-claim ("A demo case"), delete the claim banner,
   update the alt text and the two speaker-note paragraphs that quote the old wording.
4. `slides/styles/index.css`: `.slidev-layout sub { font-style: italic }` so every z-index renders
   cursive; keep `.dogma-title` as is.
5. Verify on GPU 1: regenerate the SVG, `pnpm build`, Selenium layout measurements on `/1` and `/2`
   (banner must sit above the grid content, no spill) + screenshots.

## 4. Terminal Execution & Verification (GPU 1)

### 4.1 Figure regeneration
```
$ CUDA_VISIBLE_DEVICES=1 uv run python generate_slide2_diagram.py
EXIT=0
== Slide 2 - multimodal & multiscale AI graph ==
   figure           : public/s2_multimodal_ai_graph.svg (12,319 bytes)
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
(the generator prints the length of the SVG *string* — 12,319 chars, i.e. 12,356 bytes on disk, the
difference being the UTF-8 characters `α`, `·` and `→`.)
SVG is valid XML (`xml.etree.ElementTree`, 95 nodes), and the new notation is physically in the file:
```
z<tspan dy="2.8" font-size="7.4" font-style="italic">adversity</tspan>
z<tspan dy="2.4" font-size="7.6" font-style="italic">adversity</tspan>
```

### 4.2 Production build
```
$ pnpm build
EXIT=0   ✓ built in 2.00s
```
`dist/assets/slidev/md-*.js` contains `Homeostasis or Adversity?`; the old banner string
`Today</strong> a validated` is **absent**; `math-badge` is compiled into `dist/assets/md-*.css`.

### 4.3 Layout measurement (headless Firefox, dev server `:3037`, viewport 1400×734)

Before the fix (slide 1): `.main-grid` bottom 630 while the keyword cards ran to ~600 **behind** the
banner at 584–630 → the banner text collided with the "Pillars: (q)AOPs & Disease Maps" card, and the
raw `$P(Z_{\text{homeo}}...` occupied two lines.

After the fix:
```
slide 1: verticalOverflowPx 0 | grid 93–640 | banner 662–713 (1400x820) / 584–630 (1400x734)
         .panel.context-panel   clipped 0
         .panel.kwestie-panel   clipped 0 (scrollHeight == clientHeight == 467)
         QR overhang into the padding: 3 px (well inside the panel border)
         .math-badge innerText "P(zhomeo, zadversity | context, exposure, dose)"
         font-family "Cambria Math, Latin Modern Math, Georgia, Times New Roman, serif"
slide 2: verticalOverflowPx 0 | banner null | figure 1103x522 | imgNatural 1040x406
built bundle (dist via :3038, 1400×734): slide 1 overflow 0, slide 2 overflow 0, banner null
```
Screenshots: `/home/disaacspat/ffshot/s1_fix3.png` (dev, slide 1), `s2_fix1.png` (dev, slide 2),
`dist_s1.png` / `dist_s2.png` (built bundle).

## 5. Verification of Completed Tasks

| # | Request | Status |
|---|---------|--------|
| 1 | Slide 1 bottom claim must be visible again | ✅ banner now sits below the grid (662 / 584) with no overlap; both panels `clipped 0` |
| 2 | LaTeX must render properly | ✅ raw `$…$` removed; HTML chip renders `P(z_homeo, z_adversity | context, exposure, dose)` in a math serif stack, offline-safe |
| 3 | Slide 2 title "Homeostasis or Adversity?" | ✅ `<h1 class="main-title dogma-title">Homeostasis or Adversity?</h1>` |
| 4 | Sub-claim becomes "A demo case: …" | ✅ tag "A demo case" + "a multimodal & multiscale model that fuses protein, gene, and chemical data into one representation — and turns it into probabilities, not a label." |
| 5 | `P(adversity …)` → `P(z_adversity …)`, cursive subscripts generally | ✅ SVG (`sub()` → `font-style="italic"`), rose card, chips, plus `.slidev-layout sub { font-style: italic }` |
| 6 | Delete slide 2 bottom banner | ✅ block removed; bundle has no `Today … a validated` string; `banner: null` |

Files created/modified: `generate_slide2_diagram.py`, `public/s2_multimodal_ai_graph.svg`,
`slides/slides.md`, `slides/styles/index.css`, `logs/task_slide12_polish_thought.md`.
All of them were re-read after the edits and exist with the expected content (verified via `grep`,
`git status --porcelain`, XML parse and the two rendered slide screenshots).

## 6. Reflection & Next Step

Everything asked for is implemented and verified on both the dev server and the production bundle.
Two things worth knowing:
- The formula is **not** KaTeX. Slidev only runs its math plugin on markdown text, never inside a raw
  HTML block (that is why the old `$…$` stayed literal), and a CDN `katex:` stylesheet would break an
  offline pitch. The chip is therefore real HTML (`<em>` + `<sub>`) in a math font stack — same
  notation as the SVG figure and the notes.
- The slide-1 grid now uses `grid-template-rows: minmax(0, 1fr)`, so a long panel header can never push
  the claim banner out again; the price was ~30 px of media (AOP figure 200→190, CSD video 270×135 →
  252×126, QR 156→140) plus slightly tighter gaps, which keeps everything inside the panels.
- Follow-up: slide 1 is still Dutch (title, eyebrow, cards) — translating it would need the same
  treatment as slide 2; and `public/csd_experiment.webm` still shows the v1 simulation.
