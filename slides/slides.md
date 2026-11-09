---
theme: default
title: Multimodal en Multiscale AI for Toxicology
info: |
  ## Pitch (max. 3 minuten, 2 slides)
  Onderzoeksvoorstel: Multimodal en Multiscale AI for Toxicology
colorSchema: dark
transition: fade
aspectRatio: 16/9
canvasWidth: 1120
---

<!-- ⚠ ONDERHOUD: houd alle ruwe HTML in dit bestand FLUSH-LEFT (kolom 0).
     Slidev rendert met markdown-exit (html: true): een regel die na een lege regel
     4 of meer spaties inspringt wordt een "indented code block" en verschijnt dan als
     ge-escapte tekst (pre/code) in plaats van als layout.
     Zet dus geen markdown formatter met HTML-herinspringing (bv. Prettier format-on-save) op dit bestand.
-->

<div class="slide-container">
<!-- Header -->
<div class="header-bar">
<div>
<div class="eyebrow">Onderzoeksvoorstel Pitch · Innovative Testing</div>
<h1 class="main-title">Multimodal &amp; Multiscale AI for Toxicology</h1>
</div>
<div class="header-pill">Slide 1 / 2 · NAMs &amp; Beyond Linear AOPs</div>
</div>
<!-- Main 2-Column Grid -->
<div class="main-grid">
<!-- LEFT COLUMN: NEW APPROACH METHODOLOGIES -->
<div class="panel context-panel">
<div class="panel-header">
<span class="badge badge-blue">1. New Approach Methodologies (NAM's)</span>
</div>
<!-- NAM Image / Placeholder -->
<div class="nam-visual-wrap">
<img
:src="'/nam_overview.png'"
onerror="this.onerror=null;this.src='/nam_overview_placeholder.svg';"
alt="New Approach Methodologies (NAMs)"
class="nam-img"
/>
</div>
<!-- Minimal Keyword Cards -->
<div class="keyword-stack">
<div class="kw-card">
<div class="kw-dot bg-cyan"></div>
<div>
<div class="kw-title">Probabilistic Chemical Risk Assessment</div>
<div class="kw-desc">Accelerating the transition to human-relevant, animal-free biomedical innovation</div>
</div>
</div>
<div class="kw-card">
<div class="kw-dot bg-emerald"></div>
<div>
<div class="kw-title">Pillars: (q)AOPs &amp; Physiological Maps</div>
<div class="kw-desc">Mechanistically mapping and explaining <em>Adverse Outcomes</em></div>
</div>
</div>
</div>
</div>
<!-- RIGHT COLUMN: BEYOND LINEAR AOPS -->
<div class="panel kwestie-panel">
<div class="panel-header">
<span class="badge badge-rose">2. Expanding on Linear AOPs</span>
<span class="panel-subtitle">This research will deliver a toolbox for the development of quantitative AOP's. From linear chains to dynamic, quantitative systems predicting <span class="math-badge"><br>><em>P</em>(<em>z</em><sub>homeo</sub>, <em>z</em><sub>adversity</sub> | context, exposure, dose)</span></span>
</div>
<!-- DEEL A: AOP 34 -->
<div class="part">
<div class="part-label">
<span class="part-title">AOP 34</span>
<span class="part-note">LXR (MIE 167) → lipogenese → TG-accumulation (KE 291) → steatose</span>
</div>
<div class="part-a-body">
<div class="aop-fig-stack">
<img
:src="'/aop_linear_before.svg'"
alt="Lineair AOP 34"
class="aop-fig-img"
/>
<img
v-click="1"
:src="'/aop_homeostatic_after.svg'"
alt="AOP 34 met adaptieve homeostatische buffer"
class="aop-fig-img aop-fig-overlay"
/>
</div>
</div>
</div>
<!-- DEEL B: CSD EXPERIMENT + GROTE VIDEO + DIRECT SCANBARE QR CODE -->
<div class="part">
<div class="part-label">
<span class="part-title">Cortical Spreading Depression</span>
</div>
<div class="video-row">
<!-- Brede Video Card voor csd_experiment.webm (2:1 aspect ratio) -->
<figure class="video-card" onclick="var v=this.querySelector('video');this.classList.toggle('playing',v.paused);v.paused?v.play():v.pause();">
<video :src="'/csd_experiment.webm#t=0.05'" class="csd-video" muted loop playsinline preload="metadata"></video>
<div class="play-badge">▶</div>
<figcaption><strong>Breakthrough (left) vs. Propagation Failure (right)</strong><br/>Same MIE: tissue structure determines the fate</figcaption>
</figure>
<!-- Toelichting Tekst -->
<div class="video-note">
<div class="vn-title">Same MIE, different outcomes</div>
<div class="vn-foot">scan QR for live simulation</div>
</div>
<!-- QR Card: echte, door zxing-cpp gedecodeerde QR (bron: slides/make-qr.mjs) -->
<a href="https://raw.githack.com/MeesterDavid/Levertoxiciteit/main/cortical_spreading_depression_simulator.html" target="_blank" rel="noopener" class="qr-card">
<img class="qr-img" :src="'/qr_csd_simulator.png'" alt="QR-code naar de live CSD-simulator" width="140" height="140" />
<div class="qr-caption">
<span>Live Simulator</span>
</div>
</a>
</div>
</div>
</div>
</div>
</div>

<style>
.slidev-layout {
  padding: 1.1rem 1.5rem !important;
  background: radial-gradient(circle at 15% 20%, #0f172a 0%, #090d16 100%);
  color: #f8fafc;
  font-family: 'Inter', system-ui, sans-serif;
}
#slide-container > div:not(.slidev-slide-content) {
  display: none !important;
}
.slide-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 0.7rem;
}
.header-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #1e293b;
  padding-bottom: 0.45rem;
}
.eyebrow {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #38bdf8;
  font-weight: 700;
}
.main-title {
  font-size: 1.45rem !important;
  font-weight: 800 !important;
  margin: 0 !important;
  line-height: 1.15 !important;
  color: #f8fafc;
}
.header-pill {
  background: #1e293b;
  border: 1px solid #334155;
  color: #cbd5e1;
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.25rem 0.65rem;
  border-radius: 999px;
}
.main-grid {
  display: grid;
  grid-template-columns: 31% 67.5%;
  /* a definite row height is what lets the panels shrink; with the default `auto`
     row a long panel header overflows the grid box and ends up painted behind the
     claim banner (the banner stayed at y 584–630 while the cards ran to ~600) */
  grid-template-rows: minmax(0, 1fr);
  gap: 1.5%;
  flex: 1;
  min-height: 0;
}
.panel {
  background: rgba(15, 23, 42, 0.78);
  border: 1px solid #1e293b;
  border-radius: 14px;
  padding: 0.75rem 0.9rem;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 0;
}
.panel-header {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.4rem;
}
.badge {
  font-size: 0.68rem;
  font-weight: 800;
  padding: 0.18rem 0.55rem;
  border-radius: 6px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  flex: 0 0 auto;
  white-space: nowrap;
}
.badge-blue {
  background: rgba(56, 189, 248, 0.15);
  color: #38bdf8;
  border: 1px solid rgba(56, 189, 248, 0.35);
}
.badge-rose {
  background: rgba(244, 63, 94, 0.15);
  color: #fb7185;
  border: 1px solid rgba(244, 63, 94, 0.35);
}
.panel-subtitle {
  font-size: 0.8rem;
  color: #cbd5e1;
  font-weight: 600;
  line-height: 1.3;
}
/* Inline formula. KaTeX only runs on markdown text, never inside a raw HTML block,
   so the formula is real HTML: italic variables + cursive subscripts, set in a math
   font stack (no CDN, so the deck also works offline). No chip: the sentence around it
   is body text, so the formula must not sit in a coloured rectangle of its own. */
.math-badge {
  display: inline;
  font-family: 'Cambria Math', 'Latin Modern Math', Georgia, 'Times New Roman', serif;
  font-size: 0.84rem;
  color: #e2e8f0;
}

.math-badge em {
  font-style: italic;
}
.math-badge sub {
  font-size: 0.68em;
  font-style: italic;
  letter-spacing: 0.01em;
  color: #7dd3fc;
}
.nam-visual-wrap {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 0;
  margin: 0.25rem 0;
}
.nam-img {
  max-height: 215px;
  width: 100%;
  object-fit: contain;
  border-radius: 10px;
}
.keyword-stack {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}
.kw-card {
  display: flex;
  align-items: flex-start;
  gap: 0.55rem;
  background: #0b1120;
  border: 1px solid #1e293b;
  border-radius: 9px;
  padding: 0.45rem 0.65rem;
}
.kw-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  margin-top: 0.28rem;
  flex-shrink: 0;
}
.bg-cyan { background: #38bdf8; box-shadow: 0 0 8px #38bdf8; }
.bg-emerald { background: #10b981; box-shadow: 0 0 8px #10b981; }
.kw-title {
  font-size: 0.82rem;
  font-weight: 700;
  color: #f1f5f9;
}
.kw-desc {
  font-size: 0.72rem;
  color: #94a3b8;
  line-height: 1.3;
}
.panel.kwestie-panel {
  justify-content: flex-start;
  gap: 0.3rem;
}
.part {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-height: 0;
}
.part-label {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  flex-wrap: wrap;
  line-height: 1.2;
}
.part-title {
  font-size: 0.82rem;
  font-weight: 800;
  color: #f1f5f9;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.part-note {
  font-size: 0.7rem;
  color: #94a3b8;
  font-weight: 600;
}
.click-hint {
  margin-left: auto;
  font-size: 0.54rem;
  font-weight: 700;
  color: #fbbf24;
  background: rgba(251, 191, 36, 0.12);
  border: 1px solid rgba(251, 191, 36, 0.4);
  border-radius: 999px;
  padding: 0.04rem 0.4rem;
  animation: hint-pulse 1.8s ease-in-out infinite;
}
@keyframes hint-pulse {
  0%, 100% { opacity: 0.55; }
  50% { opacity: 1; }
}
.part-a-body {
  display: flex;
  align-items: stretch;
  gap: 0.4rem;
  min-height: 0;
}
.aop-fig-stack {
  position: relative;
  flex: 1;
  min-width: 0;
  /* aop_linear_before.svg and aop_homeostatic_after.svg share the same 1490 x 490 canvas
     (public/generate_aop34_svg.py regenerates the "after" one); a fixed aspect ratio plus
     absolutely positioned layers keeps the click-overlay pixel aligned, so the "after" state
     only *adds* the homeostatic buffer. */
  aspect-ratio: 1490 / 490;
  border-radius: 9px;
  border: 1px solid #334155;
  background: #0b0f19;
  overflow: hidden;
  line-height: 0;
}
.aop-fig-img {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.aop-fig-overlay {
  z-index: 1;
}
.video-row {
  display: flex;
  align-items: center;
  gap: 0.55rem;
}
.video-card {
  position: relative;
  margin: 0;
  flex: 0 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.15rem;
  cursor: pointer;
}
.csd-video {
  width: 288px;
  height: 144px;
  display: block;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid #334155;
  background: #020617;
  transition: border-color 0.2s ease, transform 0.15s ease;
}
.video-card:hover .csd-video {
  border-color: #fb7185;
}
.play-badge {
  position: absolute;
  top: 72px;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
  color: #f8fafc;
  background: rgba(2, 6, 23, 0.68);
  border: 1px solid rgba(248, 250, 252, 0.6);
  pointer-events: none;
  transition: opacity 0.2s ease;
}
.video-card.playing .play-badge {
  opacity: 0;
}
.video-card figcaption {
  font-size: 0.6rem;
  color: #cbd5e1;
  text-align: center;
  line-height: 1.2;
}
.video-card figcaption strong {
  color: #fda4af;
}
.video-note {
  flex: 1 1 0;
  min-width: 0;
  max-width: 240px;
  display: flex;
  flex-direction: column;
  gap: 0.18rem;
  border-left: 2px solid rgba(244, 63, 94, 0.5);
  padding-left: 0.45rem;
}
.vn-title {
  font-size: 0.68rem;
  font-weight: 800;
  color: #fda4af;
  text-transform: uppercase;
  letter-spacing: 0.02em;
}
.vn-body {
  font-size: 0.62rem;
  color: #cbd5e1;
  line-height: 1.3;
}
.vn-body strong {
  color: #f1f5f9;
}
.vn-foot {
  font-size: 0.54rem;
  color: #94a3b8;
  font-style: italic;
  margin-top: 0.1rem;
}
.qr-card {
  flex: 0 0 auto;
  margin-left: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #0b1120;
  border: 1px solid #334155;
  border-radius: 9px;
  padding: 0.28rem 0.5rem;
  text-decoration: none;
  transition: border-color 0.2s ease, transform 0.15s ease;
}
.qr-card:hover {
  border-color: #38bdf8;
  transform: translateY(-1px);
}
.qr-img {
  width: 140px;
  height: 140px;
  display: block;
  border-radius: 4px;
  background: #ffffff;
}
.qr-caption {
  margin-top: 0.22rem;
  font-size: 0.5rem;
  font-weight: 700;
  color: #f1f5f9;
  text-align: center;
  display: flex;
  flex-direction: column;
  line-height: 1.1;
}
.qr-sub {
  font-size: 0.5rem;
  color: #38bdf8;
  font-weight: 500;
}
</style>

<!--
SPREEKTEKST SLIDE 1 (± 100–110 seconden | ~250 woorden):

[1. NEW APPROACH METHODOLOGIES — Links op de slide]
"Goedemiddag. Binnen ons vakgebied werken we aan de transitie naar Probabilistic Chemical Risk Assessment met behulp van New Approach Methodologies — oftewel NAMs. We combineren daarin geavanceerde humane in vitro en in silico modellen om chemische veiligheidsbeoordeling volledig dierproefvrij én mechanistisch nauwkeurig te maken. Twee belangrijke pijlers hierin zijn Adverse Outcome Pathways — (q)AOP's — en Physiological Maps, waarmee we vastleggen en verklaren hoe een blootstelling leidt tot een Adverse Outcome."

[2. BEYOND LINEAR AOPS — Rechts op de slide, in twee delen]
"Maar daar ligt een fundamentele wetenschappelijke kwestie, en die splits ik in twee delen.

Deel A — lineair. AOP 34, LXR-activatie die tot steatose leidt, is sterk reductionistisch: MIE, key events, adverse outcome. [klik] Maar zo'n AOP is in de basis lineair, en dat is precies het probleem. In de cel is de triglyceride-accumulatie namelijk eerst een reversibele buffer, die actief wordt leeggehouden door beta-oxidatie — PPARalpha en CPT1A — en door VLDL-export via MTTP en APOB. Zolang de perturbatie, dus LXR-gedreven de-novo-lipogenese en CD36-gemedieerde instroom, binnen die dynamische klaringscapaciteit blijft, veert het systeem terug naar homeostase: een adaptieve homeostatische attractor. Pas als de perturbatie groter wordt dan die capaciteit, kantelt het systeem naar een pathologische attractor: lipotoxiciteit, celdood en steatose. Dat kantelpunt — die bifurcatie — kan een lineair AOP niet voorspellen, en dat moeten we kwantitatief modelleren.

Deel B — niet-lineair weefselgedrag. Soms vertelt de Molecular Initiating Event op zichzelf onvoldoende of de Adverse Outcome daadwerkelijk optreedt. Neem Cortical Spreading Depression: we kennen de MIE — ionkanaal-verstoring of kaliumophoping — maar de uitkomst ontstaat pas door ruimtelijk-temporele emergentie in het weefsel, meetbaar met bijvoorbeeld Multi-Electrode Arrays.

[klik op de video csd_experiment]
Hier zien we twee simulaties met exact dezelfde centrale startpuls. In kwetsbaar weefsel met weinig gliale buffers breekt de golf door en propageert over de cortex. In robuust weefsel met voldoende actieve astrocyten dooft de stimulus direct uit. U kunt met de QR-code rechts direct zelf interactief met deze parameters en weefselstructuur spelen in de simulator.

Kortom: om een Adverse Outcome betrouwbaar te voorspellen, moeten we de adaptieve attractor én de emergente weefseldynamiek samen met de (q)AOP's en Physiological Maps meenemen. Mijn centrale claim in dit onderzoeksvoorstel is dat juist Multimodal en Multiscale AI bij uitstek geschikt is om deze enorme diversiteit aan data en biologische schalen wiskundig samen te brengen."
-->

---

<div class="slide-container">
<!-- Header: only the central dogma, deliberately without eyebrow or pill -->
<div class="header-bar">
<h1 class="main-title dogma-title">Homeostasis or Adversity?</h1>
</div>
<!-- Demo case, in one line -->
<div class="sub-claim">
<span class="sub-claim-tag">A demo case</span>
<span>a <strong>multimodal &amp; multiscale model</strong> that fuses chemical, molecular and physiological data into one representation — and turns it into <strong>probabilities</strong>, not a label.</span>
</div>
<!-- Central figure: 5 data sources -> 1 AI node -> 2 answers (solid = built, dashed = future work) -->
<figure class="figure-card">
<img :src="'/s2_multimodal_ai_graph.svg'" alt="Five data sources (chemical structure and dose, protein interactions, gene expression, gene regulatory interactions, and as future work physiological MEA recordings) feed one multimodal and multiscale AI model; it returns the predicted MIE class of a compound from structure alone (solid arrow, built and validated) and, as the next step, the probability of z-adversity given context, exposure and dose via Neural ODEs (dashed arrow). A legend marks solid green as done and dashed purple as future work." class="pipeline-figure" />
</figure>
</div>

<!--
SPEAKER NOTES SLIDE 2 (± 90–100 seconds | ~230 words):

[1. TITLE — homeostasis or adversity?]
"Homeostasis or adversity? That is the question this proposal answers. We do not simply predict 'toxic' or 'non-toxic'. We predict the fate of the liver cell itself: does it spring back to the homeostatic attractor z-homeo, or does it tip over into the pathological attractor z-adversity — and then on to z-steatosis or to z-cholestasis? That tipping point is the only thing that biologically matters, and it is exactly what a linear AOP cannot give us."

[2. FIVE DATA SOURCES — left]
"This is the demo case: one multimodal and multiscale model, and it takes in five data sources. At the top — with a solid green arrow, because it is the modality we already use in our baseline today — is chemical information: SMILES converted into 2D OGB molecular graphs and SELFIES sequences for twelve reference compounds across three MIE classes. Below it — with dashed purple arrows for the next steps — are the three biological modalities: protein-protein interactions from STRING-db (eighteen genes with a mean interaction score of 0.81), gene expression from GTEx, the Human Protein Atlas, GEO and HepaRG (eighteen genes across three states: healthy, steatosis and cholestasis), and gene-gene interactions from OECD AOP 34 and 27 plus the Physiological Maps (twenty directed interactions, fourteen activating and six repressing, summarised in a 19-by-19 shortest-path matrix). The fifth entry, also dashed, is the one thing that no molecule or annotation can give us: physiological measurements from multi-electrode arrays — that is where tissue-level emergent behaviour such as cortical spreading depression enters, and it is explicitly future work."

[3. THE FUSION NODE — middle]
"Everything converges in that single node: Microsoft Graphormer with 47.7 million parameters plus ToxFormer with 25.4 million, extended with a final classification layer that fuses their 3-class probability distributions 50/50."

[4. TWO ANSWERS — right]
"Then the two answers. The solid green arrow is what we have today: a structure-only classifier that predicts which MIE class a compound belongs to — LXR activation, BSEP inhibition, or neither — from its 2D molecular graph and SELFIES representation alone, without any biological context. We report that the honest way: leave-one-out cross-validation on the 50/50 dual consensus, without any hand-crafted RDKit descriptors. Ten of the twelve compounds land on the green diagonal, with a mean probability of 0.52 for the true class. The confusion matrix next to it shows where the models agree and where uncertainty remains: all four LXR activators, three of the four BSEP inhibitors, and three of the four controls are recovered, while only Chlorpromazine and Fenofibrate are confused. And that is the point: we can measure the uncertainty — which is exactly what probabilistic risk assessment asks for.
The dashed purple arrow — labelled Neural ODEs in the figure — is the next step: the probability of z-adversity given context, exposure and dose. For that we are now coupling the three biological modalities from milestone 1.2 to 4: the Graphormer encoder on the genes, the perturbation encoder, the Neural ODE over twenty-four hours, and finally the bifurcation that gives the point of departure. A single look at the legend tells the panel which half of this figure already runs on GPU 1 and which half is the proposal."
-->