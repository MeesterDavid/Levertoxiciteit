---
theme: default
title: Multimodal en Multiscale AI in de Toxicologie
info: |
  ## Pitch (max. 3 minuten, 2 slides)
  Onderzoeksvoorstel: Multimodal en Multiscale AI in de Toxicologie
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
<h1 class="main-title">Multimodal &amp; Multiscale AI in de Toxicologie</h1>
</div>
<div class="header-pill">Slide 1 / 2 · NAMs &amp; Beyond Linear AOPs</div>
</div>
<!-- Main 2-Column Grid -->
<div class="main-grid">
<!-- LEFT COLUMN: NEW APPROACH METHODOLOGIES -->
<div class="panel context-panel">
<div class="panel-header">
1. <span class="badge badge-blue">New Approach Methodologies</span>
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
<div class="kw-desc">Van dierproeven naar humane <em>in vitro</em> &amp; <em>in silico</em> voorspelling</div>
</div>
</div>
<div class="kw-card">
<div class="kw-dot bg-emerald"></div>
<div>
<div class="kw-title">Pijlers: (q)AOP's &amp; Physiological Maps</div>
<div class="kw-desc">Mechanistisch vastleggen en verklaren van <em>Adverse Outcomes</em></div>
</div>
</div>
</div>
</div>
<!-- RIGHT COLUMN: BEYOND LINEAR AOPS -->
<div class="panel kwestie-panel">
<div class="panel-header">
<span class="badge badge-rose">2. Beyond Linear AOPs</span>
<span class="panel-subtitle">From qualitative linear chains to dynamic, quantitative systems</span>
</div>
<!-- DEEL A: AOP 34 -->
<div class="part">
<div class="part-label">
<span class="step-num step-a">A</span>
<span class="part-title">AOP 34</span>
<span class="part-note">LXR (MIE 167) → lipogenese → TG-accumulatie (KE 291) → steatose</span>
<span v-click.hide="1" class="click-hint">klik ▸</span>
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
<!-- DEEL B: CSD EXPERIMENT + GROTE VIDEO + VEILIGE QR CODE -->
<div class="part">
<div class="part-label">
<span class="step-num step-b">B</span>
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
<div class="vn-body">
Both tissues initiate with an <strong>identical central potassium perturbation (MIE)</strong>. The outcome only arises through <strong>spatiotemporal emergence</strong>: the local density and buffering capacity of astrocytes (yellow cells) dictate whether the tissue recovers or the wave propagates across the cortex.
</div>
<div class="vn-foot">scan QR for live simulation</div>
</div>
<!-- QR Card met base64-veilige SVG -->
<a href="https://github.com/MeesterDavid/Levertoxiciteit/blob/main/cortical_spreading_depression_simulator.html" target="_blank" class="qr-card">
<img :src="'data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 33 33\' fill=\'%230B1120\'><rect width=\'33\' height=\'33\' rx=\'2\'/><path fill=\'%23F8FAFC\' d=\'M3 3h7v7H3V3zm2 2v3h3V5H5zM3 23h7v7H3v-7zm2 2v3h3v-3H5zM23 3h7v7h-7V3zm2 2v3h3V5h-3z\'/><path fill=\'%2338BDF8\' d=\'M12 3h2v3h-2zm5 0h1v2h-1zm3 0h1v4h-1zm-6 4h3v1h-3zm-2 2h1v2h-1zm7 0h2v1h-2zm-5 2h2v2h-2zm4 0h1v1h-1zm3 0h2v3h-1v-2h-1zm-9 2h1v2h-1zm3 1h2v1h-2zm-5 2h3v1H7zm5 0h1v3h-1zm7 0h3v1h-3zm-9 2h2v1h-2zm5 0h2v2h-1v-1h-1zm-7 2h1v1H6zm4 0h1v3h-1zm4 0h3v1h-3zm-5 2h2v1H9zm6 0h2v2h-1v-1h-1zm3-7h1v2h-1zm2 1h1v3h-1zm-2 3h1v2h-1zm3 1h1v1h-1z\'/></svg>'" alt="QR Code" class="qr-img" />
<div class="qr-caption">
<span>Live Simulator</span>
<span class="qr-sub">Scan of klik</span>
</div>
</a>
</div>
</div>
</div>
</div>
<!-- Bottom Claim Banner -->
<div class="claim-banner">
<div class="claim-icon">✦</div>
<div class="claim-text">
<em>Adverse Outcomes</em> are not linear and not always predictable from a purely reductionist approach. Some adverse outcomes depend on <em>emergent dynamics</em>. This is why we need <strong>Multimodal &amp; Multiscale AI
</strong>.
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
  font-size: 0.75rem;
  color: #cbd5e1;
  font-weight: 600;
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
  max-height: 205px;
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
  font-size: 0.76rem;
  font-weight: 700;
  color: #f1f5f9;
}
.kw-desc {
  font-size: 0.66rem;
  color: #94a3b8;
  line-height: 1.25;
}
.panel.kwestie-panel {
  justify-content: flex-start;
  gap: 0.42rem;
}
.part {
  display: flex;
  flex-direction: column;
  gap: 0.28rem;
  min-height: 0;
}
.part-label {
  display: flex;
  align-items: center;
  gap: 0.38rem;
  flex-wrap: wrap;
  line-height: 1.15;
}
.step-num {
  font-size: 0.56rem;
  font-weight: 900;
  width: 15px;
  height: 15px;
  border-radius: 4px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.step-a {
  background: rgba(56, 189, 248, 0.18);
  color: #7dd3fc;
  border: 1px solid rgba(56, 189, 248, 0.45);
}
.step-b {
  background: rgba(244, 63, 94, 0.18);
  color: #fda4af;
  border: 1px solid rgba(244, 63, 94, 0.45);
}
.part-title {
  font-size: 0.66rem;
  font-weight: 800;
  color: #f1f5f9;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.part-note {
  font-size: 0.56rem;
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
  border-radius: 9px;
  border: 1px solid #334155;
  background: #0b0f19;
  overflow: hidden;
  line-height: 0;
}
.aop-fig-img {
  display: block;
  width: 100%;
  height: auto;
  max-height: 200px;
  object-fit: contain;
}
.aop-fig-overlay {
  position: absolute;
  inset: 0;
  height: 100%;
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
  width: 270px;
  height: 135px;
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
  top: 67px;
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
  font-size: 0.52rem;
  color: #cbd5e1;
  text-align: center;
  line-height: 1.15;
}
.video-card figcaption strong {
  color: #fda4af;
}
.video-note {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.18rem;
  border-left: 2px solid rgba(244, 63, 94, 0.5);
  padding-left: 0.45rem;
}
.vn-title {
  font-size: 0.58rem;
  font-weight: 800;
  color: #fda4af;
  text-transform: uppercase;
  letter-spacing: 0.02em;
}
.vn-body {
  font-size: 0.52rem;
  color: #cbd5e1;
  line-height: 1.25;
}
.vn-body strong {
  color: #f1f5f9;
}
.vn-foot {
  font-size: 0.46rem;
  color: #94a3b8;
  font-style: italic;
  margin-top: 0.1rem;
}
.qr-card {
  flex: 0 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #0b1120;
  border: 1px solid #334155;
  border-radius: 9px;
  padding: 0.35rem 0.45rem;
  text-decoration: none;
  transition: border-color 0.2s ease, transform 0.15s ease;
}
.qr-card:hover {
  border-color: #38bdf8;
  transform: translateY(-1px);
}
.qr-img {
  width: 76px;
  height: 76px;
  display: block;
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
  font-size: 0.44rem;
  color: #38bdf8;
  font-weight: 500;
}
.claim-banner {
  margin-top: 0.45rem;
  background: linear-gradient(90deg, rgba(56, 189, 248, 0.14) 0%, rgba(168, 85, 247, 0.16) 100%);
  border: 1px solid rgba(168, 85, 247, 0.4);
  border-radius: 9px;
  padding: 0.42rem 0.7rem;
  display: flex;
  align-items: center;
  gap: 0.55rem;
}
.claim-icon {
  font-size: 1rem;
  color: #c084fc;
}
.claim-text {
  font-size: 0.7rem;
  color: #f1f5f9;
  line-height: 1.3;
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
layout: center
class: text-center
---

# Slide 2 — Aanpak, CV & Motivatie

*(Wordt in de volgende stap uitgewerkt)*