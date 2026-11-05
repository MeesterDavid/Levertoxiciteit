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
<div class="header-pill">Slide 1 / 2 · Context &amp; De Kwestie</div>
</div>
<!-- Main 2-Column Grid -->
<div class="main-grid">
<!-- LEFT COLUMN: DE CONTEXT -->
<div class="panel context-panel">
<div class="panel-header">
<span class="badge badge-blue">1. De Context</span>
<span class="panel-subtitle">Dierproefvrije &amp; Probabilistische Risicobeoordeling</span>
</div>
<!-- NAM Image / Placeholder -->
<div class="nam-visual-wrap">
<!-- Zet je eigen beeld als /public/nam_overview.png; ontbreekt het, dan valt de slide terug op de SVG-placeholder.
     Bewust een Vue-binding (:src) i.p.v. een statisch src-attribuut: Vite probeert een statisch src-pad
     als build-asset te resolven en faalt hard als het bestand (nog) niet bestaat. -->
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
<!-- RIGHT COLUMN: DE KWESTIE (CONVERGENCE HUB TO CSD) -->
<div class="panel kwestie-panel">
<div class="panel-header">
<span class="badge badge-rose">2. De Kwestie</span>
<span class="panel-subtitle">Van Reductionisme naar Multiscale Emergentie &amp; AI-Integratie</span>
</div>
<!-- Organization / Hub Diagram: AOP + Phys Map + Reaction-Diffusion -> CSD -->
<div class="hub-canvas">
<!-- SVG Connector Arrows pointing to CSD in the center -->
<svg class="hub-arrows" viewBox="0 0 640 340" preserveAspectRatio="none">
<defs>
<marker id="arr-cyan" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
<path d="M 0 1 L 10 5 L 0 9 z" fill="#38BDF8" />
</marker>
<marker id="arr-emerald" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
<path d="M 0 1 L 10 5 L 0 9 z" fill="#10B981" />
</marker>
<marker id="arr-purple" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
<path d="M 0 1 L 10 5 L 0 9 z" fill="#A855F7" />
</marker>
</defs>
<!-- Top-left (AOP) -> Center (CSD) -->
<line x1="205" y1="82" x2="252" y2="128" stroke="#38BDF8" stroke-width="2.8" stroke-dasharray="5,3" marker-end="url(#arr-cyan)" />
<!-- Bottom-left (Physiological Map) -> Center (CSD) -->
<line x1="205" y1="258" x2="252" y2="212" stroke="#10B981" stroke-width="2.8" stroke-dasharray="5,3" marker-end="url(#arr-emerald)" />
<!-- Right (Reaction-Diffusion + MEA) -> Center (CSD) -->
<line x1="435" y1="170" x2="388" y2="170" stroke="#A855F7" stroke-width="2.8" stroke-dasharray="5,3" marker-end="url(#arr-purple)" />
</svg>
<!-- Node 1: Top-Left — (q)AOP -->
<div class="spoke-card spoke-aop">
<div class="spoke-tag text-cyan">(q)AOP · Moleculair</div>
<!-- Optioneel eigen beeld: /public/aop_mini.png (valt anders terug op /public/aop_mini.svg) -->
<img
:src="'/aop_mini.png'"
onerror="this.onerror=null;this.src='/aop_mini.svg';"
alt="AOP Mini"
class="spoke-img"
/>
<div class="spoke-caption">
Sterk reductionistisch, maar <strong>lineair</strong><br/>
<span class="sub-note">(MIE = Ion channel disruption)</span>
</div>
</div>
<!-- Node 2: Bottom-Left — Physiological Map -->
<div class="spoke-card spoke-phys">
<div class="spoke-tag text-emerald">Physiological Map · Cel</div>
<!-- Optioneel eigen beeld: /public/physiological_map.png (valt anders terug op /public/physiological_map_mini.svg) -->
<img
:src="'/physiological_map.png'"
onerror="this.onerror=null;this.src='/physiological_map_mini.svg';"
alt="Physiological Map Mini"
class="spoke-img"
/>
<div class="spoke-caption">
<strong>Non-lineaire</strong> feedbackloops &amp; homeostase<br/>
<span class="sub-note">(ook weefsel-IE zoals asbest)</span>
</div>
</div>
<!-- CENTER HUB: CSD -->
<div class="center-csd-card">
<div class="csd-halo"></div>
<!-- Optioneel eigen beeld: /public/csd_center.png (valt anders terug op /public/csd_center_placeholder.svg) -->
<img
:src="'/csd_center.png'"
onerror="this.onerror=null;this.src='/csd_center_placeholder.svg';"
alt="Cortical Spreading Depression (CSD)"
class="csd-img"
/>
<div class="csd-label">CSD</div>
<div class="csd-sublabel">Emergente Adverse Outcome</div>
</div>
<!-- Node 3: Right — Reaction-Diffusion GIF & MEA Data -->
<div class="spoke-card spoke-rd">
<div class="spoke-tag text-purple">Reaction-Diffusion &amp; MEA · Weefsel</div>
<img
src="/reaction_diffusion.gif"
alt="Reaction-Diffusion Wave & MEA"
class="spoke-img rd-gif"
/>
<div class="spoke-caption">
<strong>Tissue-level emergence</strong><br/>
<span class="sub-note">Ruimtelijk-temporele MEA-data</span>
</div>
</div>
</div>
<!-- Bottom Claim Bar -->
<div class="claim-banner">
<div class="claim-icon">✦</div>
<div class="claim-text">
<strong>De Claim:</strong> Multimodale &amp; multiscale data (<em>AOP + Physiological Maps + MEA</em>) vereist <strong>AI</strong> om schalen te verbinden en het kantelpunt naar een <em>Adverse Outcome</em> te voorspellen.
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
  grid-template-columns: 36% 62.5%;
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
  font-size: 0.78rem;
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

/* Hub Diagram Styles */
.hub-canvas {
  position: relative;
  flex: 1;
  min-height: 315px;
  background: #090d16;
  border: 1px solid #1e293b;
  border-radius: 12px;
  overflow: hidden;
}
.hub-arrows {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 1;
}
.spoke-card {
  position: absolute;
  width: 196px;
  background: rgba(15, 23, 42, 0.94);
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 0.35rem 0.45rem;
  z-index: 2;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}
.spoke-aop {
  top: 10px;
  left: 10px;
  border-color: rgba(56, 189, 248, 0.45);
}
.spoke-phys {
  bottom: 10px;
  left: 10px;
  border-color: rgba(16, 185, 129, 0.45);
}
.spoke-rd {
  top: 50%;
  right: 10px;
  transform: translateY(-50%);
  width: 198px;
  border-color: rgba(168, 85, 247, 0.55);
}
.spoke-tag {
  font-size: 0.62rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin-bottom: 0.2rem;
}
.text-cyan { color: #38bdf8; }
.text-emerald { color: #10b981; }
.text-purple { color: #c084fc; }
.spoke-img {
  width: 100%;
  height: 78px;
  object-fit: contain;
  border-radius: 6px;
}
.rd-gif {
  height: 92px;
  object-fit: cover;
  border: 1px solid rgba(168, 85, 247, 0.4);
}
.spoke-caption {
  font-size: 0.62rem;
  color: #e2e8f0;
  line-height: 1.2;
  margin-top: 0.22rem;
}
.sub-note {
  font-size: 0.56rem;
  color: #94a3b8;
}

/* Center CSD Node */
.center-csd-card {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 138px;
  background: rgba(15, 23, 42, 0.96);
  border: 2px solid #f43f5e;
  border-radius: 14px;
  padding: 0.4rem;
  z-index: 3;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  box-shadow: 0 0 28px rgba(244, 63, 94, 0.28);
}
.csd-img {
  width: 112px;
  height: 96px;
  object-fit: contain;
  border-radius: 8px;
}
.csd-label {
  font-size: 1.05rem;
  font-weight: 900;
  color: #f8fafc;
  letter-spacing: 0.06em;
  margin-top: 0.15rem;
  line-height: 1.1;
}
.csd-sublabel {
  font-size: 0.56rem;
  color: #fda4af;
  font-weight: 600;
}

/* Bottom Claim Banner */
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
SPREEKTEKST SLIDE 1 (± 90–100 seconden | ~210 woorden):

[1. DE CONTEXT — Links op de slide]
"Goedemiddag. Binnen ons vakgebied werken we aan de transitie naar Probabilistic Chemical Risk Assessment met behulp van New Approach Methodologies — oftewel NAMs. We combineren daarin geavanceerde humane in vitro en in silico modellen om chemische veiligheidsbeoordeling volledig dierproefvrij én mechanistisch nauwkeurig te maken. Twee belangrijke pijlers hierin zijn Adverse Outcome Pathways — (q)AOP's — en Physiological Maps, waarmee we vastleggen en verklaren hoe een blootstelling leidt tot een Adverse Outcome."

[2. DE KWESTIE — Rechts op de slide]
"Maar daar ligt een fundamentele wetenschappelijke kwestie:
Ten eerste zijn AOP's ontzettend sterk voor een reductionistische aanpak, maar ze zijn in de basis lineair. Er wordt hard gewerkt om ze kwantitatief te maken (qAOP's), maar daar ligt nog een grote uitdaging — mede omdat biologische data inherent multimodaal is.

Ten tweede is biologie niet lineair: gezonde cellen beschikken over homeostatische feedbackloops die verstoringen opvangen, en sommige initiating events grijpen niet aan op één losse receptor, maar direct op weefselniveau — denk bijvoorbeeld aan asbestvezels. Dit soort niet-lineaire interacties vangen we in Physiological Maps.

Ten derde zijn er processen waarbij de Molecular Initiating Event op zichzelf onvoldoende vertelt over of de Adverse Outcome daadwerkelijk optreedt. Een sprekend voorbeeld in het midden van het schema is Cortical Spreading Depression (CSD). Daar kennen we de MIE — ion channel disruption — maar of die verstoring daadwerkelijk uitmondt in een pathologische CSD-golf, hangt af van tissue-level emergence: een ruimtelijk reaction-diffusion-achtig proces over het weefsel, dat we bijvoorbeeld meten met Multi-Electrode Arrays (MEA-data).

Kortom: om een Adverse Outcome betrouwbaar te voorspellen, moeten we data uit (q)AOP's, niet-lineaire Physiological Maps én ruimtelijk-temporele metingen zoals MEA-data tegelijkertijd meenemen. Mijn centrale claim in dit onderzoeksvoorstel is dat juist Multimodal en Multiscale AI bij uitstek geschikt is om deze enorme diversiteit aan data en biologische schalen wiskundig samen te brengen."
-->

---
layout: center
class: text-center
---

# Slide 2 — Aanpak, CV & Motivatie

*(Wordt in de volgende stap uitgewerkt)*

<!--
Placeholder voor Slide 2 (max. 90 seconden):
- Hoe het voorstel dit oplost (Multimodal & Multiscale AI architectuur)
- Jouw CV, achtergrond & persoonlijke motivatie
-->
