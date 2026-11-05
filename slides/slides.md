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
<!-- RIGHT COLUMN: DE KWESTIE — TWEE DELEN: (A) lineair AOP 34 -> attractor, (B) emergente CSD -->
<div class="panel kwestie-panel">
<div class="panel-header">
<span class="badge badge-rose">2. De Kwestie</span>
<span class="panel-subtitle">Van lineair AOP naar emergente weefseldynamiek</span>
</div>
<!-- DEEL A: AOP 34 (lineair) — één klik voegt de adaptieve homeostatische attractor toe -->
<div class="part">
<div class="part-label">
<span class="step-num step-a">A</span>
<span class="part-title">Lineair: AOP 34</span>
<span class="part-note">LXR (MIE 167) → TG-accumulatie (KE 291) → steatose</span>
<span v-click.hide class="click-hint">klik ▸</span>
</div>
<div class="part-a-body">
<div class="aop34-wrap">
<img
:src="'/AOP34.png'"
onerror="this.onerror=null;this.src='/aop_mini.svg';"
alt="AOP 34: LXR-activatie leidt tot leververvetting (steatose)"
class="aop34-img"
/>
</div>
<!-- v-click: adaptieve laag bovenop AOP 34 (Gemini-concept, inhoudelijk met AOP 34 gecombineerd) -->
<div v-click class="attractor-card">
<div class="ac-title">Adaptieve homeostatische attractor <span class="ac-mono">z<sub>homeo</sub> = z<sub>0</sub></span></div>
<div class="ac-lane">
<span class="chip chip-input">p: LXR → DNL · CD36<span class="chip-sub">SREBF1 · FASN · ACACA · SCD</span></span>
<span class="ac-arrow">▶</span>
<span class="chip chip-buffer">TG-accumulatie · KE 291<span class="chip-sub">reversibele buffer</span></span>
</div>
<div class="ac-lane">
<span class="ac-fb-tag">adaptieve klaring ⊣</span>
<span class="chip chip-homeo">PPARα / CPT1A<span class="chip-sub">β-oxidatie</span></span>
<span class="chip chip-homeo">MTTP / APOB<span class="chip-sub">VLDL-export</span></span>
</div>
<div class="ac-lane ac-lane-tip">
<span class="ac-arrow ac-arrow-red">⟹</span>
<span class="chip chip-path">Pathologische attractor<span class="chip-sub">p &gt; dynamische capaciteit → lipotoxiciteit → celdood</span></span>
</div>
</div>
</div>
</div>
<!-- DEEL B: dezelfde MIE op hoger abstractieniveau — klikbare CSD-video's -->
<div class="part">
<div class="part-label">
<span class="step-num step-b">B</span>
<span class="part-title">Niet-lineair: dezelfde MIE, hoger abstractieniveau</span>
<span class="part-note">CSD = emergent weefselproces</span>
</div>
<div class="video-row">
<figure class="video-card" onclick="var v=this.querySelector('video');this.classList.toggle('playing',v.paused);v.paused?v.play():v.pause();">
<video :src="'/csd_wave.mp4#t=0.05'" class="csd-video" muted loop playsinline preload="metadata"></video>
<div class="play-badge">▶</div>
<figcaption><strong>CSD-golf</strong><br/>reaction-diffusion over het weefsel</figcaption>
</figure>
<figure class="video-card" onclick="var v=this.querySelector('video');this.classList.toggle('playing',v.paused);v.paused?v.play():v.pause();">
<video :src="'/csd_recovery.mp4#t=0.05'" class="csd-video" muted loop playsinline preload="metadata"></video>
<div class="play-badge">▶</div>
<figcaption><strong>Herstel</strong><br/>terug naar homeostase</figcaption>
</figure>
<div class="video-note">
<div class="vn-title">Zelfde MIE — andere schaal</div>
<div class="vn-body">Ionkanaal-verstoring alleen voorspelt géén <em>Adverse Outcome</em>: de uitkomst ontstaat pas door <strong>ruimtelijk-temporele emergentie</strong> in het weefsel (MEA-data).</div>
<div class="vn-foot">klik op een video om af te spelen</div>
</div>
</div>
</div>
</div>
</div>
<!-- Bottom Claim Banner — full width onder beide kolommen -->
<div class="claim-banner">
<div class="claim-icon">✦</div>
<div class="claim-text">
<strong>De Claim:</strong> Een <em>Adverse Outcome</em> voorspel je niet lineair — je moet de <em>adaptieve attractor</em> én de <em>emergente weefseldynamiek</em> samen modelleren. Precies daarom is <strong>Multimodale &amp; Multiscale AI</strong> nodig om de schalen te verbinden.
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
/* ===== PITCH-MODUS: Slidev's eigen bedieningsbalk niet over de slide =====
   Slidev 53 pint de nav-balk permanent vast zodra persistNav waar is:
     persistNav        = isScreenVertical || showEditor        (pages/play.vue)
     isScreenVertical  = (vensterhoogte - vensterbreedte / (16/9)) > 120
   In een gewoon browservenster (of op een 16:10-scherm in fullscreen) is dat
   altijd waar -> de balk blijft over de slides hangen én vangt hover/clicks
   linksonder weg. Hieronder zetten we die controls-laag volledig uit.
   #slide-container heeft exact twee children: #slide-content.slidev-slide-content
   (de slide) en de controls-slot (de balk) -> ::not() is genoeg.
   Navigeren blijft: spatie / → / PageDown = volgende klik, ↓ of Shift+→ =
   volgende slide, ← = terug. F = fullscreen, O = overview, G = goto.
   Sprekersnotities: open /presenter in een tweede venster (localhost:3030/presenter).
   Wil je de balk tóch terug? Zet de regel hieronder in commentaar. */
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
  grid-template-columns: 33% 65.5%;
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


/* ===== De Kwestie in twee delen (A: AOP 34 + attractor, B: CSD-video's) ===== */
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
  min-height: 178px;
}
.aop34-wrap {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  overflow: hidden;
  border-radius: 9px;
  border: 1px solid #334155;
  background: #f8fafc;
}
.aop34-img {
  height: 178px;
  width: auto;
  display: block;
}
.attractor-card {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 0.18rem;
  background: #0b1120;
  border: 1px dashed rgba(16, 185, 129, 0.55);
  border-radius: 10px;
  padding: 0.3rem 0.4rem;
}
.ac-title {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.58rem;
  font-weight: 800;
  color: #6ee7b7;
  text-transform: uppercase;
  letter-spacing: 0.02em;
  line-height: 1.15;
}
.ac-mono {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.5rem;
  text-transform: none;
  color: #a7f3d0;
  background: rgba(16, 185, 129, 0.14);
  border-radius: 4px;
  padding: 0 0.25rem;
}
.ac-lane {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.22rem;
}
.ac-lane-tip { gap: 0.3rem; }
.ac-arrow {
  font-size: 0.5rem;
  font-weight: 900;
  color: #38bdf8;
}
.ac-arrow-red {
  font-size: 0.62rem;
  color: #f43f5e;
}
.ac-fb-tag {
  font-size: 0.46rem;
  font-weight: 800;
  color: #34d399;
  text-transform: uppercase;
  letter-spacing: 0.02em;
}
.chip {
  display: inline-flex;
  flex-direction: column;
  gap: 1px;
  font-size: 0.55rem;
  font-weight: 700;
  line-height: 1.1;
  padding: 0.12rem 0.34rem;
  border-radius: 6px;
  border: 1px solid;
}
.chip-sub {
  font-size: 0.44rem;
  font-weight: 600;
  opacity: 0.82;
}
.chip-input {
  background: #1e293b;
  border-color: rgba(56, 189, 248, 0.6);
  color: #e2e8f0;
}
.chip-input .chip-sub { color: #93c5fd; }
.chip-buffer {
  background: #0f172a;
  border-style: dashed;
  border-color: rgba(59, 130, 246, 0.65);
  color: #e2e8f0;
}
.chip-buffer .chip-sub { color: #93c5fd; }
.chip-homeo {
  background: #064e3b;
  border-color: rgba(16, 185, 129, 0.7);
  color: #ecfdf5;
}
.chip-homeo .chip-sub { color: #a7f3d0; }
.chip-path {
  background: #450a0a;
  border-color: rgba(239, 68, 68, 0.8);
  color: #fee2e2;
}
.chip-path .chip-sub { color: #fca5a5; }

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

/* ===== Deel B: klikbare CSD-video's ===== */
.video-row {
  display: flex;
  align-items: flex-start;
  gap: 0.4rem;
}
.video-card {
  position: relative;
  margin: 0;
  flex: 0 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.12rem;
  cursor: pointer;
}
.csd-video {
  width: 128px;
  height: 128px;
  display: block;
  object-fit: cover;
  border-radius: 9px;
  border: 1px solid #334155;
  background: #020617;
  transition: border-color 0.2s ease;
}
.video-card:hover .csd-video {
  border-color: #fb7185;
}
.play-badge {
  position: absolute;
  top: 64px;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.6rem;
  color: #f8fafc;
  background: rgba(2, 6, 23, 0.62);
  border: 1px solid rgba(248, 250, 252, 0.6);
  pointer-events: none;
  transition: opacity 0.2s ease;
}
.video-card.playing .play-badge {
  opacity: 0;
}
.video-card figcaption {
  width: 128px;
  font-size: 0.52rem;
  color: #cbd5e1;
  text-align: center;
  line-height: 1.2;
}
.video-card figcaption strong {
  color: #fda4af;
}
.video-note {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.16rem;
  border-left: 2px solid rgba(244, 63, 94, 0.5);
  padding-left: 0.4rem;
}
.vn-title {
  font-size: 0.58rem;
  font-weight: 800;
  color: #fda4af;
  text-transform: uppercase;
  letter-spacing: 0.02em;
}
.vn-body {
  font-size: 0.54rem;
  color: #cbd5e1;
  line-height: 1.25;
}
.vn-body strong {
  color: #f1f5f9;
}
.vn-foot {
  font-size: 0.48rem;
  color: #94a3b8;
  font-style: italic;
}
</style>

<!--
SPREEKTEKST SLIDE 1 (± 100–110 seconden | ~250 woorden):

[1. DE CONTEXT — Links op de slide]
"Goedemiddag. Binnen ons vakgebied werken we aan de transitie naar Probabilistic Chemical Risk Assessment met behulp van New Approach Methodologies — oftewel NAMs. We combineren daarin geavanceerde humane in vitro en in silico modellen om chemische veiligheidsbeoordeling volledig dierproefvrij én mechanistisch nauwkeurig te maken. Twee belangrijke pijlers hierin zijn Adverse Outcome Pathways — (q)AOP's — en Physiological Maps, waarmee we vastleggen en verklaren hoe een blootstelling leidt tot een Adverse Outcome."

[2. DE KWESTIE — Rechts op de slide, in twee delen]
"Maar daar ligt een fundamentele wetenschappelijke kwestie, en die splits ik in twee delen.

Deel A — lineair. AOP 34, LXR-activatie die tot steatose leidt, is sterk reductionistisch: MIE, key events, adverse outcome. [klik] Maar zo'n AOP is in de basis lineair, en dat is precies het probleem. In de cel is de triglyceride-accumulatie namelijk eerst een reversibele buffer, die actief wordt leeggehouden door beta-oxidatie — PPARalpha en CPT1A — en door VLDL-export via MTTP en APOB. Zolang de perturbatie, dus LXR-gedreven de-novo-lipogenese en CD36-gemedieerde instroom, binnen die dynamische klaringscapaciteit blijft, veert het systeem terug naar homeostase: een adaptieve homeostatische attractor. Pas als de perturbatie groter wordt dan die capaciteit, kantelt het systeem naar een pathologische attractor: lipotoxiciteit, celdood en steatose. Dat kantelpunt — die bifurcatie — kan een lineair AOP niet voorspellen, en dat moeten we kwantitatief modelleren.

Deel B — niet-lineair. Soms vertelt de Molecular Initiating Event op zichzelf onvoldoende of de Adverse Outcome daadwerkelijk optreedt. Neem Cortical Spreading Depression: we kennen de MIE — ion channel disruption — maar of die uitmondt in een pathologische CSD-golf hangt af van tissue-level emergence: een ruimtelijk reaction-diffusion-proces over het weefsel, dat we bijvoorbeeld met Multi-Electrode Arrays meten. [klik op de video's] Twee simulaties: de golf die zich over het weefsel voortplant, en het herstel terug naar homeostase. Dezelfde MIE, maar de uitkomst ontstaat pas op een hoger abstractieniveau.

In de context van Cortical Spreading Depression (CSD) veroorzaakt door verstoring van ionkanalen en kaliumophoping, vertalen $u$ en $a$ zich direct naar specifieke biofysische componenten van het neurale weefsel:1. Wat stelt $u$ voor?$u$ is de dimensieloze maat voor de lokale excitatietoestand: voornamelijk de extracellulaire kaliumconcentratie ($[\text{K}^+]_e$) en de gekoppelde membraandepolarisatie ($V_m$).Biologische betekenis:In rusttoestand houden neuronen $[\text{K}^+]_e$ strak gereguleerd op een laag basisniveau ($\approx 3\text{ tot }4\text{ mM}$), wat overeenkomt met $u \approx 0$.Autocatalytische dynamiek:Wanneer ionkanalen verstoord raken of cellen depolariseren, lekken grote hoeveelheden $\text{K}^+$ de nauwe extracellulaire ruimte in. Omdat de evenwichtspotentiaal voor kalium ($E_K$) hierdoor minder negatief wordt, depolariseren omliggende neuronen. Deze depolarisatie opent spanningsafhankelijke kanalen ($\text{Na}^+$ en persistente $\text{Ca}^{2+}$-stromen), wat leidt tot nóg meer depolarisatie en massale $\text{K}^+$-uitstoot.In de vergelijking weerspiegelt de term $u(1 - u)$ deze positieve terugkoppeling (de autocatalytische ontbranding) tot aan de verzadigde CSD-piek ($[\text{K}^+]_e \approx 50\text{–}80\text{ mM}$, wat overeenkomt met $u \approx 1$).2. Wat stelt $a$ voor?$a$ is de drempelwaarde (de tipping point of bifurcatiedrempel) waarbij de lokale opnamecapaciteit van astrocyten en pompen bezwijkt.Biologische betekenis:Onder normale omstandigheden werken twee fysiologische verdedigingsmechanismen om $[\text{K}^+]_e$ direct af te voeren:Astrocytaire buffering: Astrocyten nemen overtollig $\text{K}^+$ op via $\text{Kir}4.1$-kanalen en verspreiden dit via hun gap-junctionnetwerk (spatial potassium buffering).$\text{Na}^+/\text{K}^+$-ATPase pompen: Pompen actief kalium terug de cellen in tegen de gradiënt in.Het kantelpunt $a$:Zolang de kaliumstijging onder de drempelwaarde $a$ blijft ($[\text{K}^+]_e \lesssim 10\text{–}12\text{ mM}$), zijn deze pompen en gliale buffers snel genoeg om alle kalium weg te werken. Zodra de lokale verstoring echter groter is dan $a$, raakt de opnamecapaciteit van de pompen en astrocyten lokaal verzadigd.Vanaf dat moment overtreft de stroom van vrijkomend kalium de klaring, waardoor de term $(u - a)$ positief wordt en de ongecontroleerde, zelfvoorstuwing van de CSD-golf ontstaat.Samengevat in toxicologische termen:$u$: De directe toestandsparameter (de concentratie $[\text{K}^+]_e$ die stijgt als gevolg van een toxische interventie op ionkanalen of pomp-inhibitie).$a$: De resilience-grens van het weefsel; een toxische stof die bijvoorbeeld de $\text{Na}^+/\text{K}^+$-ATPase remt (zoals ouabaïne) of de astrocytaire koppeling verzwakt, verlaagt effectief de waarde van $a$. Hierdoor heeft het weefsel een veel kleinere prikkel nodig om direct in een CSD-golf te kantelen.

Kortom: om een Adverse Outcome betrouwbaar te voorspellen, moeten we de adaptieve attractor én de emergente weefseldynamiek samen met de (q)AOP's en Physiological Maps meenemen. Mijn centrale claim in dit onderzoeksvoorstel is dat juist Multimodal en Multiscale AI bij uitstek geschikt is om deze enorme diversiteit aan data en biologische schalen wiskundig samen te brengen."
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
