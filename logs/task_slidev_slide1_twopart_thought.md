# Thought & Progress Log: Slidev slide 1 — toolbar wegwerken + "De Kwestie" in twee delen
- **Date / Time:** 2026-11-05 (session 3, na de raw-HTML-indent fix)
- **Task Goal:** Pitchdeck (2 slides, slide 1 = Context & De Kwestie) presentatieklaar maken:
  (1) de Slidev-navigatiebalk mag niet meer permanent over de slides hangen, ook niet in fullscreen;
  (2) slide 1 simpeler: "De Kwestie" opgesplitst in deel A (AOP34.png, met één klik wordt het concept
  "adaptieve homeostatische attractor" toegevoegd) en deel B (twee klikbare video's die aantonen dat
  CSD op een hoger abstractieniveau gebeurt).

## 1. Analysis & Problem Statement
- **Klacht 1 — toolbar:** "de toolbar blijft de hele tijd voor mn slides hangen, ook als ik naar fullscreen ga".
  Onderzocht in `slides/node_modules/@slidev/client` (Slidev **53.1.0**):
  - `pages/play.vue:60` -> `const persistNav = computed(() => isScreenVertical.value || showEditor.value)`
  - `pages/play.vue:104-113` -> de controls-container krijgt `opacity-0 hover:opacity-100`,
    maar bij `persistNav` de override `!opacity-100 right-0` (= altijd zichtbaar, over de volle breedte).
  - `state/storage.ts:42` -> `isScreenVertical = windowSize.height - windowSize.width / slideAspect > 120`
    met `slideAspect = 16/9`. Elk venster dat niet exact 16:9 is (niet-gefullscreen browservenster, of een
    16:10-scherm in fullscreen) is dus "verticaal" -> balk permanent vastgepind. Dat verklaart ook
    "ook in fullscreen".
  - Extra probleem: de container blijft (ook onzichtbaar) hoveren/clicks opvangen linksonder.
  - **Fix:** de controls-laag volledig uitzetten met globale CSS. Stabiele selector:
    `#slide-container > div:not(.slidev-slide-content)` (bevestigd in `internals/SlideContainer.vue`:
    `#slide-container` bevat exact twee children: `#slide-content.slidev-slide-content` + de controls-slot).
    Navigeren blijft mogelijk met spatie/pijltjes/clicker; notities via `/presenter`.
- **Klacht 2 — slide simpeler + twee delen:** de huidige rechterkolom is een hub met 3 spoke-cards
  + een centrale CSD-kaart + claim-banner. Dat wordt vervangen door twee expliciete delen.
  - **Deel A:** `public/AOP34.png` (bestaat, ~1000x640, wit) is de lineaire AOP 34-keten.
    Eén klik voegt de *adaptieve homeostatische attractor* toe. Gemini's Mermaid-voorstel is
    inhoudelijk samengevoegd met AOP 34 (i.p.v. ernaast geplakt):
    perturbatie p = LXR->DNL (SREBF1/FASN/ACACA/SCD) + CD36-instroom -> TG-accumulatie (KE 291)
    als **reversibele buffer** -> adaptieve klaring via PPARα/CPT1A (β-oxidatie) en MTTP/APOB
    (VLDL-export) remt TG (negatieve feedback) -> bij p > dynamische capaciteit bifurcatie naar de
    pathologische attractor (lipotoxiciteit -> celdood). Dit is exact dezelfde fysiologie als het
    18-genen netwerk uit Milestone 1.1/1.2 (groepen AOP34 + Homeostasis + Stress_Apoptosis).
  - **Deel B:** `public/csd_wave.mp4` en `public/csd_recovery.mp4` (beide 400x400, 4.0 s, 30 fps,
    h264, zonder audiospoor) naast elkaar, klikbaar om af te spelen, als bewijs dat dezelfde MIE op
    weefselniveau (CSD, MEA) een emergente uitkomst geeft.
- **Pitfalls / boundary conditions:**
  - Slidev 53 heeft **geen** click-to-advance op slide-content: `pages/play.vue:23-34` controleert
    `e.target.id === 'slide-container'` (alleen de letterbox). Er is ook geen
    `addEventListener('click')` in de client. Gevolg: (a) `v-click` avanzert via spatie/pijl/clicker,
    niet via een muisklik op de slide; (b) een klik op een `<video>` avanzert de slide dus níet.
  - Video's starten via een inline `onclick` (native DOM-attribuut; Vue zet dit via `setAttribute`,
    want `isOn()` matcht alleen `/^on[^a-z]/`). Geen `<script setup>` nodig.
  - Statische `src` naar een niet-bestaand bestand laat de Vite-build hard falen (eerdere bug);
    daarom overal `:src="'...'"`. Alle gebruikte bestanden bestaan nu écht.
  - `slides/public` is een symlink naar `../public`, dus `/AOP34.png`, `/csd_wave.mp4`,
    `/csd_recovery.mp4` worden zowel in dev als in de build geserveerd.
  - **Flush-left regel** (vorige bug): alle ruwe HTML blijft op kolom 0, geen lege regels binnen
    een ingesprongen blok.

## 2. Methodological & Scientific Considerations
- Waarom `v-click` i.p.v. eigen JS-state? `v-click` is de enige reveal die meesynchroniseert met
  presentator-modus/PDF-export en met een presentatieclicker (PageDown/ArrowRight). Eigen
  `ref()`-state zou in presentatorweergave niet naar het publieksvenster syncen.
- Waarom de attractor-visual als HTML/CSS-chips i.p.v. Mermaid? Mermaid-fences binnen een raw
  HTML-blok worden niet geparsed door markdown-exit (het blok is letterlijk HTML), en de huisstijl
  (donker, chips, kleurcodering) blijft zo consistent. Bovendien: geen extra runtime-afhankelijkheid.
- Wetenschappelijk: de attractor-laag volgt de AOP-ontologie (MIE 167 -> KE 264/116 -> KE 291 -> AO)
  en koppelt die aan de drie empirisch verankerde attractors van Milestone 1.2
  (z_homeo = z0, z_steatose, z_cholestase); de bifurcatie is precies "Point of Departure" uit
  Milestone 4. De video's illustreren de niet-lineaire/emergente schaal.

## 3. Step-by-Step Execution Plan
- Bestand: `slides/slides.md` (slide 1 markup + globale `<style>`).
  1. Hub-blok (`<div class="hub-canvas">` ... bijhorende `</div>`) verwijderen.
  2. Rechterkolom-opbouw vervangen: panel-header -> **Deel A** (label + AOP34.png + `v-click`
     attractor-card) -> **Deel B** (label + 2 klikbare video-cards + MEA-noot).
  3. Claim-banner uit het rechterpaneel halen -> full-width onder de grid.
  4. Dode hub/spoke/CSD-CSS verwijderen; nieuwe CSS voor `.part*`, `.attractor-card`, `.chip*`,
     `.video-*` toevoegen; `.main-grid` naar 33% / 65.5%.
  5. Toolbar-fix: globale CSS-regel die de Slidev-controlslaag uitzet (met onderhoudscommentaar).
- Verificatie: parse-check op de slide (0 ge-escapte `<pre>`-blokken, divs gebalanceerd), daarna
  `pnpm exec slidev build` (EXIT=0) en controleren dat `AOP34.png` + beide mp4's in `dist/` staan.

## 4. Terminal Execution & Verification (GPU 1)
- Dit is een Node/Slidev-taak (geen PyTorch), dus zonder `uv`/CUDA — wel op dezelfde machine.
- **Parse-check** (tijdelijk script `slides/.verify_slide1b.mjs`, daarna verwijderd):
  `node .verify_slide1b.mjs` -> `EXIT=0`
  ```text
  PASS  slides gevonden  (aantal=2)
  PASS  geen <pre> (indented code block bug)  (pre=0)
  PASS  geen <p> in slide 1  (p=0)
  PASS  divs gebalanceerd  (43/43)
  PASS  geen dubbel-geescapeerde entiteiten  (amp;amp=0)
  PASS  AOP34.png aanwezig
  PASS  csd_wave.mp4 aanwezig
  PASS  csd_recovery.mp4 aanwezig
  PASS  v-click attractor aanwezig
  PASS  v-click.hide hint aanwezig
  PASS  onclick video-handler aanwezig  (n=2)
  PASS  video :src bindingen  (n=2)
  PASS  geen statische src="/ (Vite UNRESOLVED_IMPORT)  (n=0)
  PASS  oude hub-markup weg
  PASS  sprekersnotities intact  (note_len=2623)
  ALLE CHECKS OK
  ```
- **Build:** `pnpm exec slidev build` -> `✓ 454 modules transformed` / `✓ built in 2.12s` / `BUILD_EXIT=0`.
  - `dist/AOP34.png` (158958 B), `dist/csd_wave.mp4` (13293 B), `dist/csd_recovery.mp4` (6353 B) staan klaar.
  - Slide-chunk `dist/assets/slidev/md-Ckhiwrtv.js`: `&lt;div` = **0** (geen ge-escapte HTML),
    `onclick` = 2, clicks-directive refs = 4 (v-click + v-click.hide), `csd-video` = 2.
  - Gebouwde CSS `dist/assets/md-*.css` bevat `#slide-container>div:not(.slidev-slide-content){display:none!important}`
    plus alle nieuwe klassen (`.part-a-body`, `.aop34-img`, `.attractor-card`, `.chip-*`, `.video-row`,
    `.csd-video`, `.video-note`, `.play-badge`).
  - Compilatie van de inline handlers bevestigd: `onclick:` staat als **string** in de props
    (dus `setAttribute`, geen Vue-listener) en `:src` is een string-literal -> runtime-URL.
- **Dev-server end-to-end:** `pnpm exec slidev slides.md --port 3031`, daarna
  `/` -> 200, `/AOP34.png` -> 200, `/csd_wave.mp4` -> 200, `/csd_recovery.mp4` -> 200,
  0 errors/warnings in het dev-log. Server daarna gestopt.
- **Niet gelukt:** visuele screenshot (`images_for_AI_debugging/`) — er is geen headless browser
  (playwright ontbreekt; `/usr/bin/firefox` is de snap die door de lopende gebruikersinstantie
  geblokkeerd is). Layout/overflow is daarom niet pixelmatig geverifieerd.

## 5. Verification of Completed Tasks
- `slides/slides.md` daadwerkelijk gewijzigd (gecontroleerd met `sed -n`/`grep -n`):
  - Hub-blok (`<div class="hub-canvas">` … `</div>`, 78 regels) verwijderd; `grep -n 'hub-canvas|spoke-|center-csd-card'` -> geen markup-hits meer.
  - Deel A (regels 73-109) en Deel B (110-133) aanwezig; claim-banner nu buiten de grid (regels 138-145).
  - Toolbar-fix in de globale `<style>` (regels 153-167) met onderhoudscommentaar.
  - Dode CSS (`.hub-*`, `.spoke-*`, `.center-csd-card`, `.csd-*`, `.sub-note`, `.text-*`) verwijderd;
    nieuwe CSS toegevoegd (`.part*` 298-480, video's 502-590).
  - `.main-grid` van `36% 62.5%` naar `33% 65.5%`.
  - Sprekersnotities (regels 593-607) herschreven naar Deel A/Deel B, inclusief `[klik]`-markeringen.
- Alle hierboven genoemde bestanden bestaan; het tijdelijke verificatiescript is verwijderd
  (`rm -f .verify_slide1b.mjs`, bevestigd met `ls .verify*` -> niets).
- Niet gecontroleerd (geen browser): werkelijke pixel-layout, video-playback en de reveal-animatie.

## 6. Reflection & Next Step
- De toolbar-klacht was geen CSS-vergissing van de slide maar een Slidev-gedrag:
  `persistNav = isScreenVertical || showEditor`. De gekozen fix (`display:none` op de controls-laag)
  is robuust, verwijdert ook de onzichtbare hover-zone linksonder, en laat toetsenbordnavigatie intact.
  Prijs: de presenter-knop zit in die balk -> presentatormodus open je via `http://localhost:3030/presenter/`.
- De slide is nu lineair te vertellen: A (lineair AOP 34) -> klik -> attractor -> B (emergente CSD, video's).
- Directe follow-up voor de gebruiker:
  1. `cd slides && pnpm run dev`, slide 1 checken: spatie/`→` = attractor verschijnt, klik op een video = play.
  2. Visueel controleren of de rechterkolom niet overflowt (video 128px, AOP34 178px hoog);
     zo nodig `.csd-video`/`.aop34-img` in hoogte bijstellen.
  3. Sprekersnotities nalopen: de timing (~210 woorden) is iets langer geworden door de attractor-uitleg.

