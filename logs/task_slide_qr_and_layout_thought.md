# Thought & Progress Log: Slide 1 — QR-code gerepareerd + video/QR-verhouding (deel B)

- **Date / Time:** 2026-11-06 12:5x
- **Task Goal:** Deel B van de "Beyond Linear AOPs"-kwestie (CSD) gebruiken `csd_experiment` in plaats van de
  losse `csd_wave`/`csd_recovery`-video's, de QR-code daadwerkelijk scanbaar maken en de tekstkolom
  ("Same MIE, different outcomes") verkleinen ten gunste van de QR-code.

## 1. Analysis & Problem Statement
- **Exacte vraag:** (a) gebruik voor kwestie **deel 2** `csd_experiment` i.p.v. `csd_recovery` + `csd_wave`;
  (b) de QR werkt niet — een telefoon herkent niets; (c) minder ruimte voor de tekst onder de kop
  "SAME MIE, different outcome" en meer ruimte voor de QR.
- **Interpretatie / bevindingen in de repo:**
  - `slides/slides.md` verwees in de werkboom al naar `/csd_experiment.webm` (960x480, VP8, 435 frames,
    2:1). `public/csd_wave.mp4` en `public/csd_recovery.mp4` waren dood gewicht (nergens meer gerefereerd).
  - De QR was een **handgetekende SVG-matrix** (`<path d="M2 2h7v1h-7z...">`, 6376 tekens, viewBox 41x41)
    met quiet zone 2 en weergaveformaat 78x78 px. Twee onafhankelijke fouten: de matrix is geen geldige
    QR-codering (geen Reed-Solomon/masker/format-info) en 78 px voor 41 modules (~1,9 px/module) is op een
    scherm/projectie onleesbaar voor een camera.
- **Boundary conditions:** de QR moet *vanaf een scherm* scanbaar zijn (echte encoderkeuze, quiet zone
  >= 4 modules, hoog contrast, voldoende fysieke grootte) en de lay-out van deel B mag niet overstromen
  (video 270 px + tekst + QR moeten binnen de ~695 px paneelbreedte blijven).

## 2. Methodological & Scientific Considerations
- **Echte encoder i.p.v. handwerk:** `uqr@0.1.3` zit al in `slides/node_modules` (zero-dependency, algoritme
  van Nayuki's QR-Code-generator). ECC **M** (15% redundantie) is de juiste balans voor een foto van een
  scherm; quiet zone **4** modules is de QR-spec.
- **Verifieerbaarheid:** een QR die je niet decodeert is een gok. Daarom wordt de output onafhankelijk
  getest met **zxing-cpp** (tijdelijke uv-omgeving) op meerdere weergaveformaten (490/300/200/150/120 px)
  en met gesimuleerde camera-onscherpte. Ook de oude QR is met dezelfde decoder getest om de oorzaak hard
  te maken (reproduceerbaar bewijs i.p.v. "waarschijnlijk kapot").
- **Lay-out:** de QR groeit van 78 -> 156 px (+78 px) en de tekstkolom krimpt automatisch mee (flex-krimp)
  tot ~233 px; met `max-width: 240px` blijft de tekstkolom gegarandeerd smaller dan voorheen zonder dat er
  een gat in de rij valt.
- **Consistentie met de rest van het project:** geen nieuwe Python-dependencies; `public/` blijft de
  enige plek voor webassets (`.clinerules`), en de deck wordt met de bestaande `pnpm build` (slidev)
  gevalideerd.

## 3. Step-by-Step Execution Plan
1. `slides/make-qr.mjs` — QR (ECC M, quiet zone 4) -> `public/qr_csd_simulator.png` (490x490, 10 px/module),
   met een minimale PNG-encoder (zlib zit in Node, dus geen extra packages).
2. Verificatie: `/tmp/verify_qr.py` (zxing-cpp) op 5 formaten + blur; `/tmp/decode_old_qr.py` om de oude
   matrix te ontleden en te bewijzen dat die niet decodeert.
3. `slides/slides.md`: oude SVG-QR-blok (regels 115-122) vervangen door `<img class="qr-img" :src="'/qr_csd_simulator.png'">`;
   CSS `.qr-svg` -> `.qr-img` (156 px), `.video-note` `flex: 1 1 0` + `max-width: 240px`, `.qr-card` padding iets ruimer.
4. `git rm public/csd_wave.mp4 public/csd_recovery.mp4` (dode v1-assets) en `pnpm build` als eindvalidatie.

## 4. Terminal Execution & Verification (GPU 1)
- **QR genereren:** `cd slides && node make-qr.mjs`
  ```text
  QR-URL        : https://raw.githack.com/MeesterDavid/Levertoxiciteit/main/cortical_spreading_depression_simulator.html (102 tekens)
  QR-versie     : 6 -> 41x41 modules (+4 quiet zone)
  ECC / masker  : M / 2
  PNG           : 490x490 px (10 px per module) -> .../public/qr_csd_simulator.png
  ```
- **QR decoderen (nieuwe code):** `CUDA_VISIBLE_DEVICES=1 uv run --with zxing-cpp --with pillow --with numpy python /tmp/verify_qr.py`
  ```text
  bron: 490x490 px, mode=L
     490 px -> OK  QR Code ecc=M
     300 px -> OK  QR Code ecc=M
     200 px -> OK  QR Code ecc=M
     150 px -> OK  QR Code ecc=M
     120 px -> OK  QR Code ecc=M
    150 px + blur(0.8) -> OK
  RESULTAAT: ALLE DECODES OK
  ```
- **Oude QR decoderen (foutdiagnose):** `... python /tmp/decode_old_qr.py`
  ```text
  path-lengte: 6376 tekens
  subpaden: 468 | punten totaal: 1872
  donkere modulen: 638 van 1369 (46.6%)
  decode oude QR: MISLUKT (geen QR gevonden)
    finder op (0,0): correct
    finder op (30,0): correct
    finder op (0,30): correct
  ```
  -> de drie finder-patterns kloppen (daarom "ziet" een camera iets), maar de data/RS/format-bits zijn
  ongeldig: geen enkele decoder haalt er een URL uit. Dit is exact het gerapporteerde telefoongedrag.
- **Doel-URL bereikbaar:** `curl -s -o /dev/null -w '%{http_code} %{size_download}' -L <url>` -> `200 31464`.
- **Deck bouwen (eindvalidatie):** `cd slides && pnpm build` -> `EXIT=0`, `✓ built in 2.02s`.
- **Build-inhoud gecontroleerd:** `ls dist | grep -i 'csd\|qr'` -> `csd_experiment.webm`, `qr_csd_simulator.png`;
  `grep -rlo 'csd_wave\|csd_recovery' dist` -> *geen treffers*; `grep '/qr_csd_simulator.png' dist/index.html` -> treffer.

## 5. Verification of Completed Tasks
| # | Bestand | Status | Bewijs |
|---|---|---|---|
| 1 | `public/qr_csd_simulator.png` | OK | 1679 B, PNG 490x490 grijswaarden, decodeert op 120-490 px + blur |
| 2 | `slides/make-qr.mjs` | OK | nieuw script; `node make-qr.mjs` geeft versie 6 / ECC M / masker 2 |
| 3 | `slides/slides.md` — video | OK | regel 103: `/csd_experiment.webm`; geen `csd_wave`/`csd_recovery` meer in het bestand |
| 4 | `slides/slides.md` — QR | OK | regel 117: `<img class="qr-img" ... '/qr_csd_simulator.png'>`; oude 6,4 kB SVG-path weg (21 686 -> 15 291 B) |
| 5 | `slides/slides.md` — CSS | OK | `.qr-img` 156x156 (was 78x78), `.video-note` `flex: 1 1 0; max-width: 240px`, `.qr-card` padding 0.4/0.5rem |
| 6 | `public/csd_wave.mp4`, `public/csd_recovery.mp4` | verwijderd | `git rm` uitgevoerd; deck verwijst er niet meer naar, build bevat ze niet |
| 7 | `slides/dist/` (build) | OK | `pnpm build` EXIT=0, bevat `csd_experiment.webm` + `qr_csd_simulator.png` |

## 6. Reflection & Next Step
- **Werkt het?** De QR decodeert aantoonbaar op elk realistisch weergaveformaat (tot 120 px, zelfs met
  blur), staat 2x groter op de slide en heeft een correcte quiet zone; de deck bouwt schoon zonder de oude
  video's. De tekstkolom is ~80 px smaller geworden, de QR ~78 px breder.
- **Beperking / open punt:** de inhoud van `public/csd_experiment.webm` is nog de **v1**-simulatie
  ("Condition A/B, Inhibitor 4% vs 10.5%", astrocyt-buffering). De v2-pipeline (`csd_sim_v2.py`) laat zien
  dat de eerlijke, kwantitatieve variant "zelfde MIE, andere **dosis**" is (PoD = 0,05564; bistabiliteit
  K = 0,15 < K_crit = 0,1936; r_c = 4,25 px). Wil je die boodschap op de slide, dan kan `csd_sim_v2.py`
  een gecombineerde `public/csd_experiment.mp4` (2:1, 960x480, identieke state-hash) renderen en hoeft
  alleen de bestandsnaam in regel 103 te wijzigen.
- **Volgende stap:** (1) slide in `pnpm dev` visueel controleren; (2) eventueel de v2-composiet als
  `csd_experiment.mp4` renderen; (3) de PDF-export (`pnpm export`) als back-up gebruiken voor de QR
  (een geprojecteerde PDF verkleint de QR niet).
