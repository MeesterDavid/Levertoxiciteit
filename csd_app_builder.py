#!/usr/bin/env python3
"""
csd_app_builder.py - Cortical Spreading Depolarization (CSD) Simulator & HTML Builder.

Dit script genereert een standalone interactieve HTML/JS simulatie-omgeving
met periodieke torus-randvoorwaarden, video-opname en real-time parameters.
"""

import os
import webbrowser
from pathlib import Path

# ==============================================================================
# HTML, CSS & JAVASCRIPT TEMPLATE
# ==============================================================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Cortical Spreading Depolarization (CSD) Simulator</title>
<style>
  :root {
    --bg-main: #0B1120;
    --bg-card: #0F172A;
    --border-color: #1E293B;
    --text-main: #E2E8F0;
    --text-muted: #94A3B8;
    --accent-purple: #C026D3;
    --accent-yellow: #FACC15;
    --accent-blue: #38BDF8;
    --accent-green: #10B981;
    --color-darkred: #1A0404;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-main);
    color: var(--text-main);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 20px 24px;
    display: flex;
    justify-content: center;
  }

  .container {
    width: 100%;
    max-width: 1100px;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  /* Grid Panelen */
  .grid-dual {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
    font-weight: 600;
  }

  .badge {
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 6px;
    background: #1E293B;
    color: var(--text-muted);
    font-weight: 500;
  }

  /* Canvas Container */
  .canvas-wrapper {
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    background: #050101;
    border-radius: 10px;
    overflow: hidden;
    cursor: crosshair;
    border: 1px solid #1E293B;
  }

  canvas {
    width: 100%;
    height: 100%;
    display: block;
    image-rendering: pixelated;
  }

  .canvas-hint {
    position: absolute;
    bottom: 8px;
    left: 8px;
    font-size: 10px;
    background: rgba(0, 0, 0, 0.65);
    padding: 2px 6px;
    border-radius: 4px;
    color: var(--text-muted);
    pointer-events: none;
  }

  /* Sliders & Stats */
  .slider-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .slider-header {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    font-weight: 500;
  }

  input[type=range] {
    width: 100%;
    accent-color: #38BDF8;
    height: 5px;
    border-radius: 3px;
    background: #1E293B;
    outline: none;
  }

  .slider-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-muted);
  }

  .status-row {
    display: flex;
    justify-content: space-between;
    font-size: 11.5px;
    color: var(--text-muted);
    font-family: monospace;
  }

  /* Knoppenbalk */
  .toolbar {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 12px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 9px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: opacity 0.15s ease, transform 0.05s ease;
  }
  button:hover { opacity: 0.92; }
  button:active { transform: scale(0.98); }

  .btn-blue { background: #0284C7; color: white; }
  .btn-purple { background: #9333EA; color: white; }
  .btn-dark { background: #1E293B; color: #CBD5E1; }
  .btn-green { background: #059669; color: white; margin-left: auto; }
  .btn-red { background: #DC2626; color: white; margin-left: auto; }

  /* Info & Parameter Panelen */
  .panel-box {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 16px 18px;
    font-size: 12px;
  }

  .panel-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    color: #CBD5E1;
  }

  .legend-items {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .legend-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .param-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .param-desc {
    font-size: 10.5px;
    color: var(--text-muted);
    margin-top: 2px;
  }
</style>
</head>
<body>

<div class="container">

  <!-- 1. DUAL SIMULATIE GRID -->
  <div class="grid-dual">
    
    <!-- Conditie A: Kwetsbaarder weefsel -->
    <div class="card">
      <div class="card-header">
        <span style="color: #C084FC;">&#9679; Condition A: Vulnerable Tissue</span>
        <span class="badge" id="stateBadgeA">Quenched / Rest</span>
      </div>
      <div class="canvas-wrapper">
        <canvas id="canvasA" width="400" height="400"></canvas>
        <div class="canvas-hint">Click to trigger MIE</div>
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <span style="color: var(--accent-yellow);">&#9679; Inhibitor Density (Yellow)</span>
          <span id="labelDensityA">1.5%</span>
        </div>
        <input type="range" id="sliderDensityA" min="0" max="35" step="0.5" value="1.5" />
        <div class="slider-labels">
          <span>0% (No brake)</span>
          <span>Critical threshold ~12-14%</span>
          <span>35% (Dense)</span>
        </div>
      </div>
      <div class="status-row">
        <span>Active (Purple): <b id="valActA" style="color:#C084FC;">0.0%</b></span>
        <span>Inhibition (Yellow): <b id="valInhA" style="color:#FACC15;">0.0%</b></span>
      </div>
    </div>

    <!-- Conditie B: Weerbaarder weefsel -->
    <div class="card">
      <div class="card-header">
        <span style="color: var(--accent-yellow);">&#9679; Condition B: Resilient Tissue</span>
        <span class="badge" id="stateBadgeB">Quenched / Rest</span>
      </div>
      <div class="canvas-wrapper">
        <canvas id="canvasB" width="400" height="400"></canvas>
        <div class="canvas-hint">Click to trigger MIE</div>
      </div>
      <div class="slider-group">
        <div class="slider-header">
          <span style="color: var(--accent-yellow);">&#9679; Inhibitor Density (Yellow)</span>
          <span id="labelDensityB">15.5%</span>
        </div>
        <input type="range" id="sliderDensityB" min="0" max="35" step="0.5" value="15.5" />
        <div class="slider-labels">
          <span>0% (No brake)</span>
          <span>Critical threshold ~12-14%</span>
          <span>35% (Dense)</span>
        </div>
      </div>
      <div class="status-row">
        <span>Active (Purple): <b id="valActB" style="color:#C084FC;">0.0%</b></span>
        <span>Inhibition (Yellow): <b id="valInhB" style="color:#FACC15;">0.0%</b></span>
      </div>
    </div>

  </div>

  <!-- 2. TOOLBAR -->
  <div class="toolbar">
    <button class="btn-blue" id="btnPause">&#9654; Resume</button>
    <button class="btn-purple" id="btnTrigger">&#8853; Trigger Identical MIE (Pulse)</button>
    <button class="btn-dark" id="btnReset">&#8635; Reset</button>
    <button class="btn-green" id="btnRecord">&#9679; OK - Record Video & Download</button>
  </div>

  <!-- 3. LEGENDA -->
  <div class="panel-box">
    <div class="panel-title">LEGEND & CELLULAR STATES</div>
    <div class="legend-items">
      <div class="legend-row">
        <div class="legend-dot" style="background: #2A0404; border: 1px solid #7F1D1D;"></div>
        <div><b>Deep Dark Red:</b> Resting state of the cortex (neutral pyramidal cells in homeostatic balance).</div>
      </div>
      <div class="legend-row">
        <div class="legend-dot" style="background: var(--accent-yellow);"></div>
        <div><b>Bright Yellow (Inhibitor):</b> Fixed astrocytic buffers or refractory repolarizing cells behind the wavefront.</div>
      </div>
      <div class="legend-row">
        <div class="legend-dot" style="background: var(--accent-purple);"></div>
        <div><b>Bright Purple (Activator):</b> Propagating cortical depolarization front (massive K+ efflux / ionic breakdown).</div>
      </div>
    </div>
  </div>

  <!-- 4. REACTION-DIFFUSION PARAMETERS -->
  <div class="panel-box">
    <div class="panel-title">
      <span>REACTION-DIFFUSION PARAMETERS</span>
      <span style="cursor:pointer; color:#38BDF8;" id="btnDefaultParams">Reset Defaults</span>
    </div>
    <div class="param-grid">
      
      <div class="slider-group">
        <div class="slider-header">
          <span>K+ Diffusion Rate (D)</span>
          <span id="labelD">0.70</span>
        </div>
        <input type="range" id="sliderD" min="0.10" max="2.00" step="0.05" value="0.70" />
        <div class="param-desc">Determines how rapidly potassium leaks across the torus to adjacent columns.</div>
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span>Ignition Threshold (a)</span>
          <span id="labelA">0.015</span>
        </div>
        <input type="range" id="sliderA" min="0.005" max="0.100" step="0.005" value="0.015" />
        <div class="param-desc">Amount of local depolarization required to ignite self-sustaining positive feedback.</div>
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span>Buffering Strength (Yellow Cells)</span>
          <span id="labelP">0.14</span>
        </div>
        <input type="range" id="sliderP" min="0.02" max="0.40" step="0.01" value="0.14" />
        <div class="param-desc">Potassium clearance efficiency of individual astrocytic/pump sites.</div>
      </div>

      <div class="slider-group">
        <div class="slider-header">
          <span>MIE Stimulus Radius</span>
          <span id="labelR">3.5 cells</span>
        </div>
        <input type="range" id="sliderR" min="1.5" max="7.0" step="0.5" value="3.5" />
        <div class="param-desc">Spatial footprint of the primary hypoxic core / micro-injection.</div>
      </div>

    </div>
  </div>

</div>

<script>
// ==============================================================================
// BIO-COMPUTATIONAL ENGINE (FitzHugh-Nagumo / Tuckwell-Miura Lattice)
// ==============================================================================
const N = 40;            // 40x40 discrete cortical units (1600 cells)
const DT = 0.02;         // Integration time step
const SUBSTEPS = 8;      // Steps per animation frame

// Dynamic Parameters
let param_D = 0.70;
let param_A = 0.015;
let param_P = 0.14;
let param_R = 3.5;
let isRunning = false;
let continuousFocus = true; // Enables pacemaker focus for recurrent waves

// Tissue State Arrays: Condition A (Vulnerable) & Condition B (Resilient)
let uA = new Float64Array(N * N); // Activator [K+]e
let vA = new Float64Array(N * N); // Inhibitor load
let maskA = new Uint8Array(N * N); // Fixed inhibitor locations

let uB = new Float64Array(N * N);
let vB = new Float64Array(N * N);
let maskB = new Uint8Array(N * N);

const canvasA = document.getElementById("canvasA");
const canvasB = document.getElementById("canvasB");
const ctxA = canvasA.getContext("2d");
const ctxB = canvasB.getContext("2d");

// Pseudo-random tissue generator with deterministic seed
function generateMask(densityPct, seed) {
  const mask = new Uint8Array(N * N);
  let s = seed;
  for (let i = 0; i < N * N; i++) {
    s = (s * 9301 + 49297) % 233280;
    mask[i] = (s / 233280.0) < (densityPct / 100.0) ? 1 : 0;
  }
  return mask;
}

// Inject identical central MIE
function injectMIE() {
  const center = Math.floor(N / 2);
  const r2 = param_R * param_R;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const d2 = (x - center) * (x - center) + (y - center) * (y - center);
      if (d2 <= r2) {
        const idx = y * N + x;
        uA[idx] = 0.95;
        uB[idx] = 0.95;
      }
    }
  }
}

function resetSim() {
  const densA = parseFloat(document.getElementById("sliderDensityA").value);
  const densB = parseFloat(document.getElementById("sliderDensityB").value);
  maskA = generateMask(densA, 101);
  maskB = generateMask(densB, 202);
  uA.fill(0.0); vA.fill(0.0);
  uB.fill(0.0); vB.fill(0.0);
  injectMIE();
}

// Numerical Integration over Torus (Periodic Boundaries)
function stepModel(u, v, mask) {
  const lapU = new Float64Array(N * N);
  for (let y = 0; y < N; y++) {
    const yUp = (y + 1) % N;
    const yDown = (y - 1 + N) % N;
    for (let x = 0; x < N; x++) {
      const xRight = (x + 1) % N;
      const xLeft = (x - 1 + N) % N;
      const idx = y * N + x;
      lapU[idx] = u[y * N + xRight] + u[y * N + xLeft] + u[yUp * N + x] + u[yDown * N + x] - 4.0 * u[idx];
    }
  }

  const center = Math.floor(N / 2);
  for (let i = 0; i < N * N; i++) {
    const valU = u[i];
    const valV = v[i];
    
    // Non-linear cubic reaction + spatial diffusion
    let du = param_D * lapU[i] + valU * (1.0 - valU) * (valU - param_A) - valV;
    
    // Fixed buffer cells actively pump away K+
    if (mask[i]) {
      du -= param_P * valU;
    }

    // Continuous hypoxic core pacemaker (triggers recurrent waves when recovered)
    if (continuousFocus) {
      const cx = i % N;
      const cy = Math.floor(i / N);
      const dCenter2 = (cx - center) * (cx - center) + (cy - center) * (cy - center);
      if (dCenter2 <= 3.0 * 3.0) {
        du += 0.035; // Chronic focal potassium leakage
      }
    }

    // Inhibitor repolarization kinetics
    let dv = 0.012 * (1.2 * valU - 0.45 * valV);

    u[i] = Math.max(0.0, Math.min(1.2, valU + du * DT));
    v[i] = Math.max(0.0, Math.min(1.2, valV + dv * DT));
  }
}

// Render Tissue Lattice as Discrete Dots
function renderGrid(ctx, u, v, mask) {
  const cellSize = canvasA.width / N;
  const radius = cellSize * 0.44;
  ctx.fillStyle = "#050101";
  ctx.fillRect(0, 0, canvasA.width, canvasA.height);

  let activeCount = 0;
  let inhCount = 0;

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const idx = y * N + x;
      const valU = u[idx];
      const valV = v[idx];
      const isBuffer = mask[idx];

      let col = "#220303"; // Neutral resting cortex (Deep Dark Red)

      if (valU >= 0.20) {
        col = "#C026D3"; // Depolarization wavefront (Bright Purple)
        activeCount++;
      } else if (isBuffer || valV >= 0.035) {
        col = "#FACC15"; // Inhibitor active/buffer (Bright Yellow)
        inhCount++;
      }

      ctx.beginPath();
      ctx.arc(x * cellSize + cellSize / 2, y * cellSize + cellSize / 2, radius, 0, 2 * Math.PI);
      ctx.fillStyle = col;
      ctx.fill();
    }
  }

  return {
    actPct: ((activeCount / (N * N)) * 100).toFixed(1),
    inhPct: ((inhCount / (N * N)) * 100).toFixed(1)
  };
}

// Main Animation Loop
function loop() {
  if (isRunning) {
    for (let s = 0; s < SUBSTEPS; s++) {
      stepModel(uA, vA, maskA);
      stepModel(uB, vB, maskB);
    }
  }

  const statsA = renderGrid(ctxA, uA, vA, maskA);
  const statsB = renderGrid(ctxB, uB, vB, maskB);

  document.getElementById("valActA").innerText = statsA.actPct + "%";
  document.getElementById("valInhA").innerText = statsA.inhPct + "%";
  document.getElementById("valActB").innerText = statsB.actPct + "%";
  document.getElementById("valInhB").innerText = statsB.inhPct + "%";

  document.getElementById("stateBadgeA").innerText = parseFloat(statsA.actPct) > 1.0 ? "Spreading Wave" : "Quenched / Rest";
  document.getElementById("stateBadgeB").innerText = parseFloat(statsB.actPct) > 1.0 ? "Spreading Wave" : "Quenched / Rest";

  requestAnimationFrame(loop);
}

// ==============================================================================
// UI EVENT HANDLERS & SLIDERS
// ==============================================================================
const btnPause = document.getElementById("btnPause");
btnPause.onclick = () => {
  isRunning = !isRunning;
  btnPause.innerText = isRunning ? "❚❚ Pause" : "▶ Resume";
};

document.getElementById("btnTrigger").onclick = () => {
  injectMIE();
};

document.getElementById("btnReset").onclick = () => {
  resetSim();
};

// Interactive Canvas Click
function addClickStimulus(e, uArr) {
  const rect = e.target.getBoundingClientRect();
  const x = Math.floor(((e.clientX - rect.left) / rect.width) * N);
  const y = Math.floor(((e.clientY - rect.top) / rect.height) * N);
  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      if (dx * dx + dy * dy <= 9) {
        const nx = (x + dx + N) % N;
        const ny = (y + dy + N) % N;
        uArr[ny * N + nx] = 0.95;
      }
    }
  }
}
canvasA.onclick = (e) => addClickStimulus(e, uA);
canvasB.onclick = (e) => addClickStimulus(e, uB);

// Density Sliders
document.getElementById("sliderDensityA").oninput = (e) => {
  document.getElementById("labelDensityA").innerText = parseFloat(e.target.value).toFixed(1) + "%";
  maskA = generateMask(parseFloat(e.target.value), 101);
};

document.getElementById("sliderDensityB").oninput = (e) => {
  document.getElementById("labelDensityB").innerText = parseFloat(e.target.value).toFixed(1) + "%";
  maskB = generateMask(parseFloat(e.target.value), 202);
};

// Parameter Sliders
document.getElementById("sliderD").oninput = (e) => {
  param_D = parseFloat(e.target.value);
  document.getElementById("labelD").innerText = param_D.toFixed(2);
};
document.getElementById("sliderA").oninput = (e) => {
  param_A = parseFloat(e.target.value);
  document.getElementById("labelA").innerText = param_A.toFixed(3);
};
document.getElementById("sliderP").oninput = (e) => {
  param_P = parseFloat(e.target.value);
  document.getElementById("labelP").innerText = param_P.toFixed(2);
};
document.getElementById("sliderR").oninput = (e) => {
  param_R = parseFloat(e.target.value);
  document.getElementById("labelR").innerText = param_R.toFixed(1) + " cells";
};

document.getElementById("btnDefaultParams").onclick = () => {
  document.getElementById("sliderD").value = 0.70; param_D = 0.70; document.getElementById("labelD").innerText = "0.70";
  document.getElementById("sliderA").value = 0.015; param_A = 0.015; document.getElementById("labelA").innerText = "0.015";
  document.getElementById("sliderP").value = 0.14; param_P = 0.14; document.getElementById("labelP").innerText = "0.14";
  document.getElementById("sliderR").value = 3.5; param_R = 3.5; document.getElementById("labelR").innerText = "3.5 cells";
  document.getElementById("sliderDensityA").value = 1.5; document.getElementById("labelDensityA").innerText = "1.5%";
  document.getElementById("sliderDensityB").value = 15.5; document.getElementById("labelDensityB").innerText = "15.5%";
  resetSim();
};

// ==============================================================================
// SYNCHRONIZED WEBM VIDEO RECORDER
// ==============================================================================
const btnRecord = document.getElementById("btnRecord");
let mediaRecorder = null;
let recordedChunks = [];

btnRecord.onclick = () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }

  // Combined off-screen canvas
  const recCanvas = document.createElement("canvas");
  recCanvas.width = 820;
  recCanvas.height = 440;
  const rCtx = recCanvas.getContext("2d");

  const stream = recCanvas.captureStream(30);
  mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9" });
  recordedChunks = [];

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    clearInterval(recordInterval);
    btnRecord.innerText = "● OK - Record Video & Download";
    btnRecord.className = "btn-green";

    const blob = new Blob(recordedChunks, { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "csd_simulation_export.webm";
    a.click();
    URL.revokeObjectURL(url);
  };

  btnRecord.innerText = "■ Stop Recording & Save";
  btnRecord.className = "btn-red";

  // Re-trigger fresh MIE upon recording start
  injectMIE();
  if (!isRunning) {
    isRunning = true;
    btnPause.innerText = "❚❚ Pause";
  }

  mediaRecorder.start();

  const recordInterval = setInterval(() => {
    rCtx.fillStyle = "#0B1120";
    rCtx.fillRect(0, 0, recCanvas.width, recCanvas.height);
    
    // Draw Both Grids side-by-side
    rCtx.drawImage(canvasA, 10, 30, 390, 390);
    rCtx.drawImage(canvasB, 420, 30, 390, 390);

    // Headers & Labels
    rCtx.fillStyle = "#C084FC";
    rCtx.font = "bold 13px sans-serif";
    rCtx.fillText("Condition A (1.5% Inhibitor)", 10, 20);

    rCtx.fillStyle = "#FACC15";
    rCtx.fillText("Condition B (15.5% Inhibitor)", 420, 20);
  }, 1000 / 30);
};

// Initialize Application
resetSim();
loop();
</script>
</body>
</html>
"""

# ==============================================================================
# MAIN EXECUTION (File creation & launch)
# ==============================================================================
def main():
    output_dir = Path(__file__).resolve().parent / "public"
    output_dir.mkdir(parents=True, exist_ok=True)
    html_file = output_dir / "csd_interactive_simulator.html"

    print("=" * 80)
    print("CSD SIMULATOR & INTERACTIVE WEB BUILDER")
    print("=" * 80)
    print(f"Schrijven van interactieve HTML-applicatie naar:\n-> {html_file}")
    
    html_file.write_text(HTML_CONTENT, encoding="utf-8")
    print(f"Bestandsgrootte: {html_file.stat().st_size / 1024:.1f} kB")
    print("\nAutomatisch openen in standaardbrowser...")
    
    webbrowser.open(html_file.as_uri())
    print("Klaar! Pas de parameters direct aan in de browser of start de opname.")

if __name__ == "__main__":
    main()