"""
Generates visual assets and placeholders in `public/` for Slide 1 of the 3-minute pitch:
- `public/aop_mini.svg`                  (Clear miniature schematic of a linear AOP: MIE -> KE -> AO)
- `public/physiological_map_mini.svg`    (Clear miniature schematic of a non-linear Physiological Map with feedback loops)
- `public/reaction_diffusion.gif`        (Animated 2D reaction-diffusion traveling wave representing tissue-level emergence)
- `public/nam_overview_placeholder.svg`  (Visual placeholder for `/public/nam_overview.png`)
- `public/csd_center_placeholder.svg`    (Visual placeholder for `/public/csd_center.png`)
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

PUBLIC_DIR = Path(__file__).resolve().parent / "public"
PUBLIC_DIR.mkdir(exist_ok=True)


def create_aop_mini_svg(out_path: Path) -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 150" width="100%" height="100%">
  <defs>
    <marker id="arrow-aop" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#94A3B8"/>
    </marker>
  </defs>
  <rect width="360" height="150" rx="14" fill="#0F172A" stroke="#38BDF8" stroke-width="2"/>
  <text x="180" y="26" text-anchor="middle" fill="#38BDF8" font-family="Inter, sans-serif" font-size="13" font-weight="700">(q)AOP — Lineair &amp; Reductionistisch</text>

  <!-- MIE -->
  <rect x="18" y="48" width="68" height="48" rx="8" fill="#1E293B" stroke="#10B981" stroke-width="2"/>
  <text x="52" y="70" text-anchor="middle" fill="#10B981" font-family="Inter, sans-serif" font-size="13" font-weight="800">MIE</text>
  <text x="52" y="85" text-anchor="middle" fill="#CBD5E1" font-family="Inter, sans-serif" font-size="9">Ionkanaal</text>

  <line x1="88" y1="72" x2="108" y2="72" stroke="#94A3B8" stroke-width="2.2" marker-end="url(#arrow-aop)"/>

  <!-- KE 1 -->
  <rect x="112" y="48" width="60" height="48" rx="8" fill="#1E293B" stroke="#38BDF8" stroke-width="2"/>
  <text x="142" y="70" text-anchor="middle" fill="#38BDF8" font-family="Inter, sans-serif" font-size="12" font-weight="700">KE 1</text>
  <text x="142" y="85" text-anchor="middle" fill="#CBD5E1" font-family="Inter, sans-serif" font-size="9">Cel</text>

  <line x1="174" y1="72" x2="194" y2="72" stroke="#94A3B8" stroke-width="2.2" marker-end="url(#arrow-aop)"/>

  <!-- KE 2 -->
  <rect x="198" y="48" width="60" height="48" rx="8" fill="#1E293B" stroke="#F59E0B" stroke-width="2"/>
  <text x="228" y="70" text-anchor="middle" fill="#F59E0B" font-family="Inter, sans-serif" font-size="12" font-weight="700">KE 2</text>
  <text x="228" y="85" text-anchor="middle" fill="#CBD5E1" font-family="Inter, sans-serif" font-size="9">Weefsel</text>

  <line x1="260" y1="72" x2="280" y2="72" stroke="#94A3B8" stroke-width="2.2" marker-end="url(#arrow-aop)"/>

  <!-- AO -->
  <rect x="284" y="48" width="58" height="48" rx="8" fill="#1E293B" stroke="#F43F5E" stroke-width="2"/>
  <text x="313" y="70" text-anchor="middle" fill="#F43F5E" font-family="Inter, sans-serif" font-size="13" font-weight="800">AO</text>
  <text x="313" y="85" text-anchor="middle" fill="#CBD5E1" font-family="Inter, sans-serif" font-size="9">Schade</text>

  <text x="180" y="128" text-anchor="middle" fill="#94A3B8" font-family="Inter, sans-serif" font-size="10.5">Mist niet-lineaire feedback &amp; weefsel-emergentie</text>
</svg>"""
    out_path.write_text(svg, encoding="utf-8")


def create_phys_map_mini_svg(out_path: Path) -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 150" width="100%" height="100%">
  <defs>
    <marker id="arrow-act" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#10B981"/>
    </marker>
    <marker id="arrow-inh" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#F43F5E"/>
    </marker>
  </defs>
  <rect width="360" height="150" rx="14" fill="#0F172A" stroke="#10B981" stroke-width="2"/>
  <text x="180" y="25" text-anchor="middle" fill="#10B981" font-family="Inter, sans-serif" font-size="13" font-weight="700">Physiological Map — Feedbackloops</text>

  <!-- Edges & Feedback loops -->
  <path d="M 85 65 Q 140 35 190 58" fill="none" stroke="#10B981" stroke-width="2" marker-end="url(#arrow-act)"/>
  <path d="M 190 78 Q 140 105 85 78" fill="none" stroke="#F43F5E" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrow-inh)"/>
  <path d="M 218 65 Q 255 40 285 65" fill="none" stroke="#10B981" stroke-width="2" marker-end="url(#arrow-act)"/>
  <path d="M 285 82 Q 250 112 215 82" fill="none" stroke="#F43F5E" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrow-inh)"/>
  <line x1="138" y1="95" x2="188" y2="75" stroke="#10B981" stroke-width="1.8" marker-end="url(#arrow-act)"/>
  <line x1="82" y1="82" x2="115" y2="98" stroke="#10B981" stroke-width="1.8" marker-end="url(#arrow-act)"/>

  <!-- Nodes -->
  <circle cx="68" cy="72" r="16" fill="#1E293B" stroke="#38BDF8" stroke-width="2.2"/>
  <text x="68" y="76" text-anchor="middle" fill="#E2E8F0" font-family="Inter, sans-serif" font-size="9.5" font-weight="700">Prikkel</text>

  <circle cx="128" cy="102" r="14" fill="#1E293B" stroke="#A855F7" stroke-width="2"/>
  <text x="128" y="105" text-anchor="middle" fill="#E2E8F0" font-family="Inter, sans-serif" font-size="8.5">Buffer</text>

  <circle cx="204" cy="70" r="18" fill="#1E293B" stroke="#10B981" stroke-width="2.5"/>
  <text x="204" y="73" text-anchor="middle" fill="#10B981" font-family="Inter, sans-serif" font-size="9.5" font-weight="700">Homeo</text>

  <circle cx="298" cy="72" r="16" fill="#1E293B" stroke="#F59E0B" stroke-width="2.2"/>
  <text x="298" y="75" text-anchor="middle" fill="#E2E8F0" font-family="Inter, sans-serif" font-size="9" font-weight="700">Flux</text>

  <text x="180" y="134" text-anchor="middle" fill="#94A3B8" font-family="Inter, sans-serif" font-size="10.5">Niet-lineaire terugkoppeling &amp; adaptieve compensatie</text>
</svg>"""
    out_path.write_text(svg, encoding="utf-8")


def create_reaction_diffusion_gif(out_path: Path) -> None:
    """
    Generates a smooth animated 2D reaction-diffusion / concentric spiral wave GIF
    with a superimposed Multi-Electrode Array (MEA) grid to visually represent
    tissue-level spatial emergence (like Cortical Spreading Depression).
    """
    width, height = 220, 140
    y, x = np.ogrid[:height, :width]
    cx, cy = width * 0.42, height * 0.52
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = np.arctan2(y - cy, x - cx)

    frames = []
    n_frames = 36
    for i in range(n_frames):
        phase = 2.0 * np.pi * (i / n_frames)
        # Reaction-diffusion spiral/target traveling wave
        wave1 = np.sin(0.16 * r - phase + 0.5 * np.sin(theta))
        wave2 = np.cos(0.11 * np.sqrt((x - width * 0.7) ** 2 + (y - height * 0.35) ** 2) - phase)
        u = 0.65 * wave1 + 0.35 * wave2
        norm = (u + 1.0) * 0.5  # [0, 1]

        # Colorize using dark navy -> cyan -> amber -> crimson wave front
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip(15 + 230 * (norm ** 2.2), 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip(23 + 175 * np.sin(norm * np.pi), 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip(42 + 200 * (1.0 - norm * 0.6), 0, 255).astype(np.uint8)

        img = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(img)

        # Draw subtle MEA (Multi-Electrode Array) recording grid dots
        for gx in range(25, width - 15, 28):
            for gy in range(22, height - 15, 26):
                draw.ellipse((gx - 2, gy - 2, gx + 2, gy + 2), fill=(226, 232, 240))

        # Border
        draw.rectangle((0, 0, width - 1, height - 1), outline=(168, 85, 247), width=2)
        frames.append(img)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=55,
        loop=0,
        optimize=True,
    )


def create_nam_placeholder_svg(out_path: Path) -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 230" width="100%" height="100%">
  <rect width="420" height="230" rx="16" fill="#0F172A" stroke="#38BDF8" stroke-width="2" stroke-dasharray="7,5"/>
  <text x="210" y="34" text-anchor="middle" fill="#38BDF8" font-family="Inter, sans-serif" font-size="14" font-weight="800">New Approach Methodologies (NAMs)</text>
  <text x="210" y="54" text-anchor="middle" fill="#94A3B8" font-family="Inter, sans-serif" font-size="11">Dierproefvrije &amp; Probabilistische Risicobeoordeling</text>

  <!-- In Vitro box -->
  <rect x="28" y="75" width="110" height="82" rx="10" fill="#1E293B" stroke="#10B981" stroke-width="2"/>
  <text x="83" y="106" text-anchor="middle" fill="#10B981" font-family="Inter, sans-serif" font-size="13" font-weight="700">In Vitro</text>
  <text x="83" y="126" text-anchor="middle" fill="#CBD5E1" font-family="Inter, sans-serif" font-size="10">Humane cellen</text>
  <text x="83" y="141" text-anchor="middle" fill="#CBD5E1" font-family="Inter, sans-serif" font-size="10">MEA / Omics</text>

  <!-- Plus -->
  <text x="155" y="122" text-anchor="middle" fill="#64748B" font-family="Inter, sans-serif" font-size="20" font-weight="800">+</text>

  <!-- In Silico box -->
  <rect x="172" y="75" width="110" height="82" rx="10" fill="#1E293B" stroke="#38BDF8" stroke-width="2"/>
  <text x="227" y="106" text-anchor="middle" fill="#38BDF8" font-family="Inter, sans-serif" font-size="13" font-weight="700">In Silico</text>
  <text x="227" y="126" text-anchor="middle" fill="#CBD5E1" font-family="Inter, sans-serif" font-size="10">(q)AOPs &amp; Maps</text>
  <text x="227" y="141" text-anchor="middle" fill="#CBD5E1" font-family="Inter, sans-serif" font-size="10">AI Modellen</text>

  <!-- Arrow -->
  <text x="298" y="122" text-anchor="middle" fill="#64748B" font-family="Inter, sans-serif" font-size="18" font-weight="800">→</text>

  <!-- PCRA box -->
  <rect x="312" y="75" width="84" height="82" rx="10" fill="#1E293B" stroke="#A855F7" stroke-width="2"/>
  <text x="354" y="106" text-anchor="middle" fill="#A855F7" font-family="Inter, sans-serif" font-size="12" font-weight="700">PCRA</text>
  <text x="354" y="126" text-anchor="middle" fill="#CBD5E1" font-family="Inter, sans-serif" font-size="9.5">Dierproefvrij</text>
  <text x="354" y="141" text-anchor="middle" fill="#CBD5E1" font-family="Inter, sans-serif" font-size="9.5">Probabilistisch</text>

  <rect x="45" y="176" width="330" height="34" rx="8" fill="#1E293B" stroke="#475569" stroke-width="1"/>
  <text x="210" y="197" text-anchor="middle" fill="#FBBF24" font-family="monospace" font-size="11">Placeholder → plaats eigen beeld als /public/nam_overview.png</text>
</svg>"""
    out_path.write_text(svg, encoding="utf-8")


def create_csd_center_placeholder_svg(out_path: Path) -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 220" width="100%" height="100%">
  <defs>
    <radialGradient id="csd-wave" cx="50%" cy="45%" r="50%">
      <stop offset="0%" stop-color="#F43F5E" stop-opacity="0.85"/>
      <stop offset="45%" stop-color="#F59E0B" stop-opacity="0.55"/>
      <stop offset="75%" stop-color="#38BDF8" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#0F172A" stop-opacity="0.0"/>
    </radialGradient>
  </defs>
  <rect width="260" height="220" rx="18" fill="#0F172A" stroke="#F43F5E" stroke-width="2.5" stroke-dasharray="6,4"/>

  <!-- Stylized cortex + spreading depolarization wave -->
  <ellipse cx="130" cy="92" rx="76" ry="54" fill="#1E293B" stroke="#475569" stroke-width="2"/>
  <circle cx="130" cy="92" r="52" fill="url(#csd-wave)"/>
  <circle cx="130" cy="92" r="24" fill="none" stroke="#F43F5E" stroke-width="2" stroke-dasharray="4,3"/>
  <circle cx="130" cy="92" r="42" fill="none" stroke="#F59E0B" stroke-width="1.8" stroke-dasharray="5,3"/>

  <text x="130" y="168" text-anchor="middle" fill="#F8FAFC" font-family="Inter, sans-serif" font-size="20" font-weight="900">CSD</text>
  <text x="130" y="185" text-anchor="middle" fill="#CBD5E1" font-family="Inter, sans-serif" font-size="10">Cortical Spreading Depression</text>
  <text x="130" y="204" text-anchor="middle" fill="#FBBF24" font-family="monospace" font-size="9.5">/public/csd_center.png</text>
</svg>"""
    out_path.write_text(svg, encoding="utf-8")


def create_standalone_slide1_preview(out_path: Path) -> None:
    """
    Creates a standalone HTML preview (`public/slide1_pitch_preview.html`)
    that renders Slide 1 in 16:9 widescreen plus the speaker notes and
    placeholder overview directly in the browser without needing a running Node server.
    """
    html = """<!DOCTYPE html>
<html lang="nl">
<head>
  <meta charset="UTF-8" />
  <title>Pitch Slide 1 Preview — Multimodal &amp; Multiscale AI in de Toxicologie</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 24px;
      background: #060911;
      color: #f8fafc;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 24px;
    }
    .slide-frame {
      width: 1120px;
      height: 630px;
      background: radial-gradient(circle at 15% 20%, #0f172a 0%, #090d16 100%);
      border: 1px solid #334155;
      border-radius: 16px;
      padding: 18px 24px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.65);
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .header-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #1e293b;
      padding-bottom: 8px;
    }
    .eyebrow {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #38bdf8;
      font-weight: 700;
    }
    .main-title {
      font-size: 23px;
      font-weight: 800;
      margin: 2px 0 0 0;
      color: #f8fafc;
    }
    .header-pill {
      background: #1e293b;
      border: 1px solid #334155;
      color: #cbd5e1;
      font-size: 11.5px;
      font-weight: 600;
      padding: 4px 12px;
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
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .panel-header {
      display: flex;
      align-items: center;
      gap: 9px;
      margin-bottom: 6px;
    }
    .badge {
      font-size: 11px;
      font-weight: 800;
      padding: 3px 9px;
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
      font-size: 12.5px;
      color: #cbd5e1;
      font-weight: 600;
    }
    .nam-visual-wrap {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 0;
      margin: 4px 0;
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
      gap: 8px;
    }
    .kw-card {
      display: flex;
      align-items: flex-start;
      gap: 9px;
      background: #0b1120;
      border: 1px solid #1e293b;
      border-radius: 9px;
      padding: 8px 11px;
    }
    .kw-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      margin-top: 4px;
      flex-shrink: 0;
    }
    .bg-cyan { background: #38bdf8; box-shadow: 0 0 8px #38bdf8; }
    .bg-emerald { background: #10b981; box-shadow: 0 0 8px #10b981; }
    .kw-title { font-size: 12px; font-weight: 700; color: #f1f5f9; }
    .kw-desc { font-size: 10.5px; color: #94a3b8; line-height: 1.3; }

    .hub-canvas {
      position: relative;
      flex: 1;
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
      width: 205px;
      background: rgba(15, 23, 42, 0.94);
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 6px 8px;
      z-index: 2;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
    }
    .spoke-aop { top: 14px; left: 14px; border-color: rgba(56, 189, 248, 0.45); }
    .spoke-phys { bottom: 14px; left: 14px; border-color: rgba(16, 185, 129, 0.45); }
    .spoke-rd {
      top: 50%;
      right: 14px;
      transform: translateY(-50%);
      width: 205px;
      border-color: rgba(168, 85, 247, 0.55);
    }
    .spoke-tag {
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      margin-bottom: 4px;
    }
    .text-cyan { color: #38bdf8; }
    .text-emerald { color: #10b981; }
    .text-purple { color: #c084fc; }
    .spoke-img { width: 100%; height: 84px; object-fit: contain; border-radius: 6px; }
    .rd-gif { height: 96px; object-fit: cover; border: 1px solid rgba(168, 85, 247, 0.4); }
    .spoke-caption { font-size: 10px; color: #e2e8f0; line-height: 1.25; margin-top: 4px; }
    .sub-note { font-size: 9px; color: #94a3b8; }

    .center-csd-card {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 148px;
      background: rgba(15, 23, 42, 0.96);
      border: 2px solid #f43f5e;
      border-radius: 14px;
      padding: 7px;
      z-index: 3;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      box-shadow: 0 0 28px rgba(244, 63, 94, 0.28);
    }
    .csd-img { width: 120px; height: 102px; object-fit: contain; border-radius: 8px; }
    .csd-label { font-size: 17px; font-weight: 900; color: #f8fafc; letter-spacing: 0.06em; margin-top: 3px; }
    .csd-sublabel { font-size: 9px; color: #fda4af; font-weight: 600; }

    .claim-banner {
      margin-top: 8px;
      background: linear-gradient(90deg, rgba(56, 189, 248, 0.14) 0%, rgba(168, 85, 247, 0.16) 100%);
      border: 1px solid rgba(168, 85, 247, 0.4);
      border-radius: 9px;
      padding: 7px 12px;
      display: flex;
      align-items: center;
      gap: 9px;
    }
    .claim-icon { font-size: 16px; color: #c084fc; }
    .claim-text { font-size: 11.5px; color: #f1f5f9; line-height: 1.35; }

    .notes-box {
      width: 1120px;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 18px 24px;
      line-height: 1.6;
      font-size: 14px;
      color: #e2e8f0;
    }
    .notes-box h2 { margin-top: 0; font-size: 16px; color: #38bdf8; }
    .notes-box code { background: #1e293b; padding: 2px 6px; border-radius: 4px; color: #fbbf24; }
  </style>
</head>
<body>

  <div class="slide-frame">
    <div class="header-bar">
      <div>
        <div class="eyebrow">Onderzoeksvoorstel Pitch · Innovative Testing</div>
        <h1 class="main-title">Multimodal &amp; Multiscale AI in de Toxicologie</h1>
      </div>
      <div class="header-pill">Slide 1 / 2 · Context &amp; De Kwestie</div>
    </div>

    <div class="main-grid">
      <!-- LEFT COLUMN -->
      <div class="panel">
        <div class="panel-header">
          <span class="badge badge-blue">1. De Context</span>
          <span class="panel-subtitle">Dierproefvrije &amp; Probabilistische Risicobeoordeling</span>
        </div>

        <div class="nam-visual-wrap">
          <img
            src="nam_overview.png"
            onerror="this.onerror=null;this.src='nam_overview_placeholder.svg';"
            alt="New Approach Methodologies (NAMs)"
            class="nam-img"
          />
        </div>

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

      <!-- RIGHT COLUMN -->
      <div class="panel">
        <div class="panel-header">
          <span class="badge badge-rose">2. De Kwestie</span>
          <span class="panel-subtitle">Van Reductionisme naar Multiscale Emergentie &amp; AI-Integratie</span>
        </div>

        <div class="hub-canvas">
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
            <line x1="205" y1="82" x2="252" y2="128" stroke="#38BDF8" stroke-width="2.8" stroke-dasharray="5,3" marker-end="url(#arr-cyan)" />
            <line x1="205" y1="258" x2="252" y2="212" stroke="#10B981" stroke-width="2.8" stroke-dasharray="5,3" marker-end="url(#arr-emerald)" />
            <line x1="435" y1="170" x2="388" y2="170" stroke="#A855F7" stroke-width="2.8" stroke-dasharray="5,3" marker-end="url(#arr-purple)" />
          </svg>

          <div class="spoke-card spoke-aop">
            <div class="spoke-tag text-cyan">(q)AOP · Moleculair</div>
            <img src="aop_mini.png" onerror="this.onerror=null;this.src='aop_mini.svg';" alt="AOP Mini" class="spoke-img" />
            <div class="spoke-caption">
              Sterk reductionistisch, maar <strong>lineair</strong><br/>
              <span class="sub-note">(MIE = Ion channel disruption)</span>
            </div>
          </div>

          <div class="spoke-card spoke-phys">
            <div class="spoke-tag text-emerald">Physiological Map · Cel</div>
            <img src="physiological_map.png" onerror="this.onerror=null;this.src='physiological_map_mini.svg';" alt="Physiological Map Mini" class="spoke-img" />
            <div class="spoke-caption">
              <strong>Non-lineaire</strong> feedbackloops &amp; homeostase<br/>
              <span class="sub-note">(ook weefsel-IE zoals asbest)</span>
            </div>
          </div>

          <div class="center-csd-card">
            <img src="csd_center.png" onerror="this.onerror=null;this.src='csd_center_placeholder.svg';" alt="CSD" class="csd-img" />
            <div class="csd-label">CSD</div>
            <div class="csd-sublabel">Emergente Adverse Outcome</div>
          </div>

          <div class="spoke-card spoke-rd">
            <div class="spoke-tag text-purple">Reaction-Diffusion &amp; MEA · Weefsel</div>
            <img src="reaction_diffusion.gif" alt="Reaction-Diffusion Wave & MEA" class="spoke-img rd-gif" />
            <div class="spoke-caption">
              <strong>Tissue-level emergence</strong><br/>
              <span class="sub-note">Ruimtelijk-temporele MEA-data</span>
            </div>
          </div>
        </div>

        <div class="claim-banner">
          <div class="claim-icon">✦</div>
          <div class="claim-text">
            <strong>De Claim:</strong> Multimodale &amp; multiscale data (<em>AOP + Physiological Maps + MEA</em>) vereist <strong>AI</strong> om schalen te verbinden en het kantelpunt naar een <em>Adverse Outcome</em> te voorspellen.
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="notes-box">
    <h2>🎤 Spreektekst / Presenter Notes bij Slide 1 (± 90–100 seconden)</h2>
    <p><strong>[1. De Context — Links op de slide]</strong><br/>
    "Goedemiddag. Binnen ons onderzoeksveld werken we aan <em>Probabilistic Chemical Risk Assessment</em> met behulp van <em>New Approach Methodologies</em> (NAMs). Daarbij zetten we geavanceerde humane <em>in vitro</em> en <em>in silico</em> modellen in om chemische risicobeoordeling dierproefvrij én mechanistisch onderbouwd te maken. Twee belangrijke pijlers hierin zijn <em>(quantitative) Adverse Outcome Pathways</em> — (q)AOP's — en <em>Physiological Maps</em>, waarin we vastleggen en verklaren hoe chemische blootstelling tot een <em>Adverse Outcome</em> leidt."</p>

    <p><strong>[2. De Kwestie — Rechts op de slide]</strong><br/>
    "Maar precies daar ligt een fundamentele uitdaging:<br/>
    • <strong>Ten eerste:</strong> AOP's zijn erg sterk voor een reductionistische aanpak, maar ze zijn in de basis <em>lineair</em>. Er is al werk gaande om ze kwantitatief te maken (qAOP's), maar daar ligt nog veel werk — temeer omdat biologische data inherent multimodaal is.<br/>
    • <strong>Ten tweede:</strong> Biologie is niet lineair; gezonde systemen bevatten <em>homeostatische feedbackloops</em>. Bovendien grijpen sommige <em>initiating events</em> niet aan op één receptor, maar direct op weefselniveau — denk aan de fysische verstoring door <em>asbest</em>. Zulke niet-lineaire netwerken worden gedekt door <em>Physiological Maps</em>.<br/>
    • <strong>Ten derde:</strong> Soms vertelt een <em>Molecular Initiating Event</em> op zichzelf nog onvoldoende over of de <em>Adverse Outcome</em> daadwerkelijk optreedt, zoals bij <strong>Cortical Spreading Depression (CSD)</strong> hier in het midden. Daar is wel een MIE — <em>ion channel disruption</em> — maar of de Adverse Outcome daadwerkelijk optreedt, hangt af van <em>tissue-level emergence</em>: een ruimtelijk, <em>reaction-diffusion</em>-achtig golfproces over het weefsel, dat we bijvoorbeeld meten met <em>MEA-data</em> (elektrisch signaal over een heel gebied).<br/>
    • <strong>Mijn claim:</strong> Om zo'n Adverse Outcome te kunnen voorspellen, moeten we al deze heterogene data — van AOP's, van Physiological Maps én ruimtelijke MEA-data — samenbrengen. Dat is enorm veel en veelsoortige data, en mijn claim is dat <strong>AI uitermate geschikt is om dit over schalen en modaliteiten heen samen te brengen</strong>."</p>
  </div>

</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    aop_path = PUBLIC_DIR / "aop_mini.svg"
    phys_path = PUBLIC_DIR / "physiological_map_mini.svg"
    rd_path = PUBLIC_DIR / "reaction_diffusion.gif"
    nam_path = PUBLIC_DIR / "nam_overview_placeholder.svg"
    csd_path = PUBLIC_DIR / "csd_center_placeholder.svg"
    preview_path = PUBLIC_DIR / "slide1_pitch_preview.html"

    create_aop_mini_svg(aop_path)
    create_phys_map_mini_svg(phys_path)
    create_reaction_diffusion_gif(rd_path)
    create_nam_placeholder_svg(nam_path)
    create_csd_center_placeholder_svg(csd_path)
    create_standalone_slide1_preview(preview_path)

    for p in (aop_path, phys_path, rd_path, nam_path, csd_path, preview_path):
        print(f"Generated: {p.name} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
