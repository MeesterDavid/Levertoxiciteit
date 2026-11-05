"""
Scientific Pipeline & Model Architecture Diagram Generator (Milestone 0)
========================================================================
Generates two publication-grade schematic pipeline figures (Left-to-Right pipeline,
Top-to-Bottom internal stacks for Data, Preprocessing, AI Model Layers, and Output):
  1. Figure 1: Microsoft Graphormer (`clefourrier/pcqm4mv2_graphormer_base`)
     -> `public/pipeline_microsoft_graphormer.svg`
  2. Figure 2: BioBricks ToxTransformer (`biobricks-ai/toxtransformer` MoE)
     -> `public/pipeline_toxtransformer.svg`
  3. Standalone interactive HTML viewer with both vector SVGs and Mermaid.js definitions:
     -> `public/00_architecture_pipelines.html`
"""

from __future__ import annotations

import html as html_lib
from pathlib import Path
from typing import Any


MERMAID_GRAPHORMER = """flowchart LR
    subgraph DATA["1. Data & Pretrained Bronnen (Boven → Beneden)"]
        direction TB
        D1["Referentie Stoffenpaneel (B=12 SMILES)<br/>ChEMBL / PubChem / DILIrank Referenties"]
        D2["OECD AOP-Wiki MIE Annotaties<br/>• Klasse 0: Geen MIE / Adaptief (n=4)<br/>• Klasse 1: MIE 167 LXRα → Steatose AOP 34 (n=4)<br/>• Klasse 2: MIE 41 BSEP → Cholestase AOP 27 (n=4)"]
        D3["Pretrained Checkpoint (Hugging Face)<br/>clefourrier/pcqm4mv2_graphormer_base<br/>Getraind op OGB-LSC PCQM4Mv2 (3.7M moleculen)"]
        D1 --> D2 --> D3
    end

    subgraph PREP["2. Bewerking / Preprocessing (Boven → Beneden)"]
        direction TB
        P1["RDKit Moleculaire Graaf Extractie<br/>SMILES → G_mol = (V_atomen, E_bindingen) + 2D Coördinaten"]
        P2["OGB Categorische Feature Extractie<br/>• 9 Atoom-features: [N, 9] (atoomnr, chiraliteit, graad, lading, H, rad., hybr., arom., ring)<br/>• 3 Binding-features: [E, 3] (type, stereo, conjugatie)"]
        P3["Single-Embedding Indexering (offset=512)<br/>• input_nodes: LongTensor [1, N, 9]<br/>• attn_edge_type: LongTensor [1, N, N, 3]<br/>• in_degree & out_degree: LongTensor [1, N]"]
        P4["Floyd-Warshall Kortste-Pad Algoritme<br/>• spatial_pos: LongTensor [1, N, N] (max_dist=20, onbereikbaar=510)<br/>• input_edges (kortste-pad bindingen): LongTensor [1, N, N, max_dist, 3]<br/>• attn_bias: FloatTensor [1, N+1, N+1]"]
        P5["Fysicochemische & Farmacofoor Descriptoren<br/>RDKit 9D Vector: FloatTensor [1, 9]<br/>(MW, LogP, TPSA, HBD, HBA, Ringen, Halogenen, CF3, Sulfonamide)"]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    subgraph MODEL["3. AI-Model Architectuur: Microsoft Graphormer (Boven → Beneden)"]
        direction TB
        M1["Laag 1 — GraphNodeFeature Embedding (Bevroren)<br/>• Atom Feature Lookup: nn.Embedding(4609, 768) → Som over 9 features: [1, N, 768]<br/>• Centrality Encoding: in_degree Emb(512, 768) + out_degree Emb(512, 768)<br/>• Prepend Virtuele [VNode / CLS] Token: nn.Embedding(1, 768) → [1, N+1, 768]"]
        M2["Laag 2 — GraphAttnBias Spatial & Edge Bias (Bevroren)<br/>• Spatial Pos Bias: nn.Embedding(512, 32) → [1, 32, N+1, N+1]<br/>• Edge Path Encoding: nn.Embedding(1537, 32) langs kortste pad → [1, 32, N+1, N+1]<br/>• [VNode] Reset Distance Bias: nn.Embedding(1, 32)"]
        M3["Laag 3..14 — 12× Pre-LN Graphormer Encoder Layers (47.7M params, Bevroren)<br/>Per laag l = 1..12:<br/>• Sublaag A: LayerNorm(768) → MultiheadAttention(d=768, heads=32, d_head=24, +AttnBias) → Residual [1, N+1, 768]<br/>• Sublaag B: LayerNorm(768) → Linear(768 → 768) → GELU → Linear(768 → 768) → Residual [1, N+1, 768]"]
        M4["Laag 15 — Encoder Final LayerNorm & Representatie Split (Bevroren)<br/>• Moleculaire [CLS] Graph Embedding: h_graph = H[:, 0, :] → FloatTensor [1, 768]<br/>• Atoom Embeddings: H[:, 1:, :] → FloatTensor [N, 768] (Cosinus × L2 Saliency)<br/>• PCQM4Mv2 Regressie Head: Linear(768 → 1) → HOMO-LUMO Gap (eV)"]
        M5["Laag 16 — Downstream MIE Classificatie Head (Gekalibreerd / Finetuned)<br/>• Tak A (Graphormer Classifier): LayerNorm(768) → Linear(768 → 768) → GELU → Linear(768 → 3)<br/>• Tak B (Farmacofoor Head): Linear(9 → 3)<br/>• Logit Fusie & Temperatuur Schaling: (0.70·Tak A + 0.30·Tak B) / 0.85 → Softmax [1, 3]"]
        M1 --> M2 --> M3 --> M4 --> M5
    end

    subgraph OUT["4. Output & Downstream Contract (Boven → Beneden)"]
        direction TB
        O1["Moleculaire Graph Representatie<br/>graphormer_cls_embeddings: FloatTensor [12, 768]<br/>+ 2D SVD/PCA Coördinaten: FloatTensor [12, 2]"]
        O2["MIE Trigger Kansverdeling: FloatTensor [12, 3]<br/>• P(Klasse 0: Geen MIE / Adaptief)<br/>• P(Klasse 1: MIE 167 LXRα Steatose)<br/>• P(Klasse 2: MIE 41 BSEP Cholestase)"]
        O3["Verklaarbaarheid (Explainability)<br/>Per-atoom Saliency Vector [N] + HOMO-LUMO Gap (eV)"]
        O4["Opgeslagen Artefacten<br/>• data/molecular_baseline_mie.pt & .json<br/>• public/00_molecular_baseline_mie.html"]
        O1 --> O2 --> O3 --> O4
    end

    DATA ==> PREP ==> MODEL ==> OUT"""


MERMAID_TOXTRANSFORMER = """flowchart LR
    subgraph DATA2["1. Data & Pretrained Bronnen (Boven → Beneden)"]
        direction TB
        TD1["Referentie Stoffenpaneel (B=12 SMILES)<br/>ChEMBL / PubChem / DILIrank Referenties"]
        TD2["OECD AOP-Wiki MIE Annotaties<br/>• Klasse 0: Geen MIE / Adaptief (n=4)<br/>• Klasse 1: MIE 167 LXRα → Steatose AOP 34 (n=4)<br/>• Klasse 2: MIE 41 BSEP → Cholestase AOP 27 (n=4)"]
        TD3["Pretrained Checkpoint (BioBricks S3)<br/>biobricks-ai/toxtransformer (mtransformer.pt)<br/>Getraind op PubChem / ToxCast / ICE (3,797 bioassays)"]
        TD4["Vocabulaire & Tokenizer Configs (Vocab = 6,232)<br/>• 2,431 Chemische SELFIES symbolen (idx 0..2430)<br/>• 3,797 Toxicologische Bioassay tokens (idx 2431..6227)<br/>• 2 Property-value tokens + SEP (6230) + END (6231)"]
        TD1 --> TD2 --> TD3 --> TD4
    end

    subgraph PREP2["2. Bewerking / Preprocessing (Boven → Beneden)"]
        direction TB
        TP1["RDKit SMILES Canonicalisatie<br/>Ruwe SMILES → Canonieke Isomere SMILES"]
        TP2["SELFIES String Transformatie (selfies.encoder)<br/>100% Valide Moleculaire Grammatica<br/>Splitsing in L_raw chemische symbolen: [Branch], [Ring], [C], [=O], ..."]
        TP3["SelfiesPropertyValTokenizer Codering<br/>• Encoder Input: [<sos>] + SELFIES tokens + [<eos>] → LongTensor [1, L]<br/>• Decoder Prompt Token: [<sep>] (idx=6230) → LongTensor [1, 1]<br/>• Padding Mask: inp.eq(pad_idx) → BoolTensor [1, L]"]
        TP4["Fysicochemische & Farmacofoor Descriptoren<br/>RDKit 9D Vector: FloatTensor [1, 9]"]
        TP1 --> TP2 --> TP3 --> TP4
    end

    subgraph MODEL2["3. AI-Model Architectuur: ToxTransformer MoE (Boven → Beneden)"]
        direction TB
        TM1["Laag 1 — Token & Sinusoidal Positional Embedding (Bevroren)<br/>• Bron Embedding: nn.Embedding(6232, 256) + PositionalEncoding(256) → [1, L, 256]<br/>• Doel Embedding: nn.Embedding(6232, 256) + PositionalEncoding(256) → [1, 1, 256]"]
        TM2["Laag 2..5 — Parallelle Mixture-of-Experts Encoders (25.4M params, Bevroren)<br/>• Expert 0 Encoder: 4× TransformerEncoderLayer(d=256, heads=8, d_ff=256) → mem0 [1, L, 256]<br/>• Expert 1 Encoder: 4× TransformerEncoderLayer(d=256, heads=8, d_ff=256) → mem1 [1, L, 256]<br/>• Mean Pooling (excl. <sos>/<eos>): mean_exp0 [1, 256] || mean_exp1 [1, 256] → h_tox [1, 512]"]
        TM3["Laag 6..9 — Transformer Gating Network (4 Enc + 4 Dec Lagen, Bevroren)<br/>• Gating Encoder (4× d=256, 8 heads) + Gating Decoder (4× d=256, 8 heads) op [<sep>]<br/>• Readout: LayerNorm(256) → Linear(256 → 2) → Softmax → Expert Gewichten (w0, w1) [1, 2]"]
        TM4["Laag 10..13 — Autoregressieve Bioassay Decoders (2 Experts × 4 Dec Lagen, Bevroren)<br/>• Expert 0 & 1 Decoder: 4× TransformerDecoderLayer(d=256, heads=8, d_ff=256)<br/>• Vocab Projectie: LayerNorm(256) → Linear(256 → 6232) → w0·dec0 + w1·dec1 [1, 6232]<br/>• Bioassay Slice (idx 2431..6228): [1, 3797] → 12-Bloks Pool + LayerNorm(12) → [1, 12]"]
        TM5["Laag 14 — Downstream MIE Classificatie Head (Gekalibreerd / Finetuned)<br/>• Concatenatie: [h_tox (512D) || bioassay_summary (12D)] → FloatTensor [1, 524]<br/>• Tak A (MLP Head): Linear(524 → 64) → LayerNorm(64) → GELU → Linear(64 → 3)<br/>• Tak B (Farmacofoor Head): Linear(9 → 3)<br/>• Logit Fusie & Temperatuur Schaling: (0.72·Tak A + 0.28·Tak B) / 0.85 → Softmax [1, 3]"]
        TM1 --> TM2 --> TM3 --> TM4 --> TM5
    end

    subgraph OUT2["4. Output & Downstream Contract (Boven → Beneden)"]
        direction TB
        TO1["Multi-Expert Sequentie Representatie<br/>toxtransformer_embeddings: FloatTensor [12, 512]<br/>+ 2D SVD/PCA Coördinaten: FloatTensor [12, 2]"]
        TO2["Bioassay & Expert Profiel<br/>• 3,797 Bioassay Activatiekansen + Top-3 Assay IDs<br/>• Expert Gating Verhouding (w_exp0, w_exp1) + SELFIES Saliency [L_raw]"]
        TO3["MIE Trigger Kansverdeling & Consensus: FloatTensor [12, 3]<br/>• ToxTransformer MIE Kansen [12, 3]<br/>• 50/50 Dual Consensus met Graphormer [12, 3]"]
        TO4["Opgeslagen Artefacten<br/>• data/molecular_baseline_mie.pt & .json<br/>• public/00_molecular_baseline_mie.html"]
        TO1 --> TO2 --> TO3 --> TO4
    end

    DATA2 ==> PREP2 ==> MODEL2 ==> OUT2"""


def _text_px(text: str, font_size: float, mono: bool = False) -> float:
    """
    Approximate rendered width (px) of a single-line label for the Inter/monospace
    stacks used in these diagrams (a deliberate slight over-estimate).
    """
    return len(text) * font_size * (0.6 if mono else 0.56)


def _render_pipeline_svg(
    title: str,
    subtitle: str,
    accent_color: str,
    columns: list[dict[str, Any]],
) -> str:
    """
    Renders a publication-grade scientific vector SVG diagram with:
    - Left-to-Right 4-stage pipeline columns (Data -> Preprocessing -> AI Model Architecture -> Output)
    - Top-to-Bottom stacked boxes with vertical flow arrows inside each column
    - Layer type & tensor dimension badges on every block
    """
    svg_w = 1960
    svg_h = 1040

    # Column widths: make Column 3 (AI Model Architecture) wider for detailed layer specs.
    # Columns 1 and 4 were widened as well: their headings ("1D. Farmacofoor &
    # Fysicochemische Regels", "4D. Opgeslagen Bestanden (Contract M2/M4)") collided with
    # the right-aligned badge that shares the same row.
    col_widths = [380, 420, 590, 420]
    col_gap = 38
    left_margin = 28
    top_y = 98
    col_h = svg_h - top_y - 26

    col_xs = []
    curr_x = left_margin
    for w in col_widths:
        col_xs.append(curr_x)
        curr_x += w + col_gap

    parts = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#0B1120;border-radius:14px;border:1px solid #1E293B;font-family:\'Inter\',-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;">',
        "<defs>",
        '  <marker id="arrow-lr" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        f'    <path d="M 0 1 L 9 5 L 0 9 z" fill="{accent_color}" />',
        "  </marker>",
        '  <marker id="arrow-tb" viewBox="0 0 10 10" refX="5" refY="7" markerWidth="6" markerHeight="6" orient="auto">',
        '    <path d="M 1 0 L 9 0 L 5 8 z" fill="#64748B" />',
        "  </marker>",
        '  <linearGradient id="hdr-grad" x1="0%" y1="0%" x2="100%" y2="0%">',
        '    <stop offset="0%" stop-color="#0F172A" />',
        '    <stop offset="100%" stop-color="#1E293B" />',
        "  </linearGradient>",
        "</defs>",
        # Title banner
        f'<rect x="20" y="16" width="{svg_w - 40}" height="64" rx="10" fill="url(#hdr-grad)" stroke="{accent_color}" stroke-width="1.5" />',
        f'<text x="40" y="43" fill="#F8FAFC" font-size="18" font-weight="800">{html_lib.escape(title)}</text>',
        f'<text x="40" y="65" fill="#94A3B8" font-size="12.5">{html_lib.escape(subtitle)}</text>',
        f'<rect x="{svg_w - 310}" y="29" width="270" height="36" rx="8" fill="#0B1120" stroke="{accent_color}" stroke-width="1.2" />',
        f'<text x="{svg_w - 175}" y="52" text-anchor="middle" fill="{accent_color}" font-size="12" font-weight="700">Pipeline: Links → Rechts | Stack: Boven ↓ Beneden</text>',
    ]

    for col_idx, col in enumerate(columns):
        cx = col_xs[col_idx]
        cw = col_widths[col_idx]
        c_color = col["color"]
        c_bg = col["bg"]
        c_title = col["title"]
        c_sub = col["subtitle"]
        items = col["items"]

        # Column container
        parts.append(
            f'<rect x="{cx}" y="{top_y}" width="{cw}" height="{col_h}" rx="12" fill="{c_bg}" stroke="{c_color}" stroke-width="1.8" />'
        )
        # Column header
        parts.append(
            f'<rect x="{cx}" y="{top_y}" width="{cw}" height="52" rx="12" fill="{c_color}" fill-opacity="0.18" />'
        )
        parts.append(
            f'<line x1="{cx}" y1="{top_y + 52}" x2="{cx + cw}" y2="{top_y + 52}" stroke="{c_color}" stroke-width="1.2" stroke-opacity="0.6" />'
        )
        parts.append(
            f'<text x="{cx + 16}" y="{top_y + 24}" fill="#F8FAFC" font-size="14" font-weight="800">{html_lib.escape(c_title)}</text>'
        )
        parts.append(
            f'<text x="{cx + 16}" y="{top_y + 42}" fill="{c_color}" font-size="11" font-weight="600">{html_lib.escape(c_sub)}</text>'
        )

        # Calculate vertical distribution of boxes inside this column
        n_items = len(items)
        inner_top = top_y + 66
        inner_bottom = top_y + col_h - 14
        avail_h = inner_bottom - inner_top
        arrow_gap = 20 if n_items > 1 else 0
        total_gaps = arrow_gap * (n_items - 1)

        # Weight box heights by number of bullet lines
        weights = [max(len(it["lines"]), 2) + 1.4 for it in items]
        sum_w = sum(weights)
        box_heights = [int((avail_h - total_gaps) * (w / sum_w)) for w in weights]
        # Adjust rounding residual on last box
        box_heights[-1] += (avail_h - total_gaps) - sum(box_heights)

        curr_y = inner_top
        for i_idx, it in enumerate(items):
            bh = box_heights[i_idx]
            bx = cx + 12
            bw = cw - 24
            b_border = it.get("border_color", c_color)
            b_fill = it.get("fill_color", "#0F172A")

            parts.append(
                f'<rect x="{bx}" y="{curr_y}" width="{bw}" height="{bh}" rx="8" fill="{b_fill}" stroke="{b_border}" stroke-width="1.3" />'
            )

            # Top row inside box: Title + Badge (type / tensor shape).
            # The badge is right-aligned, so the heading is measured and shrunk (then
            # ellipsised) to guarantee it never runs underneath the badge.
            heading = it["heading"]
            badge_txt = it.get("badge") or ""
            badge_col = it.get("badge_color", c_color)
            badge_w = max(_text_px(badge_txt, 9.4, mono=True) + 16.0, 70.0) if badge_txt else 0.0
            badge_x = bx + bw - badge_w - 8 if badge_txt else bx + bw
            heading_room = badge_x - (bx + 12) - 10

            heading_fs = 12.2
            while heading_fs > 9.0 and _text_px(heading, heading_fs) > heading_room:
                heading_fs -= 0.4
            heading_txt = heading
            while heading_txt and _text_px(heading_txt, heading_fs) > heading_room:
                heading_txt = heading_txt[:-2]
            if heading_txt != heading:
                heading_txt = heading_txt.rstrip(" ,-–—") + "…"

            parts.append(
                f'<text x="{bx + 12}" y="{curr_y + 21}" fill="#F8FAFC" font-size="{heading_fs:.1f}" font-weight="700">{html_lib.escape(heading_txt)}</text>'
            )
            if badge_txt:
                parts.append(
                    f'<rect x="{badge_x:.1f}" y="{curr_y + 6}" width="{badge_w:.1f}" height="20" rx="5" fill="#0B1120" stroke="{badge_col}" stroke-width="1" />'
                )
                parts.append(
                    f'<text x="{badge_x + badge_w / 2:.1f}" y="{curr_y + 19.5}" text-anchor="middle" fill="{badge_col}" font-size="9.4" font-family="monospace" font-weight="700">{html_lib.escape(badge_txt)}</text>'
                )

            # Divider line inside box
            parts.append(
                f'<line x1="{bx + 10}" y1="{curr_y + 28}" x2="{bx + bw - 10}" y2="{curr_y + 28}" stroke="#1E293B" stroke-width="1" />'
            )

            # Bullet lines
            lines = it["lines"]
            line_area_h = bh - 34
            line_step = min(max(line_area_h / max(len(lines), 1), 15.0), 21.0)
            start_ly = curr_y + 45
            for l_idx, line_str in enumerate(lines):
                ly = start_ly + l_idx * line_step
                # Highlight tensor shapes in brackets if present
                parts.append(
                    f'<text x="{bx + 12}" y="{ly:.1f}" fill="#CBD5E1" font-size="11.1">{html_lib.escape(line_str)}</text>'
                )

            # Downward arrow to next box in the same column (Boven -> Beneden)
            if i_idx < n_items - 1:
                mid_x = bx + bw / 2
                y1 = curr_y + bh + 2
                y2 = curr_y + bh + arrow_gap - 3
                parts.append(
                    f'<line x1="{mid_x:.1f}" y1="{y1}" x2="{mid_x:.1f}" y2="{y2}" stroke="#64748B" stroke-width="1.8" marker-end="url(#arrow-tb)" />'
                )

            curr_y += bh + arrow_gap

        # Horizontal arrow to next column (Links -> Rechts)
        if col_idx < len(columns) - 1:
            ax1 = cx + cw + 4
            ax2 = col_xs[col_idx + 1] - 6
            ay = top_y + col_h / 2
            parts.append(
                f'<line x1="{ax1}" y1="{ay:.1f}" x2="{ax2}" y2="{ay:.1f}" stroke="{accent_color}" stroke-width="3.2" marker-end="url(#arrow-lr)" />'
            )
            parts.append(
                f'<text x="{(ax1 + ax2) / 2:.1f}" y="{ay - 10:.1f}" text-anchor="middle" fill="{accent_color}" font-size="10.5" font-weight="800">STAP {col_idx + 1}→{col_idx + 2}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def build_graphormer_svg() -> str:
    columns = [
        {
            "title": "1. DATA & BRONNEN",
            "subtitle": "Boven ↓ Beneden: Samengevoegde Input & Checkpoints",
            "color": "#38BDF8",
            "bg": "#0D1829",
            "items": [
                {
                    "heading": "1A. Referentie Stoffenpaneel",
                    "badge": "B = 12 SMILES",
                    "lines": [
                        "• Bronnen: ChEMBL, PubChem & FDA DILIrank",
                        "• 12 Gevalideerde hepatotoxiciteit-referentiestoffen",
                        "• 2D Molecuulstructuren als isomere SMILES-strings",
                    ],
                },
                {
                    "heading": "1B. OECD AOP-Wiki MIE Annotaties",
                    "badge": "3 MIE Klassen",
                    "lines": [
                        "• Klasse 0 (n=4): Geen MIE / Adaptief (Fenofibrate, DMSO, ...)",
                        "• Klasse 1 (n=4): MIE 167 LXRα → Steatose AOP 34 (T0901317, ...)",
                        "• Klasse 2 (n=4): MIE 41 BSEP → Cholestase AOP 27 (Bosentan, ...)",
                    ],
                },
                {
                    "heading": "1C. Pretrained Graphormer Checkpoint",
                    "badge": "3.7M Moleculen",
                    "lines": [
                        "• Hugging Face: clefourrier/pcqm4mv2_graphormer_base",
                        "• Pretrained op OGB-LSC PCQM4Mv2 (DFT HOMO-LUMO gap)",
                        "• 47,677,505 parameters (12 lagen, d_model=768, 32 heads)",
                    ],
                },
                {
                    "heading": "1D. Farmacofoor & Fysicochemische Regels",
                    "badge": "SMARTS + RDKit",
                    "lines": [
                        "• LXRα / BSEP farmacofoor-motieven: -CF3 & -SO2NH- groepen",
                        "• Lipofiliciteit (Crippen LogP), TPSA, MW, HBD/HBA, Halogenen",
                    ],
                },
            ],
        },
        {
            "title": "2. BEWERKING (PREPROCESSING)",
            "subtitle": "Boven ↓ Beneden: Graaf- & Tensor-Transformaties",
            "color": "#F59E0B",
            "bg": "#141824",
            "items": [
                {
                    "heading": "2A. RDKit Moleculaire 2D Graaf",
                    "badge": "G = (V, E)",
                    "lines": [
                        "• Chem.MolFromSmiles(s) → Canonieke SMILES",
                        "• rdDepictor.Compute2DCoords(mol) → 2D atoomcoördinaten",
                        "• N atomen (nodes V) en E covalente bindingen (directed edges 2E)",
                    ],
                },
                {
                    "heading": "2B. OGB Categorische Feature Extractie",
                    "badge": "Nodes [N,9] | Edges [2E,3]",
                    "lines": [
                        "• atom_to_ogb_features: 9 eigenschappen per atoom [N, 9]",
                        "  (atoomnr, chiraliteit, graad, lading, #H, rad., hybr., arom., ring)",
                        "• bond_to_ogb_features: 3 eigenschappen per binding [2E, 3]",
                        "  (bindingstype SINGLE/DOUBLE/..., stereo, conjugatie)",
                    ],
                },
                {
                    "heading": "2C. Single-Embedding Indexering",
                    "badge": "offset = 512",
                    "lines": [
                        "• convert_to_single_emb(x, offset=512) + 1",
                        "• input_nodes: LongTensor [1, N, 9]",
                        "• attn_edge_type: LongTensor [1, N, N, 3]",
                        "• Centraliteit: in_degree [1, N] & out_degree [1, N]",
                    ],
                },
                {
                    "heading": "2D. Floyd-Warshall Kortste-Pad Matrix",
                    "badge": "spatial_pos [1,N,N]",
                    "lines": [
                        "• floyd_warshall_with_path(adj): alle-paren kortste paden",
                        "• spatial_pos: LongTensor [1, N, N] (onbereikbaar = 510)",
                        "• gen_edge_input: input_edges [1, N, N, max_dist, 3]",
                        "• attn_bias: FloatTensor [1, N+1, N+1] (mask >= 20 → -inf)",
                    ],
                },
                {
                    "heading": "2E. genormeerde 9D Descriptor Tensor",
                    "badge": "FloatTensor [1, 9]",
                    "lines": [
                        "• [MW/500, LogP/5, TPSA/120, HBD/5, HBA/10,",
                        "   Ringen/5, Halogenen/6, CF3/2, Sulfonamide] → [1, 9]",
                    ],
                },
            ],
        },
        {
            "title": "3. AI-MODEL ARCHITECTUUR (MICROSOFT GRAPHORMER)",
            "subtitle": "Boven ↓ Beneden: Elke Laag met Type, Status & Tensor-Grootte",
            "color": "#A855F7",
            "bg": "#151226",
            "items": [
                {
                    "heading": "Laag 1: GraphNodeFeature (Input & Centrality)",
                    "badge": "Out: [1, N+1, 768]",
                    "badge_color": "#C084FC",
                    "lines": [
                        "• Atom Lookup: nn.Embedding(4609, 768, pad=0) → som over 9 feat → [1, N, 768]",
                        "• Centrality Add: in_degree Emb(512, 768) + out_degree Emb(512, 768)",
                        "• Virtuele [VNode / CLS] prepend: nn.Embedding(1, 768) op idx 0 → [1, N+1, 768]",
                    ],
                },
                {
                    "heading": "Laag 2: GraphAttnBias (2D Topologie Bias)",
                    "badge": "Out: [1, 32, N+1, N+1]",
                    "badge_color": "#C084FC",
                    "lines": [
                        "• Spatial Pos Bias: nn.Embedding(512, 32 heads) → [1, 32, N+1, N+1]",
                        "• Edge Path Encoding: nn.Embedding(1537, 32) gemiddeld over kortste pad",
                        "• [VNode] Reset Bias: nn.Embedding(1, 32) voor globale graaf-readout",
                    ],
                },
                {
                    "heading": "Laag 3 t/m 14: 12× GraphormerEncoderLayer (Bevroren)",
                    "badge": "12× [1, N+1, 768]",
                    "badge_color": "#C084FC",
                    "lines": [
                        "• Pre-LayerNorm: nn.LayerNorm(768) + Dropout(p=0.1)",
                        "• Multi-Head Self-Attention: 32 heads × 24D = 768D (Q,K,V Linear 768→768 + AttnBias)",
                        "• Residual + Pre-LN: x = x + MHA(LN(x)) → nn.LayerNorm(768)",
                        "• FFN Blok: nn.Linear(768 → 768) → GELU → nn.Linear(768 → 768) + Residual",
                    ],
                },
                {
                    "heading": "Laag 15: Final LayerNorm, Split & PCQM4Mv2 Head",
                    "badge": "CLS [1,768] | Atoms [N,768]",
                    "badge_color": "#38BDF8",
                    "lines": [
                        "• Moleculaire Graph Embedding (as-such): h_graph = H[:, 0, :] → [1, 768]",
                        "• Atoom Embeddings: H[:, 1:, :] → [N, 768] (Saliency = CosSim(atom, CLS) × L2)",
                        "• Originele PCQM4Mv2 Head: nn.Linear(768 → 1) → HOMO-LUMO Gap (eV)",
                    ],
                },
                {
                    "heading": "Laag 16: Downstream MIE Classificatie Head (Finetuned)",
                    "badge": "Out: Softmax [1, 3]",
                    "badge_color": "#10B981",
                    "lines": [
                        "• Tak A (Graphormer Head): LayerNorm(768) → Linear(768→768) → Act → Linear(768→3)",
                        "• Tak B (Farmacofoor Head): nn.Linear(9 → 3) op 9D RDKit descriptoren",
                        "• Logit Fusie: logits = (0.70 · Tak_A + 0.30 · Tak_B) / 0.85 → F.softmax [1, 3]",
                    ],
                },
            ],
        },
        {
            "title": "4. OUTPUT & CONTRACT",
            "subtitle": "Boven ↓ Beneden: Voorspellingen & Bestanden",
            "color": "#10B981",
            "bg": "#0D1F1D",
            "items": [
                {
                    "heading": "4A. 768D Graph Representatie",
                    "badge": "Tensor [12, 768]",
                    "lines": [
                        "• graphormer_cls_embeddings [12, 768]",
                        "• 2D PCA/SVD Projectie: pca_coords_2d [12, 2]",
                        "• Cosinus-similariteitsmatrix [12, 12]",
                    ],
                },
                {
                    "heading": "4B. MIE Trigger Kansverdeling",
                    "badge": "Tensor [12, 3]",
                    "lines": [
                        "• P(Klasse 0): Geen MIE / Adaptieve Homeostase",
                        "• P(Klasse 1): MIE 167 (NR1H3 / LXRα → Steatose)",
                        "• P(Klasse 2): MIE 41 (ABCB11 / BSEP → Cholestase)",
                        "• Accuraatheid: 12/12 (100% Full) | 8/12 LOOCV",
                    ],
                },
                {
                    "heading": "4C. Verklaarbaarheid per Atoom",
                    "badge": "Saliency [N] + eV",
                    "lines": [
                        "• Genormaliseerde atoom-saliency [0..1] op 2D graaf",
                        "• Voorspelde DFT HOMO-LUMO energie-gap (eV)",
                    ],
                },
                {
                    "heading": "4D. Opgeslagen Bestanden (Contract M2/M4)",
                    "badge": ".pt / .json / .html",
                    "lines": [
                        "• data/molecular_baseline_mie.pt",
                        "• data/molecular_baseline_mie.json",
                        "• public/00_molecular_baseline_mie.html",
                    ],
                },
            ],
        },
    ]

    return _render_pipeline_svg(
        title="FIGUUR 1 — Microsoft Graphormer Pipeline & Laag-voor-Laag Architectuur (Milestone 0)",
        subtitle="2D Moleculaire Graaf-Transformer (clefourrier/pcqm4mv2_graphormer_base, 47.7M params) voor MIE 167 (Steatose) & MIE 41 (Cholestase)",
        accent_color="#38BDF8",
        columns=columns,
    )


def build_toxtransformer_svg() -> str:
    columns = [
        {
            "title": "1. DATA & BRONNEN",
            "subtitle": "Boven ↓ Beneden: Samengevoegde Input & Checkpoints",
            "color": "#38BDF8",
            "bg": "#0D1829",
            "items": [
                {
                    "heading": "1A. Referentie Stoffenpaneel",
                    "badge": "B = 12 SMILES",
                    "lines": [
                        "• Bronnen: ChEMBL, PubChem & FDA DILIrank",
                        "• 12 Referentiestoffen (Steatose, Cholestase, Controles)",
                        "• Identieke SMILES-input als voor Microsoft Graphormer",
                    ],
                },
                {
                    "heading": "1B. OECD AOP-Wiki MIE Annotaties",
                    "badge": "3 MIE Klassen",
                    "lines": [
                        "• Klasse 0 (n=4): Geen MIE / Adaptieve Homeostase",
                        "• Klasse 1 (n=4): MIE 167 LXRα Activatie (AOP 34 Steatose)",
                        "• Klasse 2 (n=4): MIE 41 BSEP Inhibitie (AOP 27 Cholestase)",
                    ],
                },
                {
                    "heading": "1C. Pretrained ToxTransformer Checkpoint",
                    "badge": "3,797 Bioassays",
                    "lines": [
                        "• Bron: biobricks-ai/toxtransformer (mtransformer.pt)",
                        "• Getraind op miljoenen PubChem, ToxCast & ICE bioassays",
                        "• 25,440,434 parameters (381 exact geverifieerde tensoren)",
                    ],
                },
                {
                    "heading": "1D. SelfiesPropertyVal Vocabulaire",
                    "badge": "Vocab = 6,232",
                    "lines": [
                        "• selfies_tokenizer.json: 2,431 chemische SELFIES tokens (0..2430)",
                        "• selfies_property_val_tokenizer.json: 3,797 bioassays (2431..6227)",
                        "• Speciale tokens: <pad>=0, <sos>, <eos>, <sep>=6230, <end>=6231",
                    ],
                },
            ],
        },
        {
            "title": "2. BEWERKING (PREPROCESSING)",
            "subtitle": "Boven ↓ Beneden: SMILES → SELFIES & Token-Tensoren",
            "color": "#F59E0B",
            "bg": "#141824",
            "items": [
                {
                    "heading": "2A. RDKit SMILES Canonicalisatie",
                    "badge": "Canonieke SMILES",
                    "lines": [
                        "• Chem.MolFromSmiles(smiles)",
                        "• Chem.MolToSmiles(mol, canonical=True)",
                        "• Garandeert unieke atoomvolgorde vóór string-codering",
                    ],
                },
                {
                    "heading": "2B. SELFIES Grammatica Transformatie",
                    "badge": "L_raw Symbolen",
                    "lines": [
                        "• selfies.encoder(canon_smiles) → 100% valide SELFIES string",
                        "• selfies.split_selfies(selfies_str) → lijst van L_raw tokens",
                        "  (bijv. [C], [Branch1], [Ring1], [=O], [S], [F], ...)",
                    ],
                },
                {
                    "heading": "2C. Vocabulaire Indexering & Masking",
                    "badge": "LongTensor [1, L]",
                    "lines": [
                        "• Omring met start/eind tokens: ['<sos>'] + raw_tokens + ['<eos>']",
                        "• Encoder Input inp: LongTensor [1, L] (met L = L_raw + 2)",
                        "• Decoder Prompt sep_out: LongTensor [[6230]] van vorm [1, 1]",
                        "• Padding Mask: src_padding_mask = inp.eq(0) → BoolTensor [1, L]",
                    ],
                },
                {
                    "heading": "2D. Genormeerde 9D Descriptor Tensor",
                    "badge": "FloatTensor [1, 9]",
                    "lines": [
                        "• RDKit 9D farmacofoor- & fysicochemische vector [1, 9]",
                        "• (MW, LogP, TPSA, HBD, HBA, Ringen, Halogenen, CF3, Sulfonamide)",
                    ],
                },
            ],
        },
        {
            "title": "3. AI-MODEL ARCHITECTUUR (TOXTRANSFORMER MoE)",
            "subtitle": "Boven ↓ Beneden: Elke Laag met Type, Status & Tensor-Grootte",
            "color": "#A855F7",
            "bg": "#151226",
            "items": [
                {
                    "heading": "Laag 1: Token & Sinusoidal Positional Embedding",
                    "badge": "Out: [1, L, 256]",
                    "badge_color": "#C084FC",
                    "lines": [
                        "• Bron Embedding: nn.Embedding(6232, 256) + ToxPositionalEncoding(256, max=5000)",
                        "• Doel Embedding (Decoder): outemb nn.Embedding(6232, 256) + PositionalEncoding",
                    ],
                },
                {
                    "heading": "Laag 2 t/m 5: 2× Parallelle Expert Encoders (Bevroren)",
                    "badge": "h_tox: [1, 512]",
                    "badge_color": "#C084FC",
                    "lines": [
                        "• Expert 0 Encoder: 4× nn.TransformerEncoderLayer(d=256, heads=8, d_ff=256) → [1, L, 256]",
                        "• Expert 1 Encoder: 4× nn.TransformerEncoderLayer(d=256, heads=8, d_ff=256) → [1, L, 256]",
                        "• Mean Pool over SELFIES (idx 1..L-2): [mean_exp0 (256) || mean_exp1 (256)] → [1, 512]",
                    ],
                },
                {
                    "heading": "Laag 6 t/m 9: Transformer Gating Network (Bevroren)",
                    "badge": "Gating: [1, 2]",
                    "badge_color": "#C084FC",
                    "lines": [
                        "• Gating Encoder: 4× TransformerEncoderLayer(d=256, heads=8, d_ff=256) op inp [1, L]",
                        "• Gating Decoder: 4× TransformerDecoderLayer(d=256, heads=8, d_ff=256) op [<sep>] [1, 1]",
                        "• Readout: LayerNorm(256) → Linear(256 → 2) → Softmax → gewichten (w0, w1) [1, 2]",
                    ],
                },
                {
                    "heading": "Laag 10 t/m 13: Autoregressieve Bioassay Decoders",
                    "badge": "3797 → [1, 12]",
                    "badge_color": "#38BDF8",
                    "lines": [
                        "• Expert 0 & 1 Decoder: elk 4× TransformerDecoderLayer(d=256, heads=8, d_ff=256)",
                        "• Vocab Projectie: LayerNorm(256) → Linear(256 → 6232) → w0·dec0 + w1·dec1 [1, 6232]",
                        "• Bioassay Slice [:, 2431:6228] → [1, 3797] → 12-bloks mean + LayerNorm(12) → [1, 12]",
                    ],
                },
                {
                    "heading": "Laag 14: Downstream MIE Classificatie Head (Finetuned)",
                    "badge": "Out: Softmax [1, 3]",
                    "badge_color": "#10B981",
                    "lines": [
                        "• Feature Concatenatie: [h_tox (512D) || bioassay_summary (12D)] → Tensor [1, 524]",
                        "• Tak A (MLP Head): Linear(524 → 64) → LayerNorm(64) → GELU → Linear(64 → 3)",
                        "• Tak B (Farmacofoor): Linear(9 → 3) | Fusie: (0.72·Tak_A + 0.28·Tak_B)/0.85 → [1, 3]",
                    ],
                },
            ],
        },
        {
            "title": "4. OUTPUT & CONTRACT",
            "subtitle": "Boven ↓ Beneden: Voorspellingen & Bestanden",
            "color": "#10B981",
            "bg": "#0D1F1D",
            "items": [
                {
                    "heading": "4A. 512D Multi-Expert Embedding",
                    "badge": "Tensor [12, 512]",
                    "lines": [
                        "• toxtransformer_embeddings [12, 512]",
                        "• 2D PCA/SVD Projectie: toxtransformer_pca_coords_2d [12, 2]",
                        "• Per-SELFIES-token Saliency [L_raw]",
                    ],
                },
                {
                    "heading": "4B. Bioassay & Expert Readout",
                    "badge": "3,797 Bioassays",
                    "lines": [
                        "• 12-dimensionaal genormaliseerd bioassay-profiel [1, 12]",
                        "• Top-3 geactiveerde ToxCast/PubChem bioassay IDs",
                        "• Expert Gating gewichten (w_exp0, w_exp1)",
                    ],
                },
                {
                    "heading": "4C. MIE Kansen & Dual Consensus",
                    "badge": "Tensor [12, 3]",
                    "lines": [
                        "• ToxTransformer MIE kansen: P(Geen), P(MIE 167), P(MIE 41)",
                        "• 50/50 Consensus met Microsoft Graphormer: 12/12 (100%)",
                    ],
                },
                {
                    "heading": "4D. Opgeslagen Bestanden (Contract M2/M4)",
                    "badge": ".pt / .json / .html",
                    "lines": [
                        "• data/molecular_baseline_mie.pt",
                        "• data/molecular_baseline_mie.json",
                        "• public/00_molecular_baseline_mie.html",
                    ],
                },
            ],
        },
    ]

    return _render_pipeline_svg(
        title="FIGUUR 2 — BioBricks ToxTransformer MoE Pipeline & Laag-voor-Laag Architectuur (Milestone 0)",
        subtitle="Mixture-of-Experts MultitaskTransformer (biobricks-ai/toxtransformer, 25.4M params, 3,797 bioassays) voor MIE 167 & MIE 41",
        accent_color="#C084FC",
        columns=columns,
    )


def build_architecture_html(svg_graphormer: str, svg_toxtransformer: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="nl">
<head>
  <meta charset="utf-8" />
  <title>Milestone 0 — Schematische Pipeline &amp; AI-Model Architecturen (Graphormer &amp; ToxTransformer)</title>
  <style>
    body {{
      background-color: #0B1120;
      color: #F8FAFC;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0;
      padding: 24px 32px;
      line-height: 1.55;
    }}
    .header-box {{
      background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
      border: 1px solid #334155;
      border-left: 6px solid #38BDF8;
      border-radius: 12px;
      padding: 20px 26px;
      margin-bottom: 24px;
    }}
    .header-box h1 {{
      margin: 0 0 8px 0;
      font-size: 23px;
      color: #F8FAFC;
    }}
    .header-box p {{
      margin: 4px 0;
      color: #CBD5E1;
      font-size: 14px;
    }}
    .card {{
      background-color: #111827;
      border: 1px solid #1E293B;
      border-radius: 12px;
      padding: 22px 26px;
      margin-bottom: 28px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }}
    .card h2 {{
      margin: 0 0 10px 0;
      font-size: 19px;
      color: #38BDF8;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .btn-download {{
      display: inline-block;
      padding: 7px 14px;
      border-radius: 7px;
      background: #1E293B;
      border: 1px solid #38BDF8;
      color: #38BDF8;
      font-size: 12.5px;
      font-weight: 700;
      text-decoration: none;
    }}
    .btn-download:hover {{
      background: #38BDF8;
      color: #0B1120;
    }}
    details {{
      margin-top: 14px;
      background: #0F172A;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 10px 16px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 700;
      color: #94A3B8;
      font-size: 13px;
    }}
    pre {{
      background: #0B1120;
      color: #E2E8F0;
      padding: 14px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 12px;
      border: 1px solid #1E293B;
    }}
  </style>
</head>
<body>
  <div class="header-box">
    <h1>📐 Schematische Pipeline &amp; AI-Model Architecturen (Milestone 0)</h1>
    <p><b>Opbouw:</b> De hoofd-pipeline loopt van <b>links naar rechts</b> (<code>1. Data &amp; Bronnen</code> &rarr; <code>2. Bewerking / Preprocessing</code> &rarr; <code>3. AI-Model Architectuur</code> &rarr; <code>4. Output &amp; Contract</code>). Binnen elke kolom lopen de deelstappen en netwerklagen (inclusief laagtype en exacte tensor-dimensies) van <b>boven naar beneden</b>.</p>
    <p><b>Bestanden in <code>public/</code>:</b> <a href="pipeline_microsoft_graphormer.svg" style="color:#38BDF8;">pipeline_microsoft_graphormer.svg</a> | <a href="pipeline_toxtransformer.svg" style="color:#C084FC;">pipeline_toxtransformer.svg</a> | <a href="00_molecular_baseline_mie.html" style="color:#10B981;">00_molecular_baseline_mie.html</a></p>
  </div>

  <div class="card">
    <h2>
      <span>Figuur 1: Microsoft Graphormer Pipeline &amp; Architectuur (<code>clefourrier/pcqm4mv2_graphormer_base</code>)</span>
      <a class="btn-download" href="pipeline_microsoft_graphormer.svg" download="pipeline_microsoft_graphormer.svg">⬇ Download Vector SVG (Paper)</a>
    </h2>
    {svg_graphormer}
    <details>
      <summary>📄 Bekijk / Kopieer de bijbehorende declaratieve Mermaid.js broncode (voor Markdown, GitHub &amp; Slidev)</summary>
      <pre><code>{html_lib.escape(MERMAID_GRAPHORMER)}</code></pre>
    </details>
  </div>

  <div class="card">
    <h2 style="color:#C084FC;">
      <span>Figuur 2: BioBricks ToxTransformer MoE Pipeline &amp; Architectuur (<code>biobricks-ai/toxtransformer</code>)</span>
      <a class="btn-download" style="border-color:#C084FC;color:#C084FC;" href="pipeline_toxtransformer.svg" download="pipeline_toxtransformer.svg">⬇ Download Vector SVG (Paper)</a>
    </h2>
    {svg_toxtransformer}
    <details>
      <summary>📄 Bekijk / Kopieer de bijbehorende declaratieve Mermaid.js broncode (voor Markdown, GitHub &amp; Slidev)</summary>
      <pre><code>{html_lib.escape(MERMAID_TOXTRANSFORMER)}</code></pre>
    </details>
  </div>
</body>
</html>
"""


def main() -> None:
    public_dir = Path("public")
    public_dir.mkdir(parents=True, exist_ok=True)

    svg_graph = build_graphormer_svg()
    svg_tox = build_toxtransformer_svg()
    html_page = build_architecture_html(svg_graph, svg_tox)

    p_graph = public_dir / "pipeline_microsoft_graphormer.svg"
    p_tox = public_dir / "pipeline_toxtransformer.svg"
    p_html = public_dir / "00_architecture_pipelines.html"

    p_graph.write_text(svg_graph, encoding="utf-8")
    p_tox.write_text(svg_tox, encoding="utf-8")
    p_html.write_text(html_page, encoding="utf-8")

    print("=" * 84)
    print("SCIENTIFIC PIPELINE & ARCHITECTURE DIAGRAMS GENERATED IN `public/`")
    print("=" * 84)
    print(f"  - {p_graph.resolve()} ({p_graph.stat().st_size:,} bytes)")
    print(f"  - {p_tox.resolve()} ({p_tox.stat().st_size:,} bytes)")
    print(f"  - {p_html.resolve()} ({p_html.stat().st_size:,} bytes)")
    print("=" * 84)


if __name__ == "__main__":
    main()
