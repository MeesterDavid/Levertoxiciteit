"""
Generates the central figure for Slide 2 of the 3-minute pitch:

- `public/s2_multimodal_ai_graph.svg`

The figure states the central dogma of the proposal (a hepatocyte either returns to the
homeostatic attractor z_homeo or tips over into z_adversity = z_steatosis / z_cholestasis) and
shows the solution: ONE multimodal & multiscale AI node that fuses five data modalities
(protein-protein interaction, gene expression, gene-gene interaction, chemical structure + dose
and - as future work - physiological measurements such as multi-electrode arrays) and produces
two different answers:

  1. predicted MIE class                      -> solid arrow  (built and validated, M0 / M1.1)
  2. P(z_adversity | context, exposure, dose)  -> dashed arrow (next step, M1.2 -> M4)

A small key in the bottom of the figure makes the colour code explicit instead of implicit: a
solid green arrow is built and validated today, a dashed purple arrow is future work. Every
arrow head is drawn in user space with its tip exactly on the border of the node it points at,
and the two answer arrows leave the model node orthogonally, so a line always runs straight into
the axis of its own head. All numbers are derived from the milestone artifacts themselves
(`data/molecular_baseline_mie.json` and `data/hepatocyte_baseline_data.json`), and the provenance
line is assembled from `metadata.sources` of the hepatocyte JSON (minus MyGene.info), so the
figure cannot silently drift away from the data. No matplotlib is used: the SVG is written
directly, in line with the other slide asset generators in this repository.

Run with:
    CUDA_VISIBLE_DEVICES=1 uv run python generate_slide2_diagram.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PUBLIC_DIR = ROOT / "public"
MOLECULAR_JSON = DATA_DIR / "molecular_baseline_mie.json"
HEPATOCYTE_JSON = DATA_DIR / "hepatocyte_baseline_data.json"
OUT_SVG = PUBLIC_DIR / "s2_multimodal_ai_graph.svg"

# ------------------------------------------------------------------ canvas ----
W, H = 1040, 460
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"

# ----------------------------------------------------------------- palette ----
CYAN = "#38bdf8"
EMERALD = "#10b981"
AMBER = "#f59e0b"
ROSE = "#f43f5e"
VIOLET = "#a855f7"
PINK = "#f472b6"
SLATE = "#64748b"
BORDER = "#1e293b"
CARD_BG = "#111c30"
PANEL_BG = "#0b1220"
TEXT = "#e2e8f0"
SOFT = "#cbd5e1"
MUTED = "#94a3b8"
DIM = "#7c8aa0"


# ----------------------------------------------------------------- helpers ----
def esc(value: str) -> str:
    """Minimal XML escaping for text nodes and attribute values."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(
    x: float,
    y: float,
    body: str,
    size: float = 10.0,
    fill: str = TEXT,
    weight: int | str = 400,
    anchor: str = "start",
    family: str = FONT,
    opacity: float | None = None,
    style: str | None = None,
) -> str:
    attrs = [
        f'x="{x:.1f}"',
        f'y="{y:.1f}"',
        f'font-family="{family}"',
        f'font-size="{size:.1f}"',
        f'fill="{fill}"',
        f'font-weight="{weight}"',
    ]
    if anchor != "start":
        attrs.append(f'text-anchor="{anchor}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    if style:
        attrs.append(f'font-style="{style}"')
    return f'<text {" ".join(attrs)}>{body}</text>'


def sub(main: str, subscript: str, size: float = 7.6, dy: float = 2.9) -> str:
    """Inline subscript that survives plain SVG rendering (no MathML/CSS needed).

    The subscript is italic, matching the deck: `homeo`, `adversity`, `steatosis` and
    `cholestasis` are indices on the state variable z, not part of its name.
    """
    return (
        f'{main}<tspan dy="{dy}" font-size="{size}" font-style="italic">{esc(subscript)}</tspan>'
        f'<tspan dy="{-dy}"></tspan>'
    )


def card(
    x: float,
    y: float,
    w: float,
    h: float,
    stroke: str,
    fill: str = CARD_BG,
    rx: float = 12.0,
    dash: str | None = None,
    width: float = 1.4,
    opacity: float | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    op_attr = f' opacity="{opacity}"' if opacity is not None else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash_attr}{op_attr}/>'
    )


def marker(marker_id: str, color: str, size: float = 10.4) -> str:
    """Arrow head drawn in user space, with its tip exactly on the end of the path.

    `markerUnits="userSpaceOnUse"` keeps every head the same size no matter the stroke width, and
    `refX=11.4` places the tip of the triangle on the path end point, so the line runs into the
    axis of its own head instead of stopping in the middle of it. The head is fill-only: a stroked
    triangle would overshoot the node border by half a stroke width.
    """
    return (
        f'<marker id="{marker_id}" viewBox="0 0 12 12" refX="11.4" refY="6" '
        f'markerUnits="userSpaceOnUse" markerWidth="{size:.1f}" markerHeight="{size:.1f}" '
        f'orient="auto-start-reverse">'
        f'<path d="M 0.8 1.7 L 11.4 6 L 0.8 10.3 Z" fill="{color}" stroke-linejoin="round"/>'
        f'</marker>'
    )


def curve(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    marker_id: str,
    dash: str | None = None,
    width: float = 1.8,
    opacity: float = 1.0,
) -> str:
    """Smooth left-to-right S-curve between two points (cubic Bezier)."""
    cx = (x2 - x1) * 0.55
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="M {x1:.1f} {y1:.1f} C {x1 + cx:.1f} {y1:.1f}, {x2 - cx:.1f} {y2:.1f}, '
        f'{x2:.1f} {y2:.1f}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round"{dash_attr} opacity="{opacity}" marker-end="url(#{marker_id})"/>'
    )


def elbow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    marker_id: str,
    dash: str | None = None,
    width: float = 2.0,
    radius: float = 16.0,
) -> str:
    """Orthogonal connector with rounded corners, used for the two answer arrows.

    A cubic S-curve arrives at an angle, so the head visually detaches from the line it belongs
    to. An elbow arrives horizontally through the middle of its own band, which is what the eye
    expects from a block diagram, and it keeps the vertical run free for a label.
    """
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    mid_x = (x1 + x2) / 2.0
    direction = 1.0 if y2 >= y1 else -1.0
    r = min(radius, abs(y2 - y1) / 2.0 - 0.5, abs(x2 - x1) / 2.0)
    d = (
        f"M {x1:.1f} {y1:.1f} "
        f"L {mid_x - r:.1f} {y1:.1f} "
        f"Q {mid_x:.1f} {y1:.1f} {mid_x:.1f} {y1 + direction * r:.1f} "
        f"L {mid_x:.1f} {y2 - direction * r:.1f} "
        f"Q {mid_x:.1f} {y2:.1f} {mid_x + r:.1f} {y2:.1f} "
        f"L {x2:.1f} {y2:.1f}"
    )
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash_attr} '
        f'marker-end="url(#{marker_id})"/>'
    )


# ------------------------------------------------------------------- data ----
LOOCV_PROB_KEYS = {0: "loocv_P_no_mie", 1: "loocv_P_mie167", 2: "loocv_P_mie41"}
LOOCV_MODEL_BLOCKS = {"graphormer": "loocv", "toxtransformer": "tox_loocv"}


def _loocv_confusion(compounds: list[dict], block_key: str) -> tuple[list[list[int]], list[float]]:
    """Fallback path: rebuild one LOOCV confusion matrix straight from the per-compound cache."""
    matrix = [[0, 0, 0] for _ in range(3)]
    true_probs: list[float] = []
    for compound in compounds:
        truth = int(compound["true_class"])
        block = compound[block_key]
        vector = [float(block[LOOCV_PROB_KEYS[class_idx]]) for class_idx in range(3)]
        pred = max(range(3), key=lambda class_idx: vector[class_idx])
        matrix[truth][pred] += 1
        true_probs.append(vector[truth])
    return matrix, true_probs


def load_molecular_metrics() -> dict:
    """
    Leave-one-out metrics of Milestone 0.

    In-sample accuracy is deliberately absent: the MIE heads of both models were calibrated on the
    same 12 reference compounds, so an in-sample diagonal is perfect by construction (and therefore
    uninformative). The figure reports the LOOCV accuracy, the uncertainty of the true class and the
    3x3 confusion matrix - all read from `validation` in the milestone JSON.
    """
    mol = json.loads(MOLECULAR_JSON.read_text(encoding="utf-8"))
    compounds = mol["compounds"]
    validation = mol.get("validation") or {}

    consensus = validation.get("consensus") or validation.get("graphormer")
    if consensus:
        matrix = [[int(value) for value in row] for row in consensus["confusion_matrix"]]
        n_correct = int(consensus["n_correct"])
        mean_p = float(consensus["mean_p_true_class"])
        min_p = float(consensus["min_p_true_class"])
        max_p = float(consensus["max_p_true_class"])
    else:  # older cache without the `validation` block
        matrix, true_probs = _loocv_confusion(compounds, LOOCV_MODEL_BLOCKS["graphormer"])
        n_correct = sum(matrix[i][i] for i in range(3))
        mean_p = sum(true_probs) / len(true_probs)
        min_p, max_p = min(true_probs), max(true_probs)

    consensus = validation.get("consensus") or {}
    n_params = int(mol["graphormer_num_parameters"]) + int(mol["toxtransformer_num_parameters"])
    return {
        "n_compounds": len(compounds),
        "n_classes": len({int(c["true_class"]) for c in compounds}),
        "loo_acc": n_correct,
        "loo_mean": mean_p,
        "loo_min": min_p,
        "loo_max": max_p,
        "loo_matrix": matrix,
        "cons_acc": int(consensus["n_correct"]) if consensus else n_correct,
        "cons_mean": float(consensus["mean_p_true_class"]) if consensus else mean_p,
        "graphormer_params": int(mol["graphormer_num_parameters"]),
        "toxtransformer_params": int(mol["toxtransformer_num_parameters"]),
        "total_params": n_params,
    }


def load_hepatocyte_metrics() -> dict:
    """Graph size, edge signs and STRING support of the AOP / physiological-map network."""
    hep = json.loads(HEPATOCYTE_JSON.read_text(encoding="utf-8"))
    meta = hep["metadata"]
    genes = hep["genes"]
    edges = hep["edges"]
    matrices = hep["matrices"]

    string_scores = [float(e["string_score"]) for e in edges if e.get("string_score") is not None]
    n_pos = sum(1 for e in edges if int(e["sign"]) > 0)
    n_neg = sum(1 for e in edges if int(e["sign"]) < 0)
    n_features = len(matrices["X_healthy_raw"][0])
    n_states = sum(1 for key in ("X_healthy_raw", "X_steatosis_raw", "X_cholestasis_raw") if key in matrices)

    return {
        "n_genes": len(genes),
        "n_features": n_features,
        "n_states": n_states,
        "n_edges": len(edges),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_sources": len(meta["sources"]),
        "sources": provenance_sources(meta["sources"]),
        "string_mean": sum(string_scores) / len(string_scores),
        "string_min": min(string_scores),
        "string_max": max(string_scores),
        "dist_matrix_shape": (len(matrices["dist_matrix_19_cls"]), len(matrices["dist_matrix_19_cls"][0])),
    }


def fmt_params(n_params: int) -> str:
    """47677505 -> '47.7M' (keeps the figure readable)."""
    return f"{n_params / 1e6:.1f}M"


# Audience-facing labels for the sources of `fetch_real_hepatocyte_data.py`, matched by keyword.
SOURCE_ALIASES: tuple[tuple[str, str], ...] = (
    ("AOP 34", "OECD AOP 34"),
    ("AOP 27", "OECD AOP 27"),
    ("STRING", "STRING-db v12"),
    ("GTEx", "GTEx v8 / HPA"),
    ("GEO", "GEO GSE126848"),
    ("HepaRG", "HepaRG (Rodrigues 2018)"),
)
# MyGene.info only supplies gene/protein structure annotations and is kept off the pitch slide.
PROVENANCE_SKIP: tuple[str, ...] = ("mygene",)


def provenance_sources(meta_sources: list[str]) -> list[str]:
    """Short labels for the sources that really feed the model, in the order of the JSON."""
    short: list[str] = []
    for raw in meta_sources:
        if any(skip in raw.lower() for skip in PROVENANCE_SKIP):
            continue
        label = next((alias for key, alias in SOURCE_ALIASES if key in raw), raw.split(" (")[0])
        if label not in short:
            short.append(label)
    return short



def hexagon(cx: float, cy: float, r: float, stroke: str, width: float = 1.5) -> str:
    """Six-membered ring used as the 'chemical structure' glyph."""
    points = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        points.append(f"{cx + r * math.cos(angle):.1f},{cy + r * math.sin(angle):.1f}")
    return (
        f'<polygon points="{" ".join(points)}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linejoin="round"/>'
    )


# ------------------------------------------------------------------ drawing ----
def draw_icon(kind: str, cx: float, cy: float, color: str) -> str:
    """Small schematic glyph (no external icon font needed)."""
    parts = [f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="13" fill="{color}" fill-opacity="0.12" '
             f'stroke="{color}" stroke-width="1.1" stroke-opacity="0.55"/>']
    if kind == "ppi":  # two interaction partners
        parts.append(f'<line x1="{cx - 5:.1f}" y1="{cy - 3:.1f}" x2="{cx + 5:.1f}" y2="{cy + 3:.1f}" '
                     f'stroke="{color}" stroke-width="1.3"/>')
        parts.append(f'<circle cx="{cx - 5.5:.1f}" cy="{cy - 4:.1f}" r="3.1" fill="none" '
                     f'stroke="{color}" stroke-width="1.3"/>')
        parts.append(f'<circle cx="{cx + 5.5:.1f}" cy="{cy + 4:.1f}" r="3.1" fill="none" '
                     f'stroke="{color}" stroke-width="1.3"/>')
    elif kind == "expr":  # expression bars
        for i, bar_h in enumerate((4.5, 8.5, 6.5)):
            bx = cx - 6 + i * 5
            parts.append(f'<rect x="{bx:.1f}" y="{cy + 5 - bar_h:.1f}" width="3.2" '
                         f'height="{bar_h:.1f}" rx="1" fill="{color}" fill-opacity="0.85"/>')
    elif kind == "gene":  # directed gene-gene edge
        parts.append(f'<circle cx="{cx - 5.5:.1f}" cy="{cy:.1f}" r="2.8" fill="{color}"/>')
        parts.append(f'<circle cx="{cx + 5.5:.1f}" cy="{cy:.1f}" r="2.8" fill="none" '
                     f'stroke="{color}" stroke-width="1.3"/>')
        parts.append(f'<line x1="{cx - 2.6:.1f}" y1="{cy:.1f}" x2="{cx + 3.4:.1f}" y2="{cy:.1f}" '
                     f'stroke="{color}" stroke-width="1.2"/>')
    elif kind == "chem":  # chemical structure
        parts.append(hexagon(cx, cy, 6.2, color, width=1.3))
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.5" fill="{color}"/>')
    elif kind == "signal":  # multi-electrode array: a probe with an action-potential trace
        parts.append(f'<line x1="{cx - 7:.1f}" y1="{cy + 6:.1f}" x2="{cx + 7:.1f}" y2="{cy + 6:.1f}" '
                     f'stroke="{color}" stroke-width="1.1" stroke-opacity="0.6"/>')
        for spike in (-4.0, 0.0, 4.0):
            parts.append(f'<line x1="{cx + spike:.1f}" y1="{cy + 6:.1f}" x2="{cx + spike:.1f}" '
                         f'y2="{cy + 3.2:.1f}" stroke="{color}" stroke-width="1.1"/>')
        trace = ((cx - 7.5, cy - 1.5), (cx - 3.5, cy - 1.5), (cx - 1.8, cy - 6.5),
                 (cx + 0.8, cy + 2.5), (cx + 2.8, cy - 3.5), (cx + 7.5, cy - 3.5))
        points = " ".join("%.1f,%.1f" % (px, py) for px, py in trace)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.3" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    return "".join(parts)


def draw_input_card(
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    source: str,
    metric: str,
    color: str,
    icon: str,
) -> str:
    rows = [card(x, y, w, h, stroke=BORDER, fill=CARD_BG, rx=11)]
    # left accent bar so the four modalities stay distinguishable at projector distance
    rows.append(
        f'<rect x="{x:.1f}" y="{y + 8:.1f}" width="3" height="{h - 16:.1f}" rx="1.5" '
        f'fill="{color}" fill-opacity="0.85"/>'
    )
    rows.append(draw_icon(icon, x + 30, y + h / 2 - 2, color))
    rows.append(text(x + 50, y + 25, esc(label), size=11.2, fill=TEXT, weight=700))
    rows.append(text(x + 50, y + 39.5, esc(source), size=8.6, fill=MUTED))
    rows.append(text(x + 50, y + 60, esc(metric), size=8.9, fill=color, weight=600))
    return "".join(rows)


def draw_hub(x: float, y: float, w: float, h: float, total_params: tuple[str, str]) -> str:
    """The single fusion node: one multimodal & multiscale model + added classification layer."""
    rows = [
        # soft halo to read as the model's "core"
        f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="{w + 10:.1f}" height="{h + 10:.1f}" rx="17" '
        f'fill="{VIOLET}" fill-opacity="0.07"/>',
        card(x, y, w, h, stroke=VIOLET, fill="#161033", rx=14, width=1.8),
        text(x + w / 2, y + 86, "Multi-modal &amp;", size=15.0, fill=VIOLET, weight=800, anchor="middle"),
        text(x + w / 2, y + 106, "Multi-scale AI", size=15.0, fill=VIOLET, weight=800, anchor="middle"),
        f'<line x1="{x + 24:.1f}" y1="{y + 66:.1f}" x2="{x + w - 24:.1f}" y2="{y + 66:.1f}" '
        f'stroke="{VIOLET}" stroke-opacity="0.35" stroke-width="1"/>',
        text(x + w / 2, y + 33, f"Graphormer {total_params[0]} + ToxFormer {total_params[1]}",
             size=10.2, fill=EMERALD, weight=600, anchor="middle"),
        text(x + w / 2, y + 53, "+ final classification layer (50/50 fusion)",
             size=9.5, fill=EMERALD, weight=700, anchor="middle"),
    ]
    return "".join(rows)



def tag(
    x: float,
    y: float,
    plain: str,
    color: str,
    body: str | None = None,
    size: float = 8.0,
    height: float = 14.5,
    pad_x: float = 5.5,
) -> tuple[str, float]:
    """Rounded tag whose visible body may contain <tspan> markup (subscripts)."""
    width = len(plain) * size * 0.58 + 2 * pad_x
    svg = (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{height / 2:.1f}" fill="{color}" fill-opacity="0.13" '
        f'stroke="{color}" stroke-width="0.9" stroke-opacity="0.7"/>'
        + text(x + width / 2, y + height / 2 + size * 0.36, body or esc(plain), size=size,
               fill=color, weight=700, anchor="middle")
    )
    return svg, width


def chip_on_arrow(
    x: float,
    y: float,
    lines: tuple[str, ...],
    color: str,
    size: float = 8.2,
) -> str:
    """Small multi-line chip centred on the vertical run of an elbow connector.

    Used to name the purple answer arrow (`(Neural)` / `ODEs`) inside the 72 px gutter between
    the model node and the answer nodes, where a horizontal label would collide with the cards.
    The border is solid and the fill opaque on purpose: a dashed border would let the dashes of
    the arrow underneath show through its gaps.
    """
    width = max(len(line) for line in lines) * size * 0.54 + 13.0
    height = 11.6 * len(lines) + 7.0
    rows = [
        f'<rect x="{x - width / 2:.1f}" y="{y - height / 2:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" rx="6" fill="{PANEL_BG}" stroke="{color}" stroke-width="1.1" '
        f'stroke-opacity="0.9"/>'
    ]
    for i, line in enumerate(lines):
        rows.append(text(x, y - height / 2 + 12.3 + i * 11.6, esc(line), size=size,
                         fill="#C084FC", weight=700, anchor="middle"))
    return "".join(rows)


def draw_legend(x: float, y: float) -> str:
    """Two-row key: solid green = built and validated, dashed purple = future work.

    The figure encodes model status by line style only; without a key that code is invisible to
    a reader who did not write the generator, so the legend makes the claim boundary explicit.
    """
    rows = [text(x, y, "LEGEND", size=7.0, fill=DIM, weight=700)]
    entries = (
        (EMERALD, None, "arr-solid", "done"),
        (VIOLET, "7 6", "arr-dot", "possible future work"),
    )
    for i, (color, dash, marker_id, label) in enumerate(entries):
        row_y = y + 16.0 + i * 17.0
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        rows.append(
            f'<path d="M {x:.1f} {row_y:.1f} L {x + 26:.1f} {row_y:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="2.0" stroke-linecap="round"{dash_attr} '
            f'marker-end="url(#{marker_id})"/>'
        )
        rows.append(text(x + 34, row_y + 2.7, label, size=8.2, fill=SOFT))
    return "".join(rows)


CONFUSION_SHORT_LABELS: tuple[str, str, str] = ("no MIE", "167", "41")


def draw_confusion_matrix(
    x: float,
    y: float,
    matrix: list[list[int]],
    cell_w: float = 30.0,
    cell_h: float = 17.0,
) -> str:
    """Compact 3x3 LOOCV confusion matrix (rows = true MIE class, columns = prediction)."""
    rows: list[str] = []
    for c, label in enumerate(CONFUSION_SHORT_LABELS):
        rows.append(text(x + c * cell_w + cell_w / 2, y - 5, label,
                         size=7.2, fill=DIM, weight=700, anchor="middle"))
    for r, row_label in enumerate(CONFUSION_SHORT_LABELS):
        top = y + r * cell_h
        rows.append(text(x - 5, top + cell_h / 2 + 2.5, row_label,
                         size=7.2, fill=DIM, weight=700, anchor="end"))
        for c in range(3):
            value = int(matrix[r][c])
            if r == c and value:  # diagonal = confirmed MIE call
                fill, stroke, ink, fill_op, stroke_op = EMERALD, EMERALD, "#6ee7b7", 0.22, 0.75
            elif value:  # off-diagonal = the four confusions
                fill, stroke, ink, fill_op, stroke_op = ROSE, ROSE, "#fda4af", 0.20, 0.70
            else:
                fill, stroke, ink, fill_op, stroke_op = PANEL_BG, BORDER, "#475569", 1.0, 1.0
            rows.append(
                f'<rect x="{x + c * cell_w + 1:.1f}" y="{top + 1:.1f}" width="{cell_w - 2:.1f}" '
                f'height="{cell_h - 2:.1f}" rx="3" fill="{fill}" fill-opacity="{fill_op}" '
                f'stroke="{stroke}" stroke-opacity="{stroke_op}" stroke-width="0.9"/>'
            )
            rows.append(text(x + c * cell_w + cell_w / 2, top + cell_h / 2 + 3.4, str(value),
                             size=9.6, fill=ink, weight=700, anchor="middle"))
    return "".join(rows)


def draw_output_mie(x: float, y: float, w: float, h: float, mol: dict) -> str:
    """Answer 1 (solid arrow): the predicted MIE class of a compound, structure only.

    The title deliberately does not say "probability that the compound induces a MIE": what is
    reported is a 3-class discriminative assignment (no MIE / MIE 167 / MIE 41) obtained by cosine
    similarity to class prototypes in the frozen foundation embedding, validated leave-one-out on
    the 12 reference compounds. That is a structure-only prediction, not a mechanistic probability.
    """
    rows = [card(x, y, w, h, stroke=EMERALD, fill="#0c1f22", rx=11, width=1.6)]
    rows.append(text(x + 14, y + 32, "Predicted MIE from 12 compounds",
                     size=12.4, fill=EMERALD, weight=800))
    rows.append(text(x + 14, y + 48, "structure-only classifier · held-out LOOCV",
                     size=8.6, fill=SOFT))
    chip_svg, chip_w = tag(x + 14, y + 59, "MIE 167 · LXRα", AMBER)
    rows.append(chip_svg)
    chip2_svg, _ = tag(x + 14 + chip_w + 6, y + 59, "MIE 41 · BSEP", ROSE)
    rows.append(chip2_svg)
    rows.append(f'<line x1="{x + 14:.1f}" y1="{y + 86:.1f}" x2="{x + w - 14:.1f}" y2="{y + 86:.1f}" '
                f'stroke="{EMERALD}" stroke-opacity="0.3" stroke-width="1"/>')
    # left column: 50/50 dual consensus leave-one-out score (Cell 3 of report)
    rows.append(text(x + 14, y + 100, "50/50 consensus", size=8.0, fill=EMERALD, weight=700))
    rows.append(text(x + 14, y + 110, "leave-one-out", size=8.0, fill=DIM, weight=700))
    rows.append(text(x + 14, y + 126, f"{mol['loo_acc']}/{mol['n_compounds']}",
                     size=15.5, fill=EMERALD, weight=800))
    rows.append(text(x + 68, y + 126, "correct", size=8.6, fill=EMERALD, weight=600))
    rows.append(text(x + 14, y + 140, f"mean P {mol['loo_mean']:.2f}",
                     size=9.6, fill=AMBER, weight=700))
    rows.append(text(x + 14, y + 151, f"range {mol['loo_min']:.2f}–{mol['loo_max']:.2f}",
                     size=8.2, fill=AMBER, weight=600))
    rows.append(text(x + 14, y + 165, "uncertainty is", size=7.8, fill=MUTED, style="italic"))
    rows.append(text(x + 14, y + 175, "measurable →", size=7.8, fill=MUTED, style="italic"))
    # right column: 3x3 confusion matrix of the 50/50 Dual Consensus LOOCV run
    rows.append(draw_confusion_matrix(x + 140, y + 104, mol["loo_matrix"]))
    rows.append(text(x + 140, y + 169, "rows = true · cols = 50/50 LOOCV", size=7.1, fill=DIM))
    return "".join(rows)


def draw_output_adversity(x: float, y: float, w: float, h: float) -> str:
    """Answer 2 (dashed arrow): P(z_homeo, z_adversity | context, exposure, dose) - next step."""
    rows = [card(x, y, w, h, stroke=VIOLET, fill="#1d1020", rx=11, width=1.6)]
    rows.append(text(x + 14, y + 32,
                     "P(z" + sub("", "homeo", size=7.4, dy=2.8) + ", z"
                     + sub("", "adversity", size=7.4, dy=2.8) + " | context,",
                     size=12.0, fill=VIOLET, weight=800))
    rows.append(text(x + 14, y + 47, "exposure, dose)", size=12.0, fill=VIOLET, weight=800))
    rows.append(text(x + 14, y + 66, "Probability that it tips to z"
                     + sub("", "adversity", dy=2.4), size=8.6, fill=SOFT))
    rows.append(text(x + 14, y + 78, "instead of returning to z" + sub("", "homeo", dy=2.4)
                     + ", given context,", size=8.6, fill=SOFT))
    rows.append(text(x + 14, y + 90, "exposure and dose", size=8.6, fill=SOFT))
 
    return "".join(rows)


# ---------------------------------------------------------------- assembly ----
# Geometry contract (1040 x 460 canvas):
#   left column   x 18 .. 304      five data-source cards, 72 px high, 80 px apart
#   model node    x 372 .. 668     y 157 .. 287 (vertically centred on the middle card)
#   answer column x 744 .. 1022    card 1 y 58 .. 254, card 2 y 262 .. 392
#   gutter        x 668 .. 744     (72 px) holds the two orthogonal answer connectors
#   legend        x 380 .., y 386  two rows, 24 px above the provenance footer
IN_X, IN_W, IN_H, IN_STEP, IN_TOP = 18.0, 286.0, 72.0, 80.0, 26.0
HUB_X, HUB_W, HUB_Y, HUB_H = 372.0, 296.0, 157.0, 130.0
OUT_X, OUT_W = 744.0, 278.0
OUT1_Y, OUT1_H = 58.0, 196.0
OUT2_Y, OUT2_H = 262.0, 130.0
HUB_FAN_STEP = 22.0      # vertical distance between two entry points on the hub's left border
HUB_FAN_OUT = 20.0       # answer arrows leave the hub at centre -20 / +30


def build_svg(mol: dict, hep: dict) -> str:
    dist_n = hep["dist_matrix_shape"][0]
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="Five data sources (chemical structure, protein interactions, gene '
        f'expression, gene regulatory edges and, as future work, physiological MEA recordings) feed '
        f'one multimodal and multiscale AI model, which returns two answers: the predicted MIE class '
        f'of a compound from structure alone, and the probability that the hepatocyte tips to '
        f'z-adversity instead of returning to z-homeo.">',
        "<defs>"
        + marker("arr-in", SLATE)
        + marker("arr-solid", EMERALD)
        + marker("arr-dot", VIOLET)
        + "</defs>",
        card(4, 4, W - 8, H - 8, stroke=BORDER, fill=PANEL_BG, rx=16),
        text(IN_X, 17, "FIVE DATA SOURCES", size=7.6, fill=DIM, weight=700),
    ]

    # Chemical information is placed at the top (index 0) because it is the only source used so far (M0).
    inputs = [
        ("Chemical information", "SMILES · 2D OGB graph + SELFIES",
         f"{mol['n_compounds']} compounds · {mol['n_classes']} MIE classes", EMERALD, "chem"),
        ("Protein–protein interactions", "STRING-db v12 · taxon 9606 · PPI",
         f"{hep['n_genes']} genes · mean score {hep['string_mean']:.2f}", CYAN, "ppi"),
        ("Gene expression", "GTEx v8 · HPA · GEO GSE126848 · HepaRG",
         f"{hep['n_genes']} genes × {hep['n_states']} states", PINK, "expr"),
        ("Gene–gene interactions", "OECD AOP 34 & 27 + Physiological Maps",
         f"{hep['n_edges']} edges · {hep['n_pos']} + / {hep['n_neg']} − · [{dist_n},{dist_n}]",
         AMBER, "gene"),
        # Tissue-level electrophysiology is the one modality that cannot be derived from molecules,
        # genes or annotations: it is measured (MEA), and it is explicitly not built yet.
        ("Physiological measurements", "MEA · multi-electrode array · in vitro",
         "", VIOLET, "signal"),
    ]
    for i, (label, source, metric, color, icon) in enumerate(inputs):
        top = IN_TOP + i * IN_STEP
        parts.append(draw_input_card(IN_X, top, IN_W, IN_H, label, source, metric, color, icon))

    # Fan of five arrows into the single model node, symmetric around the hub's centre line:
    # - top source (i == 0, Chemical information): solid green arrow -> used and validated today!
    # - other 4 sources (i > 0): dashed purple arrows -> future work (M1.2 -> M4)
    hub_centre = HUB_Y + HUB_H / 2
    hub_targets = tuple(
        hub_centre + (i - (len(inputs) - 1) / 2) * HUB_FAN_STEP for i in range(len(inputs))
    )
    for i, target_y in enumerate(hub_targets):
        start_y = IN_TOP + i * IN_STEP + IN_H / 2
        if i == 0:
            parts.append(curve(IN_X + IN_W, start_y, HUB_X, target_y, EMERALD, "arr-solid",
                               dash=None, width=2.3, opacity=1.0))
        else:
            parts.append(curve(IN_X + IN_W, start_y, HUB_X, target_y, VIOLET, "arr-dot",
                               dash="7 6", width=1.9, opacity=0.92))

    parts.append(draw_hub(HUB_X, HUB_Y, HUB_W, HUB_H,
                          (fmt_params(mol["graphormer_params"]), fmt_params(mol["toxtransformer_params"]))))

    # The two answers leave the hub orthogonally and enter the answer cards straight through the
    # middle of the head, so the head never looks detached from its line.
    ans1_y = hub_centre - HUB_FAN_OUT
    ans2_y = hub_centre + HUB_FAN_OUT + 10
    out1_mid = OUT1_Y + OUT1_H / 2
    out2_mid = OUT2_Y + OUT2_H / 2
    parts.append(elbow(HUB_X + HUB_W, ans1_y, OUT_X, out1_mid, EMERALD, "arr-solid", width=2.3))
    parts.append(elbow(HUB_X + HUB_W, ans2_y, OUT_X, out2_mid, VIOLET, "arr-dot",
                       dash="7 6", width=2.3))
    # name the purple arrow on its vertical run (the gutter is only 72 px wide)
    parts.append(chip_on_arrow((HUB_X + HUB_W + OUT_X) / 2, (ans2_y + out2_mid) / 2,
                               ("e.g. (N)ODEs",), VIOLET))

    parts.append(draw_output_mie(OUT_X, OUT1_Y, OUT_W, OUT1_H, mol))
    parts.append(draw_output_adversity(OUT_X, OUT2_Y, OUT_W, OUT2_H))

    # key for the colour/line-style code of the whole figure
    parts.append(draw_legend(IN_X + HUB_X, 386.0))

    # provenance footer (MyGene.info is omitted: it only annotates gene/protein structure)
    parts.append(text(IN_X, H - 14, "Sources: " + " · ".join(hep["sources"]),
                      size=7.8, fill=DIM))
    parts.append(text(OUT_X - 6, H - 14,
                      f"{len(hep['sources'])} open sources · {hep['n_genes']} genes · "
                      f"{mol['n_compounds']} compounds · {mol['total_params'] / 1e6:.1f}M parameters",
                      size=7.8, fill=DIM, anchor="end"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    PUBLIC_DIR.mkdir(exist_ok=True)
    mol = load_molecular_metrics()
    hep = load_hepatocyte_metrics()

    numeric = [v for v in (*mol.values(), *hep.values()) if isinstance(v, (int, float))]
    assert all(math.isfinite(float(v)) for v in numeric), "non-finite metric detected"

    svg = build_svg(mol, hep)
    OUT_SVG.write_text(svg + "\n", encoding="utf-8")

    print("== Slide 2 - multimodal & multiscale AI graph ==")
    print(f"   figure           : {OUT_SVG.relative_to(ROOT)} ({len(svg):,} bytes)")
    print("   topology         : 5 data sources -> 1 model node -> 2 answers (solid / dashed)")
    print(f"   MIE leave-one-out: {mol['loo_acc']}/{mol['n_compounds']} correct, "
          f"mean P {mol['loo_mean']:.3f} ({mol['loo_min']:.3f}-{mol['loo_max']:.3f}) [graphormer]")
    print(f"   dual consensus   : {mol['cons_acc']}/{mol['n_compounds']} correct, "
          f"mean P {mol['cons_mean']:.3f}  (in-sample numbers are not reported)")
    print("   confusion matrix : rows = true class, columns = LOOCV prediction "
          "[no MIE, MIE 167, MIE 41]")
    for label, row in zip(CONFUSION_SHORT_LABELS, mol["loo_matrix"]):
        print(f"                      true {label:<7s} {row}")
    print(f"   parameters       : {mol['graphormer_params']:,} + {mol['toxtransformer_params']:,} "
          f"= {mol['total_params']:,}")
    print(f"   hepatocyte       : {hep['n_genes']} genes x {hep['n_features']} features x "
          f"{hep['n_states']} states, {hep['n_edges']} edges ({hep['n_pos']}+ / {hep['n_neg']}-)")
    print(f"   STRING score     : mean {hep['string_mean']:.3f} "
          f"({hep['string_min']:.2f}-{hep['string_max']:.2f})")
    print(f"   sources shown    : {len(hep['sources'])} of {hep['n_sources']} "
          f"({' · '.join(hep['sources'])})")
    print(f"   dist matrix      : {list(hep['dist_matrix_shape'])} ([CLS] prepended)")


if __name__ == "__main__":
    main()
