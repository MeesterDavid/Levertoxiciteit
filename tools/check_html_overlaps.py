#!/usr/bin/env python3
"""
Static overlap auditor for the standalone HTML dashboards in ``public/``.

There is no headless browser in this environment, so nothing is rendered.
Instead every inline ``<svg>`` block is parsed and approximate geometry is
reconstructed (``<g transform="translate()/scale()">`` aware) to report the
three failure modes that actually make these dashboards unreadable:

  1. TEXT-TEXT      a label box collides with another label box
                    ("text is on top of other text").
  2. TEXT-COVERED   an opaque <rect>/<circle>/<ellipse> is painted AFTER a
                    label and covers its centre ("the graph is on top of
                    the title").
  3. TEXT-OUTSIDE   a label sticks out of the SVG viewBox (clipped text).

Text advance widths come from the Helvetica/AFM table (Inter is metrically
close), which is accurate to a few percent - plenty to separate a real
collision from kerning noise. A hit is only reported when the overlap
exceeds ``--min-ratio`` of the *smaller* box on BOTH axes.

Usage (GPU 1 convention, although this script needs no GPU):

    CUDA_VISIBLE_DEVICES=1 uv run python tools/check_html_overlaps.py
    CUDA_VISIBLE_DEVICES=1 uv run python tools/check_html_overlaps.py --out /tmp/overlaps.md
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = REPO_ROOT / "public"

# Helvetica/AFM advance widths in em units (subset, sane defaults elsewhere).
_EM = {
    " ": 0.278, "!": 0.278, '"': 0.355, "#": 0.556, "$": 0.556, "%": 0.889, "&": 0.667,
    "'": 0.191, "(": 0.333, ")": 0.333, "*": 0.389, "+": 0.584, ",": 0.278, "-": 0.333,
    ".": 0.278, "/": 0.278, ":": 0.278, ";": 0.278, "<": 0.584, "=": 0.584, ">": 0.584,
    "?": 0.556, "@": 1.015, "[": 0.278, "\\": 0.278, "]": 0.278, "^": 0.469, "_": 0.556,
    "`": 0.333, "{": 0.334, "|": 0.26, "}": 0.334, "~": 0.584,
    "A": 0.667, "B": 0.667, "C": 0.722, "D": 0.722, "E": 0.667, "F": 0.611, "G": 0.778,
    "H": 0.722, "I": 0.278, "J": 0.5, "K": 0.667, "L": 0.556, "M": 0.833, "N": 0.722,
    "O": 0.778, "P": 0.667, "Q": 0.778, "R": 0.722, "S": 0.667, "T": 0.611, "U": 0.722,
    "V": 0.667, "W": 0.944, "X": 0.667, "Y": 0.667, "Z": 0.611,
    "a": 0.556, "b": 0.556, "c": 0.5, "d": 0.556, "e": 0.556, "f": 0.278, "g": 0.556,
    "h": 0.556, "i": 0.222, "j": 0.222, "k": 0.5, "l": 0.222, "m": 0.833, "n": 0.556,
    "o": 0.556, "p": 0.556, "q": 0.556, "r": 0.333, "s": 0.5, "t": 0.278, "u": 0.556,
    "v": 0.5, "w": 0.722, "x": 0.5, "y": 0.5, "z": 0.5,
    "0": 0.556, "1": 0.556, "2": 0.556, "3": 0.556, "4": 0.556, "5": 0.556, "6": 0.556,
    "7": 0.556, "8": 0.556, "9": 0.556,
    "\u2014": 1.0, "\u2013": 0.556, "\u2192": 1.0, "\u2190": 1.0, "\u2191": 1.0, "\u2193": 1.0,
    "\u2264": 0.584, "\u2265": 0.584, "\u00b1": 0.584, "\u00d7": 0.584, "\u2022": 0.35,
    "\u00b7": 0.278, "\u2248": 0.584, "\u03b1": 0.6, "\u03b2": 0.6, "\u03b3": 0.6,
    "\u03ba": 0.5, "\u03bc": 0.83, "\u03c3": 0.6, "\u0394": 0.7, "\u2713": 0.8, "\u2717": 0.8,
}

_NUM_RE = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _nums(value: str | None) -> list[float]:
    """Extract every number from an SVG attribute value."""
    if not value:
        return []
    return [float(v) for v in _NUM_RE.findall(value)]


def text_width(text: str, font_size: float, bold: bool = False, letter_spacing: float = 0.0) -> float:
    """Approximate rendered width (px) of a single-line SVG label."""
    width = 0.0
    for ch in text:
        width += _EM.get(ch, 1.15 if ord(ch) > 0x2000 else 0.6)
    width *= font_size
    if bold:
        width *= 1.04
    width += letter_spacing * max(len(text) - 1, 0)
    return width



@dataclass
class Box:
    """Axis-aligned bounding box of one painted SVG element."""

    x0: float
    y0: float
    x1: float
    y1: float
    kind: str          # "text" | "rect" | "circle" | "ellipse"
    label: str
    order: int
    opaque: bool = False

    @property
    def w(self) -> float:
        return max(self.x1 - self.x0, 0.0)

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 0.0)

    def overlap(self, other: "Box") -> tuple[float, float]:
        """Return (overlap width, overlap height) with another box."""
        ow = min(self.x1, other.x1) - max(self.x0, other.x0)
        oh = min(self.y1, other.y1) - max(self.y0, other.y0)
        return max(ow, 0.0), max(oh, 0.0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def fmt(self) -> str:
        return f"[{self.x0:.0f},{self.y0:.0f} \u2192 {self.x1:.0f},{self.y1:.0f}]"


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _apply_transform(transform: str | None, ctx: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Fold an SVG transform list into the running (dx, dy, sx, sy) context."""
    dx, dy, sx, sy = ctx
    if not transform:
        return ctx
    for name, raw in re.findall(r"([a-zA-Z]+)\s*\(([^)]*)\)", transform):
        vals = _nums(raw)
        if not vals:
            continue
        if name == "translate":
            dx += vals[0] * sx
            if len(vals) > 1:
                dy += vals[1] * sy
        elif name == "scale":
            fx = vals[0]
            fy = vals[1] if len(vals) > 1 else fx
            sx *= fx
            sy *= fy
        elif name == "matrix" and len(vals) == 6:
            dx += vals[4] * sx
            dy += vals[5] * sy
            sx *= abs(vals[0]) or 1.0
            sy *= abs(vals[3]) or 1.0
    return dx, dy, sx, sy


def _rotation_of(el: ET.Element) -> tuple[float, float, float] | None:
    """Return (angle_deg, cx, cy) for a ``transform="rotate(a cx cy)"`` attribute."""
    match = re.search(r"rotate\s*\(([^)]*)\)", el.get("transform") or "")
    if not match:
        return None
    vals = _nums(match.group(1))
    if not vals:
        return None
    cx, cy = (vals[1], vals[2]) if len(vals) >= 3 else (0.0, 0.0)
    return vals[0], cx, cy


def _rotate_bbox(x0: float, y0: float, x1: float, y1: float, angle_deg: float, cx: float, cy: float):
    """Axis-aligned bounds of a rectangle rotated about (cx, cy)."""
    theta = math.radians(angle_deg)
    ca, sa = math.cos(theta), math.sin(theta)
    xs, ys = [], []
    for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        xs.append(cx + (px - cx) * ca - (py - cy) * sa)
        ys.append(cy + (px - cx) * sa + (py - cy) * ca)
    return min(xs), min(ys), max(xs), max(ys)


def _element_text(el: ET.Element) -> str:
    """Concatenate the character data of a <text> element (incl. <tspan>)."""
    chunks: list[str] = []
    if el.text:
        chunks.append(el.text)
    for child in el:
        name = _local(child.tag)
        if name in ("tspan", "a") and child.text:
            chunks.append(child.text)
        if child.tail:
            chunks.append(child.tail)
    return html_lib.unescape("".join(chunks)).strip()


def _float_attr(el: ET.Element, name: str, default: float) -> float:
    vals = _nums(el.get(name))
    return vals[0] if vals else default


@dataclass
class Finding:
    """One reported layout defect inside one inline SVG."""

    kind: str
    detail: str


def collect_boxes(svg: ET.Element) -> tuple[list[Box], tuple[float, float]]:
    """Walk an <svg> subtree and return every text/shape box + the viewBox size."""
    vb = _nums(svg.get("viewBox"))
    view = (vb[2], vb[3]) if len(vb) == 4 else (
        _float_attr(svg, "width", 1000.0), _float_attr(svg, "height", 600.0)
    )
    boxes: list[Box] = []
    counter = {"n": 0}

    def walk(el: ET.Element, ctx: tuple[float, float, float, float], inherited: dict[str, str]) -> None:
        style = dict(inherited)
        for key in ("font-size", "font-weight", "text-anchor", "letter-spacing"):
            if el.get(key) is not None:
                style[key] = el.get(key)  # type: ignore[assignment]
        local_ctx = _apply_transform(el.get("transform"), ctx)
        dx, dy, sx, sy = local_ctx
        tag = _local(el.tag)

        if tag == "text":
            counter["n"] += 1
            content = _element_text(el)
            x = _float_attr(el, "x", 0.0) * sx + dx
            y = _float_attr(el, "y", 0.0) * sy + dy
            fs = _nums(style.get("font-size"))[0] if _nums(style.get("font-size")) else 16.0
            fs *= sy
            bold = str(style.get("font-weight", "")).strip() in ("700", "800", "900", "bold")
            ls_vals = _nums(style.get("letter-spacing"))
            ls = (ls_vals[0] if ls_vals else 0.0) * sx
            width = text_width(content, fs, bold, ls) * sx
            anchor = str(style.get("text-anchor", "start")).strip()
            if anchor == "middle":
                x0, x1 = x - width / 2.0, x + width / 2.0
            elif anchor == "end":
                x0, x1 = x - width, x
            else:
                x0, x1 = x, x + width
            y0, y1 = y - 0.78 * fs, y + 0.22 * fs
            rotation = _rotation_of(el)
            if rotation:
                angle, rcx, rcy = rotation
                x0, y0, x1, y1 = _rotate_bbox(x0, y0, x1, y1, angle, rcx * sx + dx, rcy * sy + dy)
            boxes.append(Box(x0, y0, x1, y1, "text", content or "(leeg)", counter["n"]))
        elif tag in ("rect", "circle", "ellipse"):
            counter["n"] += 1
            fill = str(el.get("fill", style.get("fill", "#000"))).strip().lower()
            fill_op = _nums(el.get("fill-opacity") or style.get("fill-opacity"))
            opacity = (fill_op[0] if fill_op else 1.0) * (_nums(el.get("opacity"))[0] if _nums(el.get("opacity")) else 1.0)
            opaque = fill not in ("none", "transparent", "") and opacity >= 0.5
            if tag == "rect":
                x = _float_attr(el, "x", 0.0) * sx + dx
                y = _float_attr(el, "y", 0.0) * sy + dy
                w = _float_attr(el, "width", 0.0) * sx
                h = _float_attr(el, "height", 0.0) * sy
                box = Box(x, y, x + w, y + h, "rect", f"<rect {w:.0f}x{h:.0f}>", counter["n"], opaque)
            elif tag == "circle":
                cx = _float_attr(el, "cx", 0.0) * sx + dx
                cy = _float_attr(el, "cy", 0.0) * sy + dy
                r = _float_attr(el, "r", 0.0) * sx
                box = Box(cx - r, cy - r, cx + r, cy + r, "circle", f"<circle r={r:.0f}>", counter["n"], opaque)
            else:
                cx = _float_attr(el, "cx", 0.0) * sx + dx
                cy = _float_attr(el, "cy", 0.0) * sy + dy
                rx = _float_attr(el, "rx", 0.0) * sx
                ry = _float_attr(el, "ry", 0.0) * sy
                box = Box(cx - rx, cy - ry, cx + rx, cy + ry, "ellipse", f"<ellipse {rx:.0f}x{ry:.0f}>", counter["n"], opaque)
            boxes.append(box)

        for child in el:
            if _local(child.tag) in ("defs", "marker", "clipPath", "title", "desc", "style"):
                continue
            walk(child, local_ctx, style)

    walk(svg, (0.0, 0.0, 1.0, 1.0), {})
    return boxes, view



def find_overlaps(boxes: list[Box], view: tuple[float, float], min_ratio: float) -> list[Finding]:
    """Detect text collisions, covered labels and clipped labels."""
    findings: list[Finding] = []
    texts = [b for b in boxes if b.kind == "text"]
    shapes = [b for b in boxes if b.kind != "text"]
    vw, vh = view
    pad = 2.0

    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            ow, oh = a.overlap(b)
            if ow <= 0 or oh <= 0:
                continue
            if ow > min_ratio * min(a.w, b.w) and oh > min_ratio * min(a.h, b.h):
                findings.append(
                    Finding("TEXT-TEXT", f'"{a.label[:70]}" {a.fmt()} botsen met "{b.label[:70]}" {b.fmt()}')
                )

    for t in texts:
        cx, cy = (t.x0 + t.x1) / 2.0, (t.y0 + t.y1) / 2.0
        for s in shapes:
            if s.order <= t.order or not s.opaque:
                continue
            if s.contains_point(cx, cy):
                findings.append(
                    Finding("TEXT-COVERED", f'"{t.label[:70]}" {t.fmt()} wordt bedekt door {s.label} {s.fmt()}')
                )

    for t in texts:
        if t.x0 < -pad or t.y0 < -pad or t.x1 > vw + pad or t.y1 > vh + pad:
            findings.append(
                Finding("TEXT-OUTSIDE", f'"{t.label[:70]}" {t.fmt()} valt buiten viewBox {vw:.0f}x{vh:.0f}')
            )
    return findings


_MARIMO_HTML_OUTPUT_RE = re.compile(r'"text/html"\s*:\s*("(?:[^"\\]|\\.)*")')


def _marimo_embedded_html(raw: str) -> list[str]:
    """
    Decode ``mo.Html`` outputs of a Marimo static export.

    Marimo >= 0.16 serialises every cell output into ``__MARIMO_MOUNT_CONFIG__`` as a
    JSON string with ``\\u003C``-escaped markup, so a plain ``<svg`` regex finds nothing
    in the file even though eleven charts are present. These payloads are decoded here
    and audited like any inline SVG.
    """
    payloads: list[str] = []
    for match in _MARIMO_HTML_OUTPUT_RE.finditer(raw):
        try:
            decoded = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if "<svg" in decoded:
            payloads.append(decoded)
    return payloads


def audit_html(path: Path, min_ratio: float) -> list[tuple[str, tuple[float, float], list[Finding]]]:
    """Audit every inline SVG of one HTML file; returns [(label, viewBox, findings)]."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    svg_re = re.compile(r"<svg\b.*?</svg>", flags=re.S)

    sources: list[tuple[str, str]] = [("inline", m.group(0)) for m in svg_re.finditer(raw)]
    for payload in _marimo_embedded_html(raw):
        sources.extend(("marimo", m.group(0)) for m in svg_re.finditer(payload))

    results: list[tuple[str, tuple[float, float], list[Finding]]] = []
    counters = {"inline": 0, "marimo": 0}
    for origin, chunk in sources:
        counters[origin] += 1
        label = f"SVG #{counters[origin]}" + (" (marimo output)" if origin == "marimo" else "")
        chunk = re.sub(r"&nbsp;", " ", chunk)
        try:
            root = ET.fromstring(chunk)
        except ET.ParseError as exc:  # malformed inline SVG: report, keep going
            results.append((label, (0.0, 0.0), [Finding("PARSE-ERROR", str(exc))]))
            continue
        boxes, view = collect_boxes(root)
        results.append((label, view, find_overlaps(boxes, view, min_ratio)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Static overlap auditor for public/*.html")
    parser.add_argument("--min-ratio", type=float, default=0.35,
                        help="minimum overlap fraction of the smaller box on both axes (default 0.35)")
    parser.add_argument("--out", type=Path, default=None, help="write the markdown report to this path")
    parser.add_argument("files", nargs="*", type=Path, help="HTML files (default: every public/*.html)")
    args = parser.parse_args()

    files = args.files or sorted(PUBLIC_DIR.glob("*.html"))
    lines: list[str] = ["# Overlap audit", ""]
    total = 0

    for path in files:
        if not path.is_absolute():
            path = PUBLIC_DIR / path if (PUBLIC_DIR / path).exists() else path
        if not path.exists():
            lines.append(f"## {path.name}\n- **ONTBREEKT**\n")
            continue
        results = audit_html(path, args.min_ratio)
        per_file = sum(len(f) for _, _, f in results)
        total += per_file
        lines.append(f"## {path.name} — {len(results)} inline SVG('s), {per_file} bevinding(en)")
        print(f"\n=== {path.name} === {len(results)} SVG('s), {per_file} bevinding(en)")
        for idx, view, findings in results:
            if not findings:
                continue
            header = f"  --- {idx} (viewBox {view[0]:.0f}x{view[1]:.0f}) ---"
            print(header)
            lines.append("\n```text")
            lines.append(f"{idx} (viewBox {view[0]:.0f}x{view[1]:.0f})")
            for f in findings:
                row = f"{f.kind:13s} {f.detail}"
                print("    " + row)
                lines.append(row)
            lines.append("```")
        if per_file == 0:
            lines.append("- geen overlappende of afgekapte tekst gevonden ✅")

    lines.append(f"\n**Totaal: {total} bevinding(en)**")
    print(f"\nTOTAAL: {total} bevinding(en)")
    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Rapport weggeschreven naar {args.out}")


if __name__ == "__main__":
    main()

