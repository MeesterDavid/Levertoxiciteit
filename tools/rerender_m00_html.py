#!/usr/bin/env python3
"""
Rebuild ``public/00_molecular_baseline_mie.html`` from the cached result JSON.

A full ``m00_molecular_baseline_mie.py`` run loads the Microsoft Graphormer and
ToxTransformer checkpoints on GPU 1 and re-calibrates the MIE heads (minutes of GPU
time). Every field that the renderers consume is already persisted in
``data/molecular_baseline_mie.json``, so *layout-only* changes (SVG geometry, text
wrapping, label rotation) can be re-rendered from that cache. The science - embeddings,
MIE probabilities, LOOCV, PCA coordinates - is reused byte-for-byte from the cache.

The leave-one-out validation block (accuracy, 3x3 confusion matrices, per-class
recall/precision/F1) is recomputed from the cached per-compound ``loocv`` and
``tox_loocv`` entries and written back into the JSON as ``validation``, so the
dashboard and the slide figure can report LOOCV without a GPU rerun.

This script also refreshes ``public/00_architecture_pipelines.html`` and the two
pipeline SVGs, because ``generate_standalone_html`` calls
``generate_pipeline_diagrams.main()``.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python tools/rerender_m00_html.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import m00_molecular_baseline_mie as m00  # noqa: E402


def main() -> None:
    json_path = REPO_ROOT / "data" / "molecular_baseline_mie.json"
    html_path = REPO_ROOT / "public" / "00_molecular_baseline_mie.html"
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} ontbreekt. Draai eerst `CUDA_VISIBLE_DEVICES=1 uv run python "
            f"m00_molecular_baseline_mie.py` op GPU 1."
        )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    compounds = payload["compounds"]
    graph_pca = np.asarray(payload["pca_coords_2d"], dtype=float)
    tox_pca = np.asarray(payload["toxtransformer_pca_coords_2d"], dtype=float)
    cos_sim = np.asarray(payload["cosine_similarity_matrix"], dtype=float)

    # LOOCV metrics only need the cached per-compound blocks (`loocv`, `tox_loocv`, `true_class`),
    # so the validation block can be refreshed without reloading the checkpoints on GPU 1.
    validation = m00.build_validation_metrics(compounds)
    payload["validation"] = validation
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    m00.print_validation_report(validation)

    print(f"[cache] {len(compounds)} compounds | PC shapes {graph_pca.shape}, {tox_pca.shape} | cos {cos_sim.shape}")
    m00.generate_standalone_html(compounds, graph_pca, tox_pca, cos_sim, html_path)
    print(f"[ok] re-rendered {html_path.relative_to(REPO_ROOT)} ({html_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
