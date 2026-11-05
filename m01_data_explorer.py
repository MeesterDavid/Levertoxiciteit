import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Milestone 1.1 — Hepatocyte AOP & Graphormer Data Explorer")


@app.cell
def _():
    import html as html_lib
    import json
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, html_lib, json, make_subplots, mo, np, pd


@app.cell
def _(Path, json, mo):
    data_path = Path("data/hepatocyte_baseline_data.json")
    if not data_path.exists():
        import fetch_real_hepatocyte_data

        fetch_real_hepatocyte_data.main()

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    genes = dataset["genes"]
    edges = dataset["edges"]
    matrices = dataset["matrices"]
    meta = dataset["metadata"]

    intro_md = mo.md(
        f"""
        # 🧬 Milestone 1, Step 1.1: Hepatocyte AOP Network & Real Data Explorer
        **Project:** *Multimodal & Multiscale AI for Liver Toxicity (Graphormer + Neural ODE)* | **Framework:** *Research Group Innovative Testing (HU University of Applied Sciences Utrecht)*

        ### 📘 General User Guide & Overview
        This notebook constitutes **Milestone 1, Step 1.1** of the hybrid **Graphormer + Neural ODE Liver Toxicity Simulator**. Before training/initializing the `HepatocyteGraphormer` in **Step 1.2** to summarize a liver cell into a 64-dimensional latent state vector ($z_0 \\in \\mathbb{{R}}^{{64}}$), we explore and validate the **real biological input data** here:
        1. **The 18 Key Genes (`index 0 through 17`):** Divided across four coupled biological modules:
           - 🔴 **Group 1 — `AOP34_Steatosis` (`NR1H3`, `SREBF1`, `FASN`, `ACACA`, `SCD`, `CD36`):** The lipogenic cascade of OECD AOP 34 that leads to fatty liver (steatosis) under excessive LXR activation.
           - 🟢 **Group 2 — `Homeostasis` (`PPARA`, `CPT1A`, `MTTP`, `APOB`):** The physiological counter-force from the *Physiological Map* that oxidizes excess fatty acids ($\beta$-oxidation) or exports them (VLDL) to keep the cell in the healthy attractor ($z_{{\\text{{homeo}}}}$).
           - 🟠 **Group 3 — `AOP27_Cholestasis` (`NR1H4`, `ABCB11`, `CYP7A1`, `SLC10A1`):** Bile acid homeostasis and the target of OECD AOP 27 (`ABCB11`/BSEP inhibition) that leads to toxic bile accumulation (cholestasis) when blocked.
           - 🟣 **Group 4 — `Stress_Apoptosis` (`NFE2L2`, `NFKB1`, `TP53`, `CASP3`):** The convergent injury and inflammatory pathway that drives the tipping point toward cell death (*Adverse Outcome*) when homeostatic buffers fail.
        2. **Live API Data + Empirical Expression Profiles:** Combined from **MyGene.info v3** (gene length, exons, Ensembl/UniProt), **STRING-db v12.0** (150 human protein-protein interaction scores), **GTEx/HPA** (healthy liver TPM), and **NCBI GEO** (pathological expression in Steatosis and Cholestasis).
        """
    )
    intro_md
    return data_path, dataset, edges, genes, intro_md, matrices, meta


@app.cell
def _(mo):
    state_selector = mo.ui.dropdown(
        options=[
            "Steatosis (AOP 34 - LXR/NAFLD)",
            "Healthy (Homeostasis Baseline)",
            "Cholestasis (AOP 27 - BSEP Inhibition)",
            "Delta: Steatosis vs Healthy (log2FC)",
            "Delta: Cholestasis vs Healthy (log2FC)",
        ],
        value="Steatosis (AOP 34 - LXR/NAFLD)",
        label="🔬 Cellular State on Network (Node Size & Color):",
    )

    group_filter = mo.ui.multiselect(
        options=[
            "AOP34_Steatosis",
            "Homeostasis",
            "AOP27_Cholestasis",
            "Stress_Apoptosis",
        ],
        value=[
            "AOP34_Steatosis",
            "Homeostasis",
            "AOP27_Cholestasis",
            "Stress_Apoptosis",
        ],
        label="🧩 Filter Functional AOP Groups:",
    )

    string_slider = mo.ui.slider(
        start=0.0,
        stop=0.95,
        step=0.05,
        value=0.50,
        label="🔗 Min. STRING-db Interaction Score:",
    )

    show_extra_ppi = mo.ui.checkbox(
        value=True,
        label="Also show non-causal STRING-db protein interactions (dashed background lines)",
    )

    norm_selector = mo.ui.dropdown(
        options=[
            "Z-Score (Normalized vs. Healthy Baseline)",
            "Raw Biological Values (log2 TPM, log10 bp, exons, etc.)",
        ],
        value="Z-Score (Normalized vs. Healthy Baseline)",
        label="📊 Scale for Feature Matrix X [18, 8]:",
    )

    controls_view = mo.vstack(
        [
            mo.md(
                """
                ### 🎛️ Interactive Filters Guide (Cell 1)
                - **How to use:** Use the **Cellular State on Network** dropdown to directly switch the biological network (Visualization 1) and the Feature Matrix (Visualization 3) between *Healthy*, *Steatosis (AOP 34)*, *Cholestasis (AOP 27)*, and the *Fold-Change shifts ($\Delta\log_2\text{FC}$)*. Use the **STRING-db slider** to filter out weaker protein interactions.
                """
            ),
            mo.hstack(
                [state_selector, group_filter, norm_selector],
                justify="start",
                gap=1.5,
            ),
            mo.hstack([string_slider, show_extra_ppi], justify="start", gap=2.0),
        ]
    )
    controls_view
    return (
        controls_view,
        group_filter,
        norm_selector,
        show_extra_ppi,
        state_selector,
        string_slider,
    )


@app.cell
def _(
    edges,
    genes,
    go,
    group_filter,
    html_lib,
    matrices,
    mo,
    np,
    show_extra_ppi,
    state_selector,
    string_slider,
):
    NODE_POS = {
        "NR1H3": (0.09, 0.82),
        "SREBF1": (0.29, 0.82),
        "FASN": (0.52, 0.91),
        "ACACA": (0.52, 0.77),
        "SCD": (0.52, 0.63),
        "CD36": (0.29, 0.61),
        "PPARA": (0.29, 0.43),
        "CPT1A": (0.52, 0.46),
        "MTTP": (0.69, 0.46),
        "APOB": (0.85, 0.46),
        "NR1H4": (0.09, 0.22),
        "ABCB11": (0.36, 0.27),
        "CYP7A1": (0.26, 0.09),
        "SLC10A1": (0.48, 0.13),
        "NFE2L2": (0.69, 0.71),
        "NFKB1": (0.76, 0.25),
        "TP53": (0.87, 0.68),
        "CASP3": (0.93, 0.35),
    }

    GROUP_COLORS = {
        "AOP34_Steatosis": "#FF5964",
        "Homeostasis": "#2EC4B6",
        "AOP27_Cholestasis": "#FF9F1C",
        "Stress_Apoptosis": "#A855F7",
    }

    GROUP_LABELS = {
        "AOP34_Steatosis": "AOP 34: Lipogenesis & Steatosis",
        "Homeostasis": "Homeostasis: β-Oxidation & VLDL Export",
        "AOP27_Cholestasis": "AOP 27: Bile Acid & Cholestasis",
        "Stress_Apoptosis": "Convergent Stress & Apoptosis (AO)",
    }

    def render_network_svg(selected_state_str, active_groups_set, min_str_score, include_bg_ppi):
        """Generates a crisp, interactive SVG (with native hover tooltips) that works in every viewer."""
        W, H = 1160, 620
        pad_x, pad_top, pad_bot = 60, 78, 56
        plot_w = W - 2 * pad_x
        plot_h = H - pad_top - pad_bot

        def to_px(nx, ny):
            return pad_x + nx * plot_w, pad_top + (1.0 - ny) * plot_h

        is_delta = selected_state_str.startswith("Delta:")
        string_adj = np.array(matrices["string_adjacency_18"], dtype=float)
        causal_pairs = {(e["source_idx"], e["target_idx"]) for e in edges} | {
            (e["target_idx"], e["source_idx"]) for e in edges
        }

        svg = [
            f'<svg viewBox="0 0 {W} {H}" width="100%" height="auto" style="background:#0F172A;border-radius:10px;border:1px solid #1E293B;font-family:Inter,system-ui,sans-serif;">',
            "<defs>",
            '<marker id="arrow-act" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#10B981"/></marker>',
            '<marker id="arrow-rep" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#F43F5E"/></marker>',
            '<filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.6"/></filter>',
            "</defs>",
            f'<text x="22" y="28" fill="#F8FAFC" font-size="16" font-weight="700">Visualization 1: Causal Hepatocyte AOP Network Graph (AOP 34 Steatosis + AOP 27 Cholestasis + Homeostasis)</text>',
            f'<text x="22" y="48" fill="#94A3B8" font-size="12.5">Active state: {html_lib.escape(selected_state_str)} | Min. STRING-db score: {min_str_score:.2f} | Hover over nodes and arrows for OECD AOP citations</text>',
        ]

        # Column headers
        headers = [
            (0.09, "1. Nuclear Receptors (MIE)"),
            (0.32, "2. Transcription & Influx (KE)"),
            (0.55, "3. Metabolic Enzymes & Export"),
            (0.84, "4. Stress & Apoptosis (AO)"),
        ]
        for hx, htxt in headers:
            px, _ = to_px(hx, 1.0)
            svg.append(
                f'<text x="{px:.1f}" y="70" fill="#64748B" font-size="11.5" font-weight="700" text-anchor="middle">{html_lib.escape(htxt)}</text>'
            )

        # 1. Background STRING-db lines
        if include_bg_ppi:
            for i_idx in range(len(genes)):
                for j_idx in range(i_idx + 1, len(genes)):
                    if (i_idx, j_idx) in causal_pairs:
                        continue
                    s_val = string_adj[i_idx, j_idx]
                    if s_val >= max(min_str_score, 0.45):
                        g_i, g_j = genes[i_idx], genes[j_idx]
                        if g_i["group"] in active_groups_set and g_j["group"] in active_groups_set:
                            x0, y0 = to_px(*NODE_POS[g_i["symbol"]])
                            x1, y1 = to_px(*NODE_POS[g_j["symbol"]])
                            tip = f"STRING-db PPI cross-talk: {g_i['symbol']} ↔ {g_j['symbol']} (score={s_val:.3f})"
                            svg.append(
                                f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#475569" stroke-opacity="0.30" stroke-width="1.3" stroke-dasharray="3,4"><title>{html_lib.escape(tip)}</title></line>'
                            )

        # 2. Causal AOP edges
        for e in edges:
            if e["string_score"] < min_str_score:
                continue
            g_src = genes[e["source_idx"]]
            g_dst = genes[e["target_idx"]]
            if g_src["group"] not in active_groups_set or g_dst["group"] not in active_groups_set:
                continue
            x0, y0 = to_px(*NODE_POS[g_src["symbol"]])
            x1, y1 = to_px(*NODE_POS[g_dst["symbol"]])
            dx, dy = x1 - x0, y1 - y0
            dist = max(np.hypot(dx, dy), 1.0)
            ux, uy = dx / dist, dy / dist
            sx, sy = x0 + ux * 24.0, y0 + uy * 24.0
            ex, ey = x1 - ux * 28.0, y1 - uy * 28.0

            is_act = e["sign"] > 0
            col = "#10B981" if is_act else "#F43F5E"
            dash = "" if is_act else 'stroke-dasharray="7,4"'
            marker = "url(#arrow-act)" if is_act else "url(#arrow-rep)"
            lw = 1.8 + 2.2 * float(e["string_score"])
            tip = (
                f"{e['source_symbol']} → {e['target_symbol']} ({e['interaction_type']})\n"
                f"OECD AOP Citation: {e['citation']}\n"
                f"STRING-db Score: {e['string_score']:.3f}"
            )
            svg.append(
                f'<g class="aop-edge" style="cursor:pointer;">'
                f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{col}" stroke-width="{lw:.2f}" {dash} marker-end="{marker}" opacity="0.92">'
                f"<title>{html_lib.escape(tip)}</title></line></g>"
            )

        # 3. The 18 Gene nodes
        for _g in genes:
            if _g["group"] not in active_groups_set:
                continue
            sym = _g["symbol"]
            grp_key = _g["group"]
            px, py = to_px(*NODE_POS[sym])

            if selected_state_str == "Healthy (Homeostasis Baseline)":
                val_expr = _g["expr_healthy_log2"]
                val_tpm = _g["expr_healthy_tpm"]
                radius = 14.0 + 1.8 * val_expr
                fill_col = GROUP_COLORS[grp_key]
                b_col = "#F8FAFC"
                sub_lbl = f"{val_tpm:.0f} TPM"
            elif selected_state_str == "Steatosis (AOP 34 - LXR/NAFLD)":
                val_expr = _g["expr_steatosis_log2"]
                val_tpm = _g["expr_steatosis_tpm"]
                radius = 14.0 + 1.8 * val_expr
                fill_col = GROUP_COLORS[grp_key]
                b_col = "#FDE047" if _g["log2fc_steatosis_vs_healthy"] > 0.8 else "#F8FAFC"
                sub_lbl = f"{val_tpm:.0f} TPM ({_g['log2fc_steatosis_vs_healthy']:+.2f} FC)"
            elif selected_state_str == "Cholestasis (AOP 27 - BSEP Inhibition)":
                val_expr = _g["expr_cholestasis_log2"]
                val_tpm = _g["expr_cholestasis_tpm"]
                radius = 14.0 + 1.8 * val_expr
                fill_col = GROUP_COLORS[grp_key]
                b_col = "#FDE047" if abs(_g["log2fc_cholestasis_vs_healthy"]) > 0.8 else "#F8FAFC"
                sub_lbl = f"{val_tpm:.0f} TPM ({_g['log2fc_cholestasis_vs_healthy']:+.2f} FC)"
            elif selected_state_str == "Delta: Steatosis vs Healthy (log2FC)":
                fc = _g["log2fc_steatosis_vs_healthy"]
                radius = 15.0 + 5.5 * abs(fc)
                fill_col = "#EF4444" if fc >= 0 else "#3B82F6"
                b_col = GROUP_COLORS[grp_key]
                sub_lbl = f"Δlog2FC: {fc:+.2f}"
            else:
                fc = _g["log2fc_cholestasis_vs_healthy"]
                radius = 15.0 + 5.5 * abs(fc)
                fill_col = "#EF4444" if fc >= 0 else "#3B82F6"
                b_col = GROUP_COLORS[grp_key]
                sub_lbl = f"Δlog2FC: {fc:+.2f}"

            tip = (
                f"{sym} — {_g['name']}\n"
                f"Group: {GROUP_LABELS[grp_key]} ({_g['role_label']})\n"
                f"OECD AOP: {_g['aop_ref']}\n"
                f"Description: {_g['description']}\n"
                f"---\n"
                f"Healthy (GTEx/HPA): {_g['expr_healthy_tpm']:.1f} TPM (log2={_g['expr_healthy_log2']:.2f})\n"
                f"Steatosis (AOP 34): {_g['expr_steatosis_tpm']:.1f} TPM (log2={_g['expr_steatosis_log2']:.2f}, Δ={_g['log2fc_steatosis_vs_healthy']:+.2f})\n"
                f"Cholestasis (AOP 27): {_g['expr_cholestasis_tpm']:.1f} TPM (log2={_g['expr_cholestasis_log2']:.2f}, Δ={_g['log2fc_cholestasis_vs_healthy']:+.2f})\n"
                f"MyGene.info: {_g['gene_length_bp']:,} bp | {_g['exon_count']} exons | {_g['protein_length_aa']} aa | Chr {_g['chr']}\n"
                f"Network: In-deg={_g['in_degree']} | Out-deg={_g['out_degree']} | Mean STRING={_g['mean_string_score']:.3f}"
            )
            bw = 3.2 if (b_col == "#FDE047" or is_delta) else 2.0
            svg.append(
                f'<g style="cursor:pointer;" filter="url(#glow)">'
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{fill_col}" stroke="{b_col}" stroke-width="{bw}"/>'
                f'<text x="{px:.1f}" y="{py + 4:.1f}" fill="#FFFFFF" font-size="11" font-weight="700" text-anchor="middle">{sym}</text>'
                f'<text x="{px:.1f}" y="{py - radius - 6:.1f}" fill="#E2E8F0" font-size="10.5" font-weight="600" text-anchor="middle">{html_lib.escape(sub_lbl)}</text>'
                f"<title>{html_lib.escape(tip)}</title></g>"
            )

        # Legend at bottom
        leg_items = [
            ("#FF5964", "AOP 34: Steatosis/Lipogenesis"),
            ("#2EC4B6", "Homeostasis: β-Oxidation & VLDL"),
            ("#FF9F1C", "AOP 27: Cholestasis/Bile Acid"),
            ("#A855F7", "Stress & Apoptosis (AO)"),
            ("#10B981", "Activation (+1 arrow)"),
            ("#F43F5E", "Repression / Injury (-1 arrow)"),
        ]
        lx_cur = 28
        for lcol, llbl in leg_items:
            svg.append(f'<circle cx="{lx_cur}" cy="{H - 18}" r="6" fill="{lcol}"/>')
            svg.append(f'<text x="{lx_cur + 11}" y="{H - 14}" fill="#CBD5E1" font-size="11.5">{html_lib.escape(llbl)}</text>')
            lx_cur += len(llbl) * 6.8 + 34

        svg.append("</svg>")
        return "\n".join(svg)

    active_groups = set(group_filter.value) if group_filter.value else set(GROUP_COLORS.keys())
    net_svg_str = render_network_svg(
        state_selector.value,
        active_groups,
        float(string_slider.value),
        bool(show_extra_ppi.value),
    )

    guide_v1 = mo.md(
        """
        ### 🕸️ User Guide Visualization 1 — Causal Hepatocyte AOP Network Graph
        - **How to use this cell:**
          - Select above under **🔬 Cellular State on Network** which state you want to project onto the nodes (*Steatosis AOP 34*, *Healthy*, *Cholestasis AOP 27*, or *$\Delta\log_2\text{FC}$*).
          - **Hover your mouse** over a **gene node** to read the exact Ensembl/UniProt ID, exon count, gene length, and expression across all 3 states.
          - **Hover your mouse** over a **green (+1) or red (-1) arrow** to read the **official OECD AOP Key Event citation** and **STRING-db interaction score**.
        - **What do you see in this chart?**
          - In **Steatosis (AOP 34)**, you see in the upper left how activation of `NR1H3` (LXR$\alpha$) via `SREBF1` causes the lipogenic enzymes (`FASN`, `ACACA`, `SCD`) and fatty acid influx (`CD36`) to swell strongly (outlined in yellow), while the homeostatic counter-force `PPARA` shrinks.
          - In **Cholestasis (AOP 27)**, you see in the lower left how `ABCB11` (BSEP) collapses (from 115 to 18.5 TPM), after which `NR1H4` (FXR) compensatorily represses `CYP7A1` and `SLC10A1`, and on the right the stress/apoptosis cascade (`NFE2L2` $\rightarrow$ `NFKB1` $\rightarrow$ `TP53` $\rightarrow$ `CASP3`) flares up.
        """
    )

    v1_view = mo.vstack([guide_v1, mo.Html(net_svg_str)])
    v1_view
    return (
        GROUP_COLORS,
        GROUP_LABELS,
        NODE_POS,
        active_groups,
        guide_v1,
        net_svg_str,
        render_network_svg,
        v1_view,
    )


@app.cell
def _(genes, group_filter, html_lib, mo):
    def render_expression_svg(filtered_genes_list):
        """Generates a dual-panel SVG for Visualization 2: Absolute log2(TPM+1) and Delta log2FC."""
        W, H = 1160, 450
        svg = [
            f'<svg viewBox="0 0 {W} {H}" width="100%" height="auto" style="background:#0F172A;border-radius:10px;border:1px solid #1E293B;font-family:Inter,system-ui,sans-serif;">',
            f'<text x="22" y="28" fill="#F8FAFC" font-size="16" font-weight="700">Visualization 2: Empirical Expression Profiles — Healthy Homeostasis vs. Steatosis (AOP 34) vs. Cholestasis (AOP 27)</text>',
            f'<text x="22" y="47" fill="#94A3B8" font-size="12">Left: Absolute expression log2(TPM + 1) [GTEx/HPA &amp; GEO] | Right: Pathological shift Δlog2(Fold-Change) vs. Healthy Hepatocyte</text>',
        ]

        n_g = max(len(filtered_genes_list), 1)
        # Panel 1 (Left): x from 55 to 565, y from 80 to 380
        p1_x0, p1_w, p1_y0, p1_h = 52, 505, 78, 295
        max_log2 = 10.0
        svg.append(f'<text x="{p1_x0 + p1_w/2:.1f}" y="68" fill="#38BDF8" font-size="12.5" font-weight="700" text-anchor="middle">Panel A: Absolute mRNA Expression log2(TPM + 1)</text>')
        for y_tick in [0, 2, 4, 6, 8, 10]:
            py = p1_y0 + p1_h - (y_tick / max_log2) * p1_h
            svg.append(f'<line x1="{p1_x0}" y1="{py:.1f}" x2="{p1_x0 + p1_w}" y2="{py:.1f}" stroke="#1E293B" stroke-width="1"/>')
            svg.append(f'<text x="{p1_x0 - 8}" y="{py + 4:.1f}" fill="#94A3B8" font-size="10.5" text-anchor="end">{y_tick}</text>')

        slot_w = p1_w / n_g
        bar_w = max(slot_w * 0.25, 4.0)
        for idx, _g in enumerate(filtered_genes_list):
            cx = p1_x0 + (idx + 0.5) * slot_w
            vals = [
                (_g["expr_healthy_log2"], _g["expr_healthy_tpm"], "#2EC4B6", "Healthy"),
                (_g["expr_steatosis_log2"], _g["expr_steatosis_tpm"], "#FF5964", "Steatosis AOP 34"),
                (_g["expr_cholestasis_log2"], _g["expr_cholestasis_tpm"], "#FF9F1C", "Cholestasis AOP 27"),
            ]
            for b_i, (v_log2, v_tpm, b_col, b_lbl) in enumerate(vals):
                bh = (min(v_log2, max_log2) / max_log2) * p1_h
                bx = cx + (b_i - 1.5) * bar_w
                by = p1_y0 + p1_h - bh
                tip = f"{_g['symbol']} ({b_lbl}): {v_log2:.2f} log2(TPM+1) [{v_tpm:.1f} TPM]"
                svg.append(
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w - 1:.1f}" height="{bh:.1f}" fill="{b_col}" rx="1.5">'
                    f"<title>{html_lib.escape(tip)}</title></rect>"
                )
            svg.append(
                f'<text x="{cx:.1f}" y="{p1_y0 + p1_h + 14:.1f}" fill="#E2E8F0" font-size="9.5" font-weight="600" text-anchor="end" transform="rotate(-40 {cx:.1f} {p1_y0 + p1_h + 14:.1f})">{_g["symbol"]}</text>'
            )

        # Panel 2 (Right): Delta log2FC from -3.5 to +3.5
        p2_x0, p2_w, p2_y0, p2_h = 620, 515, 78, 295
        zero_y = p2_y0 + p2_h / 2.0
        max_fc = 3.2
        svg.append(f'<text x="{p2_x0 + p2_w/2:.1f}" y="68" fill="#38BDF8" font-size="12.5" font-weight="700" text-anchor="middle">Panel B: Pathological Shift Δlog2(Fold-Change vs. Healthy)</text>')
        for fc_tick in [-3, -2, -1, 0, 1, 2, 3]:
            py = zero_y - (fc_tick / max_fc) * (p2_h / 2.0)
            stroke_c = "#64748B" if fc_tick == 0 else "#1E293B"
            sw = "1.5" if fc_tick == 0 else "1"
            svg.append(f'<line x1="{p2_x0}" y1="{py:.1f}" x2="{p2_x0 + p2_w}" y2="{py:.1f}" stroke="{stroke_c}" stroke-width="{sw}"/>')
            svg.append(f'<text x="{p2_x0 - 8}" y="{py + 4:.1f}" fill="#94A3B8" font-size="10.5" text-anchor="end">{fc_tick:+d}</text>')

        slot2_w = p2_w / n_g
        bar2_w = max(slot2_w * 0.34, 5.0)
        for idx, _g in enumerate(filtered_genes_list):
            cx = p2_x0 + (idx + 0.5) * slot2_w
            fcs = [
                (_g["log2fc_steatosis_vs_healthy"], "#FB7185", "ΔSteatosis vs Healthy"),
                (_g["log2fc_cholestasis_vs_healthy"], "#FBBF24", "ΔCholestasis vs Healthy"),
            ]
            for b_i, (fc_v, b_col, b_lbl) in enumerate(fcs):
                bh = (min(abs(fc_v), max_fc) / max_fc) * (p2_h / 2.0)
                bx = cx + (b_i - 1.0) * bar2_w
                by = zero_y - bh if fc_v >= 0 else zero_y
                tip = f"{_g['symbol']} ({b_lbl}): {fc_v:+.2f} log2FC"
                svg.append(
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar2_w - 1.5:.1f}" height="{max(bh, 1.5):.1f}" fill="{b_col}" rx="1.5">'
                    f"<title>{html_lib.escape(tip)}</title></rect>"
                )
            svg.append(
                f'<text x="{cx:.1f}" y="{p2_y0 + p2_h + 14:.1f}" fill="#E2E8F0" font-size="9.5" font-weight="600" text-anchor="end" transform="rotate(-40 {cx:.1f} {p2_y0 + p2_h + 14:.1f})">{_g["symbol"]}</text>'
            )

        # Legend at bottom
        legs = [
            ("#2EC4B6", "Healthy (GTEx/HPA Homeostasis)"),
            ("#FF5964", "Steatosis (GEO NAFLD / AOP 34)"),
            ("#FF9F1C", "Cholestasis (BSEP Inhibition / AOP 27)"),
            ("#FB7185", "Δlog2FC Steatosis vs. Healthy"),
            ("#FBBF24", "Δlog2FC Cholestasis vs. Healthy"),
        ]
        lx = 45
        for lcol, llbl in legs:
            svg.append(f'<rect x="{lx}" y="{H - 22}" width="12" height="12" fill="{lcol}" rx="2"/>')
            svg.append(f'<text x="{lx + 17}" y="{H - 12}" fill="#CBD5E1" font-size="11.5">{html_lib.escape(llbl)}</text>')
            lx += len(llbl) * 6.8 + 42

        svg.append("</svg>")
        return "\n".join(svg)

    active_grps_v2 = set(group_filter.value) if group_filter.value else {_g["group"] for _g in genes}
    filtered_genes_v2 = [_g for _g in genes if _g["group"] in active_grps_v2]
    expr_svg_str = render_expression_svg(filtered_genes_v2)

    guide_v2 = mo.md(
        """
        ### 📊 User Guide Visualization 2 — Healthy vs. Pathological Expression Profile (Homeostasis vs. Steatosis vs. Cholestasis)
        - **How to use this cell:**
          - Use the **🧩 Filter Functional AOP Groups** filter at the top to isolate specific gene groups, and hover your mouse over the bars for the exact TPM and $\log_2\text{Fold-Change}$ values.
        - **What do you see in this chart?**
          - **Panel A (left)** shows the absolute mRNA expression $\log_2(\text{TPM}+1)$ for each of the 18 genes in healthy hepatocytes (green) versus steatosis (red) and cholestasis (orange).
          - **Panel B (right)** shows the relative shift ($\Delta\log_2\text{FC}$) compared to the healthy baseline:
            1. **In Steatosis (pink bars):** Strong positive peaks on the lipogenic genes `NR1H3`, `SREBF1`, `FASN`, `ACACA`, `SCD`, and `CD36` (+1.5 to +2.8 $\log_2\text{FC}$), accompanied by a drop (below the zero line) of the $\beta$-oxidation and VLDL export genes (`PPARA`, `CPT1A`, `MTTP`, `APOB`).
            2. **In Cholestasis (yellow bars):** Strong negative deflection (repression/loss of function) on `ABCB11` (-2.6 $\log_2\text{FC}$), `CYP7A1` (-2.8 $\log_2\text{FC}$), and `SLC10A1` (-2.2 $\log_2\text{FC}$), and strong induction of the four stress/apoptosis genes on the right (`NFE2L2`, `NFKB1`, `TP53`, `CASP3`).
        """
    )

    v2_view = mo.vstack([guide_v2, mo.Html(expr_svg_str)])
    v2_view
    return (
        active_grps_v2,
        expr_svg_str,
        filtered_genes_v2,
        guide_v2,
        render_expression_svg,
        v2_view,
    )


@app.cell
def _(genes, html_lib, matrices, meta, mo, norm_selector, np, state_selector):
    def render_feature_heatmap_svg(selected_state_str, use_zscore_bool):
        """Generates an annotated SVG heatmap of the [18 x 8] Feature Matrix X."""
        if "Steatosis" in selected_state_str and not selected_state_str.startswith("Healthy"):
            raw_mat = np.array(matrices["X_steatosis_raw"], dtype=float)
            z_mat = np.array(matrices["X_steatosis_zscore"], dtype=float)
            state_tag = "Steatosis (AOP 34)"
        elif "Cholestasis" in selected_state_str:
            raw_mat = np.array(matrices["X_cholestasis_raw"], dtype=float)
            z_mat = np.array(matrices["X_cholestasis_zscore"], dtype=float)
            state_tag = "Cholestasis (AOP 27)"
        else:
            raw_mat = np.array(matrices["X_healthy_raw"], dtype=float)
            z_mat = np.array(matrices["X_healthy_zscore"], dtype=float)
            state_tag = "Healthy (Homeostasis Baseline)"

        feat_names = meta["feature_names"]
        feat_descs = [
            "mRNA expression in hepatocytes log2(TPM + 1) [GTEx/HPA Healthy vs. GEO Steatosis/Cholestasis]",
            "Genomic span on the human genome in log10(base pairs) [MyGene.info / Ensembl]",
            "Exon count / canonical transcript structure [MyGene.info / RefSeq]",
            "Canonical protein length (amino acid count / 1000) [MyGene.info / UniProt]",
            "In-degree (number of incoming regulatory arrows) in the causal AOP network",
            "Out-degree (number of outgoing regulatory arrows) in the causal AOP network",
            "Mean experimental STRING-db protein interaction score with connected network genes",
            "Functional role index (0.25=Receptor/TF, 0.50=Enzyme, 0.75=Transporter, 1.00=Stress/Apoptosis)",
        ]
        col_short = [
            "F0: log2(TPM+1)",
            "F1: log10(bp)",
            "F2: Exons",
            "F3: Protein (kAA)",
            "F4: In-Deg",
            "F5: Out-Deg",
            "F6: STRING",
            "F7: Role Index",
        ]

        W, H = 1160, 610
        x0, y0, gw, gh = 210, 82, 880, 495
        cw = gw / 8.0
        rh = gh / 18.0

        def z_to_rgb(z_val):
            # Diverging blue (-2.5) -> dark slate (0) -> red (+2.5) color scale
            v = float(np.clip(z_val / 2.5, -1.0, 1.0))
            if v < 0:
                t = abs(v)
                r = int(30 * (1 - t) + 29 * t)
                g = int(41 * (1 - t) + 78 * t)
                b = int(59 * (1 - t) + 216 * t)
            else:
                t = v
                r = int(30 * (1 - t) + 225 * t)
                g = int(41 * (1 - t) + 29 * t)
                b = int(59 * (1 - t) + 72 * t)
            return f"rgb({r},{g},{b})"

        scale_lbl = "Z-Score (σ vs. Healthy Baseline)" if use_zscore_bool else "Raw Biological Units"
        svg = [
            f'<svg viewBox="0 0 {W} {H}" width="100%" height="auto" style="background:#0F172A;border-radius:10px;border:1px solid #1E293B;font-family:Inter,system-ui,sans-serif;">',
            f'<text x="22" y="28" fill="#F8FAFC" font-size="16" font-weight="700">Visualization 3: Graphormer Input Feature Matrix X [18 genes × 8 features] — {html_lib.escape(state_tag)}</text>',
            f'<text x="22" y="48" fill="#94A3B8" font-size="12">Display: {html_lib.escape(scale_lbl)} | Each row i is the 8D feature vector x_i projected by nn.Linear(8, 64) in Milestone 1.2</text>',
        ]

        for c_i, clbl in enumerate(col_short):
            cx = x0 + (c_i + 0.5) * cw
            svg.append(f'<text x="{cx:.1f}" y="{y0 - 10:.1f}" fill="#38BDF8" font-size="11.5" font-weight="700" text-anchor="middle">{html_lib.escape(clbl)}</text>')

        for r_i, _g in enumerate(genes):
            ry = y0 + r_i * rh
            glbl = f"{_g['idx']:02d}: {_g['symbol']} ({_g['group'].split('_')[0]})"
            svg.append(f'<text x="{x0 - 10}" y="{ry + rh/2 + 4:.1f}" fill="#E2E8F0" font-size="11" font-weight="600" text-anchor="end">{html_lib.escape(glbl)}</text>')

            for c_i in range(8):
                cx = x0 + c_i * cw
                raw_v = raw_mat[r_i, c_i]
                z_v = z_mat[r_i, c_i]
                cell_col = z_to_rgb(z_v)
                cell_txt = f"{z_v:+.2f}σ" if use_zscore_bool else f"{raw_v:.2f}"
                tip = (
                    f"Gene: {_g['symbol']} ({_g['name']})\n"
                    f"Feature [{c_i}] ({feat_names[c_i]}): {feat_descs[c_i]}\n"
                    f"Raw Value ({state_tag}): {raw_v:.3f}\n"
                    f"Z-Score (vs. Healthy): {z_v:+.3f} σ"
                )
                svg.append(
                    f'<g style="cursor:pointer;">'
                    f'<rect x="{cx + 1:.1f}" y="{ry + 1:.1f}" width="{cw - 2:.1f}" height="{rh - 2:.1f}" fill="{cell_col}" rx="3"/>'
                    f'<text x="{cx + cw/2:.1f}" y="{ry + rh/2 + 4:.1f}" fill="#F8FAFC" font-size="10.5" font-weight="600" text-anchor="middle">{cell_txt}</text>'
                    f"<title>{html_lib.escape(tip)}</title></g>"
                )

        svg.append("</svg>")
        return "\n".join(svg)

    feat_svg_str = render_feature_heatmap_svg(
        state_selector.value,
        norm_selector.value.startswith("Z-Score"),
    )

    guide_v3 = mo.md(
        """
        ### 🧮 User Guide Visualization 3 — The `[18 × 8]` Feature Matrix $X$ Heatmap
        - **How to use this cell:**
          - Switch via **📊 Scale for Feature Matrix X [18, 8]** between **Z-Scores ($\sigma$)** and **Raw Biological Values**, and use **🔬 Cellular State on Network** to inspect `X_healthy`, `X_steatosis`, or `X_cholestasis`.
        - **What do you see in this chart?**
          - This is the exact tensor $X \in \mathbb{R}^{18 \times 8}$ saved in `data/hepatocyte_features.pt` for Milestone 1.2:
            - **Column `F0: log2(TPM+1)`** changes dynamically per cellular state (red = strongly upregulated vs. healthy baseline, blue = repressed).
            - **Columns `F1` through `F3`** contain the real gene and protein structure from MyGene.info (`ACACA` is for example the longest gene with 56–59 exons; `APOB` is the largest protein with 4,563 amino acids, `+3.66σ`).
            - **Columns `F4` through `F7`** contain the topological role in the AOP network (in-degree, out-degree, mean STRING-db score, and functional role index).
        """
    )

    v3_view = mo.vstack([guide_v3, mo.Html(feat_svg_str)])
    v3_view
    return feat_svg_str, guide_v3, render_feature_heatmap_svg, v3_view


@app.cell
def _(genes, html_lib, matrices, mo, np):
    def render_spatial_centrality_svg():
        """Generates a dual-panel SVG for Visualization 4: [19x19] Shortest-Path Matrix + Degree Centrality."""
        dist_19 = np.array(matrices["dist_matrix_19_cls"], dtype=int)
        cls_labels = ["00:[CLS]"] + [f"{_g['idx'] + 1:02d}:{_g['symbol']}" for _g in genes]

        W, H = 1160, 600
        svg = [
            f'<svg viewBox="0 0 {W} {H}" width="100%" height="auto" style="background:#0F172A;border-radius:10px;border:1px solid #1E293B;font-family:Inter,system-ui,sans-serif;">',
            f'<text x="22" y="28" fill="#F8FAFC" font-size="16" font-weight="700">Visualization 4: Graphormer Structural Encodings — [19×19] Shortest-Path Matrix D (incl. [CLS]) &amp; Degree Centrality</text>',
            f'<text x="22" y="48" fill="#94A3B8" font-size="12">Left: Floyd-Warshall distance matrix [19×19] for Spatial Attention Bias b_φ(v_i, v_j) | Right: In-Degree &amp; Out-Degree for Centrality Encoding</text>',
        ]

        # Panel 1 (Left): [19 x 19] matrix
        mx0, my0, mw, mh = 92, 95, 550, 475
        cw = mw / 19.0
        ch = mh / 19.0

        def dist_to_col(d):
            palette = {
                0: "#0EA5E9",
                1: "#10B981",
                2: "#22C55E",
                3: "#84CC16",
                4: "#EAB308",
                5: "#F59E0B",
                6: "#F97316",
                7: "#EF4444",
                8: "#DC2626",
                9: "#64748B",
                10: "#7C3AED",
            }
            return palette.get(int(d), "#475569")

        svg.append(f'<text x="{mx0 + mw/2:.1f}" y="72" fill="#38BDF8" font-size="12.5" font-weight="700" text-anchor="middle">Panel A: [19 × 19] Floyd-Warshall Shortest-Path Matrix D</text>')

        for i in range(19):
            cx = mx0 + (i + 0.5) * cw
            cy = my0 + (i + 0.5) * ch
            svg.append(
                f'<text x="{cx:.1f}" y="{my0 - 6:.1f}" fill="#CBD5E1" font-size="8.5" font-weight="600" text-anchor="start" transform="rotate(-45 {cx:.1f} {my0 - 6:.1f})">{cls_labels[i]}</text>'
            )
            svg.append(
                f'<text x="{mx0 - 6:.1f}" y="{cy + 3:.1f}" fill="#CBD5E1" font-size="9" font-weight="600" text-anchor="end">{cls_labels[i]}</text>'
            )
            for j in range(19):
                d_val = int(dist_19[i, j])
                rx = mx0 + j * cw
                ry = my0 + i * ch
                c_fill = dist_to_col(d_val)
                tip = f"From {cls_labels[i]} to {cls_labels[j]}: shortest-path distance = {d_val}"
                svg.append(
                    f'<g style="cursor:pointer;">'
                    f'<rect x="{rx + 0.8:.1f}" y="{ry + 0.8:.1f}" width="{cw - 1.6:.1f}" height="{ch - 1.6:.1f}" fill="{c_fill}" rx="2"/>'
                    f'<text x="{rx + cw/2:.1f}" y="{ry + ch/2 + 3.5:.1f}" fill="#FFFFFF" font-size="9.5" font-weight="700" text-anchor="middle">{d_val}</text>'
                    f"<title>{html_lib.escape(tip)}</title></g>"
                )

        # Panel 2 (Right): In-Degree & Out-Degree per gene
        bx0, by0, bw, bh = 720, 95, 405, 475
        rh = bh / 18.0
        max_deg = 4.0
        svg.append(f'<text x="{bx0 + bw/2:.1f}" y="72" fill="#38BDF8" font-size="12.5" font-weight="700" text-anchor="middle">Panel B: Graphormer Degree Centrality (In vs. Out)</text>')

        for d_tick in [0, 1, 2, 3, 4]:
            tx = bx0 + (d_tick / max_deg) * bw
            svg.append(f'<line x1="{tx:.1f}" y1="{by0}" x2="{tx:.1f}" y2="{by0 + bh}" stroke="#1E293B" stroke-width="1"/>')
            svg.append(f'<text x="{tx:.1f}" y="{by0 + bh + 16:.1f}" fill="#94A3B8" font-size="10.5" text-anchor="middle">{d_tick}</text>')

        for idx, _g in enumerate(genes):
            ry = by0 + idx * rh
            svg.append(f'<text x="{bx0 - 8}" y="{ry + rh/2 + 3.5:.1f}" fill="#E2E8F0" font-size="10" font-weight="600" text-anchor="end">{_g["symbol"]}</text>')
            w_in = (_g["in_degree"] / max_deg) * bw
            w_out = (_g["out_degree"] / max_deg) * bw
            svg.append(
                f'<rect x="{bx0}" y="{ry + 2:.1f}" width="{max(w_in, 2.0):.1f}" height="{rh/2 - 2.5:.1f}" fill="#38BDF8" rx="2">'
                f"<title>{_g['symbol']} In-Degree: {_g['in_degree']}</title></rect>"
            )
            svg.append(
                f'<rect x="{bx0}" y="{ry + rh/2 + 0.5:.1f}" width="{max(w_out, 2.0):.1f}" height="{rh/2 - 2.5:.1f}" fill="#F472B6" rx="2">'
                f"<title>{_g['symbol']} Out-Degree: {_g['out_degree']}</title></rect>"
            )

        svg.append(f'<rect x="{bx0 + 40}" y="{H - 18}" width="12" height="10" fill="#38BDF8" rx="2"/>')
        svg.append(f'<text x="{bx0 + 57}" y="{H - 9}" fill="#CBD5E1" font-size="11">In-Degree (z_deg⁻)</text>')
        svg.append(f'<rect x="{bx0 + 190}" y="{H - 18}" width="12" height="10" fill="#F472B6" rx="2"/>')
        svg.append(f'<text x="{bx0 + 207}" y="{H - 9}" fill="#CBD5E1" font-size="11">Out-Degree (z_deg⁺)</text>')
        svg.append("</svg>")
        return "\n".join(svg)

    sp_svg_str = render_spatial_centrality_svg()

    guide_v4 = mo.md(
        """
        ### 🧭 User Guide Visualization 4 — The `[19 × 19]` Graphormer Shortest-Path Matrix $D$ & Centrality
        - **How to use this cell:**
          - Hover your mouse over a cell `(i, j)` in the left panel to read the exact shortest-path distance $\phi(v_i, v_j)$ between two genes, or inspect on the right how many arrows each gene receives (`In-Degree`) and sends out (`Out-Degree`).
        - **What do you see in this chart?**
          - **Why `[19 × 19]` instead of `[18 × 18]`?** The `HepatocyteGraphormer` (Milestone 1.2) prepends a virtual **`[CLS]` readout node at index `0`** before the 18 genes to summarize the entire graph into $z_0 \in \mathbb{R}^{64}$.
          - **Purple border (`distance = 10`):** Row `0` and column `0` carry the special virtual distance `10` between `[CLS]` and all 18 genes (and `0` at `[0, 0]`), preventing any dimension mismatch (`19 vs 18`) during self-attention in Milestone 1.2.
          - **Right panel:** Immediately shows that `SREBF1` is the strongest hub (`Out-Degree = 3`, `In-Degree = 2`), which receives additional weight via Graphormer Centrality Encoding.
        """
    )

    v4_view = mo.vstack([guide_v4, mo.Html(sp_svg_str)])
    v4_view
    return guide_v4, render_spatial_centrality_svg, sp_svg_str, v4_view


@app.cell
def _(
    GROUP_COLORS,
    Path,
    edges,
    expr_svg_str,
    genes,
    mo,
    pd,
    render_feature_heatmap_svg,
    render_network_svg,
    sp_svg_str,
):
    genes_df = pd.DataFrame(
        [
            {
                "Index": _g["idx"],
                "Gene": _g["symbol"],
                "Group": _g["group"],
                "Role": _g["role_label"],
                "OECD AOP Reference": _g["aop_ref"],
                "Healthy (TPM)": _g["expr_healthy_tpm"],
                "Steatosis (TPM)": _g["expr_steatosis_tpm"],
                "Cholestasis (TPM)": _g["expr_cholestasis_tpm"],
                "Δlog2FC Steatosis": _g["log2fc_steatosis_vs_healthy"],
                "Δlog2FC Cholestasis": _g["log2fc_cholestasis_vs_healthy"],
                "Length (bp)": _g["gene_length_bp"],
                "Exons": _g["exon_count"],
                "Protein (AA)": _g["protein_length_aa"],
                "STRING Score": _g["mean_string_score"],
                "Ensembl ID": _g["ensembl_id"],
                "UniProt": _g["uniprot_id"],
            }
            for _g in genes
        ]
    )

    edges_df = pd.DataFrame(
        [
            {
                "Edge #": e["edge_id"] + 1,
                "Source": f"{e['source_idx']}: {e['source_symbol']}",
                "Target": f"{e['target_idx']}: {e['target_symbol']}",
                "Sign": "+1 (Activation)" if e["sign"] > 0 else "-1 (Repression)",
                "STRING-db Score": e["string_score"],
                "OECD AOP & Mechanistic Citation": e["citation"],
            }
            for e in edges
        ]
    )

    # Build public/01_1_data_explorer.html with pure CSS radio tabs + inline interactive SVGs
    # Guaranteed to work 100% in VS Code HTML Preview (even when JavaScript is blocked!) and in every browser.
    all_grps = set(GROUP_COLORS.keys())
    net_states = [
        ("Steatosis (AOP 34 - LXR/NAFLD)", "Steatosis (AOP 34 - LXR/NAFLD)"),
        ("Healthy (Homeostasis Baseline)", "Healthy (Homeostasis Baseline)"),
        ("Cholestasis (AOP 27 - BSEP Inhibition)", "Cholestasis (AOP 27 - BSEP Inhibition)"),
        ("Δlog2FC: Steatosis vs. Healthy", "Delta: Steatosis vs Healthy (log2FC)"),
        ("Δlog2FC: Cholestasis vs. Healthy", "Delta: Cholestasis vs Healthy (log2FC)"),
    ]
    net_svgs = [render_network_svg(full_st, all_grps, 0.50, True) for _, full_st in net_states]

    feat_states = [
        ("Steatosis (Z-Score vs. Healthy)", "Steatosis (AOP 34 - LXR/NAFLD)", True),
        ("Healthy (Z-Score Baseline)", "Healthy (Homeostasis Baseline)", True),
        ("Cholestasis (Z-Score vs. Healthy)", "Cholestasis (AOP 27 - BSEP Inhibition)", True),
        ("Raw Biological Values (Healthy)", "Healthy (Homeostasis Baseline)", False),
    ]
    feat_svgs = [render_feature_heatmap_svg(f_st, f_z) for _, f_st, f_z in feat_states]

    genes_table_html = genes_df.to_html(index=False, classes="aop-table")
    edges_table_html = edges_df.to_html(index=False, classes="aop-table")

    standalone_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Milestone 1.1 — Hepatocyte AOP & Graphormer Data Explorer</title>
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
      border-radius: 12px;
      padding: 20px 26px;
      margin-bottom: 24px;
    }}
    h1 {{ margin: 0 0 8px 0; font-size: 23px; color: #38BDF8; }}
    .guide-box {{
      background: rgba(30, 41, 59, 0.75);
      border-left: 4px solid #38BDF8;
      border-radius: 8px;
      padding: 12px 18px;
      margin-bottom: 16px;
      font-size: 13.5px;
      color: #E2E8F0;
    }}
    .guide-box h3 {{
      margin: 0 0 6px 0;
      font-size: 15px;
      color: #38BDF8;
    }}
    .guide-box ul {{
      margin: 4px 0 0 18px;
      padding: 0;
    }}
    .guide-box li {{
      margin-bottom: 4px;
    }}
    .card {{
      background-color: #0F172A;
      border: 1px solid #1E293B;
      border-radius: 12px;
      padding: 18px 20px;
      margin-bottom: 26px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }}
    /* Pure CSS Radio Tabs — works even if VS Code HTML Preview disables JavaScript! */
    .tab-group input[type="radio"] {{
      display: none;
    }}
    .tab-labels {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
      align-items: center;
    }}
    .tab-labels span {{
      font-weight: 700;
      color: #94A3B8;
      margin-right: 6px;
      font-size: 13.5px;
    }}
    .tab-labels label {{
      background: #1E293B;
      color: #E2E8F0;
      border: 1px solid #475569;
      border-radius: 8px;
      padding: 7px 14px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      user-select: none;
    }}
    .tab-panes > .pane {{
      display: none;
    }}
    #net_r0:checked ~ .tab-labels label[for="net_r0"],
    #net_r1:checked ~ .tab-labels label[for="net_r1"],
    #net_r2:checked ~ .tab-labels label[for="net_r2"],
    #net_r3:checked ~ .tab-labels label[for="net_r3"],
    #net_r4:checked ~ .tab-labels label[for="net_r4"],
    #feat_r0:checked ~ .tab-labels label[for="feat_r0"],
    #feat_r1:checked ~ .tab-labels label[for="feat_r1"],
    #feat_r2:checked ~ .tab-labels label[for="feat_r2"],
    #feat_r3:checked ~ .tab-labels label[for="feat_r3"] {{
      background: #0284C7;
      border-color: #38BDF8;
      color: #FFFFFF;
    }}
    #net_r0:checked ~ .tab-panes > #net_p0,
    #net_r1:checked ~ .tab-panes > #net_p1,
    #net_r2:checked ~ .tab-panes > #net_p2,
    #net_r3:checked ~ .tab-panes > #net_p3,
    #net_r4:checked ~ .tab-panes > #net_p4,
    #feat_r0:checked ~ .tab-panes > #feat_p0,
    #feat_r1:checked ~ .tab-panes > #feat_p1,
    #feat_r2:checked ~ .tab-panes > #feat_p2,
    #feat_r3:checked ~ .tab-panes > #feat_p3 {{
      display: block;
    }}
    .table-wrap {{
      overflow-x: auto;
      max-height: 460px;
    }}
    table.aop-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
    }}
    table.aop-table th {{
      background: #1E293B;
      color: #38BDF8;
      text-align: left;
      padding: 10px;
      position: sticky;
      top: 0;
      border-bottom: 2px solid #334155;
    }}
    table.aop-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid #1E293B;
      color: #E2E8F0;
    }}
    table.aop-table tr:hover td {{
      background: rgba(56, 189, 248, 0.08);
    }}
  </style>
</head>
<body>
  <div class="header-box">
    <h1>🧬 Milestone 1, Step 1.1: Hepatocyte AOP Network &amp; Real Data Explorer</h1>
    <p style="margin:4px 0;color:#CBD5E1;font-size:14px;"><b>Project:</b> Multimodal &amp; Multiscale AI for Liver Toxicity (Graphormer + Neural ODE) | <b>Research Group Innovative Testing (HU University of Applied Sciences Utrecht)</b></p>
    <p style="margin:4px 0;color:#94A3B8;font-size:13.5px;">This dashboard displays the validated input data from <code>data/hepatocyte_features.pt</code> and <code>data/hepatocyte_baseline_data.json</code> (MyGene.info v3, STRING-db v12.0, GTEx/HPA Healthy &amp; NCBI GEO Steatosis/Cholestasis). <b>Each cell below includes its own dedicated user guide.</b></p>
  </div>

  <!-- CELL 1: NETWORK GRAPH -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 User Guide Cell 1: Causal Hepatocyte AOP Network Graph (Visualization 1)</h3>
      <ul>
        <li><b>How to use:</b> Click the blue buttons below to switch between the 5 cellular states. Hover your mouse over a <b>node (gene)</b> or <b>arrow (interaction)</b> to inspect the exact TPM values, MyGene.info structural data, STRING-db scores, and the official <b>OECD AOP Key Event citation</b>.</li>
        <li><b>What you see happening:</b>
          <ul>
            <li>In <b>Steatosis (AOP 34)</b>, the coral-red lipogenesis genes (<code>NR1H3</code>, <code>SREBF1</code>, <code>FASN</code>, <code>ACACA</code>, <code>SCD</code>, <code>CD36</code>) swell strongly (outlined in yellow: &gt;+0.8 log2FC), while the turquoise homeostasis genes (<code>PPARA</code>, <code>CPT1A</code>, <code>MTTP</code>, <code>APOB</code>) shrink.</li>
            <li>In <b>Cholestasis (AOP 27)</b>, the bile salt export pump <code>ABCB11</code> (BSEP) shrinks dramatically and <code>CYP7A1</code> and <code>SLC10A1</code> are repressed via FXR (<code>NR1H4</code>), while on the right the purple stress and apoptosis cascade (<code>NFE2L2</code> → <code>NFKB1</code> → <code>TP53</code> → <code>CASP3</code>) lights up strongly.</li>
            <li>In the two <b>Δlog2FC modes</b>, upregulated genes turn bright red and repressed genes turn blue, with node size scaling with the magnitude of perturbation.</li>
          </ul>
        </li>
      </ul>
    </div>
    <div class="tab-group">
      <input type="radio" name="net_tab" id="net_r0" checked />
      <input type="radio" name="net_tab" id="net_r1" />
      <input type="radio" name="net_tab" id="net_r2" />
      <input type="radio" name="net_tab" id="net_r3" />
      <input type="radio" name="net_tab" id="net_r4" />
      <div class="tab-labels">
        <span>🔬 Switch Cellular State on Network:</span>
        <label for="net_r0">Steatosis (AOP 34 - LXR/NAFLD)</label>
        <label for="net_r1">Healthy (Homeostasis Baseline)</label>
        <label for="net_r2">Cholestasis (AOP 27 - BSEP Inhibition)</label>
        <label for="net_r3">Δlog2FC: Steatosis vs. Healthy</label>
        <label for="net_r4">Δlog2FC: Cholestasis vs. Healthy</label>
      </div>
      <div class="tab-panes">
        <div class="pane" id="net_p0">{net_svgs[0]}</div>
        <div class="pane" id="net_p1">{net_svgs[1]}</div>
        <div class="pane" id="net_p2">{net_svgs[2]}</div>
        <div class="pane" id="net_p3">{net_svgs[3]}</div>
        <div class="pane" id="net_p4">{net_svgs[4]}</div>
      </div>
    </div>
  </div>

  <!-- CELL 2: EXPRESSION & FOLD-CHANGES -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 User Guide Cell 2: Healthy vs. Pathological Expression Profile (Visualization 2)</h3>
      <ul>
        <li><b>How to use:</b> Compare per gene (on the x-axis) the three colored bars in <b>Panel A (left)</b> for absolute mRNA expression $\\log_2(\\text{{TPM}}+1)$, and inspect in <b>Panel B (right)</b> the relative shift ($\\Delta\\log_2\\text{{Fold-Change}}$) above or below the zero line. Hover over any bar for the exact number.</li>
        <li><b>What you see happening:</b> This chart illustrates why the three attractors in Milestone 1.2 and Milestone 3 ($z_{{\\text{{homeo}}}}$, $z_{{\\text{{steatose}}}}$, $z_{{\\text{{cholestase}}}}$) are sharply distinguishable: Steatosis (pink bars on the right) has an exclusive peak on genes 0 through 5, Cholestasis (yellow bars on the right) has a deep dip on genes 11 through 13, and both pathologies converge on the stress/apoptosis genes 14 through 17.</li>
      </ul>
    </div>
    {expr_svg_str}
  </div>

  <!-- CELL 3: FEATURE MATRIX [18 x 8] -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 User Guide Cell 3: Graphormer Input Feature Matrix X [18 genes × 8 features] (Visualization 3)</h3>
      <ul>
        <li><b>How to use:</b> Click the buttons below to toggle between the Z-score normalized matrices (<code>X_steatosis</code>, <code>X_healthy</code>, <code>X_cholestasis</code>) and the raw biological values (<code>X_healthy_raw</code>). Hover over any cell for the full feature description.</li>
        <li><b>What you see happening:</b> All three states are normalized relative to the <b>same healthy baseline</b> ($\\mu_{{\\text{{healthy}}}}, \\sigma_{{\\text{{healthy}}}}$). When switching between <i>Healthy</i>, <i>Steatosis</i>, and <i>Cholestasis</i>, you see in column <b>F0: log2(TPM+1)</b> the exact Z-score shifts appear that the <code>HepatocyteGraphormer</code> uses in Milestone 1.2 to place the cell into the appropriate Waddington valley.</li>
      </ul>
    </div>
    <div class="tab-group">
      <input type="radio" name="feat_tab" id="feat_r0" checked />
      <input type="radio" name="feat_tab" id="feat_r1" />
      <input type="radio" name="feat_tab" id="feat_r2" />
      <input type="radio" name="feat_tab" id="feat_r3" />
      <div class="tab-labels">
        <span>📊 Switch Feature Matrix X [18 × 8] View:</span>
        <label for="feat_r0">Steatosis (Z-Score vs. Healthy)</label>
        <label for="feat_r1">Healthy (Z-Score Baseline)</label>
        <label for="feat_r2">Cholestasis (Z-Score vs. Healthy)</label>
        <label for="feat_r3">Raw Biological Values (Healthy)</label>
      </div>
      <div class="tab-panes">
        <div class="pane" id="feat_p0">{feat_svgs[0]}</div>
        <div class="pane" id="feat_p1">{feat_svgs[1]}</div>
        <div class="pane" id="feat_p2">{feat_svgs[2]}</div>
        <div class="pane" id="feat_p3">{feat_svgs[3]}</div>
      </div>
    </div>
  </div>

  <!-- CELL 4: [19 x 19] SHORTEST PATH MATRIX & CENTRALITY -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 User Guide Cell 4: [19 × 19] Graphormer Shortest-Path Matrix D &amp; Centrality Encoding (Visualization 4)</h3>
      <ul>
        <li><b>How to use:</b> Inspect in <b>Panel A (left)</b> the shortest-path distance between every pair of nodes in the network (computed via Floyd-Warshall), and in <b>Panel B (right)</b> the number of incoming (blue) and outgoing (pink) AOP interactions per gene.</li>
        <li><b>What you see happening:</b> Row and column <code>00:[CLS]</code> form the virtual <b>Graphormer Readout node</b> with special distance <code>10</code> (purple) to all 18 genes and <code>0</code> to itself. Within the 18 genes (indices 1 through 18), shortest paths range from <code>0</code> (diagonal, blue) and <code>1</code> (directly connected, green) up to <code>8</code> (red, e.g., from bile acid synthesis <code>CYP7A1</code> all the way to lipogenic enzyme <code>ACACA</code>).</li>
      </ul>
    </div>
    {sp_svg_str}
  </div>

  <!-- CELL 5: TABLES -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 User Guide Cell 5: Validated Sources, MyGene.info Annotations &amp; OECD AOP Citations</h3>
      <ul>
        <li><b>What you find here:</b> The complete reference tables containing the Ensembl and UniProt ID, genomic length, exons, protein length, and TPM values per state for each of the 18 genes, followed by the 20 directed interactions with their STRING-db score and OECD AOP reference.</li>
      </ul>
    </div>
    <h3 style="color:#38BDF8;margin-top:4px;">📋 18 Key Genes &amp; Empirical Expression Profiles (MyGene.info, GTEx/HPA &amp; GEO)</h3>
    <div class="table-wrap">{genes_table_html}</div>
    <h3 style="color:#38BDF8;margin-top:22px;">🔗 20 Directed Causal OECD AOP Interactions &amp; STRING-db Scores</h3>
    <div class="table-wrap">{edges_table_html}</div>
  </div>
</body>
</html>
"""
    _public_dir = Path("public")
    _public_dir.mkdir(parents=True, exist_ok=True)
    (_public_dir / "01_1_data_explorer.html").write_text(standalone_html, encoding="utf-8")

    tables_view = mo.vstack(
        [
            mo.md(
                """
                ### 📋 User Guide Cell 5 — Source & OECD AOP Annotation Tables
                - **How to use this cell:** Click the two tabs below to browse through all 18 genes (including Ensembl/UniProt IDs from MyGene.info) and all 20 causal OECD AOP interactions (including STRING-db scores and literature citations).
                """
            ),
            mo.ui.tabs(
                {
                    "18 Key Genes & Expression Profiles": mo.ui.table(genes_df, selection=None),
                    "20 Directed OECD AOP Interactions": mo.ui.table(edges_df, selection=None),
                }
            ),
        ]
    )
    tables_view
    return (
        all_grps,
        edges_df,
        edges_table_html,
        feat_states,
        feat_svgs,
        genes_df,
        genes_table_html,
        net_states,
        net_svgs,
        standalone_html,
        tables_view,
    )


if __name__ == "__main__":
    app.run()
