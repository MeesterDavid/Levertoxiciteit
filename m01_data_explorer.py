import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Mijlpaal 1.1 — Hepatocyt AOP & Graphormer Data Explorer")


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
        # 🧬 Mijlpaal 1, Stap 1.1: Hepatocyt AOP-Netwerk & Echte Data Verkenner
        **Project:** *{meta['project']}* | **Kader:** *{meta['institution']}*

        ### 📘 Algemene Handleiding & Overzicht
        Dit notebook vormt **Mijlpaal 1, Stap 1.1** van de hybride **Graphormer + Neural ODE Levertoxiciteit Simulator**. Voordat we in **Stap 1.2** de `HepatocyteGraphormer` trainen/initiëren om een levercel samen te vatten in een 64-dimensionale latente toestandsvector ($z_0 \\in \\mathbb{{R}}^{{64}}$), verkennen en valideren we hier de **echte biologische invoerdata**:
        1. **De 18 Sleutelgenen (`index 0 t/m 17`):** Verdeeld over vier gekoppelde biologische modules:
           - 🔴 **Groep 1 — `AOP34_Steatosis` (`NR1H3`, `SREBF1`, `FASN`, `ACACA`, `SCD`, `CD36`):** De lipogene cascade van OECD AOP 34 die bij overmatige LXR-activatie leidt tot leververvetting (steatose).
           - 🟢 **Groep 2 — `Homeostasis` (`PPARA`, `CPT1A`, `MTTP`, `APOB`):** De fysiologische tegenkracht uit de *Physiological Map* die overtollige vetzuren verbrandt ($\beta$-oxidatie) of exporteert (VLDL) om de cel in de gezonde attractor ($z_{{\\text{{homeo}}}}$) te houden.
           - 🟠 **Groep 3 — `AOP27_Cholestasis` (`NR1H4`, `ABCB11`, `CYP7A1`, `SLC10A1`):** De galzuurhomeostase en het doelwit van OECD AOP 27 (`ABCB11`/BSEP-remming) die bij blokkade leidt tot toxische galstuwing (cholestase).
           - 🟣 **Groep 4 — `Stress_Apoptosis` (`NFE2L2`, `NFKB1`, `TP53`, `CASP3`):** De convergente schade- en ontstekingsroute die bij het falen van de homeostatische buffers het kantelpunt naar celdood (*Adverse Outcome*) aandrijft.
        2. **Live API Data + Empirische Expressieprofielen:** Gecombineerd uit **MyGene.info v3** (genlengte, exonen, Ensembl/UniProt), **STRING-db v12.0** (150 humane eiwit-eiwit interactiescores), **GTEx/HPA** (gezonde lever-TPM) en **NCBI GEO** (pathologische expressie bij Steatose en Cholestase).
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
        label="🔬 Celtoestand op Netwerk (Knoopgrootte & Kleur):",
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
        label="🧩 Filter Functionele AOP-Groepen:",
    )

    string_slider = mo.ui.slider(
        start=0.0,
        stop=0.95,
        step=0.05,
        value=0.50,
        label="🔗 Min. STRING-db Interactiescore:",
    )

    show_extra_ppi = mo.ui.checkbox(
        value=True,
        label="Toon ook niet-causale STRING-db eiwit-interacties (gestippelde achtergrondlijnen)",
    )

    norm_selector = mo.ui.dropdown(
        options=[
            "Z-Score (Genormaliseerd t.o.v. Gezonde Baseline)",
            "Ruwe Biologische Waarden (log2 TPM, log10 bp, exonen, etc.)",
        ],
        value="Z-Score (Genormaliseerd t.o.v. Gezonde Baseline)",
        label="📊 Schaal voor Feature Matrix X [18, 8]:",
    )

    controls_view = mo.vstack(
        [
            mo.md(
                """
                ### 🎛️ Handleiding Interactieve Filters (Cel 1)
                - **Hoe te gebruiken:** Gebruik de dropdown **Celtoestand op Netwerk** om het biologische netwerk (Visualisatie 1) en de Feature Matrix (Visualisatie 3) direct tussen *Gezond*, *Steatose (AOP 34)*, *Cholestase (AOP 27)* en de *Fold-Change verschuivingen ($\Delta\log_2\text{FC}$)* te schakelen. Met de **STRING-db slider** filter je zwakkere eiwit-interacties weg.
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
        "AOP34_Steatosis": "AOP 34: Lipogenese & Steatose",
        "Homeostasis": "Homeostase: β-Oxidatie & VLDL-Export",
        "AOP27_Cholestasis": "AOP 27: Galzuur & Cholestase",
        "Stress_Apoptosis": "Convergente Stress & Apoptose (AO)",
    }

    def render_network_svg(selected_state_str, active_groups_set, min_str_score, include_bg_ppi):
        """Genereert een haarscherpe, interactieve SVG (met native hover-tooltips) die in elke viewer werkt."""
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
            f'<text x="22" y="28" fill="#F8FAFC" font-size="16" font-weight="700">Visualisatie 1: Causale Hepatocyt AOP-Netwerkgraaf (AOP 34 Steatose + AOP 27 Cholestase + Homeostase)</text>',
            f'<text x="22" y="48" fill="#94A3B8" font-size="12.5">Actieve toestand: {html_lib.escape(selected_state_str)} | Min. STRING-db score: {min_str_score:.2f} | Beweeg je muis over knopen en pijlen voor OECD AOP-citaties</text>',
        ]

        # Kolom-headers
        headers = [
            (0.09, "1. Nucleaire Receptoren (MIE)"),
            (0.32, "2. Transcriptie & Influx (KE)"),
            (0.55, "3. Metabole Enzymen & Export"),
            (0.84, "4. Stress & Apoptose (AO)"),
        ]
        for hx, htxt in headers:
            px, _ = to_px(hx, 1.0)
            svg.append(
                f'<text x="{px:.1f}" y="70" fill="#64748B" font-size="11.5" font-weight="700" text-anchor="middle">{html_lib.escape(htxt)}</text>'
            )

        # 1. Achtergrond STRING-db lijnen
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

        # 2. Causale AOP-randen
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
                f"OECD AOP Citatie: {e['citation']}\n"
                f"STRING-db Score: {e['string_score']:.3f}"
            )
            svg.append(
                f'<g class="aop-edge" style="cursor:pointer;">'
                f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{col}" stroke-width="{lw:.2f}" {dash} marker-end="{marker}" opacity="0.92">'
                f"<title>{html_lib.escape(tip)}</title></line></g>"
            )

        # 3. De 18 Gen-knopen
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
                f"Groep: {GROUP_LABELS[grp_key]} ({_g['role_label']})\n"
                f"OECD AOP: {_g['aop_ref']}\n"
                f"Toelichting: {_g['description']}\n"
                f"---\n"
                f"Gezond (GTEx/HPA): {_g['expr_healthy_tpm']:.1f} TPM (log2={_g['expr_healthy_log2']:.2f})\n"
                f"Steatose (AOP 34): {_g['expr_steatosis_tpm']:.1f} TPM (log2={_g['expr_steatosis_log2']:.2f}, Δ={_g['log2fc_steatosis_vs_healthy']:+.2f})\n"
                f"Cholestase (AOP 27): {_g['expr_cholestasis_tpm']:.1f} TPM (log2={_g['expr_cholestasis_log2']:.2f}, Δ={_g['log2fc_cholestasis_vs_healthy']:+.2f})\n"
                f"MyGene.info: {_g['gene_length_bp']:,} bp | {_g['exon_count']} exonen | {_g['protein_length_aa']} aa | Chr {_g['chr']}\n"
                f"Netwerk: In-deg={_g['in_degree']} | Out-deg={_g['out_degree']} | Mean STRING={_g['mean_string_score']:.3f}"
            )
            bw = 3.2 if (b_col == "#FDE047" or is_delta) else 2.0
            svg.append(
                f'<g style="cursor:pointer;" filter="url(#glow)">'
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{fill_col}" stroke="{b_col}" stroke-width="{bw}"/>'
                f'<text x="{px:.1f}" y="{py + 4:.1f}" fill="#FFFFFF" font-size="11" font-weight="700" text-anchor="middle">{sym}</text>'
                f'<text x="{px:.1f}" y="{py - radius - 6:.1f}" fill="#E2E8F0" font-size="10.5" font-weight="600" text-anchor="middle">{html_lib.escape(sub_lbl)}</text>'
                f"<title>{html_lib.escape(tip)}</title></g>"
            )

        # Legenda onderaan
        leg_items = [
            ("#FF5964", "AOP 34: Steatose/Lipogenese"),
            ("#2EC4B6", "Homeostase: β-Oxidatie & VLDL"),
            ("#FF9F1C", "AOP 27: Cholestase/Galzuur"),
            ("#A855F7", "Stress & Apoptose (AO)"),
            ("#10B981", "Activatie (+1 pijl)"),
            ("#F43F5E", "Repressie / Schade (-1 pijl)"),
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
        ### 🕸️ Handleiding Visualisatie 1 — Causale Hepatocyt AOP-Netwerkgraaf
        - **Hoe gebruik je deze cel?**
          - Kies hierboven bij **🔬 Celtoestand op Netwerk** welke toestand je op de knopen wilt projecteren (*Steatose AOP 34*, *Gezond*, *Cholestase AOP 27* of *$\Delta\log_2\text{FC}$*).
          - **Hover met je muis** over een **gen-knoop** om het exacte Ensembl/UniProt ID, het aantal exonen, de genlengte en de expressie in alle 3 de toestanden te lezen.
          - **Hover met je muis** over een **groene (+1) of rode (-1) pijl** om de **officiële OECD AOP Key Event citatie** en de **STRING-db interactiescore** te lezen.
        - **Wat zie je in deze grafiek?**
          - In **Steatose (AOP 34)** zie je linksboven hoe activatie van `NR1H3` (LXR$\alpha$) via `SREBF1` de lipogene enzymen (`FASN`, `ACACA`, `SCD`) en vetzuur-influx (`CD36`) sterk laat opzwellen (geel omrand), terwijl de homeostatische tegenkracht `PPARA` krimpt.
          - In **Cholestase (AOP 27)** zie je linksonder hoe `ABCB11` (BSEP) instort (van 115 naar 18.5 TPM), waarna `NR1H4` (FXR) compensatoir `CYP7A1` en `SLC10A1` onderdrukt en rechts de stress/apoptose-cascade (`NFE2L2` $\rightarrow$ `NFKB1` $\rightarrow$ `TP53` $\rightarrow$ `CASP3`) opvlamt.
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
        """Genereert een dual-panel SVG voor Visualisatie 2: Absolute log2(TPM+1) én Delta log2FC."""
        W, H = 1160, 450
        svg = [
            f'<svg viewBox="0 0 {W} {H}" width="100%" height="auto" style="background:#0F172A;border-radius:10px;border:1px solid #1E293B;font-family:Inter,system-ui,sans-serif;">',
            f'<text x="22" y="28" fill="#F8FAFC" font-size="16" font-weight="700">Visualisatie 2: Empirische Expressieprofielen — Gezonde Homeostase vs. Steatose (AOP 34) vs. Cholestase (AOP 27)</text>',
            f'<text x="22" y="47" fill="#94A3B8" font-size="12">Links: Absolute expressie log2(TPM + 1) [GTEx/HPA &amp; GEO] | Rechts: Pathologische verschuiving Δlog2(Fold-Change) t.o.v. Gezonde Hepatocyt</text>',
        ]

        n_g = max(len(filtered_genes_list), 1)
        # Paneel 1 (Links): x van 55 tot 565, y van 80 tot 380
        p1_x0, p1_w, p1_y0, p1_h = 52, 505, 78, 295
        max_log2 = 10.0
        svg.append(f'<text x="{p1_x0 + p1_w/2:.1f}" y="68" fill="#38BDF8" font-size="12.5" font-weight="700" text-anchor="middle">Paneel A: Absolute mRNA-Expressie log2(TPM + 1)</text>')
        for y_tick in [0, 2, 4, 6, 8, 10]:
            py = p1_y0 + p1_h - (y_tick / max_log2) * p1_h
            svg.append(f'<line x1="{p1_x0}" y1="{py:.1f}" x2="{p1_x0 + p1_w}" y2="{py:.1f}" stroke="#1E293B" stroke-width="1"/>')
            svg.append(f'<text x="{p1_x0 - 8}" y="{py + 4:.1f}" fill="#94A3B8" font-size="10.5" text-anchor="end">{y_tick}</text>')

        slot_w = p1_w / n_g
        bar_w = max(slot_w * 0.25, 4.0)
        for idx, _g in enumerate(filtered_genes_list):
            cx = p1_x0 + (idx + 0.5) * slot_w
            vals = [
                (_g["expr_healthy_log2"], _g["expr_healthy_tpm"], "#2EC4B6", "Gezond"),
                (_g["expr_steatosis_log2"], _g["expr_steatosis_tpm"], "#FF5964", "Steatose AOP 34"),
                (_g["expr_cholestasis_log2"], _g["expr_cholestasis_tpm"], "#FF9F1C", "Cholestase AOP 27"),
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

        # Paneel 2 (Rechts): Delta log2FC van -3.5 tot +3.5
        p2_x0, p2_w, p2_y0, p2_h = 620, 515, 78, 295
        zero_y = p2_y0 + p2_h / 2.0
        max_fc = 3.2
        svg.append(f'<text x="{p2_x0 + p2_w/2:.1f}" y="68" fill="#38BDF8" font-size="12.5" font-weight="700" text-anchor="middle">Paneel B: Pathologische Verschuiving Δlog2(Fold-Change vs. Gezond)</text>')
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
                (_g["log2fc_steatosis_vs_healthy"], "#FB7185", "ΔSteatose vs Gezond"),
                (_g["log2fc_cholestasis_vs_healthy"], "#FBBF24", "ΔCholestase vs Gezond"),
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

        # Legenda onderaan
        legs = [
            ("#2EC4B6", "Gezond (GTEx/HPA Homeostase)"),
            ("#FF5964", "Steatose (GEO NAFLD / AOP 34)"),
            ("#FF9F1C", "Cholestase (BSEP-remming / AOP 27)"),
            ("#FB7185", "Δlog2FC Steatose vs. Gezond"),
            ("#FBBF24", "Δlog2FC Cholestase vs. Gezond"),
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
        ### 📊 Handleiding Visualisatie 2 — Gezond vs. Pathologisch Expressieprofiel (Homeostase vs. Steatose vs. Cholestase)
        - **Hoe gebruik je deze cel?**
          - Gebruik het filter **🧩 Filter Functionele AOP-Groepen** bovenin om specifieke gen-groepen te isoleren, en beweeg je muis over de staven voor de exacte TPM- en $\log_2\text{Fold-Change}$ waarden.
        - **Wat zie je in deze grafiek?**
          - **Paneel A (links)** toont de absolute mRNA-expressie $\log_2(\text{TPM}+1)$ voor elk van de 18 genen in gezonde hepatocyten (groen) versus steatose (rood) en cholestase (oranje).
          - **Paneel B (rechts)** toont de relatieve verschuiving ($\Delta\log_2\text{FC}$) ten opzichte van de gezonde baseline:
            1. **Bij Steatose (roze staven):** Sterke positieve pieken op de lipogene genen `NR1H3`, `SREBF1`, `FASN`, `ACACA`, `SCD` en `CD36` (+1.5 tot +2.8 $\log_2\text{FC}$), gepaard met een daling (onder de nullijn) van de $\beta$-oxidatie en VLDL-exportgenen (`PPARA`, `CPT1A`, `MTTP`, `APOB`).
            2. **Bij Cholestase (gele staven):** Sterke negatieve uitslag (repressie/functieverlies) op `ABCB11` (-2.6 $\log_2\text{FC}$), `CYP7A1` (-2.8 $\log_2\text{FC}$) en `SLC10A1` (-2.2 $\log_2\text{FC}$), en sterke inductie van de vier stress/apoptose-genen rechts (`NFE2L2`, `NFKB1`, `TP53`, `CASP3`).
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
        """Genereert een geannoteerde SVG heatmap van de [18 x 8] Feature Matrix X."""
        if "Steatosis" in selected_state_str and not selected_state_str.startswith("Healthy"):
            raw_mat = np.array(matrices["X_steatosis_raw"], dtype=float)
            z_mat = np.array(matrices["X_steatosis_zscore"], dtype=float)
            state_tag = "Steatose (AOP 34)"
        elif "Cholestasis" in selected_state_str:
            raw_mat = np.array(matrices["X_cholestasis_raw"], dtype=float)
            z_mat = np.array(matrices["X_cholestasis_zscore"], dtype=float)
            state_tag = "Cholestase (AOP 27)"
        else:
            raw_mat = np.array(matrices["X_healthy_raw"], dtype=float)
            z_mat = np.array(matrices["X_healthy_zscore"], dtype=float)
            state_tag = "Gezond (Homeostase Baseline)"

        feat_names = meta["feature_names"]
        feat_descs = meta["feature_descriptions"]
        col_short = [
            "F0: log2(TPM+1)",
            "F1: log10(bp)",
            "F2: Exonen",
            "F3: Eiwit (kAA)",
            "F4: In-Deg",
            "F5: Out-Deg",
            "F6: STRING",
            "F7: Rol-Index",
        ]

        W, H = 1160, 610
        x0, y0, gw, gh = 210, 82, 880, 495
        cw = gw / 8.0
        rh = gh / 18.0

        def z_to_rgb(z_val):
            # Divergente blauw (-2.5) -> donker leigrijs (0) -> rood (+2.5) kleurenschaal
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

        scale_lbl = "Z-Score (σ t.o.v. Gezonde Baseline)" if use_zscore_bool else "Ruwe Biologische Eenheden"
        svg = [
            f'<svg viewBox="0 0 {W} {H}" width="100%" height="auto" style="background:#0F172A;border-radius:10px;border:1px solid #1E293B;font-family:Inter,system-ui,sans-serif;">',
            f'<text x="22" y="28" fill="#F8FAFC" font-size="16" font-weight="700">Visualisatie 3: Graphormer Input Feature Matrix X [18 genen × 8 features] — {html_lib.escape(state_tag)}</text>',
            f'<text x="22" y="48" fill="#94A3B8" font-size="12">Weergave: {html_lib.escape(scale_lbl)} | Elke rij i is de 8D feature-vector x_i die in Mijlpaal 1.2 door nn.Linear(8, 64) wordt geprojecteerd</text>',
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
                    f"Gen: {_g['symbol']} ({_g['name']})\n"
                    f"Feature [{c_i}] ({feat_names[c_i]}): {feat_descs[c_i]}\n"
                    f"Ruwe Waarde ({state_tag}): {raw_v:.3f}\n"
                    f"Z-Score (t.o.v. Gezond): {z_v:+.3f} σ"
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
        ### 🧮 Handleiding Visualisatie 3 — De `[18 × 8]` Feature Matrix $X$ Heatmap
        - **Hoe gebruik je deze cel?**
          - Schakel via **📊 Schaal voor Feature Matrix X [18, 8]** tussen **Z-Scores ($\sigma$)** en **Ruwe Biologische Waarden**, en gebruik **🔬 Celtoestand op Netwerk** om `X_healthy`, `X_steatosis` of `X_cholestasis` te inspecteren.
        - **Wat zie je in deze grafiek?**
          - Dit is exact de tensor $X \in \mathbb{R}^{18 \times 8}$ die in `data/hepatocyte_features.pt` is opgeslagen voor Mijlpaal 1.2:
            - **Kolom `F0: log2(TPM+1)`** verandert dynamisch per celtoestand (rood = sterk opgereguleerd t.o.v. gezonde baseline, blauw = onderdrukt).
            - **Kolommen `F1` t/m `F3`** bevatten de echte gen- en eiwitstructuur uit MyGene.info (`ACACA` is bijvoorbeeld het langste gen met 59 exonen; `APOB` het grootste eiwit met 4.563 aminozuren, `+3.66σ`).
            - **Kolommen `F4` t/m `F7`** bevatten de topologische rol in het AOP-netwerk (in-degree, out-degree, gemiddelde STRING-db score en functionele rol-index).
        """
    )

    v3_view = mo.vstack([guide_v3, mo.Html(feat_svg_str)])
    v3_view
    return feat_svg_str, guide_v3, render_feature_heatmap_svg, v3_view


@app.cell
def _(genes, html_lib, matrices, mo, np):
    def render_spatial_centrality_svg():
        """Genereert een dual-panel SVG voor Visualisatie 4: [19x19] Shortest-Path Matrix + Degree Centrality."""
        dist_19 = np.array(matrices["dist_matrix_19_cls"], dtype=int)
        cls_labels = ["00:[CLS]"] + [f"{_g['idx'] + 1:02d}:{_g['symbol']}" for _g in genes]

        W, H = 1160, 600
        svg = [
            f'<svg viewBox="0 0 {W} {H}" width="100%" height="auto" style="background:#0F172A;border-radius:10px;border:1px solid #1E293B;font-family:Inter,system-ui,sans-serif;">',
            f'<text x="22" y="28" fill="#F8FAFC" font-size="16" font-weight="700">Visualisatie 4: Graphormer Structurele Encodings — [19×19] Shortest-Path Matrix D (incl. [CLS]) &amp; Degree Centrality</text>',
            f'<text x="22" y="48" fill="#94A3B8" font-size="12">Links: Floyd-Warshall afstandsmatrix [19×19] voor Spatial Attention Bias b_φ(v_i, v_j) | Rechts: In-Degree &amp; Out-Degree voor Centrality Encoding</text>',
        ]

        # Paneel 1 (Links): [19 x 19] matrix
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

        svg.append(f'<text x="{mx0 + mw/2:.1f}" y="72" fill="#38BDF8" font-size="12.5" font-weight="700" text-anchor="middle">Paneel A: [19 × 19] Floyd-Warshall Shortest-Path Matrix D</text>')

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
                tip = f"Van {cls_labels[i]} naar {cls_labels[j]}: kortste-padafstand = {d_val}"
                svg.append(
                    f'<g style="cursor:pointer;">'
                    f'<rect x="{rx + 0.8:.1f}" y="{ry + 0.8:.1f}" width="{cw - 1.6:.1f}" height="{ch - 1.6:.1f}" fill="{c_fill}" rx="2"/>'
                    f'<text x="{rx + cw/2:.1f}" y="{ry + ch/2 + 3.5:.1f}" fill="#FFFFFF" font-size="9.5" font-weight="700" text-anchor="middle">{d_val}</text>'
                    f"<title>{html_lib.escape(tip)}</title></g>"
                )

        # Paneel 2 (Rechts): In-Degree & Out-Degree per gen
        bx0, by0, bw, bh = 720, 95, 405, 475
        rh = bh / 18.0
        max_deg = 4.0
        svg.append(f'<text x="{bx0 + bw/2:.1f}" y="72" fill="#38BDF8" font-size="12.5" font-weight="700" text-anchor="middle">Paneel B: Graphormer Degree Centrality (In vs. Out)</text>')

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
        ### 🧭 Handleiding Visualisatie 4 — De `[19 × 19]` Graphormer Shortest-Path Matrix $D$ & Centrality
        - **Hoe gebruik je deze cel?**
          - Beweeg je muis over een vakje `(i, j)` in het linkerpaneel om de exacte kortste-padafstand $\phi(v_i, v_j)$ tussen twee genen te lezen, of bekijk rechts hoeveel pijlen elk gen ontvangt (`In-Degree`) en uitstuurt (`Out-Degree`).
        - **Wat zie je in deze grafiek?**
          - **Waarom `[19 × 19]` in plaats van `[18 × 18]`?** De `HepatocyteGraphormer` (Mijlpaal 1.2) plaatst een virtuele **`[CLS]`-readout knoop op index `0`** vóór de 18 genen om de hele graaf samen te vatten in $z_0 \in \mathbb{R}^{64}$.
          - **Paarse rand (`afstand = 10`):** Rij `0` en kolom `0` hebben de speciale virtuele afstand `10` tussen `[CLS]` en alle 18 genen (en `0` op `[0, 0]`), zodat de self-attention in Mijlpaal 1.2 geen dimensie-mismatch (`19 vs 18`) krijgt.
          - **Rechterpaneel:** Laat direct zien dat `SREBF1` de sterkste hub is (`Out-Degree = 3`, `In-Degree = 2`), wat door de Graphormer Centrality Encoding extra gewicht krijgt.
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
                "Gen": _g["symbol"],
                "Groep": _g["group"],
                "Rol": _g["role_label"],
                "OECD AOP Referentie": _g["aop_ref"],
                "Gezond (TPM)": _g["expr_healthy_tpm"],
                "Steatose (TPM)": _g["expr_steatosis_tpm"],
                "Cholestase (TPM)": _g["expr_cholestasis_tpm"],
                "Δlog2FC Steatose": _g["log2fc_steatosis_vs_healthy"],
                "Δlog2FC Cholestase": _g["log2fc_cholestasis_vs_healthy"],
                "Lengte (bp)": _g["gene_length_bp"],
                "Exonen": _g["exon_count"],
                "Eiwit (AA)": _g["protein_length_aa"],
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
                "Bron": f"{e['source_idx']}: {e['source_symbol']}",
                "Doel": f"{e['target_idx']}: {e['target_symbol']}",
                "Teken": "+1 (Activatie)" if e["sign"] > 0 else "-1 (Repressie)",
                "STRING-db Score": e["string_score"],
                "OECD AOP & Mechanistische Citatie": e["citation"],
            }
            for e in edges
        ]
    )

    # Bouw 01_1_data_explorer.html met pure CSS radio-tabs + inline interactieve SVG's
    # Werkt 100% gegarandeerd in VS Code HTML Preview (zelfs als JavaScript geblokkeerd is!) én in elke browser.
    all_grps = set(GROUP_COLORS.keys())
    net_states = [
        ("Steatose (AOP 34 - LXR/NAFLD)", "Steatosis (AOP 34 - LXR/NAFLD)"),
        ("Gezond (Homeostase Baseline)", "Healthy (Homeostasis Baseline)"),
        ("Cholestase (AOP 27 - BSEP Remming)", "Cholestasis (AOP 27 - BSEP Inhibition)"),
        ("Δlog2FC: Steatose vs. Gezond", "Delta: Steatosis vs Healthy (log2FC)"),
        ("Δlog2FC: Cholestase vs. Gezond", "Delta: Cholestasis vs Healthy (log2FC)"),
    ]
    net_svgs = [render_network_svg(full_st, all_grps, 0.50, True) for _, full_st in net_states]

    feat_states = [
        ("Steatose (Z-Score t.o.v. Gezond)", "Steatosis (AOP 34 - LXR/NAFLD)", True),
        ("Gezond (Z-Score Baseline)", "Healthy (Homeostasis Baseline)", True),
        ("Cholestase (Z-Score t.o.v. Gezond)", "Cholestasis (AOP 27 - BSEP Inhibition)", True),
        ("Ruwe Biologische Waarden (Gezond)", "Healthy (Homeostasis Baseline)", False),
    ]
    feat_svgs = [render_feature_heatmap_svg(f_st, f_z) for _, f_st, f_z in feat_states]

    genes_table_html = genes_df.to_html(index=False, classes="aop-table")
    edges_table_html = edges_df.to_html(index=False, classes="aop-table")

    standalone_html = f"""<!DOCTYPE html>
<html lang="nl">
<head>
  <meta charset="utf-8" />
  <title>Mijlpaal 1.1 — Hepatocyt AOP & Graphormer Data Explorer</title>
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
    /* Pure CSS Radio Tabs — werkt zelfs als VS Code HTML Preview JavaScript uitschakelt! */
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
    <h1>🧬 Mijlpaal 1, Stap 1.1: Hepatocyt AOP-Netwerk &amp; Echte Data Verkenner</h1>
    <p style="margin:4px 0;color:#CBD5E1;font-size:14px;"><b>Project:</b> Multimodal &amp; Multiscale AI voor Levertoxiciteit (Graphormer + Neural ODE) | <b>Lectoraat Innovative Testing (Hogeschool Utrecht)</b></p>
    <p style="margin:4px 0;color:#94A3B8;font-size:13.5px;">Dit dashboard toont de gevalideerde invoerdata uit <code>data/hepatocyte_features.pt</code> en <code>data/hepatocyte_baseline_data.json</code> (MyGene.info v3, STRING-db v12.0, GTEx/HPA Gezond &amp; NCBI GEO Steatose/Cholestase). <b>Elke cel heeft hieronder een eigen Nederlandstalige handleiding.</b></p>
  </div>

  <!-- CEL 1: NETWERKGRAAF -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 Handleiding Cel 1: Causale Hepatocyt AOP-Netwerkgraaf (Visualisatie 1)</h3>
      <ul>
        <li><b>Hoe te gebruiken:</b> Klik op de blauwe knoppen hieronder om te schakelen tussen de 5 celtoestanden. Beweeg je muis over een <b>knoop (gen)</b> of <b>pijl (interactie)</b> om de exacte TPM-waarden, MyGene.info structuurdata, STRING-db scores en de officiële <b>OECD AOP Key Event citatie</b> te bekijken.</li>
        <li><b>Wat je ziet gebeuren:</b>
          <ul>
            <li>In <b>Steatose (AOP 34)</b> zwellen de koraalrode lipogenese-genen (<code>NR1H3</code>, <code>SREBF1</code>, <code>FASN</code>, <code>ACACA</code>, <code>SCD</code>, <code>CD36</code>) sterk op (geel omrand: &gt;+0.8 log2FC), terwijl de turkooizen homeostase-genen (<code>PPARA</code>, <code>CPT1A</code>, <code>MTTP</code>, <code>APOB</code>) krimpen.</li>
            <li>In <b>Cholestase (AOP 27)</b> krimpt de galzoutpomp <code>ABCB11</code> (BSEP) dramatisch en worden <code>CYP7A1</code> en <code>SLC10A1</code> via FXR (<code>NR1H4</code>) onderdrukt, terwijl rechts de paarse stress- en apoptose-cascade (<code>NFE2L2</code> → <code>NFKB1</code> → <code>TP53</code> → <code>CASP3</code>) sterk oplicht.</li>
            <li>In de twee <b>Δlog2FC modi</b> kleuren opgereguleerde genen felrood en onderdrukte genen blauw, waarbij de knoopgrootte schaalt met de sterkte van de verstoring.</li>
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
        <span>🔬 Schakel Celtoestand op Netwerk:</span>
        <label for="net_r0">Steatose (AOP 34 - LXR/NAFLD)</label>
        <label for="net_r1">Gezond (Homeostase Baseline)</label>
        <label for="net_r2">Cholestase (AOP 27 - BSEP Remming)</label>
        <label for="net_r3">Δlog2FC: Steatose vs. Gezond</label>
        <label for="net_r4">Δlog2FC: Cholestase vs. Gezond</label>
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

  <!-- CEL 2: EXPRESSIE & FOLD-CHANGES -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 Handleiding Cel 2: Gezond vs. Pathologisch Expressieprofiel (Visualisatie 2)</h3>
      <ul>
        <li><b>Hoe te gebruiken:</b> Vergelijk per gen (op de x-as) de drie gekleurde staven in <b>Paneel A (links)</b> voor de absolute mRNA-expressie $\\log_2(\\text{{TPM}}+1)$, en bekijk in <b>Paneel B (rechts)</b> de relatieve verschuiving ($\\Delta\\log_2\\text{{Fold-Change}}$) boven of onder de nullijn. Hover over elke staaf voor het exacte getal.</li>
        <li><b>Wat je ziet gebeuren:</b> Deze grafiek laat zien waarom de drie attractoren in Mijlpaal 1.2 en Mijlpaal 3 ($z_{{\\text{{homeo}}}}$, $z_{{\\text{{steatose}}}}$, $z_{{\\text{{cholestase}}}}$) scherp van elkaar te onderscheiden zijn: Steatose (roze staven rechts) heeft een exclusieve piek op genen 0 t/m 5, Cholestase (gele staven rechts) heeft een diepe dip op genen 11 t/m 13, en beide ziektebeelden convergeren op de stress/apoptose-genen 14 t/m 17.</li>
      </ul>
    </div>
    {expr_svg_str}
  </div>

  <!-- CEL 3: FEATURE MATRIX [18 x 8] -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 Handleiding Cel 3: Graphormer Input Feature Matrix X [18 genen × 8 features] (Visualisatie 3)</h3>
      <ul>
        <li><b>Hoe te gebruiken:</b> Klik op de knoppen hieronder om te wisselen tussen de Z-score genormaliseerde matrices (<code>X_steatosis</code>, <code>X_healthy</code>, <code>X_cholestasis</code>) en de ruwe biologische waarden (<code>X_healthy_raw</code>). Hover over een cel voor de volledige feature-uitleg.</li>
        <li><b>Wat je ziet gebeuren:</b> Alle drie de toestanden zijn genormaliseerd ten opzichte van <b>dezelfde gezonde baseline</b> ($\\mu_{{\\text{{healthy}}}}, \\sigma_{{\\text{{healthy}}}}$). Wanneer je tussen <i>Gezond</i>, <i>Steatose</i> en <i>Cholestase</i> schakelt, zie je in kolom <b>F0: log2(TPM+1)</b> de exacte Z-score verschuivingen verschijnen die de <code>HepatocyteGraphormer</code> in Mijlpaal 1.2 gebruikt om de cel in de juiste Waddington-vallei te plaatsen.</li>
      </ul>
    </div>
    <div class="tab-group">
      <input type="radio" name="feat_tab" id="feat_r0" checked />
      <input type="radio" name="feat_tab" id="feat_r1" />
      <input type="radio" name="feat_tab" id="feat_r2" />
      <input type="radio" name="feat_tab" id="feat_r3" />
      <div class="tab-labels">
        <span>📊 Schakel Feature Matrix X [18 × 8] Weergave:</span>
        <label for="feat_r0">Steatose (Z-Score t.o.v. Gezond)</label>
        <label for="feat_r1">Gezond (Z-Score Baseline)</label>
        <label for="feat_r2">Cholestase (Z-Score t.o.v. Gezond)</label>
        <label for="feat_r3">Ruwe Biologische Waarden (Gezond)</label>
      </div>
      <div class="tab-panes">
        <div class="pane" id="feat_p0">{feat_svgs[0]}</div>
        <div class="pane" id="feat_p1">{feat_svgs[1]}</div>
        <div class="pane" id="feat_p2">{feat_svgs[2]}</div>
        <div class="pane" id="feat_p3">{feat_svgs[3]}</div>
      </div>
    </div>
  </div>

  <!-- CEL 4: [19 x 19] SHORTEST PATH MATRIX & CENTRALITY -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 Handleiding Cel 4: [19 × 19] Graphormer Shortest-Path Matrix D &amp; Centrality Encoding (Visualisatie 4)</h3>
      <ul>
        <li><b>Hoe te gebruiken:</b> Bekijk in <b>Paneel A (links)</b> de kortste-padafstand tussen elk paar knopen in het netwerk (berekend via Floyd-Warshall), en in <b>Paneel B (rechts)</b> het aantal inkomende (blauw) en uitgaande (roze) AOP-interacties per gen.</li>
        <li><b>Wat je ziet gebeuren:</b> Rij en kolom <code>00:[CLS]</code> vormen de virtuele <b>Graphormer Readout-knoop</b> met speciale afstand <code>10</code> (paars) naar alle 18 genen en <code>0</code> naar zichzelf. Binnen de 18 genen (indices 1 t/m 18) lopen de kortste paden van <code>0</code> (diagonaal, blauw) en <code>1</code> (direct verbonden, groen) tot <code>8</code> (rood, bijv. van galzuursynthese <code>CYP7A1</code> helemaal naar lipogenese-enzym <code>ACACA</code>).</li>
      </ul>
    </div>
    {sp_svg_str}
  </div>

  <!-- CEL 5: TABELLEN -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 Handleiding Cel 5: Gevalideerde Bronnen, MyGene.info Annotaties &amp; OECD AOP Citaties</h3>
      <ul>
        <li><b>Wat je hier vindt:</b> De volledige referentietabellen met voor elk van de 18 genen het Ensembl- en UniProt-ID, genomische lengte, exonen, eiwitlengte en TPM-waarden per toestand, gevolgd door de 20 gerichte interacties met hun STRING-db score en OECD AOP-verwijzing.</li>
      </ul>
    </div>
    <h3 style="color:#38BDF8;margin-top:4px;">📋 18 Sleutelgenen &amp; Empirische Expressieprofielen (MyGene.info, GTEx/HPA &amp; GEO)</h3>
    <div class="table-wrap">{genes_table_html}</div>
    <h3 style="color:#38BDF8;margin-top:22px;">🔗 20 Gerichte Causale OECD AOP-Interacties &amp; STRING-db Scores</h3>
    <div class="table-wrap">{edges_table_html}</div>
  </div>
</body>
</html>
"""
    Path("01_1_data_explorer.html").write_text(standalone_html, encoding="utf-8")

    tables_view = mo.vstack(
        [
            mo.md(
                """
                ### 📋 Handleiding Cel 5 — Bron- & OECD AOP-Annotatietabellen
                - **Hoe gebruik je deze cel?** Klik op de twee tabbladen hieronder om door alle 18 genen (inclusief Ensembl/UniProt ID's uit MyGene.info) en alle 20 causale OECD AOP-interacties (inclusief STRING-db scores en literatuurcitaten) te bladeren.
                """
            ),
            mo.ui.tabs(
                {
                    "18 Sleutelgenen & Expressieprofielen": mo.ui.table(genes_df, selection=None),
                    "20 Gerichte OECD AOP-Interacties": mo.ui.table(edges_df, selection=None),
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
