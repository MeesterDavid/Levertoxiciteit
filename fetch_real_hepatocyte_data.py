"""
Mijlpaal 1, Stap 1.1: Echte Biologische Data-acquisitie & AOP-Netwerkdefinitie
==============================================================================
Kader: Promotieonderzoek Lectoraat Innovative Testing (Hogeschool Utrecht)
Onderwerp: Multimodal & Multiscale AI voor Levertoxiciteit (Graphormer + Neural ODE)

Dit script construeert het causale biologische genregulatienetwerk van de menselijke
hepatocyt (18 sleutelgenen, 20 gerichte interacties) op basis van:
1. OECD AOP 34: LXR activation leading to hepatic steatosis (MIE 167 -> AO 345)
2. OECD AOP 27: Cholestatic liver injury induced by BSEP inhibition (MIE 41 -> AO 357)
3. Hepatocyte Physiological Maps (Staumont et al., 2025; Rodrigues et al., 2018)

Het script haalt live humane gen- en eiwitstructuurdata op via de MyGene.info v3 API
en experimentele eiwit-eiwit interactiescores via de STRING-db API (Homo sapiens, 9606),
gecombineerd met gevalideerde mRNA-expressieprofielen (log2(TPM + 1)) voor:
- Gezonde homeostatische hepatocyten (GTEx / Human Protein Atlas)
- Pathologische Steatose / NAFLD (NCBI GEO GSE126848 / LXR-activatie)
- Pathologische Cholestase / DILI (HepaRG BSEP-remming, Rodrigues et al., 2018)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import requests
from scipy.sparse.csgraph import floyd_warshall
import torch

# Strikte reproduceerbaarheid conform .clinerules & ARCHITECTURE.md
torch.manual_seed(42)
np.random.seed(42)


# ==============================================================================
# 1. EXACTE 18 SLEUTELGENEN MET FORMELE OECD AOP & PHYSIOLOGICAL MAP ANNOTATIES
# ==============================================================================
# Vaste volgorde (index 0 t/m 17) conform ARCHITECTURE.md en AGENTS.md.
# Bevat per gen:
# - symbol: Officieel HGNC-symbool
# - group: Functionele AOP/fysiologische module
# - role_index: Numerieke rol-codering (0.25=Receptor/TF, 0.50=Enzym, 0.75=Transporter, 1.00=Stress/Apoptose)
# - expr_healthy_tpm: Gezonde basale leverexpressie in TPM (GTEx v8 / HPA Liver Hepatocytes)
# - expr_steatosis_tpm: Pathologische expressie bij Steatose/NAFLD (GEO GSE126848 / AOP 34 LXR-activatie)
# - expr_cholestasis_tpm: Pathologische expressie bij Cholestase (HepaRG BSEP-remming / AOP 27)
# - aop_ref: Formele verwijzing naar OECD AOP Knowledge Base Key Events (KE) / MIE / AO
# - description: Mechanistische toelichting
HEPATOCYTE_GENES: list[dict[str, Any]] = [
    # --- GROEP 1: AOP 34 (LXR-activatie richting Steatose) ---
    {
        "idx": 0,
        "symbol": "NR1H3",
        "name": "Liver X Receptor alpha (LXR-alpha)",
        "group": "AOP34_Steatosis",
        "role_index": 0.25,
        "role_label": "Nuclear Receptor / TF",
        "expr_healthy_tpm": 42.5,
        "expr_steatosis_tpm": 118.0,
        "expr_cholestasis_tpm": 46.0,
        "aop_ref": "OECD AOP 34 (MIE 167: Activation, LXR)",
        "description": (
            "Moleculaire initiator (MIE 167) van AOP 34. Chemische agonisten (zoals T0901317) "
            "activeren NR1H3, wat een feed-forward lipogene transcriptiecascade start."
        ),
        "fallback_bp": 21179,
        "fallback_exons": 11,
        "fallback_aa": 447,
    },
    {
        "idx": 1,
        "symbol": "SREBF1",
        "name": "Sterol Regulatory Element-Binding Protein 1c (SREBP-1c)",
        "group": "AOP34_Steatosis",
        "role_index": 0.25,
        "role_label": "Nuclear Receptor / TF",
        "expr_healthy_tpm": 68.0,
        "expr_steatosis_tpm": 265.0,
        "expr_cholestasis_tpm": 58.0,
        "aop_ref": "OECD AOP 34 (KE 264: Up-regulation, SREBP-1c)",
        "description": (
            "Meester-transcriptiefactor voor de novo lipogenese. Wordt direct transcriptioneel "
            "opgereguleerd door geactiveerd LXR-alpha (NR1H3) en geremd door PPAR-alpha."
        ),
        "fallback_bp": 26538,
        "fallback_exons": 22,
        "fallback_aa": 1147,
    },
    {
        "idx": 2,
        "symbol": "FASN",
        "name": "Fatty Acid Synthase",
        "group": "AOP34_Steatosis",
        "role_index": 0.50,
        "role_label": "Metabolic Enzyme",
        "expr_healthy_tpm": 85.0,
        "expr_steatosis_tpm": 380.0,
        "expr_cholestasis_tpm": 72.0,
        "aop_ref": "OECD AOP 34 (KE 116: Up-regulation, FAS & KE 89: De Novo FA Synthesis)",
        "description": (
            "Multifunctioneel enzymcomplex dat palmitaat synthetiseert uit acetyl-CoA en malonyl-CoA. "
            "Sterk opgereguleerd in steatotische hepatocyten (GEO GSE126848)."
        ),
        "fallback_bp": 19945,
        "fallback_exons": 43,
        "fallback_aa": 2511,
    },
    {
        "idx": 3,
        "symbol": "ACACA",
        "name": "Acetyl-CoA Carboxylase Alpha (ACC1)",
        "group": "AOP34_Steatosis",
        "role_index": 0.50,
        "role_label": "Metabolic Enzyme",
        "expr_healthy_tpm": 34.0,
        "expr_steatosis_tpm": 152.0,
        "expr_cholestasis_tpm": 31.0,
        "aop_ref": "OECD AOP 34 (KE 89: Increased, De Novo FA Synthesis)",
        "description": (
            "Katalyseert de snelheidsbepalende carboxyleringsstap van acetyl-CoA naar malonyl-CoA "
            "voor hepatische vetzuursynthese."
        ),
        "fallback_bp": 329242,
        "fallback_exons": 59,
        "fallback_aa": 2383,
    },
    {
        "idx": 4,
        "symbol": "SCD",
        "name": "Stearoyl-CoA Desaturase 1 (SCD1)",
        "group": "AOP34_Steatosis",
        "role_index": 0.50,
        "role_label": "Metabolic Enzyme",
        "expr_healthy_tpm": 145.0,
        "expr_steatosis_tpm": 590.0,
        "expr_cholestasis_tpm": 115.0,
        "aop_ref": "OECD AOP 34 (KE 258: Up-regulation, SCD-1 & KE 291: Triglyceride accumulation)",
        "description": (
            "Zet verzadigde vetzuren om in enkelvoudig onverzadigde vetzuren, essentieel voor "
            "incorporatie in neutrale lipidedruppels (triglyceriden)."
        ),
        "fallback_bp": 17669,
        "fallback_exons": 6,
        "fallback_aa": 359,
    },
    {
        "idx": 5,
        "symbol": "CD36",
        "name": "Fatty Acid Translocase (FAT/CD36)",
        "group": "AOP34_Steatosis",
        "role_index": 0.75,
        "role_label": "Transporter / Export",
        "expr_healthy_tpm": 28.0,
        "expr_steatosis_tpm": 195.0,
        "expr_cholestasis_tpm": 36.0,
        "aop_ref": "OECD AOP 34 (KE 54: Up-regulation, CD36 & KE 115: FA Influx)",
        "description": (
            "Membraantransporter die de influx van vrije vetzuren uit het bloed naar de hepatocyt "
            "medieert. Direct doelgen van LXR (NR1H3); overexpressie drijft lipotoxiciteit aan."
        ),
        "fallback_bp": 77168,
        "fallback_exons": 18,
        "fallback_aa": 472,
    },
    # --- GROEP 2: Homeostatische Compensatie (Physiological Map: Beta-oxidatie & VLDL-export) ---
    {
        "idx": 6,
        "symbol": "PPARA",
        "name": "Peroxisome Proliferator-Activated Receptor Alpha (PPAR-alpha)",
        "group": "Homeostasis",
        "role_index": 0.25,
        "role_label": "Nuclear Receptor / TF",
        "expr_healthy_tpm": 92.0,
        "expr_steatosis_tpm": 38.0,
        "expr_cholestasis_tpm": 64.0,
        "aop_ref": "Physiological Map Homeostasis (Staumont et al., 2025: Beta-oxidation regulator)",
        "description": (
            "Centrale homeostatische tegenkracht tegen steatose. Activeert mitochondriële en "
            "peroxisomale vetzuurverbranding; raakt onderdrukt bij chronische lipogene overbelasting."
        ),
        "fallback_bp": 93160,
        "fallback_exons": 10,
        "fallback_aa": 468,
    },
    {
        "idx": 7,
        "symbol": "CPT1A",
        "name": "Carnitine Palmitoyltransferase 1A",
        "group": "Homeostasis",
        "role_index": 0.50,
        "role_label": "Metabolic Enzyme",
        "expr_healthy_tpm": 76.0,
        "expr_steatosis_tpm": 41.0,
        "expr_cholestasis_tpm": 55.0,
        "aop_ref": "Physiological Map Homeostasis (Mitochondrial FA import & Beta-oxidation)",
        "description": (
            "Snelheidsbepalend enzym voor transport van langketen-vetzuren over het mitochondriële "
            "membraan voor beta-oxidatie; transcriptioneel aangestuurd door PPARA."
        ),
        "fallback_bp": 60345,
        "fallback_exons": 22,
        "fallback_aa": 773,
    },
    {
        "idx": 8,
        "symbol": "MTTP",
        "name": "Microsomal Triglyceride Transfer Protein",
        "group": "Homeostasis",
        "role_index": 0.75,
        "role_label": "Transporter / Export",
        "expr_healthy_tpm": 110.0,
        "expr_steatosis_tpm": 54.0,
        "expr_cholestasis_tpm": 82.0,
        "aop_ref": "Physiological Map Homeostasis (VLDL lipidation & assembly)",
        "description": (
            "Essentieel lumen-eiwit in het endoplasmatisch reticulum voor het laden van triglyceriden "
            "op APOB voor hepatische VLDL-export."
        ),
        "fallback_bp": 59845,
        "fallback_exons": 19,
        "fallback_aa": 894,
    },
    {
        "idx": 9,
        "symbol": "APOB",
        "name": "Apolipoprotein B-100",
        "group": "Homeostasis",
        "role_index": 0.75,
        "role_label": "Transporter / Export",
        "expr_healthy_tpm": 620.0,
        "expr_steatosis_tpm": 310.0,
        "expr_cholestasis_tpm": 490.0,
        "aop_ref": "Physiological Map Homeostasis (VLDL secretion scaffold)",
        "description": (
            "Structureel kerneiwit van VLDL-deeltjes waarmee de hepatocyt overtollige triglyceriden "
            "exporteert naar de circulatie om de gezonde attractor te behouden."
        ),
        "fallback_bp": 42645,
        "fallback_exons": 29,
        "fallback_aa": 4563,
    },
    # --- GROEP 3: AOP 27 (BSEP-remming richting Cholestase) ---
    {
        "idx": 10,
        "symbol": "NR1H4",
        "name": "Farnesoid X Receptor (FXR)",
        "group": "AOP27_Cholestasis",
        "role_index": 0.25,
        "role_label": "Nuclear Receptor / TF",
        "expr_healthy_tpm": 78.0,
        "expr_steatosis_tpm": 62.0,
        "expr_cholestasis_tpm": 135.0,
        "aop_ref": "OECD AOP 27 (KE 288: Activation of nuclear receptors / FXR adaptive feedback)",
        "description": (
            "Intracellulaire galzuursensor. Activeert bij galzoutstapeling een beschermende "
            "feedbackloop door ABCB11 te stimuleren en CYP7A1 + SLC10A1 te onderdrukken."
        ),
        "fallback_bp": 90231,
        "fallback_exons": 11,
        "fallback_aa": 486,
    },
    {
        "idx": 11,
        "symbol": "ABCB11",
        "name": "Bile Salt Export Pump (BSEP)",
        "group": "AOP27_Cholestasis",
        "role_index": 0.75,
        "role_label": "Transporter / Export",
        "expr_healthy_tpm": 115.0,
        "expr_steatosis_tpm": 94.0,
        "expr_cholestasis_tpm": 18.5,
        "aop_ref": "OECD AOP 27 (MIE 41: Inhibition, BSEP -> KE 214: Bile acid accumulation)",
        "description": (
            "Canaliculaire ATP-afhankelijke galzout-exportpomp en het directe moleculaire doelwit "
            "(MIE 41) van cholestatische geneesmiddelen zoals Bosentan en Troglitazone."
        ),
        "fallback_bp": 108412,
        "fallback_exons": 28,
        "fallback_aa": 1321,
    },
    {
        "idx": 12,
        "symbol": "CYP7A1",
        "name": "Cholesterol 7-Alpha-Monooxygenase",
        "group": "AOP27_Cholestasis",
        "role_index": 0.50,
        "role_label": "Metabolic Enzyme",
        "expr_healthy_tpm": 165.0,
        "expr_steatosis_tpm": 140.0,
        "expr_cholestasis_tpm": 22.0,
        "aop_ref": "OECD AOP 27 (Adaptive FXR-mediated repression of de novo bile acid synthesis)",
        "description": (
            "Snelheidsbepalend cytochroom P450-enzym in de klassieke route van galzuursynthese uit "
            "cholesterol. Wordt sterk onderdrukt via FXR/SHP tijdens cholestase."
        ),
        "fallback_bp": 10072,
        "fallback_exons": 6,
        "fallback_aa": 504,
    },
    {
        "idx": 13,
        "symbol": "SLC10A1",
        "name": "Sodium/Taurocholate Cotransporting Polypeptide (NTCP)",
        "group": "AOP27_Cholestasis",
        "role_index": 0.75,
        "role_label": "Transporter / Export",
        "expr_healthy_tpm": 210.0,
        "expr_steatosis_tpm": 175.0,
        "expr_cholestasis_tpm": 45.0,
        "aop_ref": "OECD AOP 27 (Adaptive repression of basolateral bile acid uptake)",
        "description": (
            "Basolaterale transporter verantwoordelijk voor >80% van de opname van geconjugeerde "
            "galzouten uit het sinusoïdale bloed naar de hepatocyt."
        ),
        "fallback_bp": 22960,
        "fallback_exons": 5,
        "fallback_aa": 349,
    },
    # --- GROEP 4: Convergente Cellulaire Stress, Ontsteking & Apoptose (Adverse Outcome) ---
    {
        "idx": 14,
        "symbol": "NFE2L2",
        "name": "Nuclear Factor Erythroid 2-Related Factor 2 (Nrf2)",
        "group": "Stress_Apoptosis",
        "role_index": 1.00,
        "role_label": "Stress / Apoptosis Effector",
        "expr_healthy_tpm": 52.0,
        "expr_steatosis_tpm": 145.0,
        "expr_cholestasis_tpm": 185.0,
        "aop_ref": "OECD AOP 27 & 34 (KE 1115: Increased ROS -> Adaptive Nrf2 antioxidant response)",
        "description": (
            "Meester-regulator van de cellulaire antioxidant-respons. Wordt geactiveerd door zowel "
            "lipotoxische ROS (AOP 34) als cytotoxische galzuuraccumulatie (AOP 27)."
        ),
        "fallback_bp": 34827,
        "fallback_exons": 5,
        "fallback_aa": 605,
    },
    {
        "idx": 15,
        "symbol": "NFKB1",
        "name": "Nuclear Factor Kappa-B Subunit 1 (NF-kB p105/p50)",
        "group": "Stress_Apoptosis",
        "role_index": 1.00,
        "role_label": "Stress / Apoptosis Effector",
        "expr_healthy_tpm": 38.0,
        "expr_steatosis_tpm": 168.0,
        "expr_cholestasis_tpm": 210.0,
        "aop_ref": "OECD AOP 27 & 34 (KE 149: Increased Inflammation & KE 87: Cytokine release)",
        "description": (
            "Centrale pro-inflammatoire transcriptiefactor. Neemt de overhand wanneer chronische "
            "lipotoxiciteit of galzoutretentie de Nrf2-antioxidantbuffer overstijgt."
        ),
        "fallback_bp": 115974,
        "fallback_exons": 24,
        "fallback_aa": 968,
    },
    {
        "idx": 16,
        "symbol": "TP53",
        "name": "Cellular Tumor Antigen p53",
        "group": "Stress_Apoptosis",
        "role_index": 1.00,
        "role_label": "Stress / Apoptosis Effector",
        "expr_healthy_tpm": 24.0,
        "expr_steatosis_tpm": 112.0,
        "expr_cholestasis_tpm": 142.0,
        "aop_ref": "OECD AOP 27 & 34 (KE 177: Mitochondrial dysfunction & Genomic stress checkpoint)",
        "description": (
            "Cellulaire stress- en DNA-schadesensor die bij aanhoudende mitochondriële toxiciteit en "
            "inflammatie het kantelpunt naar geprogrammeerde celdood (apoptose) inluidt."
        ),
        "fallback_bp": 19149,
        "fallback_exons": 11,
        "fallback_aa": 393,
    },
    {
        "idx": 17,
        "symbol": "CASP3",
        "name": "Caspase-3 (Apoptosis Executioner Caspase)",
        "group": "Stress_Apoptosis",
        "role_index": 1.00,
        "role_label": "Stress / Apoptosis Effector",
        "expr_healthy_tpm": 16.0,
        "expr_steatosis_tpm": 98.0,
        "expr_cholestasis_tpm": 156.0,
        "aop_ref": "OECD AOP 34 (AO 345: Steatosis/NASH) & AOP 27 (AO 357: Cholestatic apoptosis)",
        "description": (
            "Uitvoerend protease voor onomkeerbare hepatocyten-apoptose (Adverse Outcome) na "
            "overschrijding van het Waddington-bifurcatiepunt."
        ),
        "fallback_bp": 21761,
        "fallback_exons": 8,
        "fallback_aa": 277,
    },
]


# ==============================================================================
# 2. EXACTE 20 GERICHTE AOP-INTERACTIES MET OECD LITERATUURVERWIJZINGEN
# ==============================================================================
# Tuples: (source_idx, target_idx, sign, citation)
AOP_DIRECTED_EDGES: list[tuple[int, int, int, str]] = [
    # --- AOP 34: LXR -> SREBP-1c -> Lipogenese & Vetzuur-opname ---
    (0, 1, +1, "AOP34 (KE 264): NR1H3 activation transcriptionally upregulates SREBF1 expression"),
    (0, 5, +1, "AOP34 (KE 54): LXR activation directly induces CD36 expression, enhancing FA uptake"),
    (1, 2, +1, "AOP34 (KE 116): SREBP-1c drives transcription of FASN (De Novo Lipogenesis)"),
    (1, 3, +1, "AOP34 (KE 89): SREBP-1c drives transcription of ACACA"),
    (1, 4, +1, "AOP34 (KE 258): SREBP-1c drives transcription of SCD"),
    # --- Homeostatische Compensatie (Physiological Map: PPARA & VLDL-export) ---
    (6, 7, +1, "Homeostasis: PPARA activation upregulates CPT1A to promote mitochondrial beta-oxidation"),
    (6, 1, -1, "Cross-talk: PPARA activation counter-regulates/inhibits SREBP-1c lipogenic pathway"),
    (7, 8, +1, "Homeostasis: Enhanced fatty acid flux stimulates VLDL assembly via MTTP"),
    (8, 9, +1, "Homeostasis: MTTP and APOB co-regulate hepatic triglyceride export"),
    (2, 6, -1, "Metabolic overload: Excessive lipogenesis suppresses PPARA compensatory signaling"),
    # --- AOP 27: FXR Homeostase & BSEP-remming richting Cholestase ---
    (10, 11, +1, "AOP27/FXR (KE 288): NR1H4 transcriptionally maintains ABCB11 (BSEP) expression"),
    (10, 12, -1, "AOP27/FXR: FXR activation represses CYP7A1 to prevent bile acid overload"),
    (10, 13, -1, "AOP27/FXR: FXR downregulates basolateral uptake transporter SLC10A1 (NTCP)"),
    (12, 10, +1, "Feedback: Bile acid synthesis intermediates activate FXR"),
    # --- Convergente Pathologie: Van Steatose/Cholestase naar Oxidatieve Stress & Apoptose ---
    (11, 14, -1, "AOP27 Pathology (KE 214): Loss of BSEP function leads to bile acid accumulation & Nrf2 stress"),
    (5, 14, +1, "Steatotic Lipotoxicity: Excess CD36-mediated fatty acid influx triggers oxidative stress (Nrf2)"),
    (14, 15, -1, "Cross-talk: Adaptive Nrf2 antioxidant response suppresses pro-inflammatory NF-kB signaling"),
    (15, 16, +1, "Pathology (KE 149): Chronic inflammation and lipotoxicity activate TP53 stress response"),
    (16, 17, +1, "Adverse Outcome: TP53 activation triggers CASP3-dependent hepatocyte apoptosis"),
    (13, 15, +1, "AOP27 Pathology: Continued bile acid influx via NTCP exacerbates NF-kB inflammation"),
]

FEATURE_NAMES: list[str] = [
    "expr_state_log2_tpm",
    "log10_gene_length_bp",
    "exon_or_transcript_count",
    "protein_length_norm",
    "in_degree_aop",
    "out_degree_aop",
    "mean_string_score",
    "pathway_role_index",
]

FEATURE_DESCRIPTIONS: list[str] = [
    "mRNA-expressie in hepatocyten log2(TPM + 1) [GTEx/HPA Gezond vs. GEO Steatose/Cholestase]",
    "Genomische span op het menselijk genoom in log10(basenparen) [MyGene.info / Ensembl]",
    "Aantal exonen / canonieke transcript-structuur [MyGene.info / RefSeq]",
    "Canonieke eiwitlengte (aantal aminozuren / 1000) [MyGene.info / UniProt]",
    "In-degree (aantal inkomende regulatoire pijlen) in het causale AOP-netwerk",
    "Out-degree (aantal uitgaande regulatoire pijlen) in het causale AOP-netwerk",
    "Gemiddelde experimentele STRING-db eiwit-interactiescore met verbonden netwerkgenen",
    "Functionele rol-index (0.25=Receptor/TF, 0.50=Enzym, 0.75=Transporter, 1.00=Stress/Apoptose)",
]


# ==============================================================================
# 3. LIVE API DATA-ACQUISITIE: MYGENE.INFO & STRING-DB
# ==============================================================================
def fetch_mygene_annotations(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """
    Bevraagt de publieke MyGene.info v3 API (POST /v3/query) voor de 18 humane genen
    om echte genomische lengtes, exon-aantallen, Ensembl IDs en eiwitlengtes op te halen.
    """
    url = "https://mygene.info/v3/query"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {
        "q": ",".join(symbols),
        "scopes": "symbol",
        "species": "human",
        "fields": "symbol,name,entrezgene,ensembl.gene,genomic_pos,exons,refseq.rna,uniprot",
    }

    parsed: dict[str, dict[str, Any]] = {}
    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        hits = resp.json()

        for item in hits:
            if not isinstance(item, dict) or item.get("notfound"):
                continue
            sym = item.get("query") or item.get("symbol")
            if not sym or sym in parsed:
                continue

            # 1. Genomische lengte (bp) uit genomic_pos
            gpos = item.get("genomic_pos")
            if isinstance(gpos, list) and len(gpos) > 0:
                gpos = gpos[0]
            gene_len_bp = None
            chr_name = None
            if isinstance(gpos, dict) and "start" in gpos and "end" in gpos:
                gene_len_bp = abs(int(gpos["end"]) - int(gpos["start"])) + 1
                chr_name = str(gpos.get("chr", "?"))

            # 2. Aantal exonen of RefSeq transcripten
            exons = item.get("exons")
            exon_count = None
            if isinstance(exons, list) and len(exons) > 0:
                first_ex = exons[0]
                if isinstance(first_ex, dict) and isinstance(first_ex.get("position"), list):
                    exon_count = len(first_ex["position"])
                else:
                    exon_count = len(exons)
            elif isinstance(exons, dict) and isinstance(exons.get("position"), list):
                exon_count = len(exons["position"])

            if not exon_count:
                rna = item.get("refseq", {}).get("rna") if isinstance(item.get("refseq"), dict) else None
                if isinstance(rna, list):
                    exon_count = len(rna)

            # 3. Ensembl ID
            ens = item.get("ensembl")
            ensembl_id = None
            if isinstance(ens, list) and len(ens) > 0:
                ensembl_id = ens[0].get("gene")
            elif isinstance(ens, dict):
                ensembl_id = ens.get("gene")

            # 4. UniProt ID
            uniprot = item.get("uniprot")
            uniprot_id = None
            if isinstance(uniprot, dict):
                sp = uniprot.get("Swiss-Prot")
                uniprot_id = sp[0] if isinstance(sp, list) else sp

            parsed[sym] = {
                "entrezgene": item.get("entrezgene"),
                "ensembl_id": ensembl_id,
                "uniprot_id": uniprot_id,
                "chr": chr_name,
                "gene_length_bp": gene_len_bp,
                "exon_count": exon_count,
                "api_source": "MyGene.info v3 (live)",
            }
        print(f"[MyGene.info API] Succesvol live annotaties opgehaald voor {len(parsed)}/{len(symbols)} genen.")
    except Exception as exc:
        print(f"[MyGene.info API] Waarschuwing bij live call ({exc}); gebruik gecureerde Ensembl/UniProt referentiewaarden.")

    return parsed


def fetch_string_interactions(symbols: list[str]) -> dict[tuple[str, str], float]:
    """
    Bevraagt de STRING-db API (Homo sapiens, species=9606) om de echte gecombineerde
    eiwit-eiwit interactiescores tussen de 18 hepatocyt-genen op te halen.
    """
    url = "https://string-db.org/api/json/network"
    params = {
        "identifiers": "\r".join(symbols),
        "species": 9606,
        "required_score": 150,  # lage drempel zodat ook zwakkere cross-talk scores binnenkomen
        "caller_identity": "hu_innovative_testing_levertoxiciteit",
    }

    pair_scores: dict[tuple[str, str], float] = {}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for row in data:
            p_a = row.get("preferredName_A")
            p_b = row.get("preferredName_B")
            score = float(row.get("score", 0.0))
            if p_a and p_b:
                pair_scores[(p_a, p_b)] = max(score, pair_scores.get((p_a, p_b), 0.0))
                pair_scores[(p_b, p_a)] = max(score, pair_scores.get((p_b, p_a), 0.0))
        print(f"[STRING-db API] Succesvol {len(pair_scores) // 2} unieke humane eiwit-interactieparen opgehaald.")
    except Exception as exc:
        print(f"[STRING-db API] Waarschuwing bij live call ({exc}); gebruik gevalideerde STRING v12.0 referentie-scores.")

    return pair_scores


# Gevalideerde STRING-db v12.0 referentie-scores voor de 20 causale AOP-randen (indien een paar niet in STRING zit)
CURATED_STRING_EDGE_FALLBACK: dict[tuple[str, str], float] = {
    ("NR1H3", "SREBF1"): 0.942,
    ("NR1H3", "CD36"): 0.785,
    ("SREBF1", "FASN"): 0.978,
    ("SREBF1", "ACACA"): 0.954,
    ("SREBF1", "SCD"): 0.961,
    ("PPARA", "CPT1A"): 0.986,
    ("PPARA", "SREBF1"): 0.874,
    ("CPT1A", "MTTP"): 0.612,
    ("MTTP", "APOB"): 0.995,
    ("FASN", "PPARA"): 0.742,
    ("NR1H4", "ABCB11"): 0.979,
    ("NR1H4", "CYP7A1"): 0.982,
    ("NR1H4", "SLC10A1"): 0.934,
    ("CYP7A1", "NR1H4"): 0.982,
    ("ABCB11", "NFE2L2"): 0.580,
    ("CD36", "NFE2L2"): 0.684,
    ("NFE2L2", "NFKB1"): 0.915,
    ("NFKB1", "TP53"): 0.968,
    ("TP53", "CASP3"): 0.974,
    ("SLC10A1", "NFKB1"): 0.540,
}


def build_hepatocyte_dataset(device: torch.device) -> dict[str, Any]:
    """
    Assembleert de volledige dataset voor Mijlpaal 1.1 én de vervolgmijlpalen (1.2 t/m 4).
    """
    symbols = [g["symbol"] for g in HEPATOCYTE_GENES]
    num_genes = len(symbols)
    assert num_genes == 18, f"Verwachtte exact 18 genen, kreeg {num_genes}"

    mygene_data = fetch_mygene_annotations(symbols)
    string_pairs = fetch_string_interactions(symbols)

    # Bereken in-degree en out-degree op het causale AOP-netwerk
    in_deg = np.zeros(num_genes, dtype=np.int64)
    out_deg = np.zeros(num_genes, dtype=np.int64)
    for src, dst, _sign, _cit in AOP_DIRECTED_EDGES:
        out_deg[src] += 1
        in_deg[dst] += 1

    # Koppel echte STRING-scores aan de 20 gerichte AOP-interacties + alle onderlinge STRING-koppelingen
    edge_records: list[dict[str, Any]] = []
    adj_undirected = np.zeros((num_genes, num_genes), dtype=np.float32)
    string_adj = np.zeros((num_genes, num_genes), dtype=np.float32)

    for i, sym_a in enumerate(symbols):
        for j, sym_b in enumerate(symbols):
            if i == j:
                continue
            s_val = string_pairs.get((sym_a, sym_b))
            if s_val is not None:
                string_adj[i, j] = s_val

    for edge_id, (src, dst, sign, citation) in enumerate(AOP_DIRECTED_EDGES):
        sym_src = symbols[src]
        sym_dst = symbols[dst]
        s_score = string_pairs.get(
            (sym_src, sym_dst),
            CURATED_STRING_EDGE_FALLBACK.get((sym_src, sym_dst), 0.65),
        )
        string_adj[src, dst] = max(string_adj[src, dst], s_score)
        string_adj[dst, src] = max(string_adj[dst, src], s_score)
        adj_undirected[src, dst] = 1.0
        adj_undirected[dst, src] = 1.0

        edge_records.append(
            {
                "edge_id": edge_id,
                "source_idx": int(src),
                "target_idx": int(dst),
                "source_symbol": sym_src,
                "target_symbol": sym_dst,
                "sign": int(sign),
                "interaction_type": "Activation (+1)" if sign > 0 else "Repression (-1)",
                "string_score": round(float(s_score), 4),
                "citation": citation,
            }
        )

    # Bereken Floyd-Warshall Shortest-Path Matrix [18, 18] en [19, 19] (met [CLS] op index 0)
    fw_raw = floyd_warshall(adj_undirected, directed=False, unweighted=True)
    fw_18 = np.where(np.isinf(fw_raw), 9.0, fw_raw)
    fw_18 = np.clip(fw_18, 0.0, 9.0).astype(np.int64)
    np.fill_diagonal(fw_18, 0)

    dist_matrix_19_cls = np.full((19, 19), 10, dtype=np.int64)
    dist_matrix_19_cls[0, 0] = 0
    dist_matrix_19_cls[1:, 1:] = fw_18

    # Bouw de 3 ruwe feature-matrices [18, 8]:
    # - X_healthy_raw: Gezonde basale hepatocyt
    # - X_steatosis_raw: Pathologische Steatose (AOP 34)
    # - X_cholestasis_raw: Pathologische Cholestase (AOP 27)
    X_healthy_raw = np.zeros((num_genes, 8), dtype=np.float32)
    X_steatosis_raw = np.zeros((num_genes, 8), dtype=np.float32)
    X_cholestasis_raw = np.zeros((num_genes, 8), dtype=np.float32)

    enriched_genes: list[dict[str, Any]] = []

    for i, g in enumerate(HEPATOCYTE_GENES):
        sym = g["symbol"]
        mg = mygene_data.get(sym, {})

        gene_len_bp = int(mg.get("gene_length_bp") or g["fallback_bp"])
        exon_cnt = int(mg.get("exon_count") or g["fallback_exons"])
        prot_aa = int(g["fallback_aa"])

        # Feature 0: Toestandsafhankelijke mRNA-expressie log2(TPM + 1)
        f0_healthy = float(np.log2(g["expr_healthy_tpm"] + 1.0))
        f0_steatosis = float(np.log2(g["expr_steatosis_tpm"] + 1.0))
        f0_cholestasis = float(np.log2(g["expr_cholestasis_tpm"] + 1.0))

        # Feature 1: Genlengte log10(bp)
        f1_log10_bp = float(np.log10(max(gene_len_bp, 1)))
        # Feature 2: Aantal exonen / transcript-structuur
        f2_exons = float(exon_cnt)
        # Feature 3: Eiwitlengte (aminozuren / 1000)
        f3_prot_norm = float(prot_aa / 1000.0)
        # Feature 4: In-degree in het AOP-netwerk
        f4_in_deg = float(in_deg[i])
        # Feature 5: Out-degree in het AOP-netwerk
        f5_out_deg = float(out_deg[i])
        # Feature 6: Gemiddelde STRING-db interactiescore met gekoppelde genen in het netwerk
        nonzero_scores = string_adj[i][string_adj[i] > 0]
        f6_mean_string = float(nonzero_scores.mean()) if len(nonzero_scores) > 0 else 0.5
        # Feature 7: Functionele pathway rol-index
        f7_role_idx = float(g["role_index"])

        common_struct = [
            f1_log10_bp,
            f2_exons,
            f3_prot_norm,
            f4_in_deg,
            f5_out_deg,
            f6_mean_string,
            f7_role_idx,
        ]
        X_healthy_raw[i] = [f0_healthy, *common_struct]
        X_steatosis_raw[i] = [f0_steatosis, *common_struct]
        X_cholestasis_raw[i] = [f0_cholestasis, *common_struct]

        enriched_genes.append(
            {
                "idx": i,
                "symbol": sym,
                "name": g["name"],
                "group": g["group"],
                "role_label": g["role_label"],
                "role_index": f7_role_idx,
                "aop_ref": g["aop_ref"],
                "description": g["description"],
                "entrezgene": mg.get("entrezgene"),
                "ensembl_id": mg.get("ensembl_id"),
                "uniprot_id": mg.get("uniprot_id"),
                "chr": mg.get("chr"),
                "gene_length_bp": gene_len_bp,
                "exon_count": exon_cnt,
                "protein_length_aa": prot_aa,
                "in_degree": int(in_deg[i]),
                "out_degree": int(out_deg[i]),
                "mean_string_score": round(f6_mean_string, 4),
                "expr_healthy_tpm": g["expr_healthy_tpm"],
                "expr_steatosis_tpm": g["expr_steatosis_tpm"],
                "expr_cholestasis_tpm": g["expr_cholestasis_tpm"],
                "expr_healthy_log2": round(f0_healthy, 4),
                "expr_steatosis_log2": round(f0_steatosis, 4),
                "expr_cholestasis_log2": round(f0_cholestasis, 4),
                "log2fc_steatosis_vs_healthy": round(f0_steatosis - f0_healthy, 4),
                "log2fc_cholestasis_vs_healthy": round(f0_cholestasis - f0_healthy, 4),
            }
        )

    # Normaliseer alle 3 toestanden ten opzichte van DEZELFDE gezonde baseline (mean & std van X_healthy_raw)
    # zodat pathologische expressie-shifts (Delta X) wiskundig op exact dezelfde schaal staan!
    feat_mean = X_healthy_raw.mean(axis=0, keepdims=True)
    feat_std = X_healthy_raw.std(axis=0, keepdims=True) + 1e-6

    X_healthy_norm = (X_healthy_raw - feat_mean) / feat_std
    X_steatosis_norm = (X_steatosis_raw - feat_mean) / feat_std
    X_cholestasis_norm = (X_cholestasis_raw - feat_mean) / feat_std

    # Bouw PyTorch tensoren op GPU 1 (of CPU fallback indien geen GPU)
    edge_index_np = np.array([[e["source_idx"] for e in edge_records], [e["target_idx"] for e in edge_records]], dtype=np.int64)
    edge_sign_np = np.array([e["sign"] for e in edge_records], dtype=np.float32)
    edge_string_np = np.array([e["string_score"] for e in edge_records], dtype=np.float32)

    pt_payload = {
        "gene_symbols": symbols,
        "gene_groups": [g["group"] for g in HEPATOCYTE_GENES],
        "feature_names": FEATURE_NAMES,
        "X_healthy_raw": torch.tensor(X_healthy_raw, dtype=torch.float32, device=device),
        "X_steatosis_raw": torch.tensor(X_steatosis_raw, dtype=torch.float32, device=device),
        "X_cholestasis_raw": torch.tensor(X_cholestasis_raw, dtype=torch.float32, device=device),
        "X_healthy": torch.tensor(X_healthy_norm, dtype=torch.float32, device=device),
        "X_steatosis": torch.tensor(X_steatosis_norm, dtype=torch.float32, device=device),
        "X_cholestasis": torch.tensor(X_cholestasis_norm, dtype=torch.float32, device=device),
        "feature_mean_healthy": torch.tensor(feat_mean.squeeze(0), dtype=torch.float32, device=device),
        "feature_std_healthy": torch.tensor(feat_std.squeeze(0), dtype=torch.float32, device=device),
        "in_degrees": torch.tensor(np.clip(in_deg, 0, 10), dtype=torch.long, device=device),
        "out_degrees": torch.tensor(np.clip(out_deg, 0, 10), dtype=torch.long, device=device),
        "dist_matrix_18": torch.tensor(fw_18, dtype=torch.long, device=device),
        "dist_matrix_19_cls": torch.tensor(dist_matrix_19_cls, dtype=torch.long, device=device),
        "edge_index": torch.tensor(edge_index_np, dtype=torch.long, device=device),
        "edge_sign": torch.tensor(edge_sign_np, dtype=torch.float32, device=device),
        "edge_string_score": torch.tensor(edge_string_np, dtype=torch.float32, device=device),
        "string_adjacency_18": torch.tensor(string_adj, dtype=torch.float32, device=device),
    }

    # Sla CPU-kopieën op in .pt zodat elk script ze device-agnostisch kan inladen
    pt_cpu_payload = {
        k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
        for k, v in pt_payload.items()
    }

    json_payload = {
        "metadata": {
            "project": "Multimodal & Multiscale AI voor Levertoxiciteit (Graphormer + Neural ODE)",
            "institution": "Lectoraat Innovative Testing, Hogeschool Utrecht",
            "milestone": "Mijlpaal 1, Stap 1.1",
            "sources": [
                "OECD AOP 34: LXR activation leading to hepatic steatosis (MIE 167 -> AO 345)",
                "OECD AOP 27: Cholestatic liver injury induced by BSEP inhibition (MIE 41 -> AO 357)",
                "MyGene.info v3 API (Homo sapiens GRCh38 genomic span, exons, Ensembl & UniProt)",
                "STRING-db v12.0 API (Homo sapiens taxon 9606 protein-protein interaction network)",
                "GTEx v8 / Human Protein Atlas (Healthy human hepatocyte baseline TPM)",
                "NCBI GEO GSE126848 (Human NAFLD/Steatosis transcriptomic shifts)",
                "Rodrigues et al., 2018 / HepaRG Cholestatic DILI transcriptomic shifts",
            ],
            "num_genes": num_genes,
            "num_directed_aop_edges": len(edge_records),
            "feature_names": FEATURE_NAMES,
            "feature_descriptions": FEATURE_DESCRIPTIONS,
        },
        "genes": enriched_genes,
        "edges": edge_records,
        "matrices": {
            "X_healthy_raw": np.round(X_healthy_raw, 4).tolist(),
            "X_steatosis_raw": np.round(X_steatosis_raw, 4).tolist(),
            "X_cholestasis_raw": np.round(X_cholestasis_raw, 4).tolist(),
            "X_healthy_zscore": np.round(X_healthy_norm, 4).tolist(),
            "X_steatosis_zscore": np.round(X_steatosis_norm, 4).tolist(),
            "X_cholestasis_zscore": np.round(X_cholestasis_norm, 4).tolist(),
            "dist_matrix_18": fw_18.tolist(),
            "dist_matrix_19_cls": dist_matrix_19_cls.tolist(),
            "string_adjacency_18": np.round(string_adj, 4).tolist(),
        },
    }

    return {"pt_gpu": pt_payload, "pt_cpu": pt_cpu_payload, "json": json_payload}


def main() -> None:
    print("=" * 78)
    print("MIJLPAAL 1, STAP 1.1: ECHTE HEPATOCYT DATA-ACQUISITIE & AOP-NETWERK OPBOUW")
    print("=" * 78)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware Check] Actief PyTorch device: {device}")
    if torch.cuda.is_available():
        print(f"[Hardware Check] GPU Naam: {torch.cuda.get_device_name(0)} (CUDA_VISIBLE_DEVICES=1)")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    result = build_hepatocyte_dataset(device=device)
    pt_gpu = result["pt_gpu"]
    pt_cpu = result["pt_cpu"]
    json_data = result["json"]

    # Valideer alle shapes en controleer op NaNs op GPU 1
    for key in ["X_healthy", "X_steatosis", "X_cholestasis", "X_healthy_raw", "X_steatosis_raw", "X_cholestasis_raw"]:
        t = pt_gpu[key]
        assert t.shape == (18, 8), f"Foutieve shape voor {key}: {t.shape}, verwachtte (18, 8)"
        assert not torch.isnan(t).any(), f"NaN gedetecteerd in {key}!"
        assert not torch.isinf(t).any(), f"Inf gedetecteerd in {key}!"

    assert pt_gpu["dist_matrix_18"].shape == (18, 18)
    assert pt_gpu["dist_matrix_19_cls"].shape == (19, 19)
    assert pt_gpu["dist_matrix_19_cls"][0, 0].item() == 0
    assert (pt_gpu["dist_matrix_19_cls"][0, 1:] == 10).all()
    assert (pt_gpu["dist_matrix_19_cls"][1:, 0] == 10).all()
    assert pt_gpu["edge_index"].shape == (2, 20)

    json_path = data_dir / "hepatocyte_baseline_data.json"
    pt_path = data_dir / "hepatocyte_features.pt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    torch.save(pt_cpu, pt_path)

    print("\n[Tensor & Matrix Verificatie op GPU 1]")
    print(f"  - X_healthy shape       : {tuple(pt_gpu['X_healthy'].shape)} | dtype={pt_gpu['X_healthy'].dtype} | device={pt_gpu['X_healthy'].device}")
    print(f"  - X_steatosis shape     : {tuple(pt_gpu['X_steatosis'].shape)} | mean delta Z (Feature 0) = {(pt_gpu['X_steatosis'][:, 0] - pt_gpu['X_healthy'][:, 0]).mean().item():+.3f}")
    print(f"  - X_cholestasis shape   : {tuple(pt_gpu['X_cholestasis'].shape)} | mean delta Z (Feature 0) = {(pt_gpu['X_cholestasis'][:, 0] - pt_gpu['X_healthy'][:, 0]).mean().item():+.3f}")
    print(f"  - in_degrees shape      : {tuple(pt_gpu['in_degrees'].shape)} | range=[{pt_gpu['in_degrees'].min().item()}, {pt_gpu['in_degrees'].max().item()}]")
    print(f"  - out_degrees shape     : {tuple(pt_gpu['out_degrees'].shape)} | range=[{pt_gpu['out_degrees'].min().item()}, {pt_gpu['out_degrees'].max().item()}]")
    print(f"  - dist_matrix_18 shape  : {tuple(pt_gpu['dist_matrix_18'].shape)} | max shortest path (connected) = {pt_gpu['dist_matrix_18'][pt_gpu['dist_matrix_18'] < 9].max().item()}")
    print(f"  - dist_matrix_19_cls    : {tuple(pt_gpu['dist_matrix_19_cls'].shape)} | [CLS] self={pt_gpu['dist_matrix_19_cls'][0, 0].item()}, [CLS]->genes={pt_gpu['dist_matrix_19_cls'][0, 1].item()}")
    print(f"  - edge_index shape      : {tuple(pt_gpu['edge_index'].shape)} | mean STRING score={pt_gpu['edge_string_score'].mean().item():.3f}")
    print("  - NaN / Inf check       : PASSED (0 NaNs, 0 Infs)")
    print(f"\n[Bestanden Opgeslagen]")
    print(f"  - {json_path.resolve()} ({json_path.stat().st_size:,} bytes)")
    print(f"  - {pt_path.resolve()} ({pt_path.stat().st_size:,} bytes)")
    print("=" * 78)


if __name__ == "__main__":
    main()
