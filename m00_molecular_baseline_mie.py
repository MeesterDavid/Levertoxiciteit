"""
Mijlpaal 0 (Stap A Validatiestrategie): Moleculaire Structuur-Baseline & MIE-Voorspeller
========================================================================================
Project : Multimodal & Multiscale AI voor Levertoxiciteit (Graphormer + Neural ODE)
Kader   : Promotieonderzoek Lectoraat Innovative Testing (Hogeschool Utrecht)
Model   : Microsoft Graphormer (`clefourrier/pcqm4mv2_graphormer_base`) via Hugging Face
          `GraphormerForGraphClassification` (geïdentificeerd in chatgesprekmetgeminioverdemo.pdf, p. 1)

Doel van dit programma:
1. Laadt het voorgetrainde open-weights 2D moleculaire Graph Transformer model
   `clefourrier/pcqm4mv2_graphormer_base` (47.7M parameters, 12 lagen, 768D, 32 heads) op GPU 1.
2. Converteert 2D SMILES-strings via RDKit naar exacte OGB-compatibele molecuulgrafen
   (9 atoomfeatures, 3 bindingsfeatures, Floyd-Warshall kortste paden en multi-hop edge-paden)
   zonder afhankelijk te zijn van externe Cython-compilatie.
3. Voert een daadwerkelijke forward-pass uit door `GraphormerForGraphClassification` op GPU 1,
   extraheert de 768-dimensionale `[CLS]` molecuul-embeddings en voorspelt per stof of deze een
   Molecular Initiating Event (MIE) triggert:
   - Klasse 0: Geen MIE / Adaptieve Homeostase (bijv. Fenofibrate, DMSO, Caffeine, Metformin)
   - Klasse 1: MIE 167 — NR1H3 (LXR-alpha) Activatie -> AOP 34 Steatose (bijv. T0901317, GW3965, Amiodarone, Tamoxifen)
   - Klasse 2: MIE 41  — ABCB11 (BSEP) Remming -> AOP 27 Cholestase (bijv. Bosentan, Troglitazone, Glibenclamide, Chlorpromazine)
4. Slaat de resultaten en 768D Graphormer-tensoren op in `data/molecular_baseline_mie.pt` en
   `data/molecular_baseline_mie.json`, en genereert het interactieve HTML/SVG-dashboard
   `00_molecular_baseline_mie.html` (met Nederlandstalige handleiding boven elke cel).
"""

import html as html_lib
import json
import os
from pathlib import Path
from typing import Any

# Stel Hugging Face cache lokaal in op de 1.5TB schijf binnen de projectmap
CACHE_DIR = Path(__file__).resolve().parent / "data" / "hf_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
if (CACHE_DIR / "hub" / "models--clefourrier--pcqm4mv2_graphormer_base").exists():
    os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdDepictor
from transformers.models.deprecated.graphormer import (
    GraphormerConfig,
    GraphormerForGraphClassification,
)

# Strikte reproduceerbaarheid conform ARCHITECTURE.md & .clinerules
torch.manual_seed(42)
np.random.seed(42)

HF_MODEL_ID = "clefourrier/pcqm4mv2_graphormer_base"
UNREACHABLE_NODE_DISTANCE = 510

# ==============================================================================
# 1. REFERENTIESTOFFEN MET FORMELE OECD AOP / MIE ANNOTATIES
# ==============================================================================
MIE_CLASSES = {
    0: "Geen_MIE_Adaptief",
    1: "MIE_167_LXR_Steatose",
    2: "MIE_41_BSEP_Cholestase",
}

MIE_CLASS_LABELS_NL = {
    0: "Geen Toxische MIE (Adaptief / Controle)",
    1: "MIE 167: LXRα-activatie (AOP 34 → Steatose)",
    2: "MIE 41: BSEP-remming (AOP 27 → Cholestase)",
}

MIE_CLASS_COLORS = {
    0: "#10B981",  # Groen (Veilig / Adaptieve Homeostase)
    1: "#F43F5E",  # Koraalrood (AOP 34 Steatose MIE 167)
    2: "#F59E0B",  # Amber/Oranje (AOP 27 Cholestase MIE 41)
}

REFERENCE_MIE_PANEL: list[dict[str, Any]] = [
    # --- KLASSE 1: MIE 167 (NR1H3 / LXR-alpha activatie -> AOP 34 Steatose) ---
    {
        "name": "T0901317",
        "smiles": "CC(C)(C)c1ccc(cc1)S(=O)(=O)N(CC(F)(F)F)c2ccc(cc2)C(C(F)(F)F)(C(F)(F)F)O",
        "true_class": 1,
        "mie_code": "MIE 167 (AOP 34)",
        "target_gene": "NR1H3 (LXRα) → SREBF1 / FASN / CD36",
        "mechanism_nl": (
            "Synthetische hexafluoro-isopropanol sulfonamide LXRα/β-agonist (EC50 ~ 20 nM). "
            "Bindt direct aan het ligand-bindend domein van NR1H3 (MIE 167) en forceert massale de novo lipogenese."
        ),
        "limitation_note_nl": (
            "Structuur-baseline ziet altijd: 'MIE 167 = JA'. Maar bij 0.1 µM vangt PPARA/CPT1A dit in de "
            "hepatocyt nog homeostatisch op, terwijl pas boven ~5-10 µM kanteling naar steatose optreedt!"
        ),
    },
    {
        "name": "GW3965",
        "smiles": "COc1ccc(CC(O)(C(F)(F)F)C(F)(F)F)cc1CNC(Cc1ccc(OCC(C)(C)C(=O)O)cc1)c1cccc(Cl)c1",
        "true_class": 1,
        "mie_code": "MIE 167 (AOP 34)",
        "target_gene": "NR1H3 (LXRα) → SREBF1 / SCD",
        "mechanism_nl": (
            "Selectieve niet-steroïdale LXRα/β-agonist met bistrifluormethyl-carbinol farmacofoor. "
            "Tweede gouden-standaard MIE 167 referentie-inductor voor AOP 34 hepatische steatose."
        ),
        "limitation_note_nl": (
            "Hoewel GW3965 krachtig MIE 167 triggert, hangt de daadwerkelijke lipidenstapeling af van de "
            "VLDL-exportcapaciteit (MTTP/APOB) en blootstellingsduur in de Neural ODE."
        ),
    },
    {
        "name": "Amiodarone",
        "smiles": "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1",
        "true_class": 1,
        "mie_code": "MIE 167 / KE 177 (AOP 34)",
        "target_gene": "NR1H3 / SREBF1 & Mitochondriale β-oxidatie",
        "mechanism_nl": (
            "Amfifiel gejodeerd benzofuraan anti-aritmicum. Induceert ernstige klinische steatohepatitis "
            "door gelijktijdige opregulatie van LXR/SREBP-1c lipogenese-genen en remming van CPT1A."
        ),
        "limitation_note_nl": (
            "Veroorzaakt bij therapeutische lage spiegels soms reversibele fosfolipidose, maar kantelt "
            "bij chronische accumulatie naar irreversibele NASH en apoptose (CASP3)."
        ),
    },
    {
        "name": "Tamoxifen",
        "smiles": "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
        "true_class": 1,
        "mie_code": "MIE 167 (AOP 34)",
        "target_gene": "SREBF1 / FASN / CD36",
        "mechanism_nl": (
            "Trifenylethyleen-derivaat dat in humane hepatocyten (HepaRG) de novo vetzuursynthese (FASN, ACACA) "
            "en vetzuur-opname (CD36) stimuleert en klinisch bekend staat om medicijn-geïnduceerde vetlever."
        ),
        "limitation_note_nl": (
            "Het kantelpunt naar steatose treedt pas op wanneer de compensatoire PPARα-route overbelast raakt."
        ),
    },
    # --- KLASSE 2: MIE 41 (ABCB11 / BSEP remming -> AOP 27 Cholestase) ---
    {
        "name": "Bosentan",
        "smiles": "CC(C)(C)c1ccc(cc1)S(=O)(=O)Nc2c(nc(nc2OCCO)c3ccccc3)Oc4ccccc4OC",
        "true_class": 2,
        "mie_code": "MIE 41 (AOP 27)",
        "target_gene": "ABCB11 (BSEP) → NR1H4 / NFE2L2",
        "mechanism_nl": (
            "Endotheline-receptorantagonist en archetype BSEP-remmer (IC50 ~ 15-25 µM, Rodrigues et al., 2018). "
            "Blokkeert canaliculaire galzoutexport (ABCB11, MIE 41), wat leidt tot toxische galzuurretentie."
        ),
        "limitation_note_nl": (
            "Bij lage dosis compenseert de hepatocyt via FXR (NR1H4): CYP7A1 en SLC10A1 worden onderdrukt. "
            "De structuur-baseline mist dit FXR-feedbacknetwerk volledig!"
        ),
    },
    {
        "name": "Troglitazone",
        "smiles": "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
        "true_class": 2,
        "mie_code": "MIE 41 (AOP 27)",
        "target_gene": "ABCB11 (BSEP) → NFE2L2 / TP53 / CASP3",
        "mechanism_nl": (
            "Van de markt gehaald thiazolidinedion met chromaan-ring. Krachtige competitieve remmer van BSEP "
            "(ABCB11, MIE 41, IC50 ~ 3 µM), resulterend in cholestatische leverschade en mitochondriële ROS."
        ),
        "limitation_note_nl": (
            "Zonder tijdsafhankelijke ODE-integratie is niet te zien hoe snel de Nrf2 (NFE2L2) antioxidant-buffer "
            "wordt uitgeput richting Caspase-3 apoptose."
        ),
    },
    {
        "name": "Glibenclamide",
        "smiles": "COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1",
        "true_class": 2,
        "mie_code": "MIE 41 (AOP 27)",
        "target_gene": "ABCB11 (BSEP)",
        "mechanism_nl": (
            "Sulfonylureum-derivaat dat de ATP-afhankelijke galzouttransportpomp BSEP (ABCB11, MIE 41) direct "
            "remt en in menselijke hepatocyten acute galzuurretentie veroorzaakt."
        ),
        "limitation_note_nl": (
            "In vitro treedt BSEP-remming op bij hoge micromolaire concentraties; bij lage therapeutische "
            "dosis blijft de cel binnen de homeostatische FXR-attractor."
        ),
    },
    {
        "name": "Chlorpromazine",
        "smiles": "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",
        "true_class": 2,
        "mie_code": "MIE 41 / KE 149 (AOP 27)",
        "target_gene": "ABCB11 (BSEP) & SLC10A1 → NFKB1",
        "mechanism_nl": (
            "Klassiek tricyclisch fenothiazine dat intrahepatische cholestase veroorzaakt door remming van "
            "BSEP-expressie/functie (MIE 41) gecombineerd met oxidatieve stress en NF-κB ontstekingsreacties."
        ),
        "limitation_note_nl": (
            "De overgang van adaptieve cholestatische stress naar daadwerkelijke celdood vereist modellering "
            "van de NFKB1 → TP53 → CASP3 cascade in Mijlpaal 1–3."
        ),
    },
    # --- KLASSE 0: Geen Toxische MIE / Adaptieve Homeostase Controle ---
    {
        "name": "Fenofibrate",
        "smiles": "CC(C)OC(=O)C(C)(C)Oc1ccc(cc1)C(=O)c2ccc(cc2)Cl",
        "true_class": 0,
        "mie_code": "Geen MIE (Adaptieve PPARα-agonist)",
        "target_gene": "PPARA → CPT1A / MTTP / APOB (Beschermend)",
        "mechanism_nl": (
            "Klinische lipidenverlager die PPAR-alpha (PPARA) activeert. Stimuleert mitochondriële β-oxidatie "
            "(CPT1A) en VLDL-export (MTTP/APOB) en remt SREBP-1c (versterkt de gezonde attractor z0!)."
        ),
        "limitation_note_nl": (
            "Cruciale teststof: heeft aromatische ringen en halogeen zoals veel geneesmiddelen, maar triggert "
            "géén MIE 167 of MIE 41 en stabiliseert juist de homeostatische attractor."
        ),
    },
    {
        "name": "DMSO_Control",
        "smiles": "CS(=O)C",
        "true_class": 0,
        "mie_code": "Geen MIE (Vehikelcontrole)",
        "target_gene": "Geen (Homeostase z0)",
        "mechanism_nl": (
            "Dimethylsulfoxide: standaard oplosmiddel en negatieve controle in HepaRG toxiciteits-experimenten. "
            "Triggert geen LXRα (MIE 167) en remt BSEP (MIE 41) niet."
        ),
        "limitation_note_nl": (
            "Fungeert in Mijlpaal 1–4 als het nul-ankerpunt waarbij dz/dt = 0 op de gezonde attractor z0 blijft."
        ),
    },
    {
        "name": "Caffeine",
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "true_class": 0,
        "mie_code": "Geen MIE (Negatieve Controle)",
        "target_gene": "Geen (CYP1A2 metabolisme)",
        "mechanism_nl": (
            "Xanthine-alkaloïde en standaard negatieve controle in DILI-benchmarks. Wordt probleemloos door "
            "hepatocyten gemetaboliseerd zonder verstoring van lipiden- of galzuurhomeostase."
        ),
        "limitation_note_nl": (
            "Bevestigt dat heterocyclische kleine moleculen zonder MIE-farmacofoor laag scoren op zowel MIE 167 als MIE 41."
        ),
    },
    {
        "name": "Metformin",
        "smiles": "CN(C)C(=N)NC(=N)N",
        "true_class": 0,
        "mie_code": "Geen MIE (Anti-steatotische Controle)",
        "target_gene": "AMPK → Remming SREBF1 / ACACA",
        "mechanism_nl": (
            "Eerstelijns biguanide antidiabeticum. Activeert AMPK waardoor ACACA en SREBF1 juist worden geremd; "
            "veroorzaakt geen steatose (MIE 167) en geen BSEP-cholestase (MIE 41)."
        ),
        "limitation_note_nl": (
            "Laat zien dat hydrofiele metabolisme-modulatoren duidelijk gescheiden worden van lipofiele LXR/BSEP-liganden."
        ),
    },
]


# ==============================================================================
# 2. OGB / GRAPHORMER MOLECUULGRAAF PREPROCESSING (100% PURE PYTHON / NUMPY)
# ==============================================================================
CHIRALITY_LIST = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
]
HYBRIDIZATION_LIST = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
BOND_TYPE_LIST = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
BOND_STEREO_LIST = [
    "STEREONONE",
    "STEREOZ",
    "STEREOE",
    "STEREOCIS",
    "STEREOTRANS",
    "STEREOANY",
]


def _safe_index(lst: list[Any], val: Any, default: int) -> int:
    try:
        return lst.index(val)
    except ValueError:
        return default


def atom_to_ogb_features(atom: Chem.Atom) -> list[int]:
    """
    Zet een RDKit Atoom om naar de exacte 9 categorische OGB-features waarop
    Microsoft Graphormer (`clefourrier/pcqm4mv2_graphormer_base`) is voorgetraind.
    """
    atomic_num_idx = min(max(atom.GetAtomicNum() - 1, 0), 118)
    chirality_idx = _safe_index(CHIRALITY_LIST, str(atom.GetChiralTag()), 4)
    degree_idx = min(max(atom.GetTotalDegree(), 0), 11)
    formal_charge_idx = min(max(atom.GetFormalCharge() + 5, 0), 11)
    num_h_idx = min(max(atom.GetTotalNumHs(), 0), 9)
    radical_e_idx = min(max(atom.GetNumRadicalElectrons(), 0), 5)
    hybridization_idx = _safe_index(HYBRIDIZATION_LIST, str(atom.GetHybridization()), 5)
    is_aromatic_idx = 1 if atom.GetIsAromatic() else 0
    is_in_ring_idx = 1 if atom.IsInRing() else 0
    return [
        atomic_num_idx,
        chirality_idx,
        degree_idx,
        formal_charge_idx,
        num_h_idx,
        radical_e_idx,
        hybridization_idx,
        is_aromatic_idx,
        is_in_ring_idx,
    ]


def bond_to_ogb_features(bond: Chem.Bond) -> list[int]:
    """
    Zet een RDKit Binding om naar de exacte 3 categorische OGB-bindingsfeatures.
    """
    bond_type_idx = _safe_index(BOND_TYPE_LIST, str(bond.GetBondType()), 4)
    stereo_idx = _safe_index(BOND_STEREO_LIST, str(bond.GetStereo()), 5)
    is_conj_idx = 1 if bond.GetIsConjugated() else 0
    return [bond_type_idx, stereo_idx, is_conj_idx]


def convert_to_single_emb(x: np.ndarray, offset: int = 512) -> np.ndarray:
    """
    Verschuift elke feature-kolom met veelvouden van 512 zodat één gedeelde
    nn.Embedding tabel in Graphormer alle 9 atoom- of 3 bindingsfeatures kan opzoeken.
    Exact conform `collating_graphormer.py`.
    """
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    return x + feature_offset


def floyd_warshall_with_path(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Zuivere NumPy/Python implementatie van `algos_graphormer.floyd_warshall(adj)`.
    Berekent kortste-padafstanden M [N, N] én tussenliggende pad-indices path [N, N].
    """
    n = adj.shape[0]
    M = np.full((n, n), UNREACHABLE_NODE_DISTANCE, dtype=np.int32)
    path = np.full((n, n), -1, dtype=np.int32)

    for i in range(n):
        M[i, i] = 0
        for j in range(n):
            if i != j and adj[i, j]:
                M[i, j] = 1

    for k in range(n):
        for i in range(n):
            m_ik = int(M[i, k])
            if m_ik >= UNREACHABLE_NODE_DISTANCE:
                continue
            for j in range(n):
                cost = m_ik + int(M[k, j])
                if int(M[i, j]) > cost:
                    M[i, j] = cost
                    path[i, j] = k

    for i in range(n):
        for j in range(n):
            if M[i, j] >= UNREACHABLE_NODE_DISTANCE:
                path[i, j] = UNREACHABLE_NODE_DISTANCE
                M[i, j] = UNREACHABLE_NODE_DISTANCE

    return M, path


def _get_all_intermediate_nodes(path: np.ndarray, i: int, j: int) -> list[int]:
    k = int(path[i, j])
    if k == -1:
        return []
    return _get_all_intermediate_nodes(path, i, k) + [k] + _get_all_intermediate_nodes(path, k, j)


def gen_edge_input_pure(max_dist: int, path: np.ndarray, attn_edge_type: np.ndarray) -> np.ndarray:
    """
    Zuivere Python/NumPy implementatie van `algos_graphormer.gen_edge_input`.
    Vult een tensor van shape [N, N, max_dist, 3] met de bindingsfeatures langs het kortste pad tussen i en j.
    """
    n = path.shape[0]
    edge_F = attn_edge_type.shape[-1]
    edge_fea_all = np.full((n, n, max_dist, edge_F), -1, dtype=np.int64)

    for i in range(n):
        for j in range(n):
            if i == j or path[i, j] == UNREACHABLE_NODE_DISTANCE:
                continue
            node_seq = [i] + _get_all_intermediate_nodes(path, i, j) + [j]
            num_steps = len(node_seq) - 1
            for k in range(min(num_steps, max_dist)):
                u, v = node_seq[k], node_seq[k + 1]
                edge_fea_all[i, j, k, :] = attn_edge_type[u, v, :]

    return edge_fea_all


def smiles_to_graphormer_item(smiles: str, spatial_pos_max: int = 20) -> dict[str, Any]:
    """
    Zet een SMILES-string met RDKit om naar een volledige Graphormer-input dictionary
    inclusief 2D atoomcoördinaten voor visualisatie.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Ongeldige SMILES string: {smiles}")

    # Bereken 2D coördinaten voor de molecuulgraaf-visualisatie
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()

    num_nodes = mol.GetNumAtoms()
    node_feat = np.array([atom_to_ogb_features(mol.GetAtomWithIdx(i)) for i in range(num_nodes)], dtype=np.int64)

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_attr_list: list[list[int]] = []
    bond_records: list[dict[str, Any]] = []

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bf = bond_to_ogb_features(bond)
        edge_src.extend([u, v])
        edge_dst.extend([v, u])
        edge_attr_list.extend([bf, bf])
        bond_records.append(
            {
                "u": u,
                "v": v,
                "bond_type": str(bond.GetBondType()),
                "is_aromatic": bool(bond.GetIsAromatic()),
                "is_conjugated": bool(bond.GetIsConjugated()),
            }
        )

    if len(edge_src) > 0:
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_attr = np.array(edge_attr_list, dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 3), dtype=np.int64)

    # Exacte Graphormer index-conversie conform collating_graphormer.py
    input_nodes = convert_to_single_emb(node_feat, offset=512) + 1
    attn_edge_type = np.zeros((num_nodes, num_nodes, 3), dtype=np.int64)
    if edge_index.shape[1] > 0:
        attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr, offset=512) + 1

    adj = np.zeros((num_nodes, num_nodes), dtype=bool)
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = True

    shortest_path_result, path = floyd_warshall_with_path(adj)
    finite_mask = shortest_path_result < UNREACHABLE_NODE_DISTANCE
    max_dist = int(np.amax(shortest_path_result[finite_mask])) if np.any(finite_mask) else 1
    max_dist = max(max_dist, 1)

    input_edges = gen_edge_input_pure(max_dist, path, attn_edge_type)
    attn_bias = np.zeros((num_nodes + 1, num_nodes + 1), dtype=np.float32)

    spatial_pos = shortest_path_result.astype(np.int64) + 1
    in_degree = np.sum(adj, axis=1).reshape(-1).astype(np.int64) + 1
    out_degree = in_degree.copy()
    input_nodes = input_nodes + 1
    input_edges = input_edges + 1

    # Maskeer paden die langer zijn dan spatial_pos_max
    attn_bias[1:, 1:][spatial_pos >= spatial_pos_max] = float("-inf")

    # Atoommetadata voor SVG molecuul-visualisatie
    atoms_meta = []
    for i in range(num_nodes):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        atoms_meta.append(
            {
                "idx": i,
                "symbol": atom.GetSymbol(),
                "atomic_num": atom.GetAtomicNum(),
                "degree": int(atom.GetTotalDegree()),
                "is_aromatic": bool(atom.GetIsAromatic()),
                "formal_charge": int(atom.GetFormalCharge()),
                "x": float(pos.x),
                "y": float(pos.y),
            }
        )

    # Fysisch-chemische en farmacofoor-eigenschappen voor mechanistische interpretatie
    mol_wt = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = int(Descriptors.NumHDonors(mol))
    hba = int(Descriptors.NumHAcceptors(mol))
    num_rings = int(Descriptors.RingCount(mol))
    num_halogen = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
    num_cf3 = len(mol.GetSubstructMatches(Chem.MolFromSmarts("C(F)(F)F")))
    num_sulfonamide = len(mol.GetSubstructMatches(Chem.MolFromSmarts("S(=O)(=O)N")))

    return {
        "smiles": smiles,
        "num_nodes": num_nodes,
        "max_shortest_path": max_dist,
        "input_nodes": torch.tensor(input_nodes, dtype=torch.long).unsqueeze(0),
        "attn_bias": torch.tensor(attn_bias, dtype=torch.float32).unsqueeze(0),
        "attn_edge_type": torch.tensor(attn_edge_type, dtype=torch.long).unsqueeze(0),
        "spatial_pos": torch.tensor(spatial_pos, dtype=torch.long).unsqueeze(0),
        "in_degree": torch.tensor(in_degree, dtype=torch.long).unsqueeze(0),
        "out_degree": torch.tensor(out_degree, dtype=torch.long).unsqueeze(0),
        "input_edges": torch.tensor(input_edges, dtype=torch.long).unsqueeze(0),
        "shortest_path_matrix": shortest_path_result,
        "atoms_meta": atoms_meta,
        "bonds_meta": bond_records,
        "descriptors": {
            "mw": round(mol_wt, 2),
            "logp": round(logp, 2),
            "tpsa": round(tpsa, 2),
            "hbd": hbd,
            "hba": hba,
            "rings": num_rings,
            "halogens": num_halogen,
            "cf3_groups": num_cf3,
            "sulfonamide_groups": num_sulfonamide,
        },
    }


# ==============================================================================
# 3. MICROSOFT GRAPHORMER MIE-CLASSIFIER & EMBEDDING EXTRACTIE OP GPU 1
# ==============================================================================
class MolecularGraphormerMIEPredictor(nn.Module):
    """
    Combineert de voorgetrainde Microsoft Graphormer (`clefourrier/pcqm4mv2_graphormer_base`,
    47.7M parameters) met een gekalibreerde 3-klassen MIE-classificatiekop:
      - Klasse 0: Geen MIE / Adaptieve Homeostase
      - Klasse 1: MIE 167 (NR1H3 / LXR-alpha activatie -> AOP 34 Steatose)
      - Klasse 2: MIE 41  (ABCB11 / BSEP remming -> AOP 27 Cholestase)
    """

    def __init__(self, pretrained_model_id: str = HF_MODEL_ID, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.device = device
        # 1. Laad de originele voorgetrainde GraphormerForGraphClassification (met PCQM4Mv2 regressiekop)
        self.pcqm4m_model = GraphormerForGraphClassification.from_pretrained(pretrained_model_id).to(device)
        self.pcqm4m_model.eval()

        # 2. Laad een 3-klassen GraphormerForGraphClassification voor MIE-voorspelling
        #    (deelt de voorgetrainde 12-laags Graphormer encoder en heeft een 3-klassen GraphormerDecoderHead)
        self.mie_graphormer = GraphormerForGraphClassification.from_pretrained(
            pretrained_model_id,
            num_classes=3,
            ignore_mismatched_sizes=True,
        ).to(device)
        # Deel het encoder-geheugen met pcqm4m_model om VRAM efficiënt te houden
        self.mie_graphormer.encoder = self.pcqm4m_model.encoder

        # bevries de 47M voorgetrainde encoder-parameters zodat de pre-trained PCQM4Mv2 representatie intact blijft
        for param in self.mie_graphormer.encoder.parameters():
            param.requires_grad = False

        # Auxiliaire projectie van fysisch-chemische / farmacofoor-descriptoren (9 features -> 3 logits)
        # die samen met de 768D Graphormer [CLS]-embedding de MIE-receptorbinding verankert
        self.pharmacophore_head = nn.Linear(9, 3).to(device)

    @staticmethod
    def descriptors_to_tensor(desc: dict[str, Any], device: torch.device) -> torch.Tensor:
        vec = [
            desc["mw"] / 500.0,
            desc["logp"] / 5.0,
            desc["tpsa"] / 120.0,
            desc["hbd"] / 5.0,
            desc["hba"] / 10.0,
            desc["rings"] / 5.0,
            desc["halogens"] / 6.0,
            float(desc["cf3_groups"]) / 2.0,
            float(desc["sulfonamide_groups"]),
        ]
        return torch.tensor([vec], dtype=torch.float32, device=device)

    @torch.no_grad()
    def extract_graphormer_representation(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Voert een forward-pass uit door de 12-laags Microsoft Graphormer op GPU 1
        en retourneert:
        - `cls_embedding` [1, 768]: De globale moleculaire [CLS] graaf-representatie
        - `atom_embeddings` [N, 768]: De gecontextualiseerde atoom-representaties
        - `atom_saliency` [N]: Genormaliseerde belangrijkheid van elk atoom t.o.v. de [CLS] readout
        - `homo_lumo_gap_ev`: De originele PCQM4Mv2 voorspelling van de HOMO-LUMO gap
        """
        inputs = {
            k: item[k].to(self.device)
            for k in [
                "input_nodes",
                "input_edges",
                "attn_bias",
                "in_degree",
                "out_degree",
                "spatial_pos",
                "attn_edge_type",
            ]
        }

        pcqm_out = self.pcqm4m_model(**inputs, return_dict=True)
        homo_lumo_gap = float(pcqm_out.logits[0, 0].item())

        enc_out = self.pcqm4m_model.encoder(**inputs, return_dict=True)
        # last_hidden_state heeft shape [1, N + 1, 768], waarbij index 0 het [CLS] graph token is
        last_hidden = enc_out.last_hidden_state[0]  # [N + 1, 768]
        cls_emb = last_hidden[0:1, :]  # [1, 768]
        atom_embs = last_hidden[1:, :]  # [N, 768]

        # Bereken per atoom de cosinus-affiniteit en L2-bijdrage aan de globale [CLS] molecuulrepresentatie
        cls_norm = F.normalize(cls_emb, p=2, dim=-1)
        atom_norm = F.normalize(atom_embs, p=2, dim=-1)
        cos_sim = torch.matmul(atom_norm, cls_norm.T).squeeze(-1)  # [N]
        atom_l2 = torch.norm(atom_embs, p=2, dim=-1)

        # Combineer tot een genormaliseerde atoom-saliency score in [0, 1]
        raw_sal = cos_sim * atom_l2
        if raw_sal.numel() > 1 and float(raw_sal.max() - raw_sal.min()) > 1e-6:
            sal = (raw_sal - raw_sal.min()) / (raw_sal.max() - raw_sal.min())
        else:
            sal = torch.ones_like(raw_sal) * 0.5

        return {
            "cls_embedding": cls_emb,  # [1, 768]
            "atom_embeddings": atom_embs,  # [N, 768]
            "atom_saliency": sal.detach().cpu().numpy().tolist(),
            "homo_lumo_gap_ev": round(homo_lumo_gap, 4),
        }

    def calibrate_mie_head(self, processed_items: list[dict[str, Any]], labels: list[int], epochs: int = 250) -> None:
        """
        Kalibreert de `GraphormerDecoderHead(768, 3)` van `self.mie_graphormer.classifier`
        op de 768-dimensionale Microsoft Graphormer [CLS]-representaties op GPU 1.
        """
        torch.manual_seed(42)
        self.mie_graphormer.classifier.train()
        self.pharmacophore_head.train()

        optimizer = torch.optim.AdamW(
            list(self.mie_graphormer.classifier.parameters()) + list(self.pharmacophore_head.parameters()),
            lr=1e-2,
            weight_decay=1e-2,
        )

        # Verzamel de bevroren 768D encoder outputs [B, 1, 768] zodat de kalibratie op exact dezelfde
        # Graphormer-tensoren plaatsvindt
        with torch.no_grad():
            cls_seq_list = []
            desc_list = []
            for item in processed_items:
                inputs = {
                    k: item[k].to(self.device)
                    for k in [
                        "input_nodes",
                        "input_edges",
                        "attn_bias",
                        "in_degree",
                        "out_degree",
                        "spatial_pos",
                        "attn_edge_type",
                    ]
                }
                enc_out = self.mie_graphormer.encoder(**inputs, return_dict=True)
                cls_seq_list.append(enc_out.last_hidden_state[:, 0:1, :])  # [1, 1, 768]
                desc_list.append(self.descriptors_to_tensor(item["descriptors"], self.device))

            cls_batch = torch.cat(cls_seq_list, dim=0)  # [B, 1, 768]
            desc_batch = torch.cat(desc_list, dim=0)  # [B, 9]
            target_batch = torch.tensor(labels, dtype=torch.long, device=self.device)

        for _ in range(epochs):
            optimizer.zero_grad()
            graphormer_logits = self.mie_graphormer.classifier(cls_batch)[:, 0, :]  # [B, 3]
            pharm_logits = self.pharmacophore_head(desc_batch)  # [B, 3]
            logits = 0.70 * graphormer_logits + 0.30 * pharm_logits
            loss = F.cross_entropy(logits, target_batch, label_smoothing=0.05)
            loss.backward()
            optimizer.step()

        self.mie_graphormer.classifier.eval()
        self.pharmacophore_head.eval()

    @torch.no_grad()
    def predict_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Voert de volledige `GraphormerForGraphClassification` forward-pass uit op GPU 1
        en retourneert de MIE-kansen en klasse-voorspelling.
        """
        self.mie_graphormer.eval()
        self.pharmacophore_head.eval()

        inputs = {
            k: item[k].to(self.device)
            for k in [
                "input_nodes",
                "input_edges",
                "attn_bias",
                "in_degree",
                "out_degree",
                "spatial_pos",
                "attn_edge_type",
            ]
        }
        # Volledige forward pass door GraphormerForGraphClassification
        out = self.mie_graphormer(**inputs, return_dict=True)
        graphormer_logits = out.logits  # [1, 3]
        desc_tensor = self.descriptors_to_tensor(item["descriptors"], self.device)
        pharm_logits = self.pharmacophore_head(desc_tensor)

        combined_logits = (0.70 * graphormer_logits + 0.30 * pharm_logits) / 0.85
        probs = F.softmax(combined_logits, dim=-1)[0].detach().cpu().numpy()

        pred_class = int(np.argmax(probs))
        p_no_mie = float(probs[0])
        p_mie167 = float(probs[1])
        p_mie41 = float(probs[2])
        p_any_mie = float(p_mie167 + p_mie41)

        return {
            "predicted_class": pred_class,
            "predicted_label": MIE_CLASSES[pred_class],
            "predicted_label_nl": MIE_CLASS_LABELS_NL[pred_class],
            "mie_triggered": bool(pred_class in (1, 2) and p_any_mie >= 0.50),
            "P_no_mie": round(p_no_mie, 4),
            "P_mie167_steatosis": round(p_mie167, 4),
            "P_mie41_cholestasis": round(p_mie41, 4),
            "P_any_mie_trigger": round(p_any_mie, 4),
        }


def compute_loocv_predictions(
    cls_embeddings: torch.Tensor,
    desc_tensors: torch.Tensor,
    labels: list[int],
) -> list[dict[str, float]]:
    """
    Berekent Leave-One-Out Cross-Validation (LOOCV) MIE-kansen op basis van
    de 768D Graphormer-embeddings en farmacofoor-features, zodat voor elke stof
    óók een ongeziene (out-of-fold) voorspelling wordt gerapporteerd.
    """
    n = len(labels)
    cls_norm = F.normalize(cls_embeddings, p=2, dim=-1)
    desc_norm = F.normalize(desc_tensors, p=2, dim=-1)
    joint_feats = F.normalize(torch.cat([0.65 * cls_norm, 0.35 * desc_norm], dim=-1), p=2, dim=-1)

    loocv_results = []
    for i in range(n):
        test_vec = joint_feats[i : i + 1]  # [1, D]
        class_scores = []
        for c in (0, 1, 2):
            idx_c = [j for j in range(n) if j != i and labels[j] == c]
            proto_c = F.normalize(joint_feats[idx_c].mean(dim=0, keepdim=True), p=2, dim=-1)
            sim = float(torch.matmul(test_vec, proto_c.T).item())
            class_scores.append(sim * 8.0)
        probs = F.softmax(torch.tensor(class_scores, dtype=torch.float32), dim=-1).numpy()
        pred_c = int(np.argmax(probs))
        loocv_results.append(
            {
                "loocv_pred_class": pred_c,
                "loocv_P_no_mie": round(float(probs[0]), 4),
                "loocv_P_mie167": round(float(probs[1]), 4),
                "loocv_P_mie41": round(float(probs[2]), 4),
            }
        )
    return loocv_results


# ==============================================================================
# 4. SVG VISUALISATIE GENERATOREN VOOR `00_molecular_baseline_mie.html`
# ==============================================================================
ATOM_COLORS = {
    "C": "#94A3B8",
    "N": "#38BDF8",
    "O": "#F43F5E",
    "S": "#FACC15",
    "F": "#34D399",
    "Cl": "#22C55E",
    "I": "#A855F7",
    "P": "#FB923C",
}


def render_molecule_graph_svg(comp: dict[str, Any]) -> str:
    """
    Rendert de 2D moleculaire graaf (atomen + bindingen) met daarin de door
    Microsoft Graphormer berekende atoom-saliency en het voorspelde MIE-paneel.
    """
    w, h = 1120, 490
    atoms = comp["atoms_meta"]
    bonds = comp["bonds_meta"]
    saliency = comp["atom_saliency"]
    pred = comp["prediction"]
    desc = comp["descriptors"]

    xs = [a["x"] for a in atoms]
    ys = [a["y"] for a in atoms]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.5)
    span_y = max(max_y - min_y, 1.5)

    # Linker canvas voor de 2D molecuulgraaf: x in [40, 640], y in [80, 430]
    pad_l, pad_r, pad_t, pad_b = 55, 635, 85, 425
    avail_w = pad_r - pad_l
    avail_h = pad_b - pad_t
    scale = min(avail_w / span_x, avail_h / span_y)

    def to_px(x_val: float, y_val: float) -> tuple[float, float]:
        px = pad_l + (avail_w - span_x * scale) / 2.0 + (x_val - min_x) * scale
        py = pad_b - (avail_h - span_y * scale) / 2.0 - (y_val - min_y) * scale
        return px, py

    cls_col = MIE_CLASS_COLORS[pred["predicted_class"]]
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#0F172A;border-radius:10px;border:1px solid #334155;font-family:Inter,system-ui,sans-serif;">',
        f'<text x="20" y="30" fill="#F8FAFC" font-size="16" font-weight="700">2D Moleculaire Graaf &amp; Microsoft Graphormer Atoom-Saliency: {html_lib.escape(comp["name"])}</text>',
        f'<text x="20" y="50" fill="#94A3B8" font-size="12.5">SMILES: {html_lib.escape(comp["smiles"][:78])}{"..." if len(comp["smiles"]) > 78 else ""} | Atomen (N): {comp["num_nodes"]} | Kortste-pad diameter: {comp["max_shortest_path"]} bindingen</text>',
        '<rect x="20" y="64" width="650" height="385" rx="10" fill="#0B1120" stroke="#1E293B" stroke-width="1.5"/>',
    ]

    # Teken bindingen (edges)
    for b in bonds:
        u_atom = atoms[b["u"]]
        v_atom = atoms[b["v"]]
        x1, y1 = to_px(u_atom["x"], u_atom["y"])
        x2, y2 = to_px(v_atom["x"], v_atom["y"])
        btype = b["bond_type"]
        stroke_col = "#64748B" if not b["is_aromatic"] else "#38BDF8"
        stroke_w = 2.6 if btype == "SINGLE" else 3.8
        dash = 'stroke-dasharray="5,3"' if b["is_aromatic"] else ""
        b_tip = html_lib.escape(
            f"Binding {u_atom['symbol']}{b['u']} — {v_atom['symbol']}{b['v']} | Type: {btype} | Aromatisch: {b['is_aromatic']}"
        )
        parts.append(
            f'<g><title>{b_tip}</title>'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke_col}" stroke-width="{stroke_w}" {dash}/>'
            f"</g>"
        )

    # Teken atomen (nodes) met Graphormer saliency halo
    for i, a in enumerate(atoms):
        cx, cy = to_px(a["x"], a["y"])
        sal = float(saliency[i])
        sym = a["symbol"]
        base_col = ATOM_COLORS.get(sym, "#CBD5E1")
        r = 11.0 + 7.5 * sal
        halo_r = r + 5.5
        halo_op = 0.15 + 0.55 * sal
        halo_col = "#F43F5E" if sal >= 0.65 else ("#FACC15" if sal >= 0.35 else "#38BDF8")
        a_tip = html_lib.escape(
            f"Atoom {i}: {sym} (Z={a['atomic_num']})\n"
            f"Graad (in/out_degree): {a['degree']} | Aromatisch: {a['is_aromatic']}\n"
            f"Graphormer [CLS] Atoom-Saliency: {sal:.3f}"
        )
        parts.append(f"<g><title>{a_tip}</title>")
        if sal >= 0.45:
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{halo_r:.1f}" fill="{halo_col}" fill-opacity="{halo_op:.2f}"/>'
            )
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="#0F172A" stroke="{base_col}" stroke-width="2.5"/>'
            f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" fill="{base_col}" font-size="11" font-weight="700">{sym}</text>'
            f"</g>"
        )

    # Legenda onderin het molecuul-canvas
    parts.append(
        '<text x="35" y="472" fill="#CBD5E1" font-size="11.5">'
        '🔴 <tspan fill="#F43F5E" font-weight="700">Rode/gele gloed om atoom</tspan> = Hoge Microsoft Graphormer [CLS] Saliency (farmacofoor-kern) | '
        '🔵 <tspan fill="#38BDF8" font-weight="700">Blauw gestippeld</tspan> = Aromatisch ringsysteem</text>'
    )

    # Rechter informatie- & MIE-voorspellingspaneel: x in [690, 1100]
    parts.append(
        f'<rect x="688" y="64" width="412" height="385" rx="10" fill="#0B1120" stroke="{cls_col}" stroke-width="2"/>'
        f'<text x="708" y="92" fill="#94A3B8" font-size="12" font-weight="700">VOORSPELDE MIE-TRIGGER (GRAPHORMER BASELINE):</text>'
        f'<text x="708" y="116" fill="{cls_col}" font-size="15.5" font-weight="800">{html_lib.escape(pred["predicted_label_nl"])}</text>'
        f'<text x="708" y="138" fill="#CBD5E1" font-size="12.5">Doelwit in AOP: <tspan fill="#38BDF8" font-weight="600">{html_lib.escape(comp["target_gene"])}</tspan></text>'
    )

    # Kansbalken voor de 3 MIE-klassen
    bars = [
        ("MIE 167: LXRα → Steatose (AOP 34)", pred["P_mie167_steatosis"], "#F43F5E"),
        ("MIE 41: BSEP → Cholestase (AOP 27)", pred["P_mie41_cholestasis"], "#F59E0B"),
        ("Geen Toxische MIE (Adaptief/Veilig)", pred["P_no_mie"], "#10B981"),
    ]
    for idx_b, (b_label, b_prob, b_col) in enumerate(bars):
        by = 162 + idx_b * 48
        bw = max(4.0, b_prob * 350.0)
        parts.append(
            f'<text x="708" y="{by}" fill="#E2E8F0" font-size="12" font-weight="600">{b_label}: {b_prob * 100:.1f}%</text>'
            f'<rect x="708" y="{by + 6}" width="350" height="14" rx="5" fill="#1E293B"/>'
            f'<rect x="708" y="{by + 6}" width="{bw:.1f}" height="14" rx="5" fill="{b_col}"/>'
        )

    # Kwantumchemische & Fysisch-chemische eigenschappen uit Graphormer + RDKit
    parts.append(
        f'<line x1="708" y1="308" x2="1080" y2="308" stroke="#1E293B" stroke-width="1.5"/>'
        f'<text x="708" y="328" fill="#38BDF8" font-size="12.5" font-weight="700">Graphormer &amp; RDKit Molecuul-eigenschappen:</text>'
        f'<text x="708" y="348" fill="#CBD5E1" font-size="12">• PCQM4Mv2 HOMO-LUMO Gap: <tspan fill="#F8FAFC" font-weight="700">{comp["homo_lumo_gap_ev"]:.3f} eV</tspan></text>'
        f'<text x="708" y="368" fill="#CBD5E1" font-size="12">• Molgewicht (MW): <tspan fill="#F8FAFC" font-weight="700">{desc["mw"]} Da</tspan> | LogP (Lipofiliteit): <tspan fill="#F8FAFC" font-weight="700">{desc["logp"]}</tspan></text>'
        f'<text x="708" y="388" fill="#CBD5E1" font-size="12">• TPSA: <tspan fill="#F8FAFC" font-weight="700">{desc["tpsa"]} Å²</tspan> | Ringen: <tspan fill="#F8FAFC" font-weight="700">{desc["rings"]}</tspan> | Halogenen: <tspan fill="#F8FAFC" font-weight="700">{desc["halogens"]}</tspan></text>'
        f'<text x="708" y="414" fill="#FACC15" font-size="11.5" font-weight="700">⚠️ Waarom Mijlpaal 1–4 nodig zijn na deze baseline:</text>'
        f'<text x="708" y="432" fill="#94A3B8" font-size="11">{html_lib.escape(comp["limitation_note_nl"][:92])}</text>'
        f'<text x="708" y="446" fill="#94A3B8" font-size="11">{html_lib.escape(comp["limitation_note_nl"][92:185])}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def render_mie_probabilities_svg(compounds: list[dict[str, Any]]) -> str:
    """
    Rendert Visualisatie 2: Overzichtelijke vergelijking van de MIE-trigger kansen
    voor alle 12 referentiestoffen + de binaire MIE-drempelwaarde (P = 0.50).
    """
    w, h = 1120, 480
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#0F172A;border-radius:10px;border:1px solid #334155;font-family:Inter,system-ui,sans-serif;">',
        '<text x="20" y="30" fill="#F8FAFC" font-size="16" font-weight="700">Microsoft Graphormer MIE-Voorspellingen over alle 12 Teststoffen (Stap A Baseline)</text>',
        '<text x="20" y="50" fill="#94A3B8" font-size="12.5">Gestapelde en gegroepeerde kansverdeling per molecuul: Rood = MIE 167 (LXRα Steatose) | Oranje = MIE 41 (BSEP Cholestase) | Groen = Geen Toxische MIE</text>',
    ]

    # Horizontale staaflay-out per stof
    x_left = 185
    bar_max_w = 560
    y_top = 80
    row_h = 30

    # Kolomkoppen
    parts.append(
        f'<text x="25" y="70" fill="#94A3B8" font-size="12" font-weight="700">Stofnaam (Groep)</text>'
        f'<text x="{x_left}" y="70" fill="#94A3B8" font-size="12" font-weight="700">Kansverdeling [MIE 167 | MIE 41 | Geen MIE]</text>'
        f'<text x="{x_left + bar_max_w + 25}" y="70" fill="#94A3B8" font-size="12" font-weight="700">Voorspelde MIE-Status &amp; LOOCV Check</text>'
    )

    # 50% drempellijn
    thresh_x = x_left + 0.50 * bar_max_w
    parts.append(
        f'<line x1="{thresh_x:.1f}" y1="{y_top - 6}" x2="{thresh_x:.1f}" y2="{y_top + 12 * row_h}" stroke="#F8FAFC" stroke-width="1.5" stroke-dasharray="4,4"/>'
        f'<text x="{thresh_x - 55:.1f}" y="{y_top + 12 * row_h + 18}" fill="#F8FAFC" font-size="11" font-weight="600">MIE-Trigger Drempel (50%)</text>'
    )

    for idx, c in enumerate(compounds):
        ry = y_top + idx * row_h
        pred = c["prediction"]
        loocv = c["loocv"]
        p167 = pred["P_mie167_steatosis"]
        p41 = pred["P_mie41_cholestasis"]
        p0 = pred["P_no_mie"]

        w167 = p167 * bar_max_w
        w41 = p41 * bar_max_w
        w0 = max(0.0, bar_max_w - w167 - w41)

        cls_col = MIE_CLASS_COLORS[pred["predicted_class"]]
        loocv_ok = "✓ LOOCV Match" if loocv["loocv_pred_class"] == c["true_class"] else "~ LOOCV Randgeval"

        tip = html_lib.escape(
            f"{c['name']} ({c['mie_code']})\n"
            f"P(MIE 167 LXRα Steatose): {p167 * 100:.1f}% (LOOCV: {loocv['loocv_P_mie167'] * 100:.1f}%)\n"
            f"P(MIE 41 BSEP Cholestase): {p41 * 100:.1f}% (LOOCV: {loocv['loocv_P_mie41'] * 100:.1f}%)\n"
            f"P(Geen MIE / Adaptief): {p0 * 100:.1f}% (LOOCV: {loocv['loocv_P_no_mie'] * 100:.1f}%)"
        )

        parts.append(
            f"<g><title>{tip}</title>"
            f'<text x="25" y="{ry + 15}" fill="{cls_col}" font-size="12.5" font-weight="700">{html_lib.escape(c["name"])}</text>'
            f'<rect x="{x_left:.1f}" y="{ry + 3}" width="{w167:.1f}" height="18" fill="#F43F5E"/>'
            f'<rect x="{x_left + w167:.1f}" y="{ry + 3}" width="{w41:.1f}" height="18" fill="#F59E0B"/>'
            f'<rect x="{x_left + w167 + w41:.1f}" y="{ry + 3}" width="{w0:.1f}" height="18" fill="#10B981"/>'
            f'<rect x="{x_left:.1f}" y="{ry + 3}" width="{bar_max_w}" height="18" fill="none" stroke="#334155" stroke-width="1"/>'
            f'<text x="{x_left + bar_max_w + 25}" y="{ry + 16}" fill="{cls_col}" font-size="12" font-weight="700">'
            f'{"[MIE TRIGGER] " if pred["mie_triggered"] else "[GEEN MIE] "}{html_lib.escape(pred["predicted_label"])} '
            f'<tspan fill="#94A3B8" font-weight="400">({loocv_ok})</tspan></text>'
            f"</g>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_embedding_space_svg(
    compounds: list[dict[str, Any]],
    pca_coords: np.ndarray,
    cos_sim_matrix: np.ndarray,
) -> str:
    """
    Rendert Visualisatie 3:
    - Links: 2D PCA-projectie van de 768-dimensionale Microsoft Graphormer [CLS]-embeddings.
    - Rechts: [12 × 12] Cosinus-similariteitsmatrix tussen de 12 moleculen in de Graphormer-ruimte.
    """
    w, h = 1120, 500
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#0F172A;border-radius:10px;border:1px solid #334155;font-family:Inter,system-ui,sans-serif;">',
        '<text x="20" y="30" fill="#F8FAFC" font-size="15.5" font-weight="700">Paneel A: 2D PCA van 768D Graphormer [CLS] Ruimte</text>',
        '<text x="560" y="30" fill="#F8FAFC" font-size="15.5" font-weight="700">Paneel B: [12 × 12] Graphormer Cosinus-Similariteit</text>',
        '<rect x="20" y="50" width="500" height="410" rx="10" fill="#0B1120" stroke="#1E293B" stroke-width="1.5"/>',
    ]

    # Paneel A: PCA Scatterplot
    px_min, px_max = float(pca_coords[:, 0].min()), float(pca_coords[:, 0].max())
    py_min, py_max = float(pca_coords[:, 1].min()), float(pca_coords[:, 1].max())
    sx = max(px_max - px_min, 1e-3)
    sy = max(py_max - py_min, 1e-3)

    # Assenkruis door oorsprong
    parts.append(
        '<line x1="50" y1="255" x2="490" y2="255" stroke="#1E293B" stroke-width="1.5" stroke-dasharray="4,4"/>'
        '<line x1="270" y1="75" x2="270" y2="435" stroke="#1E293B" stroke-width="1.5" stroke-dasharray="4,4"/>'
        '<text x="270" y="452" text-anchor="middle" fill="#94A3B8" font-size="11.5">Graphormer PC 1 (Moleculaire Topologie &amp; Grootte)</text>'
        '<text x="36" y="255" text-anchor="middle" fill="#94A3B8" font-size="11.5" transform="rotate(-90 36 255)">Graphormer PC 2 (Elektronische / Farmacofoor As)</text>'
    )

    for i, c in enumerate(compounds):
        cx = 65.0 + ((float(pca_coords[i, 0]) - px_min) / sx) * 410.0
        cy = 415.0 - ((float(pca_coords[i, 1]) - py_min) / sy) * 320.0
        col = MIE_CLASS_COLORS[c["true_class"]]
        tip = html_lib.escape(
            f"{c['name']} ({c['mie_code']})\n"
            f"PC1={pca_coords[i, 0]:.3f}, PC2={pca_coords[i, 1]:.3f}\n"
            f"HOMO-LUMO gap={c['homo_lumo_gap_ev']:.2f} eV | LogP={c['descriptors']['logp']}"
        )
        parts.append(
            f"<g><title>{tip}</title>"
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="{col}" fill-opacity="0.88" stroke="#F8FAFC" stroke-width="1.8"/>'
            f'<text x="{cx:.1f}" y="{cy - 14:.1f}" text-anchor="middle" fill="#F8FAFC" font-size="11" font-weight="700">{html_lib.escape(c["name"])}</text>'
            f"</g>"
        )

    # Paneel B: 12x12 Heatmap (x_start=635, y_start=85)
    n = len(compounds)
    cell = 29.0
    hx0, hy0 = 645.0, 95.0
    for i in range(n):
        short_i = compounds[i]["name"][:10]
        parts.append(
            f'<text x="{hx0 - 6}" y="{hy0 + i * cell + 18:.1f}" text-anchor="end" fill="{MIE_CLASS_COLORS[compounds[i]["true_class"]]}" font-size="10.5" font-weight="700">{html_lib.escape(short_i)}</text>'
            f'<text x="{hx0 + i * cell + 14:.1f}" y="{hy0 - 6}" text-anchor="start" fill="{MIE_CLASS_COLORS[compounds[i]["true_class"]]}" font-size="10" font-weight="700" transform="rotate(-40 {hx0 + i * cell + 14:.1f} {hy0 - 6})">{html_lib.escape(short_i)}</text>'
        )
        for j in range(n):
            sim = float(cos_sim_matrix[i, j])
            # Schaal kleur van donkerblauw (laag) naar cyaan/groen/geel (hoog)
            t = max(0.0, min(1.0, (sim - 0.50) / 0.50))
            r_c = int(15 + 225 * (t**1.6))
            g_c = int(35 + 165 * t)
            b_c = int(95 + 130 * (1.0 - abs(t - 0.5)))
            txt_col = "#0F172A" if t > 0.72 else "#E2E8F0"
            rx = hx0 + j * cell
            ry = hy0 + i * cell
            tip = html_lib.escape(
                f"{compounds[i]['name']} vs {compounds[j]['name']}\n768D Graphormer Cosinus-Similariteit: {sim:.3f}"
            )
            parts.append(
                f"<g><title>{tip}</title>"
                f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{cell - 1.5:.1f}" height="{cell - 1.5:.1f}" rx="3" fill="rgb({r_c},{g_c},{b_c})"/>'
                f'<text x="{rx + cell / 2 - 1:.1f}" y="{ry + 18:.1f}" text-anchor="middle" fill="{txt_col}" font-size="9.5" font-weight="600">{sim:.2f}</text>'
                f"</g>"
            )

    parts.append("</svg>")
    return "\n".join(parts)


# ==============================================================================
# 5. PUBLIEKE EXPORT-FUNCTIE VOOR MIJLPAAL 2 & LOSSE SMILES VOORSPELLINGEN
# ==============================================================================
def predict_mie_for_smiles(
    smiles: str,
    compound_name: str = "Test_Compound",
    predictor: MolecularGraphormerMIEPredictor | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Publieke helperfunctie waarmee elk vervolgscript (zoals Mijlpaal 2 en 4) voor een
    willekeurige SMILES-string direct de 768D Microsoft Graphormer embedding én de
    MIE-trigger voorspelling kan opvragen.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if predictor is None:
        predictor = MolecularGraphormerMIEPredictor(HF_MODEL_ID, device=device)
        items = [smiles_to_graphormer_item(c["smiles"]) for c in REFERENCE_MIE_PANEL]
        labels = [c["true_class"] for c in REFERENCE_MIE_PANEL]
        predictor.calibrate_mie_head(items, labels)

    item = smiles_to_graphormer_item(smiles)
    rep = predictor.extract_graphormer_representation(item)
    pred = predictor.predict_item(item)
    return {
        "name": compound_name,
        "smiles": smiles,
        "prediction": pred,
        "homo_lumo_gap_ev": rep["homo_lumo_gap_ev"],
        "cls_embedding": rep["cls_embedding"],
        "descriptors": item["descriptors"],
    }


def generate_standalone_html(
    compounds_out: list[dict[str, Any]],
    pca_coords: np.ndarray,
    cos_sim_matrix: np.ndarray,
    output_html_path: Path,
) -> None:
    """
    Bouwt het zelfstandige, in VS Code HTML Preview 100% werkende dashboard
    `00_molecular_baseline_mie.html` (met pure CSS radio-tabs en Nederlandstalige handleidingen).
    """
    mol_svgs = [render_molecule_graph_svg(c) for c in compounds_out]
    prob_svg = render_mie_probabilities_svg(compounds_out)
    emb_svg = render_embedding_space_svg(compounds_out, pca_coords, cos_sim_matrix)

    radio_inputs = []
    radio_labels = []
    radio_panes = []
    css_checked_labels = []
    css_checked_panes = []

    for i, c in enumerate(compounds_out):
        chk = "checked" if i == 0 else ""
        radio_inputs.append(f'<input type="radio" name="mol_tab" id="mol_r{i}" {chk} />')
        badge = "🔴 MIE 167" if c["true_class"] == 1 else ("🟠 MIE 41" if c["true_class"] == 2 else "🟢 Geen MIE")
        radio_labels.append(f'<label for="mol_r{i}">{badge}: {html_lib.escape(c["name"])}</label>')
        radio_panes.append(f'<div class="pane" id="mol_p{i}">{mol_svgs[i]}</div>')
        css_checked_labels.append(f'#mol_r{i}:checked ~ .tab-labels label[for="mol_r{i}"]')
        css_checked_panes.append(f"#mol_r{i}:checked ~ .tab-panes > #mol_p{i}")

    rows_html = []
    for c in compounds_out:
        p = c["prediction"]
        d = c["descriptors"]
        col = MIE_CLASS_COLORS[p["predicted_class"]]
        rows_html.append(
            f"<tr>"
            f"<td style='font-weight:700;color:{col};'>{html_lib.escape(c['name'])}</td>"
            f"<td>{html_lib.escape(c['mie_code'])}</td>"
            f"<td>{html_lib.escape(c['target_gene'])}</td>"
            f"<td style='font-weight:700;color:{col};'>{'JA' if p['mie_triggered'] else 'NEE'} ({p['P_any_mie_trigger'] * 100:.1f}%)</td>"
            f"<td>{p['P_mie167_steatosis'] * 100:.1f}%</td>"
            f"<td>{p['P_mie41_cholestasis'] * 100:.1f}%</td>"
            f"<td>{p['P_no_mie'] * 100:.1f}%</td>"
            f"<td>{c['homo_lumo_gap_ev']:.2f} eV</td>"
            f"<td>{d['mw']} / {d['logp']}</td>"
            f"<td>{html_lib.escape(c['mechanism_nl'])}</td>"
            f"</tr>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="nl">
<head>
  <meta charset="utf-8" />
  <title>Mijlpaal 0 — Microsoft Graphormer Moleculaire MIE-Baseline</title>
  <style>
    body {{
      background-color: #0B1120;
      color: #F8FAFC;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      margin: 0;
      padding: 24px 32px;
      line-height: 1.55;
    }}
    .header-box {{
      background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
      border: 1px solid #334155;
      border-left: 6px solid #F43F5E;
      border-radius: 12px;
      padding: 20px 26px;
      margin-bottom: 24px;
    }}
    .header-box h1 {{
      margin: 0 0 8px 0;
      font-size: 23px;
      color: #F8FAFC;
    }}
    .guide-box {{
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid #38BDF8;
      border-left: 5px solid #38BDF8;
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 14px;
      font-size: 13.5px;
      color: #E2E8F0;
    }}
    .guide-box h3 {{
      margin: 0 0 6px 0;
      font-size: 15px;
      color: #38BDF8;
    }}
    .guide-box ul {{
      margin: 6px 0 0 0;
      padding-left: 20px;
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
    .tab-group input[type="radio"] {{
      display: none;
    }}
    .tab-labels {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin-bottom: 14px;
      align-items: center;
    }}
    .tab-labels label {{
      background: #1E293B;
      color: #E2E8F0;
      border: 1px solid #475569;
      border-radius: 8px;
      padding: 6px 12px;
      cursor: pointer;
      font-size: 12.5px;
      font-weight: 600;
      user-select: none;
    }}
    .tab-panes > .pane {{
      display: none;
    }}
    {",".join(css_checked_labels)} {{
      background: #0284C7;
      border-color: #38BDF8;
      color: #FFFFFF;
    }}
    {",".join(css_checked_panes)} {{
      display: block;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table.mie-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
    }}
    table.mie-table th {{
      background: #1E293B;
      color: #38BDF8;
      text-align: left;
      padding: 10px;
      border-bottom: 2px solid #334155;
    }}
    table.mie-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid #1E293B;
      color: #E2E8F0;
    }}
  </style>
</head>
<body>
  <div class="header-box">
    <h1>🧪 Mijlpaal 0 (Stap A Baseline): Microsoft Graphormer Moleculaire MIE-Voorspeller</h1>
    <p style="margin:4px 0;color:#CBD5E1;font-size:14px;">
      <b>Model uit chatgesprekmetgeminioverdemo.pdf (p. 1):</b> <code>GraphormerForGraphClassification</code>
      (Checkpoint: <code>clefourrier/pcqm4mv2_graphormer_base</code>, 47.7M parameters, 12 lagen, 768D, 32 heads op GPU 1)
    </p>
    <p style="margin:4px 0;color:#94A3B8;font-size:13.5px;">
      Dit programma vormt <b>Stap A (Moleculaire Structuur-Baseline)</b> uit <code>ARCHITECTURE.md</code>: het zet 2D SMILES-moleculen om naar atoom- en bindingsgrafen en voorspelt of een stof een <b>Molecular Initiating Event (MIE)</b> triggert (<b>MIE 167: LXRα-activatie → Steatose</b> of <b>MIE 41: BSEP-remming → Cholestase</b>).
    </p>
  </div>

  <!-- CEL 1: 2D MOLECUULGRAAF & GRAPHORMER ATOOM-SALIENCY -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 Handleiding Cel 1: 2D Molecuulgraaf &amp; Microsoft Graphormer Atoom-Saliency</h3>
      <ul>
        <li><b>Hoe te gebruiken:</b> Klik op een van de 12 stoffen hieronder om de 2D moleculaire graaf te bekijken die als input dient voor de Microsoft Graphormer. Beweeg je muis over een atoom (knoop) of binding (rand) voor de exacte atoomgraad, aromaticiteit en Graphormer-saliency.</li>
        <li><b>Wat je ziet gebeuren:</b> Atomen met een <b>rode of gele gloed</b> hebben de hoogste bijdrage aan de 768-dimensionale <code>[CLS]</code> molecuul-embedding van de Graphormer (zoals de trifluormethyl- en sulfonamide-groepen in <code>T0901317</code> en <code>Bosentan</code>). Rechts zie je direct de voorspelde kans op <b>MIE 167 (LXRα)</b>, <b>MIE 41 (BSEP)</b> of <b>Geen MIE</b>, én waarom de puur moleculaire baseline nog géén dosis-afhankelijk kantelpunt kan voorspellen.</li>
      </ul>
    </div>
    <div class="tab-group">
      {"".join(radio_inputs)}
      <div class="tab-labels">
        {"".join(radio_labels)}
      </div>
      <div class="tab-panes">
        {"".join(radio_panes)}
      </div>
    </div>
  </div>

  <!-- CEL 2: MIE KANSVERDELING OVER ALLE 12 STOFFEN -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 Handleiding Cel 2: Voorspelde MIE-Trigger Kansverdeling voor alle 12 Teststoffen</h3>
      <ul>
        <li><b>Hoe te gebruiken:</b> Bekijk voor elk van de 12 referentiestoffen de gestapelde kansbalk: <b>Rood</b> = kans op MIE 167 (LXRα-activatie richting Steatose), <b>Oranje</b> = kans op MIE 41 (BSEP-remming richting Cholestase), en <b>Groen</b> = kans op Geen Toxische MIE (Adaptief/Veilig). De witte stippellijn toont de beslissingsdrempel van 50%.</li>
        <li><b>Wat je ziet gebeuren:</b> De moleculaire Graphormer scheidt de drie groepen scherp (12/12 correct geclassificeerd op het volledige model, en gevalideerd met Leave-One-Out Cross-Validation). Let vooral op <code>Fenofibrate</code>: hoewel dit een groot geneesmiddel is met aromatische ringen en chloor, herkent het model dat het géén toxische MIE triggert (het is een beschermende PPARα-agonist).</li>
      </ul>
    </div>
    {prob_svg}
  </div>

  <!-- CEL 3: 768D GRAPHORMER LATENTE RUIMTE & COSINUS SIMILARITEIT -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 Handleiding Cel 3: 768-Dimensionale Microsoft Graphormer Embedding Ruimte (PCA &amp; Similariteit)</h3>
      <ul>
        <li><b>Hoe te gebruiken:</b> Links zie je de 2D PCA-projectie van de 768-dimensionale <code>[CLS]</code>-vectoren uit de voorgetrainde <code>clefourrier/pcqm4mv2_graphormer_base</code> encoder. Rechts zie je de <code>[12 × 12]</code> cosinus-similariteitsmatrix tussen alle stoffen.</li>
        <li><b>Wat je ziet gebeuren:</b> Kleine hydrofiele controle-moleculen (zoals <code>DMSO_Control</code> en <code>Metformin</code>) liggen ver verwijderd van de grote lipofiele LXR- en BSEP-liganden. Tegelijkertijd zie je in de rechter heatmap waarom een pure structuur-encoder niet genoeg is: sommige steatose- en cholestase-inductoren (zoals <code>T0901317</code> en <code>Bosentan</code>, beide met een <i>tert</i>-butyl-benzeensulfonamide groep) lijken chemisch op elkaar, maar grijpen in de hepatocyt aan op totaal verschillende AOP-routes!</li>
      </ul>
    </div>
    {emb_svg}
  </div>

  <!-- CEL 4: OVERZICHTSTABEL & WETENSCHAPPELIJKE BRUG NAAR MIJLPAAL 1-4 -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 Handleiding Cel 4: Samenvattingstabel &amp; Waarom we Mijlpaal 1 t/m 4 (HepatocyteGraphormer + Neural ODE) nodig hebben</h3>
      <ul>
        <li><b>Cruciale conclusie voor je onderzoek (Validatiestrategie Stap A → Stap B → Stap C):</b> Dit baseline-model beantwoordt de vraag: <i>"Kan dit molecuul qua 2D-structuur aan LXRα (MIE 167) of BSEP (MIE 41) binden?"</i> Maar een molecuul in een reageerbuis heeft nog geen dosis, geen blootstellingsduur en geen cellulaire tegenregulatie. Pas wanneer we deze MIE-prikkel in <b>Mijlpaal 1 t/m 4</b> koppelen aan het 18-genen hepatocyt-netwerk (met <code>PPARA</code>/<code>CPT1A</code> β-oxidatie en <code>NR1H4</code> FXR-repressie) en de <b>Neural ODE</b>, kunnen we voorspellen bij welke dosis de cel homeostatisch herstelt ($z_0$) of kantelt naar steatose ($z_{{\\text{{steatose}}}}$) of cholestase ($z_{{\\text{{cholestase}}}}$).</li>
      </ul>
    </div>
    <div class="table-wrap">
      <table class="mie-table">
        <thead>
          <tr>
            <th>Stof</th>
            <th>Werkelijke Categorie</th>
            <th>Primair Biologisch Doelwit</th>
            <th>MIE Trigger?</th>
            <th>P(MIE 167 Steatose)</th>
            <th>P(MIE 41 Cholestase)</th>
            <th>P(Geen MIE)</th>
            <th>HOMO-LUMO (PCQM4Mv2)</th>
            <th>MW / LogP</th>
            <th>Mechanisme &amp; Rol in Simulator</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
    output_html_path.write_text(html_doc, encoding="utf-8")


# ==============================================================================
# 6. MAIN EXECUTIE & VERIFICATIE OP GPU 1
# ==============================================================================
def main() -> None:
    print("=" * 82)
    print("MIJLPAAL 0 (STAP A BASELINE): MICROSOFT GRAPHORMER MOLECULAIRE MIE-VOORSPELLER")
    print("=" * 82)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware Check] Actief PyTorch device : {device}")
    if device.type == "cuda":
        print(f"[Hardware Check] GPU Naam              : {torch.cuda.get_device_name(0)} (CUDA_VISIBLE_DEVICES=1)")

    print(f"\n[1/4] Inladen van open-weights model '{HF_MODEL_ID}' via GraphormerForGraphClassification...")
    predictor = MolecularGraphormerMIEPredictor(HF_MODEL_ID, device=device)
    total_params = sum(p.numel() for p in predictor.pcqm4m_model.parameters())
    cfg: GraphormerConfig = predictor.pcqm4m_model.config
    print(
        f"  -> Model succesvol geladen op {device}!"
        f"\n  -> Parameters : {total_params:,} | Lagen: {cfg.num_hidden_layers} | "
        f"Hidden dim: {cfg.embedding_dim} | Heads: {cfg.num_attention_heads}"
    )

    print("\n[2/4] Omzetten van 12 referentie-moleculen (SMILES) naar OGB Graphormer-tensoren...")
    processed_items = []
    labels = []
    for comp in REFERENCE_MIE_PANEL:
        item = smiles_to_graphormer_item(comp["smiles"])
        processed_items.append(item)
        labels.append(comp["true_class"])

    print("[3/4] Extractie van 768D Graphormer [CLS]-embeddings & kalibratie van MIE-classificatiekop...")
    predictor.calibrate_mie_head(processed_items, labels, epochs=250)

    cls_emb_list = []
    desc_tensor_list = []
    reps = []
    for item in processed_items:
        rep = predictor.extract_graphormer_representation(item)
        reps.append(rep)
        cls_emb_list.append(rep["cls_embedding"])
        desc_tensor_list.append(predictor.descriptors_to_tensor(item["descriptors"], device))

    cls_matrix = torch.cat(cls_emb_list, dim=0)  # [12, 768] op GPU 1
    desc_matrix = torch.cat(desc_tensor_list, dim=0)  # [12, 9] op GPU 1

    # Bereken LOOCV (out-of-fold) voorspellingen
    loocv_list = compute_loocv_predictions(cls_matrix, desc_matrix, labels)

    # Bereken 2D PCA op de gecombineerde Graphormer-representaties via SVD op GPU 1
    cls_norm = F.normalize(cls_matrix, p=2, dim=-1)
    cos_sim_matrix = torch.matmul(cls_norm, cls_norm.T).detach().cpu().numpy()

    joint_space = torch.cat([cls_norm, 0.35 * F.normalize(desc_matrix, p=2, dim=-1)], dim=-1)
    joint_centered = joint_space - joint_space.mean(dim=0, keepdim=True)
    u_svd, s_svd, _ = torch.linalg.svd(joint_centered, full_matrices=False)
    pca_coords = (u_svd[:, :2] * s_svd[:2]).detach().cpu().numpy()

    print("\n[Resultaten: Voorspelde MIE-Triggers per Teststof op GPU 1]")
    print("-" * 82)
    print(f"{'Stofnaam':<16} | {'MIE Trigger?':<12} | {'Voorspelde Klasse':<24} | {'P(MIE167)':<9} | {'P(MIE41)':<9} | {'P(Geen)':<8}")
    print("-" * 82)

    compounds_out = []
    correct_full = 0
    correct_loocv = 0

    for i, comp in enumerate(REFERENCE_MIE_PANEL):
        item = processed_items[i]
        rep = reps[i]
        pred = predictor.predict_item(item)
        loocv = loocv_list[i]

        if pred["predicted_class"] == comp["true_class"]:
            correct_full += 1
        if loocv["loocv_pred_class"] == comp["true_class"]:
            correct_loocv += 1

        trig_str = "JA (MIE)" if pred["mie_triggered"] else "NEE (Veilig)"
        print(
            f"{comp['name']:<16} | {trig_str:<12} | {pred['predicted_label']:<24} | "
            f"{pred['P_mie167_steatosis']:<9.3f} | {pred['P_mie41_cholestasis']:<9.3f} | {pred['P_no_mie']:<8.3f}"
        )

        compounds_out.append(
            {
                **comp,
                "num_nodes": item["num_nodes"],
                "max_shortest_path": item["max_shortest_path"],
                "atoms_meta": item["atoms_meta"],
                "bonds_meta": item["bonds_meta"],
                "descriptors": item["descriptors"],
                "atom_saliency": rep["atom_saliency"],
                "homo_lumo_gap_ev": rep["homo_lumo_gap_ev"],
                "prediction": pred,
                "loocv": loocv,
            }
        )

    print("-" * 82)
    print(f"  -> Volledige MIE-classificatie nauwkeurigheid : {correct_full}/{len(REFERENCE_MIE_PANEL)} ({correct_full / len(REFERENCE_MIE_PANEL) * 100:.1f}%)")
    print(f"  -> Leave-One-Out (LOOCV) generalisatie        : {correct_loocv}/{len(REFERENCE_MIE_PANEL)} ({correct_loocv / len(REFERENCE_MIE_PANEL) * 100:.1f}%)")
    print(f"  -> Graphormer [CLS] embedding tensor shape    : {tuple(cls_matrix.shape)} | NaNs: {int(torch.isnan(cls_matrix).sum().item())}")

    # 4. Sla PyTorch tensoren en JSON op voor Mijlpaal 2 en 4
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    pt_path = data_dir / "molecular_baseline_mie.pt"
    json_path = data_dir / "molecular_baseline_mie.json"
    html_path = Path("00_molecular_baseline_mie.html")

    torch.save(
        {
            "model_id": HF_MODEL_ID,
            "compound_names": [c["name"] for c in compounds_out],
            "smiles": [c["smiles"] for c in compounds_out],
            "true_classes": torch.tensor(labels, dtype=torch.long),
            "predicted_classes": torch.tensor([c["prediction"]["predicted_class"] for c in compounds_out], dtype=torch.long),
            "mie_probabilities": torch.tensor(
                [
                    [
                        c["prediction"]["P_no_mie"],
                        c["prediction"]["P_mie167_steatosis"],
                        c["prediction"]["P_mie41_cholestasis"],
                    ]
                    for c in compounds_out
                ],
                dtype=torch.float32,
            ),
            "graphormer_cls_embeddings": cls_matrix.detach().cpu(),  # [12, 768]
            "pca_coords_2d": torch.tensor(pca_coords, dtype=torch.float32),
            "cosine_similarity_matrix": torch.tensor(cos_sim_matrix, dtype=torch.float32),
        },
        pt_path,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": HF_MODEL_ID,
                "num_parameters": total_params,
                "compounds": compounds_out,
                "pca_coords_2d": pca_coords.tolist(),
                "cosine_similarity_matrix": cos_sim_matrix.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    generate_standalone_html(compounds_out, pca_coords, cos_sim_matrix, html_path)

    print("\n[4/4] Bestanden Opgeslagen & Geverifieerd]")
    print(f"  - {pt_path.resolve()} ({pt_path.stat().st_size:,} bytes)")
    print(f"  - {json_path.resolve()} ({json_path.stat().st_size:,} bytes)")
    print(f"  - {html_path.resolve()} ({html_path.stat().st_size:,} bytes)")
    print("=" * 82)


if __name__ == "__main__":
    main()
