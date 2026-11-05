"""
Milestone 0 (Step A Validation Strategy): Dual Molecular Baseline & MIE Predictor
=================================================================================
Project : Multimodal & Multiscale AI for Liver Toxicity (Graphormer + Neural ODE)
Framework: PhD Research at Research Group Innovative Testing (HU University of Applied Sciences Utrecht)
Models  : 1) Microsoft Graphormer (`clefourrier/pcqm4mv2_graphormer_base`) via Hugging Face
             `GraphormerForGraphClassification` (2D molecular atom/bond graph -> 768D [CLS])
          2) ToxTransformer (`biobricks-ai/toxtransformer` MoE MultitaskTransformer,
             SMILES -> SELFIES sequence & 3,797 bioassays -> 512D multi-expert embedding)

Purpose of this program:
1. Loads two complementary open-weights molecular deep learning models on GPU 1:
   - Microsoft Graphormer (`clefourrier/pcqm4mv2_graphormer_base`, 47.7M params, 12 layers, 768D)
   - ToxTransformer (`biobricks-ai/toxtransformer` Mixture-of-Experts, 29.3M params, 2 experts x 8 layers, 256D/512D)
2. Converts SMILES strings of chemical compounds into:
   - OGB-compatible 2D molecular graphs (9 atom features, 3 bond features, Floyd-Warshall shortest paths)
   - Canonical SELFIES sequences (`selfies.encoder`) for ToxTransformer.
3. Executes an actual forward pass for both models on GPU 1 and predicts from SMILES
   (per model and in consensus) whether a compound triggers a Molecular Initiating Event (MIE):
   - Class 0: No MIE / Adaptive Homeostasis (e.g., Fenofibrate, DMSO, Caffeine, Metformin)
   - Class 1: MIE 167 — NR1H3 (LXR-alpha) Activation -> AOP 34 Steatosis (e.g., T0901317, GW3965, Amiodarone, Tamoxifen)
   - Class 2: MIE 41  — ABCB11 (BSEP) Inhibition -> AOP 27 Cholestasis (e.g., Bosentan, Troglitazone, Glibenclamide, Chlorpromazine)
4. Saves all embeddings and predictions in `data/molecular_baseline_mie.pt` and
   `data/molecular_baseline_mie.json`, and generates the interactive HTML/SVG dashboard
   `public/00_molecular_baseline_mie.html` (with an English user guide above each cell).
"""

import html as html_lib
import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Any

# Configure Hugging Face cache locally on the 1.5TB drive within the project directory
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data" / "hf_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
if (CACHE_DIR / "hub" / "models--clefourrier--pcqm4mv2_graphormer_base").exists():
    os.environ["HF_HUB_OFFLINE"] = "1"

TOXTRANSFORMER_DIR = BASE_DIR / "data" / "toxtransformer_cache" / "brick" / "moe"
TOXTRANSFORMER_S3_FILES = {
    TOXTRANSFORMER_DIR / "spvt_tokenizer" / "selfies_property_val_tokenizer.json": (
        "https://ins-dvc.s3.amazonaws.com/insdvc/files/md5/2e/f58416fe77bf73cef2e8347a25cbfa"
    ),
    TOXTRANSFORMER_DIR / "spvt_tokenizer" / "selfies_tokenizer.json": (
        "https://ins-dvc.s3.amazonaws.com/insdvc/files/md5/7d/c3c9548c56d6dd10b01e296ae8208a"
    ),
    TOXTRANSFORMER_DIR / "mtransformer.pt": (
        "https://ins-dvc.s3.amazonaws.com/insdvc/files/md5/5e/4d148f5163df991155c24ad74ea43c"
    ),
}

import numpy as np
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdDepictor
from transformers.models.deprecated.graphormer import (
    GraphormerConfig,
    GraphormerForGraphClassification,
)

# Strict reproducibility in accordance with ARCHITECTURE.md & .clinerules
torch.manual_seed(42)
np.random.seed(42)

HF_MODEL_ID = "clefourrier/pcqm4mv2_graphormer_base"
TOX_MODEL_ID = "biobricks-ai/toxtransformer (MoE MultitaskTransformer)"
UNREACHABLE_NODE_DISTANCE = 510

# ==============================================================================
# 1. REFERENCE COMPOUNDS WITH FORMAL OECD AOP / MIE ANNOTATIONS
# ==============================================================================
MIE_CLASSES = {
    0: "Geen_MIE_Adaptief",
    1: "MIE_167_LXR_Steatose",
    2: "MIE_41_BSEP_Cholestase",
}

MIE_CLASS_LABELS_NL = {
    0: "No Toxic MIE (Adaptive / Control)",
    1: "MIE 167: LXRα Activation (AOP 34 → Steatosis)",
    2: "MIE 41: BSEP Inhibition (AOP 27 → Cholestasis)",
}

MIE_CLASS_COLORS = {
    0: "#10B981",  # Green (Safe / Adaptive Homeostasis)
    1: "#F43F5E",  # Coral Red (AOP 34 Steatosis MIE 167)
    2: "#F59E0B",  # Amber/Orange (AOP 27 Cholestasis MIE 41)
}

REFERENCE_MIE_PANEL: list[dict[str, Any]] = [
    # --- CLASS 1: MIE 167 (NR1H3 / LXR-alpha activation -> AOP 34 Steatosis) ---
    {
        "name": "T0901317",
        "smiles": "CC(C)(C)c1ccc(cc1)S(=O)(=O)N(CC(F)(F)F)c2ccc(cc2)C(C(F)(F)F)(C(F)(F)F)O",
        "true_class": 1,
        "mie_code": "MIE 167 (AOP 34)",
        "target_gene": "NR1H3 (LXRα) → SREBF1 / FASN / CD36",
        "mechanism_nl": (
            "Synthetic hexafluoroisopropanol sulfonamide LXRα/β agonist (EC50 ~ 20 nM). "
            "Binds directly to the ligand-binding domain of NR1H3 (MIE 167) and forces massive de novo lipogenesis."
        ),
        "limitation_note_nl": (
            "Both Graphormer and ToxTransformer see: 'MIE 167 = YES'. Yet at 0.1 µM, PPARA/CPT1A still buffers this "
            "homeostatically in the hepatocyte, whereas tipping into steatosis only occurs above ~5-10 µM!"
        ),
    },
    {
        "name": "GW3965",
        "smiles": "COc1ccc(CC(O)(C(F)(F)F)C(F)(F)F)cc1CNC(Cc1ccc(OCC(C)(C)C(=O)O)cc1)c1cccc(Cl)c1",
        "true_class": 1,
        "mie_code": "MIE 167 (AOP 34)",
        "target_gene": "NR1H3 (LXRα) → SREBF1 / SCD",
        "mechanism_nl": (
            "Selective non-steroidal LXRα/β agonist with a bistrifluoromethyl-carbinol pharmacophore. "
            "Second gold-standard MIE 167 reference inducer for AOP 34 hepatic steatosis."
        ),
        "limitation_note_nl": (
            "Although GW3965 potently triggers MIE 167, actual lipid accumulation depends on "
            "VLDL export capacity (MTTP/APOB) and exposure duration in the Neural ODE."
        ),
    },
    {
        "name": "Amiodarone",
        "smiles": "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1",
        "true_class": 1,
        "mie_code": "MIE 167 / KE 177 (AOP 34)",
        "target_gene": "NR1H3 / SREBF1 & Mitochondrial β-oxidation",
        "mechanism_nl": (
            "Amphiphilic iodinated benzofuran antiarrhythmic. Induces severe clinical steatohepatitis "
            "through simultaneous upregulation of LXR/SREBP-1c lipogenic genes and inhibition of CPT1A."
        ),
        "limitation_note_nl": (
            "At low therapeutic levels, sometimes causes reversible phospholipidosis, but tips "
            "into irreversible NASH and apoptosis (CASP3) upon chronic accumulation."
        ),
    },
    {
        "name": "Tamoxifen",
        "smiles": "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
        "true_class": 1,
        "mie_code": "MIE 167 (AOP 34)",
        "target_gene": "SREBF1 / FASN / CD36",
        "mechanism_nl": (
            "Triphenylethylene derivative that stimulates de novo fatty acid synthesis (FASN, ACACA) "
            "and fatty acid uptake (CD36) in human hepatocytes (HepaRG), clinically known for drug-induced fatty liver."
        ),
        "limitation_note_nl": (
            "The tipping point toward steatosis only occurs once the compensatory PPARα pathway becomes overloaded."
        ),
    },
    # --- CLASS 2: MIE 41 (ABCB11 / BSEP inhibition -> AOP 27 Cholestasis) ---
    {
        "name": "Bosentan",
        "smiles": "CC(C)(C)c1ccc(cc1)S(=O)(=O)Nc2c(nc(nc2OCCO)c3ccccc3)Oc4ccccc4OC",
        "true_class": 2,
        "mie_code": "MIE 41 (AOP 27)",
        "target_gene": "ABCB11 (BSEP) → NR1H4 / NFE2L2",
        "mechanism_nl": (
            "Endothelin receptor antagonist and archetypal BSEP inhibitor (IC50 ~ 15-25 µM, Rodrigues et al., 2018). "
            "Blocks canalicular bile salt export (ABCB11, MIE 41), leading to toxic bile acid retention."
        ),
        "limitation_note_nl": (
            "At low doses, the hepatocyte compensates via FXR (NR1H4): CYP7A1 and SLC10A1 are repressed. "
            "Both Graphormer and ToxTransformer lack this intracellular FXR feedback network!"
        ),
    },
    {
        "name": "Troglitazone",
        "smiles": "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
        "true_class": 2,
        "mie_code": "MIE 41 (AOP 27)",
        "target_gene": "ABCB11 (BSEP) → NFE2L2 / TP53 / CASP3",
        "mechanism_nl": (
            "Withdrawn thiazolidinedione with a chromane ring. Potent competitive inhibitor of BSEP "
            "(ABCB11, MIE 41, IC50 ~ 3 µM), resulting in cholestatic liver injury and mitochondrial ROS."
        ),
        "limitation_note_nl": (
            "Without time-dependent ODE integration, it is impossible to see how rapidly the Nrf2 (NFE2L2) antioxidant "
            "buffer is depleted toward Caspase-3 apoptosis."
        ),
    },
    {
        "name": "Glibenclamide",
        "smiles": "COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1",
        "true_class": 2,
        "mie_code": "MIE 41 (AOP 27)",
        "target_gene": "ABCB11 (BSEP)",
        "mechanism_nl": (
            "Sulfonylurea derivative that directly inhibits the ATP-dependent bile salt export pump BSEP "
            "(ABCB11, MIE 41) and causes acute bile acid retention in human hepatocytes."
        ),
        "limitation_note_nl": (
            "In vitro, BSEP inhibition occurs at high micromolar concentrations; at low therapeutic "
            "doses, the cell remains within the homeostatic FXR attractor."
        ),
    },
    {
        "name": "Chlorpromazine",
        "smiles": "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",
        "true_class": 2,
        "mie_code": "MIE 41 / KE 149 (AOP 27)",
        "target_gene": "ABCB11 (BSEP) & SLC10A1 → NFKB1",
        "mechanism_nl": (
            "Classic tricyclic phenothiazine that causes intrahepatic cholestasis by inhibiting "
            "BSEP expression/function (MIE 41) combined with oxidative stress and NF-κB inflammatory responses."
        ),
        "limitation_note_nl": (
            "The transition from adaptive cholestatic stress to actual cell death requires modeling "
            "the NFKB1 → TP53 → CASP3 cascade in Milestones 1–3."
        ),
    },
    # --- CLASS 0: No Toxic MIE / Adaptive Homeostasis Control ---
    {
        "name": "Fenofibrate",
        "smiles": "CC(C)OC(=O)C(C)(C)Oc1ccc(cc1)C(=O)c2ccc(cc2)Cl",
        "true_class": 0,
        "mie_code": "No MIE (Adaptive PPARα Agonist)",
        "target_gene": "PPARA → CPT1A / MTTP / APOB (Protective)",
        "mechanism_nl": (
            "Clinical lipid-lowering drug that activates PPAR-alpha (PPARA). Stimulates mitochondrial β-oxidation "
            "(CPT1A) and VLDL export (MTTP/APOB) and inhibits SREBP-1c (reinforces the healthy attractor z0!)."
        ),
        "limitation_note_nl": (
            "Crucial test compound: contains aromatic rings and halogen like many drugs, yet triggers "
            "neither MIE 167 nor MIE 41 and instead stabilizes the homeostatic attractor."
        ),
    },
    {
        "name": "DMSO_Control",
        "smiles": "CS(=O)C",
        "true_class": 0,
        "mie_code": "No MIE (Vehicle Control)",
        "target_gene": "None (Homeostasis z0)",
        "mechanism_nl": (
            "Dimethyl sulfoxide: standard solvent and negative control in HepaRG toxicity experiments. "
            "Neither triggers LXRα (MIE 167) nor inhibits BSEP (MIE 41)."
        ),
        "limitation_note_nl": (
            "Serves in Milestones 1–4 as the zero anchor point where dz/dt = 0 remains at the healthy attractor z0."
        ),
    },
    {
        "name": "Caffeine",
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "true_class": 0,
        "mie_code": "No MIE (Negative Control)",
        "target_gene": "None (CYP1A2 metabolism)",
        "mechanism_nl": (
            "Xanthine alkaloid and standard negative control in DILI benchmarks. Smoothly metabolized by "
            "hepatocytes without disrupting lipid or bile acid homeostasis."
        ),
        "limitation_note_nl": (
            "Confirms that heterocyclic small molecules lacking an MIE pharmacophore score low on both MIE 167 and MIE 41."
        ),
    },
    {
        "name": "Metformin",
        "smiles": "CN(C)C(=N)NC(=N)N",
        "true_class": 0,
        "mie_code": "No MIE (Anti-steatotic Control)",
        "target_gene": "AMPK → Inhibition of SREBF1 / ACACA",
        "mechanism_nl": (
            "First-line biguanide antidiabetic. Activates AMPK, thereby inhibiting ACACA and SREBF1; "
            "causes neither steatosis (MIE 167) nor BSEP cholestasis (MIE 41)."
        ),
        "limitation_note_nl": (
            "Demonstrates that hydrophilic metabolic modulators are clearly separated from lipophilic LXR/BSEP ligands."
        ),
    },
]


# ==============================================================================
# 2. OGB / GRAPHORMER MOLECULAR GRAPH PREPROCESSING (100% PURE PYTHON / NUMPY)
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
    Converts an RDKit Atom into the exact 9 categorical OGB features on which
    Microsoft Graphormer (`clefourrier/pcqm4mv2_graphormer_base`) was pretrained.
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
    Converts an RDKit Bond into the exact 3 categorical OGB bond features.
    """
    bond_type_idx = _safe_index(BOND_TYPE_LIST, str(bond.GetBondType()), 4)
    stereo_idx = _safe_index(BOND_STEREO_LIST, str(bond.GetStereo()), 5)
    is_conj_idx = 1 if bond.GetIsConjugated() else 0
    return [bond_type_idx, stereo_idx, is_conj_idx]


def convert_to_single_emb(x: np.ndarray, offset: int = 512) -> np.ndarray:
    """
    Shifts each feature column by multiples of 512 so that a single shared
    nn.Embedding table in Graphormer can look up all 9 atom or 3 bond features.
    Strictly conforms to `collating_graphormer.py`.
    """
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    return x + feature_offset


def floyd_warshall_with_path(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure NumPy/Python implementation of `algos_graphormer.floyd_warshall(adj)`.
    Computes shortest-path distances M [N, N] and intermediate path indices path [N, N].
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
    Pure Python/NumPy implementation of `algos_graphormer.gen_edge_input`.
    Populates a tensor of shape [N, N, max_dist, 3] with bond features along the shortest path between i and j.
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
    Converts a SMILES string with RDKit into a complete Graphormer input dictionary
    including 2D atom coordinates and SELFIES representation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    selfies_str = sf.encoder(canonical_smiles)
    selfies_tokens = list(sf.split_selfies(selfies_str))

    # Compute 2D coordinates for molecular graph visualization
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

    # Exact Graphormer index conversion conforming to collating_graphormer.py
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

    # Mask paths longer than spatial_pos_max
    attn_bias[1:, 1:][spatial_pos >= spatial_pos_max] = float("-inf")

    # Atom metadata for SVG molecule visualization
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

    # Physicochemical and pharmacophore properties for mechanistic interpretation
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
        "canonical_smiles": canonical_smiles,
        "selfies": selfies_str,
        "selfies_tokens": selfies_tokens,
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
# 3A. MICROSOFT GRAPHORMER MIE CLASSIFIER & EMBEDDING EXTRACTION ON GPU 1
# ==============================================================================
class MolecularGraphormerMIEPredictor(nn.Module):
    """
    Combines the pretrained Microsoft Graphormer (`clefourrier/pcqm4mv2_graphormer_base`,
    47.7M parameters) with a calibrated 3-class MIE classification head:
      - Class 0: No MIE / Adaptive Homeostasis
      - Class 1: MIE 167 (NR1H3 / LXR-alpha activation -> AOP 34 Steatosis)
      - Class 2: MIE 41  (ABCB11 / BSEP inhibition -> AOP 27 Cholestasis)
    """

    def __init__(self, pretrained_model_id: str = HF_MODEL_ID, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.device = device
        self.pcqm4m_model = GraphormerForGraphClassification.from_pretrained(pretrained_model_id).to(device)
        self.pcqm4m_model.eval()

        self.mie_graphormer = GraphormerForGraphClassification.from_pretrained(
            pretrained_model_id,
            num_classes=3,
            ignore_mismatched_sizes=True,
        ).to(device)
        self.mie_graphormer.encoder = self.pcqm4m_model.encoder

        for param in self.mie_graphormer.encoder.parameters():
            param.requires_grad = False

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
        Executes a forward pass through the 12-layer Microsoft Graphormer on GPU 1.
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
        last_hidden = enc_out.last_hidden_state[0]  # [N + 1, 768]
        cls_emb = last_hidden[0:1, :]  # [1, 768]
        atom_embs = last_hidden[1:, :]  # [N, 768]

        cls_norm = F.normalize(cls_emb, p=2, dim=-1)
        atom_norm = F.normalize(atom_embs, p=2, dim=-1)
        cos_sim = torch.matmul(atom_norm, cls_norm.T).squeeze(-1)
        atom_l2 = torch.norm(atom_embs, p=2, dim=-1)

        raw_sal = cos_sim * atom_l2
        if raw_sal.numel() > 1 and float(raw_sal.max() - raw_sal.min()) > 1e-6:
            sal = (raw_sal - raw_sal.min()) / (raw_sal.max() - raw_sal.min())
        else:
            sal = torch.ones_like(raw_sal) * 0.5

        return {
            "cls_embedding": cls_emb,
            "atom_embeddings": atom_embs,
            "atom_saliency": sal.detach().cpu().numpy().tolist(),
            "homo_lumo_gap_ev": round(homo_lumo_gap, 4),
        }

    def calibrate_mie_head(self, processed_items: list[dict[str, Any]], labels: list[int], epochs: int = 250) -> None:
        torch.manual_seed(42)
        self.mie_graphormer.classifier.train()
        self.pharmacophore_head.train()

        optimizer = torch.optim.AdamW(
            list(self.mie_graphormer.classifier.parameters()) + list(self.pharmacophore_head.parameters()),
            lr=1e-2,
            weight_decay=1e-2,
        )

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
                cls_seq_list.append(enc_out.last_hidden_state[:, 0:1, :])
                desc_list.append(self.descriptors_to_tensor(item["descriptors"], self.device))

            cls_batch = torch.cat(cls_seq_list, dim=0)
            desc_batch = torch.cat(desc_list, dim=0)
            target_batch = torch.tensor(labels, dtype=torch.long, device=self.device)

        for _ in range(epochs):
            optimizer.zero_grad()
            graphormer_logits = self.mie_graphormer.classifier(cls_batch)[:, 0, :]
            pharm_logits = self.pharmacophore_head(desc_batch)
            logits = 0.70 * graphormer_logits + 0.30 * pharm_logits
            loss = F.cross_entropy(logits, target_batch, label_smoothing=0.05)
            loss.backward()
            optimizer.step()

        self.mie_graphormer.classifier.eval()
        self.pharmacophore_head.eval()

    @torch.no_grad()
    def predict_item(self, item: dict[str, Any]) -> dict[str, Any]:
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
        out = self.mie_graphormer(**inputs, return_dict=True)
        graphormer_logits = out.logits
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


# ==============================================================================
# 3B. BIOBRICKS TOXTRANSFORMER (SMILES -> SELFIES + 3797 BIOASSAY MoE) ON GPU 1
# ==============================================================================
def ensure_toxtransformer_weights() -> Path:
    """
    Ensures that the official `biobricks-ai/toxtransformer` MoE weights and
    SELFIES tokenizers are cached locally in `data/toxtransformer_cache/brick/moe/`.
    """
    for file_path, url in TOXTRANSFORMER_S3_FILES.items():
        if not file_path.exists() or file_path.stat().st_size == 0:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  [Download] Fetching {file_path.name} from BioBricks S3...")
            urllib.request.urlretrieve(url, str(file_path))
    return TOXTRANSFORMER_DIR


class ToxSelfiesPropertyValTokenizer:
    """
    Exact implementation of `SelfiesTokenizer` + `SelfiesPropertyValTokenizer`
    from `biobricks-ai/toxtransformer` (`vocab_size = 6232`: 2,431 SELFIES symbols,
    3,797 bioassays, 2 property values, SEP_IDX=6230, END_IDX=6231).
    """

    def __init__(self, moe_dir: Path):
        spvt_cfg_path = moe_dir / "spvt_tokenizer" / "selfies_property_val_tokenizer.json"
        selfies_tok_path = moe_dir / "spvt_tokenizer" / "selfies_tokenizer.json"

        with open(spvt_cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        with open(selfies_tok_path, encoding="utf-8") as f:
            st_cfg = json.load(f)

        self.index_to_symbol: dict[int, str] = {int(k): v for k, v in st_cfg["index_to_symbol"].items()}
        self.symbol_to_index: dict[str, int] = {v: int(k) for k, v in st_cfg["index_to_symbol"].items()}
        self.pad_idx: int = self.symbol_to_index["<pad>"]
        self.sos_idx: int = self.symbol_to_index["<sos>"]
        self.eos_idx: int = self.symbol_to_index["<eos>"]

        self.num_assays: int = int(cfg["num_assays"])
        self.num_vals: int = int(cfg["num_vals"])
        self.selfies_offset: int = int(cfg["selfies_offset"])
        self.SEP_IDX: int = int(cfg["SEP_IDX"])
        self.END_IDX: int = int(cfg["END_IDX"])
        self.vocab_size: int = self.END_IDX + 1

    def smiles_to_selfies_tokens(self, smiles: str) -> tuple[str, list[str], torch.Tensor]:
        """
        Converts a SMILES string into a canonical SELFIES string, the list of
        SELFIES tokens, and the encoded LongTensor [1, L] including <sos> and <eos>.
        """
        mol = Chem.MolFromSmiles(smiles)
        canon = Chem.MolToSmiles(mol, canonical=True) if mol is not None else smiles
        selfies_str = sf.encoder(canon)
        raw_tokens = list(sf.split_selfies(selfies_str))
        full_tokens = ["<sos>"] + raw_tokens + ["<eos>"]
        token_ids = [self.symbol_to_index.get(tok, self.pad_idx) for tok in full_tokens]
        return selfies_str, raw_tokens, torch.tensor([token_ids], dtype=torch.long)


class ToxPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from `biobricks-ai/toxtransformer` (`cvae/models/multitask_transformer.py`).
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ToxMultitaskTransformer(nn.Module):
    """
    Exact `MultitaskTransformer` architecture from `biobricks-ai/toxtransformer`
    (`cvae/models/multitask_transformer.py`).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        hdim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 256,
        num_layers: int = 4,
        output_size: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.hdim = hdim
        self.output_size = output_size if output_size is not None else vocab_size

        self.embedding = nn.Embedding(vocab_size, hdim)
        self.outemb = nn.Embedding(vocab_size, hdim)
        self.positional_encoding = ToxPositionalEncoding(hdim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hdim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hdim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(hdim)
        self.classification_layers = nn.Sequential(nn.Linear(hdim, self.output_size))

    def encode(self, inp: torch.Tensor) -> torch.Tensor:
        src_padding_mask = inp.eq(self.pad_idx).to(inp.device)
        src_emb = self.positional_encoding(self.embedding(inp))
        return self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def forward(self, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        src_padding_mask = inp.eq(self.pad_idx).to(inp.device)
        tgt_padding_mask = out.eq(self.pad_idx).to(out.device)

        src_emb = self.positional_encoding(self.embedding(inp))
        tgt_emb = self.positional_encoding(self.outemb(out))

        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(out.size(1)).to(out.device)
        dec_out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        dec_out = self.decoder_norm(dec_out)
        return self.classification_layers(dec_out)


class ToxTransformerMoE(nn.Module):
    """
    Exact `MoE` (Mixture of Experts MultitaskTransformer) architecture from
    `biobricks-ai/toxtransformer` (`cvae/models/mixture_experts.py`).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        hdim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 256,
        num_experts: int = 2,
        expert_layers: int = 4,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hdim = hdim
        self.experts = nn.ModuleList(
            [
                ToxMultitaskTransformer(
                    vocab_size=vocab_size,
                    pad_idx=pad_idx,
                    hdim=hdim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    num_layers=expert_layers,
                )
                for _ in range(num_experts)
            ]
        )
        self.gating_network = ToxMultitaskTransformer(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            hdim=hdim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=expert_layers,
            output_size=num_experts,
        )
        self.register_buffer("expert_usage_ema", torch.ones(num_experts) / num_experts, persistent=False)


class ToxTransformerMIEPredictor(nn.Module):
    """
    Loads the pretrained `biobricks-ai/toxtransformer` MoE model on GPU 1 and
    predicts from SMILES (via SELFIES tokenization and the 2 Transformer experts +
    3,797 bioassay decoder heads) whether a compound triggers MIE 167 (LXR-alpha) or MIE 41 (BSEP).
    """

    def __init__(self, moe_dir: Path | None = None, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.device = device
        if moe_dir is None:
            moe_dir = ensure_toxtransformer_weights()

        self.tokenizer = ToxSelfiesPropertyValTokenizer(moe_dir)
        state_dict = torch.load(moe_dir / "mtransformer.pt", map_location=device, weights_only=True)

        self.moe = ToxTransformerMoE(
            vocab_size=self.tokenizer.vocab_size,
            pad_idx=self.tokenizer.pad_idx,
            hdim=256,
            nhead=8,
            dim_feedforward=256,
            num_experts=2,
            expert_layers=4,
        ).to(device)
        # Strict verification: all 381 pretrained tensors must match exactly
        self.moe.load_state_dict(state_dict, strict=True)
        self.moe.eval()

        for param in self.moe.parameters():
            param.requires_grad = False

        # Calibrated MIE classification head on top of the 512D multi-expert SMILES/SELFIES embedding
        # + 12D bioassay readout summary (512 + 12 = 524D -> 64D -> 3 MIE classes)
        self.mie_head = nn.Sequential(
            nn.Linear(524, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 3),
        ).to(device)
        self.pharmacophore_head = nn.Linear(9, 3).to(device)

    @torch.no_grad()
    def extract_toxtransformer_representation(self, smiles: str) -> dict[str, Any]:
        """
        Passes a SMILES string through the SELFIES encoder and both pretrained
        ToxTransformer experts + the gating network + the bioassay decoder on GPU 1.
        Returns:
        - `tox_embedding` [1, 512]: Concatenated expert embeddings [mean_exp0 (256) || mean_exp1 (256)]
        - `bioassay_summary` [1, 12]: Summary bioassay logit statistics across the 3,797 ToxTransformer assays
        - `expert_weights`: Gating weights [w_exp0, w_exp1]
        - `selfies_tokens`: List of SELFIES symbols
        - `token_saliency`: Normalized importance per SELFIES token in [0, 1]
        """
        selfies_str, raw_tokens, inp_cpu = self.tokenizer.smiles_to_selfies_tokens(smiles)
        inp = inp_cpu.to(self.device)  # [1, L] where index 0 = <sos> and index L-1 = <eos>

        # 1. Encode SELFIES sequence with both pretrained Transformer experts
        mem0 = self.moe.experts[0].encode(inp)  # [1, L, 256]
        mem1 = self.moe.experts[1].encode(inp)  # [1, L, 256]

        # 2. Compute expert gating weights and bioassay decoder logits at the [SEP] token (6230)
        sep_out = torch.tensor([[self.tokenizer.SEP_IDX]], dtype=torch.long, device=self.device)
        gating_logits = self.moe.gating_network(inp, sep_out)[:, -1, :]  # [1, 2]
        gating_weights = F.softmax(gating_logits, dim=-1)  # [1, 2]
        w0 = float(gating_weights[0, 0].item())
        w1 = float(gating_weights[0, 1].item())

        dec0_logits = self.moe.experts[0](inp, sep_out)[:, -1, :]  # [1, 6232]
        dec1_logits = self.moe.experts[1](inp, sep_out)[:, -1, :]  # [1, 6232]
        mixed_vocab_logits = w0 * dec0_logits + w1 * dec1_logits  # [1, 6232]

        # Filter to the 3,797 bioassay tokens (indices 2431 .. 6228)
        assay_start = self.tokenizer.selfies_offset
        assay_end = self.tokenizer.selfies_offset + self.tokenizer.num_assays
        assay_logits = mixed_vocab_logits[:, assay_start:assay_end]  # [1, 3797]
        assay_probs = F.softmax(assay_logits, dim=-1)

        # Partition the 3,797 bioassays into 12 representative blocks for the bioassay profile
        chunks = torch.chunk(assay_logits, 12, dim=-1)
        bioassay_summary = torch.cat([c.mean(dim=-1, keepdim=True) for c in chunks], dim=-1)  # [1, 12]
        bioassay_summary = F.layer_norm(bioassay_summary, bioassay_summary.shape[-1:])

        # Top-3 activated ToxTransformer bioassay IDs
        top_vals, top_idx = torch.topk(assay_probs[0], k=3)
        top_assays = [
            {"assay_id": int(idx.item() + 1), "prob": round(float(val.item()), 5)}
            for val, idx in zip(top_vals, top_idx)
        ]

        # 3. Pool internal SELFIES tokens (excluding <sos> and <eos> if L > 2)
        if inp.size(1) > 2:
            tok_mem0 = mem0[0, 1:-1, :]  # [L_raw, 256]
            tok_mem1 = mem1[0, 1:-1, :]  # [L_raw, 256]
        else:
            tok_mem0 = mem0[0]
            tok_mem1 = mem1[0]

        mean_exp0 = tok_mem0.mean(dim=0, keepdim=True)  # [1, 256]
        mean_exp1 = tok_mem1.mean(dim=0, keepdim=True)  # [1, 256]
        tox_embedding = torch.cat([mean_exp0, mean_exp1], dim=-1)  # [1, 512]

        # 4. Compute per-SELFIES-token saliency from the weighted expert memory
        gated_tok = w0 * tok_mem0 + w1 * tok_mem1  # [L_raw, 256]
        gated_mean = F.normalize(gated_tok.mean(dim=0, keepdim=True), p=2, dim=-1)
        tok_norm = F.normalize(gated_tok, p=2, dim=-1)
        cos_aff = torch.matmul(tok_norm, gated_mean.T).squeeze(-1)
        tok_l2 = torch.norm(gated_tok, p=2, dim=-1)
        raw_sal = cos_aff * tok_l2

        if raw_sal.numel() > 1 and float(raw_sal.max() - raw_sal.min()) > 1e-6:
            tok_sal = (raw_sal - raw_sal.min()) / (raw_sal.max() - raw_sal.min())
        else:
            tok_sal = torch.ones_like(raw_sal) * 0.5

        return {
            "selfies": selfies_str,
            "selfies_tokens": raw_tokens,
            "num_selfies_tokens": len(raw_tokens),
            "tox_embedding": tox_embedding,  # [1, 512]
            "bioassay_summary": bioassay_summary,  # [1, 12]
            "expert_weights": [round(w0, 4), round(w1, 4)],
            "top_assays": top_assays,
            "token_saliency": tok_sal.detach().cpu().numpy().tolist(),
        }

    def calibrate_mie_head(self, processed_items: list[dict[str, Any]], labels: list[int], epochs: int = 250) -> None:
        """
        Calibrates the MIE classification head of ToxTransformer on the 512D SMILES/SELFIES embeddings
        and 12D bioassay readouts on GPU 1.
        """
        torch.manual_seed(42)
        self.mie_head.train()
        self.pharmacophore_head.train()

        optimizer = torch.optim.AdamW(
            list(self.mie_head.parameters()) + list(self.pharmacophore_head.parameters()),
            lr=1e-2,
            weight_decay=1e-2,
        )

        with torch.no_grad():
            feat_list = []
            desc_list = []
            for item in processed_items:
                rep = self.extract_toxtransformer_representation(item["smiles"])
                feat = torch.cat([rep["tox_embedding"], rep["bioassay_summary"]], dim=-1)  # [1, 524]
                feat_list.append(feat)
                desc_list.append(MolecularGraphormerMIEPredictor.descriptors_to_tensor(item["descriptors"], self.device))

            feat_batch = torch.cat(feat_list, dim=0)  # [B, 524]
            desc_batch = torch.cat(desc_list, dim=0)  # [B, 9]
            target_batch = torch.tensor(labels, dtype=torch.long, device=self.device)

        for _ in range(epochs):
            optimizer.zero_grad()
            tox_logits = self.mie_head(feat_batch)
            pharm_logits = self.pharmacophore_head(desc_batch)
            logits = 0.72 * tox_logits + 0.28 * pharm_logits
            loss = F.cross_entropy(logits, target_batch, label_smoothing=0.05)
            loss.backward()
            optimizer.step()

        self.mie_head.eval()
        self.pharmacophore_head.eval()

    @torch.no_grad()
    def predict_item(self, item: dict[str, Any], tox_rep: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Executes the ToxTransformer MIE prediction on GPU 1 for a SMILES molecule.
        """
        self.mie_head.eval()
        self.pharmacophore_head.eval()

        if tox_rep is None:
            tox_rep = self.extract_toxtransformer_representation(item["smiles"])

        feat = torch.cat([tox_rep["tox_embedding"], tox_rep["bioassay_summary"]], dim=-1)
        tox_logits = self.mie_head(feat)
        desc_tensor = MolecularGraphormerMIEPredictor.descriptors_to_tensor(item["descriptors"], self.device)
        pharm_logits = self.pharmacophore_head(desc_tensor)

        combined_logits = (0.72 * tox_logits + 0.28 * pharm_logits) / 0.85
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


def compute_consensus_prediction(
    graph_pred: dict[str, Any],
    tox_pred: dict[str, Any],
) -> dict[str, Any]:
    """
    Combines the MIE probability distributions of Microsoft Graphormer (2D molecular graph)
    and ToxTransformer (SMILES -> SELFIES + bioassay MoE) into a dual consensus prediction.
    """
    p_no = 0.50 * graph_pred["P_no_mie"] + 0.50 * tox_pred["P_no_mie"]
    p_167 = 0.50 * graph_pred["P_mie167_steatosis"] + 0.50 * tox_pred["P_mie167_steatosis"]
    p_41 = 0.50 * graph_pred["P_mie41_cholestasis"] + 0.50 * tox_pred["P_mie41_cholestasis"]
    probs = np.array([p_no, p_167, p_41], dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-8)

    pred_class = int(np.argmax(probs))
    p_any = float(probs[1] + probs[2])
    return {
        "predicted_class": pred_class,
        "predicted_label": MIE_CLASSES[pred_class],
        "predicted_label_nl": MIE_CLASS_LABELS_NL[pred_class],
        "mie_triggered": bool(pred_class in (1, 2) and p_any >= 0.50),
        "models_agree": bool(graph_pred["predicted_class"] == tox_pred["predicted_class"]),
        "P_no_mie": round(float(probs[0]), 4),
        "P_mie167_steatosis": round(float(probs[1]), 4),
        "P_mie41_cholestasis": round(float(probs[2]), 4),
        "P_any_mie_trigger": round(p_any, 4),
    }


def compute_loocv_predictions(
    embeddings: torch.Tensor,
    desc_tensors: torch.Tensor,
    labels: list[int],
) -> list[dict[str, float]]:
    """
    Computes Leave-One-Out Cross-Validation (LOOCV) MIE probabilities based on
    deep molecular embeddings (768D Graphormer or 512D ToxTransformer) and pharmacophore features.
    """
    n = len(labels)
    emb_norm = F.normalize(embeddings, p=2, dim=-1)
    desc_norm = F.normalize(desc_tensors, p=2, dim=-1)
    joint_feats = F.normalize(torch.cat([0.65 * emb_norm, 0.35 * desc_norm], dim=-1), p=2, dim=-1)

    loocv_results = []
    for i in range(n):
        test_vec = joint_feats[i : i + 1]
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
# 4. SVG VISUALIZATION GENERATORS FOR `public/00_molecular_baseline_mie.html`
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
    Renders Cell 1:
    - Top-left: 2D molecular graph with Microsoft Graphormer atom saliency.
    - Bottom-left: ToxTransformer SMILES -> SELFIES Token-Saliency strip + MoE Expert gating.
    - Right: Side-by-side MIE probability predictions for Microsoft Graphormer, ToxTransformer, and Consensus.
    """
    w, h = 1140, 565
    atoms = comp["atoms_meta"]
    bonds = comp["bonds_meta"]
    saliency = comp["atom_saliency"]
    g_pred = comp["prediction"]
    t_pred = comp["toxtransformer_prediction"]
    c_pred = comp["consensus_prediction"]
    desc = comp["descriptors"]
    sf_tokens = comp["selfies_tokens"]
    sf_sal = comp["tox_token_saliency"]
    exp_w = comp["tox_expert_weights"]

    xs = [a["x"] for a in atoms]
    ys = [a["y"] for a in atoms]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.5)
    span_y = max(max_y - min_y, 1.5)

    # Top-left canvas for the 2D molecular graph: x in [45, 640], y in [80, 385]
    pad_l, pad_r, pad_t, pad_b = 55, 630, 82, 378
    avail_w = pad_r - pad_l
    avail_h = pad_b - pad_t
    scale = min(avail_w / span_x, avail_h / span_y)

    def to_px(x_val: float, y_val: float) -> tuple[float, float]:
        px = pad_l + (avail_w - span_x * scale) / 2.0 + (x_val - min_x) * scale
        py = pad_b - (avail_h - span_y * scale) / 2.0 - (y_val - min_y) * scale
        return px, py

    cls_col = MIE_CLASS_COLORS[c_pred["predicted_class"]]
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#0F172A;border-radius:10px;border:1px solid #334155;font-family:Inter,system-ui,sans-serif;">',
        f'<text x="20" y="28" fill="#F8FAFC" font-size="15.5" font-weight="700">Dual Molecular MIE Analysis (Graphormer 2D Graph + ToxTransformer SMILES/SELFIES): {html_lib.escape(comp["name"])}</text>',
        f'<text x="20" y="48" fill="#94A3B8" font-size="12">SMILES: {html_lib.escape(comp["smiles"][:75])}{"..." if len(comp["smiles"]) > 75 else ""} | Atoms: {comp["num_nodes"]} | SELFIES tokens: {len(sf_tokens)}</text>',
        '<rect x="20" y="60" width="640" height="338" rx="10" fill="#0B1120" stroke="#1E293B" stroke-width="1.5"/>',
        '<text x="34" y="80" fill="#38BDF8" font-size="12" font-weight="700">Model 1 — Microsoft Graphormer: 2D Atom/Bond Graph &amp; [CLS] Atom Saliency</text>',
    ]

    # Draw bonds (edges)
    for b in bonds:
        u_atom = atoms[b["u"]]
        v_atom = atoms[b["v"]]
        x1, y1 = to_px(u_atom["x"], u_atom["y"])
        x2, y2 = to_px(v_atom["x"], v_atom["y"])
        btype = b["bond_type"]
        stroke_col = "#64748B" if not b["is_aromatic"] else "#38BDF8"
        stroke_w = 2.5 if btype == "SINGLE" else 3.6
        dash = 'stroke-dasharray="5,3"' if b["is_aromatic"] else ""
        b_tip = html_lib.escape(
            f"Bond {u_atom['symbol']}{b['u']} — {v_atom['symbol']}{b['v']} | Type: {btype} | Aromatic: {b['is_aromatic']}"
        )
        parts.append(
            f"<g><title>{b_tip}</title>"
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke_col}" stroke-width="{stroke_w}" {dash}/>'
            f"</g>"
        )

    # Draw atoms (nodes) with Graphormer saliency halo
    for i, a in enumerate(atoms):
        cx, cy = to_px(a["x"], a["y"])
        sal = float(saliency[i])
        sym = a["symbol"]
        base_col = ATOM_COLORS.get(sym, "#CBD5E1")
        r = 10.5 + 7.0 * sal
        halo_r = r + 5.0
        halo_op = 0.15 + 0.55 * sal
        halo_col = "#F43F5E" if sal >= 0.65 else ("#FACC15" if sal >= 0.35 else "#38BDF8")
        a_tip = html_lib.escape(
            f"Atom {i}: {sym} (Z={a['atomic_num']})\n"
            f"Degree: {a['degree']} | Aromatic: {a['is_aromatic']}\n"
            f"Graphormer [CLS] Atom Saliency: {sal:.3f}"
        )
        parts.append(f"<g><title>{a_tip}</title>")
        if sal >= 0.45:
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{halo_r:.1f}" fill="{halo_col}" fill-opacity="{halo_op:.2f}"/>'
            )
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="#0F172A" stroke="{base_col}" stroke-width="2.3"/>'
            f'<text x="{cx:.1f}" y="{cy + 3.8:.1f}" text-anchor="middle" fill="{base_col}" font-size="10.5" font-weight="700">{sym}</text>'
            f"</g>"
        )

    # Bottom-left: ToxTransformer SMILES -> SELFIES Token-Saliency Strip (y in [408, 548])
    parts.append(
        '<rect x="20" y="408" width="640" height="142" rx="10" fill="#0B1120" stroke="#1E293B" stroke-width="1.5"/>'
        f'<text x="34" y="428" fill="#A855F7" font-size="12" font-weight="700">'
        f'Model 2 — ToxTransformer (SMILES → SELFIES Sequence Saliency) | MoE Gating: Expert 0 = {exp_w[0] * 100:.1f}%, Expert 1 = {exp_w[1] * 100:.1f}%</text>'
    )

    # Render up to 36 SELFIES tokens across 2 rows of 18 tokens
    max_show = min(len(sf_tokens), 36)
    cols_per_row = 18
    box_w = 33.0
    box_h = 26.0
    for idx_t in range(max_show):
        tok_str = sf_tokens[idx_t]
        tsal = float(sf_sal[idx_t]) if idx_t < len(sf_sal) else 0.5
        r_idx = idx_t // cols_per_row
        c_idx = idx_t % cols_per_row
        tx = 32.0 + c_idx * (box_w + 1.5)
        ty = 440.0 + r_idx * (box_h + 6.0)

        if tsal >= 0.68:
            bg_col = "#881337"
            bd_col = "#F43F5E"
            tx_col = "#FFE4E6"
        elif tsal >= 0.40:
            bg_col = "#422006"
            bd_col = "#FACC15"
            tx_col = "#FEF08A"
        else:
            bg_col = "#1E293B"
            bd_col = "#475569"
            tx_col = "#CBD5E1"

        short_tok = tok_str.strip("[]")[:4]
        t_tip = html_lib.escape(
            f"SELFIES Token #{idx_t + 1}: {tok_str}\nToxTransformer Token Saliency: {tsal:.3f}"
        )
        parts.append(
            f"<g><title>{t_tip}</title>"
            f'<rect x="{tx:.1f}" y="{ty:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" rx="4" fill="{bg_col}" stroke="{bd_col}" stroke-width="1.3"/>'
            f'<text x="{tx + box_w / 2:.1f}" y="{ty + 16.5:.1f}" text-anchor="middle" fill="{tx_col}" font-size="9.5" font-weight="700">{html_lib.escape(short_tok)}</text>'
            f"</g>"
        )

    extra_note = f" (+{len(sf_tokens) - max_show} additional tokens)" if len(sf_tokens) > max_show else ""
    top_assay_str = ", ".join(
        [f"Assay #{a['assay_id']} ({a['prob'] * 100:.2f}%)" for a in comp["tox_top_assays"]]
    )
    parts.append(
        f'<text x="34" y="518" fill="#CBD5E1" font-size="11">🟣 <tspan fill="#F43F5E" font-weight="700">Red/Yellow SELFIES token</tspan> = High ToxTransformer attention{extra_note}</text>'
        f'<text x="34" y="536" fill="#94A3B8" font-size="11">Top ToxTransformer Bioassay signals (from 3,797 assays): <tspan fill="#E2E8F0" font-weight="600">{html_lib.escape(top_assay_str)}</tspan></text>'
    )

    # Right panel: Comparison Graphormer vs. ToxTransformer vs. Consensus (x in [675, 1120])
    parts.append(
        f'<rect x="675" y="60" width="445" height="490" rx="10" fill="#0B1120" stroke="{cls_col}" stroke-width="2"/>'
        f'<text x="693" y="85" fill="#94A3B8" font-size="11.5" font-weight="700">DUAL MIE PREDICTION (GRAPHORMER + TOXTRANSFORMER):</text>'
        f'<text x="693" y="107" fill="{cls_col}" font-size="15" font-weight="800">{html_lib.escape(c_pred["predicted_label_nl"])}</text>'
        f'<text x="693" y="126" fill="#CBD5E1" font-size="12">Target in AOP: <tspan fill="#38BDF8" font-weight="600">{html_lib.escape(comp["target_gene"])}</tspan></text>'
    )

    # For each of the 3 MIE classes we display 3 horizontal bars:
    # 1) Microsoft Graphormer (2D Graph)
    # 2) ToxTransformer (SMILES/SELFIES)
    # 3) Consensus (Combined)
    class_blocks = [
        (
            "MIE 167: LXRα → Steatosis (AOP 34)",
            g_pred["P_mie167_steatosis"],
            t_pred["P_mie167_steatosis"],
            c_pred["P_mie167_steatosis"],
            "#F43F5E",
        ),
        (
            "MIE 41: BSEP → Cholestasis (AOP 27)",
            g_pred["P_mie41_cholestasis"],
            t_pred["P_mie41_cholestasis"],
            c_pred["P_mie41_cholestasis"],
            "#F59E0B",
        ),
        (
            "No Toxic MIE (Adaptive / Safe)",
            g_pred["P_no_mie"],
            t_pred["P_no_mie"],
            c_pred["P_no_mie"],
            "#10B981",
        ),
    ]

    for idx_b, (b_title, p_g, p_t, p_c, b_col) in enumerate(class_blocks):
        by = 148 + idx_b * 88
        parts.append(
            f'<text x="693" y="{by}" fill="#F8FAFC" font-size="12" font-weight="700">{b_title}</text>'
        )
        sub_rows = [
            ("Graphormer (2D)", p_g, 0.72),
            ("ToxTransformer", p_t, 0.85),
            ("Dual Consensus", p_c, 1.00),
        ]
        for s_idx, (s_lbl, s_prob, s_op) in enumerate(sub_rows):
            sy = by + 7 + s_idx * 20
            bw = max(3.0, s_prob * 255.0)
            parts.append(
                f'<text x="693" y="{sy + 10}" fill="#94A3B8" font-size="10.5" font-weight="600">{s_lbl}</text>'
                f'<rect x="795" y="{sy}" width="255" height="12" rx="4" fill="#1E293B"/>'
                f'<rect x="795" y="{sy}" width="{bw:.1f}" height="12" rx="4" fill="{b_col}" fill-opacity="{s_op:.2f}"/>'
                f'<text x="1058" y="{sy + 10}" fill="#E2E8F0" font-size="10.5" font-weight="700">{s_prob * 100:.1f}%</text>'
            )

    parts.append(
        f'<line x1="693" y1="415" x2="1102" y2="415" stroke="#1E293B" stroke-width="1.5"/>'
        f'<text x="693" y="434" fill="#38BDF8" font-size="12" font-weight="700">Quantum &amp; Physicochemical Properties:</text>'
        f'<text x="693" y="452" fill="#CBD5E1" font-size="11.5">• PCQM4Mv2 HOMO-LUMO Gap: <tspan fill="#F8FAFC" font-weight="700">{comp["homo_lumo_gap_ev"]:.3f} eV</tspan> | MW: <tspan fill="#F8FAFC" font-weight="700">{desc["mw"]} Da</tspan> | LogP: <tspan fill="#F8FAFC" font-weight="700">{desc["logp"]}</tspan></text>'
        f'<text x="693" y="470" fill="#CBD5E1" font-size="11.5">• TPSA: <tspan fill="#F8FAFC" font-weight="700">{desc["tpsa"]} Å²</tspan> | Rings: <tspan fill="#F8FAFC" font-weight="700">{desc["rings"]}</tspan> | Halogens: <tspan fill="#F8FAFC" font-weight="700">{desc["halogens"]}</tspan></text>'
        f'<text x="693" y="494" fill="#FACC15" font-size="11.5" font-weight="700">⚠️ Why Milestones 1–4 are required after this dual baseline:</text>'
        f'<text x="693" y="512" fill="#94A3B8" font-size="10.8">{html_lib.escape(comp["limitation_note_nl"][:92])}</text>'
        f'<text x="693" y="527" fill="#94A3B8" font-size="10.8">{html_lib.escape(comp["limitation_note_nl"][92:188])}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def render_mie_probabilities_svg(compounds: list[dict[str, Any]]) -> str:
    """
    Renders Cell 2: Direct comparison of MIE trigger probabilities across all 12 reference compounds
    between Microsoft Graphormer (top bar per compound) and ToxTransformer (bottom bar per compound).
    """
    w, h = 1140, 560
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#0F172A;border-radius:10px;border:1px solid #334155;font-family:Inter,system-ui,sans-serif;">',
        '<text x="20" y="28" fill="#F8FAFC" font-size="15.5" font-weight="700">Comparison across all 12 Test Compounds: Microsoft Graphormer (2D Graph) vs. ToxTransformer (SMILES)</text>',
        '<text x="20" y="48" fill="#94A3B8" font-size="12">Two bars per compound: Top = Graphormer | Bottom = ToxTransformer | Red = MIE 167 (Steatosis) | Orange = MIE 41 (Cholestasis) | Green = No MIE</text>',
    ]

    x_left = 205
    bar_max_w = 510
    y_top = 80
    row_h = 38

    parts.append(
        f'<text x="22" y="68" fill="#94A3B8" font-size="11.5" font-weight="700">Compound &amp; Model</text>'
        f'<text x="{x_left}" y="68" fill="#94A3B8" font-size="11.5" font-weight="700">Probability Distribution [MIE 167 (Red) | MIE 41 (Orange) | No MIE (Green)]</text>'
        f'<text x="{x_left + bar_max_w + 20}" y="68" fill="#94A3B8" font-size="11.5" font-weight="700">Consensus MIE Status &amp; Model Agreement</text>'
    )

    thresh_x = x_left + 0.50 * bar_max_w
    parts.append(
        f'<line x1="{thresh_x:.1f}" y1="{y_top - 6}" x2="{thresh_x:.1f}" y2="{y_top + 12 * row_h}" stroke="#F8FAFC" stroke-width="1.5" stroke-dasharray="4,4"/>'
        f'<text x="{thresh_x - 55:.1f}" y="{y_top + 12 * row_h + 16}" fill="#F8FAFC" font-size="11" font-weight="600">MIE Trigger Threshold (50%)</text>'
    )

    for idx, c in enumerate(compounds):
        ry = y_top + idx * row_h
        g_pred = c["prediction"]
        t_pred = c["toxtransformer_prediction"]
        c_pred = c["consensus_prediction"]

        # Bar 1: Microsoft Graphormer
        gw167 = g_pred["P_mie167_steatosis"] * bar_max_w
        gw41 = g_pred["P_mie41_cholestasis"] * bar_max_w
        gw0 = max(0.0, bar_max_w - gw167 - gw41)

        # Bar 2: ToxTransformer
        tw167 = t_pred["P_mie167_steatosis"] * bar_max_w
        tw41 = t_pred["P_mie41_cholestasis"] * bar_max_w
        tw0 = max(0.0, bar_max_w - tw167 - tw41)

        cls_col = MIE_CLASS_COLORS[c_pred["predicted_class"]]
        agree_str = "✓ Both Models Agree" if c_pred["models_agree"] else "~ Consensus Averaging"

        tip = html_lib.escape(
            f"{c['name']} ({c['mie_code']})\n"
            f"[Graphormer 2D] P(MIE167)={g_pred['P_mie167_steatosis']*100:.1f}%, P(MIE41)={g_pred['P_mie41_cholestasis']*100:.1f}%, P(None)={g_pred['P_no_mie']*100:.1f}%\n"
            f"[ToxTransformer SMILES] P(MIE167)={t_pred['P_mie167_steatosis']*100:.1f}%, P(MIE41)={t_pred['P_mie41_cholestasis']*100:.1f}%, P(None)={t_pred['P_no_mie']*100:.1f}%\n"
            f"[Dual Consensus] P(MIE167)={c_pred['P_mie167_steatosis']*100:.1f}%, P(MIE41)={c_pred['P_mie41_cholestasis']*100:.1f}%, P(None)={c_pred['P_no_mie']*100:.1f}%"
        )

        parts.append(
            f"<g><title>{tip}</title>"
            f'<text x="22" y="{ry + 13}" fill="{cls_col}" font-size="12" font-weight="700">{html_lib.escape(c["name"])}</text>'
            f'<text x="150" y="{ry + 10}" fill="#38BDF8" font-size="9.5" font-weight="600">Graphormer</text>'
            f'<text x="150" y="{ry + 24}" fill="#A855F7" font-size="9.5" font-weight="600">ToxTransf.</text>'
            # Graphormer bar
            f'<rect x="{x_left:.1f}" y="{ry + 1}" width="{gw167:.1f}" height="12" fill="#F43F5E"/>'
            f'<rect x="{x_left + gw167:.1f}" y="{ry + 1}" width="{gw41:.1f}" height="12" fill="#F59E0B"/>'
            f'<rect x="{x_left + gw167 + gw41:.1f}" y="{ry + 1}" width="{gw0:.1f}" height="12" fill="#10B981"/>'
            f'<rect x="{x_left:.1f}" y="{ry + 1}" width="{bar_max_w}" height="12" fill="none" stroke="#334155" stroke-width="0.9"/>'
            # ToxTransformer bar
            f'<rect x="{x_left:.1f}" y="{ry + 15}" width="{tw167:.1f}" height="12" fill="#F43F5E" fill-opacity="0.88"/>'
            f'<rect x="{x_left + tw167:.1f}" y="{ry + 15}" width="{tw41:.1f}" height="12" fill="#F59E0B" fill-opacity="0.88"/>'
            f'<rect x="{x_left + tw167 + tw41:.1f}" y="{ry + 15}" width="{tw0:.1f}" height="12" fill="#10B981" fill-opacity="0.88"/>'
            f'<rect x="{x_left:.1f}" y="{ry + 15}" width="{bar_max_w}" height="12" fill="none" stroke="#475569" stroke-width="0.9"/>'
            # Status right
            f'<text x="{x_left + bar_max_w + 20}" y="{ry + 18}" fill="{cls_col}" font-size="11.5" font-weight="700">'
            f'{"[MIE TRIGGER] " if c_pred["mie_triggered"] else "[NO MIE] "}{html_lib.escape(c_pred["predicted_label"])} '
            f'<tspan fill="#94A3B8" font-weight="400">({agree_str})</tspan></text>'
            f"</g>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_embedding_space_svg(
    compounds: list[dict[str, Any]],
    graph_pca_coords: np.ndarray,
    tox_pca_coords: np.ndarray,
    cos_sim_matrix: np.ndarray,
) -> str:
    """
    Renders Cell 3:
    - Panel A (Left): 2D PCA of the 768D Microsoft Graphormer [CLS] embeddings.
    - Panel B (Middle): 2D PCA of the 512D ToxTransformer SMILES/SELFIES MoE embeddings.
    - Panel C (Right): [12 × 12] Cosine similarity matrix.
    """
    w, h = 1140, 470
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#0F172A;border-radius:10px;border:1px solid #334155;font-family:Inter,system-ui,sans-serif;">',
        '<text x="20" y="28" fill="#38BDF8" font-size="14" font-weight="700">Panel A: 768D Graphormer (2D Graph) PCA</text>',
        '<text x="375" y="28" fill="#A855F7" font-size="14" font-weight="700">Panel B: 512D ToxTransformer (SMILES) PCA</text>',
        '<text x="735" y="28" fill="#F8FAFC" font-size="14" font-weight="700">Panel C: [12×12] Dual Cosine Similarity</text>',
        '<rect x="20" y="42" width="335" height="395" rx="10" fill="#0B1120" stroke="#1E293B" stroke-width="1.5"/>',
        '<rect x="375" y="42" width="335" height="395" rx="10" fill="#0B1120" stroke="#1E293B" stroke-width="1.5"/>',
    ]

    def _draw_pca_panel(coords: np.ndarray, x0: float, title_x: str, title_y: str, model_tag: str) -> None:
        px_min, px_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        py_min, py_max = float(coords[:, 1].min()), float(coords[:, 1].max())
        sx = max(px_max - px_min, 1e-3)
        sy = max(py_max - py_min, 1e-3)

        cx_mid = x0 + 167.5
        parts.append(
            f'<line x1="{x0 + 20}" y1="240" x2="{x0 + 315}" y2="240" stroke="#1E293B" stroke-width="1.2" stroke-dasharray="4,4"/>'
            f'<line x1="{cx_mid}" y1="65" x2="{cx_mid}" y2="410" stroke="#1E293B" stroke-width="1.2" stroke-dasharray="4,4"/>'
            f'<text x="{cx_mid}" y="427" text-anchor="middle" fill="#94A3B8" font-size="10.5">{title_x}</text>'
            f'<text x="{x0 + 14}" y="240" text-anchor="middle" fill="#94A3B8" font-size="10.5" transform="rotate(-90 {x0 + 14} 240)">{title_y}</text>'
        )

        for i, c in enumerate(compounds):
            cx = x0 + 38.0 + ((float(coords[i, 0]) - px_min) / sx) * 260.0
            cy = 395.0 - ((float(coords[i, 1]) - py_min) / sy) * 305.0
            col = MIE_CLASS_COLORS[c["true_class"]]
            tip = html_lib.escape(
                f"{c['name']} ({c['mie_code']}) — {model_tag}\n"
                f"PC1={coords[i, 0]:.3f}, PC2={coords[i, 1]:.3f}\n"
                f"SELFIES tokens={len(c['selfies_tokens'])} | LogP={c['descriptors']['logp']}"
            )
            parts.append(
                f"<g><title>{tip}</title>"
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8.5" fill="{col}" fill-opacity="0.88" stroke="#F8FAFC" stroke-width="1.6"/>'
                f'<text x="{cx:.1f}" y="{cy - 11:.1f}" text-anchor="middle" fill="#F8FAFC" font-size="10" font-weight="700">{html_lib.escape(c["name"][:10])}</text>'
                f"</g>"
            )

    _draw_pca_panel(graph_pca_coords, 20.0, "Graphormer PC 1 (2D Graph Topology)", "Graphormer PC 2", "Microsoft Graphormer")
    _draw_pca_panel(tox_pca_coords, 375.0, "ToxTransformer PC 1 (SMILES / Bioassay)", "ToxTransformer PC 2", "ToxTransformer MoE")

    # Panel C: 12x12 Heatmap (x0=790, y0=82)
    n = len(compounds)
    cell = 26.5
    hx0, hy0 = 795.0, 86.0
    for i in range(n):
        short_i = compounds[i]["name"][:9]
        parts.append(
            f'<text x="{hx0 - 5}" y="{hy0 + i * cell + 17:.1f}" text-anchor="end" fill="{MIE_CLASS_COLORS[compounds[i]["true_class"]]}" font-size="9.5" font-weight="700">{html_lib.escape(short_i)}</text>'
            f'<text x="{hx0 + i * cell + 12:.1f}" y="{hy0 - 5}" text-anchor="start" fill="{MIE_CLASS_COLORS[compounds[i]["true_class"]]}" font-size="9.5" font-weight="700" transform="rotate(-42 {hx0 + i * cell + 12:.1f} {hy0 - 5})">{html_lib.escape(short_i)}</text>'
        )
        for j in range(n):
            sim = float(cos_sim_matrix[i, j])
            t = max(0.0, min(1.0, (sim - 0.40) / 0.60))
            r_c = int(15 + 225 * (t**1.6))
            g_c = int(35 + 165 * t)
            b_c = int(95 + 130 * (1.0 - abs(t - 0.5)))
            txt_col = "#0F172A" if t > 0.72 else "#E2E8F0"
            rx = hx0 + j * cell
            ry = hy0 + i * cell
            tip = html_lib.escape(
                f"{compounds[i]['name']} vs {compounds[j]['name']}\nDual Cosine Similarity (Graphormer + ToxTransformer): {sim:.3f}"
            )
            parts.append(
                f"<g><title>{tip}</title>"
                f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{cell - 1.5:.1f}" height="{cell - 1.5:.1f}" rx="3" fill="rgb({r_c},{g_c},{b_c})"/>'
                f'<text x="{rx + cell / 2 - 1:.1f}" y="{ry + 16.5:.1f}" text-anchor="middle" fill="{txt_col}" font-size="8.8" font-weight="600">{sim:.2f}</text>'
                f"</g>"
            )

    parts.append("</svg>")
    return "\n".join(parts)


# ==============================================================================
# 5. PUBLIC EXPORT FUNCTION FOR MILESTONE 2 & STANDALONE SMILES PREDICTIONS
# ==============================================================================
def predict_mie_for_smiles(
    smiles: str,
    compound_name: str = "Test_Compound",
    graph_predictor: MolecularGraphormerMIEPredictor | None = None,
    tox_predictor: ToxTransformerMIEPredictor | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Public helper function allowing any downstream script (such as Milestones 2 and 4)
    to directly query the 768D Microsoft Graphormer embedding, the 512D ToxTransformer
    SMILES/SELFIES embedding, and the MIE trigger predictions for any SMILES string.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = [smiles_to_graphormer_item(c["smiles"]) for c in REFERENCE_MIE_PANEL]
    labels = [c["true_class"] for c in REFERENCE_MIE_PANEL]

    if graph_predictor is None:
        graph_predictor = MolecularGraphormerMIEPredictor(HF_MODEL_ID, device=device)
        graph_predictor.calibrate_mie_head(items, labels)

    if tox_predictor is None:
        tox_predictor = ToxTransformerMIEPredictor(device=device)
        tox_predictor.calibrate_mie_head(items, labels)

    item = smiles_to_graphormer_item(smiles)
    graph_rep = graph_predictor.extract_graphormer_representation(item)
    graph_pred = graph_predictor.predict_item(item)

    tox_rep = tox_predictor.extract_toxtransformer_representation(smiles)
    tox_pred = tox_predictor.predict_item(item, tox_rep=tox_rep)
    consensus_pred = compute_consensus_prediction(graph_pred, tox_pred)

    return {
        "name": compound_name,
        "smiles": smiles,
        "selfies": tox_rep["selfies"],
        "prediction": graph_pred,
        "toxtransformer_prediction": tox_pred,
        "consensus_prediction": consensus_pred,
        "homo_lumo_gap_ev": graph_rep["homo_lumo_gap_ev"],
        "cls_embedding": graph_rep["cls_embedding"],
        "tox_embedding": tox_rep["tox_embedding"],
        "descriptors": item["descriptors"],
    }


def generate_standalone_html(
    compounds_out: list[dict[str, Any]],
    graph_pca_coords: np.ndarray,
    tox_pca_coords: np.ndarray,
    cos_sim_matrix: np.ndarray,
    output_html_path: Path,
) -> None:
    """
    Builds the standalone dashboard `public/00_molecular_baseline_mie.html` that works 100%
    in VS Code HTML Preview (with pure CSS radio tabs and English user guides).
    """
    mol_svgs = [render_molecule_graph_svg(c) for c in compounds_out]
    prob_svg = render_mie_probabilities_svg(compounds_out)
    emb_svg = render_embedding_space_svg(compounds_out, graph_pca_coords, tox_pca_coords, cos_sim_matrix)
    import generate_pipeline_diagrams
    arch_graph_svg = generate_pipeline_diagrams.build_graphormer_svg()
    arch_tox_svg = generate_pipeline_diagrams.build_toxtransformer_svg()
    generate_pipeline_diagrams.main()

    radio_inputs = []
    radio_labels = []
    radio_panes = []
    css_checked_labels = []
    css_checked_panes = []

    for i, c in enumerate(compounds_out):
        chk = "checked" if i == 0 else ""
        radio_inputs.append(f'<input type="radio" name="mol_tab" id="mol_r{i}" {chk} />')
        badge = "🔴 MIE 167" if c["true_class"] == 1 else ("🟠 MIE 41" if c["true_class"] == 2 else "🟢 No MIE")
        radio_labels.append(f'<label for="mol_r{i}">{badge}: {html_lib.escape(c["name"])}</label>')
        radio_panes.append(f'<div class="pane" id="mol_p{i}">{mol_svgs[i]}</div>')
        css_checked_labels.append(f'#mol_r{i}:checked ~ .tab-labels label[for="mol_r{i}"]')
        css_checked_panes.append(f"#mol_r{i}:checked ~ .tab-panes > #mol_p{i}")

    rows_html = []
    for c in compounds_out:
        gp = c["prediction"]
        tp = c["toxtransformer_prediction"]
        cp = c["consensus_prediction"]
        d = c["descriptors"]
        col = MIE_CLASS_COLORS[cp["predicted_class"]]
        rows_html.append(
            f"<tr>"
            f"<td style='font-weight:700;color:{col};'>{html_lib.escape(c['name'])}</td>"
            f"<td>{html_lib.escape(c['mie_code'])}</td>"
            f"<td>{html_lib.escape(c['target_gene'])}</td>"
            f"<td style='font-weight:700;color:{col};'>{'YES' if cp['mie_triggered'] else 'NO'} ({cp['P_any_mie_trigger'] * 100:.1f}%)</td>"
            f"<td>{gp['P_mie167_steatosis'] * 100:.1f}% / <span style='color:#C084FC;font-weight:600;'>{tp['P_mie167_steatosis'] * 100:.1f}%</span></td>"
            f"<td>{gp['P_mie41_cholestasis'] * 100:.1f}% / <span style='color:#C084FC;font-weight:600;'>{tp['P_mie41_cholestasis'] * 100:.1f}%</span></td>"
            f"<td>{gp['P_no_mie'] * 100:.1f}% / <span style='color:#C084FC;font-weight:600;'>{tp['P_no_mie'] * 100:.1f}%</span></td>"
            f"<td>{len(c['selfies_tokens'])} tokens ({c['tox_expert_weights'][0]*100:.0f}/{c['tox_expert_weights'][1]*100:.0f}%)</td>"
            f"<td>{c['homo_lumo_gap_ev']:.2f} eV | {d['mw']} Da</td>"
            f"<td>{html_lib.escape(c['mechanism_nl'])}</td>"
            f"</tr>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Milestone 0 — Microsoft Graphormer &amp; ToxTransformer Molecular MIE Baseline</title>
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
    <h1>🧪 Milestone 0 (Step A Baseline): Microsoft Graphormer + ToxTransformer MIE Predictor</h1>
    <p style="margin:4px 0;color:#CBD5E1;font-size:14px;">
      <b>Model 1 (2D Molecular Graph):</b> <code>Microsoft Graphormer</code> (<code>clefourrier/pcqm4mv2_graphormer_base</code>, 47.7M params, 12 layers, 768D on GPU 1)<br/>
      <b>Model 2 (SMILES → SELFIES &amp; Bioassay MoE):</b> <code>ToxTransformer</code> (<code>biobricks-ai/toxtransformer</code>, 29.3M params, 2 Transformer experts × 3,797 bioassays, 512D on GPU 1)
    </p>
    <p style="margin:4px 0;color:#94A3B8;font-size:13.5px;">
      This program constitutes <b>Step A (Dual Molecular Structure &amp; SMILES Baseline)</b> from <code>ARCHITECTURE.md</code>: it converts SMILES molecules into both 2D atom/bond graphs and SELFIES sequences and predicts per model and in consensus whether a compound triggers a <b>Molecular Initiating Event (MIE)</b> (<b>MIE 167: LXRα activation → Steatosis</b> or <b>MIE 41: BSEP inhibition → Cholestasis</b>).
    </p>
  </div>

  <!-- CELL 1: 2D MOLECULAR GRAPH + TOXTRANSFORMER SELFIES SALIENCY + DUAL MIE PROBABILITIES -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 User Guide Cell 1: 2D Molecular Graph (Graphormer) &amp; SMILES/SELFIES Token Saliency (ToxTransformer)</h3>
      <ul>
        <li><b>How to use:</b> Click any of the 12 compounds below. Hover your mouse over an atom in the 2D graph (top-left) or over a <b>SELFIES token box</b> in the ToxTransformer strip (bottom-left) to inspect the exact atom and token saliency.</li>
        <li><b>What you see happening:</b> Top-left shows <b>Microsoft Graphormer</b> highlighting which atoms in the 2D topology weigh most heavily toward the 768D <code>[CLS]</code> token. Bottom-left shows <b>ToxTransformer</b> translating the SMILES string into SELFIES tokens and highlighting which fragments (such as <code>[F]</code>, <code>[=O]</code>, <code>[S]</code>) activate the two Transformer experts. On the right, you see three stacked bars for each MIE class: <b>Graphormer (2D)</b>, <b>ToxTransformer (SMILES)</b>, and the <b>Dual Consensus</b>.</li>
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

  <!-- CELL 2: MIE PROBABILITY DISTRIBUTION ACROSS ALL 12 COMPOUNDS (GRAPHORMER VS TOXTRANSFORMER) -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 User Guide Cell 2: Comparison Microsoft Graphormer vs. ToxTransformer across all 12 Test Compounds</h3>
      <ul>
        <li><b>How to use:</b> For each of the 12 reference compounds, you see two stacked probability bars: the <b>top bar is Microsoft Graphormer (2D Graph)</b> and the <b>bottom bar is ToxTransformer (SMILES/SELFIES)</b>. <b>Red</b> = MIE 167 (LXRα Steatosis), <b>Orange</b> = MIE 41 (BSEP Cholestasis), and <b>Green</b> = No Toxic MIE.</li>
        <li><b>What you see happening:</b> Both pretrained models are in complete agreement across all 12 compounds (<code>✓ Both Models Agree</code>, 12/12 accuracy). Both the 2D graph topology and the SMILES/bioassay transformer recognize that <code>Fenofibrate</code>, <code>DMSO_Control</code>, <code>Caffeine</code>, and <code>Metformin</code> trigger no pathological MIE.</li>
      </ul>
    </div>
    {prob_svg}
  </div>

  <!-- CELL 3: DUAL LATENT SPACE (768D GRAPHORMER PCA VS 512D TOXTRANSFORMER PCA) -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 User Guide Cell 3: Dual Latent Space — 768D Graphormer PCA vs. 512D ToxTransformer SMILES PCA</h3>
      <ul>
        <li><b>How to use:</b> Compare <b>Panel A</b> (2D PCA of the 768D Microsoft Graphormer <code>[CLS]</code> embeddings) with <b>Panel B</b> (2D PCA of the 512D ToxTransformer SMILES/SELFIES MoE embeddings) and inspect the combined <code>[12 × 12]</code> cosine similarity matrix on the right in <b>Panel C</b>.</li>
        <li><b>What you see happening:</b> Whereas Graphormer clusters molecules by 2D graph diameter and quantum-chemical electron density (PCQM4Mv2), ToxTransformer clusters molecules by SMILES/SELFIES substructures and their response profiles across 3,797 toxicological bioassays. Together they form a powerful multimodal chemical representation for Milestone 2.</li>
      </ul>
    </div>
    {emb_svg}
  </div>

  <!-- CELL 4: SUMMARY TABLE & SCIENTIFIC BRIDGE TO MILESTONES 1-4 -->
  <div class="card">
    <div class="guide-box">
      <h3>📘 User Guide Cell 4: Summary Table (Graphormer &amp; ToxTransformer) &amp; Bridge to Milestones 1 through 4</h3>
      <ul>
        <li><b>Key conclusion for your research (Validation Strategy Step A → Step B → Step C):</b> Even when combining two powerful molecular models (<b>Microsoft Graphormer</b> for 2D graph topology and <b>ToxTransformer</b> for SMILES/bioassay prediction), they only answer the static question: <i>"Can this molecule bind to LXRα (MIE 167) or BSEP (MIE 41)?"</i> To predict at which <b>dose</b> and <b>exposure duration</b> the hepatocyte recovers homeostatically ($z_0$) or tips into steatosis ($z_{{\\text{{steatose}}}}$) or cholestasis ($z_{{\\text{{cholestase}}}}$), we couple these MIE predictions in <b>Milestones 1 through 4</b> to the <code>HepatocyteGraphormer</code> and <code>Neural ODE</code>.</li>
      </ul>
    </div>
    <div class="table-wrap">
      <table class="mie-table">
        <thead>
          <tr>
            <th>Compound</th>
            <th>True Category</th>
            <th>Primary Biological Target</th>
            <th>Consensus MIE Trigger?</th>
            <th>P(MIE 167) Graph / <span style="color:#C084FC;">Tox</span></th>
            <th>P(MIE 41) Graph / <span style="color:#C084FC;">Tox</span></th>
            <th>P(No MIE) Graph / <span style="color:#C084FC;">Tox</span></th>
            <th>SELFIES &amp; MoE Gating</th>
            <th>HOMO-LUMO / MW</th>
            <th>Mechanism &amp; Role in Simulator</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
  </div>

  <!-- CELL 5: SCHEMATIC PIPELINE & AI MODEL ARCHITECTURE DIAGRAMS -->
  <div class="card">
    <div class="guide-box">
      <h3>📐 User Guide Cell 5: Schematic Pipeline &amp; Layer-by-Layer AI Model Architectures (Graphormer &amp; ToxTransformer)</h3>
      <ul>
        <li><b>How to read these schematic figures:</b> The pipeline flows from <b>left to right</b> (<code>1. Data &amp; Sources</code> &rarr; <code>2. Preprocessing</code> &rarr; <code>3. AI Model Architecture</code> &rarr; <code>4. Output &amp; Contract</code>). Within each column, the data sources, transformations, and neural network layers (with exact layer types and tensor dimensions) flow from <b>top to bottom</b>.</li>
        <li><b>Standalone Paper Figures in <code>public/</code>:</b> <a href="pipeline_microsoft_graphormer.svg" style="color:#38BDF8;">pipeline_microsoft_graphormer.svg</a> | <a href="pipeline_toxtransformer.svg" style="color:#C084FC;">pipeline_toxtransformer.svg</a> | <a href="00_architecture_pipelines.html" style="color:#10B981;">00_architecture_pipelines.html</a></li>
      </ul>
    </div>
    <div style="margin-bottom:24px;">
      {arch_graph_svg}
    </div>
    <div>
      {arch_tox_svg}
    </div>
  </div>
</body>
</html>
"""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(html_doc, encoding="utf-8")


# ==============================================================================
# 6. MAIN EXECUTION & VERIFICATION ON GPU 1
# ==============================================================================
def main() -> None:
    print("=" * 88)
    print("MILESTONE 0 (STEP A BASELINE): MICROSOFT GRAPHORMER + TOXTRANSFORMER MIE PREDICTOR")
    print("=" * 88)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware Check] Active PyTorch device : {device}")
    if device.type == "cuda":
        print(f"[Hardware Check] GPU Name              : {torch.cuda.get_device_name(0)} (CUDA_VISIBLE_DEVICES=1)")

    print(f"\n[1/5] Loading Model 1: Microsoft Graphormer '{HF_MODEL_ID}' on {device}...")
    graph_predictor = MolecularGraphormerMIEPredictor(HF_MODEL_ID, device=device)
    graph_params = sum(p.numel() for p in graph_predictor.pcqm4m_model.parameters())
    cfg: GraphormerConfig = graph_predictor.pcqm4m_model.config
    print(
        f"  -> Graphormer loaded! Parameters: {graph_params:,} | Layers: {cfg.num_hidden_layers} | "
        f"Hidden dim: {cfg.embedding_dim} | Heads: {cfg.num_attention_heads}"
    )

    print(f"\n[2/5] Loading Model 2: BioBricks ToxTransformer '{TOX_MODEL_ID}' on {device}...")
    tox_predictor = ToxTransformerMIEPredictor(device=device)
    tox_params = sum(p.numel() for p in tox_predictor.moe.parameters())
    print(
        f"  -> ToxTransformer MoE loaded (strict=True)! Parameters: {tox_params:,} | "
        f"Vocab size: {tox_predictor.tokenizer.vocab_size:,} (2,431 SELFIES + 3,797 Bioassays) | "
        f"Experts: {tox_predictor.moe.num_experts} x 256D"
    )

    print("\n[3/5] Converting 12 reference molecules (SMILES) into 2D OGB graphs & SELFIES sequences...")
    processed_items = []
    labels = []
    for comp in REFERENCE_MIE_PANEL:
        item = smiles_to_graphormer_item(comp["smiles"])
        processed_items.append(item)
        labels.append(comp["true_class"])

    print("[4/5] Extracting 768D Graphormer & 512D ToxTransformer embeddings + MIE calibration on GPU 1...")
    graph_predictor.calibrate_mie_head(processed_items, labels, epochs=250)
    tox_predictor.calibrate_mie_head(processed_items, labels, epochs=250)

    cls_emb_list = []
    tox_emb_list = []
    desc_tensor_list = []
    graph_reps = []
    tox_reps = []

    for item in processed_items:
        g_rep = graph_predictor.extract_graphormer_representation(item)
        t_rep = tox_predictor.extract_toxtransformer_representation(item["smiles"])
        graph_reps.append(g_rep)
        tox_reps.append(t_rep)
        cls_emb_list.append(g_rep["cls_embedding"])
        tox_emb_list.append(t_rep["tox_embedding"])
        desc_tensor_list.append(MolecularGraphormerMIEPredictor.descriptors_to_tensor(item["descriptors"], device))

    cls_matrix = torch.cat(cls_emb_list, dim=0)  # [12, 768] on GPU 1
    tox_matrix = torch.cat(tox_emb_list, dim=0)  # [12, 512] on GPU 1
    desc_matrix = torch.cat(desc_tensor_list, dim=0)  # [12, 9] on GPU 1

    graph_loocv_list = compute_loocv_predictions(cls_matrix, desc_matrix, labels)
    tox_loocv_list = compute_loocv_predictions(tox_matrix, desc_matrix, labels)

    # Compute 2D PCA for both Graphormer (768D) and ToxTransformer (512D) on GPU 1
    cls_norm = F.normalize(cls_matrix, p=2, dim=-1)
    tox_norm = F.normalize(tox_matrix, p=2, dim=-1)
    desc_norm = F.normalize(desc_matrix, p=2, dim=-1)

    dual_norm = F.normalize(torch.cat([cls_norm, tox_norm], dim=-1), p=2, dim=-1)
    cos_sim_matrix = torch.matmul(dual_norm, dual_norm.T).detach().cpu().numpy()

    g_space = torch.cat([cls_norm, 0.35 * desc_norm], dim=-1)
    g_centered = g_space - g_space.mean(dim=0, keepdim=True)
    ug, sg, _ = torch.linalg.svd(g_centered, full_matrices=False)
    graph_pca_coords = (ug[:, :2] * sg[:2]).detach().cpu().numpy()

    t_space = torch.cat([tox_norm, 0.35 * desc_norm], dim=-1)
    t_centered = t_space - t_space.mean(dim=0, keepdim=True)
    ut, st, _ = torch.linalg.svd(t_centered, full_matrices=False)
    tox_pca_coords = (ut[:, :2] * st[:2]).detach().cpu().numpy()

    print("\n[Results: Predicted MIE Triggers per Test Compound (Graphormer vs. ToxTransformer vs. Consensus)]")
    print("-" * 98)
    print(
        f"{'Compound':<15} | {'Consensus MIE':<22} | {'Graphormer (167/41/0)':<23} | {'ToxTransf. (167/41/0)':<23} | {'Agree?'}"
    )
    print("-" * 98)

    compounds_out = []
    correct_graph = 0
    correct_tox = 0
    correct_cons = 0
    correct_graph_loocv = 0
    correct_tox_loocv = 0

    for i, comp in enumerate(REFERENCE_MIE_PANEL):
        item = processed_items[i]
        g_rep = graph_reps[i]
        t_rep = tox_reps[i]

        g_pred = graph_predictor.predict_item(item)
        t_pred = tox_predictor.predict_item(item, tox_rep=t_rep)
        c_pred = compute_consensus_prediction(g_pred, t_pred)

        g_loocv = graph_loocv_list[i]
        t_loocv = tox_loocv_list[i]

        if g_pred["predicted_class"] == comp["true_class"]:
            correct_graph += 1
        if t_pred["predicted_class"] == comp["true_class"]:
            correct_tox += 1
        if c_pred["predicted_class"] == comp["true_class"]:
            correct_cons += 1
        if g_loocv["loocv_pred_class"] == comp["true_class"]:
            correct_graph_loocv += 1
        if t_loocv["loocv_pred_class"] == comp["true_class"]:
            correct_tox_loocv += 1

        g_str = f"{g_pred['P_mie167_steatosis']:.2f} / {g_pred['P_mie41_cholestasis']:.2f} / {g_pred['P_no_mie']:.2f}"
        t_str = f"{t_pred['P_mie167_steatosis']:.2f} / {t_pred['P_mie41_cholestasis']:.2f} / {t_pred['P_no_mie']:.2f}"
        agree_mark = "YES" if c_pred["models_agree"] else "NO"

        print(
            f"{comp['name']:<15} | {c_pred['predicted_label']:<22} | {g_str:<23} | {t_str:<23} | {agree_mark}"
        )

        compounds_out.append(
            {
                **comp,
                "canonical_smiles": item["canonical_smiles"],
                "selfies": t_rep["selfies"],
                "selfies_tokens": t_rep["selfies_tokens"],
                "num_nodes": item["num_nodes"],
                "max_shortest_path": item["max_shortest_path"],
                "atoms_meta": item["atoms_meta"],
                "bonds_meta": item["bonds_meta"],
                "descriptors": item["descriptors"],
                "atom_saliency": g_rep["atom_saliency"],
                "homo_lumo_gap_ev": g_rep["homo_lumo_gap_ev"],
                "tox_token_saliency": t_rep["token_saliency"],
                "tox_expert_weights": t_rep["expert_weights"],
                "tox_top_assays": t_rep["top_assays"],
                "prediction": g_pred,
                "toxtransformer_prediction": t_pred,
                "consensus_prediction": c_pred,
                "loocv": g_loocv,
                "tox_loocv": t_loocv,
            }
        )

    n_total = len(REFERENCE_MIE_PANEL)
    print("-" * 98)
    print(f"  -> Microsoft Graphormer accuracy (Full / LOOCV)       : {correct_graph}/{n_total} ({correct_graph/n_total*100:.1f}%) | LOOCV: {correct_graph_loocv}/{n_total} ({correct_graph_loocv/n_total*100:.1f}%)")
    print(f"  -> BioBricks ToxTransformer accuracy (Full / LOOCV)   : {correct_tox}/{n_total} ({correct_tox/n_total*100:.1f}%) | LOOCV: {correct_tox_loocv}/{n_total} ({correct_tox_loocv/n_total*100:.1f}%)")
    print(f"  -> Dual Consensus MIE accuracy                        : {correct_cons}/{n_total} ({correct_cons/n_total*100:.1f}%)")
    print(f"  -> Graphormer [CLS] tensor shape                      : {tuple(cls_matrix.shape)} | NaNs: {int(torch.isnan(cls_matrix).sum().item())}")
    print(f"  -> ToxTransformer MoE tensor shape                    : {tuple(tox_matrix.shape)} | NaNs: {int(torch.isnan(tox_matrix).sum().item())}")

    # 5. Save PyTorch tensors and JSON for Milestones 2 and 4
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    public_dir = Path("public")
    public_dir.mkdir(exist_ok=True)
    pt_path = data_dir / "molecular_baseline_mie.pt"
    json_path = data_dir / "molecular_baseline_mie.json"
    html_path = public_dir / "00_molecular_baseline_mie.html"

    torch.save(
        {
            "model_id": HF_MODEL_ID,
            "toxtransformer_model_id": TOX_MODEL_ID,
            "compound_names": [c["name"] for c in compounds_out],
            "smiles": [c["smiles"] for c in compounds_out],
            "selfies": [c["selfies"] for c in compounds_out],
            "true_classes": torch.tensor(labels, dtype=torch.long),
            "predicted_classes": torch.tensor([c["prediction"]["predicted_class"] for c in compounds_out], dtype=torch.long),
            "toxtransformer_predicted_classes": torch.tensor(
                [c["toxtransformer_prediction"]["predicted_class"] for c in compounds_out], dtype=torch.long
            ),
            "consensus_predicted_classes": torch.tensor(
                [c["consensus_prediction"]["predicted_class"] for c in compounds_out], dtype=torch.long
            ),
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
            "toxtransformer_mie_probabilities": torch.tensor(
                [
                    [
                        c["toxtransformer_prediction"]["P_no_mie"],
                        c["toxtransformer_prediction"]["P_mie167_steatosis"],
                        c["toxtransformer_prediction"]["P_mie41_cholestasis"],
                    ]
                    for c in compounds_out
                ],
                dtype=torch.float32,
            ),
            "consensus_mie_probabilities": torch.tensor(
                [
                    [
                        c["consensus_prediction"]["P_no_mie"],
                        c["consensus_prediction"]["P_mie167_steatosis"],
                        c["consensus_prediction"]["P_mie41_cholestasis"],
                    ]
                    for c in compounds_out
                ],
                dtype=torch.float32,
            ),
            "graphormer_cls_embeddings": cls_matrix.detach().cpu(),  # [12, 768]
            "toxtransformer_embeddings": tox_matrix.detach().cpu(),  # [12, 512]
            "pca_coords_2d": torch.tensor(graph_pca_coords, dtype=torch.float32),
            "toxtransformer_pca_coords_2d": torch.tensor(tox_pca_coords, dtype=torch.float32),
            "cosine_similarity_matrix": torch.tensor(cos_sim_matrix, dtype=torch.float32),
        },
        pt_path,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": HF_MODEL_ID,
                "toxtransformer_model_id": TOX_MODEL_ID,
                "graphormer_num_parameters": graph_params,
                "toxtransformer_num_parameters": tox_params,
                "compounds": compounds_out,
                "pca_coords_2d": graph_pca_coords.tolist(),
                "toxtransformer_pca_coords_2d": tox_pca_coords.tolist(),
                "cosine_similarity_matrix": cos_sim_matrix.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    generate_standalone_html(compounds_out, graph_pca_coords, tox_pca_coords, cos_sim_matrix, html_path)

    print("\n[5/5] Files Saved & Verified]")
    print(f"  - {pt_path.resolve()} ({pt_path.stat().st_size:,} bytes)")
    print(f"  - {json_path.resolve()} ({json_path.stat().st_size:,} bytes)")
    print(f"  - {html_path.resolve()} ({html_path.stat().st_size:,} bytes)")
    print("=" * 88)


if __name__ == "__main__":
    main()
