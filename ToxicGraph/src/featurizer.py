
import hashlib
import inspect as _inspect
import numpy as np
import torch
from functools import lru_cache
from torch_geometric.data import Data
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')  # suppress UFFTYPER / sanitisation warnings


def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_atom_features(atom):
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), [6, 7, 8, 9, 15, 16, 17, 35, 53])   # 10
    features += one_hot_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])                # 7
    features += one_hot_encoding(atom.GetHybridization(), [                                 # 6
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ])
    features += [int(atom.GetIsAromatic())]                                                 # 1
    features += one_hot_encoding(atom.GetFormalCharge(), [-1, 0, 1, 2, 3])                 # 6
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])                    # 6
    features += [int(atom.IsInRing())]                                                      # 1
    features += one_hot_encoding(atom.GetChiralTag(), [                                     # 4
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ])
    # Gasteiger partial charge: electrostatic proxy, relevant for toxicity mechanisms
    charge = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
    charge = 0.0 if not np.isfinite(charge) else float(np.clip(charge, -1.0, 1.0))
    features += [charge]                                                                     # 1
    return torch.tensor(features, dtype=torch.float)  # 42 dims


def get_bond_features(bond):
    features = []
    features += one_hot_encoding(bond.GetBondType(), [                                      # 5
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ])
    features += [int(bond.GetIsConjugated())]                                               # 1
    features += [int(bond.IsInRing())]                                                      # 1
    return torch.tensor(features, dtype=torch.float)  # 7 dims


def _generate_conformer(mol):
    """
    Generate a 3D conformer for heavy atoms.
    Returns an (N, 3) numpy array or None if embedding fails.
    Adds hydrogens for better geometry then strips them back out.
    """
    try:
        mol_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        conf = mol_h.GetConformer()
        # Heavy atoms keep their original indices 0..n-1 after AddHs
        n = mol.GetNumAtoms()
        return np.array([list(conf.GetAtomPosition(i)) for i in range(n)], dtype=np.float32)
    except Exception:
        return None


def _compute_angle_features(mol, pos):
    """
    Per-atom bond-angle statistics (mean, std) normalised by π.
    Returns a (N, 2) tensor. Zero-filled when pos is None or atom has < 2 neighbours.
    These are SE(3)-invariant: they don't change under rotation or translation.
    """
    feats = []
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if pos is None or len(nbrs) < 2:
            feats.append([0.0, 0.0])
            continue
        angles = []
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                v1 = pos[nbrs[a]] - pos[j]
                v2 = pos[nbrs[b]] - pos[j]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-8 or n2 < 1e-8:
                    continue
                cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                angles.append(np.arccos(cos_a))
        if angles:
            feats.append([float(np.mean(angles)) / np.pi,
                          float(np.std(angles)) / np.pi])
        else:
            feats.append([0.0, 0.0])
    return torch.tensor(feats, dtype=torch.float)  # (N, 2)


@lru_cache(maxsize=1024)
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    AllChem.ComputeGasteigerCharges(mol)

    # ── base node features (42 dims) ──────────────────────────────────────────
    x = torch.stack([get_atom_features(atom) for atom in mol.GetAtoms()])

    # ── base edge features (7 dims) ───────────────────────────────────────────
    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_features += [feat, feat]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features)            # (E, 7)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 8), dtype=torch.float)  # skip to final dim

    # ── 3D geometry (SE(3)-invariant) ─────────────────────────────────────────
    pos = _generate_conformer(mol)

    # Distance appended to each edge (1 dim → total 8)
    if edge_indices:
        if pos is not None:
            src = edge_index[0].tolist()
            dst = edge_index[1].tolist()
            dists = np.linalg.norm(pos[src] - pos[dst], axis=1, keepdims=True)
            dist_feat = torch.tensor(dists, dtype=torch.float)
        else:
            dist_feat = torch.zeros(edge_attr.shape[0], 1)
        edge_attr = torch.cat([edge_attr, dist_feat], dim=1)  # (E, 8)

    # Bond-angle stats appended to each node (2 dims → total 44)
    angle_feat = _compute_angle_features(mol, pos)        # (N, 2)
    x = torch.cat([x, angle_feat], dim=1)                 # (N, 44)

    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _compute_featurizer_hash() -> str:
    src = (
        _inspect.getsource(get_atom_features) +
        _inspect.getsource(get_bond_features) +
        _inspect.getsource(smiles_to_graph)
    )
    return hashlib.sha256(src.encode()).hexdigest()[:8]


FEATURIZER_HASH = _compute_featurizer_hash()
