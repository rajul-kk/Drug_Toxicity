
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolops

def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def get_atom_features(atom):
    features = []
    # Atomic number (one-hot)
    features += one_hot_encoding(atom.GetAtomicNum(), [6, 7, 8, 9, 15, 16, 17, 35, 53])
    # Degree
    features += one_hot_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
    # Hybridization
    features += one_hot_encoding(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ])
    # Aromaticity
    features += [1 if atom.GetIsAromatic() else 0]
    return torch.tensor(features, dtype=torch.float)

def get_bond_features(bond):
    features = []
    # Bond type
    bt = bond.GetBondType()
    features += one_hot_encoding(bt, [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ])
    # Conjugation
    features += [1 if bond.GetIsConjugated() else 0]
    # Ring
    features += [1 if bond.IsInRing() else 0]
    return torch.tensor(features, dtype=torch.float)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features (Nodes)
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_features)

    # Bond features (Edges)
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Add both directions for undirected graph
        edge_indices.append([start, end])
        edge_indices.append([end, start])
        
        b_feat = get_bond_features(bond)
        edge_features.append(b_feat)
        edge_features.append(b_feat)

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(get_bond_features(mol.GetBondWithIdx(0)))), dtype=torch.float) if mol.GetNumBonds() > 0 else torch.empty((0, 7), dtype=torch.float) # 7 is rough dimension of bond features
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
