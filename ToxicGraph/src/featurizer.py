
import torch
from torch_geometric.data import Data
from rdkit import Chem


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
    return torch.tensor(features, dtype=torch.float)  # 41 dims total


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
    return torch.tensor(features, dtype=torch.float)  # 7 dims total


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.stack([get_atom_features(atom) for atom in mol.GetAtoms()])

    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_features += [feat, feat]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
