
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import matplotlib.pyplot as plt
import numpy as np

# CPK element colors and relative atom sizes
_CPK_COLORS = {
    1:  '#DDDDDD',  # H
    6:  '#404040',  # C
    7:  '#3050F8',  # N
    8:  '#FF0D0D',  # O
    9:  '#90E050',  # F
    15: '#FF8000',  # P
    16: '#FFFF30',  # S
    17: '#1FF01F',  # Cl
    35: '#A62929',  # Br
    53: '#940094',  # I
}
_ATOM_SIZE = {1: 80, 6: 180, 7: 160, 8: 150, 9: 130, 15: 220, 16: 220, 17: 200, 35: 240, 53: 260}


@lru_cache(maxsize=512)
def smiles_to_sdf(smiles: str) -> str | None:
    """Return an SDF/molblock string for 3Dmol.js; None if embedding fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return Chem.MolToMolBlock(mol)


def visualize_molecule(smiles, save_path=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    img = Draw.MolToImage(mol)
    if save_path:
        img.save(save_path)
        print(f"Saved 2D visualization to {save_path}")
    return img


def visualize_molecule_3d(smiles, save_path=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        print(f"3D embedding failed for: {smiles}")
        return None
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    pos = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

    fig = plt.figure(figsize=(7, 7), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            [pos[i, 2], pos[j, 2]],
            color='#888888', linewidth=2, zorder=1,
        )

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        an = atom.GetAtomicNum()
        ax.scatter(
            *pos[idx],
            s=_ATOM_SIZE.get(an, 180),
            c=_CPK_COLORS.get(an, '#FF69B4'),
            edgecolors='#222222',
            linewidths=0.4,
            depthshade=True,
            zorder=2,
        )

    ax.set_title(smiles, fontsize=8)
    ax.set_axis_off()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D visualization to {save_path}")

    plt.close()
    return fig
