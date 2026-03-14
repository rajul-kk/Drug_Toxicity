
from rdkit import Chem
from rdkit.Chem import Draw

def visualize_molecule(smiles, save_path=None):
    """
    Visualizes a molecule from a SMILES string.
    
    Args:
        smiles (str): The SMILES string of the molecule.
        save_path (str, optional): Path to save the image. If None, the image is not saved.
        
    Returns:
        PIL.Image.Image: The image of the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None
    
    img = Draw.MolToImage(mol)
    
    if save_path:
        img.save(save_path)
        print(f"Saved visualization to {save_path}")
        
    return img
