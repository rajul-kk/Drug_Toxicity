
import os
import yaml
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from src.models import GNN
from src.featurizer import smiles_to_graph

# Try importing Explainer (PyG 2.2+)
try:
    from torch_geometric.explain import Explainer, GNNExplainer
    PYG_EXPLAINER_V2 = True
except ImportError:
    from torch_geometric.nn import GNNExplainer
    PYG_EXPLAINER_V2 = False

def explain_molecule(smiles, target_task_idx=0, output_dir='explanations'):
    """
    Explain the prediction for a specific molecule and task.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup Model
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Auto-detect input dimension from a dummy molecule
    dummy_data = smiles_to_graph('C')
    num_node_features = dummy_data.x.shape[1]
    print(f"Detected {num_node_features} input features.")

    num_classes = 12 if config['dataset']['name'] == 'tox21' else 1
    
    model = GNN(num_node_features=num_node_features, 
                hidden_channels=config['model']['hidden_channels'], 
                num_classes=num_classes).to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load('model.pth', map_location=device))
    except FileNotFoundError:
        print("Model file not found. Please train first.")
        return

    model.eval()

    # 2. Prepare Data
    data = smiles_to_graph(smiles)
    if data is None:
        print("Invalid SMILES")
        return
        
    data = data.to(device)
    # Add batch dim for the model forward pass (Expects batch index)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)

    # 3. Explain
    print(f"Explaining SMILES: {smiles} for Task {target_task_idx}...")
    
    mol = Chem.MolFromSmiles(smiles)
    
    if PYG_EXPLAINER_V2:
        # Configuration for Explainer
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification', # Treating each task output as a logit
                task_level='graph',
                return_type='raw',
            ),
        )
        
        # We want to explain the output for the specific task index
        # For multi-label, we can look at the specific output neuron.
        explanation = explainer(
            x=data.x, 
            edge_index=data.edge_index, 
            edge_attr=data.edge_attr, 
            batch=data.batch,
            target=torch.tensor([target_task_idx]).to(device) # Target class index implies which output neuron we care about? 
            # Actually for multi-label regression/binary classification 'target' usually selects the output index 
            # if using multiclass mode or if the return_type is raw.
        )
        
        node_mask = explanation.node_mask
        # Reduce node mask to scalar per node (sum over features)
        node_weights = node_mask.sum(dim=1).cpu().detach().numpy()
        
    else:
        # Legacy GNNExplainer
        explainer = GNNExplainer(model, epochs=200, return_type='raw')
        node_mask, edge_mask = explainer.explain_graph(
            data.x, 
            data.edge_index, 
            edge_attr=data.edge_attr, 
            batch=data.batch
        )
        # Note: Legacy returns masks for the predicted class usually? 
        # Or we might need to specify it. 
        # For simplicity, we assume generic importance.
        node_weights = node_mask.sum(dim=1).cpu().detach().numpy()

    # Normalize weights for visualization
    # Optional: Min-Max normalization or just pass raw weights to SimilarityMap
    
    # 4. Visualize
    # RDKit SimilarityMaps expects weights for each atom.
    # Note: rdkit atom order must match data.x order. 
    # smiles_to_graph iterates mol.GetAtoms(), so order should be preserved.
    
    save_path = os.path.join(output_dir, f'explanation_task{target_task_idx}.png')
    
    # Visualization with fallback to MolToImage
    try:
        from rdkit.Chem import Draw
        from matplotlib import cm
        
        # Normalize weights to 0-1 for coloring
        min_w = node_weights.min()
        max_w = node_weights.max()
        if max_w > min_w:
            norm_weights = (node_weights - min_w) / (max_w - min_w)
        else:
            norm_weights = node_weights # All same
            
        # Create color map
        highlight_atom_colors = {}
        for i, w in enumerate(norm_weights):
            # Use a colormap (e.g., Reds)
            # cm.Reds(w) returns (r, g, b, a)
            color = cm.Reds(w)[:3] 
            highlight_atom_colors[i] = color
            
        img = Draw.MolToImage(mol, highlightAtoms=range(len(node_weights)), highlightAtomColors=highlight_atom_colors)
        img.save(save_path)
        print(f"Saved explanation to {save_path}")

    except Exception as e:
        print(f"Visualization failed: {e}")
        with open('error_log.txt', 'w') as f:
            f.write(str(e))
        print("Node weights:", node_weights)

if __name__ == '__main__':
    # Test with Caffeine
    # Task 0 in Tox21 is usually NR-AR (Androgen Receptor)
    caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' 
    explain_molecule(caffeine_smiles, target_task_idx=0)
