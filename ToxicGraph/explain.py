
import os
import yaml
import torch
import numpy as np
from matplotlib import cm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data as PyGData
from src.models import GNN
from src.featurizer import smiles_to_graph
from src.dataset import TOX21_TASKS

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    PYG_EXPLAINER_V2 = True
except ImportError:
    from torch_geometric.nn import GNNExplainer
    PYG_EXPLAINER_V2 = False

def explain_molecule(smiles, target_task_idx=0, output_dir='explanations'):
    os.makedirs(output_dir, exist_ok=True)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dummy_data = smiles_to_graph('CCO')  # molecule with bonds so edge_attr has shape[1]
    num_node_features = dummy_data.x.shape[1]
    edge_dim = dummy_data.edge_attr.shape[1]
    num_classes = 12 if config['dataset']['name'] == 'tox21' else 1

    model = GNN(
        num_node_features=num_node_features,
        hidden_channels=config['model']['hidden_channels'],
        num_classes=num_classes,
        edge_dim=edge_dim,
    ).to(device)

    model_path = 'model_0.pth' if os.path.exists('model_0.pth') else 'model.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print("No model file found. Please run train.py first.")
        return

    model.eval()

    data = smiles_to_graph(smiles)
    if data is None:
        print(f"Invalid SMILES: {smiles}")
        return

    data = data.to(device)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    task_name = TOX21_TASKS[target_task_idx] if target_task_idx < len(TOX21_TASKS) else str(target_task_idx)
    print(f"Explaining '{smiles}' for task {target_task_idx} ({task_name})...")

    mol = Chem.MolFromSmiles(smiles)

    if PYG_EXPLAINER_V2:
        # mode='regression' treats each output neuron independently — correct for multi-label
        # binary classification where outputs are NOT mutually exclusive (no softmax).
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='graph',
                return_type='raw',
            ),
        )
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
            target=torch.tensor([target_task_idx], device=device),
        )
        node_weights = explanation.node_mask.sum(dim=1).cpu().detach().numpy()
    else:
        explainer = GNNExplainer(model, epochs=200, return_type='raw')
        node_mask, _ = explainer.explain_graph(
            data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch
        )
        node_weights = node_mask.sum(dim=1).cpu().detach().numpy()

    save_path = os.path.join(output_dir, f'explanation_task{target_task_idx}.png')

    try:
        min_w, max_w = node_weights.min(), node_weights.max()
        norm_weights = (node_weights - min_w) / (max_w - min_w) if max_w > min_w else node_weights

        highlight_atom_colors = {i: cm.Reds(float(w))[:3] for i, w in enumerate(norm_weights)}

        img = Draw.MolToImage(
            mol,
            highlightAtoms=list(range(len(node_weights))),
            highlightAtomColors=highlight_atom_colors,
        )
        img.save(save_path)
        print(f"Saved explanation to {save_path}")
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Node weights:", node_weights)

def get_atom_importance_svg(smiles: str, task_idx: int, ensemble, task_names: list,
                            device) -> dict | None:
    """
    Gradient saliency for atom importance. Fast: one forward + backward pass.
    Returns {'atom_weights': [...], 'svg': '<svg>...', 'task_name': str} or None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    data = smiles_to_graph(smiles)
    if data is None:
        return None

    # Use first ensemble member for saliency (consistent, deterministic)
    model = ensemble.models[0]
    model.eval()

    # Build grad-enabled data object
    x_grad = data.x.clone().detach().to(device).requires_grad_(True)
    d = PyGData(
        x=x_grad,
        edge_index=data.edge_index.to(device),
        edge_attr=data.edge_attr.to(device),
        batch=torch.zeros(data.x.shape[0], dtype=torch.long, device=device),
        fp=data.fp.to(device) if hasattr(data, 'fp') and data.fp is not None else None,
    )

    out = model(d)  # (1, num_tasks)
    if task_idx >= out.shape[1]:
        task_idx = 0
    out[0, task_idx].backward()

    importance = x_grad.grad.abs().sum(dim=-1).detach().cpu().numpy()
    min_v, max_v = float(importance.min()), float(importance.max())
    norm = (importance - min_v) / (max_v - min_v + 1e-8)

    # Render: green colormap, heavier = more important
    colors = {i: cm.Greens(float(v) * 0.85 + 0.1)[:3] for i, v in enumerate(norm)}

    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(320, 260)
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().padding = 0.15
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=list(range(mol.GetNumAtoms())),
            highlightAtomColors=colors,
            highlightAtomRadii={i: 0.4 for i in range(mol.GetNumAtoms())},
        )
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    except Exception:
        # Fallback: plain molecule SVG without highlights
        drawer = rdMolDraw2D.MolDraw2DSVG(320, 260)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

    task_name = task_names[task_idx] if task_idx < len(task_names) else str(task_idx)
    return {
        'atom_weights': norm.tolist(),
        'svg': svg,
        'task_name': task_name,
    }


if __name__ == '__main__':
    caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    explain_molecule(caffeine_smiles, target_task_idx=0)
