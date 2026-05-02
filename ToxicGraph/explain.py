
import os
import yaml
import torch
from matplotlib import cm
from rdkit import Chem
from rdkit.Chem import Draw
from src.models import GNN
from src.featurizer import smiles_to_graph

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    PYG_EXPLAINER_V2 = True
except ImportError:
    from torch_geometric.nn import GNNExplainer
    PYG_EXPLAINER_V2 = False

TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
]

def explain_molecule(smiles, target_task_idx=0, output_dir='explanations'):
    os.makedirs(output_dir, exist_ok=True)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dummy_data = smiles_to_graph('C')
    num_node_features = dummy_data.x.shape[1]
    num_classes = 12 if config['dataset']['name'] == 'tox21' else 1

    model = GNN(
        num_node_features=num_node_features,
        hidden_channels=config['model']['hidden_channels'],
        num_classes=num_classes,
    ).to(device)

    try:
        model.load_state_dict(torch.load('model.pth', map_location=device))
    except FileNotFoundError:
        print("model.pth not found. Please run train.py first.")
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

if __name__ == '__main__':
    caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    explain_molecule(caffeine_smiles, target_task_idx=0)
