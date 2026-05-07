
import os
import yaml
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from src.dataset import ToxicDataset
from src.models import GNN, DMPNN, EnsembleGNN
from src.splitter import scaffold_split
from src.utils import visualize_molecule_3d
from train import eval_full_metrics, TOX21_TASKS

OUT_DIR = 'evaluations'
TARGET_IMAGES = 20


def load_ensemble(config, device):
    dataset = ToxicDataset(root=config['dataset']['root'], name=config['dataset']['name'])
    _, _, test_dataset = scaffold_split(dataset)

    num_node_features = dataset.num_node_features
    edge_dim = dataset[0].edge_attr.shape[1]
    num_tasks = dataset[0].y.shape[1]
    hidden = config['model']['hidden_channels']
    depth = config['model'].get('depth', 4)
    model_type = config['model'].get('type', 'gnn')
    ensemble_size = config['model'].get('ensemble_size', 3)

    models = []
    for i in range(ensemble_size):
        path = f'model_{i}.pth'
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found — run train.py first')
        if model_type == 'dmpnn':
            m = DMPNN(num_node_features, edge_dim, hidden, num_tasks, depth=depth).to(device)
        else:
            m = GNN(num_node_features, hidden, num_tasks, edge_dim=edge_dim).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)

    return EnsembleGNN(models), test_dataset


def collect_predictions(ensemble, test_dataset, device, temperature):
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    all_logits, all_y = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            all_logits.append(ensemble(data).cpu())
            all_y.append(data.y.cpu())
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    probs = torch.sigmoid(logits / temperature)
    return probs.numpy(), y.numpy()


def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temperature = float(torch.load('temperature.pt', map_location='cpu')) if os.path.exists('temperature.pt') else 1.0
    print(f'Temperature: {temperature:.4f}')

    ensemble, test_dataset = load_ensemble(config, device)

    # ── metrics ───────────────────────────────────────────────────────────────
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    task_names = TOX21_TASKS if config['dataset']['name'] == 'tox21' else None
    num_tasks = test_dataset[0].y.shape[1]
    print('\n=== Test Set Metrics ===')
    eval_full_metrics(ensemble, loader, num_tasks, device, task_names, temperature=temperature)

    # ── 3D visualisations ────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    probs, _ = collect_predictions(ensemble, test_dataset, device, temperature)

    test_smiles = test_dataset.smiles_list

    # Rank by confidence (max prob across tasks) and attempt 3D embedding
    max_prob = probs.max(axis=1)
    ranked = np.argsort(max_prob)[::-1]

    saved = 0
    attempted = 0
    print(f'\nGenerating 3D images → {OUT_DIR}/')
    for rank, idx in enumerate(ranked):
        if saved >= TARGET_IMAGES:
            break
        smiles = test_smiles[idx]
        attempted += 1
        path = os.path.join(OUT_DIR, f'mol_{rank:03d}_p{max_prob[idx]:.3f}.png')
        fig = visualize_molecule_3d(smiles, save_path=path)
        if fig is not None:
            saved += 1
            print(f'  [{saved:2d}/{TARGET_IMAGES}] {smiles[:60]}  (max_prob={max_prob[idx]:.3f})')

    print(f'\nSaved {saved} images from {attempted} attempts.')


if __name__ == '__main__':
    main()
