
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from src.dataset import ToxicDataset
from src.models import GNN, EnsembleGNN
from src.splitter import scaffold_split

TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
]


def collect_preds(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            ys.append(data.y.cpu())
            preds.append(model(data).cpu())
    return torch.cat(ys, dim=0), torch.cat(preds, dim=0)


def eval_auc(model, loader, num_tasks, device):
    y_true, y_pred = collect_preds(model, loader, device)
    scores = []
    for i in range(num_tasks):
        mask = y_true[:, i] > -0.5
        if mask.sum() > 0:
            try:
                scores.append(roc_auc_score(y_true[mask, i], y_pred[mask, i]))
            except ValueError:
                pass
    return float(np.mean(scores)) if scores else 0.0


def eval_full_metrics(model, loader, num_tasks, device, task_names=None):
    y_true, y_pred = collect_preds(model, loader, device)
    task_names = task_names or [str(i) for i in range(num_tasks)]

    aucs, auprcs = [], []
    rows = []
    for i in range(num_tasks):
        mask = y_true[:, i] > -0.5
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask, i].numpy(), y_pred[mask, i].numpy()
        try:
            auc = roc_auc_score(yt, yp)
            auprc = average_precision_score(yt, yp)
            aucs.append(auc)
            auprcs.append(auprc)
            rows.append((task_names[i], auc, auprc))
        except ValueError:
            pass

    print(f"\n{'Task':<20} {'AUC':>6}  {'AUPRC':>6}")
    print('-' * 36)
    for name, auc, auprc in rows:
        print(f"{name:<20} {auc:.4f}  {auprc:.4f}")
    print('-' * 36)
    print(f"{'Mean':<20} {np.mean(aucs):.4f}  {np.mean(auprcs):.4f}\n")


def train_single(model, train_loader, val_loader, num_tasks, device, config, run_idx):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'[{run_idx + 1}] Epoch {epoch:03d}')
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            y = data.y
            is_labeled = y > -0.5
            loss = criterion(out[is_labeled], y[is_labeled])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        val_auc = eval_auc(model, val_loader, num_tasks, device)
        scheduler.step(val_auc)
        lr = optimizer.param_groups[0]['lr']
        print(f'[{run_idx + 1}] Epoch {epoch:03d}  Loss {total_loss / len(train_loader):.4f}  Val AUC {val_auc:.4f}  LR {lr:.6f}')

    return model


def train():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = ToxicDataset(root=config['dataset']['root'], name=config['dataset']['name'])
    train_dataset, val_dataset, test_dataset = scaffold_split(dataset)
    print(f"Scaffold split: {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test")

    loader_kwargs = dict(
        batch_size=config['training']['batch_size'],
        num_workers=4,
        persistent_workers=True,
        pin_memory=device.type == 'cuda',
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    num_node_features = dataset.num_node_features
    edge_dim = dataset[0].edge_attr.shape[1]
    num_tasks = dataset[0].y.shape[1] if dataset[0].y.dim() > 1 else 1
    task_names = TOX21_TASKS if config['dataset']['name'] == 'tox21' else None
    ensemble_size = config['model'].get('ensemble_size', 3)

    for run_idx in range(ensemble_size):
        print(f"\n{'=' * 55}")
        print(f"  Training model {run_idx + 1} / {ensemble_size}")
        print(f"{'=' * 55}")
        torch.manual_seed(run_idx)

        model = GNN(
            num_node_features=num_node_features,
            hidden_channels=config['model']['hidden_channels'],
            num_classes=num_tasks,
            edge_dim=edge_dim,
        ).to(device)

        save_path = f'model_{run_idx}.pth'
        if os.path.exists(save_path):
            try:
                model.load_state_dict(torch.load(save_path, map_location=device))
                print(f"Resumed from {save_path}")
            except RuntimeError as e:
                print(f"Warning: incompatible checkpoint, starting fresh.\n  ({e})")

        if device.type == 'cuda':
            model = torch.compile(model)

        model = train_single(model, train_loader, val_loader, num_tasks, device, config, run_idx)

        # Save via state_dict so it's compatible with uncompiled GNN at load time
        state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        torch.save(state, save_path)
        print(f"Saved {save_path}")

    # Build ensemble from saved weights
    ensemble_models = []
    for run_idx in range(ensemble_size):
        m = GNN(num_node_features, config['model']['hidden_channels'], num_tasks, edge_dim=edge_dim).to(device)
        m.load_state_dict(torch.load(f'model_{run_idx}.pth', map_location=device))
        ensemble_models.append(m)
    ensemble = EnsembleGNN(ensemble_models)

    print("\n=== Ensemble Test Results ===")
    eval_full_metrics(ensemble, test_loader, num_tasks, device, task_names)


if __name__ == '__main__':
    train()
