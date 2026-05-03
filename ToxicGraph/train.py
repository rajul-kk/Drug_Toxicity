
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from src.dataset import ToxicDataset
from src.models import GNN, EnsembleGNN
from src.splitter import scaffold_split
from src.calibration import fit_temperature, compute_ece, compute_brier

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


def eval_full_metrics(model, loader, num_tasks, device, task_names=None, temperature=1.0):
    y_true, y_logits = collect_preds(model, loader, device)
    y_probs = torch.sigmoid(y_logits / temperature)
    task_names = task_names or [str(i) for i in range(num_tasks)]

    aucs, auprcs, briers, eces = [], [], [], []
    rows = []
    for i in range(num_tasks):
        mask = y_true[:, i] > -0.5
        if mask.sum() == 0:
            continue
        yt = y_true[mask, i].numpy()
        yp_logit = y_logits[mask, i].numpy()
        yp_prob = y_probs[mask, i].numpy()
        try:
            auc = roc_auc_score(yt, yp_logit)
            auprc = average_precision_score(yt, yp_logit)
            brier = compute_brier(yp_prob, yt)
            ece = compute_ece(yp_prob, yt)
            aucs.append(auc)
            auprcs.append(auprc)
            briers.append(brier)
            eces.append(ece)
            rows.append((task_names[i], auc, auprc, brier, ece))
        except ValueError:
            pass

    print(f"\n{'Task':<20} {'AUC':>6}  {'AUPRC':>6}  {'Brier':>6}  {'ECE':>6}")
    print('-' * 54)
    for name, auc, auprc, brier, ece in rows:
        print(f"{name:<20} {auc:.4f}  {auprc:.4f}  {brier:.4f}  {ece:.4f}")
    print('-' * 54)
    print(f"{'Mean':<20} {np.mean(aucs):.4f}  {np.mean(auprcs):.4f}  {np.mean(briers):.4f}  {np.mean(eces):.4f}\n")


def focal_loss(logits, targets, gamma=2.0):
    """Binary focal loss: down-weights easy negatives to address class imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.exp(-bce)
    return ((1 - p_t) ** gamma * bce).mean()


def train_single(model, train_loader, val_loader, num_tasks, device, config, run_idx):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    patience = config['training'].get('early_stopping_patience', 10)

    best_val_auc = 0.0
    best_state = None
    epochs_no_improve = 0

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
            loss = focal_loss(out[is_labeled], y[is_labeled])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        val_auc = eval_auc(model, val_loader, num_tasks, device)
        scheduler.step(val_auc)
        lr = optimizer.param_groups[0]['lr']
        print(f'[{run_idx + 1}] Epoch {epoch:03d}  Loss {total_loss / len(train_loader):.4f}  Val AUC {val_auc:.4f}  LR {lr:.6f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'[{run_idx + 1}] Early stop at epoch {epoch} (best val AUC {best_val_auc:.4f})')
                break

    model.load_state_dict(best_state)
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

    print("Fitting temperature scaler on validation set...")
    temperature = fit_temperature(ensemble, val_loader, device)
    torch.save(temperature, 'temperature.pt')
    print(f"\n=== Calibrated Test Results (T={temperature:.4f}) ===")
    eval_full_metrics(ensemble, test_loader, num_tasks, device, task_names, temperature=temperature)


if __name__ == '__main__':
    train()
