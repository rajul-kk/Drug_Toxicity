
import os
import pathlib
import argparse
import datetime
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from src.dataset import load_dataset, TOX21_TASKS
from src.models import GNN, DMPNN, EnsembleGNN
from src.splitter import scaffold_split, multidataset_scaffold_split
from src.calibration import fit_temperature, compute_ece, compute_brier


def compute_pos_weights(train_dataset, num_tasks, device):
    """
    Per-task pos_weight = neg_count / pos_count, clamped to [1, 10].
    Missing labels (-1) are excluded. Tasks with no positives keep weight 1.
    """
    all_y = train_dataset.data.y          # (N, T) — InMemoryDataset mega-tensor
    pos_weight = torch.ones(num_tasks)
    for i in range(num_tasks):
        mask = all_y[:, i] > -0.5         # exclude missing labels
        if mask.sum() == 0:
            continue
        labeled = all_y[mask, i]
        pos = (labeled > 0.5).float().sum()
        neg = (labeled < 0.5).float().sum()
        if pos > 0:
            pos_weight[i] = (neg / pos).clamp(max=10.0)
    return pos_weight.to(device)


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


def focal_loss(logits, targets, gamma=2.0, weights=None, pos_weight=None):
    """Binary focal loss with optional per-element class and dataset weights."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.exp(-bce)
    focal = (1 - p_t) ** gamma * bce
    if pos_weight is not None:
        # Upweight positive-class losses; negatives unchanged
        focal = focal * torch.where(targets > 0.5, pos_weight, torch.ones_like(pos_weight))
    if weights is not None:
        focal = focal * weights
    return focal.mean()


def train_single(model, train_loader, val_loader, num_tasks, device, config, run_idx,
                 task_weight_vec=None, pos_weight_vec=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs'], eta_min=1e-6
    )
    patience = config['training'].get('early_stopping_patience', 10)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_auc = -1.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0

    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'[{run_idx + 1}] Epoch {epoch:03d}')
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(data)
                y = data.y
                is_labeled = y > -0.5
                t_idx = torch.arange(y.shape[1], device=device).unsqueeze(0).expand_as(y)
                labeled_t_idx = t_idx[is_labeled]
                sample_w  = task_weight_vec[labeled_t_idx] if task_weight_vec is not None else None
                elem_pw   = pos_weight_vec[labeled_t_idx]  if pos_weight_vec  is not None else None
                loss = focal_loss(out[is_labeled], y[is_labeled], weights=sample_w, pos_weight=elem_pw)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        val_auc = eval_auc(model, val_loader, num_tasks, device)
        scheduler.step()
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=None,
                        help='Directory to save weights (default: checkpoints/<model_type>)')
    args, _ = parser.parse_known_args()

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = load_dataset(config)
    if hasattr(dataset, 'source_list'):
        print("Per-dataset scaffold split:")
        train_dataset, val_dataset, test_dataset = multidataset_scaffold_split(dataset)
    else:
        train_dataset, val_dataset, test_dataset = scaffold_split(dataset)
    print(f"Total: {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test")

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

    primary_tasks = dataset.primary_tasks
    eval_num_tasks = len(primary_tasks)   # used during training for val AUC / early stopping
    all_tasks = dataset.all_tasks if hasattr(dataset, 'all_tasks') else primary_tasks
    ensemble_size = config['model'].get('ensemble_size', 3)
    hidden = config['model']['hidden_channels']
    model_type = config['model'].get('type', 'gnn')
    depth = config['model'].get('depth', 4)
    task_dim = config['model'].get('task_dim', 64)
    fp_dim = config['model'].get('fp_dim', 64)

    date_str = datetime.date.today().strftime('%Y%m%d')
    ds_key = '-'.join(n[:3] for n in config['dataset']['names'])
    out_dir = args.out_dir or os.path.join('checkpoints', f'{model_type}_{ds_key}_{date_str}')
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    pos_weight_vec = compute_pos_weights(train_dataset, num_tasks, device)
    print(f"pos_weight range: min={pos_weight_vec.min():.2f}  max={pos_weight_vec.max():.2f}  "
          f"mean={pos_weight_vec.mean():.2f}")

    # Build per-task loss weight vector from dataset_weights config (multi-dataset only)
    task_weight_vec = None
    if hasattr(dataset, 'task_ranges'):
        dw = config['training'].get('dataset_weights', {})
        if dw:
            wv = torch.ones(num_tasks)
            for name, (start, end) in dataset.task_ranges.items():
                wv[start:end] = float(dw.get(name, 1.0))
            task_weight_vec = wv.to(device)
            print(f"Task weights: { {n: float(dw.get(n, 1.0)) for n in dataset.task_ranges} }")

    def build_model():
        if model_type == 'dmpnn':
            return DMPNN(num_node_features, edge_dim, hidden, num_tasks, depth=depth, task_dim=task_dim, fp_dim=fp_dim).to(device)
        return GNN(num_node_features, hidden, num_tasks, edge_dim=edge_dim, task_dim=task_dim, fp_dim=fp_dim).to(device)

    print(f"Model: {model_type.upper()}  hidden={hidden}  task_dim={task_dim}  depth/layers={depth}  ensemble={ensemble_size}")

    for run_idx in range(ensemble_size):
        print(f"\n{'=' * 55}")
        print(f"  Training model {run_idx + 1} / {ensemble_size}")
        print(f"{'=' * 55}")
        torch.manual_seed(run_idx)

        model = build_model()

        save_path = os.path.join(out_dir, f'model_{run_idx}.pth')
        if os.path.exists(save_path):
            try:
                model.load_state_dict(torch.load(save_path, map_location=device))
                print(f"Resumed from {save_path}")
            except RuntimeError as e:
                print(f"Warning: incompatible checkpoint, starting fresh.\n  ({e})")

        if device.type == 'cuda':
            model = torch.compile(model)

        model = train_single(model, train_loader, val_loader, eval_num_tasks, device, config,
                             run_idx, task_weight_vec, pos_weight_vec)

        # Save via state_dict so it's compatible with uncompiled model at load time
        state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        torch.save(state, save_path)
        print(f"Saved {save_path}")

    # Build ensemble from saved weights
    ensemble_models = []
    for run_idx in range(ensemble_size):
        m = build_model()
        m.load_state_dict(torch.load(os.path.join(out_dir, f'model_{run_idx}.pth'), map_location=device))
        ensemble_models.append(m)
    ensemble = EnsembleGNN(ensemble_models)

    print("\n=== Ensemble Test Results ===")
    eval_full_metrics(ensemble, test_loader, num_tasks, device, all_tasks)

    print("Fitting temperature scaler on validation set...")
    temperature = fit_temperature(ensemble, val_loader, device)
    torch.save(temperature, os.path.join(out_dir, 'temperature.pt'))
    import json as _json
    _metadata = {
        'arch':           model_type,
        'datasets':       config['dataset']['names'],
        'date':           date_str,
        'hidden_channels': hidden,
        'ensemble_size':  ensemble_size,
        'fp_dim':         fp_dim,
        'fp_bits':        config['model'].get('fp_bits', 679),
        'depth':          depth,
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as _f:
        _json.dump(_metadata, _f, indent=2)
    print(f"Saved metadata.json to {out_dir}")
    print(f"\n=== Calibrated Test Results (T={temperature:.4f}) ===")
    eval_full_metrics(ensemble, test_loader, num_tasks, device, all_tasks, temperature=temperature)


if __name__ == '__main__':
    train()
