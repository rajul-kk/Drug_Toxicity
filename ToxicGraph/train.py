
import os
import yaml
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.dataset import ToxicDataset
from src.models import GNN
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score


def eval_auc(model, loader, num_tasks, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            ys.append(data.y.cpu())
            preds.append(model(data).cpu())

    y_true = torch.cat(ys, dim=0)
    y_pred = torch.cat(preds, dim=0)

    scores = []
    for i in range(num_tasks):
        mask = y_true[:, i] > -0.5
        if mask.sum() > 0:
            try:
                scores.append(roc_auc_score(y_true[mask, i], y_pred[mask, i]))
            except ValueError:
                pass
    return float(np.mean(scores)) if scores else 0.0


def train():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = ToxicDataset(root=config['dataset']['root'], name=config['dataset']['name'])
    dataset.shuffle()

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

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
    num_tasks = dataset[0].y.shape[1] if dataset[0].y.dim() > 1 else 1

    model = GNN(
        num_node_features=num_node_features,
        hidden_channels=config['model']['hidden_channels'],
        num_classes=num_tasks,
    ).to(device)

    if os.path.exists('model.pth'):
        try:
            model.load_state_dict(torch.load('model.pth', map_location=device))
            print("Resumed from model.pth")
        except RuntimeError as e:
            print(f"Warning: model.pth is incompatible with current architecture, starting fresh.\n  ({e})")

    if device.type == 'cuda':
        model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}')
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
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch:03d}, Loss: {total_loss / len(train_loader):.4f}, Val AUC: {val_auc:.4f}, LR: {current_lr:.6f}')

    # Save before test eval so we always have weights even if something fails
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

    test_auc = eval_auc(model, test_loader, num_tasks, device)
    print(f'Test AUC: {test_auc:.4f}')


if __name__ == '__main__':
    train()
