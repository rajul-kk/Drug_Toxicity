
import yaml
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.dataset import ToxicDataset
from src.models import GNN
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

def train():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    dataset = ToxicDataset(root=config['dataset']['root'], name=config['dataset']['name'])
    dataset.shuffle()
    
    # Simple random split
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:]

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
                               num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False,
                             num_workers=4, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False,
                              num_workers=4, persistent_workers=True, pin_memory=True)

    # Model
    # Determine input dim from first data object
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes 
    # Note: dataset.num_classes in PyG often refers to number of classes per task if single label, 
    # but for multi-label binary classification it might just be the number of tasks usually stored in y.shape[1].
    # Let's check y shape.
    if dataset[0].y.dim() > 1:
        num_tasks = dataset[0].y.shape[1]
    else:
        num_tasks = 1

    model = GNN(num_node_features=num_node_features, 
                hidden_channels=config['model']['hidden_channels'], 
                num_classes=num_tasks).to(device)

    # Load existing model if available
    import os
    if os.path.exists('model.pth'):
        print("Loading existing model from model.pth...")
        model.load_state_dict(torch.load('model.pth', map_location=device))

    # Compile model for faster CPU/GPU kernel execution (PyTorch >= 2.0)
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0
        
        # Progress bar for the batch loop
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}')
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            
            # Masking NaNs (-1 in our dataset loader)
            y = data.y
            is_labeled = y > -0.5 # Filter out -1
            
            loss = criterion(out[is_labeled], y[is_labeled])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar with current batch loss
            pbar.set_postfix({'loss': loss.item()})

        # Validation
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                ys.append(data.y.cpu())
                preds.append(out.cpu())
        
        y_true = torch.cat(ys, dim=0)
        y_pred = torch.cat(preds, dim=0)

        # Compute ROC-AUC per task and average
        roc_scores = []
        for i in range(num_tasks):
            # Only evaluate on valid labels
            mask = y_true[:, i] > -0.5
            if mask.sum() > 0:
                try:
                    score = roc_auc_score(y_true[mask, i], y_pred[mask, i])
                    roc_scores.append(score)
                except ValueError:
                    pass
        
        val_auc = np.mean(roc_scores) if roc_scores else 0

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch:03d}, Avg Loss: {total_loss / len(train_loader):.4f}, Val AUC: {val_auc:.4f}, LR: {current_lr:.6f}')

    # Save model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

if __name__ == '__main__':
    train()
