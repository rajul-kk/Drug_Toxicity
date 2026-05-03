
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Set2Set


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=4, edge_dim=8):
        super().__init__()
        head_dim = hidden_channels // heads

        # Input projection to hidden_channels so residuals have the right shape
        self.input_proj = nn.Linear(num_node_features, hidden_channels)

        self.conv1 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv3 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv4 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)

        # Set2Set iteratively refines a graph-level query vector via attention
        # over all node embeddings; output dim = 2 * hidden_channels
        self.set2set = Set2Set(hidden_channels, processing_steps=4)
        self.lin = nn.Linear(2 * hidden_channels, num_classes)

        # nn.Dropout modules (not F.dropout) so enable_mc_dropout can flip them
        # independently of BatchNorm during MC uncertainty sampling
        self.dropout_mid = nn.Dropout(p=0.2)
        self.dropout_out = nn.Dropout(p=0.5)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        if edge_index is None and hasattr(x, 'x'):
            data = x
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.input_proj(x).relu()

        x = self.bn1((x + self.conv1(x, edge_index, edge_attr=edge_attr)).relu())
        x = self.dropout_mid(x)
        x = self.bn2((x + self.conv2(x, edge_index, edge_attr=edge_attr)).relu())
        x = self.dropout_mid(x)
        x = self.bn3((x + self.conv3(x, edge_index, edge_attr=edge_attr)).relu())
        x = self.dropout_mid(x)
        x = self.bn4((x + self.conv4(x, edge_index, edge_attr=edge_attr)).relu())

        x = self.set2set(x, batch)          # (B, 2 * hidden_channels)
        x = self.dropout_out(x)
        return self.lin(x)


class EnsembleGNN(nn.Module):
    """Wraps multiple GNN models and returns their mean prediction."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        return torch.stack([m(*args, **kwargs) for m in self.models]).mean(0)
