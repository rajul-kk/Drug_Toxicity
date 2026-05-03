
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Set2Set


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=4, edge_dim=8):
        super().__init__()
        head_dim = hidden_channels // heads

        self.conv1 = GATv2Conv(num_node_features, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv3 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Set2Set iteratively refines a graph-level query vector via attention
        # over all node embeddings; output dim = 2 * hidden_channels
        self.set2set = Set2Set(hidden_channels, processing_steps=4)
        self.lin = nn.Linear(2 * hidden_channels, num_classes)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        if edge_index is None and hasattr(x, 'x'):
            data = x
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr).relu())
        x = self.bn2(self.conv2(x, edge_index, edge_attr=edge_attr).relu())
        x = self.conv3(x, edge_index, edge_attr=edge_attr)

        x = self.set2set(x, batch)          # (B, 2 * hidden_channels)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


class EnsembleGNN(nn.Module):
    """Wraps multiple GNN models and returns their mean prediction."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        return torch.stack([m(*args, **kwargs) for m in self.models]).mean(0)
