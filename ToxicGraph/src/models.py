
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=4):
        super(GNN, self).__init__()
        edge_dim = 7
        head_dim = hidden_channels // heads

        self.conv1 = GATv2Conv(num_node_features, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv3 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        if edge_index is None and hasattr(x, 'x'):
            data = x
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr).relu())
        x = self.bn2(self.conv2(x, edge_index, edge_attr=edge_attr).relu())
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = global_add_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


class EnsembleGNN(nn.Module):
    """Wraps multiple GNN models and returns their mean prediction."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        return torch.stack([m(*args, **kwargs) for m in self.models]).mean(0)
