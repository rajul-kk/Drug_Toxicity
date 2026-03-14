
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=4):
        super(GNN, self).__init__()
        # Edge dim is 7 based on featurizer implementation
        edge_dim = 7
        # Each head outputs hidden_channels // heads so concat stays at hidden_channels
        head_dim = hidden_channels // heads

        # 3 Graph Attention Layers with multi-head attention
        # GATv2Conv natively handles edge_attr
        self.conv1 = GATv2Conv(num_node_features, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv3 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)

        # BatchNorm after conv1 and conv2 for stable training
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        # Output layer
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        # Support for passing a data object directly
        if batch is None and edge_index is None and hasattr(x, 'x'):
             data = x
             x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 1. Obtain node embeddings
        x = self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr).relu())
        x = self.bn2(self.conv2(x, edge_index, edge_attr=edge_attr).relu())
        x = self.conv3(x, edge_index, edge_attr=edge_attr)

        # 2. Readout layer
        # Aggregates node features into a graph representation
        # global_add_pool often works better for molecule size/count features
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
