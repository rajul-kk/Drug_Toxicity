
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Set2Set


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=4, edge_dim=8, fp_dim=2048):
        super().__init__()
        head_dim = hidden_channels // heads

        self.input_proj = nn.Linear(num_node_features, hidden_channels)

        self.conv1 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv3 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv4 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)

        self.set2set = Set2Set(hidden_channels, processing_steps=4)
        self.fp_dim = fp_dim
        self.lin = nn.Linear(2 * hidden_channels + fp_dim, num_classes)

        # nn.Dropout modules (not F.dropout) so enable_mc_dropout can flip them
        # independently of BatchNorm during MC uncertainty sampling
        self.dropout_mid = nn.Dropout(p=0.2)
        self.dropout_out = nn.Dropout(p=0.5)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        if edge_index is None and hasattr(x, 'x'):
            data = x
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            fp = getattr(data, 'fp', None)
        else:
            fp = None

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
        if self.fp_dim > 0 and fp is not None:
            x = torch.cat([x, fp], dim=-1)
        return self.lin(x)


class DMPNN(nn.Module):
    """
    Directed Message Passing Neural Network (Chemprop-style).

    Messages propagate along directed bonds. When updating edge (u→v), all edges
    entering u are aggregated except the reverse (v→u), preventing message echo.
    Uses the scatter-subtract trick: aggregate all incoming states per node, then
    subtract the single reverse-edge contribution — O(E) instead of O(E²).
    """

    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes, depth=4, fp_dim=2048):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.fp_dim = fp_dim

        self.W_i = nn.Linear(num_node_features + num_edge_features, hidden_channels)
        self.W_m = nn.Linear(hidden_channels, hidden_channels)
        self.W_a = nn.Linear(num_node_features + hidden_channels, hidden_channels)

        self.set2set = Set2Set(hidden_channels, processing_steps=4)
        self.dropout_mid = nn.Dropout(p=0.2)
        self.dropout_out = nn.Dropout(p=0.5)
        self.lin = nn.Linear(2 * hidden_channels + fp_dim, num_classes)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        if edge_index is None and hasattr(x, 'x'):
            data = x
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            fp = getattr(data, 'fp', None)
        else:
            fp = None

        num_nodes = x.size(0)

        if edge_index.size(1) == 0:
            zeros = x.new_zeros(num_nodes, self.hidden_channels)
            atom_out = self.W_a(torch.cat([x, zeros], dim=-1)).relu()
            return self.lin(self.dropout_out(self.set2set(atom_out, batch)))

        src, dst = edge_index[0], edge_index[1]
        num_edges = edge_index.size(1)
        # Featurizer stores (u,v) and (v,u) as consecutive pairs, so rev of edge e is e^1
        rev_idx = torch.arange(num_edges, device=x.device) ^ 1

        h_init = self.W_i(torch.cat([x[src], edge_attr], dim=-1)).relu()
        if h_init.isnan().any():
            raise RuntimeError(f'NaN in h_init — x_nan={x.isnan().any().item()} ea_nan={edge_attr.isnan().any().item()}')
        h = h_init

        for i in range(self.depth):
            # Sum all incoming edge states per destination node (pure PyTorch, no scatter dep)
            agg = torch.zeros(num_nodes, h.size(-1), device=h.device, dtype=h.dtype)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(h), h)
            # For edge (u→v): subtract reverse edge (v→u) to exclude it
            msg = agg[src] - h[rev_idx]
            h = F.relu(h_init + self.W_m(msg))
            if h.isnan().any():
                raise RuntimeError(f'NaN in h after step {i} — msg_nan={msg.isnan().any().item()} agg_nan={agg.isnan().any().item()}')
            h = self.dropout_mid(h)

        atom_msg = torch.zeros(num_nodes, h.size(-1), device=h.device, dtype=h.dtype)
        atom_msg.scatter_add_(0, dst.unsqueeze(-1).expand_as(h), h)
        atom_out = self.W_a(torch.cat([x, atom_msg], dim=-1)).relu()
        if atom_out.isnan().any():
            raise RuntimeError(f'NaN in atom_out — atom_msg_nan={atom_msg.isnan().any().item()}')

        out = self.set2set(atom_out, batch)
        if out.isnan().any():
            raise RuntimeError('NaN in set2set_out')
        out = self.dropout_out(out)
        if self.fp_dim > 0 and fp is not None:
            out = torch.cat([out, fp], dim=-1)
        return self.lin(out)


class EnsembleGNN(nn.Module):
    """Wraps multiple GNN models and returns their mean prediction."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        return torch.stack([m(*args, **kwargs) for m in self.models]).mean(0)
