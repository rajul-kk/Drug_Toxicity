
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Set2Set


class TaskHead(nn.Module):
    """
    Task-conditioned readout: each task has a learned embedding that is concatenated
    with the molecule embedding before scoring. Enables few-shot transfer to new tasks
    by fine-tuning only the new task embedding while freezing the encoder.
    """
    def __init__(self, mol_dim, num_tasks, task_dim):
        super().__init__()
        self.task_emb = nn.Embedding(num_tasks, task_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mol_dim + task_dim, mol_dim),
            nn.ReLU(),
            nn.Linear(mol_dim, 1),
        )

    def forward(self, mol):
        # mol: (B, mol_dim)
        T = self.task_emb.weight                                    # (T, task_dim)
        mol_exp = mol.unsqueeze(1).expand(-1, T.size(0), -1)       # (B, T, mol_dim)
        task_exp = T.unsqueeze(0).expand(mol.size(0), -1, -1)      # (B, T, task_dim)
        return self.mlp(torch.cat([mol_exp, task_exp], dim=-1)).squeeze(-1)  # (B, T)


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes,
                 heads=4, edge_dim=8, task_dim=64, fp_dim=64):
        super().__init__()
        head_dim = hidden_channels // heads

        self.input_proj = nn.Linear(num_node_features, hidden_channels)

        self.conv1 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv3 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)
        self.conv4 = GATv2Conv(hidden_channels, head_dim, edge_dim=edge_dim, heads=heads)

        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.ln3 = nn.LayerNorm(hidden_channels)
        self.ln4 = nn.LayerNorm(hidden_channels)

        self.set2set = Set2Set(hidden_channels, processing_steps=4)
        self.dropout_mid = nn.Dropout(p=0.2)
        self.dropout_out = nn.Dropout(p=0.5)

        self.fp_dim = fp_dim
        if fp_dim > 0:
            self.fp_proj = nn.Sequential(nn.Linear(512, fp_dim), nn.ReLU())
        self.task_head = TaskHead(2 * hidden_channels + fp_dim, num_classes, task_dim)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        if edge_index is None and hasattr(x, 'x'):
            data = x
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            fp = getattr(data, 'fp', None)
        else:
            fp = None

        x = self.input_proj(x).relu()

        x = self.ln1((x + self.conv1(x, edge_index, edge_attr=edge_attr)).relu())
        x = self.dropout_mid(x)
        x = self.ln2((x + self.conv2(x, edge_index, edge_attr=edge_attr)).relu())
        x = self.dropout_mid(x)
        x = self.ln3((x + self.conv3(x, edge_index, edge_attr=edge_attr)).relu())
        x = self.dropout_mid(x)
        x = self.ln4((x + self.conv4(x, edge_index, edge_attr=edge_attr)).relu())

        mol = self.set2set(x, batch)        # (B, 2H)
        mol = self.dropout_out(mol)

        if self.fp_dim > 0:
            if fp is None:
                fp = x.new_zeros(mol.shape[0], 512)
            mol = torch.cat([mol, self.fp_proj(fp)], dim=-1)   # (B, 2H + fp_dim)

        return self.task_head(mol)


class DMPNN(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels,
                 num_classes, depth=4, task_dim=64, fp_dim=64):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.depth = depth

        self.W_i = nn.Linear(num_node_features + num_edge_features, hidden_channels)
        self.W_m = nn.Linear(hidden_channels, hidden_channels)
        self.W_a = nn.Linear(num_node_features + hidden_channels, hidden_channels)
        self.ln_step = nn.LayerNorm(hidden_channels)

        self.set2set = Set2Set(hidden_channels, processing_steps=4)
        self.dropout_mid = nn.Dropout(p=0.2)
        self.dropout_out = nn.Dropout(p=0.5)

        self.fp_dim = fp_dim
        if fp_dim > 0:
            self.fp_proj = nn.Sequential(nn.Linear(512, fp_dim), nn.ReLU())
        self.task_head = TaskHead(2 * hidden_channels + fp_dim, num_classes, task_dim)

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
            mol = self.dropout_out(self.set2set(atom_out, batch))
            if self.fp_dim > 0:
                if fp is None:
                    fp = x.new_zeros(mol.shape[0], 512)
                mol = torch.cat([mol, self.fp_proj(fp)], dim=-1)
            return self.task_head(mol)

        src, dst = edge_index[0], edge_index[1]
        num_edges = edge_index.size(1)
        rev_idx = torch.arange(num_edges, device=x.device) ^ 1

        h_init = self.W_i(torch.cat([x[src], edge_attr], dim=-1)).relu()
        h = h_init

        for i in range(self.depth):
            agg = torch.zeros(num_nodes, h.size(-1), device=h.device, dtype=h.dtype)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(h), h)
            msg = agg[src] - h[rev_idx]
            h = self.ln_step(F.relu(h_init + self.W_m(msg)))
            h = self.dropout_mid(h)

        atom_msg = torch.zeros(num_nodes, h.size(-1), device=h.device, dtype=h.dtype)
        atom_msg.scatter_add_(0, dst.unsqueeze(-1).expand_as(h), h)
        atom_out = self.W_a(torch.cat([x, atom_msg], dim=-1)).relu()

        mol = self.dropout_out(self.set2set(atom_out, batch))

        if self.fp_dim > 0:
            if fp is None:
                fp = x.new_zeros(mol.shape[0], 512)
            mol = torch.cat([mol, self.fp_proj(fp)], dim=-1)

        return self.task_head(mol)


class EnsembleGNN(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        return torch.stack([m(*args, **kwargs) for m in self.models]).mean(0)


def build_and_load_ensemble(config, device, model_dir='.'):
    import os
    from src.featurizer import smiles_to_graph
    from src.dataset import DATASET_CONFIGS

    dummy = smiles_to_graph('CCO')
    num_node_features = dummy.x.shape[1]
    edge_dim = dummy.edge_attr.shape[1]
    names = config['dataset'].get('names') or [config['dataset']['name']]
    # tasks=None means auto-detected (e.g. ToxCast) — load the saved dataset to get num_classes
    if any(DATASET_CONFIGS[n]['tasks'] is None for n in names):
        from src.dataset import load_dataset
        num_classes = len(load_dataset(config).primary_tasks)
    else:
        num_classes = sum(len(DATASET_CONFIGS[n]['tasks']) for n in names)
    hidden = config['model']['hidden_channels']
    depth = config['model'].get('depth', 4)
    task_dim = config['model'].get('task_dim', 64)
    fp_dim = config['model'].get('fp_dim', 64)
    model_type = config['model'].get('type', 'gnn')
    ensemble_size = config['model'].get('ensemble_size', 3)

    models = []
    for i in range(ensemble_size):
        path = os.path.join(model_dir, f'model_{i}.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found — run train.py first')
        if model_type == 'dmpnn':
            m = DMPNN(num_node_features, edge_dim, hidden, num_classes, depth=depth, task_dim=task_dim, fp_dim=fp_dim).to(device)
        else:
            m = GNN(num_node_features, hidden, num_classes, edge_dim=edge_dim, task_dim=task_dim, fp_dim=fp_dim).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        # torch.compile is intentionally skipped: compiled wrappers store the
        # original module via object.__setattr__, so nn.Module.modules() never
        # recurses into the real submodules. enable_mc_dropout cannot reach
        # nn.Dropout layers, all 20 MC passes are identical, and std = 0.
        models.append(m)

    return EnsembleGNN(models)
