# Use after a RANDOM split only — scaffold split test sets are too sparse for per-scaffold AUC.
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score


def print_scaffold_analysis(model, test_dataset, num_tasks, device, task_names=None, min_size=2):
    task_names = task_names or [str(i) for i in range(num_tasks)]

    scaffolds = {}
    for i, smi in enumerate(test_dataset.smiles_list):
        mol = Chem.MolFromSmiles(smi)
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
        scaffolds.setdefault(sc, []).append(i)

    results = []
    model.eval()
    for scaffold, indices in scaffolds.items():
        if len(indices) < min_size:
            continue

        subset = [test_dataset[i] for i in indices]
        loader = DataLoader(subset, batch_size=len(subset))

        ys, preds = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                ys.append(data.y.cpu())
                preds.append(model(data).cpu())

        y_true = torch.cat(ys).numpy()
        y_pred = torch.cat(preds).numpy()

        scores = []
        for i in range(num_tasks):
            mask = y_true[:, i] > -0.5
            if mask.sum() > 0:
                try:
                    scores.append(roc_auc_score(y_true[mask, i], y_pred[mask, i]))
                except ValueError:
                    pass

        if scores:
            results.append((scaffold, len(indices), float(np.mean(scores))))

    if not results:
        print("\nNo scaffolds with enough molecules for analysis.")
        return

    results.sort(key=lambda x: x[2])

    print(f"\n=== Scaffold Analysis ({len(results)} scaffolds with ≥{min_size} molecules) ===")
    print(f"{'Scaffold':<46} {'N':>4}  {'AUC':>6}")
    print('-' * 60)
    n_show = min(5, len(results))
    print(f"  Worst {n_show}")
    for sc, n, auc in results[:n_show]:
        print(f"  {sc[:44]:<44} {n:>4}  {auc:.4f}")
    if len(results) > n_show:
        print(f"  Best {n_show}")
        for sc, n, auc in results[-n_show:]:
            print(f"  {sc[:44]:<44} {n:>4}  {auc:.4f}")
