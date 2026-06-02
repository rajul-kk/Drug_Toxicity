"""
eval_datasets.py — Per-dataset evaluation of GNN and DMPNN ensembles.

Metrics reported per task and per dataset:
  ROC-AUC      — primary MoleculeNet benchmark metric
  PR-AUC       — better for severely imbalanced tasks (HIV, Tox21 NR tasks)
  Brier score  — calibration-sensitive probability accuracy (lower = better)
  ECE          — Expected Calibration Error (lower = better)
  Bal-Acc      — balanced accuracy at 0.5 threshold, class-imbalance aware
  Pos%         — fraction of positives (context for imbalance)

Usage:
  python eval_datasets.py                  # both models
  python eval_datasets.py --model gnn      # GNN only
  python eval_datasets.py --model dmpnn    # DMPNN only
  python eval_datasets.py --no-cache       # force re-inference even if cache exists
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

# -- helpers --------------------------------------------------------------------

def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


# -- per-task metrics for a single (probs, labels) slice -----------------------

def task_metrics(probs: np.ndarray, labels: np.ndarray) -> dict | None:
    """
    Compute all metrics for one task column.
    Returns None if the task has too few usable samples or only one class.
    `labels` may contain -1 (missing) — these are excluded.
    """
    mask = labels > -0.5
    if mask.sum() < 10:
        return None

    yt = labels[mask].astype(float)
    yp = probs[mask].astype(float)

    n_pos = int((yt > 0.5).sum())
    n_neg = int((yt < 0.5).sum())

    if n_pos == 0 or n_neg == 0:
        return None  # single-class split — AUC undefined

    try:
        auc   = roc_auc_score(yt, yp)
        auprc = average_precision_score(yt, yp)
        brier = compute_brier(yp, yt)
        ece   = compute_ece(yp, yt)
        pred_bin = (yp >= 0.5).astype(int)
        bal_acc = balanced_accuracy_score(yt.astype(int), pred_bin)
        pos_pct = 100.0 * n_pos / (n_pos + n_neg)
        n_total = n_pos + n_neg
        return dict(
            auc=auc, auprc=auprc, brier=brier, ece=ece,
            bal_acc=bal_acc, pos_pct=pos_pct, n=n_total,
        )
    except Exception as e:
        return None


# -- load or compute test-set predictions --------------------------------------

def load_predictions(arch: str, config: dict, device: torch.device,
                     force_recompute: bool = False):
    """
    Returns (probs, labels, smiles_list, source_list, task_names, task_ranges).
    Uses the precomputed checkpoint cache when available (fast); otherwise runs
    batch inference and reports the reason.
    """
    cp_dir = os.path.join('checkpoints', arch)
    cache_files = {
        'probs':   os.path.join(cp_dir, 'test_probs.npy'),
        'labels':  os.path.join(cp_dir, 'test_labels.npy'),
        'smiles':  os.path.join(cp_dir, 'test_smiles.json'),
        'sources': os.path.join(cp_dir, 'test_sources.json'),
    }
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if cache_exists and not force_recompute:
        probs       = np.load(cache_files['probs'])
        labels      = np.load(cache_files['labels'])
        smiles_list = json.load(open(cache_files['smiles']))
        source_list = json.load(open(cache_files['sources']))
        print(f"  [{arch}] using precomputed cache — {len(smiles_list)} molecules")
    else:
        reason = "forced re-inference" if force_recompute else "no cache found"
        print(f"  [{arch}] {reason} — running batch inference (may take a few minutes)…")
        from evaluate import collect_predictions, load_test_dataset
        from src.models import build_and_load_ensemble

        model_dir = cp_dir
        temperature_path = os.path.join(cp_dir, 'temperature.pt')
        temperature = (
            float(torch.load(temperature_path, map_location='cpu'))
            if os.path.exists(temperature_path) else 1.0
        )

        ensemble = build_and_load_ensemble(config, device, model_dir=model_dir)
        test_dataset, _ = load_test_dataset(config)
        probs, labels = collect_predictions(ensemble, test_dataset, device, temperature)
        smiles_list = test_dataset.smiles_list
        source_list = (
            test_dataset.source_list
            if hasattr(test_dataset, 'source_list')
            else [config['dataset'].get('names', ['unknown'])[0]] * len(smiles_list)
        )
        print(f"  [{arch}] inference complete — {len(smiles_list)} molecules")

    # Build task_names and task_ranges from config (not stored in cache)
    from src.dataset import DATASET_CONFIGS
    dataset_names = config['dataset']['names']
    task_names, task_ranges = [], {}
    for ds in dataset_names:
        start = len(task_names)
        tasks = DATASET_CONFIGS[ds]['tasks']
        task_names.extend(tasks)
        task_ranges[ds] = (start, start + len(tasks))

    return probs, labels, smiles_list, source_list, task_names, task_ranges


# -- pretty-print helpers -------------------------------------------------------

COL_W = {
    'task':    24,
    'auc':      7,
    'auprc':    7,
    'brier':    7,
    'ece':      7,
    'bal_acc':  7,
    'pos_pct':  6,
    'n':        6,
}

HEADER = (
    f"{'Task':<{COL_W['task']}} "
    f"{'AUC':>{COL_W['auc']}} "
    f"{'PR-AUC':>{COL_W['auprc']}} "
    f"{'Brier':>{COL_W['brier']}} "
    f"{'ECE':>{COL_W['ece']}} "
    f"{'BalAcc':>{COL_W['bal_acc']}} "
    f"{'Pos%':>{COL_W['pos_pct']}} "
    f"{'N':>{COL_W['n']}}"
)
DIVIDER = '-' * len(HEADER)


def fmt_row(name: str, m: dict) -> str:
    return (
        f"{name:<{COL_W['task']}} "
        f"{m['auc']:>{COL_W['auc']}.4f} "
        f"{m['auprc']:>{COL_W['auprc']}.4f} "
        f"{m['brier']:>{COL_W['brier']}.4f} "
        f"{m['ece']:>{COL_W['ece']}.4f} "
        f"{m['bal_acc']:>{COL_W['bal_acc']}.4f} "
        f"{m['pos_pct']:>{COL_W['pos_pct']}.1f} "
        f"{m['n']:>{COL_W['n']}}"
    )


def mean_metrics(rows: list[dict]) -> dict:
    keys = ['auc', 'auprc', 'brier', 'ece', 'bal_acc', 'pos_pct']
    totals = {k: 0.0 for k in keys}
    for m in rows:
        for k in keys:
            totals[k] += m[k]
    n = len(rows)
    avg = {k: totals[k] / n for k in keys}
    avg['n'] = sum(m['n'] for m in rows)
    return avg


# -- main evaluation ------------------------------------------------------------

def evaluate_arch(arch: str, config: dict, device: torch.device,
                  force_recompute: bool = False):
    """Evaluate one model architecture across all datasets."""

    try:
        probs, labels, smiles_list, source_list, task_names, task_ranges = \
            load_predictions(arch, config, device, force_recompute)
    except FileNotFoundError as e:
        print(f"\n  [{arch}] SKIPPED — {e}")
        return None

    source_arr = np.array(source_list)
    results = {}  # dataset → {task: metrics}

    for ds, (t_start, t_end) in task_ranges.items():
        mol_mask = source_arr == ds
        n_mol = int(mol_mask.sum())

        ds_tasks = task_names[t_start:t_end]
        ds_probs  = probs[mol_mask, t_start:t_end]   # (N_ds, T_ds)
        ds_labels = labels[mol_mask, t_start:t_end]  # (N_ds, T_ds)

        task_results = {}
        for j, task in enumerate(ds_tasks):
            m = task_metrics(ds_probs[:, j], ds_labels[:, j])
            if m is not None:
                task_results[task] = m

        results[ds] = {'tasks': task_results, 'n_mol': n_mol}

    return results


def print_dataset_table(arch: str, ds: str, data: dict):
    task_results = data['tasks']
    n_mol        = data['n_mol']

    if not task_results:
        print(f"\n  [{arch.upper()} x {ds}] skipped — no molecules from this dataset in the "
              f"test cache (cache may predate adding {ds}; run --no-cache or retrain)")
        return

    title = f"  {arch.upper()}  x {ds}  ({n_mol} test molecules, "            f"{len(task_results)} evaluable tasks)"
    print(f"\n{'=' * len(HEADER)}")
    print(title)
    print(HEADER)
    print(DIVIDER)

    for task, m in sorted(task_results.items(), key=lambda x: -x[1]['auc']):
        print(fmt_row(task, m))

    print(DIVIDER)
    if len(task_results) > 1:
        mean = mean_metrics(list(task_results.values()))
        print(fmt_row('MEAN', mean))


def print_comparison_table(gnn_results: dict | None, dmpnn_results: dict | None,
                            task_ranges: dict, task_names: list):
    """Side-by-side mean ROC-AUC comparison across datasets."""
    archs = [(a, r) for a, r in [('gnn', gnn_results), ('dmpnn', dmpnn_results)]
             if r is not None]
    if not archs:
        return

    print(f"\n\n{'=' * 62}")
    print("  SUMMARY — mean ROC-AUC and PR-AUC per dataset")
    print(f"{'=' * 62}")

    arch_names = [a for a, _ in archs]
    header = f"  {'Dataset':<12}"
    for a in arch_names:
        header += f"  {a.upper()+' AUC':>10}  {a.upper()+' PR-AUC':>12}"
    print(header)
    print('  ' + '-' * (len(header) - 2))

    all_aucs = {a: [] for a, _ in archs}
    all_auprcs = {a: [] for a, _ in archs}

    for ds in task_ranges:
        row = f"  {ds:<12}"
        for a, res in archs:
            dr = res.get(ds, {}).get('tasks', {})
            if dr:
                vals = list(dr.values())
                m_auc   = np.mean([v['auc']   for v in vals])
                m_auprc = np.mean([v['auprc'] for v in vals])
                all_aucs[a].append(m_auc)
                all_auprcs[a].append(m_auprc)
                row += f"  {m_auc:>10.4f}  {m_auprc:>12.4f}"
            else:
                row += f"  {'n/a':>10}  {'n/a':>12}"
        print(row)

    print('  ' + '-' * (len(header) - 2))
    row = f"  {'OVERALL':<12}"
    for a, _ in archs:
        if all_aucs[a]:
            row += f"  {np.mean(all_aucs[a]):>10.4f}  {np.mean(all_auprcs[a]):>12.4f}"
        else:
            row += f"  {'n/a':>10}  {'n/a':>12}"
    print(row)


def main():
    parser = argparse.ArgumentParser(description='Per-dataset model evaluation')
    parser.add_argument('--model', choices=['gnn', 'dmpnn', 'both'], default='both',
                        help='Which model architecture to evaluate (default: both)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-inference even if precomputed cache exists')
    args = parser.parse_args()

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Datasets: {config['dataset']['names']}")

    # Rebuild task_ranges here for the summary table
    from src.dataset import DATASET_CONFIGS
    task_names_list, task_ranges = [], {}
    for ds in config['dataset']['names']:
        start = len(task_names_list)
        tasks = DATASET_CONFIGS[ds]['tasks']
        task_names_list.extend(tasks)
        task_ranges[ds] = (start, start + len(tasks))

    archs = (['gnn', 'dmpnn'] if args.model == 'both'
             else [args.model])

    all_results = {}
    for arch in archs:
        print(f"\nEvaluating {arch.upper()}…")
        all_results[arch] = evaluate_arch(arch, config, device, args.no_cache)

    # Per-dataset per-task tables
    for arch in archs:
        res = all_results.get(arch)
        if res is None:
            continue
        for ds in config['dataset']['names']:
            if ds in res:
                print_dataset_table(arch, ds, res[ds])

    # Summary comparison
    print_comparison_table(
        all_results.get('gnn'),
        all_results.get('dmpnn'),
        task_ranges,
        task_names_list,
    )
    print()


if __name__ == '__main__':
    main()
