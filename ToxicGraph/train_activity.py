"""
train_activity.py — Train the RF activity model on BBBP and HIV datasets.

These activity endpoints are kept separate from the toxicity GNN because they
measure pharmacological activity (permeability, antiviral efficacy) rather than
safety, and their inductive bias can negatively transfer to toxicity learning.

Run after setting up the environment and before starting the web app:
    python train_activity.py

Output: checkpoints/activity/activity_rf.pkl
"""
import os
import pathlib
import yaml
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from src.dataset import MultiToxDataset
from src.fp_model import FPEnsemble
from src.activity_model import ACTIVITY_TASKS, ActivityModel
from src.splitter import multidataset_scaffold_split


OUTPUT_DIR = os.path.join('checkpoints', 'activity')


def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    ds_names = config['dataset'].get('activity_names', ['bbbp', 'hiv'])
    root = config['dataset']['root']

    print(f"Activity datasets: {ds_names}")
    dataset = MultiToxDataset(root=root, names=ds_names)
    print(f"  {len(dataset)} molecules, {len(dataset.all_tasks)} tasks: {dataset.all_tasks}")

    train_ds, val_ds, test_ds = multidataset_scaffold_split(dataset)
    print(f"  Split: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    def collect(ds):
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        fps, ys = [], []
        for batch in loader:
            fps.append(batch.fp.numpy())
            ys.append(batch.y.numpy())
        return np.vstack(fps), np.vstack(ys)

    X_train, Y_train = collect(train_ds)
    X_test,  Y_test  = collect(test_ds)
    print(f"  X_train: {X_train.shape},  X_test: {X_test.shape}")

    print("Training RF activity model (300 trees per task)...")
    rf = FPEnsemble(n_estimators=300)
    rf.fit(X_train, Y_train)
    print(f"  Fitted {len(rf.models)} / {Y_train.shape[1]} task models")

    print("\nTest-set ROC-AUC:")
    probs = rf.predict_proba(X_test, n_tasks=len(ACTIVITY_TASKS))
    for i, task_meta in enumerate(ACTIVITY_TASKS):
        mask = Y_test[:, i] > -0.5
        if mask.sum() < 10:
            print(f"  {task_meta['label']:<28} — insufficient data")
            continue
        yt = Y_test[mask, i].astype(int)
        yp = probs[mask, i]
        if len(np.unique(yt)) < 2:
            print(f"  {task_meta['label']:<28} — single class in test split")
            continue
        auc = roc_auc_score(yt, yp)
        print(f"  {task_meta['label']:<28}  AUC={auc:.4f}  (n={mask.sum()}, pos={yt.sum()})")

    model = ActivityModel(rf)
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'activity_rf.pkl')
    model.save(out_path)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
