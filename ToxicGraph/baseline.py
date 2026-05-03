
"""
XGBoost baseline using Morgan fingerprints (ECFP4, 2048 bits).
Uses the same scaffold split as the GNN for a fair comparison.
"""
import yaml
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

from src.dataset import ToxicDataset
from src.splitter import scaffold_split

TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
]


def smiles_to_fp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits), dtype=np.float32)


def run_baseline():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    dataset = ToxicDataset(root=config['dataset']['root'], name=config['dataset']['name'])
    train_dataset, _, test_dataset = scaffold_split(dataset)
    task_names = TOX21_TASKS if config['dataset']['name'] == 'tox21' else [str(i) for i in range(12)]
    num_tasks = len(task_names)

    print("Computing Morgan fingerprints (ECFP4, 2048 bits)...")
    X_train = np.array([smiles_to_fp(s) for s in train_dataset.smiles_list])
    X_test = np.array([smiles_to_fp(s) for s in test_dataset.smiles_list])

    y_train = np.array([train_dataset[i].y.numpy().flatten() for i in range(len(train_dataset))])
    y_test = np.array([test_dataset[i].y.numpy().flatten() for i in range(len(test_dataset))])

    print(f"\n{'Task':<20} {'AUC':>6}  {'AUPRC':>6}")
    print('-' * 36)

    aucs, auprcs = [], []
    for i in range(num_tasks):
        mask_tr = y_train[:, i] > -0.5
        mask_te = y_test[:, i] > -0.5

        if mask_tr.sum() < 10 or mask_te.sum() < 5:
            continue

        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0,
        )
        clf.fit(X_train[mask_tr], y_train[mask_tr, i])
        probs = clf.predict_proba(X_test[mask_te])[:, 1]

        auc = roc_auc_score(y_test[mask_te, i], probs)
        auprc = average_precision_score(y_test[mask_te, i], probs)
        aucs.append(auc)
        auprcs.append(auprc)
        print(f"{task_names[i]:<20} {auc:.4f}  {auprc:.4f}")

    print('-' * 36)
    print(f"{'Mean':<20} {np.mean(aucs):.4f}  {np.mean(auprcs):.4f}\n")


if __name__ == '__main__':
    run_baseline()
