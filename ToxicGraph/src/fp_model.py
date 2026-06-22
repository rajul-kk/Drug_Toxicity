import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestClassifier


def smiles_to_fp(smiles: str) -> np.ndarray | None:
    """Returns (679,) float32 array: Morgan(512) + MACCS(167), or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    morgan_obj = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    maccs_obj  = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(679, dtype=np.float32)
    ConvertToNumpyArray(morgan_obj, arr[:512])
    ConvertToNumpyArray(maccs_obj,  arr[512:])
    return arr


class FPEnsemble:
    """Per-task RandomForest on 679-bit fingerprints. Handles missing labels (-1)."""

    def __init__(self, n_estimators: int = 300, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.models: dict = {}

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """X: (N, 679), Y: (N, T) with -1 for missing labels."""
        n_tasks = Y.shape[1]
        for t in range(n_tasks):
            mask = Y[:, t] > -0.5
            if mask.sum() < 10:
                continue
            yt = Y[mask, t].astype(int)
            if len(np.unique(yt)) < 2:
                continue
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                class_weight='balanced',
                n_jobs=self.n_jobs,
                random_state=42,
            )
            rf.fit(X[mask], yt)
            self.models[t] = rf

    def predict_proba(self, X: np.ndarray, n_tasks: int) -> np.ndarray:
        """Returns (N, n_tasks) probability array. Unfitted tasks default to 0.5."""
        probs = np.full((X.shape[0], n_tasks), 0.5, dtype=np.float32)
        for t, rf in self.models.items():
            if t < n_tasks:
                probs[:, t] = rf.predict_proba(X)[:, 1]
        return probs

    def predict_proba_with_std(self, X: np.ndarray, n_tasks: int):
        """Returns (mean, std) each (N, n_tasks). std = tree-vote disagreement."""
        mean = np.full((X.shape[0], n_tasks), 0.5, dtype=np.float32)
        std  = np.zeros((X.shape[0], n_tasks), dtype=np.float32)
        for t, rf in self.models.items():
            if t < n_tasks:
                tree_preds = np.array([e.predict_proba(X)[:, 1] for e in rf.estimators_], dtype=np.float32)
                mean[:, t] = tree_preds.mean(axis=0)
                std[:, t]  = tree_preds.std(axis=0)
        return mean, std

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'FPEnsemble':
        with open(path, 'rb') as f:
            return pickle.load(f)
