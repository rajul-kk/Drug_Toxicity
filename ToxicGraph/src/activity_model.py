"""
Activity model — RF fingerprint predictions for BBBP and HIV.

These are pharmacological activity endpoints (membrane permeability, antiviral
efficacy), distinct from safety/toxicity endpoints and trained separately as a
pure fingerprint RF model via train_activity.py.
"""
import pickle
import numpy as np
from src.fp_model import FPEnsemble, smiles_to_fp


# Ordered to match the task columns in the MultiToxDataset built from activity_names.
ACTIVITY_TASKS = [
    {
        'key':   'bbbp',
        'label': 'BBB Permeability',
        'desc':  'Probability the molecule penetrates the blood-brain barrier '
                 '(Martins et al. 2012, ~2k compounds)',
        'color': '#0891b2',
    },
    {
        'key':   'hiv',
        'label': 'HIV Antiviral Activity',
        'desc':  'Probability of inhibiting HIV replication '
                 '(Wu et al. 2018, ~41k compounds)',
        'color': '#7c3aed',
    },
]


class ActivityModel:
    """
    Wraps a per-task FPEnsemble trained on activity datasets (BBBP, HIV).
    Task order must match the order of `activity_names` in config.yaml.
    """

    def __init__(self, rf_ensemble: FPEnsemble):
        self.rf = rf_ensemble

    def predict(self, smiles: str) -> dict | None:
        """
        Returns {task_key: probability} for each activity task, or None if
        the SMILES is invalid.
        """
        fp = smiles_to_fp(smiles)
        if fp is None:
            return None
        probs = self.rf.predict_proba(fp[None], n_tasks=len(ACTIVITY_TASKS))[0]
        return {t['key']: float(probs[i]) for i, t in enumerate(ACTIVITY_TASKS)}

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'ActivityModel':
        with open(path, 'rb') as f:
            return pickle.load(f)
