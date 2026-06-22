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
        Returns {task_key: {prob, std}} for each activity task, or None if invalid.
        std is the standard deviation of tree votes (RF uncertainty).
        """
        fp = smiles_to_fp(smiles)
        if fp is None:
            return None
        mean, std = self.rf.predict_proba_with_std(fp[None], n_tasks=len(ACTIVITY_TASKS))
        return {
            t['key']: {'prob': float(mean[0, i]), 'std': float(std[0, i])}
            for i, t in enumerate(ACTIVITY_TASKS)
        }

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'ActivityModel':
        with open(path, 'rb') as f:
            return pickle.load(f)
