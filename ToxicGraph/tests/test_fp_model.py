import numpy as np
import pytest
from src.fp_model import FPEnsemble, smiles_to_fp


def test_smiles_to_fp_shape():
    fp = smiles_to_fp('CCO')
    assert fp is not None
    assert fp.shape == (679,)


def test_smiles_to_fp_binary():
    fp = smiles_to_fp('c1ccccc1')
    assert ((fp == 0) | (fp == 1)).all()


def test_smiles_to_fp_invalid_returns_none():
    assert smiles_to_fp('not_a_smiles_!@#') is None


def test_fp_ensemble_fit_predict():
    X = np.random.randint(0, 2, (50, 679)).astype(np.float32)
    Y = np.random.randint(0, 2, (50, 3)).astype(np.float32)
    model = FPEnsemble(n_estimators=10)
    model.fit(X, Y)
    probs = model.predict_proba(X, n_tasks=3)
    assert probs.shape == (50, 3)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_fp_ensemble_handles_missing_labels():
    X = np.random.randint(0, 2, (40, 679)).astype(np.float32)
    Y = np.full((40, 2), -1.0)
    Y[:20, 0] = np.random.randint(0, 2, 20)
    model = FPEnsemble(n_estimators=10)
    model.fit(X, Y)
    assert 0 in model.models
    assert 1 not in model.models


def test_fp_ensemble_save_load(tmp_path):
    X = np.random.randint(0, 2, (30, 679)).astype(np.float32)
    Y = np.random.randint(0, 2, (30, 2)).astype(np.float32)
    model = FPEnsemble(n_estimators=5)
    model.fit(X, Y)
    path = tmp_path / 'fp_model.pkl'
    model.save(str(path))
    loaded = FPEnsemble.load(str(path))
    p1 = model.predict_proba(X, 2)
    p2 = loaded.predict_proba(X, 2)
    assert np.allclose(p1, p2)
