import torch
from unittest.mock import MagicMock, patch
from inference import predict


def _mock_ensemble():
    ens = MagicMock()
    ens.return_value = torch.zeros(1, 3)
    return ens


def test_predict_accepts_preloaded_ensemble():
    """predict() should use the passed ensemble instead of loading from disk."""
    ens = _mock_ensemble()
    task_names = ['T1', 'T2', 'T3']
    with patch('src.calibration.mc_sample') as mock_mc:
        mock_mc.return_value = (torch.zeros(1, 3), torch.zeros(1, 3))
        means, stds, returned_tasks = predict(
            ['CCO'], ensemble=ens, task_names=task_names, temperature=1.0
        )
    assert returned_tasks == task_names
    assert means[0] is not None
    assert len(means[0]) == 3


def test_predict_returns_none_for_invalid_smiles():
    ens = _mock_ensemble()
    means, stds, _ = predict(
        ['not_a_smiles!!!'], ensemble=ens, task_names=['T1'], temperature=1.0
    )
    assert means[0] is None


def test_predict_does_not_load_from_disk_when_ensemble_provided(tmp_path):
    """No FileNotFoundError should be raised even though no weights exist."""
    ens = _mock_ensemble()
    with patch('src.calibration.mc_sample') as mock_mc:
        mock_mc.return_value = (torch.zeros(1, 1), torch.zeros(1, 1))
        means, stds, _ = predict(
            ['CCO'], ensemble=ens, task_names=['T1'], temperature=1.0,
            model_dir=str(tmp_path),
        )
    assert means[0] is not None
