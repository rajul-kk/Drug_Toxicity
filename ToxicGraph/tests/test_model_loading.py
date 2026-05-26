import pytest
from src.models import build_and_load_ensemble


def test_build_and_load_ensemble_uses_model_dir(tmp_path):
    """model_dir param should make the loader look in the given directory."""
    config = {
        'dataset': {'names': ['tox21']},
        'model': {'type': 'gnn', 'hidden_channels': 32, 'ensemble_size': 1,
                  'depth': 2, 'task_dim': 8},
    }
    with pytest.raises(FileNotFoundError) as exc:
        build_and_load_ensemble(config, 'cpu', model_dir=str(tmp_path))
    assert str(tmp_path) in str(exc.value)
