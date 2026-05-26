import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Patch heavy startup so tests run without real model weights."""
    mock_cache = {
        'smiles': ['CCO', 'c1ccccc1'],
        'sdf': ['sdf1', 'sdf2'],
        'probs': np.array([[0.9, 0.1, 0.5], [0.3, 0.7, 0.2]]),
        'labels': np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]),
        'dataset': ['tox21', 'tox21'],
        'max_conf': [0.9, 0.7],
        'auc_per_mol': [0.85, 0.72],
        'auprc_per_mol': [0.80, 0.68],
        'temperature': 1.0,
    }
    mock_ens = MagicMock()
    import app as app_module
    with patch.object(app_module, '_load_available_ensembles',
                      return_value=({'gnn': mock_ens}, {'gnn': mock_cache})):
        with TestClient(app_module.app) as c:
            yield c


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_info_returns_task_names(client):
    r = client.get('/api/info')
    assert r.status_code == 200
    data = r.json()
    assert 'task_names' in data
    assert 'task_groups' in data
    assert 'available_models' in data
    assert isinstance(data['task_names'], list)
    assert len(data['task_names']) > 0


def test_info_available_models(client):
    r = client.get('/api/info')
    data = r.json()
    assert 'gnn' in data['available_models']


def test_predict_returns_means_and_stds(client):
    with patch('app._predict') as mock_pred, patch('app.smiles_to_sdf', return_value='SDF'):
        import numpy as np
        mock_pred.return_value = ([np.array([0.5] * 41)], [np.array([0.05] * 41)], None)
        r = client.post('/api/predict', json={'smiles': 'CCO', 'n_mc': 5})
    assert r.status_code == 200
    data = r.json()
    assert 'means' in data and 'stds' in data
    assert isinstance(data['means'], list)
    assert data['model_used'] == 'gnn'


def test_predict_invalid_smiles_returns_422(client):
    with patch('app._predict') as mock_pred, patch('app.smiles_to_sdf', return_value=None):
        mock_pred.return_value = ([None], [None], None)
        r = client.post('/api/predict', json={'smiles': 'not_valid!!', 'n_mc': 5})
    assert r.status_code == 422


def test_testset_list_returns_rows(client):
    r = client.get('/api/testset')
    assert r.status_code == 200
    data = r.json()
    assert 'rows' in data and 'total' in data
    assert isinstance(data['rows'], list)


def test_testset_single_returns_probs_and_labels(client):
    r = client.get('/api/testset/0')
    assert r.status_code == 200
    data = r.json()
    assert 'probs' in data and 'labels' in data and 'smiles' in data


def test_testset_filter_by_dataset(client):
    r = client.get('/api/testset?filter=tox21')
    assert r.status_code == 200


def test_testset_out_of_range_returns_404(client):
    r = client.get('/api/testset/99999')
    assert r.status_code == 404


def test_testset_model_param_accepted(client):
    r = client.get('/api/testset?model=gnn')
    assert r.status_code == 200
    assert r.json()['model_used'] == 'gnn'


def test_predict_model_param_accepted(client):
    with patch('app._predict') as mock_pred, patch('app.smiles_to_sdf', return_value='SDF'):
        import numpy as np
        mock_pred.return_value = ([np.array([0.5] * 41)], [np.array([0.05] * 41)], None)
        r = client.post('/api/predict', json={'smiles': 'CCO', 'model': 'gnn'})
    assert r.status_code == 200
    assert r.json()['model_used'] == 'gnn'
