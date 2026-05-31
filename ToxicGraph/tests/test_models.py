import torch
import torch.nn as nn
import pytest
from src.models import GNN, DMPNN
from src.featurizer import smiles_to_graph


def _make_batch(smiles='CCO'):
    data = smiles_to_graph(smiles).clone()
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    return data


def test_gnn_no_batch_norm():
    model = GNN(44, 64, 3, edge_dim=8, fp_dim=64)
    for name, mod in model.named_modules():
        assert not isinstance(mod, nn.BatchNorm1d), f"BatchNorm found at {name}"


def test_gnn_has_layer_norm():
    model = GNN(44, 64, 3, edge_dim=8, fp_dim=64)
    ln_count = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    assert ln_count >= 4, f"Expected >=4 LayerNorm modules, found {ln_count}"


def test_dmpnn_has_layer_norm():
    model = DMPNN(44, 8, 64, 3, fp_dim=64)
    assert any(isinstance(m, nn.LayerNorm) for m in model.modules())


def test_gnn_output_shape_with_fp():
    model = GNN(44, 64, 5, edge_dim=8, fp_dim=64)
    model.eval()
    data = _make_batch()
    out = model(data)
    assert out.shape == (1, 5), f"Expected (1, 5), got {out.shape}"


def test_dmpnn_output_shape_with_fp():
    model = DMPNN(44, 8, 64, 5, fp_dim=64)
    model.eval()
    data = _make_batch()
    out = model(data)
    assert out.shape == (1, 5), f"Expected (1, 5), got {out.shape}"


def test_gnn_graceful_without_fp():
    # If data.fp is absent, model should zero-fill and still produce output
    model = GNN(44, 64, 3, edge_dim=8, fp_dim=64)
    model.eval()
    data = _make_batch()
    del data.fp
    out = model(data)
    assert out.shape == (1, 3)


def test_dmpnn_graceful_without_fp():
    model = DMPNN(44, 8, 64, 3, fp_dim=64)
    model.eval()
    data = _make_batch()
    del data.fp
    out = model(data)
    assert out.shape == (1, 3)


def test_gnn_output_differs_with_fp_vs_without():
    # fp_dim > 0: output should change when fp is zeroed vs real
    model = GNN(44, 64, 3, edge_dim=8, fp_dim=64)
    model.eval()
    data_real = _make_batch('c1ccccc1')
    data_zero = _make_batch('c1ccccc1')
    data_zero.fp = torch.zeros_like(data_zero.fp)
    with torch.no_grad():
        out_real = model(data_real)
        out_zero = model(data_zero)
    assert not torch.allclose(out_real, out_zero), \
        "Real and zeroed fingerprints should produce different outputs"
