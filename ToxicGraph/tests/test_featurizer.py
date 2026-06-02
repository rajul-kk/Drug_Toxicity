import torch
import pytest
from src.featurizer import smiles_to_graph


def test_fp_attribute_present():
    data = smiles_to_graph('CCO')
    assert hasattr(data, 'fp'), "data.fp should exist after featurizer update"


def test_fp_shape():
    data = smiles_to_graph('c1ccccc1')
    assert data.fp.shape == (1, 679), f"Expected (1, 679), got {data.fp.shape}"


def test_fp_dtype():
    data = smiles_to_graph('CC(=O)Nc1ccc(O)cc1')
    assert data.fp.dtype == torch.float


def test_fp_binary_values():
    data = smiles_to_graph('CCO')
    assert ((data.fp == 0) | (data.fp == 1)).all(), "ECFP4 bits must be 0 or 1"


def test_fp_nonzero_for_real_molecule():
    data = smiles_to_graph('c1ccccc1')
    assert data.fp.sum() > 0, "Benzene fingerprint should have set bits"


def test_fp_differs_across_molecules():
    d1 = smiles_to_graph('CCO')
    d2 = smiles_to_graph('c1ccccc1')
    assert not torch.equal(d1.fp, d2.fp), "Different molecules must produce different fingerprints"


def test_fp_deterministic():
    d1 = smiles_to_graph('CC(=O)O')
    smiles_to_graph.cache_clear()
    d2 = smiles_to_graph('CC(=O)O')
    assert torch.equal(d1.fp, d2.fp)


def test_virtual_node_present():
    data = smiles_to_graph('CCO')   # ethanol: 3 heavy atoms
    assert data.x.shape[0] == 4, \
        f"Expected 4 nodes (3 atoms + 1 virtual), got {data.x.shape[0]}"


def test_virtual_node_edges():
    data = smiles_to_graph('CCO')   # 3 atoms, 2 bonds -> 4 directed real edges
    N_real = 3
    N_virtual_edges = 2 * N_real   # (vn->i) and (i->vn) for each atom
    assert data.edge_index.shape[1] % 2 == 0, "Edge count must be even (all pairs)"
    vn_idx = N_real  # virtual node is at index 3
    vn_edges = (data.edge_index == vn_idx).any(dim=0)
    assert vn_edges.sum() == N_virtual_edges, \
        f"Expected {N_virtual_edges} virtual edges, found {vn_edges.sum().item()}"
