import torch
import pytest
from train import focal_loss, compute_pos_weights


class _FakeDataset:
    """Minimal stub — enough for compute_pos_weights."""
    def __init__(self, y_tensor):
        # Mimics InMemoryDataset: .data.y is the full (N, T) tensor
        class _Data:
            pass
        self.data = _Data()
        self.data.y = y_tensor


def test_compute_pos_weights_ratio():
    # Task 0: 1 positive, 3 negatives → weight = 3/1 = 3.0
    # Task 1: 2 positives, 2 negatives → weight = 2/2 = 1.0
    y = torch.tensor([[1., 1.], [0., 0.], [0., 1.], [0., 0.]])
    ds = _FakeDataset(y)
    pw = compute_pos_weights(ds, num_tasks=2, device=torch.device('cpu'))
    assert abs(pw[0].item() - 3.0) < 1e-5
    assert abs(pw[1].item() - 1.0) < 1e-5


def test_compute_pos_weights_clamps_at_10():
    # 1 positive, 100 negatives → raw = 100, clamped to 10
    y = torch.zeros(101, 1)
    y[0, 0] = 1.0
    ds = _FakeDataset(y)
    pw = compute_pos_weights(ds, num_tasks=1, device=torch.device('cpu'))
    assert pw[0].item() == pytest.approx(10.0)


def test_compute_pos_weights_all_negative_stays_one():
    # No positives → weight stays at 1.0 (avoid division by zero)
    y = torch.zeros(10, 2)
    ds = _FakeDataset(y)
    pw = compute_pos_weights(ds, num_tasks=2, device=torch.device('cpu'))
    assert (pw == 1.0).all()


def test_compute_pos_weights_masked_labels_ignored():
    # -1.0 = missing label, should not be counted as negative
    # Task 0: 1 pos, 1 neg (the -1 entry is ignored) → weight = 1/1 = 1.0
    y = torch.tensor([[1.], [0.], [-1.]])
    ds = _FakeDataset(y)
    pw = compute_pos_weights(ds, num_tasks=1, device=torch.device('cpu'))
    assert abs(pw[0].item() - 1.0) < 1e-5


def test_focal_loss_pos_weight_upweights_positives():
    # Identical logits: with pos_weight=2 loss must be > without
    logits = torch.zeros(4)
    targets = torch.tensor([1., 1., 0., 0.])
    pw = torch.full((4,), 2.0)
    loss_pw = focal_loss(logits, targets, pos_weight=pw)
    loss_no_pw = focal_loss(logits, targets)
    assert loss_pw.item() > loss_no_pw.item()


def test_focal_loss_pos_weight_none_unchanged():
    # pos_weight=None must produce identical result to not passing it
    logits = torch.randn(8)
    targets = torch.randint(0, 2, (8,)).float()
    assert focal_loss(logits, targets, pos_weight=None).item() == \
           focal_loss(logits, targets).item()


def test_focal_loss_dataset_weights_and_pos_weight_combine():
    # Both weights applied: result differs from either alone
    logits = torch.zeros(4)
    targets = torch.tensor([1., 0., 1., 0.])
    pw = torch.full((4,), 3.0)
    dw = torch.full((4,), 2.0)
    loss_both = focal_loss(logits, targets, weights=dw, pos_weight=pw)
    loss_pw_only = focal_loss(logits, targets, pos_weight=pw)
    loss_dw_only = focal_loss(logits, targets, weights=dw)
    assert loss_both.item() != loss_pw_only.item()
    assert loss_both.item() != loss_dw_only.item()
