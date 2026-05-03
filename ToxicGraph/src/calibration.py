
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def enable_mc_dropout(model):
    """Eval mode for BatchNorm (uses running stats) but Dropout stays active."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_sample(model, data, n_samples=30):
    """
    Return (mean_logits, std_logits) over n_samples stochastic forward passes.
    Call on a single graph Data object (batch dim already set).
    """
    enable_mc_dropout(model)
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            samples.append(model(data))
    stacked = torch.stack(samples)          # (n_samples, num_tasks)
    return stacked.mean(0), stacked.std(0)


def fit_temperature(model, val_loader, device):
    """
    Post-hoc calibration: find scalar T that minimises NLL on the val set.
    Returns T as a float. T > 1 softens probabilities toward 0.5; T < 1 sharpens them.
    """
    logits_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            logits_list.append(model(data).cpu())
            labels_list.append(data.y.cpu())

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    mask = labels > -0.5

    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits[mask] / temperature, labels[mask])
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.item())


def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error over all labeled (prob, label) pairs.
    Lower is better; 0 means predicted probabilities match empirical frequencies exactly.
    """
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1],
                      np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def compute_brier(probs, labels):
    """Brier score: MSE between predicted probabilities and true labels. Lower is better."""
    return float(np.mean((probs - labels) ** 2))
