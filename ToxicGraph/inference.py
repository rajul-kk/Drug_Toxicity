
import os
import yaml
import torch
from src.dataset import TOX21_TASKS
from src.models import build_and_load_ensemble
from src.featurizer import smiles_to_graph
from src.calibration import mc_sample


def predict(smiles_list, n_mc=30):
    """
    Returns (means, stds, task_names).
    Uses MC Dropout (n_mc forward passes) for per-prediction uncertainty.
    Applies temperature calibration if temperature.pt is present.
    """
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_and_load_ensemble(config, device)
    task_names = TOX21_TASKS if config['dataset']['name'] == 'tox21' else None

    temperature = 1.0
    if os.path.exists('temperature.pt'):
        temperature = float(torch.load('temperature.pt', map_location='cpu'))

    means, stds = [], []
    for smiles in smiles_list:
        data = smiles_to_graph(smiles)
        if data is None:
            means.append(None)
            stds.append(None)
            continue
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        data = data.to(device)
        mean_logits, std_logits = mc_sample(model, data, n_samples=n_mc)
        mean_probs = torch.sigmoid(mean_logits / temperature).cpu().numpy().flatten()
        std_probs = std_logits.cpu().numpy().flatten()
        means.append(mean_probs)
        stds.append(std_probs)

    return means, stds, task_names


if __name__ == '__main__':
    test_smiles = [
        'CCO',                                          # Ethanol
        'C1=CC=CC=C1',                                  # Benzene
        'CC(=O)OC1=CC=CC=C1C(=O)O',                    # Aspirin
        'CC(=O)Nc1ccc(O)cc1',                           # Paracetamol
        'Cn1cnc2c1c(=O)n(C)c(=O)n2C',                  # Caffeine
        'Clc1ccc(cc1)C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl',      # DDT
    ]

    means, stds, task_names = predict(test_smiles)
    col_w = 18
    header = f"{'SMILES':<45}" + "".join(f"{t:>{col_w}}" for t in task_names)
    print(header)
    print('-' * len(header))
    for smi, m, s in zip(test_smiles, means, stds):
        if m is None:
            print(f"{smi:<45}  (invalid SMILES)")
        else:
            # format each task as "mean±std"
            cols = "".join(f"{f'{v:.3f}±{u:.3f}':>{col_w}}" for v, u in zip(m, s))
            print(f"{smi:<45}{cols}")
