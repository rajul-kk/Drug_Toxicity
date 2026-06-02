
import os
import yaml
import torch
from src.models import build_and_load_ensemble
from src.featurizer import smiles_to_graph
from src.calibration import mc_sample


def predict(smiles_list, n_mc=20, ensemble=None, task_names=None, temperature=None,
            config=None, device=None, model_dir='.', fp_model=None):
    """
    Returns (means, stds, task_names).
    Pass pre-loaded ensemble/task_names/temperature for the web app path.
    Falls back to loading from disk when called from CLI (backwards compat).
    """
    if ensemble is None:
        if config is None:
            with open('config.yaml') as f:
                config = yaml.safe_load(f)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ensemble = build_and_load_ensemble(config, device, model_dir=model_dir)

    if temperature is None:
        temp_path = os.path.join(model_dir, 'temperature.pt')
        temperature = float(torch.load(temp_path, map_location='cpu')) \
                      if os.path.exists(temp_path) else 1.0

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    means, stds = [], []
    for smiles in smiles_list:
        data = smiles_to_graph(smiles)
        if data is None:
            means.append(None)
            stds.append(None)
            continue
        data = data.clone()  # don't mutate the lru_cache'd object
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        data = data.to(device)
        mean_logits, std_logits = mc_sample(ensemble, data, n_samples=n_mc)
        mean_probs = torch.sigmoid(mean_logits / temperature).cpu().numpy().flatten()
        if fp_model is not None:
            from src.fp_model import smiles_to_fp as _s2fp
            _fp = _s2fp(smiles)
            if _fp is not None:
                _rf = fp_model.predict_proba(_fp[None], len(mean_probs))[0]
                mean_probs = 0.7 * mean_probs + 0.3 * _rf
        std_probs = std_logits.cpu().numpy().flatten()
        means.append(mean_probs)
        stds.append(std_probs)

    return means, stds, task_names


if __name__ == '__main__':
    test_smiles = [
        'CCO',
        'C1=CC=CC=C1',
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CC(=O)Nc1ccc(O)cc1',
        'Cn1cnc2c1c(=O)n(C)c(=O)n2C',
        'Clc1ccc(cc1)C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl',
    ]
    means, stds, task_names = predict(test_smiles)
    if task_names:
        col_w = 18
        header = f"{'SMILES':<45}" + "".join(f"{t:>{col_w}}" for t in task_names)
        print(header)
        print('-' * len(header))
        for smi, m, s in zip(test_smiles, means, stds):
            if m is None:
                print(f"{smi:<45}  (invalid SMILES)")
            else:
                cols = "".join(f"{f'{v:.3f}±{u:.3f}':>{col_w}}" for v, u in zip(m, s))
                print(f"{smi:<45}{cols}")
