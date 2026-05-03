
import os
import yaml
import torch
from src.models import GNN, EnsembleGNN
from src.featurizer import smiles_to_graph

TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
]


def load_model(config, device):
    num_node_features = smiles_to_graph('C').x.shape[1]
    num_classes = 12 if config['dataset']['name'] == 'tox21' else 1
    hidden = config['model']['hidden_channels']
    ensemble_size = config['model'].get('ensemble_size', 3)

    models = []
    for i in range(ensemble_size):
        path = f'model_{i}.pth'
        if os.path.exists(path):
            m = GNN(num_node_features, hidden, num_classes).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)

    if not models:
        raise FileNotFoundError("No model files found (model_0.pth, …). Run train.py first.")

    return EnsembleGNN(models) if len(models) > 1 else models[0]


def predict(smiles_list):
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config, device)
    task_names = TOX21_TASKS if config['dataset']['name'] == 'tox21' else None

    predictions = []
    with torch.no_grad():
        for smiles in smiles_list:
            data = smiles_to_graph(smiles)
            if data is None:
                predictions.append(None)
                continue
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            data = data.to(device)
            probs = torch.sigmoid(model(data)).cpu().numpy().flatten()
            predictions.append(probs)

    return predictions, task_names


if __name__ == '__main__':
    test_smiles = [
        'CCO',                                          # Ethanol
        'C1=CC=CC=C1',                                  # Benzene
        'CC(=O)OC1=CC=CC=C1C(=O)O',                    # Aspirin
        'CC(=O)Nc1ccc(O)cc1',                           # Paracetamol
        'Cn1cnc2c1c(=O)n(C)c(=O)n2C',                  # Caffeine
        'Clc1ccc(cc1)C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl',      # DDT
    ]

    preds, task_names = predict(test_smiles)
    header = f"{'SMILES':<45}" + "".join(f"{t:>12}" for t in task_names)
    print(header)
    print('-' * len(header))
    for smi, p in zip(test_smiles, preds):
        if p is None:
            print(f"{smi:<45}  (invalid SMILES)")
        else:
            row = f"{smi:<45}" + "".join(f"{v:>12.3f}" for v in p)
            print(row)
