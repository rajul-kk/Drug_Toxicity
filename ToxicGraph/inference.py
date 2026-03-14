
import yaml
import torch
from src.models import GNN
from src.featurizer import smiles_to_graph
# Minimal dataset import to get simple logic if needed, but we can infer dims from saved implementation if we saved metadata 
# For now, we hardcode or load config. Since we trained with specific dims, we need to know them.
# We will assume config matches training.

def predict(smiles_list):
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # atom (10) + degree (7) + hybrid (6) + aromatic (1) = 24.
    num_node_features = 24 
    
    # Num tasks (classes) also needs to be known. Tox21=12.
    if config['dataset']['name'] == 'tox21':
        num_classes = 12
    elif config['dataset']['name'] == 'hiv':
        num_classes = 1
    else:
        num_classes = 12 # Default
        
    model = GNN(num_node_features=num_node_features, 
                hidden_channels=config['model']['hidden_channels'], 
                num_classes=num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load('model.pth', map_location=device))
    except FileNotFoundError:
        print("Model file not found. Please run train.py first.")
        return

    model.eval()
    
    predictions = []
    print(f"Predicting for {len(smiles_list)} molecules...")
    
    with torch.no_grad():
        for smiles in smiles_list:
            data = smiles_to_graph(smiles)
            if data is None:
                predictions.append(None)
                continue
                
            # Add batch dimension
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            data = data.to(device)
            
            out = model(data)
            probs = torch.sigmoid(out)
            predictions.append(probs.cpu().numpy().flatten())
            
    return predictions

if __name__ == '__main__':
    # Test with some example SMILES
    test_smiles = [
        # Existing
        'CCO', 
        'C1=CC=CC=C1',
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        
        # New examples
        'CC(=O)Nc1ccc(O)cc1',             # Paracetamol
        'Cn1cnc2c1c(=O)n(C)c(=O)n2C',     # Caffeine
        'Clc1ccc(cc1)C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl' # DDT
    ]
    preds = predict(test_smiles)
    for s, p in zip(test_smiles, preds):
        print(f"SMILES: {s}")
        print(f"Prediction: {p}")
