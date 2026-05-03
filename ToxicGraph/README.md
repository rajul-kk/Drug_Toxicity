# ToxicGraph

GNN-based molecular toxicity prediction on the [Tox21](https://tripod.nih.gov/tox21/) benchmark — 12 nuclear receptor and stress-response assays, ~8k compounds.

## Models

Two architectures are available and can be swapped via `config.yaml`:

**GNN** (`type: gnn`) — 4-layer GATv2 with residual connections, BatchNorm, and Set2Set pooling. Each layer applies multi-head graph attention over atom neighbourhoods using both node and edge features.

**D-MPNN** (`type: dmpnn`) — Directed Message Passing Neural Network ([Chemprop](https://github.com/chemprop/chemprop), Yang et al. 2019). Messages travel along directed bonds. When updating bond (u→v), all bonds entering u are aggregated *except* the reverse (v→u), preventing message echo and preserving bond identity through all message passing steps.

Both use Set2Set graph pooling, an ensemble of 3 independently-seeded models, and MC Dropout uncertainty at inference.

## Features

- **Node features (44-dim):** atom type, degree, hybridisation, aromaticity, formal charge, H count, ring membership, chirality, bond-angle statistics (SE(3)-invariant), Gasteiger partial charge
- **Edge features (8-dim):** bond type, conjugation, ring membership, 3D interatomic distance (SE(3)-invariant)
- **Training:** focal loss (γ=2) for class imbalance, early stopping on val AUC, ReduceLROnPlateau
- **Calibration:** post-hoc temperature scaling fitted on the val set; Brier score and ECE reported at test time
- **Uncertainty:** MC Dropout — 30 stochastic forward passes at inference, outputs mean ± std per task
- **Baseline:** XGBoost on Morgan ECFP4 fingerprints (radius=2, 2048 bits), same scaffold split

## Installation

```bash
pip install torch torch-geometric rdkit-pypi scikit-learn xgboost tqdm pyyaml
```

## Usage

**Train:**
```bash
python train.py
```
Switch architecture by editing `config.yaml` → `model.type: dmpnn`. Saves `model_0.pth`, `model_1.pth`, `model_2.pth`, and `temperature.pt`.

**Benchmark both architectures:**
```bash
# GNN
python train.py                          # config.yaml: type: gnn

# D-MPNN  
# edit config.yaml: type: dmpnn, then:
python train.py
```

**XGBoost baseline:**
```bash
python baseline.py
```

**Inference with uncertainty:**
```bash
python inference.py
```
Output format: `mean±std` probability per task, calibrated via temperature scaling if `temperature.pt` exists.

**GNNExplainer attribution:**
```bash
python explain.py
```
Produces a highlighted atom-importance image in `explanations/`.

## Configuration

```yaml
model:
  type: gnn          # gnn | dmpnn
  hidden_channels: 128
  ensemble_size: 3
  depth: 4           # message passing steps (dmpnn) or conv layers (gnn)

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
```

## Project Structure

```
ToxicGraph/
├── train.py              # training pipeline
├── inference.py          # ensemble prediction with MC Dropout uncertainty
├── baseline.py           # XGBoost + Morgan fingerprint benchmark
├── explain.py            # GNNExplainer atom attribution
├── config.yaml
└── src/
    ├── models.py         # GNN, DMPNN, EnsembleGNN
    ├── featurizer.py     # SMILES → PyG graph (3D geometry, Gasteiger charges)
    ├── dataset.py        # ToxicDataset (InMemoryDataset, auto-download)
    ├── splitter.py       # scaffold-based train/val/test split
    ├── calibration.py    # temperature scaling, MC Dropout, ECE, Brier
    └── utils.py          # 2D/3D molecule visualisation
```

## Results

Scaffold split, ensemble of 3 models. GNN results after ~60 epochs on CPU.

| Model | Mean AUC | Mean AUPRC |
|---|---|---|
| XGBoost (ECFP4) | 0.731 | 0.340 |
| GNN (GATv2 + Set2Set) | 0.747 | 0.365 |
| D-MPNN | — | — |
| Chemprop (reported) | ~0.840 | — |
