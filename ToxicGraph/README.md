# ToxicGraph

Multi-dataset molecular safety prediction — GNN/DMPNN ensemble across 49 tasks covering toxicity, mutagenicity, cardiac risk, liver injury, and CYP450 inhibition. Interactive web UI for single-molecule prediction, batch CSV processing, test-set browsing, and similarity search.

---

## Datasets

| Dataset | Tasks | Domain |
|---------|-------|--------|
| Tox21 | 12 | Nuclear receptor & stress-response assays |
| ClinTox | 2 | FDA approval / clinical toxicity |
| SIDER | 27 | Drug side-effect organ-system categories |
| AMES | 1 | Mutagenicity |
| CYP450 | 5 | CYP1A2 / 2C9 / 2D6 / 3A4 / 2C19 inhibition |
| hERG | 1 | Cardiac ion-channel blockade |
| DILI | 1 | Drug-induced liver injury |
| BBBP* | 1 | Blood-brain barrier permeability |
| HIV* | 1 | HIV replication inhibition |

*BBBP and HIV use a separate Random Forest + fingerprint model (`train_activity.py`), not the GNN ensemble.

---

## Models

**GNN** — 4-layer GATv2 with residual connections, BatchNorm, and Set2Set pooling. Multi-head graph attention over atom neighbourhoods using both node and edge features.

**D-MPNN** — Directed Message Passing Neural Network ([Chemprop](https://github.com/chemprop/chemprop), Yang et al. 2019). Messages travel along directed bonds; when updating bond u→v, all bonds entering u are aggregated except the reverse v→u, preventing message echo.

Both models:
- Ensemble of 2 independently-seeded models, predictions averaged
- Post-hoc temperature scaling calibration fitted on val set
- MC Dropout uncertainty: 30 stochastic forward passes at inference → mean ± std per task

**Activity model (BBBP / HIV)** — Random Forest on Morgan ECFP4 (2048-bit) + MACCS (167-bit) fingerprints.

---

## Molecular Features

**Node features (44-dim):** atom type, degree, hybridisation, aromaticity, formal charge, H count, ring membership, chirality, bond-angle statistics (SE(3)-invariant), Gasteiger partial charge

**Edge features (8-dim):** bond type, conjugation, ring membership, 3D interatomic distance (SE(3)-invariant)

---

## Results

Scaffold split (80/10/10), ensemble of 2 models, calibrated with temperature scaling.

| Model | Mean AUC | Mean AUPRC | Mean Brier | Mean ECE |
|-------|----------|------------|------------|----------|
| GNN (GATv2 + Set2Set) | 0.678 | 0.566 | 0.176 | 0.152 |
| D-MPNN | 0.693 | 0.573 | 0.178 | 0.155 |

CYP450 tasks are the strongest performers (AUC 0.80–0.86). Some SIDER endpoints are difficult (AUC 0.48–0.53) due to sparse labels and class imbalance. Full per-task metrics are in [metrics.md](metrics.md).

---

## Web UI

FastAPI backend + Vite-bundled frontend. Run locally:

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:8000
```

Or with Docker:

```bash
docker build -t toxicgraph .
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -v $(pwd)/data:/app/data:ro \
  toxicgraph
```

**Views:**

- **Predict** — paste a SMILES string; get calibrated probabilities with MC Dropout uncertainty badges, dataset filter chips, GNNExplainer atom attribution, bookmark, and PDF/print report
- **Browse** — paginated test-set table with SMARTS substructure search and dataset filter
- **Batch** — CSV upload or paste, runs all SMILES in chunks with live progress
- **Similar** — Tanimoto nearest-neighbour search across the test set (Morgan FP)
- **History** — last 50 predictions grouped by dataset, with ★ bookmarks

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model load status and task count |
| `/api/info` | GET | Task names, groups, available models |
| `/api/predict` | POST | GNN/DMPNN ensemble prediction for a SMILES |
| `/api/explain` | GET | GNNExplainer atom-importance SVG |
| `/api/similar` | GET | Tanimoto nearest neighbours from test set |
| `/api/testset` | GET | Paginated test-set rows with probabilities and labels |
| `/api/testset/{idx}` | GET | Single test-set molecule |
| `/api/search` | GET | SMARTS substructure search over test set |
| `/api/properties/{smiles}` | GET | RDKit descriptors (MW, logP, TPSA, QED, Lipinski) |
| `/api/activity` | GET | RF model prediction for BBBP / HIV |

Rate-limited: 60 req/min for predict, 30/min for similar, 10/min for explain.

---

## Training

```bash
# GNN/DMPNN multi-dataset ensemble
python train.py

# Random Forest activity model (BBBP + HIV)
python train_activity.py

# Evaluate on test set, print per-task metrics
python evaluate.py
```

Switch architecture by editing `config.yaml` → `model.type: gnn` or `dmpnn`.

```yaml
dataset:
  names: [tox21, clintox, sider, ames, cyp450, herg, dili]
  activity_names: [bbbp, hiv]

model:
  type: dmpnn          # gnn | dmpnn
  hidden_channels: 256
  ensemble_size: 2
  depth: 4

training:
  batch_size: 64
  learning_rate: 0.0003
  epochs: 60
  early_stopping_patience: 12
  dataset_weights:     # upweight rare/hard datasets
    tox21: 1.0
    clintox: 4.0
    sider: 5.0
    ames: 2.0
    cyp450: 1.5
    herg: 3.0
    dili: 4.0
```

---

## Project Structure

```
ToxicGraph/
├── app.py                  # FastAPI server — all routes, cache, rate limiting
├── train.py                # GNN/DMPNN multi-dataset training pipeline
├── train_activity.py       # RF fingerprint model for BBBP/HIV
├── inference.py            # Ensemble prediction with MC Dropout
├── evaluate.py             # Per-task AUC/AUPRC/Brier/ECE on test set
├── explain.py              # GNNExplainer atom attribution → SVG
├── baseline.py             # XGBoost + ECFP4 benchmark
├── precompute_cache.py     # Warm inference cache at startup
├── config.yaml
├── metrics.md              # Full per-task results table
├── src/
│   ├── models.py           # GNN, DMPNN, EnsembleGNN
│   ├── featurizer.py       # SMILES → PyG graph (3D geometry, Gasteiger)
│   ├── dataset.py          # ToxicDataset (multi-dataset InMemoryDataset)
│   ├── splitter.py         # Scaffold-based train/val/test split
│   ├── calibration.py      # Temperature scaling, MC Dropout, ECE, Brier
│   └── utils.py            # Fingerprint utilities
└── web/
    ├── templates/index.html
    ├── static/
    │   ├── app.css
    │   ├── components.css
    │   ├── landing.css
    │   └── dist/bundle.js  # Vite build output
    └── src/
        ├── main.js         # Entry point, window bindings, error boundary
        ├── app.js          # Boot, routing, model selection
        ├── predict.js      # Single-molecule predict view + chart
        ├── browse.js       # Test-set browse and search
        ├── multi.js        # Batch prediction + CSV upload
        ├── similar.js      # Similarity search
        ├── history.js      # Prediction history + bookmarks
        └── viewer.js       # 3Dmol.js conformer viewer
```

---

## Tests

```bash
python -m pytest tests/ -v
```

26 tests covering health, predict, browse, search, properties, `/api/similar`, SMILES validation, inference cache, output shape, and probability bounds.
