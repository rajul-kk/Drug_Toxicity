"""
Run once after training to save test-set predictions to disk.
App startup then skips batch inference entirely (~2 s vs ~30 s).

Usage:
    python precompute_cache.py          # processes all available architectures
    python precompute_cache.py --arch gnn
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

from evaluate import collect_predictions, load_test_dataset
from src.models import build_and_load_ensemble


def precompute(arch: str, config: dict) -> None:
    model_dir = os.path.join('checkpoints', arch)
    if not os.path.exists(os.path.join(model_dir, 'model_0.pth')):
        print(f'[{arch}] no weights found at {model_dir} — skipping')
        return

    device = torch.device('cpu')
    cfg = {**config, 'model': {**config['model'], 'type': arch}}

    temp_path = os.path.join(model_dir, 'temperature.pt')
    temperature = float(torch.load(temp_path, map_location='cpu')) \
                  if os.path.exists(temp_path) else 1.0

    print(f'[{arch}] loading ensemble…')
    ens = build_and_load_ensemble(cfg, device, model_dir=model_dir)

    print(f'[{arch}] running batch inference on test set…')
    test_ds, _ = load_test_dataset(config)
    probs, labels = collect_predictions(ens, test_ds, device, temperature)

    smiles_list = test_ds.smiles_list
    source_list = (test_ds.source_list if hasattr(test_ds, 'source_list')
                   else [config['dataset'].get('names', ['unknown'])[0]] * len(smiles_list))

    np.save(os.path.join(model_dir, 'test_probs.npy'),   probs)
    np.save(os.path.join(model_dir, 'test_labels.npy'),  labels)
    json.dump(smiles_list, open(os.path.join(model_dir, 'test_smiles.json'), 'w'))
    json.dump(source_list, open(os.path.join(model_dir, 'test_sources.json'), 'w'))

    print(f'[{arch}] saved {len(smiles_list)} molecules → {model_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['gnn', 'dmpnn'],
                        help='which architecture to precompute (default: all available)')
    args = parser.parse_args()

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    archs = [args.arch] if args.arch else ['gnn', 'dmpnn']
    for arch in archs:
        precompute(arch, config)

    print('done.')
