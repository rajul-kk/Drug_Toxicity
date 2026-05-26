import os
import yaml
import torch
import numpy as np
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

from src.models import build_and_load_ensemble
from src.dataset import DATASET_CONFIGS
from src.utils import smiles_to_sdf
from evaluate import collect_predictions, load_test_dataset
from inference import predict as _predict

DS_COLORS = {
    'tox21':   '#047857',
    'clintox': '#6d28d9',
    'sider':   '#1d4ed8',
}


def _build_task_info(config):
    names = config['dataset'].get('names') or [config['dataset']['name']]
    task_names, task_groups = [], {}
    for ds in names:
        tasks = DATASET_CONFIGS[ds]['tasks'] or []
        task_groups[ds] = tasks
        task_names.extend(tasks)
    return task_names, task_groups


def _load_available_ensembles(config, device):
    from sklearn.metrics import roc_auc_score, average_precision_score

    ensembles, test_caches = {}, {}
    for arch in ['gnn', 'dmpnn']:
        model_dir = os.path.join('checkpoints', arch)
        if not os.path.exists(os.path.join(model_dir, 'model_0.pth')):
            continue

        cfg = {**config, 'model': {**config['model'], 'type': arch}}
        ens = build_and_load_ensemble(cfg, device, model_dir=model_dir)

        temp_path = os.path.join(model_dir, 'temperature.pt')
        temperature = float(torch.load(temp_path, map_location='cpu')) \
                      if os.path.exists(temp_path) else 1.0

        test_dataset, _ = load_test_dataset(config)
        probs, labels = collect_predictions(ens, test_dataset, device, temperature)

        smiles_list = test_dataset.smiles_list
        source_list = test_dataset.source_list if hasattr(test_dataset, 'source_list') \
                      else [config['dataset'].get('names', ['unknown'])[0]] * len(smiles_list)
        sdf_list = [smiles_to_sdf(s) or '' for s in smiles_list]

        n_tasks = probs.shape[1]
        aucs, auprcs = [], []
        for mol_idx in range(len(smiles_list)):
            p, l = probs[mol_idx], labels[mol_idx]
            valid = l != -1
            if valid.sum() > 0:
                auc_vals = [roc_auc_score([l[t]], [p[t]]) if valid[t] else np.nan
                            for t in range(n_tasks)]
                auprc_vals = [average_precision_score([l[t]], [p[t]]) if valid[t] else np.nan
                              for t in range(n_tasks)]
                aucs.append(float(np.nanmean(auc_vals)))
                auprcs.append(float(np.nanmean(auprc_vals)))
            else:
                aucs.append(0.0)
                auprcs.append(0.0)

        ensembles[arch] = ens
        test_caches[arch] = {
            'smiles':       smiles_list,
            'sdf':          sdf_list,
            'probs':        probs,
            'labels':       labels,
            'dataset':      source_list,
            'max_conf':     probs.max(axis=1).tolist(),
            'auc_per_mol':  aucs,
            'auprc_per_mol': auprcs,
            'temperature':  temperature,
        }

    return ensembles, test_caches


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task_names, task_groups = _build_task_info(config)
    ensembles, test_caches = _load_available_ensembles(config, device)

    if not ensembles:
        raise RuntimeError(
            "No model weights found under checkpoints/gnn/ or checkpoints/dmpnn/. "
            "Run train.py first."
        )

    app.state.ensembles    = ensembles
    app.state.test_caches  = test_caches
    app.state.task_names   = task_names
    app.state.task_groups  = task_groups
    app.state.default_model = config['model'].get('type', 'gnn')
    app.state.device       = device
    app.state.executor     = ThreadPoolExecutor(max_workers=1)
    yield
    app.state.executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)
app.mount('/static', StaticFiles(directory='web/static'), name='static')
templates = Jinja2Templates(directory='web/templates')


# ── health ────────────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    return {'status': 'ok'}


# ── index ─────────────────────────────────────────────────────────────────────

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


# ── /api/info ─────────────────────────────────────────────────────────────────

@app.get('/api/info')
def api_info(request: Request):
    s = request.app.state
    return {
        'available_models': list(s.ensembles.keys()),
        'default_model':    s.default_model,
        'task_names':       s.task_names,
        'task_groups':      s.task_groups,
        'dataset_colors':   DS_COLORS,
    }


# ── /api/predict ──────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    smiles: str
    n_mc:   int = 30
    model:  Optional[str] = None


@app.post('/api/predict')
def api_predict(req: PredictRequest, request: Request):
    import concurrent.futures
    s = request.app.state
    model_key = req.model or s.default_model
    if model_key not in s.ensembles:
        model_key = next(iter(s.ensembles))

    ensemble    = s.ensembles[model_key]
    temperature = s.test_caches[model_key]['temperature']

    future = s.executor.submit(
        _predict, [req.smiles], req.n_mc,
        ensemble=ensemble, task_names=s.task_names,
        temperature=temperature, device=s.device,
    )
    try:
        means_list, stds_list, _ = future.result(timeout=60)
    except concurrent.futures.TimeoutError:
        raise HTTPException(503, 'Prediction timed out')

    if means_list[0] is None:
        raise HTTPException(422, 'Invalid SMILES — RDKit could not parse it')

    means = means_list[0].tolist()
    stds  = stds_list[0].tolist()
    sdf   = smiles_to_sdf(req.smiles) or ''

    top_idx = int(max(range(len(means)), key=lambda i: means[i]))
    return {
        'means':        means,
        'stds':         stds,
        'sdf':          sdf,
        'model_used':   model_key,
        'task_names':   s.task_names,
        'task_groups':  s.task_groups,
        'max_auc':      float(max(means)),
        'mc_std_mean':  float(sum(stds) / len(stds)),
        'top_task':     s.task_names[top_idx],
    }


# ── /api/testset ──────────────────────────────────────────────────────────────

@app.get('/api/testset')
def api_testset(request: Request, model: Optional[str] = None, page: int = 1,
                per_page: int = 20, filter: str = 'all', sort: str = 'conf'):
    s = request.app.state
    model_key = model or s.default_model
    if model_key not in s.test_caches:
        model_key = next(iter(s.test_caches))
    cache = s.test_caches[model_key]

    indices = list(range(len(cache['smiles'])))
    if filter != 'all':
        indices = [i for i in indices if cache['dataset'][i] == filter]

    sort_map = {
        'conf':  lambda i: -cache['max_conf'][i],
        'auc':   lambda i: -cache['auc_per_mol'][i],
        'auprc': lambda i: -cache['auprc_per_mol'][i],
    }
    indices.sort(key=sort_map.get(sort, sort_map['conf']))

    total = len(indices)
    start = (page - 1) * per_page
    page_slice = indices[start:start + per_page]

    rows = [{
        'idx':     i,
        'smiles':  cache['smiles'][i],
        'dataset': cache['dataset'][i],
        'max_conf': round(cache['max_conf'][i], 4),
        'auc':     round(cache['auc_per_mol'][i], 4),
        'auprc':   round(cache['auprc_per_mol'][i], 4),
    } for i in page_slice]

    return {
        'rows':       rows,
        'total':      total,
        'page':       page,
        'pages':      max(1, (total + per_page - 1) // per_page),
        'model_used': model_key,
    }


@app.get('/api/testset/{idx}')
def api_testset_single(idx: int, request: Request, model: Optional[str] = None):
    s = request.app.state
    model_key = model or s.default_model
    if model_key not in s.test_caches:
        model_key = next(iter(s.test_caches))
    cache = s.test_caches[model_key]

    if idx < 0 or idx >= len(cache['smiles']):
        raise HTTPException(404, f'Index {idx} out of range')

    return {
        'idx':        idx,
        'smiles':     cache['smiles'][idx],
        'sdf':        cache['sdf'][idx],
        'probs':      cache['probs'][idx].tolist(),
        'labels':     cache['labels'][idx].tolist(),
        'dataset':    cache['dataset'][idx],
        'model_used': model_key,
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
