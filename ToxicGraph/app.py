import io
import json
import os
import yaml
import torch
import numpy as np
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

        _cp = lambda f: os.path.join(model_dir, f)
        if all(os.path.exists(_cp(f)) for f in
               ('test_probs.npy', 'test_labels.npy', 'test_smiles.json', 'test_sources.json')):
            probs       = np.load(_cp('test_probs.npy'))
            labels      = np.load(_cp('test_labels.npy'))
            smiles_list = json.load(open(_cp('test_smiles.json')))
            source_list = json.load(open(_cp('test_sources.json')))
            print(f'  {arch}: loaded precomputed cache ({len(smiles_list)} molecules)')
        else:
            test_dataset, _ = load_test_dataset(config)
            probs, labels = collect_predictions(ens, test_dataset, device, temperature)
            smiles_list = test_dataset.smiles_list
            source_list = test_dataset.source_list if hasattr(test_dataset, 'source_list') \
                          else [config['dataset'].get('names', ['unknown'])[0]] * len(smiles_list)
            print(f'  {arch}: ran batch inference ({len(smiles_list)} molecules)'
                  f' — run precompute_cache.py to speed up future startups')
        n_tasks = probs.shape[1]
        scores = []
        for mol_idx in range(len(smiles_list)):
            p, l = probs[mol_idx], labels[mol_idx]
            valid = l != -1
            if valid.sum() > 0:
                # mean probability of the correct label across valid tasks
                vals = [float(p[t]) if l[t] == 1 else float(1.0 - p[t])
                        for t in range(n_tasks) if valid[t]]
                scores.append(float(np.mean(vals)))
            else:
                scores.append(0.0)

        n = len(smiles_list)
        max_conf_list = probs.max(axis=1).tolist()
        datasets_seen = list(dict.fromkeys(source_list))

        def _sorted_idx(vals):
            return sorted(range(n), key=lambda i: -vals[i])

        idx_conf  = _sorted_idx(max_conf_list)
        idx_score = _sorted_idx(scores)

        def _filter(idx_sorted, ds):
            return [i for i in idx_sorted if source_list[i] == ds]

        ensembles[arch] = ens
        test_caches[arch] = {
            'smiles':       smiles_list,
            'probs':        probs,
            'labels':       labels,
            'dataset':      source_list,
            'max_conf':     max_conf_list,
            'score_per_mol': scores,
            'temperature':  temperature,
            'idx': {
                'all': {'conf': idx_conf, 'score': idx_score},
                **{ds: {'conf': _filter(idx_conf, ds), 'score': _filter(idx_score, ds)}
                   for ds in datasets_seen},
            },
        }

    return ensembles, test_caches


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.getenv('CONFIG_PATH', 'config.yaml')
    with open(config_path) as f:
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

    # warm smiles_to_sdf lru_cache after server is live — daemon so it doesn't block shutdown
    import threading
    _all_smiles = list({s for cache in test_caches.values() for s in cache['smiles']})
    threading.Thread(
        target=lambda: [smiles_to_sdf(s) for s in _all_smiles],
        daemon=True,
    ).start()

    yield
    app.state.executor.shutdown(wait=False)


limiter = Limiter(
    key_func=get_remote_address,
    enabled=os.getenv('RATE_LIMIT', 'true').lower() == 'true',
)

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('CORS_ORIGINS', '*').split(','),
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)
app.mount('/static', StaticFiles(directory='web/static'), name='static')
templates = Jinja2Templates(directory='web/templates')


# ── health ────────────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    return {'status': 'ok'}


# ── thumbnail ─────────────────────────────────────────────────────────────────

@app.get('/api/thumbnail/{smiles:path}')
@limiter.limit('60/minute')
def api_thumbnail(smiles: str, request: Request, size: int = 80):
    from rdkit import Chem
    from rdkit.Chem import Draw
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(404, 'Invalid SMILES')
    img = Draw.MolToImage(mol, size=(size, size), kekulize=True)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return Response(content=buf.read(), media_type='image/png',
                    headers={'Cache-Control': 'public, max-age=86400'})


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
    n_mc:   int = 20
    model:  Optional[str] = None


@app.post('/api/predict')
@limiter.limit('20/minute')
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

    idx_group = cache['idx'].get(filter, cache['idx']['all'])
    idx_list  = idx_group.get(sort, idx_group['conf'])

    total = len(idx_list)
    start = (page - 1) * per_page
    page_slice = idx_list[start:start + per_page]

    rows = [{
        'idx':     i,
        'smiles':  cache['smiles'][i],
        'dataset': cache['dataset'][i],
        'max_conf': round(cache['max_conf'][i], 4),
        'score':   round(cache['score_per_mol'][i], 4),
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

    smiles = cache['smiles'][idx]
    return {
        'idx':        idx,
        'smiles':     smiles,
        'sdf':        smiles_to_sdf(smiles) or '',
        'probs':      cache['probs'][idx].tolist(),
        'labels':     cache['labels'][idx].tolist(),
        'dataset':    cache['dataset'][idx],
        'model_used': model_key,
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'app:app',
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 8000)),
        reload=os.getenv('DEV', 'false').lower() == 'true',
    )
