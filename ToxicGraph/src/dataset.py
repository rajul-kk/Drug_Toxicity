
import os
import gzip
import shutil
import pandas as pd
import torch
import requests
from torch_geometric.data import InMemoryDataset
from src.featurizer import smiles_to_graph, FEATURIZER_HASH
from tqdm import tqdm


def _download_file(url: str, dest_dir: str, filename: str) -> str:
    """Download url → dest_dir/filename using requests (avoids urllib 403s on S3)."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        return dest
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; ToxicGraph/1.0)'}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
    return dest

TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
]

CLINTOX_TASKS = ['FDA_APPROVED', 'CT_TOX']

SIDER_TASKS = [
    'Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues',
    'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders',
    'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders',
    'Reproductive system and breast disorders',
    'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
    'General disorders and administration site conditions', 'Endocrine disorders',
    'Surgical and medical procedures', 'Vascular disorders',
    'Blood and lymphatic system disorders', 'Skin and subcutaneous tissue disorders',
    'Congenital, familial and genetic disorders', 'Infections and infestations',
    'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
    'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions',
    'Ear and labyrinth disorders', 'Cardiac disorders', 'Nervous system disorders',
    'Injury, poisoning and procedural complications',
]

MUV_TASKS = [
    'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
    'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
    'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859',
]

DATASET_CONFIGS = {
    'tox21': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
        'smiles_col': 'smiles',
        'tasks': TOX21_TASKS,
        'gzip': True,
    },
    'clintox': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz',
        'smiles_col': 'smiles',
        'tasks': CLINTOX_TASKS,
        'gzip': True,
    },
    'sider': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz',
        'smiles_col': 'smiles',
        'tasks': SIDER_TASKS,
        'gzip': True,
    },
    'hiv': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv',
        'smiles_col': 'smiles',
        'tasks': ['HIV_active'],
        'gzip': False,
    },
    # Blood-brain barrier penetration (Martins et al. 2012): ~2k compounds, 1 task.
    'bbbp': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
        'smiles_col': 'smiles',
        'tasks': ['p_np'],
        'gzip': False,
    },
    # Beta-secretase 1 inhibition (Subramanian et al. 2016): ~1.5k compounds, 1 task.
    'bace': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
        'smiles_col': 'mol',
        'tasks': ['Class'],
        'gzip': False,
    },
    # Maximum Unbiased Validation bioassays: ~93k compounds, 17 tasks, extreme imbalance (~0.2% positive).
    'muv': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz',
        'smiles_col': 'smiles',
        'tasks': MUV_TASKS,
        'gzip': True,
    },
    # tasks=None means all columns except smiles_col are treated as tasks (auto-detected at process time).
    # ToxCast and PCBA have auto-detected tasks — not compatible with MultiToxDataset; use standalone only.
    'toxcast': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz',
        'smiles_col': 'smiles',
        'tasks': None,
        'gzip': True,
    },
    # PubChem BioAssay: ~440k compounds, 128 tasks. Large dataset, standalone only.
    'pcba': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz',
        'smiles_col': 'smiles',
        'tasks': None,
        'gzip': True,
    },
    # Ames mutagenicity (Kazius et al. 2005): ~6.5k compounds, 1 task.
    # Pre-fetched via fetch_missing_data.py → data/raw/ames.csv
    'ames': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/mutagenicity.csv.gz',
        'smiles_col': 'smiles',
        'tasks': ['mutagenic'],
        'gzip': False,
    },
    # CYP P450 inhibition (TDC isoform datasets outer-joined): ~16k compounds, 5 isoforms.
    # Key ADMET endpoint — CYP inhibition drives most drug-drug interactions.
    # Pre-fetched via fetch_missing_data.py → data/raw/cyp450.csv
    'cyp450': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/cyp450.csv.gz',
        'smiles_col': 'smiles',
        'tasks': ['cyp1a2_inhibitor', 'cyp2c9_inhibitor', 'cyp2d6_inhibitor',
                  'cyp3a4_inhibitor', 'cyp2c19_inhibitor'],
        'gzip': False,
    },
    # hERG cardiac ion channel blockade (Karim et al. 2019): ~13k compounds, binary.
    # IC50 < 10 µM → blocker (1). #1 cause of drug withdrawal — QT prolongation.
    # Pre-fetched via fetch_missing_data.py → data/raw/herg.csv
    'herg': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/herg_karim.csv',
        'smiles_col': 'smiles',
        'tasks': ['herg_blocker'],
        'gzip': False,
    },
    # Drug-induced liver injury (DILIst, Liu et al. 2015): ~1.8k FDA-approved drugs, binary.
    # DILI+ (1) = concern; DILI- (0) = no concern. #1 cause of drug market withdrawals.
    # Pre-fetched via fetch_missing_data.py → data/raw/dili.csv
    'dili': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/DILI.csv',
        'smiles_col': 'smiles',
        'tasks': ['dili_concern'],
        'gzip': False,
    },
}


class ToxicDataset(InMemoryDataset):
    def __init__(self, root, name='tox21', transform=None, pre_transform=None):
        self.name = name.lower()
        super(ToxicDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.smiles_list, self.tasks = torch.load(
            self.processed_paths[0], weights_only=False
        )
        self.primary_tasks = self.tasks

    @property
    def raw_file_names(self):
        return [f'{self.name}.csv']

    @property
    def processed_file_names(self):
        return [f'{self.name}_{FEATURIZER_HASH}.pt']

    def download(self):
        cfg = DATASET_CONFIGS.get(self.name)
        if cfg is None:
            raise ValueError(f"Dataset '{self.name}' not supported. Choose from: {list(DATASET_CONFIGS)}")
        raw_filename = cfg['url'].split('/')[-1]
        path = _download_file(cfg['url'], self.raw_dir, raw_filename)
        if cfg['gzip']:
            with gzip.open(path, 'rb') as f_in:
                with open(os.path.join(self.raw_dir, f'{self.name}.csv'), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(path)

    def process(self):
        cfg = DATASET_CONFIGS[self.name]
        smiles_col = cfg['smiles_col']
        df = pd.read_csv(self.raw_paths[0])
        col_map = cfg.get('col_map', {})
        if col_map:
            df = df.rename(columns=col_map)
        # tasks=None means every column except smiles_col is a task (e.g. ToxCast)
        tasks = cfg['tasks'] if cfg['tasks'] is not None else [c for c in df.columns if c != smiles_col]
        for task in tasks:
            df[task] = pd.to_numeric(df[task], errors='coerce')

        data_list, smiles_list = [], []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row[smiles_col]
            data = smiles_to_graph(smiles)
            if data is None:
                continue
            labels = torch.tensor(row[tasks].values.astype(float), dtype=torch.float)
            data.y = torch.nan_to_num(labels, nan=-1.0).view(1, -1)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            smiles_list.append(smiles)

        data, slices = self.collate(data_list)
        torch.save((data, slices, smiles_list, tasks), self.processed_paths[0])


class MultiToxDataset(InMemoryDataset):
    """
    Concatenates multiple toxicity datasets into one.
    Each molecule's label vector spans all tasks across all datasets;
    tasks from other datasets are filled with -1 (masked in loss).
    The first name in `names` is the primary dataset used for evaluation.
    """

    def __init__(self, root, names, transform=None, pre_transform=None):
        self.names = [n.lower() for n in names]
        for n in self.names:
            if n not in DATASET_CONFIGS:
                raise ValueError(f"Dataset '{n}' not supported. Choose from: {list(DATASET_CONFIGS)}")
            if DATASET_CONFIGS[n]['tasks'] is None:
                raise ValueError(
                    f"Dataset '{n}' uses auto-detected tasks (e.g. ToxCast) and cannot be "
                    f"combined in MultiToxDataset. Use it standalone via names: [{n}]."
                )

        # Build combined task list and per-dataset slice ranges
        self.all_tasks = []
        self.task_ranges = {}
        for n in self.names:
            start = len(self.all_tasks)
            tasks = DATASET_CONFIGS[n]['tasks']
            self.all_tasks.extend(tasks)
            self.task_ranges[n] = (start, start + len(tasks))

        super(MultiToxDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.smiles_list, self.source_list = torch.load(
            self.processed_paths[0], weights_only=False
        )
        self.primary_tasks = DATASET_CONFIGS[self.names[0]]['tasks']

    @property
    def raw_file_names(self):
        return [f'{n}.csv' for n in self.names]

    @property
    def processed_file_names(self):
        key = '_'.join(self.names)
        return [f'multi_{key}_{FEATURIZER_HASH}.pt']

    def download(self):
        for n in self.names:
            csv_path = os.path.join(self.raw_dir, f'{n}.csv')
            if os.path.exists(csv_path):
                continue
            cfg = DATASET_CONFIGS[n]
            raw_filename = cfg['url'].split('/')[-1]
            path = _download_file(cfg['url'], self.raw_dir, raw_filename)
            if cfg['gzip']:
                with gzip.open(path, 'rb') as f_in:
                    with open(csv_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(path)

    def process(self):
        num_total = len(self.all_tasks)
        data_list, smiles_list, source_list = [], [], []

        for n in self.names:
            cfg = DATASET_CONFIGS[n]
            tasks, smiles_col = cfg['tasks'], cfg['smiles_col']
            start, end = self.task_ranges[n]
            df = pd.read_csv(os.path.join(self.raw_dir, f'{n}.csv'))
            col_map = cfg.get('col_map', {})
            if col_map:
                df = df.rename(columns=col_map)
            for task in tasks:
                df[task] = pd.to_numeric(df[task], errors='coerce')

            for _, row in tqdm(df.iterrows(), total=len(df), desc=n):
                smiles = row[smiles_col]
                data = smiles_to_graph(smiles)
                if data is None:
                    continue
                labels = torch.full((num_total,), -1.0)
                src_labels = torch.tensor(row[tasks].values.astype(float), dtype=torch.float)
                labels[start:end] = torch.nan_to_num(src_labels, nan=-1.0)
                data.y = labels.unsqueeze(0)
                data_list.append(data)
                smiles_list.append(smiles)
                source_list.append(n)

        data, slices = self.collate(data_list)
        torch.save((data, slices, smiles_list, source_list), self.processed_paths[0])


def load_dataset(config):
    """Return the right dataset class based on config. Supports name: or names:."""
    root = config['dataset']['root']
    names = config['dataset'].get('names') or [config['dataset']['name']]
    if len(names) == 1:
        return ToxicDataset(root=root, name=names[0])
    return MultiToxDataset(root=root, names=names)
