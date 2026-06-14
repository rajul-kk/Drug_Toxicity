"""
Download missing datasets (ames, cyp450, herg, dili) from Harvard Dataverse (TDC mirror).
Saves files to data/raw/ in the CSV format expected by src/dataset.py.
Run once: python fetch_missing_data.py
"""

import os
import io
import requests
import pandas as pd

RAW_DIR = os.path.join('data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

DATAVERSE_URL = 'https://dataverse.harvard.edu/api/access/datafile/{}'

# TDC file IDs and their output mapping
SINGLE_TASK = [
    {
        'id':       6822246,
        'filename': 'herg.csv',
        'task_col': 'herg_blocker',
    },
    {
        'id':       4259585,
        'filename': 'dili.csv',
        'task_col': 'dili_concern',
    },
    {
        'id':       4259564,
        'filename': 'ames.csv',
        'task_col': 'mutagenic',
    },
]

CYP_ISOFORMS = [
    ('cyp1a2_inhibitor',  4259573),
    ('cyp2c9_inhibitor',  4259577),
    ('cyp2d6_inhibitor',  4259580),
    ('cyp3a4_inhibitor',  4259582),
    ('cyp2c19_inhibitor', 4259576),
]
CYP_TASKS = [t for t, _ in CYP_ISOFORMS]


def fetch_tab(file_id: int) -> pd.DataFrame:
    url = DATAVERSE_URL.format(file_id)
    r = requests.get(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    # Strip surrounding quotes from SMILES if present
    df = pd.read_csv(io.StringIO(r.text), sep='\t')
    df['Drug'] = df['Drug'].str.strip('"')
    return df


def download_single(cfg: dict) -> None:
    dest = os.path.join(RAW_DIR, cfg['filename'])
    if os.path.exists(dest):
        print(f'  skip  {cfg["filename"]} (already exists)')
        return
    print(f'  fetch {cfg["filename"]} ...')
    df = fetch_tab(cfg['id'])
    out = pd.DataFrame({'smiles': df['Drug'], cfg['task_col']: df['Y'].astype(int)})
    out.to_csv(dest, index=False)
    print(f'  saved {dest}  ({len(out):,} rows)')


def download_cyp450() -> None:
    dest = os.path.join(RAW_DIR, 'cyp450.csv')
    if os.path.exists(dest):
        print(f'  skip  cyp450.csv (already exists)')
        return

    frames = {}
    for task_name, file_id in CYP_ISOFORMS:
        print(f'  fetch cyp450/{task_name} ...')
        df = fetch_tab(file_id)
        frames[task_name] = df[['Drug', 'Y']].rename(
            columns={'Drug': 'smiles', 'Y': task_name}
        ).set_index('smiles')

    merged = frames[CYP_ISOFORMS[0][0]]
    for task_name, _ in CYP_ISOFORMS[1:]:
        merged = merged.join(frames[task_name], how='outer')

    merged = merged.reset_index()
    # Fill missing labels with -1 (treated as "no label" in dataset.py)
    task_cols = [t for t, _ in CYP_ISOFORMS]
    merged[task_cols] = merged[task_cols].fillna(-1).astype(int)

    merged.to_csv(dest, index=False)
    print(f'  saved {dest}  ({len(merged):,} rows)')


if __name__ == '__main__':
    print('Fetching missing datasets from Harvard Dataverse (TDC mirror)...')
    for cfg in SINGLE_TASK:
        download_single(cfg)
    download_cyp450()
    print('\nDone. All missing datasets saved to data/raw/.')
