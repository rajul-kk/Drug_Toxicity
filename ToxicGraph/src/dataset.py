
import os
import gzip
import shutil
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, download_url
from src.featurizer import smiles_to_graph
from tqdm import tqdm


class ToxicDataset(InMemoryDataset):
    def __init__(self, root, name='tox21', transform=None, pre_transform=None):
        self.name = name.lower()
        super(ToxicDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.smiles_list = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self):
        return [f'{self.name}.csv']

    @property
    def processed_file_names(self):
        # v2: includes smiles_list and 41-dim atom features
        return [f'{self.name}_v2.pt']

    def download(self):
        if self.name == 'tox21':
            url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
            path = download_url(url, self.raw_dir)
            with gzip.open(path, 'rb') as f_in:
                with open(os.path.join(self.raw_dir, 'tox21.csv'), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(path)
        elif self.name == 'hiv':
            url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv'
            download_url(url, self.raw_dir)
        else:
            raise ValueError(f"Dataset '{self.name}' not supported. Choose 'tox21' or 'hiv'.")

    def process(self):
        df = pd.read_csv(self.raw_paths[0])

        if self.name == 'tox21':
            tasks = [
                'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
                'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
                'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
            ]
            smiles_col = 'smiles'
        elif self.name == 'hiv':
            tasks = ['HIV_active']
            smiles_col = 'smiles'

        for task in tasks:
            df[task] = pd.to_numeric(df[task], errors='coerce')

        data_list = []
        smiles_list = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row[smiles_col]
            data = smiles_to_graph(smiles)
            if data is None:
                continue

            labels = torch.tensor(row[tasks].values.astype(float), dtype=torch.float)
            labels = torch.nan_to_num(labels, nan=-1.0)
            data.y = labels.view(1, -1)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            smiles_list.append(smiles)

        data, slices = self.collate(data_list)
        torch.save((data, slices, smiles_list), self.processed_paths[0])
