
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_split(dataset, train_frac=0.8, val_frac=0.1):
    """No scaffold appears in more than one split — prevents data leakage."""
    scaffolds = {}
    for i, smi in enumerate(dataset.smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        else:
            sc = smi
        scaffolds.setdefault(sc, []).append(i)

    # Largest scaffold groups first for a deterministic, reproducible split
    groups = sorted(scaffolds.values(), key=len, reverse=True)

    n = len(dataset)
    train_cutoff = int(train_frac * n)
    val_cutoff = int((train_frac + val_frac) * n)

    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    train_ds = dataset[train_idx]
    train_ds.smiles_list = [dataset.smiles_list[i] for i in train_idx]

    val_ds = dataset[val_idx]
    val_ds.smiles_list = [dataset.smiles_list[i] for i in val_idx]

    test_ds = dataset[test_idx]
    test_ds.smiles_list = [dataset.smiles_list[i] for i in test_idx]

    return train_ds, val_ds, test_ds
