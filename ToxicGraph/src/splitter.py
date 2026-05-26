
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _scaffold_split_indices(smiles_list, global_indices, train_frac, val_frac):
    """Return (train, val, test) as lists of indices into global_indices."""
    scaffolds = {}
    for local_i, global_i in enumerate(global_indices):
        smi = smiles_list[global_i]
        mol = Chem.MolFromSmiles(smi)
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
        scaffolds.setdefault(sc, []).append(local_i)

    groups = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(global_indices)
    train_cut = int(train_frac * n)
    val_cut = int((train_frac + val_frac) * n)

    local_train, local_val, local_test = [], [], []
    for group in groups:
        if len(local_train) + len(group) <= train_cut:
            local_train.extend(group)
        elif len(local_train) + len(local_val) + len(group) <= val_cut:
            local_val.extend(group)
        else:
            local_test.extend(group)

    return (
        [global_indices[i] for i in local_train],
        [global_indices[i] for i in local_val],
        [global_indices[i] for i in local_test],
    )


def _make_split(dataset, indices):
    ds = dataset[indices]
    ds.smiles_list = [dataset.smiles_list[i] for i in indices]
    if hasattr(dataset, 'source_list'):
        ds.source_list = [dataset.source_list[i] for i in indices]
    return ds


def scaffold_split(dataset, train_frac=0.8, val_frac=0.1):
    """No scaffold appears in more than one split — prevents data leakage."""
    all_idx = list(range(len(dataset)))
    train_idx, val_idx, test_idx = _scaffold_split_indices(
        dataset.smiles_list, all_idx, train_frac, val_frac
    )
    return _make_split(dataset, train_idx), _make_split(dataset, val_idx), _make_split(dataset, test_idx)


def multidataset_scaffold_split(dataset, train_frac=0.8, val_frac=0.1):
    """Per-source scaffold split: each sub-dataset gets its own 80/10/10 split
    so every dataset has fair test representation regardless of combined size."""
    by_source = defaultdict(list)
    for i, source in enumerate(dataset.source_list):
        by_source[source].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for source, indices in by_source.items():
        tr, va, te = _scaffold_split_indices(dataset.smiles_list, indices, train_frac, val_frac)
        train_idx.extend(tr)
        val_idx.extend(va)
        test_idx.extend(te)
        print(f"  {source}: {len(tr)} train / {len(va)} val / {len(te)} test")

    return _make_split(dataset, train_idx), _make_split(dataset, val_idx), _make_split(dataset, test_idx)
