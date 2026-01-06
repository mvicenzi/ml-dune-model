# loader/splits.py

import torch
from torch.utils.data import Subset

def compute_split_indices(n_total, val_fraction, seed):
    """Computes training and validation indices for a dataset split.
    Args:
        n_total: Total number of samples in the dataset.
        val_fraction: Fraction of data to use for validation.
        seed: Random seed for reproducibility.
    Returns:
        A tuple (train_indices, val_indices).
    """   
    n_val = int(val_fraction * n_total)

    # randomly permute indices with fixed seed
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=g)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return train_idx, val_idx

def train_val_split(dataset, val_fraction=0.2, seed=4247, use_cache=True):
    """Splits a dataset into training and validation subsets.
    Args:
        dataset: The dataset to split.
        val_fraction: Fraction of data to use for validation.
        seed: Random seed for reproducibility.
        use_cache: If True, uses a cached split if available.
    Returns:
        A tuple (train_subset, val_subset).
    """
    n_total = len(dataset)
    train_idx, val_idx = None, None

    # if use_cache is enabled, try to load cached indices
    if use_cache and hasattr(dataset, 'cache_dir'):
        cache_dir = dataset.cache_dir
        cache_dir.mkdir(exist_ok=True)
        train_cache_file = cache_dir / "train_indices.pt"
        val_cache_file = cache_dir / "val_indices.pt"

        if train_cache_file.exists() and val_cache_file.exists():
            print(f"Loading train/val split indices from {cache_dir}.")
            train_idx = torch.load(train_cache_file)
            val_idx = torch.load(val_cache_file)

        else: #if cache files do not exist, compute and also save
            train_idx, val_idx = compute_split_indices(n_total, val_fraction, seed)
            torch.save(train_idx, train_cache_file)
            torch.save(val_idx, val_cache_file)

    else: # if not using cache, just compute
        print("Computing train/val split indices without cache.")
        train_idx, val_idx = compute_split_indices(n_total, val_fraction, seed)

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        train_idx,
        val_idx
    )
