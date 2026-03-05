import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import get_worker_info


def _seed_worker(worker_id):
    """Set random seed for data loading workers to ensure reproducibility."""
    worker_info = get_worker_info()
    rank = int(os.environ.get("RANK", "0"))
    base = torch.initial_seed() % (2**31)
    seed = base + worker_id + 997 * rank
    random.seed(seed)
    np.random.seed(seed)
    ds = worker_info.dataset
    
    if hasattr(ds, "set_rng") and callable(getattr(ds, "set_rng", None)):
        ds.set_rng(np.random.RandomState(seed))
    else:
        logging.warning(f"Dataset does not have a set_rng method")
