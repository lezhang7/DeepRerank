import random
import numpy as np
import torch
import os
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_results_file(results_file):
    # Load existing results if file exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    return all_results