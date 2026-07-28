import random
import numpy as np
import torch
import os
from datetime import datetime


def seed_everything(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def experiment_name_with_timestamp(experiment_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return f"{timestamp}_{experiment_name}"


def create_experiment_dir(output_dir: str, experiment_name: str, fold_id: int):
    experiment_path = os.path.join(output_dir, experiment_name, f"fold_{fold_id}")
    os.makedirs(experiment_path, exist_ok=True)
    return experiment_path


def average_dict_values(d: dict) -> float:
    if not d:  # handle empty dict
        return 0
    x = []
    for _, val in d.items():
        x.append(val)
    x = np.mean(x, axis=0).tolist()
    return x
