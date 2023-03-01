import sys
import os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "examples/data/is2re_bs3.pt"
BATCH = None

sys.path.insert(0, str(ROOT))
os.environ["SQLALCHEMY_SILENCE_UBER_WARNING"] = "1"


def generate_inits(init_space, defaults):
    inits = []
    for key, values in init_space.items():
        for value in values:
            init = defaults.copy()
            init[key] = value
            inits.append(({key: value}, init))
    return inits


def get_batch():
    global BATCH
    if BATCH is None:
        BATCH = torch.load(DATA_PATH)
    return BATCH
