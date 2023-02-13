import torch
from t_utils import ROOT
import torch_geometric

DATA_PATH = ROOT / "examples/data/is2re_bs8.pt"


def test_load_data():
    torch.load(DATA_PATH)


def test_is_batch():
    assert isinstance(torch.load(DATA_PATH), torch_geometric.data.Batch)
