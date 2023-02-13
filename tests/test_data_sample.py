import torch
import t_utils as tu
import torch_geometric


def test_load_data():
    torch.load(tu.DATA_PATH)


def test_is_batch():
    assert isinstance(torch.load(tu.DATA_PATH), torch_geometric.data.Batch)


def test_get_batch():
    assert isinstance(tu.get_batch(), torch_geometric.data.Batch)
