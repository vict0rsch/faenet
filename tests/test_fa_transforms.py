import t_utils as tu
from faenet import FrameAveraging
from torch_geometric.data import Batch
import pytest
import torch


@pytest.mark.parametrize("frame_averaging", ["", "2D", "3D", "DA"])
@pytest.mark.parametrize(
    "fa_method",
    ["", "stochastic", "det", "all", "se3-stochastic", "se3-det", "se3-all"],
)
def test_transform(frame_averaging, fa_method):
    batch = tu.get_batch()
    transform = FrameAveraging(frame_averaging, fa_method)
    for i in range(len(batch.sid)):
        g = Batch.get_example(batch, i)
        transform(g)
        if frame_averaging not in {"DA", ""}:
            assert not torch.allclose(g.pos, g.fa_pos[0])
            assert not torch.allclose(g.cell, g.fa_cell[0])
            assert g.fa_rot[0] is not None
