import t_utils as tu
from faenet import FrameAveraging
from torch_geometric.data import Batch
import pytest

@pytest.mark.parametrize("fa_type", ["2D", "3D", "DA"])
@pytest.mark.parametrize("fa_frames", ["stochastic", "det", "all", "se3-stochastic", "se3-det", "se3-all"])
def test_transform(fa_type, fa_frames):
    batch = tu.get_batch()
    transform = FrameAveraging(fa_type, fa_frames)
    for i in range(len(batch.sid)):
        g = Batch.get_example(batch, i)
        transform(g)