import t_utils as tu
from faenet.transforms import FrameAveraging
from torch_geometric.data import Batch
import pytest

@pytest.mark.parametrize("fa_type", ["2D", "3D"])
@pytest.mark.parametrize("fa_frames", ["stochastic", "det", "all", "se3-stochastic", "se3-det", "se3-all"])
def test_transform(fa_type, fa_frames):
    batch = tu.get_batch()
    transform = FrameAveraging(fa_type, fa_frames)
    for i in range(len(batch.sid)):
        g = Batch.get_example(batch, i)
        transform(g)

# PhAST 
# rewired_data = remove_tag0_nodes(deepcopy(data))
# print(
#     "Data without tag-0 nodes contains {} graphs, a total of {} atoms and {} edges".format(
#         len(rewired_data.natoms), rewired_data.ptr[-1], len(rewired_data.cell_offsets)
#     )
# Use Compose Transforms