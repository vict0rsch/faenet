import pytest
import t_utils as tu
from faenet import frame_averaging_3D, frame_averaging_2D
from torch_geometric.data import Batch
import torch
from copy import deepcopy
import math
import random



def test_frame_averaging_3D():
    batch = tu.get_batch()
    for i in range(len(batch.sid)):
        for method in ["all", "stochastic", "det", "se3-all", "se3-stochastic", "se3-det"]:
            g = Batch.get_example(batch, i)
            pos, cell, rot = frame_averaging_3D(g.pos, g.cell, fa_method=method)
            assert (pos[0] != g.pos).any().item() # Check that positions have changed
            assert (cell[0] != g.cell).any().item() # Check that cell has changed

            # Check correct dimensions are returned for all FA methods
            if method == "all":
                assert len(pos) == len(rot) == len(cell) == 8
            elif method == "se3-all":
                assert len(pos) == len(rot) == len(cell) == 4
            else: 
                assert len(pos) == len(cell) == len(rot) == 1
            assert pos[0].shape == (g.num_nodes, 3)
            assert cell[0].shape == (1, 3, 3)
            assert rot[0].shape == (1, 3, 3)
        break


def test_frame_averaging_2D():
    batch = tu.get_batch()
    for i in range(len(batch.sid)):
        for method in ["all", "stochastic", "det", "se3-all", "se3-stochastic", "se3-det"]:
            g = Batch.get_example(batch, i)
            pos, cell, rot = frame_averaging_2D(g.pos, g.cell, fa_method=method)
            assert (pos[0] != g.pos).any().item() # Check that positions have changed
            assert (cell[0] != g.cell).any().item() # Check that cell has changed

            # Check correct dimensions are returned for all FA methods
            if method == "all":
                assert len(pos) == len(rot) == len(cell) == 4
            elif method == "se3-all":
                assert len(pos) == len(rot) == len(cell) == 2
            else: 
                assert len(pos) == len(cell) == len(rot) == 1
            assert pos[0].shape == (g.num_nodes, 3)
            assert cell[0].shape == (1, 3, 3)
            assert rot[0].shape == (1, 3, 3)
        break


def test_rotation_invariance():
    """Check that frame averaging is rotation invariant."""
    batch = tu.get_batch()
    for i in range(len(batch.sid)):
        g = Batch.get_example(batch, i)
            
        # Rotate graph
        rotated_g = deepcopy(g)
        degrees = (0, 180)
        degree = math.pi * random.uniform(*degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        R = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        rotated_g.pos = (g.pos @ R.T).to(g.pos.device, g.pos.dtype)

        # Frame averaging on both graphs
        fa_pos, fa_cell, _ = frame_averaging_3D(g.pos, g.cell, fa_method="all")
        rotated_fa_pos, rotated_fa_cell, _ = frame_averaging_3D(rotated_g.pos, rotated_g.cell, fa_method="all")

        # Check that the set of frames is the same for both graphs 
        pos_diff = 0
        for pos1, pos2 in zip(fa_pos, rotated_fa_pos):
            pos_diff += pos1 - pos2
        assert torch.abs(pos_diff).sum() < 1e-2

        # Check that the set of FA cells is the same for both graphs 
        cell_diff = 0
        for cell1, cell2 in zip(fa_cell, rotated_fa_cell):
            cell_diff += cell1 - cell2
        assert torch.abs(cell_diff).sum() < 1e-2



# Check constraints 
# Check that rotated / reflected graph have the same frame
# Test that pbc distances are preserved