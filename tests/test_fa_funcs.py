import pytest
import t_utils as tu
from faenet import frame_averaging_3D, frame_averaging_2D
from faenet.utils import get_pbc_distances
from torch_geometric.data import Batch
import torch
from copy import deepcopy
import math
import random

fa_method_list = ["all", "stochastic", "det", "se3-all", "se3-stochastic", "se3-det"]


@pytest.mark.parametrize("fa_method", fa_method_list)
def test_frame_averaging_3D(fa_method):
    batch = tu.get_batch()
    for i in range(len(batch.sid)):
        g = Batch.get_example(batch, i)
        pos, cell, rot = frame_averaging_3D(g.pos, g.cell, fa_method=fa_method)
        assert (pos[0] != g.pos).any().item()  # Check that positions have changed
        assert (cell[0] != g.cell).any().item()  # Check that cell has changed

        # Check correct dimensions are returned for all FA methods
        if fa_method == "all":
            assert len(pos) == len(rot) == len(cell) == 8
        elif fa_method == "se3-all":
            assert len(pos) == len(rot) == len(cell) == 4
        else:
            assert len(pos) == len(cell) == len(rot) == 1
        assert pos[0].shape == (g.num_nodes, 3)
        assert cell[0].shape == (1, 3, 3)
        assert rot[0].shape == (1, 3, 3)


@pytest.mark.parametrize("fa_method", fa_method_list)
def test_frame_averaging_2D(fa_method):
    batch = tu.get_batch()
    for i in range(len(batch.sid)):
        g = Batch.get_example(batch, i)
        pos, cell, rot = frame_averaging_2D(g.pos, g.cell, fa_method=fa_method)
        assert (pos[0] != g.pos).any().item()  # Check that positions have changed
        assert (cell[0] != g.cell).any().item()  # Check that cell has changed

        # Check correct dimensions are returned for all FA methods
        if fa_method == "all":
            assert len(pos) == len(rot) == len(cell) == 4
        elif fa_method == "se3-all":
            assert len(pos) == len(rot) == len(cell) == 2
        else:
            assert len(pos) == len(cell) == len(rot) == 1
        assert pos[0].shape == (g.num_nodes, 3)
        assert cell[0].shape == (1, 3, 3)
        assert rot[0].shape == (1, 3, 3)


def test_check_constraints():
    """Check that the FA requirements are satisfied"""
    batch = tu.get_batch()
    for i in range(len(batch.sid)):
        g = Batch.get_example(batch, i)
        frame_averaging_3D(g.pos, g.cell, fa_method="stochastic", check=True)


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
        rotated_fa_pos, rotated_fa_cell, _ = frame_averaging_3D(
            rotated_g.pos, rotated_g.cell, fa_method="all"
        )

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


def test_distance_preservation():
    """Check that distances are preserved with PBC when applying rotations on the graph."""
    batch = tu.get_batch()
    for i in range(len(batch.sid)):
        g = Batch.get_example(batch, i)
        out = get_pbc_distances(
            g.pos,
            g.edge_index,
            g.cell,
            g.cell_offsets,
            batch.neighbors[i],
            return_rel_pos=True,
        )
        init_distances = out["distances"]

        # Repeat for projected graph
        g.fa_pos, g.fa_cell, _ = frame_averaging_3D(g.pos, g.cell, fa_method="all")
        fa_out = get_pbc_distances(
            g.fa_pos[0],
            g.edge_index,
            g.fa_cell[0],
            g.cell_offsets,
            batch.neighbors[i],
            return_rel_pos=True,
        )
        fa_distances = fa_out["distances"]

        assert torch.allclose(init_distances, fa_distances, atol=1e-03)
