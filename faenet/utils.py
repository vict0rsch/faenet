import math
import numbers
import random

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.transforms import LinearTransformation
from torch_geometric.nn import radius_graph


def swish(x):
    """Swish activation function"""
    return torch.nn.functional.silu(x)


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets=False,
    return_rel_pos=False,
):
    """Compute distances between atoms with periodic boundary conditions

    Args:
        pos (tensor): (N, 3) tensor of atomic positions
        edge_index (tensor): (2, E) tensor of edge indices
        cell (tensor): (3, 3) tensor of cell vectors
        cell_offsets (tensor): (N, 3) tensor of cell offsets
        neighbors (tensor): (N, 3) tensor of neighbor indices
        return_offsets (bool): return the offsets
        return_rel_pos (bool): return the relative positions vectors

    Returns:
        (dict): dictionary with the updated edge_index, atom distances,
            and optionally the offsets and distance vectors.
    """
    rel_pos = pos[edge_index[0]] - pos[edge_index[1]]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    rel_pos += offsets

    # compute distances
    distances = rel_pos.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_rel_pos:
        out["rel_pos"] = rel_pos[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


def base_preprocess(data, cutoff=6.0, max_num_neighbors=40):
    """Preprocess datapoint: create a cutoff graph,
        compute distances and relative positions.

        Args:
        data (data.Data): data object with specific attributes:
                - batch (N): index of the graph to which each atom belongs to in this batch
                - pos (N,3): atom positions
                - atomic_numbers (N): atomic numbers of each atom in the batch
                - edge_index (2,E): edge indices, for all graphs of the batch
            With B is the batch size, N the number of atoms in the batch (across all graphs),
            and E the number of edges in the batch.
            If these attributes are not present, implement your own preprocess function.
        cutoff (int): cutoff radius (in Angstrom)
        max_num_neighbors (int): maximum number of neighbors per node.

    Returns:
        (tuple): atomic_numbers, batch, sparse adjacency matrix, relative positions, distances
    """
    edge_index = radius_graph(
        data.pos,
        r=cutoff,
        batch=data.batch,
        max_num_neighbors=max_num_neighbors,
    )
    rel_pos = data.pos[edge_index[0]] - data.pos[edge_index[1]]
    return (
        data.atomic_numbers.long(),
        data.batch,
        edge_index,
        rel_pos,
        rel_pos.norm(dim=-1),
    )


def pbc_preprocess(data, cutoff=6.0, max_num_neighbors=40):
    """Preprocess datapoint using periodic boundary conditions
        to improve the existing graph.

    Args:
        data (data.Data): data object with specific attributes. B is the batch size,
            N the number of atoms in the batch (across all graphs), E the number of edges in the batch.
                - batch (N): index of the graph to which each atom belongs to in this batch
                - pos (N,3): atom positions
                - atomic_numbers (N): atomic numbers of each atom in the batch
                - cell (B, 3, 3): unit cell containing each graph, for materials.
                - cell_offsets (E, 3): cell offsets for each edge, for materials
                - neighbors (B): total number of edges inside each graph.
                - edge_index (2,E): edge indices, for all graphs of the batch
            If these attributes are not present, implement your own preprocess function.

    Returns:
        (tuple): atomic_numbers, batch, sparse adjacency matrix, relative positions, distances
    """
    out = get_pbc_distances(
        data.pos,
        data.edge_index,
        data.cell,
        data.cell_offsets,
        data.neighbors,
        return_rel_pos=True,
    )

    return (
        data.atomic_numbers.long(),
        data.batch,
        out["edge_index"],
        out["rel_pos"],
        out["distances"],
    )


class GaussianSmearing(nn.Module):
    r"""Smears a distance distribution by a Gaussian function."""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RandomRotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If `degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axes (int, optional): The rotation axes. (default: `[0, 1, 2]`)
    """

    def __init__(self, degrees, axes=[0, 1, 2]):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axes = axes

    def __call__(self, data):
        if data.pos.size(-1) == 2:
            degree = math.pi * random.uniform(*self.degrees) / 180.0
            sin, cos = math.sin(degree), math.cos(degree)
            matrix = [[cos, sin], [-sin, cos]]
        else:
            m1, m2, m3 = torch.eye(3), torch.eye(3), torch.eye(3)
            if 0 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m1 = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
            if 1 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m2 = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
            if 2 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m3 = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])

            matrix = torch.mm(torch.mm(m1, m2), m3)

        data_rotated = LinearTransformation(matrix)(data)
        if torch_geometric.__version__.startswith("2."):
            matrix = matrix.T

        # LinearTransformation only rotates `.pos`; need to rotate `.cell` too.
        if hasattr(data_rotated, "cell"):
            data_rotated.cell = torch.matmul(
                data_rotated.cell, matrix.to(data_rotated.cell.device)
            )
        # And .force too
        if hasattr(data_rotated, "force"):
            data_rotated.force = torch.matmul(
                data_rotated.force, matrix.to(data_rotated.force.device)
            )

        return (
            data_rotated,
            matrix,
            torch.inverse(matrix),
        )

    def __repr__(self):
        return "{}({}, axis={})".format(
            self.__class__.__name__, self.degrees, self.axis
        )


class RandomReflect(object):
    r"""Reflect node positions around a specific axis (x, y, x=y) or the origin.
    Take a random reflection type from a list of reflection types.
        (type 0: reflect wrt x-axis, type1: wrt y-axis, type2: y=x, type3: origin)
    """

    def __init__(self):
        self.reflection_type = random.choice([0, 1, 2, 3])

    def __call__(self, data):
        if self.reflection_type == 0:
            matrix = torch.FloatTensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif self.reflection_type == 1:
            matrix = torch.FloatTensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        elif self.reflection_type == 2:
            matrix = torch.FloatTensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        elif self.reflection_type == 3:
            matrix = torch.FloatTensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        data_reflected = LinearTransformation(matrix)(data)

        if torch_geometric.__version__.startswith("2."):
            matrix = matrix.T

        # LinearTransformation only rotates `.pos`; need to rotate `.cell` too.
        if hasattr(data_reflected, "cell"):
            data_reflected.cell = torch.matmul(
                data_reflected.cell, matrix.to(data_reflected.cell.device)
            )
        # And .force too
        if hasattr(data_reflected, "force"):
            data_reflected.force = torch.matmul(
                data_reflected.force, matrix.to(data_reflected.force.device)
            )

        return (
            data_reflected,
            matrix,
            torch.inverse(matrix),
        )
