import torch
import torch.nn as nn

import math
import numbers
import random

import torch_geometric
from torch_geometric.transforms import LinearTransformation


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets=False,
    return_distance_vec=False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def swish(x):
    return x * x.sigmoid()

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
    r"""Reflect node positions around a specific axis (x, y, x=y) or the origin
    z-axis remains fixed.
    Info -- type 0: reflect wrt x-axis, type1: wrt y-axis, type2: y=x, type3: origin
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