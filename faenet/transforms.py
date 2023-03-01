import math
import numbers
import random

import torch
import torch_geometric
from torch_geometric.transforms import LinearTransformation

from faenet.frame_averaging import (
    frame_averaging_2D,
    frame_averaging_3D,
    data_augmentation,
)


class Transform:
    def __call__(self, data):
        raise NotImplementedError

    def __str__(self):
        name = self.__class__.__name__
        items = [
            f"{k}={v}"
            for k, v in self.__dict__.items()
            if not callable(v) and k != "inactive"
        ]
        s = f"{name}({', '.join(items)})"
        if self.inactive:
            s = f"[inactive] {s}"
        return s


class FrameAveraging(Transform):
    def __init__(self, fa_type=None, fa_frames=None):
        self.fa_frames = (
            "random" if (fa_frames is None or fa_frames == "") else fa_frames
        )
        self.fa_type = "" if fa_type is None else fa_type
        self.inactive = not self.fa_type
        assert self.fa_type in {
            "",
            "2D",
            "3D",
            "DA",
        }
        assert self.fa_frames in {
            "",  # equivalent to random, necessary still for sweeps
            "random",
            "det",
            "all",
            "multiple",
            "se3-random",
            "se3-det",
            "se3-all",
            "se3-multiple",
        }

        if self.fa_type:
            if self.fa_type == "2D":
                self.fa_func = frame_averaging_2D
            elif self.fa_type == "3D":
                self.fa_func = frame_averaging_3D
            elif self.fa_type == "DA":
                self.fa_func = data_augmentation
            else:
                raise ValueError(f"Unknown frame averaging: {self.fa_type}")

    def __call__(self, data):
        if self.inactive:
            return data
        return self.fa_func(data, self.fa_frames)

class RandomRotate(Transform):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.

    Borrowed from https://github.com/Open-Catalyst-Project/ocp

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


class RandomReflect(Transform):
    r"""Reflect node positions around a specific axis (x, y, x=y) or the origin
    z-axis remains fixed.

    Borrowed from https://github.com/Open-Catalyst-Project/ocp

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
