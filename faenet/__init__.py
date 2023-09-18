"""
FAENet: Frame Averaging Equivariant Network

.. note::

    documentation in progress

Refer to :py:mod:`faenet.model` for the model definition or :class:`~faenet.model.FAENet` for the class definition.

Bye bye
"""
from .model import FAENet  # noqa: F401
from .frame_averaging import frame_averaging_2D, frame_averaging_3D
from .transforms import FrameAveraging
from .fa_forward import model_forward

import importlib.metadata

__version__ = importlib.metadata.version("faenet")
