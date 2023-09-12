from .model import FAENet  # noqa: F401
from .frame_averaging import frame_averaging_2D, frame_averaging_3D
from .transforms import FrameAveraging
from .fa_forward import model_forward


import importlib.metadata

__version__ = importlib.metadata.version("faenet")
