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
    r""" Frame Averaging Transform for PyG Data objects

    Args:
        frame_averaging (str): Transform method used.
            (`""`, `"2D"`, `"3D"`, `"DA"`)
        fa_method (str): FA method used.
            (`""`, "stochastic", "all", "det", "se3-stochastic", "se3-all", "se3-det")
    """
    def __init__(self, frame_averaging=None, fa_method=None):
        self.fa_method = (
            "stochastic" if (fa_method is None or fa_method == "") else fa_method
        )
        self.frame_averaging = "" if frame_averaging is None else frame_averaging
        self.inactive = not self.frame_averaging
        assert self.frame_averaging in {
            "",
            "2D",
            "3D",
            "DA",
        }
        assert self.fa_method in {
            "",
            "stochastic",
            "det",
            "all",
            "se3-stochastic",
            "se3-det",
            "se3-all",
        }

        if self.frame_averaging:
            if self.frame_averaging == "2D":
                self.fa_func = frame_averaging_2D
            elif self.frame_averaging == "3D":
                self.fa_func = frame_averaging_3D
            elif self.frame_averaging == "DA":
                self.fa_func = data_augmentation
            else:
                raise ValueError(f"Unknown frame averaging: {self.frame_averaging}")

    def __call__(self, data):
        if self.inactive:
            return data
        elif self.frame_averaging == "DA":
            return self.fa_func(data, self.fa_method)
        else:
            data.fa_pos, data.fa_cell, data.fa_rot = self.fa_func(data.pos, data.cell, self.fa_method)
            return data