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
            "stochastic" if (fa_frames is None or fa_frames == "") else fa_frames
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
            "",
            "stochastic",
            "det",
            "all",
            "se3-stochastic",
            "se3-det",
            "se3-all",
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
        elif self.fa_type == "DA":
            return self.fa_func(data, self.fa_frames)
        else:
            data.fa_pos, data.fa_cell, data.fa_rot = self.fa_func(data.pos, data.cell, self.fa_frames)
            return data