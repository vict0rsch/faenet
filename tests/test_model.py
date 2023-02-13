import t_utils as tu  # noqa: F401

from faenet import FAENet


def test_init_no_arg():
    model = FAENet()
    assert model is not None
