import pytest

import t_utils as tu  # noqa: F401

from faenet import FAENet

defaults = {
    "complex_mp": False,
    "cutoff": 5.0,
    "energy_head": None,
    "force_decoder_type": "mlp",
    "force_decoder_model_config": {"mlp": {"hidden_channels": 16}},
    "graph_norm": True,
    "hidden_channels": 64,
    "max_num_neighbors": 40,
    "mp_type": "updownscale_base",
    "num_filters": 128,
    "num_gaussians": 5,
    "num_interactions": 1,
    "pg_hidden_channels": 8,
    "phys_embeds": True,
    "phys_hidden_channels": 0,
    "regress_forces": False,
    "second_layer_MLP": True,
    "skip_co": True,
    "tag_hidden_channels": 8,
    "preprocess": "pbc_preprocess",
}

init_space = {
    "complex_mp": [False, True],
    "energy_head": [
        None,
        "weighted-av-initial-embeds",
        "weighted-av-final-embeds",
    ],
    "force_decoder_type": ["mlp", "simple", "", "res", "res_updown"],
    "graph_norm": [False, True],
    "mp_type": [
        "simple",
        "updownscale",
        "updownscale_base",
        "updown_local_env",
        "base",
    ],
    "phys_embeds": [False, True],
    "phys_hidden_channels": [0, 8],
    "regress_forces": ["direct", "", "direct_with_gradient_target"],
    "second_layer_MLP": [False, True],
    "skip_co": [False, True],
    "tag_hidden_channels": [0, 8],
}


def test_init_no_arg():
    model = FAENet()
    assert model is not None


@pytest.mark.parametrize("kwargs", tu.generate_inits(init_space, defaults))
def test_init(kwargs):
    item, kwargs = kwargs
    if kwargs["regress_forces"] is True:
        with pytest.raises(AssertionError):
            model = FAENet(**kwargs)
    else:
        model = FAENet(**kwargs)
        assert model is not None


@pytest.mark.parametrize("kwargs", tu.generate_inits(init_space, defaults))
def test_forwards(kwargs):
    item, kwargs = kwargs
    if kwargs["regress_forces"] is True:
        return
    else:
        model = FAENet(**kwargs)
        y = model(tu.get_batch())
        assert model is not None
