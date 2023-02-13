import pytest

import t_utils as tu  # noqa: F401

from faenet import FAENet

defaults = {
    "att_heads": 0,
    "complex_mp": False,
    "cutoff": 5.0,
    "edge_embed_hidden": 64,
    "edge_embed_type": "all_rij",
    "energy_head": None,
    "force_decoder_type": "mlp",
    "force_decoder_model_config": {"hidden_channels": 16},
    "graph_norm": True,
    "graph_rewiring": None,
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
    "use_pbc": True,
}

init_space = {
    "att_heads": [0, 1],
    "complex_mp": [False, True],
    "edge_embed_type": ["rij", "all_rij", "sh", "all"],
    "energy_head": [
        None,
        "graclus",
        "pooling",
        "random",
        "weighted-av-initial-embeds",
        "weighted-av-final-embeds",
    ],
    "force_decoder_type": ["mlp", "simple", ""],
    "graph_norm": [False, True],
    "graph_rewiring": [
        None,
        "remove-tag-0",
        "one-supernode-per-graph",
        "one-supernode-per-atom-type",
        "one-supernode-per-atom-type-dist",
    ],
    "mp_type": [
        "simple",
        "sfarinet",
        "updownscale",
        "updownscale_base",
        "base_with_att",
        "att",
        "local_env",
        "updown_local_env",
        "base",
    ],
    "phys_embeds": [False, True],
    "phys_hidden_channels": [0, 8],
    "regress_forces": [False, True, "direct", "", "direct_with_gradient_target"],
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
