import pytest
import t_utils as tu
from torch_geometric.data import Batch
from faenet import FrameAveraging
from faenet.eval import eval_model_symmetries
from faenet import FAENet
from faenet.utils import pbc_preprocess


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
    "energy_head": [
        None,
        "weighted-av-final-embeds",
    ],
    "force_decoder_type": ["mlp", "res_updown"],
    "regress_forces": ["", "direct"],
    "skip_co": [False, "concat"],
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
@pytest.mark.parametrize("fa_method", ["stochastic", "all", "se3-stochastic", ""])
@pytest.mark.parametrize("frame_averaging", ["", "2D", "3D", "DA"])
def test_symmetries(kwargs, fa_method, frame_averaging):
    item, kwargs = kwargs
    batch = tu.get_batch()
    model = FAENet(**kwargs)
    transform = FrameAveraging(frame_averaging, fa_method)
    neighbors = batch.neighbors

    # Transform batch manually here instead of using Dataloader
    b = Batch.to_data_list(batch)
    for g in b:
        transform(g)
    b = Batch.from_data_list(b)
    if not hasattr(b, "neighbors"):
        b.neighbors = neighbors

    # Eval symmetries
    metrics = eval_model_symmetries(
        loader=[b],
        model=model,
        frame_averaging=frame_averaging,
        fa_method=fa_method,
        device=b.pos.device,
        task_name="energy",
        crystal_task=True,
    )
    assert metrics
