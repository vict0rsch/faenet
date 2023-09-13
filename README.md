<p align="center">
<strong><a href="https://github.com/vict0rsch/faenet" target="_blank">💻&nbsp;&nbsp;Code</a></strong>
<strong>&nbsp;&nbsp;•&nbsp;&nbsp;</strong>
<strong><a href="https://faenet.readthedocs.io/" target="_blank">Docs&nbsp;&nbsp;📑</a></strong>
</p>

<p align="center">
    <a>
	    <img src='https://img.shields.io/badge/python-3.8%2B-blue' alt='Python' />
	</a>
	<a href='https://faenet.readthedocs.io/en/latest/?badge=latest'>
    	<img src='https://readthedocs.org/projects/faenet/badge/?version=latest' alt='Documentation Status' />
	</a>
    <a href="https://github.com/psf/black">
	    <img src='https://img.shields.io/badge/code%20style-black-black' />
	</a>
<a href="https://pytorch.org">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white"/>
</a>
</p>
<br/>

# FAENet: Frame Averaging Equivariant GNN for materials modeling


🌟 This repository contains an implementation of the paper [*FAENet: Frame Averaging Equivariant GNN for Materials Modeling*](https://arxiv.org/pdf/2305.05577.pdf), accepted at ICML 2023.🌟 More precisely, you will find:

* `FrameAveraging`: the transform that projects your pytorch-geometric data into a canonical space of all euclidean transformations, as defined in the paper.
* `FAENet`: a GNN architecture for material modeling.
* `model_forward`: a high-level forward function that computes appropriate equivariant model predictions for the Frame Averaging method, i.e. handling the different frames and mapping to equivariant predictions.

## Installation

```
pip install faenet
```

⚠️ The above installation requires `Python >= 3.8`, [`torch > 1.11`](https://pytorch.org/get-started/locally/), [`torch_geometric > 2.1`](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) to the best of our knowledge. Both `mendeleev` and `pandas` package are also required to derive physics-aware atom embeddings in FAENet.

## Getting started

### Frame Averaging Transform

`FrameAveraging` is a Transform method applicable to pytorch-geometric `Data` object, which shall be used in the `get_item()` function of your `Dataset` class. This method derives a new canonical position for the atomic graph, identical for all euclidean symmetries, and stores it under the data attribute `fa_pos`. You can choose among several options for the frame averaging, ranging from *Full FA* to *Stochastic FA* (in 2D or 3D) including traditional data augmentation *DA* with rotated samples. See the full [doc](https://faenet.readthedocs.io/en/latest/autoapi/faenet/transforms/index.html#faenet.transforms.FrameAveraging) for more details. Note that, although this transform is specific to pytorch-geometric data objects, it can be easily extended to new settings since the core functions `frame_averaging_2D()` and `frame_averaging_3D()` generalise to other data format.

```python
import torch
from faenet.transforms import FrameAveraging

frame_averaging = "3D"  # symmetry preservation method used: {"3D", "2D", "DA", ""}:
fa_method = "stochastic"  # the frame averaging method: {"det", "all", "se3-stochastic", "se3-det", "se3-all", ""}:
transform = FrameAveraging(frame_averaging, fa_method)
transform(data)  # transform the PyG graph data
```

### Model forward for Frame Averaging

`model_forward()` aggregates the predictions of a chosen ML model (e.g FAENet) when Frame Averaging is applied, as stipulated by the Equation (1) of the paper. INded, applying the model on canonical positions (`fa_pos`) directly would not yield equivariant predictions. This method must be applied at training and inference time to compute all model predictions. It requires `batch` to have pos, batch and frame averaging attributes (see [docu](https://faenet.readthedocs.io/en/latest/autoapi/faenet/fa_forward/index.html)).

```python
from faenet.fa_forward import model_forward

preds = model_forward(
    batch=batch,   # batch from, dataloader
    model=model,  # FAENet(**kwargs)
    frame_averaging="3D", # ["2D", "3D", "DA", ""]
    mode="train",  # for training
    crystal_task=True,  # for crystals, with pbc conditions
)
```

### FAENet GNN

Implementation of the FAENet GNN model, compatible with any dataset or transform. In short, FAENet is a very simple, scalable and expressive model. Since does not explicitly preserve data symmetries, it has the ability to process directly and unrestrictedly atom relative positions, which is very efficient and powerful. Although it was specifically designed to be applied with Frame Averaging above, to preserve symmetries without any design restrictions, note that it can also be applied without. When applied with Frame Averaging, we need to use the `model_forward()` function above to compute model predictions, `model(data)` is not enough. Note that the training procedure is not given here, you should refer to the original [github repository](https://github.com/RolnickLab/ocp). Check the [documentation](https://faenet.readthedocs.io/en/latest/autoapi/faenet/model/index.html) to see all input parameters.

Note that the model assumes input data (e.g.`batch` below) to have certain attributes, like atomic_numbers, batch, pos or edge_index. If your data does not have these attributes, you can apply custom pre-processing functions, taking `pbc_preprocess` or `base_preprocess` in [utils.py](https://faenet.readthedocs.io/en/latest/autoapi/faenet/utils/index.html) as inspiration. You simply need to pass them as argument to FAENet (`preprocess`).

```python
from faenet.model import FAENet

preds = FAENet(**kwargs)
model(batch)
```

![FAENet architecture](https://raw.githubusercontent.com/vict0rsch/faenet/main/examples/data/faenet-archi.png)

### Eval

The `eval_model_symmetries()` function helps you evaluate the equivariant, invariant and other properties of a model, as we did in the paper.

Note: you can predict any atom-level or graph-level property, although the code explicitly refers to energy and forces.

### Tests

The `/tests` folder contains several useful unit-tests. Feel free to have a look at them to explore how the model can be used. For more advanced examples, please refer to the full [repository](https://github.com/RolnickLab/ocp) used in our ICML paper to make predictions on OC20 IS2RE, S2EF, QM9 and QM7-X dataset.

This requires [`poetry`](https://python-poetry.org/docs/). Make sure to have `torch` and `torch_geometric` installed in your environment before you can run the tests. Unfortunately because of CUDA/torch compatibilities, neither `torch` nor `torch_geometric` are part of the explicit dependencies and must be installed independently.

```bash
git clone git@github.com:vict0rsch/faenet.git
poetry install --with dev
pytest --cov=faenet --cov-report term-missing
```

Testing on Macs you may encounter a [Library Not Loaded Error](https://github.com/pyg-team/pytorch_geometric/issues/6530)

### Contact

Authors: Alexandre Duval (alexandre.duval@mila.quebec) and Victor Schmidt (schmidtv@mila.quebec). We welcome your questions and feedback via email or GitHub Issues.

