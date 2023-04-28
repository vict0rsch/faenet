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

# FAENet: Frame Averaging Equivariant GNN for Materials modeling


This repository contains an implementation of the paper *FAENet: Frame Averaging Equivariant GNN for Materials modeling*, accepted at ICML 2023. More precisely, you will find:

* `FrameAveraging`: the transform that projects your pytorch-geometric data into the canonical space defined in the paper.
* `FAENet` GNN model for material modeling. 
* `model_forward`: a high-level forward function that computes appropriate model predictions for the Frame Averaging method, i.e. handling the different frames and mapping to equivariant predictions. 

Also: https://github.com/vict0rsch/faenet

## Installation

```
pip install faenet
```

⚠️ The above installation requires `Python >= 3.8`, [`torch > 1.11`](https://pytorch.org/get-started/locally/), [`torch_geometric > 2.1`](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) to the best of our knowledge. Both `mendeleev` and `pandas` package are also required to derive physics-aware atom embeddings in FAENet.

## Getting started

### Frame Averaging Transform

`FrameAveraging` is a Transform method applicable to pytorch-geometric `Data` object. You can choose among several options ranging from *Full FA* to *Stochastic FA* (in 2D or 3D) including data augmentation *DA*. This method shall be applied in the `get_item()` function of your `Dataset` class. Note that although this transform is specific to pytorch-geometric data objects, it can be easily extended to new settings since the core functions `frame_averaging_2D()` and `frame_averaging_3D()` generalise to other data format. 

```python
import torch
from faenet.transform import FrameAveraging

frame_averaging = "3D"  # symmetry preservation method used: {"3D", "2D", "DA", ""}:
fa_method = "stochastic"  # the frame averaging method: {"det", "all", "se3-stochastic", "se3-det", "se3-all", ""}:
transform = FrameAveraging(frame_averaging, fa_method)
transform(g)  # transform the PyG graph g 
```

### Model forward for Frame Averaging

`model_forward()` aggregates model predictions when Frame Averaging is applied, as stipulated by the Equation (1) of the paper. It must be applied. 

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

Implementation of the FAENet GNN model, compatible with any dataset or transform. In short, FAENet is a very simple, scalable and expressive model. Since does not explicitly preserve data symmetries, it has the ability to process directly and unrestrictedly atom relative positions, which is very efficient. Note that the training procedure is not given here. 

```python
from faenet.model import FAENet

preds = FAENet(**kwargs)
model(batch)
```

![FAENet architecture](examples/data/faenet-archi.png)

### Eval 

The `eval_model_symmetries()` function helps you evaluate the equivariant, invariant and other properties of a model, as we did in the paper. 

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

