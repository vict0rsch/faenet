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


This repository contains an implementation of the paper ``FAENet: Frame Averaging Equivariant GNN for Materials modeling'', submitted to ICML 2023. More precisely, you will find:

* `FrameAveraging` Transform maps your pytorch-geometric data to the canonical space defined in the paper, free of symmetry-preserving constraints.  
* `FAENet` contains the GNN architecture. 
* `FA-forward-function` computes model predictions for the frame averaging extension, handling the different frames and mapping to equivariant predictions. 

Also: https://github.com/vict0rsch/faenet

## Installation

```
pip install faenet
```

⚠️ The above installation requires `Python >= 3.8`, [`torch > 1.11`](https://pytorch.org/get-started/locally/), [`torch_geometric > 2.1`](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) to the best of our knowledge. 

## Getting started

### Frame Averaging Transform

FrameAveraging acts as a transform method, applicable on each graph. It shall be applied in the Dataloader class of your pipeline, in the get_item() function. Note that this transform is specific to pytorch-geometric data object but it can be easily extended to new settings as the inner function frame_averaging_2D and frame_averaging_3D generalise to all data format types. 

```python
import torch
from faenet import FrameAveraging

frame_averaging = "3D"  # "2D", "DA", ""
fa_method = "stochastic"  # "det", "all", "se3-stochastic", "se3-det", "se3-all", ""
transform = FrameAveraging(frame_averaging, fa_method)
transform(g)  # transform the PyG graph g 
```

### Frame Averaging forward

```python
from faenet import model_forward

preds = model_forward(
    batch=batch,   # batch from, dataloader
    model=model,  # FAENet(**kwargs)
    frame_averaging="3D", # ["2D", "3D", "DA", ""]
    mode="train",  # for training 
    crystal_task=True,  # for crystals, with pbc conditions
)
```

### FAENet GNN 

You apply FAENet the FAENet model, which works with any dataset and any transform (None, Frame Averaging, Data augmentation, etc.). Here is a figure of the architecture defined in the paper. Note that this model does not explicitly preserve symmetries, which allows it to have a very scalable, easy-to-understand and expressive functioning. This is due to its ability to process directly and unrestrictedly atom relative positions. 
Add image pipeline. 

```python
from faenet import FAENet

preds = FAENet(**kwargs)
model(batch)
```

## Tests

This requires [`poetry`](https://python-poetry.org/docs/). Make sure to have `torch` and `torch_geometric` installed in your environment before you can run the tests. Unfortunately because of CUDA/torch compatibilities, neither `torch` nor `torch_geometric` are part of the explicit dependencies and must be installed independently.

Many tests are included and can be useful to explore in more details how the model can be used. For more real-use cases, you can check the full repository used to make predictions on OC20 IS2RE, S2EF, QM9 and QM7-X dataset: LINK. 

```bash
git clone git@github.com:vict0rsch/faenet.git
poetry install --with dev
pytest --cov=faenet --cov-report term-missing
```

Testing on Macs you may encounter a [Library Not Loaded Error](https://github.com/pyg-team/pytorch_geometric/issues/6530)