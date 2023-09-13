Getting started
===============

In this section, we will show you how to (1) use the Frame Averaging transform (2) use FAENet to make predictions on your own dataset (3) evaluate model properties and test the code.

.. contents:: Table of Contents
    :depth: 1
    :local:

Frame Averaging
---------------

:class:`~faenet.transforms.FrameAveraging` is a Transform method applicable to pytorch-geometric ``Data`` object, which shall be used in the ``get_item()`` function of your ``Dataset`` class. This method derives a new canonical position for the atomic graph, identical for all euclidean symmetries, and stores it under the data attribute ``fa_pos``. You can choose among several options for the frame averaging, ranging from *Full FA* to *Stochastic FA* (in 2D or 3D) including traditional data augmentation *DA* with rotated samples. Note that, although this transform is specific to pytorch-geometric data objects, it can be easily extended to new settings since the core functions :meth:`~faenet.frame_averaging.frame_averaging_2D` and :meth:`~faenet.frame_averaging.frame_averaging_3D` generalise to other data format.

.. code-block:: python

    import torch
    from faenet.transforms import FrameAveraging

    frame_averaging = "3D"  # symmetry preservation method used: {"3D", "2D", "DA", ""}:
    fa_method = "stochastic"  # the frame averaging method: {"det", "all", "se3-stochastic", "se3-det", "se3-all", ""}:
    transform = FrameAveraging(frame_averaging, fa_method)
    transform(data)  # transform the PyG graph data


Model Forward Pass with Frame Averaging
---------------------------------------   

:meth:`~faenet.fa_forward.model_forward` aggregates the predictions of a chosen ML model (e.g FAENet) when Frame Averaging is applied, as stipulated by the Equation (1) of the paper. INded, applying the model on canonical positions (``fa_pos``) directly would not yield equivariant predictions. This method must be applied at training and inference time to compute all model predictions. It requires ``batch`` to have pos, batch and frame averaging attributes.

.. code-block:: python

    from faenet.fa_forward import model_forward

    preds = model_forward(
        batch=batch,   # batch from, dataloader
        model=model,  # FAENet(**kwargs)
        frame_averaging="3D", # ["2D", "3D", "DA", ""]
        mode="train",  # for training
        crystal_task=True,  # for crystals, with pbc conditions
    )

FAENet
------

Implementation of the :class:`~faenet.model.FAENet` GNN model, compatible with any dataset or transform. In short, FAENet is a very simple, scalable and expressive model. Since does not explicitly preserve data symmetries, it has the ability to process directly and unrestrictedly atom relative positions, which is very efficient and powerful. Although it was specifically designed to be applied with Frame Averaging above, to preserve symmetries without any design restrictions, note that it can also be applied without. When applied with Frame Averaging, we need to use the :meth:`~faenet.fa_forward.model_forward` function above to compute model predictions, ``model(data)`` is not enough. Note that the training procedure is not given here, you should refer to the `original github repository <https://github.com/RolnickLab/ocp>`_. 
Check :class:`~faenet.model.FAENet` to see all input parameters.

**Assumption**: the input data (e.g. ``batch`` below) has certain attributes (e.g. atomic_numbers, batch, pos or edge_index). If your data does not have these attributes, you can apply custom pre-processing functions, taking :meth:`~faenet.utils.pbc_preprocess` or :meth:`~faenet.utils.base_preprocess` as inspiration. You simply need to pass them as argument to FAENet (``preprocess``).

.. code-block:: python

    from faenet.model import FAENet

    model = FAENet(**kwargs)
    model(batch)  # forward pass

.. image:: ../../../examples/data/faenet-archi.png

Evaluation
----------

:meth:`~faenet.eval.eval_model_symmetries` helps you evaluate the equivariant, invariant and other properties of a model, as we did in the paper.

Note: you can predict any atom-level or graph-level property, although the code explicitly refers to energy and forces.

Tests
-----

The ``/tests`` folder contains several useful unit-tests. Feel free to have a look at them to explore how the model can be used. For more advanced examples, please refer to the full `repository <https://github.com/RolnickLab/ocp>`_ used in our ICML paper to make predictions on OC20 IS2RE, S2EF, QM9 and QM7-X dataset.

This requires `poetry <https://python-poetry.org/docs/>`_. Make sure to have ``torch`` and ``torch_geometric`` installed in your environment before you can run the tests. Unfortunately because of CUDA/torch compatibilities, neither ``torch`` nor ``torch_geometric`` are part of the explicit dependencies and must be installed independently.

.. code-block:: bash
    
    git clone git@github.com:vict0rsch/faenet.git
    poetry install --with dev
    pytest --cov=faenet --cov-report term-missing

Testing on Macs you may encounter a `Library Not Loaded Error <https://github.com/pyg-team/pytorch_geometric/issues/6530>`_