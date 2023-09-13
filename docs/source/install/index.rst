Installation
------------

.. code-block:: bash

   pip install faenet


⚠️ The above installation requires: 

* Python >= 3.8
* `torch > 1.11 <https://pytorch.org/get-started/locally/>`_
* `torch_geometric > 2.1 <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#>`_ (to the best of our knowledge).

Indeed, because of CUDA/torch compatibilities, neither ``torch`` nor ``torch_geometric`` are part of the explicit dependencies and must be installed independently.
The rest of the dependencies, such as the ``mendeleev`` package to derive physics-aware atom embeddings, are installed automatically with faenet. 