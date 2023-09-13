FAENet: Frame Averaging Equivariant GNN for Materials modeling
==============================================================

🌟 This repository contains an implementation of the paper `FAENet: Frame Averaging Equivariant GNN for Materials modeling <https://arxiv.org/pdf/2305.05577.pdf>`_, accepted at *ICML 2023*.🌟 More precisely, you will find:

   * :class:`~faenet.transforms.FrameAveraging`: the transform that projects your pytorch-geometric data into a canonical space of all euclidean transformations, as defined in the paper.
   * :class:`~faenet.model.FAENet`: a GNN architecture for material modeling.
   * :meth:`~faenet.fa_forward.model_forward`: a high-level forward function that computes appropriate equivariant model predictions for the Frame Averaging method, i.e. handling the different frames and mapping to equivariant predictions.

More information are provided in the `User Guide <getting-started/index>`_.

.. toctree::
   :hidden:
   :maxdepth: 4

   About <self>

.. toctree::
   :glob:
   :caption: User Guide
   :maxdepth: 4
   
   install/index
   getting-started/index

.. toctree::
   :glob:
   :caption: Reference
   :maxdepth: 1

   api/model
   api/transforms
   api/frame_averaging
   api/fa_forward
   api/base_model
   api/embedding
   api/force_decoder
   api/eval
   api/utils

Contact
-------  

Alexandre Duval (alexandre.duval@mila.quebec) and Victor Schmidt (schmidtv@mila.quebec). 
We welcome your questions and feedback via email or GitHub Issues.

