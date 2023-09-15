FAENet: Frame Averaging Equivariant GNN for Materials modeling
==============================================================

🌟 This repository contains an implementation of the paper `FAENet: Frame Averaging Equivariant GNN for Materials modeling <https://arxiv.org/pdf/2305.05577.pdf>`_, accepted at *ICML 2023*.🌟 More precisely, you will find:

   * :class:`~faenet.transforms.FrameAveraging`: the data transform that projects your 3D graph into a canonical space of all euclidean transformations, as defined in the paper.
   * :class:`~faenet.model.FAENet`: a GNN architecture for property prediction on 3D atomic systems.
   * :meth:`~faenet.fa_forward.model_forward`: a high-level forward function that computes appropriate equivariant model predictions for the Frame Averaging method, i.e. handling the different frames and mapping to equivariant predictions.

More information are provided in the `User Guide <getting-started/index>`_.

**See the source code for the package on `Github <https://github.com/vict0rsch/faenet>`_**

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
   :caption: Reference
   :maxdepth: 1

   /api/faenet/index
   /api/faenet/base_model/index
   /api/faenet/embedding/index
   /api/faenet/eval/index
   /api/faenet/fa_forward/index
   /api/faenet/force_decoder/index
   /api/faenet/frame_averaging/index
   /api/faenet/model/index
   /api/faenet/transforms/index
   /api/faenet/utils/index

Contact
-------

Alexandre Duval (alexandre.duval@mila.quebec) and Victor Schmidt (schmidtv@mila.quebec).
We welcome your questions and feedback via email or GitHub Issues.

