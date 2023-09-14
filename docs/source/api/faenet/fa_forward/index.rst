:py:mod:`faenet.fa_forward`
===========================

.. py:module:: faenet.fa_forward


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   faenet.fa_forward.model_forward



.. py:function:: model_forward(batch, model, frame_averaging, mode='train', crystal_task=True)

   Perform a model forward pass when frame averaging is applied.

   :param batch: batch of graphs with attributes:
                 - original atom positions (`pos`)
                 - batch indices (to which graph in batch each atom belongs to) (`batch`)
                 - frame averaged positions, cell and rotation matrices (`fa_pos`, `fa_cell`, `fa_rot`)
   :type batch: data.Batch
   :param model: model instance
   :param frame_averaging: symmetry preserving method (already) applied
                           ("2D", "3D", "DA", "")
   :type frame_averaging: str
   :param mode: model mode. Defaults to "train".
                ("train", "eval")
   :type mode: str, optional
   :param crystal_task: Whether crystals (molecules) are considered.
                        If they are, the unit cell (3x3) is affected by frame averaged and expected as attribute.
                        (default: :obj:`True`)
   :type crystal_task: bool, optional

   :returns: model predictions tensor for "energy" and "forces".
   :rtype: (dict)


