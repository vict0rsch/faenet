:py:mod:`faenet.eval`
=====================

.. py:module:: faenet.eval


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   faenet.eval.eval_model_symmetries
   faenet.eval.reflect_graph
   faenet.eval.rotate_graph
   faenet.eval.transform_batch



.. py:function:: eval_model_symmetries(loader, model, frame_averaging, fa_method, device, task_name, crystal_task=True)

   Test rotation and reflection invariance & equivariance of GNNs

   :param loader: dataloader
   :type loader: data
   :param model: model instance
   :param frame_averaging: frame averaging ("2D", "3D"), data augmentation ("DA")
                           or none ("")
   :type frame_averaging: str
   :param fa_method: _description_
   :type fa_method: str
   :param task_name: the targeted task
                     ("energy", "forces")
   :type task_name: str
   :param crystal_task: whether we have a crystal (i.e. a unit cell)
                        or a molecule
   :type crystal_task: bool

   :returns:

             metrics measuring invariance/equivariance
                 of energy/force predictions
   :rtype: (dict)


.. py:function:: reflect_graph(batch, frame_averaging, fa_method, reflection=None)

   Rotate all graphs in a batch

   :param batch: batch of graphs
   :type batch: data.Batch
   :param frame_averaging: Transform method used
                           ("2D", "3D", "DA", "")
   :type frame_averaging: str
   :param fa_method: FA method used
                     ("", "stochastic", "all", "det", "se3-stochastic", "se3-all", "se3-det")
   :type fa_method: str
   :param reflection: type of reflection applied. (default: :obj:`None`)
   :type reflection: str, optional

   :returns: reflected batch sample and rotation matrix used to reflect it
   :rtype: (dict)


.. py:function:: rotate_graph(batch, frame_averaging, fa_method, rotation=None)

   Rotate all graphs in a batch

   :param batch: batch of graphs.
   :type batch: data.Batch
   :param frame_averaging: Transform method used.
                           ("2D", "3D", "DA", "")
   :type frame_averaging: str
   :param fa_method: FA method used.
                     ("", "stochastic", "all", "det", "se3-stochastic", "se3-all", "se3-det")
   :type fa_method: str
   :param rotation: type of rotation applied. (default: :obj:`None`)
                    ("z", "x", "y", None)
   :type rotation: str, optional

   :returns: rotated batch sample and rotation matrix used to rotate it
   :rtype: (dict)


.. py:function:: transform_batch(batch, frame_averaging, fa_method, neighbors=None)

   Apply a transformation to a batch of graphs

   :param batch: batch of data.Data objects.
   :type batch: data.Batch
   :param frame_averaging: Transform method used.
   :type frame_averaging: str
   :param fa_method: FA method used.
   :type fa_method: str
   :param neighbors: list containing the number of edges
                     in each graph of the batch. (default: :obj:`None`)
   :type neighbors: list, optional

   :returns: transformed batch sample
   :rtype: (data.Batch)


