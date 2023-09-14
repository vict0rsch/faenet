:py:mod:`faenet.frame_averaging`
================================

.. py:module:: faenet.frame_averaging


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   faenet.frame_averaging.check_constraints
   faenet.frame_averaging.compute_frames
   faenet.frame_averaging.data_augmentation
   faenet.frame_averaging.frame_averaging_2D
   faenet.frame_averaging.frame_averaging_3D



.. py:function:: check_constraints(eigenval, eigenvec, dim=3)

   Check that the requirements for frame averaging are satisfied

   :param eigenval: eigenvalues
   :type eigenval: tensor
   :param eigenvec: eigenvectors
   :type eigenvec: tensor
   :param dim: 2D or 3D frame averaging
   :type dim: int


.. py:function:: compute_frames(eigenvec, pos, cell, fa_method='stochastic', pos_3D=None, det_index=0)

   Compute all `frames` for a given graph, i.e. all possible
   canonical representations of the 3D graph (of all euclidean transformations).

   :param eigenvec: eigenvectors matrix
   :type eigenvec: tensor
   :param pos: centered position vector
   :type pos: tensor
   :param cell: cell direction (dxd)
   :type cell: tensor
   :param fa_method: the Frame Averaging (FA) inspired technique
                     chosen to select frames: stochastic-FA (stochastic), deterministic-FA (det),
                     Full-FA (all) or SE(3)-FA (se3).
   :type fa_method: str
   :param pos_3D: for 2D FA, pass atoms' 3rd position coordinate.

   :returns: 3D position tensors of projected representation
   :rtype: (list)


.. py:function:: data_augmentation(g, d=3, *args)

   Data augmentation: randomly rotated graphs are added
   in the dataloader transform.

   :param g: single graph
   :type g: data.Data
   :param d: dimension of the DA rotation (2D around z-axis or 3D)
   :type d: int
   :param rotation: around which axis do we rotate it.
                    Defaults to 'z'.
   :type rotation: str, optional

   :returns: rotated graph
   :rtype: (data.Data)


.. py:function:: frame_averaging_2D(pos, cell=None, fa_method='stochastic', check=False)

   Computes new positions for the graph atoms using
   frame averaging, which itself builds on the PCA of atom positions.
   2D case: we project the atoms on the plane orthogonal to the z-axis.
   Motivation: sometimes, the z-axis is not the most relevant one (e.g. fixed).

   :param pos: positions of atoms in the graph
   :type pos: tensor
   :param cell: unit cell of the graph. None if no pbc.
   :type cell: tensor
   :param fa_method: FA method used (stochastic, det, all, se3)
   :type fa_method: str
   :param check: check if constraints are satisfied. Default: False.
   :type check: bool

   :returns: updated atom positions
             (tensor): updated unit cell
             (tensor): the rotation matrix used (PCA)
   :rtype: (tensor)


.. py:function:: frame_averaging_3D(pos, cell=None, fa_method='stochastic', check=False)

   Computes new positions for the graph atoms using
   frame averaging, which itself builds on the PCA of atom positions.
   Base case for 3D inputs.

   :param pos: positions of atoms in the graph
   :type pos: tensor
   :param cell: unit cell of the graph. None if no pbc.
   :type cell: tensor
   :param fa_method: FA method used
                     (stochastic, det, all, se3-all, se3-det, se3-stochastic)
   :type fa_method: str
   :param check: check if constraints are satisfied. Default: False.
   :type check: bool

   :returns: updated atom positions
             (tensor): updated unit cell
             (tensor): the rotation matrix used (PCA)
   :rtype: (tensor)


