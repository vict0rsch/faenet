:py:mod:`faenet.utils`
======================

.. py:module:: faenet.utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   faenet.utils.GaussianSmearing
   faenet.utils.RandomReflect
   faenet.utils.RandomRotate



Functions
~~~~~~~~~

.. autoapisummary::

   faenet.utils.base_preprocess
   faenet.utils.get_pbc_distances
   faenet.utils.pbc_preprocess
   faenet.utils.swish



.. py:class:: GaussianSmearing(start=0.0, stop=5.0, num_gaussians=50)

   Bases: :py:obj:`torch.nn.Module`

   Smears a distance distribution by a Gaussian function.

   .. py:method:: forward(dist)



.. py:class:: RandomReflect

   Bases: :py:obj:`object`

   Reflect node positions around a specific axis (x, y, x=y) or the origin.
   Take a random reflection type from a list of reflection types.
       (type 0: reflect wrt x-axis, type1: wrt y-axis, type2: y=x, type3: origin)

   .. py:method:: __call__(data)



.. py:class:: RandomRotate(degrees, axes=[0, 1, 2])

   Bases: :py:obj:`object`

   Rotates node positions around a specific axis by a randomly sampled
   factor within a given interval.

   :param degrees: Rotation interval from which the rotation
                   angle is sampled. If `degrees` is a number instead of a
                   tuple, the interval is given by :math:`[-\mathrm{degrees},
                   \mathrm{degrees}]`.
   :type degrees: tuple or float
   :param axes: The rotation axes. (default: `[0, 1, 2]`)
   :type axes: int, optional

   .. py:method:: __call__(data)


   .. py:method:: __repr__()

      Return repr(self).



.. py:function:: base_preprocess(data, cutoff=6.0, max_num_neighbors=40)

   Preprocess datapoint: create a cutoff graph,
       compute distances and relative positions.

       Args:
       data (data.Data): data object with specific attributes:
               - batch (N): index of the graph to which each atom belongs to in this batch
               - pos (N,3): atom positions
               - atomic_numbers (N): atomic numbers of each atom in the batch
               - edge_index (2,E): edge indices, for all graphs of the batch
           With B is the batch size, N the number of atoms in the batch (across all graphs),
           and E the number of edges in the batch.
           If these attributes are not present, implement your own preprocess function.
       cutoff (int): cutoff radius (in Angstrom)
       max_num_neighbors (int): maximum number of neighbors per node.

   :returns: atomic_numbers, batch, sparse adjacency matrix, relative positions, distances
   :rtype: (tuple)


.. py:function:: get_pbc_distances(pos, edge_index, cell, cell_offsets, neighbors, return_offsets=False, return_rel_pos=False)

   Compute distances between atoms with periodic boundary conditions

   :param pos: (N, 3) tensor of atomic positions
   :type pos: tensor
   :param edge_index: (2, E) tensor of edge indices
   :type edge_index: tensor
   :param cell: (3, 3) tensor of cell vectors
   :type cell: tensor
   :param cell_offsets: (N, 3) tensor of cell offsets
   :type cell_offsets: tensor
   :param neighbors: (N, 3) tensor of neighbor indices
   :type neighbors: tensor
   :param return_offsets: return the offsets
   :type return_offsets: bool
   :param return_rel_pos: return the relative positions vectors
   :type return_rel_pos: bool

   :returns:

             dictionary with the updated edge_index, atom distances,
                 and optionally the offsets and distance vectors.
   :rtype: (dict)


.. py:function:: pbc_preprocess(data, cutoff=6.0, max_num_neighbors=40)

   Preprocess datapoint using periodic boundary conditions
       to improve the existing graph.

   :param data: data object with specific attributes. B is the batch size,
                N the number of atoms in the batch (across all graphs), E the number of edges in the batch.
                    - batch (N): index of the graph to which each atom belongs to in this batch
                    - pos (N,3): atom positions
                    - atomic_numbers (N): atomic numbers of each atom in the batch
                    - cell (B, 3, 3): unit cell containing each graph, for materials.
                    - cell_offsets (E, 3): cell offsets for each edge, for materials
                    - neighbors (B): total number of edges inside each graph.
                    - edge_index (2,E): edge indices, for all graphs of the batch
                If these attributes are not present, implement your own preprocess function.
   :type data: data.Data

   :returns: atomic_numbers, batch, sparse adjacency matrix, relative positions, distances
   :rtype: (tuple)


.. py:function:: swish(x)

   Swish activation function


