:py:mod:`faenet.model`
======================

.. py:module:: faenet.model

.. autoapi-nested-parse::

   FAENet: Frame Averaging Equivariant graph neural Network
   Simple, scalable and expressive model for property prediction on 3D atomic systems.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   faenet.model.EmbeddingBlock
   faenet.model.FAENet
   faenet.model.InteractionBlock
   faenet.model.OutputBlock




.. py:class:: EmbeddingBlock(num_gaussians, num_filters, hidden_channels, tag_hidden_channels, pg_hidden_channels, phys_hidden_channels, phys_embeds, act, second_layer_MLP)

   Bases: :py:obj:`torch.nn.Module`

   Initialise atom and edge representations.

   .. py:method:: forward(z, rel_pos, edge_attr, tag=None, subnodes=None)

      Forward pass of the Embedding block.
      Called in FAENet to generate initial atom and edge representations.

      :param z: atomic numbers. (num_atoms, )
      :type z: tensor
      :param rel_pos: relative atomic positions. (num_edges, 3)
      :type rel_pos: tensor
      :param edge_attr: RBF of pairwise distances. (num_edges, num_gaussians)
      :type edge_attr: tensor
      :param tag: atom information specific to OCP. Defaults to None.
      :type tag: tensor, optional

      :returns: atom embeddings, edge embeddings
      :rtype: (tensor, tensor)


   .. py:method:: reset_parameters()



.. py:class:: FAENet(cutoff = 6.0, preprocess = 'pbc_preprocess', act = 'swish', max_num_neighbors = 40, hidden_channels = 128, tag_hidden_channels = 32, pg_hidden_channels = 32, phys_embeds = True, phys_hidden_channels = 0, num_interactions = 4, num_gaussians = 50, num_filters = 128, second_layer_MLP = True, skip_co = 'concat', mp_type = 'updownscale_base', graph_norm = True, complex_mp = False, energy_head = None, out_dim = 1, pred_as_dict = True, regress_forces = None, force_decoder_type = 'mlp', force_decoder_model_config = {'hidden_channels': 128})

   Bases: :py:obj:`faenet.base_model.BaseModel`

   Non-symmetry preserving GNN model for 3D atomic systems,
   called FAENet: Frame Averaging Equivariant Network.

   :param cutoff: Cutoff distance for interatomic interactions.
                  (default: :obj:`6.0`)
   :type cutoff: float
   :param preprocess: Pre-processing function for the data. This function
                      should accept a data object as input and return a tuple containing the following:
                      atomic numbers, batch indices, final adjacency, relative positions, pairwise distances.
                      Examples of valid preprocessing functions include `pbc_preprocess`,
                      `base_preprocess`, or custom functions.
   :type preprocess: callable
   :param act: Activation function
               (default: `swish`)
   :type act: str
   :param max_num_neighbors: The maximum number of neighbors to
                             collect for each node within the :attr:`cutoff` distance.
                             (default: `40`)
   :type max_num_neighbors: int
   :param hidden_channels: Hidden embedding size.
                           (default: `128`)
   :type hidden_channels: int
   :param tag_hidden_channels: Hidden tag embedding size.
                               (default: :obj:`32`)
   :type tag_hidden_channels: int
   :param pg_hidden_channels: Hidden period and group embedding size.
                              (default: :obj:`32`)
   :type pg_hidden_channels: int
   :param phys_embeds: Do we include fixed physics-aware embeddings.
                       (default: :obj: `True`)
   :type phys_embeds: bool
   :param phys_hidden_channels: Hidden size of learnable physics-aware embeddings.
                                (default: :obj:`0`)
   :type phys_hidden_channels: int
   :param num_interactions: The number of interaction (i.e. message passing) blocks.
                            (default: :obj:`4`)
   :type num_interactions: int
   :param num_gaussians: The number of gaussians :math:`\mu` to encode distance info.
                         (default: :obj:`50`)
   :type num_gaussians: int
   :param num_filters: The size of convolutional filters.
                       (default: :obj:`128`)
   :type num_filters: int
   :param second_layer_MLP: Use 2-layers MLP at the end of the Embedding block.
                            (default: :obj:`False`)
   :type second_layer_MLP: bool
   :param skip_co: Add a skip connection between each interaction block and
                   energy-head. (`False`, `"add"`, `"concat"`, `"concat_atom"`)
   :type skip_co: str
   :param mp_type: Specificies the Message Passing type of the interaction block.
                   (`"base"`, `"updownscale_base"`, `"updownscale"`, `"updown_local_env"`, `"simple"`):
   :type mp_type: str
   :param graph_norm: Whether to apply batch norm after every linear layer.
                      (default: :obj:`True`)
   :type graph_norm: bool
   :param complex_mp: (default: :obj:`True`)
   :type complex_mp: bool
   :param energy_head: Method to compute energy prediction
                       from atom representations.
                       (`None`, `"weighted-av-initial-embeds"`, `"weighted-av-final-embeds"`)
   :type energy_head: str
   :param out_dim: size of the output tensor for graph-level predicted properties ("energy")
                   Allows to predict multiple properties at the same time.
                   (default: :obj:`1`)
   :type out_dim: int
   :param pred_as_dict: Set to False to return a (property) prediction tensor.
                        By default, predictions are returned as a dictionary with several keys (e.g. energy, forces)
                        (default: :obj:`True`)
   :type pred_as_dict: bool
   :param regress_forces: Specifies if we predict forces or not, and how
                          do we predict them. (`None` or `""`, `"direct"`, `"direct_with_gradient_target"`)
   :type regress_forces: str
   :param force_decoder_type: Specifies the type of force decoder
                              (`"simple"`, `"mlp"`, `"res"`, `"res_updown"`)
   :type force_decoder_type: str
   :param force_decoder_model_config: contains information about the
                                      for decoder architecture (e.g. number of layers, hidden size).
   :type force_decoder_model_config: dict

   .. py:method:: energy_forward(data, preproc=True)

      Predicts any graph-level property (e.g. energy) for 3D atomic systems.

      :param data: Batch of graphs data objects.
      :type data: data.Batch
      :param preproc: Whether to apply (any given) preprocessing to the graph.
                      Default to True.
      :type preproc: bool

      :returns:

                predicted properties for each graph (key: "energy")
                    and final atomic representations (key: "hidden_state")
      :rtype: (dict)


   .. py:method:: forces_forward(preds)

      Predicts forces for 3D atomic systems.
      Can be utilised to predict any atom-level property.

      :param preds: dictionnary with final atomic representations
                    (hidden_state) and predicted properties (e.g. energy)
                    for each graph
      :type preds: dict

      :returns: additional predicted properties, at an atom-level (e.g. forces)
      :rtype: (dict)



.. py:class:: InteractionBlock(hidden_channels, num_filters, act, mp_type, complex_mp, graph_norm)

   Bases: :py:obj:`torch_geometric.nn.MessagePassing`

   Updates atom representations through custom message passing.

   .. py:method:: forward(h, edge_index, e)

      Forward pass of the Interaction block.
      Called in FAENet forward pass to update atom representations.

      :param h: atom embedddings. (num_atoms, hidden_channels)
      :type h: tensor
      :param edge_index: adjacency matrix. (2, num_edges)
      :type edge_index: tensor
      :param e: edge embeddings. (num_edges, num_filters)
      :type e: tensor

      :returns: updated atom embeddings
      :rtype: (tensor)


   .. py:method:: message(x_j, W, local_env=None)


   .. py:method:: reset_parameters()



.. py:class:: OutputBlock(energy_head, hidden_channels, act, out_dim=1)

   Bases: :py:obj:`torch.nn.Module`

   Compute task-specific predictions from final atom representations.

   .. py:method:: forward(h, edge_index, edge_weight, batch, alpha)

      Forward pass of the Output block.
      Called in FAENet to make prediction from final atom representations.

      :param h: atom representations. (num_atoms, hidden_channels)
      :type h: tensor
      :param edge_index: adjacency matrix. (2, num_edges)
      :type edge_index: tensor
      :param edge_weight: edge weights. (num_edges, )
      :type edge_weight: tensor
      :param batch: batch indices. (num_atoms, )
      :type batch: tensor
      :param alpha: atom attention weights for late energy head. (num_atoms, )
      :type alpha: tensor

      :returns: graph-level representation (e.g. energy prediction)
      :rtype: (tensor)


   .. py:method:: reset_parameters()



