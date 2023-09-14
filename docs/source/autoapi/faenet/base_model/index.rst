:py:mod:`faenet.base_model`
===========================

.. py:module:: faenet.base_model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   faenet.base_model.BaseModel




.. py:class:: BaseModel(**kwargs)

   Bases: :py:obj:`torch.nn.Module`

   Base class for ML models applied to 3D atomic systems.

   .. py:property:: num_params


   .. py:method:: energy_forward(data, preproc=True)
      :abstractmethod:

      Forward pass for energy prediction.


   .. py:method:: forces_as_energy_grad(pos, energy)

      Computes forces from energy gradient

      :param pos: 3D atom positions
      :type pos: tensor
      :param energy: system's predicted energy
      :type energy: tensor

      :returns: gradient of energy w.r.t. atom positions
      :rtype: forces (tensor)


   .. py:method:: forces_forward(preds)
      :abstractmethod:

      Forward pass for force prediction.


   .. py:method:: forward(data, mode='train', preproc=True)

      Main Forward pass.

      :param data: input data object, with 3D atom positions (pos)
      :type data: Data
      :param mode: train or inference mode
      :type mode: str
      :param preproc: Whether to preprocess (pbc, cutoff graph)
                      the input graph or point cloud. Default: True.
      :type preproc: bool

      :returns: predicted energy, forces and final atomic hidden states
      :rtype: dict


   .. py:method:: reset_parameters()

      Resets all learnable parameters of the module.



