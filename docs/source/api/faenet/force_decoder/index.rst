:py:mod:`faenet.force_decoder`
==============================

.. py:module:: faenet.force_decoder


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   faenet.force_decoder.ForceDecoder
   faenet.force_decoder.LambdaLayer




.. py:class:: ForceDecoder(type, input_channels, model_configs, act)

   Bases: :py:obj:`torch.nn.Module`

   Predicts a force vector per atom from final atomic representations.

   :param type: Type of force decoder to use
   :type type: str
   :param input_channels: Number of input channels
   :type input_channels: int
   :param model_configs: Dictionary of config parameters for the
                         decoder's model
   :type model_configs: dict
   :param act: Activation function (NOT a module)
   :type act: callable

   :raises ValueError: Unknown type of decoder

   :returns: Predicted force vector per atom
   :rtype: (torch.Tensor)

   .. py:method:: forward(h)


   .. py:method:: reset_parameters()



.. py:class:: LambdaLayer(func)

   Bases: :py:obj:`torch.nn.Module`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:method:: forward(x)



