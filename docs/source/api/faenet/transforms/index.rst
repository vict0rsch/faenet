:py:mod:`faenet.transforms`
===========================

.. py:module:: faenet.transforms


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   faenet.transforms.FrameAveraging
   faenet.transforms.Transform




.. py:class:: FrameAveraging(frame_averaging=None, fa_method=None)

   Bases: :py:obj:`Transform`

   Frame Averaging (FA) Transform for (PyG) Data objects (e.g. 3D atomic graphs).

   Computes new atomic positions (`fa_pos`) for all datapoints, as well as
   new unit cells (`fa_cell`) attributes for crystal structures, when applicable.
   The rotation matrix (`fa_rot`) used for the frame averaging is also stored.

   :param frame_averaging: Transform method used.
                           Can be 2D FA, 3D FA, Data Augmentation or no FA, respectively denoted by
                           (`"2D"`, `"3D"`, `"DA"`, `""`)
   :type frame_averaging: str
   :param fa_method: the actual frame averaging technique used.
                     "stochastic" refers to sampling one frame at random (at each epoch),
                     "det" to chosing deterministically one frame, and "all" to using all frames.
                     The prefix "se3-" refers to the SE(3) equivariant version of the method.
                     "" means that no frame averaging is used.
                     (`""`, `"stochastic"`, `"all"`, `"det"`, `"se3-stochastic"`, `"se3-all"`, `"se3-det"`)
   :type fa_method: str

   :returns: updated data object with new positions (+ unit cell) attributes
             and the rotation matrices used for the frame averaging transform.
   :rtype: (data.Data)

   .. py:method:: __call__(data)

      The only requirement for the data is to have a `pos` attribute.



.. py:class:: Transform

   Base class for all transforms.

   .. py:method:: __call__(data)
      :abstractmethod:


   .. py:method:: __str__()

      Return str(self).



