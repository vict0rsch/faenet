:py:mod:`faenet.embedding`
==========================

.. py:module:: faenet.embedding


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   faenet.embedding.PhysEmbedding




.. py:class:: PhysEmbedding(props=True, props_grad=False, pg=False, short=False)

   Bases: :py:obj:`torch.nn.Module`

   Create physics-aware embeddings meta class with sub-emeddings for each atom

   :param props: Create an embedding of physical
                 properties. (default: :obj:`True`)
   :type props: bool, optional
   :param props_grad: Learn a physics-aware embedding
                      instead of keeping it fixed. (default: :obj:`False`)
   :type props_grad: bool, optional
   :param pg: Learn two embeddings based on period and
              group information respectively. (default: :obj:`False`)
   :type pg: bool, optional
   :param short: Remove all columns containing NaN values.
                 (default: :obj:`False`)
   :type short: bool, optional

   .. py:property:: device



