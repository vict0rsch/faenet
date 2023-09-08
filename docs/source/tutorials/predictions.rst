Making predictions with ``FAENet``
==================================


In this tutorial, we will learn how to use ``FAENet`` to make predictions. We will use the ``FAENet`` model trained on the ``PDBbind`` dataset.

Import modules
--------------

Import ``FAENet`` and ``FAENetPredictor`` from ``deepchem.models``.

.. code:: python

  from deepchem.models import FAENet
  from deepchem.models import FAENetPredictor

Load ``FAENet`` model
---------------------

Load ``FAENet`` model trained on the ``PDBbind`` dataset.

.. code:: python

  model = FAENet.load_from_dir(
      "/path/to/pdbbind/FAENet/model")

Load test dataset
-----------------

Load the test dataset from the ``pdbbind`` dataset.

.. code:: python

  pdbbind_tasks, pdbbind_datasets, transformers = dc.molnet.load_pdbbind(
      featurizer='FAE', split='test')

Create ``FAENetPredictor``
--------------------------

Create ``FAENetPredictor`` using the loaded ``FAENet`` model.

.. code:: python

  predictor = FAENetPredictor(model, dataset=pdbbind_datasets['test'])

Evaluate ``FAENet``
-------------------

Evaluate the ``FAENet`` model on the test dataset.

.. code:: python

  scores = predictor.evaluate(pdbbind_datasets['test'], pdbbind_tasks,
                              transformers)

The output is:

.. code:: text

  Evaluating model
  0.0
  computed_metrics: [0.0]

Make predictions with ``FAENet``
--------------------------------

Make predictions with the ``FAENet`` model on the test dataset.

.. code:: python

  y_pred = predictor.predict(pdbbind_datasets['test'])

The output is:

.. code:: text

  Making predictions
  0.0
  computed_metrics: [0.0]

