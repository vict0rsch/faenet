# FAENet

## Installation

1. Install [`torch`](https://pytorch.org/get-started/locally/)
2. Install [`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#)
3. Install `faenet`:
    ```
    $ pip install faenet
    ```
    or from source
    ```
    pip install git+https://github.com/vict0rsch/faenet.git#egg=faenet
    ```

# TODO

* [ ] averaging function
* [ ] update default args to paper HPs
* [ ] data loader utils / transforms (graph & FA)
* [ ] eval symmetries
* [ ] documentation
* [ ] check att heads > 0 if "att" in mp_type -> tests & model
* [ ] EC loss interface
* [ ] document sample data
* [ ] check docstrings
* [ ] clean up faenet mp options
* [ ] ForceDecoder takes a single model config now not a dict of configs per type
* [ ] idea: forcedecoder as nn module
