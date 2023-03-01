<p align="center">
<strong><a href="https://github.com/vict0rsch/faenet" target="_blank">Code</a></strong>
<strong>&nbsp;&nbsp;•&nbsp;&nbsp;</strong>
<strong><a href="https://faenet.readthedocs.io/" target="_blank">Docs</a></strong>
</p>

# FAENet

## Installation

1. `Python >= 3.8`
2. [`torch > 1.11`](https://pytorch.org/get-started/locally/)
3. [`torch_geometric > 2.1`](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#)
4. `faenet`:
    ```
    $ pip install faenet
    ```
    or from source
    ```
    pip install git+https://github.com/vict0rsch/faenet.git#egg=faenet
    ```

# TODO

* Functional
  * [ ] refactor averaging function
  * [ ] data loader utils / transforms (graph & FA)
  * [ ] eval symmetries
  * [ ] EC loss interface
  * [ ] clean up faenet mp options
  * [ ] ForceDecoder takes a single model config now not a dict of configs per type
  * [ ] idea: forcedecoder as nn module
* Release
  * [ ] update default args to paper HPs
  * [ ] documentation
  * [ ] check att heads > 0 if "att" in mp_type -> tests & model
  * [ ] document sample data
  * [ ] check docstrings
