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
  * [ ] ForceDecoder: merge mlp type and hidden channels in dict
  * [x] idea: forcedecoder as nn module
* Release
  * [ ] update default args to the ones used in FAENet paper
  * [ ] documentation
  * [x] check att heads > 0 if "att" in mp_type -> tests & model
  * [ ] document sample data (is2re_bs3)
