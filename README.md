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
  * [ ] Refactor averaging function (model_forward)
  * [ ] Data loader utils / transforms (graph & FA)
  * [ ] EC loss interface
  * [ ] Clean up faenet mp options
  * [ ] Update default args to the ones used in FAENet paper
  * [ ] Documentation
  * [ ] Document sample data (is2re_bs3)
  * [ ] ForceDecoder: merge mlp type and hidden channels in dict
  * [ ] ForceDecoder: update it
  * [ ] Import last changes from ICML cleanup
  * [ ] Remove se3-multiple, se3-det
  * [x] Implement faenet
  * [x] Implement FA/SFA
  * [x] Eval symmetries
  * [x] Idea: forcedecoder as nn module  
  * [x] Check att heads > 0 if "att" in mp_type -> tests & model
  
