import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for ML models applied to 3D atomic systems."""

    def __init__(self, **kwargs):
        super().__init__()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                assert isinstance(child, nn.Module)
                assert callable(child.reset_parameters)
                child.reset_parameters()
            else:
                if hasattr(child, "weight"):
                    nn.init.xavier_uniform_(child.weight)
                if hasattr(child, "bias"):
                    child.bias.data.fill_(0)

    def energy_forward(self, data, preproc=True):
        """Forward pass for energy prediction."""
        raise NotImplementedError

    def forces_forward(self, preds):
        """Forward pass for force prediction."""
        raise NotImplementedError

    def forward(self, data, mode="train", preproc=True):
        """Main Forward pass.

        Args:
            data (Data): input data object, with 3D atom positions (pos)
            mode (str): train or inference mode
            preproc (bool): Whether to preprocess (pbc, cutoff graph)
                the input graph or point cloud. Default: True.

        Returns:
            (dict): predicted energy, forces and final atomic hidden states
        """
        grad_forces = forces = None

        # energy gradient w.r.t. positions will be computed
        if mode == "train" or self.regress_forces == "from_energy":
            data.pos.requires_grad_(True)

        # predict energy
        preds = self.energy_forward(data, preproc)

        if self.regress_forces:
            if self.regress_forces in {"direct", "direct_with_gradient_target"}:
                # predict forces
                forces = self.forces_forward(preds)

            if mode == "train" or self.regress_forces == "from_energy":
                if "gemnet" in self.__class__.__name__.lower():
                    # gemnet forces are already computed
                    grad_forces = forces
                else:
                    # compute forces from energy gradient
                    grad_forces = self.forces_as_energy_grad(data.pos, preds["energy"])

            if self.regress_forces == "from_energy":
                # predicted forces are the energy gradient
                preds["forces"] = grad_forces
            elif self.regress_forces in {"direct", "direct_with_gradient_target"}:
                # predicted forces are the model's direct forces
                preds["forces"] = forces
                if mode == "train":
                    # Store the energy gradient as target for "direct_with_gradient_target"
                    # Use it as a metric only in "direct" mode.
                    preds["forces_grad_target"] = grad_forces.detach()
            else:
                raise ValueError(
                    f"Unknown forces regression mode {self.regress_forces}"
                )

        if not self.pred_as_dict:
            return preds["energy"]

        return preds

    def forces_as_energy_grad(self, pos, energy):
        """Computes forces from energy gradient

        Args:
            pos (tensor): 3D atom positions
            energy (tensor): system's predicted energy

        Returns:
            (tensor): forces as the energy gradient w.r.t. atom positions
        """

        return -1 * (
            torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
