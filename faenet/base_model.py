import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def reset_parameters(self):
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

    def energy_forward(self, data):
        raise NotImplementedError

    def forces_forward(self, preds):
        raise NotImplementedError

    def forward(self, data, mode="train"):
        grad_forces = forces = None

        # energy gradient w.r.t. positions will be computed
        if mode == "train" or self.regress_forces == "from_energy":
            data.pos.requires_grad_(True)

        # predict energy
        preds = self.energy_forward(data)

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
                    # store the energy gradient as the target. Used for metrics
                    # only in "direct" mode.
                    preds["forces_grad_target"] = grad_forces.detach()
            else:
                raise ValueError(
                    f"Unknown forces regression mode {self.regress_forces}"
                )

        return preds

    def forces_as_energy_grad(self, pos, energy):
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
