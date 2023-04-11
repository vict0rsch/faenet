import torch
import torch.nn as nn

class DDPLoss(nn.Module):
    def __init__(self, loss_fn, reduction="mean"):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "sum"
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        loss = self.loss_fn(input, target)
        if self.reduction == "mean":
            num_samples = input.shape[0]
            num_samples = dist_utils.all_reduce(num_samples, device=input.device)
            # Multiply by world size since gradients are averaged
            # across DDP replicas
            return loss * dist_utils.get_world_size() / num_samples
        else:
            return loss

class L2MAELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)

def load_loss(config, dist_utils):
    loss_fn = {}
    loss_fn["energy"] = config["optim"].get("loss_energy", "mae")
    loss_fn["force"] = config["optim"].get("loss_force", "mae")
    for loss, loss_name in loss_fn.items():
        if loss_name in ["l1", "mae"]:
            loss_fn[loss] = nn.L1Loss()
        elif loss_name == "mse":
            loss_fn[loss] = nn.MSELoss()
        elif loss_name == "l2mae":
            loss_fn[loss] = L2MAELoss()
        else:
            raise NotImplementedError(f"Unknown loss function name: {loss_name}")
        if dist_utils.initialized():
            loss_fn[loss] = DDPLoss(loss_fn[loss])


def compute_loss(preds, batch_list, config, loss_fn, task_name, normalizer, device):
    """ Compute the loss for a batch of data.

    Args:
        preds (dict): dictionary of model predictions, with entries
            "energy", "forces" and "forces_grad_target" if available.
        batch_list (data.Batch): batch of graphs
        config (dict): configuration dictionnary
        loss_fn (dict): dictionary of loss functions
        task_name (str): name of the task
        normalizers (dict): dictionary of normalizers
        device (torch.device): device on which to compute the loss

    Returns:
        dict: dictionnary with entries representing the different loss terms
            ("energy_loss", "force_loss", "total_loss", "energy_grad_loss")
    """    
    loss = {"total_loss": []}

    # Energy loss
    energy_target = torch.cat(
        [
            batch.y_relaxed.to(device)
            if task_name == "is2re"
            else batch.y.to(device)
            for batch in batch_list
        ],
        dim=0,
    )

    if normalizer.get("normalize_labels", False):
        hofs = None
        if task_name == "qm7x":
            hofs = torch.cat(
                [batch.hofs.to(device) for batch in batch_list], dim=0
            )
        target_normed = normalizers["target"].norm(energy_target, hofs=hofs)
    else:
        target_normed = energy_target
    energy_mult = config["optim"].get("energy_coefficient", 1)
    loss["energy_loss"] = loss_fn["energy"](preds["energy"], target_normed)
    loss["total_loss"].append(energy_mult * loss["energy_loss"])

    # Force loss.
    if task_name in {"is2rs", "s2ef"} or config["model"].get(
        "regress_forces"
    ):
        force_target = torch.cat(
            [batch.force.to(device) for batch in batch_list], dim=0
        )
        if (
            normalizer.get("normalize_labels", False)
            and "grad_target" in normalizers
        ):
            force_target = normalizers["grad_target"].norm(force_target)

        tag_specific_weights = config["task"].get("tag_specific_weights", [])
        if tag_specific_weights != []:
            # handle tag specific weights as introduced in forcenet
            assert len(tag_specific_weights) == 3

            batch_tags = torch.cat(
                [batch.tags.float().to(device) for batch in batch_list],
                dim=0,
            )
            weight = torch.zeros_like(batch_tags)
            weight[batch_tags == 0] = tag_specific_weights[0]
            weight[batch_tags == 1] = tag_specific_weights[1]
            weight[batch_tags == 2] = tag_specific_weights[2]

            loss_force_list = torch.abs(preds["forces"] - force_target)
            train_loss_force_unnormalized = torch.sum(
                loss_force_list * weight.view(-1, 1)
            )
            train_loss_force_normalizer = 3.0 * weight.sum()

            # add up normalizer to obtain global normalizer
            dist_utils.all_reduce(train_loss_force_normalizer)

            # perform loss normalization before backprop
            train_loss_force_normalized = train_loss_force_unnormalized * (
                dist_utils.get_world_size() / train_loss_force_normalizer
            )
            loss.append(train_loss_force_normalized)

        else:
            # Force coefficient = 30 has been working well for us.
            force_mult = config["optim"].get("force_coefficient", 30)
            mask = torch.ones_like(force_target).bool().to(device)
            if config["task"].get("train_on_free_atoms", False):
                fixed = torch.cat(
                    [batch.fixed.to(device) for batch in batch_list]
                )
                mask = fixed == 0

            loss["force_loss"] = loss_fn["force"](
                preds["forces"][mask], force_target[mask]
            )
            loss["total_loss"].append(force_mult * loss["force_loss"])
            if "forces_grad_target" in preds:
                grad_target = preds["forces_grad_target"]
                if config["model"].get("cosine_sim", False): 
                    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                    loss["energy_grad_loss"] = - torch.mean(cos(preds["forces"][mask], grad_target[mask]))
                else: 
                    loss["energy_grad_loss"] = loss_fn["force"](
                        preds["forces"][mask], grad_target[mask]
                    )
                if (
                    model.module.regress_forces
                    == "direct_with_gradient_target"
                ):
                    energy_grad_mult = config["optim"].get(
                        "energy_grad_coefficient", 10
                    )
                    loss["total_loss"].append(
                        energy_grad_mult * loss["energy_grad_loss"]
                    )
    # Sanity check to make sure the compute graph is correct.
    for lc in loss["total_loss"]:
        assert hasattr(lc, "grad_fn")

    loss["total_loss"] = sum(loss["total_loss"])
    return loss