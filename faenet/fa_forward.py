from copy import deepcopy
import torch


def model_forward(batch, model, frame_averaging, mode="train", crystal_task=True):
    """Perform a model forward pass when frame averaging is applied.

    Args:
        batch (data.Batch): batch of graphs with attributes:
            - original atom positions (`pos`)
            - batch indices (to which graph in batch each atom belongs to) (`batch`)
            - frame averaged positions, cell and rotation matrices (`fa_pos`, `fa_cell`, `fa_rot`)
        model: model instance
        frame_averaging (str): symmetry preserving method (already) applied
            ("2D", "3D", "DA", "")
        mode (str, optional): model mode. Defaults to "train".
            ("train", "eval")
        crystal_task (bool, optional): Whether crystals (molecules) are considered.
            If they are, the unit cell (3x3) is affected by frame averaged and expected as attribute.
            (default: :obj:`True`)

    Returns:
        (dict): model predictions tensor for "energy" and "forces".
    """
    if isinstance(batch, list):
        batch = batch[0]
    if not hasattr(batch, "natoms"):
        batch.natoms = torch.unique(batch.batch, return_counts=True)[1]

    # Distinguish Frame Averaging prediction from traditional case.
    if frame_averaging and frame_averaging != "DA":
        original_pos = batch.pos
        if crystal_task:
            original_cell = batch.cell
        e_all, f_all, gt_all = [], [], []

        # Compute model prediction for each frame
        for i in range(len(batch.fa_pos)):
            batch.pos = batch.fa_pos[i]
            if crystal_task:
                batch.cell = batch.fa_cell[i]
            # Forward pass
            preds = model(deepcopy(batch), mode=mode)
            e_all.append(preds["energy"])
            fa_rot = None

            # Force predictions are rotated back to be equivariant
            if preds.get("forces") is not None:
                fa_rot = torch.repeat_interleave(batch.fa_rot[i], batch.natoms, dim=0)
                # Transform forces to guarantee equivariance of FA method
                g_forces = (
                    preds["forces"]
                    .view(-1, 1, 3)
                    .bmm(fa_rot.transpose(1, 2).to(preds["forces"].device))
                    .view(-1, 3)
                )
                f_all.append(g_forces)

            # Energy conservation loss
            if preds.get("forces_grad_target") is not None:
                if fa_rot is None:
                    fa_rot = torch.repeat_interleave(
                        batch.fa_rot[i], batch.natoms, dim=0
                    )
                # Transform gradients to stay consistent with FA
                g_grad_target = (
                    preds["forces_grad_target"]
                    .view(-1, 1, 3)
                    .bmm(fa_rot.transpose(1, 2).to(preds["forces_grad_target"].device))
                    .view(-1, 3)
                )
                gt_all.append(g_grad_target)

        batch.pos = original_pos
        if crystal_task:
            batch.cell = original_cell

        # Average predictions over frames
        preds["energy"] = sum(e_all) / len(e_all)
        if len(f_all) > 0 and all(y is not None for y in f_all):
            preds["forces"] = sum(f_all) / len(f_all)
        if len(gt_all) > 0 and all(y is not None for y in gt_all):
            preds["forces_grad_target"] = sum(gt_all) / len(gt_all)

    # Traditional case (no frame averaging)
    else:
        preds = model(batch, mode=mode)

    if preds["energy"].shape[-1] == 1:
        preds["energy"] = preds["energy"].view(-1)

    return preds
