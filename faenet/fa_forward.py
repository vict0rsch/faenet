from copy import deepcopy
import torch


def model_forward(batch_list, model, frame_averaging, mode="train", crystal_task=True):
    """ Perform a forward pass of the model when frame averaging is applied. 

    Args:
        batch_list (data.Batch): batch 
        model: model instance
        frame_averaging (str): (FA) method employed 
            ("2D", "3D", "DA")
        mode (str, optional): model mode. Defaults to "train".
        crystal_task (bool, optional): Whether crystals are considered. 
            Defaults to True.

    Returns:
        dict: model predictions with entries "energy" and "forces"
    """
    
    # Distinguish Frame Averaging prediction from traditional case.
    if frame_averaging and frame_averaging != "DA":
        original_pos = batch_list.pos
        if crystal_task: 
            original_cell = batch_list.cell 
        e_all, f_all, gt_all = [], [], []

        # Compute model prediction for each frame
        for i in range(len(batch_list.fa_pos)):
            batch_list.pos = batch_list.fa_pos[i]
            if crystal_task: 
                batch_list.cell = batch_list.fa_cell[i]
            # Forward pass
            preds = model(deepcopy(batch_list), mode=mode)
            e_all.append(preds["energy"])
            fa_rot = None

            # Force predictions are rotated back to be equivariant
            if preds.get("forces") is not None:
                fa_rot = torch.repeat_interleave(
                    batch_list.fa_rot[i], batch_list.natoms, dim=0
                )
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
                        batch_list.fa_rot[i], batch_list.natoms, dim=0
                    )
                # Transform gradients to stay consistent with FA
                g_grad_target = (
                    preds["forces_grad_target"]
                    .view(-1, 1, 3)
                    .bmm(
                        fa_rot.transpose(1, 2).to(
                            preds["forces_grad_target"].device
                        )
                    )
                    .view(-1, 3)
                )
                gt_all.append(g_grad_target)

        batch_list.pos = original_pos
        if crystal_task:
            batch_list.cell = original_cell

        # Average predictions over frames
        preds["energy"] = sum(e_all) / len(e_all)
        if len(f_all) > 0 and all(y is not None for y in f_all):
            preds["forces"] = sum(f_all) / len(f_all)
        if len(gt_all) > 0 and all(y is not None for y in gt_all):
            preds["forces_grad_target"] = sum(gt_all) / len(gt_all)
    
    # Traditional case (no frame averaging)
    else:
        preds = model(batch_list)

    if preds["energy"].shape[-1] == 1:
        preds["energy"] = preds["energy"].view(-1)

    return preds