import torch
from copy import deepcopy
from faenet.fa_forward import model_forward
from faenet.utils import RandomRotate, RandomReflect


@torch.no_grad()
def test_model_symmetries(self, model, device, task_name, debug_batches=-1):
    """Test rotation and reflection invariance & equivariance
    of GNNs
    Returns:
        (tensors): metrics to measure RI difference in
        energy/force pred. or pos. between G and rotated G
    """
    model.eval()

    energy_diff = torch.zeros(1, device=device)
    energy_diff_z = torch.zeros(1, device=device)
    energy_diff_z_percentage = torch.zeros(1, device=device)
    energy_diff_refl = torch.zeros(1, device=device)
    pos_diff_total = torch.zeros(1, device=device)
    forces_diff = torch.zeros(1, device=device)
    forces_diff_z = torch.zeros(1, device=device)
    forces_diff_z_graph = torch.zeros(1, device=device)
    forces_diff_refl = torch.zeros(1, device=device)
    n_batches = 0
    n_atoms = 0

    for i, batch in enumerate(self.loaders[self.config["dataset"]["default_val"]]):
        # if debug_batches > 0 and i == debug_batches:
        #     break

        n_batches += len(batch[0].natoms)
        n_atoms += batch[0].natoms.sum()

        # Compute model prediction
        preds1 = model_forward(deepcopy(batch), mode="inference")

        # Compute prediction on rotated graph
        rotated = rotate_graph(batch, rotation="z")
        preds2 = model_forward(deepcopy(rotated["batch_list"]), mode="inference")

        # Difference in predictions, for energy and forces
        energy_diff_z += torch.abs(preds1["energy"] - preds2["energy"]).sum()

        if task_name == "forces":
            energy_diff_z_percentage += (
                torch.abs(preds1["energy"] - preds2["energy"])
                / torch.abs(batch[0].y).to(preds1["energy"].device)
            ).sum()
            forces_diff_z += torch.abs(
                preds1["forces"] @ rotated["rot"].to(preds1["forces"].device)
                - preds2["forces"]
            ).sum()
            assert torch.allclose(
                torch.abs(
                    batch[0].force @ rotated["rot"].to(batch[0].force.device)
                    - rotated["batch_list"][0].force
                ).sum(),
                torch.tensor([0.0]),
                atol=1e-05,
            )
        elif task_name == "is2re":
            energy_diff_z_percentage += (
                torch.abs(preds1["energy"] - preds2["energy"])
                / torch.abs(batch[0].y_relaxed).to(preds1["energy"].device)
            ).sum()
        else:
            energy_diff_z_percentage += (
                torch.abs(preds1["energy"] - preds2["energy"])
                / torch.abs(batch[0].y).to(preds1["energy"].device)
            ).sum()

        # Diff in positions
        pos_diff = -1
        if hasattr(batch[0], "fa_pos"):
            pos_diff = 0
            # Compute total difference across frames
            for pos1, pos2 in zip(batch[0].fa_pos, rotated["batch_list"][0].fa_pos):
                pos_diff += pos1 - pos2
            # Manhattan distance of pos matrix wrt 0 matrix.
            pos_diff_total += torch.abs(pos_diff).sum()

        # Reflect graph and compute diff in prediction
        reflected = self.reflect_graph(batch)
        preds3 = model_forward(reflected["batch_list"], mode="inference")
        energy_diff_refl += torch.abs(preds1["energy"] - preds3["energy"]).sum()
        if task_name == "forces":
            forces_diff_refl += torch.abs(
                preds1["forces"] @ reflected["rot"].to(preds1["forces"].device)
                - preds3["forces"]
            ).sum()

        # 3D Rotation and compute diff in prediction
        rotated = rotate_graph(batch)
        preds4 = model_forward(rotated["batch_list"], mode="inference")
        energy_diff += torch.abs(preds1["energy"] - preds4["energy"]).sum()
        if task_name == "forces":
            forces_diff += torch.abs(preds1["forces"] - preds4["forces"]).sum()

    # Aggregate the results
    energy_diff_z = energy_diff_z / n_batches
    energy_diff_z_percentage = energy_diff_z_percentage / n_batches
    energy_diff = energy_diff / n_batches
    energy_diff_refl = energy_diff_refl / n_batches
    pos_diff_total = pos_diff_total / n_batches

    symmetry = {
        "2D_E_ri": float(energy_diff_z),
        "2D_E_ri_percentage": float(energy_diff_z_percentage),
        "3D_E_ri": float(energy_diff),
        "2D_pos_ri": float(pos_diff_total),
        "2D_E_refl_i": float(energy_diff_refl),
    }

    # Test equivariance of forces
    if task_name == "forces":
        forces_diff_z = forces_diff_z / n_atoms
        forces_diff_z_graph = forces_diff_z / n_batches
        forces_diff = forces_diff / n_atoms
        forces_diff_refl = forces_diff_refl / n_atoms
        symmetry.update(
            {
                "2D_F_ri_graph": float(forces_diff_z_graph),
                "2D_F_ri": float(forces_diff_z),
                "3D_F_ri": float(forces_diff),
                "2D_F_refl_i": float(forces_diff_refl),
            }
        )

    # print("".join([f"\n  > {k:12}: {v:.5f}" for k, v in symmetry.items()]))

    return symmetry


def rotate_graph(self, batch, rotation=None):
    """Rotate all graphs in a batch
    Args:
        batch (data.Batch): batch of graphs
        rotation (str, optional): type of rotation applied. Defaults to None.
    Returns:
        data.Batch: rotated batch
    """
    if isinstance(batch, list):
        batch = batch[0]

    # Sampling a random rotation within [-180, 180] for all axes.
    if rotation == "z":
        transform = RandomRotate([-180, 180], [2])
    elif rotation == "x":
        transform = RandomRotate([-180, 180], [0])
    elif rotation == "y":
        transform = RandomRotate([-180, 180], [1])
    else:
        transform = RandomRotate([-180, 180], [0, 1, 2])

    # Rotate graph
    batch_rotated, rot, inv_rot = transform(deepcopy(batch))
    assert not torch.allclose(batch.pos, batch_rotated.pos, atol=1e-05)

    # Recompute fa-pos for batch_rotated
    if hasattr(batch, "fa_pos"):
        delattr(batch_rotated, "fa_pos")  # delete it otherwise can't iterate
        delattr(batch_rotated, "fa_cell")  # delete it otherwise can't iterate
        delattr(batch_rotated, "fa_rot")  # delete it otherwise can't iterate

        g_list = batch_rotated.to_data_list()
        fa_transform = FrameAveraging(
            self.config["frame_averaging"], self.config["fa_frames"]
        )
        for g in g_list:
            g = fa_transform(g)
        batch_rotated = Batch.from_data_list(g_list)
        if hasattr(batch, "neighbors"):
            batch_rotated.neighbors = batch.neighbors

    return {"batch_list": [batch_rotated], "rot": rot}

def reflect_graph(self, batch, reflection=None):
    """Rotate all graphs in a batch
    Args:
        batch (data.Batch): batch of graphs
        rotation (str, optional): type of rotation applied. Defaults to None.
    Returns:
        data.Batch: rotated batch
    """
    if isinstance(batch, list):
        batch = batch[0]

    # Sampling a random rotation within [-180, 180] for all axes.
    transform = RandomReflect()

    # Reflect batch
    batch_reflected, rot, inv_rot = transform(deepcopy(batch))
    assert not torch.allclose(batch.pos, batch_reflected.pos, atol=1e-05)

    # Recompute fa-pos for batch_rotated
    if hasattr(batch, "fa_pos"):
        delattr(batch_reflected, "fa_pos")  # delete it otherwise can't iterate
        delattr(batch_reflected, "fa_cell")  # delete it otherwise can't iterate
        delattr(batch_reflected, "fa_rot")  # delete it otherwise can't iterate
        g_list = batch_reflected.to_data_list()
        fa_transform = FrameAveraging(
            self.config["frame_averaging"], self.config["fa_frames"]
        )
        for g in g_list:
            g = fa_transform(g)
        batch_reflected = Batch.from_data_list(g_list)
        if hasattr(batch, "neighbors"):
            batch_reflected.neighbors = batch.neighbors

    return {"batch_list": [batch_reflected], "rot": rot}