import torch
from copy import deepcopy
from faenet.fa_forward import model_forward
from faenet.utils import RandomRotate, RandomReflect
from faenet import FrameAveraging
from torch_geometric.data import Batch


def transform_batch(g_list, fa_transform, neighbors=None):
    for g in g_list:
        g = fa_transform(g)
    batch_rotated = Batch.from_data_list(g_list)
    if neighbors is not None:
        batch_rotated.neighbors = neighbors
    return batch_rotated


def rotate_graph(batch, fa, fa_frames, rotation=None):
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
        fa_transform = FrameAveraging(fa, fa_frames)

        batch_rotated = transform_batch(
            g_list,
            fa_transform,
            batch.neighbors if hasattr(batch, "neighbors") else None,
        )

        return {"batch_list": batch_rotated, "rot": rot}


def reflect_graph(batch, frame_averaging, fa_frames, reflection=None):
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
        fa_transform = FrameAveraging(frame_averaging, fa_frames)

        batch_reflected = transform_batch(
            g_list,
            fa_transform,
            batch.neighbors if hasattr(batch, "neighbors") else None,
        )

    return {"batch_list": batch_reflected, "rot": rot}


@torch.no_grad()
def eval_model_symmetries(
    loader, model, fa, fa_frames, device, task_name, crystal_task=True
):
    """Test rotation and reflection invariance & equivariance of GNNs

    Args:
        loader (data): dataloader
        model: model instance
        fa (str): frame averaging ("2D", "3D") or data augmentation ("DA")
        fa_frames (str): _description_
        task_name (str): the targeted task
            ("energy", "forces")
        crystal_task (bool): whether we have a crystal (i.e. a unit cell)
            or a molecule

    Returns:
        (dict): dict of metrics to measure RI difference in
            energy/force pred. or pos. between G and rotated G
    """
    model.eval()

    energy_diff = torch.zeros(1, device=device)
    energy_diff_percentage = torch.zeros(1, device=device)
    energy_diff_refl = torch.zeros(1, device=device)
    pos_diff_total = torch.zeros(1, device=device)
    forces_diff = torch.zeros(1, device=device)
    forces_diff_graph = torch.zeros(1, device=device)
    forces_diff_refl = torch.zeros(1, device=device)
    n_batches = 0
    n_atoms = 0
    rotation = "z" if fa == "2D" else None

    for batch in loader:
        n_batches += len(batch.natoms)
        n_atoms += batch.natoms.sum()

        # Compute model prediction
        preds1 = model_forward(
            deepcopy(batch), model, fa, mode="inference", crystal_task=crystal_task
        )

        # Compute prediction on rotated graph
        rotated = rotate_graph(batch, fa, fa_frames, rotation=rotation)
        preds2 = model_forward(
            deepcopy(rotated["batch_list"]),
            model,
            fa,
            mode="inference",
            crystal_task=crystal_task,
        )

        # Difference in predictions, for energy
        energy_diff += torch.abs(preds1["energy"] - preds2["energy"]).sum()
        if task_name == "forces":
            energy_diff_percentage += (
                torch.abs(preds1["energy"] - preds2["energy"])
                / torch.abs(batch.y).to(preds1["energy"].device)
            ).sum()
            forces_diff += torch.abs(
                preds1["forces"] @ rotated["rot"].to(preds1["forces"].device)
                - preds2["forces"]
            ).sum()
            assert torch.allclose(
                torch.abs(
                    batch.force @ rotated["rot"].to(batch.force.device)
                    - rotated["batch_list"].force
                ).sum(),
                torch.tensor([0.0]),
                atol=1e-05,
            )
        elif task_name == "energy":
            energy_diff_percentage += (
                torch.abs(preds1["energy"] - preds2["energy"])
                / torch.abs(batch.y_relaxed).to(preds1["energy"].device)
            ).sum()
        else:
            energy_diff_percentage += (
                torch.abs(preds1["energy"] - preds2["energy"])
                / torch.abs(batch.y).to(preds1["energy"].device)
            ).sum()

        # Difference in positions to check that Frame Averaging works as expected
        pos_diff = -1
        if hasattr(batch, "fa_pos"):
            pos_diff = 0
            for pos1, pos2 in zip(batch.fa_pos, rotated["batch_list"].fa_pos):
                pos_diff += pos1 - pos2
            pos_diff_total += torch.abs(pos_diff).sum()

        # Reflect graph and compute diff in prediction
        reflected = reflect_graph(batch, fa, fa_frames)
        preds3 = model_forward(
            deepcopy(reflected["batch_list"]),
            model,
            fa,
            mode="inference",
            crystal_task=crystal_task,
        )
        energy_diff_refl += torch.abs(preds1["energy"] - preds3["energy"]).sum()
        if task_name == "forces":
            forces_diff_refl += torch.abs(
                preds1["forces"] @ reflected["rot"].to(preds1["forces"].device)
                - preds3["forces"]
            ).sum()

    # Aggregate the results
    energy_diff = energy_diff / n_batches
    energy_diff_percentage = energy_diff_percentage / n_batches
    energy_diff_refl = energy_diff_refl / n_batches
    pos_diff_total = pos_diff_total / n_batches

    symmetry = {
        "Rot-I": float(energy_diff),
        "Perc-diff": float(energy_diff_percentage),
        "Pos": float(pos_diff_total),
        "Refl-I": float(energy_diff_refl),
    }

    # Test equivariance of forces
    if task_name == "forces":
        forces_diff = forces_diff / n_atoms
        forces_diff_graph = forces_diff / n_batches
        forces_diff_refl = forces_diff_refl / n_atoms
        symmetry.update(
            {
                "F-RE-graph": float(forces_diff_graph),
                "F-Rot-E": float(forces_diff),
                "F-Refl-E": float(forces_diff_refl),
            }
        )

    return symmetry
