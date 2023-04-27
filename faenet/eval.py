import torch
from copy import deepcopy
from faenet import model_forward
from faenet.utils import RandomRotate, RandomReflect
from faenet import FrameAveraging
from torch_geometric.data import Batch


def transform_batch(batch, frame_averaging, fa_method, neighbors=None):
    r"""Apply a transformation to a batch of graphs

    Args:
        batch (data.Batch): batch of data.Data objects.
        frame_averaging (str): Transform method used.
        fa_method (str): FA method used.
        neighbors (list, optional): list containing the number of edges
            in each graph of the batch. (default: :obj:`None`)

    Returns:
        (data.Batch): transformed batch sample
    """
    delattr(batch, "fa_pos")  # delete it otherwise can't iterate
    delattr(batch, "fa_cell")  # delete it otherwise can't iterate
    delattr(batch, "fa_rot")  # delete it otherwise can't iterate

    # Convert batch to list of graphs
    g_list = batch.to_data_list()

    # Apply transform to individual graphs of batch
    fa_transform = FrameAveraging(frame_averaging, fa_method)
    for g in g_list:
        g = fa_transform(g)
    batch = Batch.from_data_list(g_list)
    if neighbors is not None:
        batch.neighbors = neighbors
    return batch


def rotate_graph(batch, frame_averaging, fa_method, rotation=None):
    r"""Rotate all graphs in a batch

    Args:
        batch (data.Batch): batch of graphs.
        frame_averaging (str): Transform method used.
            ("2D", "3D", "DA", "")
        fa_method (str): FA method used.
            ("", "stochastic", "all", "det", "se3-stochastic", "se3-all", "se3-det")
        rotation (str, optional): type of rotation applied. (default: :obj:`None`)
            ("z", "x", "y", None)

    Returns:
        (dict): rotated batch sample and rotation matrix used to rotate it
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
        batch_rotated = transform_batch(
            batch_rotated,
            frame_averaging,
            fa_method,
            batch.neighbors if hasattr(batch, "neighbors") else None,
        )

    return {"batch": batch_rotated, "rot": rot}


def reflect_graph(batch, frame_averaging, fa_method, reflection=None):
    r"""Rotate all graphs in a batch

    Args:
        batch (data.Batch): batch of graphs
        frame_averaging (str): Transform method used
            ("2D", "3D", "DA", "")
        fa_method (str): FA method used
            ("", "stochastic", "all", "det", "se3-stochastic", "se3-all", "se3-det")
        reflection (str, optional): type of reflection applied. (default: :obj:`None`)

    Returns:
        (dict): reflected batch sample and rotation matrix used to reflect it
    """
    if isinstance(batch, list):
        batch = batch[0]

    # Sampling a random rotation within [-180, 180] for all axes.
    transform = RandomReflect()

    # Reflect batch
    batch_reflected, rot, inv_rot = transform(deepcopy(batch))
    assert not torch.allclose(batch.pos, batch_reflected.pos, atol=1e-05)

    if hasattr(batch, "fa_pos"):
        batch_reflected = transform_batch(
            batch_reflected,
            frame_averaging,
            fa_method,
            batch.neighbors if hasattr(batch, "neighbors") else None,
        )
    return {"batch": batch_reflected, "rot": rot}


@torch.no_grad()
def eval_model_symmetries(
    loader, model, frame_averaging, fa_method, device, task_name, crystal_task=True
):
    """Test rotation and reflection invariance & equivariance of GNNs

    Args:
        loader (data): dataloader
        model: model instance
        frame_averaging (str): frame averaging ("2D", "3D"), data augmentation ("DA")
            or none ("")
        fa_method (str): _description_
        task_name (str): the targeted task
            ("energy", "forces")
        crystal_task (bool): whether we have a crystal (i.e. a unit cell)
            or a molecule

    Returns:
        (dict): metrics measuring invariance/equivariance
            of energy/force predictions
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
    rotation = "z" if frame_averaging == "2D" else None

    for batch in loader:
        n_batches += len(batch.natoms)
        n_atoms += batch.natoms.sum()

        # Compute model prediction
        preds1 = model_forward(
            deepcopy(batch),
            model,
            frame_averaging,
            mode="inference",
            crystal_task=crystal_task,
        )

        # Compute prediction on rotated graph
        rotated = rotate_graph(batch, frame_averaging, fa_method, rotation=rotation)
        preds2 = model_forward(
            deepcopy(rotated["batch"]),
            model,
            frame_averaging,
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
                    - rotated["batch"].force
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
            for pos1, pos2 in zip(batch.fa_pos, rotated["batch"].fa_pos):
                pos_diff += pos1 - pos2
            pos_diff_total += torch.abs(pos_diff).sum()

        # Reflect graph and compute diff in prediction
        reflected = reflect_graph(batch, frame_averaging, fa_method)
        preds3 = model_forward(
            deepcopy(reflected["batch"]),
            model,
            frame_averaging,
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
