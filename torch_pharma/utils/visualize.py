import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import wandb
from typing import Any, Dict, List, Optional, Union, Tuple
from matplotlib.axes._subplots import Axes
from wandb.sdk.wandb_run import Run
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

from src.utils.pylogger import get_pylogger
log = get_pylogger(__name__)

from src.datamodules.components.edm import get_bond_order
from torch_pharma.utils.io import load_files_with_ext, load_molecule_xyz

@typechecked
def visualize_mol(
    path: str,
    dataset_info: Dict[str, Any],
    max_num: int = 25,
    wandb_run: Optional[Run] = None,
    spheres_3d: bool = False,
    mode: str = "molecule",
    verbose: bool = True
):
    files = load_files_with_ext(path, ext="xyz")[0: max_num]
    for file in files:
        positions, one_hot = load_molecule_xyz(file, dataset_info)
        atom_types = torch.argmax(one_hot, dim=-1).numpy()
        dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
        dists = dists[dists > 0]

        if verbose:
            log.info(f"Average distance between atoms: {dists.mean().item()}")

        plot_data3d(
            positions=positions,
            atom_types=atom_types,
            dataset_info=dataset_info,
            save_path=f"{file[:-4]}.png",
            spheres_3d=spheres_3d
        )

        if wandb_run is not None:
            # log image(s) via WandB
            path = f"{file[:-4]}.png"
            im = plt.imread(path)
            wandb_run.log({mode: [wandb.Image(im, caption=path)]})


@typechecked
def visualize_mol_chain(
    path: str,
    dataset_info: Dict[str, Any],
    wandb_run: Optional[Run] = None,
    spheres_3d: bool = False,
    mode: str = "chain",
    verbose: bool = True
):
    files = load_files_with_ext(path, ext="xyz")
    files = sorted(files)
    save_paths = []

    for file in files:
        positions, one_hot = load_molecule_xyz(file, dataset_info=dataset_info)

        atom_types = torch.argmax(one_hot, dim=-1).numpy()
        fn = f"{file[:-4]}.png"
        plot_data3d(
            positions=positions,
            atom_types=atom_types,
            dataset_info=dataset_info,
            save_path=fn,
            spheres_3d=spheres_3d,
            alpha=1.0
        )

        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = os.path.join(dirname, "output.gif")

    if verbose:
        log.info(f"Creating GIF with {len(imgs)} images")

    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb_run is not None:
        wandb_run.log({mode: [wandb.Video(gif_path, caption=gif_path)]})


@typechecked
def draw_sphere(
    ax: plt.axis,
    x: float,
    y: float,
    z: float,
    size: float,
    color: str,
    alpha: float
):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib.
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(
        x + xs,
        y + ys,
        z + zs,
        rstride=2,
        cstride=2,
        color=color,
        linewidth=0,
        alpha=alpha
    )


@typechecked
def plot_molecule(
    ax: Axes,
    positions: TensorType["num_nodes", 3],
    atom_types: np.ndarray,
    alpha: float,
    spheres_3d: bool,
    hex_bg_color: str,
    dataset_info: Dict[str, Any]
):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    colors_dic = np.array(dataset_info["colors_dic"])
    radius_dic = np.array(dataset_info["radius_dic"])
    area_dic = 1500 * radius_dic ** 2

    areas = area_dic[atom_types]
    radii = radius_dic[atom_types]
    colors = colors_dic[atom_types]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha,
                   c=colors)

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))

            s = (atom_types[i], atom_types[j])

            draw_edge_int = get_bond_order(
                dataset_info["atom_decoder"][s[0]],
                dataset_info["atom_decoder"][s[1]],
                dist
            )
            line_width = 2

            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    linewidth_factor = 1
                ax.plot(
                    [x[i], x[j]],
                    [y[i], y[j]],
                    [z[i], z[j]],
                    linewidth=line_width * linewidth_factor,
                    c=hex_bg_color,
                    alpha=alpha
                )


@typechecked
def plot_data3d(
    positions: TensorType["num_nodes", 3],
    atom_types: np.ndarray,
    dataset_info: Dict[str, Any],
    camera_elev: int = 0,
    camera_azim: int = 0,
    save_path: str = None,
    spheres_3d: bool = False,
    bg: str = "black",
    alpha: float = 1.0
):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = "#FFFFFF" if bg == "black" else "#666666"

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("auto")
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == "black":
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)

    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == "black":
        ax.xaxis.line.set_color("black")
    else:
        ax.xaxis.line.set_color("white")

    plot_molecule(
        ax=ax,
        positions=positions,
        atom_types=atom_types,
        alpha=alpha,
        spheres_3d=spheres_3d,
        hex_bg_color=hex_bg_color,
        dataset_info=dataset_info
    )

    max_value = positions.abs().max().item()
    axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype("uint8")
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()
