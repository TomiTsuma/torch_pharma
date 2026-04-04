import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import wandb
from typing import Any, Dict, List, Optional, Union, Tuple
from matplotlib.axes import Axes
from wandb.sdk.wandb_run import Run
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

from torch_pharma.utils.logging import get_pylogger
log = get_pylogger(__name__)


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

            from torch_pharma.data.components.edm import get_bond_order
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

@typechecked
def parse_ms_value(ms: Any) -> Tuple[float, float]:
    """Parse the MS value and its error, return as a tuple."""
    if isinstance(ms, (float, int)) or ms == "N/A":
        return (float(ms) if ms != "N/A" else np.nan, 0.0)
    if not isinstance(ms, str):
        return (float(ms), 0.0)
    ms_parts = ms.split("±")
    value = float(ms_parts[0].strip())
    error = float(ms_parts[1].strip()) if len(ms_parts) > 1 else 0.0
    return (value, error)


@typechecked
def format_ms_annotation(value: Any, error: float) -> str:
    """Format the MS annotation based on value and error."""
    if value == "N/A" or np.isnan(value):
        return "N/A"
    lower = value - error
    upper = value + error
    return r"$MS \in " + f"[{lower:.1f}\%, {upper:.1f}\%]" + "$"


@typechecked
def plot_property_optimization(
    data: Dict[str, Any],
    save_path: str = "property_optimization_results.png",
    title: str = "Property Optimization Results"
):
    """
    Plot property optimization results (MAE and Molecule Stability).
    Adapted from bio-diffusion's optimization_analysis.py.
    """
    data_groups = {}
    for k, v in data.items():
        values, errors, ms_values = {}, {}, {}
        for prop in v:
            raw_value_str = v[prop]["value"].split("±")[0].strip()
            raw_value = float(raw_value_str)
            ms_value, ms_error = parse_ms_value(v[prop]["MS"])
            
            # Filtering heuristic from original code
            if raw_value > 50:
                values[prop], errors[prop], ms_values[prop] = np.nan, 0.0, ("N/A", "N/A")
            else:
                values[prop] = raw_value
                errors[prop] = float(v[prop]["value"].split("±")[1].strip())
                ms_values[prop] = (ms_value, ms_error)
        data_groups[k] = {"values": values, "errors": errors, "MS": ms_values}

    x_labels = list(next(iter(data_groups.values()))["values"].keys())

    fig, ax = plt.subplots(figsize=(10, 8))
    width = 0.15
    group_gap = 0.5
    n_groups = len(data_groups)
    total_width = n_groups * width + (n_groups - 1) * group_gap
    positions = np.arange(len(x_labels)) * (total_width + group_gap)

    for i, (group, group_data) in enumerate(data_groups.items()):
        vals = list(group_data["values"].values())
        errs = list(group_data["errors"].values())
        ms_vals = list(group_data["MS"].values())
        bar_positions = [pos + i * (width + group_gap) for pos in positions]
        bars = ax.barh(bar_positions, vals, width, label=group, xerr=errs, capsize=2, alpha=0.8, edgecolor="black")

        for j, val in enumerate(vals):
            if np.isnan(val):
                ax.text(0, bar_positions[j], "x", color="red", va="center", ha="center", fontsize=12, weight="bold")

        for bar, (ms, error) in zip(bars, ms_vals):
            if not isinstance(ms, str) or ms != "N/A":
                ms_annotation = format_ms_annotation(ms, error)
                ax.annotate(ms_annotation, (bar.get_width(), bar.get_y() + bar.get_height() / 2 + 0.35),
                            textcoords="offset points", xytext=(5, 0), ha="left",
                            fontsize=8, color="darkblue", weight="black")

    ax.set_ylabel("Task")
    ax.set_xlabel("Property MAE / Molecule Stability (MS) %")
    ax.set_yticks([pos + total_width / 2 - width / 2 for pos in positions])
    ax.set_yticklabels(x_labels, rotation=45, va="center")
    ax.set_title(title)
    ax.grid(True, which='both', axis='x', linestyle='-.', linewidth=0.5)

    for pos in positions[1:]:
        ax.axhline(y=pos - group_gap / 2, color="black", linewidth=2)

    ax.legend(loc="best")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    log.info(f"Property optimization plot saved to {save_path}")
