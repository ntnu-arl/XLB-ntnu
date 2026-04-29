import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

from xlb.compute_backend import ComputeBackend


def _normalize_boundary_layer_voxels(boundary_layer_voxels):
    if boundary_layer_voxels is None:
        return np.empty((0, 7), dtype=np.float32)

    array = np.asarray(boundary_layer_voxels, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError(
            "boundary_layer_voxels must be shaped like [N, >=5] with columns [x, y, x/c, y/c, side, (optional n_x, n_y)]."
        )
    return array


def visualize_boundary_voxel_normals(
    boundary_layer_voxels,
    obstacle_indices=None,
    sample_stride=1,
    normal_scale=3.0,
    title="Boundary voxel surface normals",
):
    """Visualize boundary voxels and their estimated surface normals.

    Args:
        boundary_layer_voxels: Array shaped [N, >=7] with columns
            [x, y, x/c, y/c, side, n_x, n_y].
        obstacle_indices: Optional obstacle cells as [2, N] or [N, 2].
        sample_stride: Plot every k-th normal vector for readability.
        normal_scale: Arrow length scale in grid-cell units.
        title: Figure title.

    Returns:
        (fig, ax): Matplotlib figure and axis objects.
    """
    import matplotlib.pyplot as plt

    boundary_layer_voxels = _normalize_boundary_layer_voxels(boundary_layer_voxels)
    if boundary_layer_voxels.size == 0:
        raise ValueError("No boundary-layer voxels available to visualize.")
    if boundary_layer_voxels.shape[1] < 7:
        raise ValueError(
            "boundary_layer_voxels must include normal columns [n_x, n_y] at columns 5 and 6."
        )

    sample_stride = max(1, int(sample_stride))

    voxel_x = boundary_layer_voxels[:, 0].astype(np.float32) + 0.5
    voxel_y = boundary_layer_voxels[:, 1].astype(np.float32) + 0.5
    side_flag = boundary_layer_voxels[:, 4].astype(np.float32)
    normal_x = boundary_layer_voxels[:, 5].astype(np.float32)
    normal_y = boundary_layer_voxels[:, 6].astype(np.float32)

    normal_norm = np.sqrt(normal_x**2 + normal_y**2)

    fig, ax = plt.subplots(figsize=(10, 6))

    if obstacle_indices is not None:
        obstacle_arr = np.asarray(obstacle_indices)
        if obstacle_arr.size > 0:
            if obstacle_arr.ndim != 2:
                raise ValueError(
                    "obstacle_indices must be shaped like [2, N] or [N, 2]."
                )
            if obstacle_arr.shape[0] == 2:
                obstacle_x = obstacle_arr[0].astype(np.float32) + 0.5
                obstacle_y = obstacle_arr[1].astype(np.float32) + 0.5
            elif obstacle_arr.shape[1] == 2:
                obstacle_x = obstacle_arr[:, 0].astype(np.float32) + 0.5
                obstacle_y = obstacle_arr[:, 1].astype(np.float32) + 0.5
            else:
                raise ValueError(
                    "obstacle_indices must be shaped like [2, N] or [N, 2]."
                )
            ax.scatter(
                obstacle_x,
                obstacle_y,
                s=5,
                c="#111111",
                alpha=0.25,
                linewidths=0,
                label="Obstacle cells",
            )

    upper_mask = side_flag > 0.0
    lower_mask = side_flag < 0.0

    if np.any(upper_mask):
        ax.scatter(
            voxel_x[upper_mask],
            voxel_y[upper_mask],
            s=16,
            c="#2b6cb0",
            alpha=0.9,
            linewidths=0,
            label="Boundary voxels (upper)",
        )
    if np.any(lower_mask):
        ax.scatter(
            voxel_x[lower_mask],
            voxel_y[lower_mask],
            s=16,
            c="#c53030",
            alpha=0.9,
            linewidths=0,
            label="Boundary voxels (lower)",
        )

    sample_slice = slice(None, None, sample_stride)
    ax.quiver(
        voxel_x[sample_slice],
        voxel_y[sample_slice],
        normal_x[sample_slice],
        normal_y[sample_slice],
        angles="xy",
        scale_units="xy",
        scale=1.0 / max(1.0e-6, float(normal_scale)),
        color="#2f855a",
        width=0.0035,
        alpha=0.9,
        label="Estimated outward normals",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (grid cells)")
    ax.set_ylabel("y (grid cells)")
    ax.set_title(
        f"{title} | normals: min={normal_norm.min():.3f}, max={normal_norm.max():.3f}, mean={normal_norm.mean():.3f}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def visualize_solid_and_boundary_voxels(
    solid_voxels,
    boundary_voxels,
    title="Solid and boundary voxel indices",
):
    """Visualize solid obstacle cells and boundary-adjacent cells."""

    def _as_xy(indices, name):
        array = np.asarray(indices)
        if array.size == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"{name} must be shaped like [2, N] or [N, 2].")
        if array.shape[0] == 2:
            return array[0].astype(np.float32) + 0.5, array[1].astype(np.float32) + 0.5
        if array.shape[1] == 2:
            return (
                array[:, 0].astype(np.float32) + 0.5,
                array[:, 1].astype(np.float32) + 0.5,
            )
        raise ValueError(f"{name} must be shaped like [2, N] or [N, 2].")

    solid_x, solid_y = _as_xy(solid_voxels, "solid_voxels")
    boundary_x, boundary_y = _as_xy(boundary_voxels, "boundary_voxels")

    fig, ax = plt.subplots(figsize=(10, 6))
    if solid_x.size > 0:
        ax.scatter(
            solid_x,
            solid_y,
            s=8,
            c="#1a202c",
            alpha=0.45,
            linewidths=0,
            label="Solid voxels",
        )
    if boundary_x.size > 0:
        ax.scatter(
            boundary_x,
            boundary_y,
            s=18,
            c="#d69e2e",
            alpha=0.9,
            linewidths=0,
            label="Boundary voxels",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (grid cells)")
    ax.set_ylabel("y (grid cells)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    from helpers import (
        LBUnitConverter,
        build_airfoil_indices,
    )
    from xlb.grid import grid_factory

    # Simulation parameters
    # Simulation Configuration
    grid_shape = (1200, 1200)
    compute_backend = ComputeBackend.WARP

    domain_length_m = 0.6  # Physical domain size in meters (length x height)
    Re = 100000.0
    air_kinematic_viscosity_m2ps = 1.511e-5
    sim_time = 5.0  # Total simulation time in seconds
    print_interval = 10000
    output_interval = 100
    eval_start_step = 20000

    # Airfoil obstacle parameters (NACA 4-digit style)
    airfoil_chord_length = 0.13  # m
    airfoil_thickness = 0.02 / airfoil_chord_length
    airfoil_camber = 0.05
    airfoil_camber_position = 0.40
    airfoil_angle_deg = -0.0
    airfoil_x_position = 0.05
    airfoil_y_position = 0.5

    # Create unit converter and grid
    units = LBUnitConverter(
        grid_shape=grid_shape,
        domain_length_m=domain_length_m,
        wind_speed_mps=None,
        chord_fraction=airfoil_chord_length / domain_length_m,
        reynolds_number=Re,
        nu_m2ps=air_kinematic_viscosity_m2ps,
    )
    grid = grid_factory(grid_shape, compute_backend=compute_backend)

    # Build airfoil obstacle indices
    solid_indices, boundary_indices, airfoil_indices, voxels = build_airfoil_indices(
        grid_shape=grid_shape,
        chord_fraction=airfoil_chord_length / domain_length_m,
        thickness=airfoil_thickness,
        camber=airfoil_camber,
        camber_position=airfoil_camber_position,
        angle_deg=airfoil_angle_deg,
        x_position=airfoil_x_position,
        y_position=airfoil_y_position,
    )

    # Visualize solid voxels and boundary voxels
    fig, ax = visualize_solid_and_boundary_voxels(
        solid_voxels=solid_indices,
        boundary_voxels=boundary_indices,
        title="Airfoil solid voxels and boundary voxels",
    )
    plt.show()

    # Visualize boundary voxels and normals
    fig, ax = visualize_boundary_voxel_normals(
        boundary_layer_voxels=voxels,
        obstacle_indices=airfoil_indices,
        sample_stride=1,
        normal_scale=2.0,
        title="Airfoil boundary voxels and normals",
    )
    plt.show()
