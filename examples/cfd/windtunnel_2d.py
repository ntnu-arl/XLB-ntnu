import xlb
import time
import numpy as np
import warp as wp
import jax.numpy as jnp

from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.velocity_set import D2Q9
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    HalfwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
    FullwayBounceBackBC,
)

from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_image, save_fields_vtk, show_image

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath

# -------------------------- Helper Functions --------------------------


def build_obstacle_indices(
    grid_shape,
    chord_fraction=0.20,
    thickness=0.12,
    camber=0.00,
    camber_position=0.4,
    angle_deg=-10.0,
    x_position=0.25,
    y_position=0.50,
    naca_points=400,
):
    """Build a rasterized NACA 4-digit style airfoil obstacle on the grid.

    Parameters are nondimensional except angle in degrees:
    - chord_fraction: fraction of domain length used as chord
    - thickness: max thickness / chord (e.g. 0.12 for NACA 0012-like thickness)
    - camber: max camber / chord
    - camber_position: location of max camber along chord (0..1)
    - x_position: leading-edge x-position as fraction of domain length
    - y_position: vertical center position as fraction of domain height

    Uses a closed trailing edge thickness coefficient so the airfoil ends sharp.
    """
    nx, ny = grid_shape
    chord = max(10, int(chord_fraction * nx))

    # Build a NACA 4-digit airfoil contour in normalized chord coordinates.
    x = np.linspace(0.0, 1.0, naca_points)
    yt = (
        5.0
        * thickness
        * (
            0.2969 * np.sqrt(np.clip(x, 1.0e-12, 1.0))
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1036 * x**4
        )
    )

    m = camber
    p = np.clip(camber_position, 1.0e-6, 1.0 - 1.0e-6)
    yc = np.where(
        x < p,
        m / (p**2) * (2.0 * p * x - x**2),
        m / ((1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
    )
    dyc_dx = np.where(
        x < p,
        2.0 * m / (p**2) * (p - x),
        2.0 * m / ((1.0 - p) ** 2) * (p - x),
    )
    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Closed polygon: upper surface (LE->TE), then lower (TE->LE).
    x_poly = np.concatenate([xu, xl[::-1]])
    y_poly = np.concatenate([yu, yl[::-1]])

    # Scale to cells, rotate, and translate into the domain.
    x_poly = x_poly * chord
    y_poly = y_poly * chord

    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    x_rot = c * x_poly - s * y_poly
    y_rot = s * x_poly + c * y_poly

    x_le = x_position * nx
    y_ref = y_position * ny
    x_airfoil = x_rot + x_le
    y_airfoil = y_rot + y_ref

    airfoil_poly = np.column_stack([x_airfoil, y_airfoil])
    path = MplPath(airfoil_poly)

    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cell_centers = np.column_stack([(xx.ravel() + 0.5), (yy.ravel() + 0.5)])
    inside = np.asarray(path.contains_points(cell_centers)).reshape(nx, ny)

    indices = np.stack(np.where(inside), axis=0).tolist()
    return indices


def plot_simulation_setup(
    grid_shape,
    voxel_size,
    bc_mask,
    wall_id,
    inlet_id,
    outlet_id,
    obstacle_id,
    boundary_linewidth=10,
):
    cmap = ListedColormap(
        [
            "#f2f2f2",  # fluid
            "#2b6cb0",  # walls
            "#38a169",  # inlet
            "#dd6b20",  # outlet
            "#c53030",  # obstacle
        ]
    )
    if isinstance(bc_mask, jnp.ndarray):
        bc_mask_np = np.array(bc_mask)
    else:
        bc_mask_np = np.array(wp.to_jax(bc_mask))

    # bc_mask is [1, nx, ny, nz] in this setup; use the first component/layer.
    mask_2d = bc_mask_np[0, :, :, 0]

    # Convert BC ids to compact plotting classes:
    # 0=fluid, 1=wall, 2=inlet, 3=outlet, 4=obstacle
    plot_mask = np.zeros_like(mask_2d, dtype=np.uint8)
    plot_mask[mask_2d == wall_id] = 1
    plot_mask[mask_2d == inlet_id] = 2
    plot_mask[mask_2d == outlet_id] = 3
    plot_mask[mask_2d == obstacle_id] = 4
    plot_mask[mask_2d == 255] = 5

    fig, ax = plt.subplots(figsize=(12, 4.5))
    extent = [0.0, grid_shape[0] * voxel_size, 0.0, grid_shape[1] * voxel_size]
    ax.imshow(
        plot_mask.T,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        extent=extent,
        aspect="equal",
    )

    # Add thick outlines so boundary-condition regions are easier to see.
    boundary_layers = [
        (1, "#2b6cb0"),  # wall
        (2, "#38a169"),  # inlet
        (3, "#dd6b20"),  # outlet
    ]
    for boundary_class, color in boundary_layers:
        boundary_binary = (plot_mask == boundary_class).astype(np.float32)
        if np.any(boundary_binary):
            ax.contour(
                boundary_binary.T,
                levels=[0.5],
                colors=[color],
                linewidths=boundary_linewidth,
                origin="lower",
                extent=extent,
            )

    ax.set_title("2D Wind Tunnel Setup")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, grid_shape[0] * voxel_size)
    ax.set_ylim(0, grid_shape[1] * voxel_size)

    legend_handles = [
        Patch(facecolor="#f2f2f2", edgecolor="none", label="Fluid"),
        Patch(facecolor="#2b6cb0", edgecolor="none", label="Wall"),
        Patch(facecolor="#38a169", edgecolor="none", label="Inlet"),
        Patch(facecolor="#dd6b20", edgecolor="none", label="Outlet"),
        Patch(facecolor="#c53030", edgecolor="none", label="Obstacle"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    ax.annotate(
        "",
        xy=(0.92, 0.95),
        xytext=(0.08, 0.95),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2.0, color="black"),
    )
    ax.text(
        0.50,
        0.98,
        "Flow direction",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)


def post_process(
    step,
    f_0,
    f_1,
    grid_shape,
    macro,
    momentum_transfer,
    missing_mask,
    bc_mask,
    wind_speed,
    field_prefix="out/windtunnel2d",
):
    # Convert to JAX array if necessary
    if not isinstance(f_0, jnp.ndarray):
        f_0_jax = wp.to_jax(f_0)
    else:
        f_0_jax = f_0

    rho, u = macro(f_0_jax)

    rho = rho[:, 1:-1, 1:-1, 0]
    u = u[:, 1:-1, 1:-1, 0]
    u_magnitude = jnp.sqrt(u[0] ** 2 + u[1] ** 2) / voxel_size
    p_static = 101325 - 0.5 * rho * u_magnitude**2

    fields = {
        "rho": rho[0],
        "u_x": u[0],
        "u_y": u[1],
        "u_magnitude": u_magnitude,
        "p_static": p_static,
    }
    # save_fields_vtk(fields, timestep=step, prefix=field_prefix)
    # save_image(fields["u_magnitude"], timestep=step, prefix=field_prefix)
    show_image(
        fields["u_magnitude"],
        timestep=step,
        prefix=field_prefix,
        vmin=0,
        vmax=wind_speed * 2,
    )

    boundary_force = momentum_transfer(f_0, f_1, bc_mask, missing_mask)
    drag = boundary_force[0]  # x-direction
    lift = boundary_force[1]  # y-direction


# Simulation Configuration
grid_shape = (1000, 600)
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32

velocity_set = D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

voxel_size = 0.002
wind_speed = 15.0
Re = 80000.0
num_steps = 500000
print_interval = 10000
output_interval = 1000

# Airfoil obstacle parameters (NACA 4-digit style)
airfoil_chord_fraction = 0.25
airfoil_thickness = 0.1
airfoil_camber = 0.1
airfoil_camber_position = 0.40
airfoil_angle_deg = -10.0
airfoil_x_position = 0.25
airfoil_y_position = 0.50

clength = (
    airfoil_chord_fraction * grid_shape[0] * voxel_size
)  # Characteristic length (in physical units, e.g., m)
visc = wind_speed * clength / Re
omega = 1.0 / (3.0 * visc + 0.5)

# Print simulation info
print("\n" + "=" * 50 + "\n")
print("Simulation Configuration:")
print(f"Grid size: {grid_shape[0]} x {grid_shape[1]}")
print(f"Grid dimensions: {grid_shape[0]*voxel_size} x {grid_shape[1]*voxel_size}")
print(f"Number of cells: {grid_shape[0]*grid_shape[1]}")
print(f"Backend: {compute_backend}")
print(f"Velocity set: {velocity_set}")
print(f"Precision policy: {precision_policy}")
print(f"Prescribed velocity: {wind_speed}")
print(f"Reynolds number: {Re}")
print(f"Relaxation factor: {omega}")
print(f"Max iterations: {num_steps}")
print("\n" + "=" * 50 + "\n")

xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

grid = grid_factory(grid_shape, compute_backend=compute_backend)

box = grid.bounding_box_indices()
box_no_edge = grid.bounding_box_indices(remove_edges=True)
inlet = box_no_edge["left"]
outlet = box_no_edge["right"]
walls = [box["bottom"][i] + box["top"][i] for i in range(velocity_set.d)]
walls = np.unique(np.array(walls), axis=-1).tolist()

obstacle_indices = build_obstacle_indices(
    grid_shape,
    chord_fraction=airfoil_chord_fraction,
    thickness=airfoil_thickness,
    camber=airfoil_camber,
    camber_position=airfoil_camber_position,
    angle_deg=airfoil_angle_deg,
    x_position=airfoil_x_position,
    y_position=airfoil_y_position,
)

bc_inlet = RegularizedBC(
    "velocity",
    prescribed_value=(wind_speed * voxel_size, 0.0),
    indices=inlet,
)
bc_outlet = ExtrapolationOutflowBC(indices=outlet)
bc_walls = FullwayBounceBackBC(indices=walls)
bc_obstacle = HalfwayBounceBackBC(indices=obstacle_indices)

boundary_conditions = [bc_walls, bc_inlet, bc_outlet, bc_obstacle]

stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
)

f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

plot_simulation_setup(
    grid_shape=grid_shape,
    voxel_size=voxel_size,
    bc_mask=bc_mask,
    wall_id=bc_walls.id,
    inlet_id=bc_inlet.id,
    outlet_id=bc_outlet.id,
    obstacle_id=bc_obstacle.id,
)

# Setup Momentum Transfer for Force Calculation
bc_obstacle = boundary_conditions[-1]
momentum_transfer = MomentumTransfer(bc_obstacle, compute_backend=compute_backend)

macro = Macroscopic(
    compute_backend=ComputeBackend.JAX,
    precision_policy=precision_policy,
    velocity_set=D2Q9(
        precision_policy=precision_policy, compute_backend=ComputeBackend.JAX
    ),
)

start_time = time.time()
for step in range(num_steps):
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0

    if step % print_interval == 0:
        if compute_backend == ComputeBackend.WARP:
            wp.synchronize()
        elapsed_time = time.time() - start_time
        print(f"Iteration: {step}/{num_steps} | Time elapsed: {elapsed_time:.2f}s")
        start_time = time.time()
    if step % output_interval == 0:
        post_process(
            step,
            f_0,
            f_1,
            grid_shape,
            macro,
            momentum_transfer,
            missing_mask,
            bc_mask,
            wind_speed,
        )


print("Simulation completed successfully.")
