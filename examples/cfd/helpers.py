from xlb.utils import save_image, save_fields_vtk, show_image
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
import numpy as np
import jax.numpy as jnp
import warp as wp

@dataclass
class LBUnitConverter:
    """Convert between SI units and lattice units for a D2Q9 simulation."""

    grid_shape: tuple
    domain_size_m: tuple
    wind_speed_mps: float
    chord_fraction: float
    reynolds_number: Optional[float] = None
    nu_m2ps: Optional[float] = None
    target_u_lb: float = 0.05
    rho_ref_kgm3: float = 1.225

    def __post_init__(self):
        nx, ny = self.grid_shape
        lx_m, ly_m = self.domain_size_m

        self.dx = lx_m / nx
        self.dy = ly_m / ny
        if not np.isclose(self.dx, self.dy):
            raise ValueError("Non-square cells detected. Please use isotropic spacing.")

        self.chord_m = self.chord_fraction * lx_m

        if self.nu_m2ps is None:
            if self.reynolds_number is None:
                raise ValueError(
                    "Provide either nu_m2ps or reynolds_number for physical scaling."
                )
            self.nu_m2ps = self.wind_speed_mps * self.chord_m / self.reynolds_number

        # Choose dt such that inlet speed in lattice units stays low (low Mach number).
        self.dt = self.target_u_lb * self.dx / self.wind_speed_mps

        self.u_lb_inlet = self.to_lattice_velocity(self.wind_speed_mps)
        self.nu_lb = self.nu_m2ps * self.dt / (self.dx**2)
        self.tau = 3.0 * self.nu_lb + 0.5
        self.omega = 1.0 / self.tau
        self.ma = self.u_lb_inlet * np.sqrt(3.0)

        self.vel_scale = self.dx / self.dt
        self.pressure_scale = self.rho_ref_kgm3 * (self.vel_scale**2)
        # D2Q9 is a 2D simulation; momentum-transfer force is force per unit depth [N/m].
        self.force_scale_per_depth = self.rho_ref_kgm3 * (self.dx**3) / (self.dt**2)

    def to_lattice_velocity(self, u_mps):
        return u_mps * self.dt / self.dx

    def to_si_velocity(self, u_lb):
        return u_lb * self.vel_scale

    def to_si_time(self, step):
        return step * self.dt

    def pressure_fluctuation_to_si(self, rho_lb, rho_lb0=1.0):
        # For weakly compressible LBM: p' = cs^2 * (rho-rho0), with cs^2=1/3 in lattice units.
        return (1.0 / 3.0) * (rho_lb - rho_lb0) * self.pressure_scale

    def to_si_force_per_depth(self, force_lb):
        """Convert lattice force to SI force per unit depth [N/m] for 2D simulations."""
        return force_lb * self.force_scale_per_depth

    def to_si_force(self, force_lb, span_m=1.0):
        """Convert lattice force to total SI force [N] using an assumed span/depth."""
        return self.to_si_force_per_depth(force_lb) * span_m


# -------------------------- Helper Functions --------------------------

def build_airfoil_indices(
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
    extent = (0.0, grid_shape[0] * voxel_size, 0.0, grid_shape[1] * voxel_size)
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
    macro,
    units,
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
    u_magnitude_lb = jnp.sqrt(u[0] ** 2 + u[1] ** 2)
    u_magnitude = units.to_si_velocity(u_magnitude_lb)
    q_dynamic = 0.5 * units.rho_ref_kgm3 * u_magnitude**2
    p_fluctuation = units.pressure_fluctuation_to_si(rho[0])

    fields = {
        "rho": rho[0],
        "u_x": u[0],
        "u_y": u[1],
        "u_magnitude": u_magnitude,
        "q_dynamic": q_dynamic,
        "p_fluctuation": p_fluctuation,
    }
    # save_fields_vtk(fields, timestep=step, prefix=field_prefix)
    # save_image(fields["u_magnitude"], timestep=step, prefix=field_prefix)
    show_image(
        fields["u_magnitude"],
        timestep=step,
        prefix=field_prefix,
        vmin=0,
        vmax=units.wind_speed_mps * 2,
    )

def plot_drag_coefficient(time_steps, drag_coefficients, lift_coefficients):
    """
    Plot the drag coefficient with various moving averages.

    Args:
        time_steps (list): List of time steps.
        drag_coefficients (list): List of drag coefficients.
    """
    # Convert lists to numpy arrays for processing
    time_steps_np = np.array(time_steps)
    drag_coefficients_np = np.array(drag_coefficients)
    lift_coefficients_np = np.array(lift_coefficients)

    # Define moving average windows
    windows = [10, 100, 1000, 10000, 100000]
    labels = ["MA 10", "MA 100", "MA 1,000", "MA 10,000", "MA 100,000"]

    plt.figure(figsize=(12, 8))
    plt.plot(
        time_steps_np, drag_coefficients_np, label="Raw Drag Coefficient", alpha=0.5
    )
    plt.plot(
        time_steps_np, lift_coefficients_np, label="Raw Lift Coefficient", alpha=0.5
    )

    for window, label in zip(windows, labels):
        if len(drag_coefficients_np) >= window:
            ma = np.convolve(
                drag_coefficients_np, np.ones(window) / window, mode="valid"
            )
            plt.plot(time_steps_np[window - 1 :], ma, label=label)

    plt.ylim(-2.0, 2.0)
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Drag coefficient")
    plt.title("Drag Coefficient Over Time with Moving Averages")
    plt.savefig("out/drag_coefficient_ma.png")
    plt.close()

def compute_forces(step, f_0, f_1, bc_mask, missing_mask, momentum_transfer, units, airfoil_chord_length):
    boundary_force = momentum_transfer(f_0, f_1, bc_mask, missing_mask)
    drag_lb = boundary_force[0]  # x-direction (lattice force)
    lift_lb = boundary_force[1]  # y-direction (lattice force)

    # In 2D, convert to force per unit depth [N/m].
    drag_n_per_m = units.to_si_force_per_depth(drag_lb)
    lift_n_per_m = units.to_si_force_per_depth(lift_lb)

    q_inf = 0.5 * units.rho_ref_kgm3 * units.wind_speed_mps**2
    cd = drag_n_per_m / (q_inf * airfoil_chord_length)
    cl = lift_n_per_m / (q_inf * airfoil_chord_length)
    print(
        f"Step {step}: Drag = {drag_n_per_m:.6f} N/m, Lift = {lift_n_per_m:.6f} N/m, "
        f"Cd = {cd:.4f}, Cl = {cl:.4f}"
    )
    return cd, cl
