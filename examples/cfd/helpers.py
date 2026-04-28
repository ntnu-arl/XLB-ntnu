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
    chord_fraction: float
    wind_speed_mps: Optional[float] = None
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

        if self.wind_speed_mps is None:
            if self.nu_m2ps is None or self.reynolds_number is None:
                raise ValueError(
                    "When wind_speed_mps is omitted, provide both nu_m2ps and reynolds_number."
                )
            self.wind_speed_mps = self.reynolds_number * self.nu_m2ps / self.chord_m

        if self.nu_m2ps is None:
            if self.reynolds_number is None:
                raise ValueError(
                    "Provide reynolds_number to derive nu_m2ps from wind_speed_mps."
                )
            self.nu_m2ps = self.wind_speed_mps * self.chord_m / self.reynolds_number

        if self.wind_speed_mps is None:
            raise ValueError("wind_speed_mps could not be determined.")

        if self.nu_m2ps is None:
            raise ValueError("nu_m2ps could not be determined.")

        if self.wind_speed_mps <= 0.0:
            raise ValueError("wind_speed_mps must be positive.")

        if self.nu_m2ps <= 0.0:
            raise ValueError("nu_m2ps must be positive.")

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

    Returns:
    - boundary_indices: solid cells adjacent to at least one fluid cell
    - boundary_layer_indices: fluid cells adjacent to the obstacle surface
    - boundary_layer_chord_fraction: projected x/c position for boundary-layer cells
    - boundary_layer_chord_distance: signed distance to chord line y/c for boundary-layer cells
    - boundary_layer_side_flag: +1 for top side, -1 for bottom side
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
    obstacle_mask = inside

    boundary_layer_mask = np.zeros_like(obstacle_mask, dtype=bool)
    neighbor_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    obstacle_cells = np.stack(np.where(obstacle_mask), axis=1)

    for x_idx, y_idx in obstacle_cells:
        for dx, dy in neighbor_offsets:
            neighbor_x = x_idx + dx
            neighbor_y = y_idx + dy
            if not (0 <= neighbor_x < nx and 0 <= neighbor_y < ny):
                continue
            if obstacle_mask[neighbor_x, neighbor_y]:
                continue
            boundary_layer_mask[neighbor_x, neighbor_y] = True

    obstacle_indices = np.stack(np.where(obstacle_mask), axis=0).tolist()
    boundary_layer_cells = np.stack(np.where(boundary_layer_mask), axis=1)
    boundary_layer_indices = boundary_layer_cells.T.tolist()

    if boundary_layer_cells.size == 0:
        boundary_layer_chord_fraction = np.array([], dtype=np.float32)
        boundary_layer_chord_distance = np.array([], dtype=np.float32)
        boundary_layer_side_flag = np.array([], dtype=np.float32)
    else:
        chord_axis = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        chord_normal = np.array([-np.sin(angle), np.cos(angle)], dtype=np.float32)
        boundary_layer_centers = boundary_layer_cells.astype(np.float32) + 0.5
        leading_edge = np.array([x_le, y_ref], dtype=np.float32)
        relative_positions = boundary_layer_centers - leading_edge
        boundary_layer_chord_fraction = (
            relative_positions @ chord_axis / float(chord)
        ).astype(np.float32)

        local_x = (relative_positions @ chord_axis) / float(chord)
        local_y = (relative_positions @ chord_normal) / float(chord)

        upper_x = xu.astype(np.float32)
        upper_y = yu.astype(np.float32)
        lower_x = xl.astype(np.float32)
        lower_y = yl.astype(np.float32)

        upper_order = np.argsort(upper_x)
        lower_order = np.argsort(lower_x)
        upper_x = upper_x[upper_order]
        upper_y = upper_y[upper_order]
        lower_x = lower_x[lower_order]
        lower_y = lower_y[lower_order]

        upper_surface_y = np.interp(local_x, upper_x, upper_y)
        lower_surface_y = np.interp(local_x, lower_x, lower_y)
        camber_midline_y = 0.5 * (upper_surface_y + lower_surface_y)

        boundary_layer_chord_distance = local_y
        boundary_layer_side_flag = np.where(
            local_y >= camber_midline_y,
            1.0,
            -1.0,
        ).astype(np.float32)

    boundary_layer_voxels = np.hstack(
        [
            boundary_layer_cells,
            boundary_layer_chord_fraction.reshape(-1, 1),
            boundary_layer_chord_distance.reshape(-1, 1),
            boundary_layer_side_flag.reshape(-1, 1),
        ]
    )
    return (
        obstacle_indices,
        boundary_layer_indices,
        boundary_layer_voxels,
    )



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
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

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
        vmax=units.wind_speed_mps*1.5,
    )


def plot_pressure_profile(
    upper_chord_fraction: np.ndarray,
    upper_pressure_coefficient: np.ndarray,
    lower_chord_fraction: np.ndarray,
    lower_pressure_coefficient: np.ndarray,
    boundary_layer_voxels: np.ndarray,
    step: int,
    airfoil_angle_deg: Optional[float] = None,
    field_prefix: str = "out/pressure_profile",
):
    if upper_chord_fraction.size == 0 and lower_chord_fraction.size == 0:
        return

    fig, (ax_profile, ax_delta) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    upper_mask = np.isfinite(upper_chord_fraction) & np.isfinite(upper_pressure_coefficient)
    lower_mask = np.isfinite(lower_chord_fraction) & np.isfinite(lower_pressure_coefficient)
    ax_profile.plot(
        upper_chord_fraction[upper_mask],
        upper_pressure_coefficient[upper_mask],
        color="#2b6cb0",
        linewidth=1.8,
        marker="o",
        markersize=3,
        alpha=0.9,
        label="Upper surface",
    )
    ax_profile.plot(
        lower_chord_fraction[lower_mask],
        lower_pressure_coefficient[lower_mask],
        color="#c53030",
        linewidth=1.8,
        marker="o",
        markersize=3,
        alpha=0.9,
        label="Lower surface",
    )
    # Plot boundary-layer samples on a secondary y-axis for separate scaling.
    ax_secondary = ax_profile.twinx()
    ax_secondary.plot(
        boundary_layer_voxels[:, 2],
        -boundary_layer_voxels[:, 3]*1.0,  # Scale distance for visibility
        color="gray",
        linestyle="",
        marker="o",
        markersize=2,
        alpha=0.5,
        label="Boundary layer samples",
    )
    ax_secondary.set_ylabel("Boundary layer distance (scaled)")
    ax_secondary.invert_yaxis()
    ax_secondary.set_ylim(-0.3, 0.3)
    ax_profile.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_profile.invert_yaxis()
    ax_profile.set_ylabel("Pressure coefficient $C_p$")
    ax_profile.set_title(f"Airfoil pressure profile (step {step})")
    ax_profile.grid(True, alpha=0.25)
    # Combine legends from primary and secondary axes (if present).
    handles1, labels1 = ax_profile.get_legend_handles_labels()
    handles2, labels2 = ([], [])
    try:
        handles2, labels2 = ax_secondary.get_legend_handles_labels()
    except NameError:
        pass
    if handles2:
        ax_profile.legend(handles1 + handles2, labels1 + labels2, loc="best")
    else:
        ax_profile.legend(handles1, labels1, loc="best")

    if np.count_nonzero(upper_mask) > 1 and np.count_nonzero(lower_mask) > 1:
        xu = upper_chord_fraction[upper_mask]
        cpu = upper_pressure_coefficient[upper_mask]
        xl = lower_chord_fraction[lower_mask]
        cpl = lower_pressure_coefficient[lower_mask]

        upper_order = np.argsort(xu)
        lower_order = np.argsort(xl)
        xu = xu[upper_order]
        cpu = cpu[upper_order]
        xl = xl[lower_order]
        cpl = cpl[lower_order]

        xl_unique, xl_unique_idx = np.unique(xl, return_index=True)
        cpl_unique = cpl[xl_unique_idx]

        overlap_min = max(xu.min(), xl_unique.min())
        overlap_max = min(xu.max(), xl_unique.max())
        overlap_mask = (xu >= overlap_min) & (xu <= overlap_max)
        if np.any(overlap_mask):
            xu_overlap = xu[overlap_mask]
            cpl_interp = np.interp(xu_overlap, xl_unique, cpl_unique)
            pressure_difference = cpl_interp - cpu[overlap_mask]
            ax_delta.plot(
                xu_overlap,
                pressure_difference,
                color="#2f855a",
                linewidth=1.8,
                marker="o",
                markersize=3,
                alpha=0.9,
            )

    ax_delta.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_delta.set_xlabel("Chord fraction, x/c")
    ax_delta.set_ylabel(r"$\Delta C_p = C_{p,lower} - C_{p,upper}$")
    ax_delta.grid(True, alpha=0.25)
    if airfoil_angle_deg is not None:
        ax_profile.set_title(
            f"Airfoil pressure profile (step {step}, angle {airfoil_angle_deg:.1f}°)"
        )
    else:
        ax_profile.set_title(f"Airfoil pressure profile (step {step})")
    fig.tight_layout()
    fig.savefig(f"{field_prefix}_{step:06d}.png", dpi=200)
    plt.close(fig)


def capture_pressure_profile(
    step,
    f_0,
    macro,
    units,
    boundary_layer_voxels,
    airfoil_angle_deg,
    field_prefix="out/pressure_profile",
):
    if not isinstance(f_0, jnp.ndarray):
        f_0_jax = wp.to_jax(f_0)
    else:
        f_0_jax = f_0

    rho, _ = macro(f_0_jax)
    rho = rho[:, 1:-1, 1:-1, 0]
    rho_mean = rho.mean()
    pressure = units.pressure_fluctuation_to_si(rho[0])-units.pressure_fluctuation_to_si(rho_mean)  # Subtract reference pressure to get actual pressure distribution.

    q_inf = 0.5 * units.rho_ref_kgm3 * units.wind_speed_mps**2
    pressure_coefficient = np.asarray(pressure / q_inf)

    boundary_layer_voxels = np.asarray(boundary_layer_voxels, dtype=np.float32)
    if boundary_layer_voxels.ndim != 2 or boundary_layer_voxels.shape[1] < 5:
        raise ValueError(
            "boundary_layer_voxels must be shaped like [N, 5] with columns [x, y, x/c, y/c, side]."
        )

    if boundary_layer_voxels.size == 0:
        return None

    sample_x = boundary_layer_voxels[:, 0].astype(np.int32)
    sample_y = boundary_layer_voxels[:, 1].astype(np.int32)
    chord_fraction = boundary_layer_voxels[:, 2].astype(np.float32)
    side_flag = boundary_layer_voxels[:, 4].astype(np.float32)

    sample_values = pressure_coefficient[sample_x, sample_y]

    finite_chord_mask = np.isfinite(chord_fraction)
    if not np.any(finite_chord_mask):
        return None

    upper_mask = side_flag > 0.0
    lower_mask = side_flag < 0.0

    upper_x = chord_fraction[upper_mask]
    upper_cp = sample_values[upper_mask]
    lower_x = chord_fraction[lower_mask]
    lower_cp = sample_values[lower_mask]

    if upper_x.size == 0 or lower_x.size == 0:
        return None

    upper_order = np.argsort(upper_x)
    lower_order = np.argsort(lower_x)

    upper_x = upper_x[upper_order]
    upper_cp = upper_cp[upper_order]
    lower_x = lower_x[lower_order]
    lower_cp = lower_cp[lower_order]

    plot_pressure_profile(
        upper_x,
        upper_cp,
        lower_x,
        lower_cp,
        boundary_layer_voxels,
        step=step,
        airfoil_angle_deg=airfoil_angle_deg,
        field_prefix=field_prefix,
    )

    return upper_x, upper_cp, lower_x, lower_cp

def plot_drag_coefficient(
    current_step, drag_coefficients: np.ndarray, lift_coefficients: np.ndarray
):
    """
    Plot drag/lift coefficients after batch-averaging samples.

    Args:
        current_step (int): Current timestep used for plot context.
        drag_coefficients (np.array): List of drag coefficients.
        lift_coefficients (np.array): List of lift coefficients.
    """
    # Convert lists to numpy arrays for processing

    if current_step <= 0:
        return

    x = np.arange(current_step)

    plt.figure(figsize=(12, 8))
    plt.plot(x, drag_coefficients[:current_step], label="Drag Coefficient", color="red", alpha=0.3)
    plt.hlines(drag_coefficients[:current_step].mean(), 0, current_step, colors="red", linestyles="dashed", label="Drag Coefficient Mean")
    plt.plot(x, lift_coefficients[:current_step], label="Lift Coefficient", color="blue", alpha=0.3)
    plt.hlines(lift_coefficients[:current_step].mean(), 0, current_step, colors="blue", linestyles="dashed", label="Lift Coefficient Mean")

    print(
        f"Batch-averaged Drag Coefficient: {drag_coefficients[:current_step].mean():.4f}, "
        f"Lift Coefficient: {lift_coefficients[:current_step].mean():.4f}"
    )


    plt.ylim(-2.0, 2.0)
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Drag coefficient")
    plt.title(f"Drag/Lift Coefficients (Batch-Averaged, step {current_step})")
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

    return cd, cl
