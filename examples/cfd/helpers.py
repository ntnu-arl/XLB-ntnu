from dataclasses import dataclass
from typing import Any, Optional

import atexit
import importlib
import multiprocessing as mp
import queue as queue_module
import numpy as np
import jax.numpy as jnp
import warp as wp

pg: Any = None
try:
    pg = importlib.import_module("pyqtgraph")
except Exception:  # pragma: no cover - optional dependency.
    pass

from matplotlib.path import Path as MplPath


@dataclass
class LBUnitConverter:
    """Convert between SI units and lattice units for a D2Q9 simulation."""

    grid_shape: tuple
    domain_length_m: float
    chord_fraction: float
    wind_speed_mps: Optional[float] = None
    reynolds_number: Optional[float] = None
    nu_m2ps: Optional[float] = None
    target_u_lb: float = 0.05
    rho_ref_kgm3: float = 1.225

    def __post_init__(self):
        nx, ny = self.grid_shape
        lx_m = self.domain_length_m

        self.dx = lx_m / nx

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

    def to_si_pressure(self, p_lb):
        return p_lb * self.pressure_scale

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
    naca_points=1000,
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
    - boundary_layer_voxels: [x, y, x/c, y/c, side, n_x, n_y, ds], where
        side is +1 for upper surface, -1 for lower surface
        (n_x, n_y) is the estimated outward unit surface normal in grid coordinates
        ds is the surface element length for proper pressure integration
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
            - 0.1015 * x**4
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
    neighbor_offsets = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (2, 0),
        (-2, 0),
        (0, 2),
        (0, -2),
    )
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
        raise ValueError(
            "No boundary layer cells found. Check airfoil parameters and grid resolution."
        )
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

    # Interpolate local surface slopes dy/dx along upper/lower curves.
    upper_dx = np.gradient(upper_x)
    lower_dx = np.gradient(lower_x)
    upper_slope = np.gradient(upper_y) / np.where(
        np.abs(upper_dx) < 1.0e-8, 1.0e-8, upper_dx
    )
    lower_slope = np.gradient(lower_y) / np.where(
        np.abs(lower_dx) < 1.0e-8, 1.0e-8, lower_dx
    )

    upper_surface_y = np.interp(local_x, upper_x, upper_y)
    lower_surface_y = np.interp(local_x, lower_x, lower_y)
    camber_midline_y = 0.5 * (upper_surface_y + lower_surface_y)

    boundary_layer_chord_distance = local_y
    boundary_layer_side_flag = np.where(
        local_y >= camber_midline_y,
        1.0,
        -1.0,
    ).astype(np.float32)

    sample_upper_slope = np.interp(local_x, upper_x, upper_slope)
    sample_lower_slope = np.interp(local_x, lower_x, lower_slope)
    sample_surface_slope = np.where(
        boundary_layer_side_flag > 0.0,
        sample_upper_slope,
        sample_lower_slope,
    )

    # Local outward normal for y=f(x): upper [-f'(x), 1], lower [f'(x), -1].
    normal_local_x = -sample_surface_slope * boundary_layer_side_flag
    normal_local_y = boundary_layer_side_flag
    normal_local_norm = np.sqrt(normal_local_x**2 + normal_local_y**2)
    normal_local_norm = np.where(normal_local_norm < 1.0e-8, 1.0, normal_local_norm)
    normal_local_x /= normal_local_norm
    normal_local_y /= normal_local_norm

    boundary_layer_normal_x = (
        normal_local_x * chord_axis[0] + normal_local_y * chord_normal[0]
    ).astype(np.float32)
    boundary_layer_normal_y = (
        normal_local_x * chord_axis[1] + normal_local_y * chord_normal[1]
    ).astype(np.float32)

    # Calculate arc lengths of upper and lower surfaces.
    # Use the sorted upper_x, upper_y and lower_x, lower_y arrays.
    upper_dx_curve = np.diff(upper_x, prepend=upper_x[0])
    upper_dy_curve = np.diff(upper_y, prepend=upper_y[0])
    upper_ds_curve = np.sqrt(upper_dx_curve**2 + upper_dy_curve**2)
    upper_arc_length = np.sum(upper_ds_curve)

    lower_dx_curve = np.diff(lower_x, prepend=lower_x[0])
    lower_dy_curve = np.diff(lower_y, prepend=lower_y[0])
    lower_ds_curve = np.sqrt(lower_dx_curve**2 + lower_dy_curve**2)
    lower_arc_length = np.sum(lower_ds_curve)

    # Count number of boundary voxels on each side.
    num_upper = np.sum(boundary_layer_side_flag > 0.0)
    num_lower = np.sum(boundary_layer_side_flag < 0.0)

    # Compute surface element length (ds) for each side.
    # This is the arc length divided by the number of voxels on that side.
    dx_upper = float(upper_arc_length) / max(1, num_upper)
    dx_lower = float(lower_arc_length) / max(1, num_lower)

    # Map each boundary voxel to its corresponding ds value based on side flag.
    boundary_layer_ds = np.where(
        boundary_layer_side_flag > 0.0,
        dx_upper,
        dx_lower,
    ).astype(np.float32)

    boundary_layer_voxels = np.hstack(
        [
            boundary_layer_cells,
            boundary_layer_chord_fraction.reshape(-1, 1),
            boundary_layer_chord_distance.reshape(-1, 1),
            boundary_layer_side_flag.reshape(-1, 1),
            boundary_layer_normal_x.reshape(-1, 1),
            boundary_layer_normal_y.reshape(-1, 1),
            boundary_layer_ds.reshape(-1, 1),
        ]
    )
    return (
        obstacle_indices,
        boundary_layer_indices,
        boundary_layer_voxels,
    )


_WINDTUNNEL_DASHBOARD = None


def _require_pyqtgraph():
    if pg is None:
        raise RuntimeError(
            "pyqtgraph is not available. Install pyqtgraph and PySide6 to use live plotting."
        )

def _normalize_boundary_layer_voxels(boundary_layer_voxels):
    if boundary_layer_voxels is None:
        return np.empty((0, 8), dtype=np.float32)

    array = np.asarray(boundary_layer_voxels, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 8), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError(
            "boundary_layer_voxels must be shaped like [N, >=5] with columns [x, y, x/c, y/c, side, (optional n_x, n_y, ds)]."
        )
    return array


def _extract_windtunnel_diagnostics(
    f_0, macro, units: LBUnitConverter, boundary_layer_voxels=None
):
    global rho_wp, u_wp
    if "rho_wp" not in globals() or "u_wp" not in globals():
        print("Initializing diagnostic buffers...")
        rho_wp = wp.zeros((1, f_0.shape[1], f_0.shape[2], f_0.shape[3]), dtype=wp.float32)
        u_wp = wp.zeros((2, f_0.shape[1], f_0.shape[2], f_0.shape[3]), dtype=wp.float32)

    rho_wp, u_wp = macro(f_0, rho_wp, u_wp)

    rho = np.asarray(wp.to_jax(rho_wp)[:,1:-1,1:-1,0])  # Remove halo and extract scalar density
    u = np.asarray(wp.to_jax(u_wp)[:,1:-1,1:-1,0])  # Remove halo and extract scalar density

    #print(rho.sum()/rho.size)

    flowfield = units.to_si_velocity(np.sqrt(np.pow(u[0], 2) + np.pow(u[1], 2)))
    # flowfield = _to_numpy(units.pressure_fluctuation_to_si(rho[0], rho_lb0=1.0)) * 100

    wind_speed_mps = units.wind_speed_mps
    if wind_speed_mps is None:
        raise ValueError(
            "units.wind_speed_mps must be set before extracting diagnostics."
        )
    
    q_dyn = 0.5 * units.rho_ref_kgm3 * units.to_si_velocity(u[0] ** 2 + u[1] ** 2)
    q_inf = 0.5 * units.rho_ref_kgm3 * wind_speed_mps**2

    pressure = units.pressure_fluctuation_to_si(rho[0], rho_lb0=1.0)
    pressure_coefficient = np.asarray(pressure / q_inf)

    if boundary_layer_voxels is None:
        return flowfield, None, None

    boundary_layer_voxels = _normalize_boundary_layer_voxels(boundary_layer_voxels)
    if boundary_layer_voxels.size == 0:
        return flowfield, None, None

    sample_x = boundary_layer_voxels[:, 0].astype(np.int32)
    sample_y = boundary_layer_voxels[:, 1].astype(np.int32)
    chord_fraction = boundary_layer_voxels[:, 2].astype(np.float32)
    side_flag = boundary_layer_voxels[:, 4].astype(np.float32)
    sample_values = pressure_coefficient[sample_x, sample_y]

    force = compute_forces_from_boundary_voxels(
        pressure_coefficient, boundary_layer_voxels, units, q_inf
    )

    upper_mask = side_flag > 0.0
    lower_mask = side_flag < 0.0
    if not np.any(upper_mask) or not np.any(lower_mask):
        return flowfield, None, force

    upper_x = chord_fraction[upper_mask]
    upper_cp = sample_values[upper_mask]
    lower_x = chord_fraction[lower_mask]
    lower_cp = sample_values[lower_mask]

    upper_order = np.argsort(upper_x)
    lower_order = np.argsort(lower_x)
    pressure_profile = {
        "upper_x": upper_x[upper_order],
        "upper_cp": upper_cp[upper_order],
        "lower_x": lower_x[lower_order],
        "lower_cp": lower_cp[lower_order],
    }
    return flowfield, pressure_profile, force


class WindTunnelDashboard:
    def __init__(self, flowfield_shape, voxel_size):
        _require_pyqtgraph()
        self.app = pg.mkQApp("XLB Wind Tunnel Dashboard")
        self.window = pg.GraphicsLayoutWidget(title="XLB Wind Tunnel Dashboard")
        self.window.resize(1500, 700)
        self.flowfield_shape = tuple(flowfield_shape)
        self.voxel_size = voxel_size
        self.flow_scaling = np.zeros(2, dtype=np.float32)
        self._counter = 0

        self.flow_plot = self.window.addPlot(row=0, col=0, rowspan=2)
        self.flow_plot.setAspectLocked(True)
        self.flow_plot.showGrid(x=True, y=True, alpha=0.2)
        self.flow_plot.setLabel("bottom", "x", units="m")
        self.flow_plot.setLabel("left", "y", units="m")
        self.flow_plot.setTitle("Flowfield magnitude")

        self.flow_image = pg.ImageItem(axisOrder="row-major")
        self.flow_image.setLookupTable(
            pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
        )
        self.flow_plot.addItem(self.flow_image)

        self.airfoil_overlay = pg.ScatterPlotItem(
            pen=pg.mkPen("#111111", width=1),
            brush=pg.mkBrush(17, 17, 17, 160),
            size=3,
        )
        self.flow_plot.addItem(self.airfoil_overlay)

        self.curve_length = 200
        self.cp_buffer_size = 10
        self.cld_buffer_size = 100
        self.upper_cp_buffer = None
        self.lower_cp_buffer = None
        self.lift_coef_buffer = np.zeros((self.cld_buffer_size,), dtype=np.float32)
        self.drag_coef_buffer = np.zeros((self.cld_buffer_size,), dtype=np.float32)
        self.drag_curve_data = np.zeros((self.curve_length,), dtype=np.float32)
        self.lift_curve_data = np.zeros((self.curve_length,), dtype=np.float32)

        # BC contour overlays for walls, inlet, outlet, obstacle
        self.wall_contour = pg.PlotCurveItem(pen=pg.mkPen("#2b6cb0", width=2))
        self.inlet_contour = pg.PlotCurveItem(pen=pg.mkPen("#38a169", width=2))
        self.outlet_contour = pg.PlotCurveItem(pen=pg.mkPen("#dd6b20", width=2))
        self.obstacle_contour = pg.PlotCurveItem(pen=pg.mkPen("#c53030", width=2))
        self.flow_plot.addItem(self.wall_contour)
        self.flow_plot.addItem(self.inlet_contour)
        self.flow_plot.addItem(self.outlet_contour)
        self.flow_plot.addItem(self.obstacle_contour)

        self.pressure_plot = self.window.addPlot(row=0, col=1)
        self.pressure_plot.showGrid(x=True, y=True, alpha=0.2)
        self.pressure_plot.setLabel("bottom", "Chord fraction", units="x/c")
        self.pressure_plot.setLabel("left", "Pressure coefficient", units="C_p")
        self.pressure_plot.invertY(True)
        self.pressure_plot.addLegend()
        self.upper_curve = self.pressure_plot.plot(
            [],
            [],
            pen=pg.mkPen("#2b6cb0", width=2),
            symbol="o",
            symbolSize=2,
            name="Upper surface",
        )
        self.lower_curve = self.pressure_plot.plot(
            [],
            [],
            pen=pg.mkPen("#c53030", width=2),
            symbol="o",
            symbolSize=2,
            name="Lower surface",
        )

        self.coefficient_plot = self.window.addPlot(row=1, col=1)
        self.coefficient_plot.showGrid(x=True, y=True, alpha=0.5)
        self.coefficient_plot.setLabel("bottom", "Time step")
        self.coefficient_plot.setLabel("left", "Coefficient")
        self.coefficient_plot.addLegend()
        self.coefficient_plot.setTitle("Drag and lift coefficients")
        self.drag_curve = self.coefficient_plot.plot(
            [],
            [],
            pen=pg.mkPen("#c53030", width=4),
            symbol="o",
            symbolSize=1,
            name="Drag coefficient",
        )
        self.lift_curve = self.coefficient_plot.plot(
            [],
            [],
            pen=pg.mkPen("#2b6cb0", width=4),
            symbol="o",
            symbolSize=1,
            name="Lift coefficient",
        )

        self.window.show()
        self.app.processEvents()

    def update(
        self,
        step,
        flowfield_np,
        pressure_profile=None,
        boundary_layer_voxels=None,
        airfoil_angle_deg=None,
        drag_coefficient=None,
        lift_coefficient=None,
    ):
        if flowfield_np.ndim != 2:
            raise ValueError("flowfield must be a 2D array of velocity magnitudes.")

        self.flow_image.setImage(flowfield_np.T, autoLevels=False)
        self._update_flowfield_range(flowfield_np)
        self.flow_plot.setXRange(0.0, flowfield_np.shape[0], padding=0.0)
        self.flow_plot.setYRange(0.0, flowfield_np.shape[1], padding=0.0)
        self.flow_plot.setTitle(f"Flowfield magnitude (step {step})")

        upper_x, upper_cp, lower_x, lower_cp = self._pressure_arrays(
            pressure_profile,
            boundary_layer_voxels,
        )

        if self.lower_cp_buffer is None or self.upper_cp_buffer is None:
            self.lower_cp_buffer = np.zeros(
                (self.cp_buffer_size, lower_cp.shape[0]), dtype=np.float32
            )
            self.upper_cp_buffer = np.zeros(
                (self.cp_buffer_size, upper_cp.shape[0]), dtype=np.float32
            )

        self.upper_cp_buffer[0, :] = upper_cp
        self.lower_cp_buffer[0, :] = lower_cp
        self.upper_cp_buffer = np.roll(self.upper_cp_buffer, 1, axis=0)
        self.lower_cp_buffer = np.roll(self.lower_cp_buffer, 1, axis=0)

        avg_upper_cp = np.mean(self.upper_cp_buffer, axis=0)
        avg_lower_cp = np.mean(self.lower_cp_buffer, axis=0)

        self.upper_curve.setData(upper_x, avg_upper_cp)
        self.lower_curve.setData(lower_x, avg_lower_cp)
        self.pressure_plot.setTitle(self._pressure_title(step, airfoil_angle_deg))
        self._update_pressure_range(avg_upper_cp, avg_lower_cp)

        self.lift_coef_buffer[0] = lift_coefficient
        self.lift_coef_buffer = np.roll(self.lift_coef_buffer, 1)
        self.drag_coef_buffer[0] = drag_coefficient
        self.drag_coef_buffer = np.roll(self.drag_coef_buffer, 1)

        self.lift_curve_data = np.roll(self.lift_curve_data, 1)
        self.lift_curve_data[0] = self.lift_coef_buffer.mean()
        self.drag_curve_data = np.roll(self.drag_curve_data, 1)
        self.drag_curve_data[0] = self.drag_coef_buffer.mean()

        _step = min(self._counter, self.curve_length)
        self.lift_curve.setData(-np.arange(-_step, 0), self.lift_curve_data[:_step])
        self.drag_curve.setData(-np.arange(-_step, 0), self.drag_curve_data[:_step])

        self._update_coefficient_range(
            self.drag_curve_data[:_step],
            self.lift_curve_data[:_step],
            self._counter,
        )

        self._counter += 1
        self.app.processEvents()

    @staticmethod
    def _pressure_arrays(pressure_profile, boundary_layer_voxels):
        if pressure_profile is None:
            empty = np.asarray([], dtype=np.float32)
            return empty, empty, empty, empty

        return (
            np.asarray(pressure_profile["upper_x"], dtype=np.float32),
            np.asarray(pressure_profile["upper_cp"], dtype=np.float32),
            np.asarray(pressure_profile["lower_x"], dtype=np.float32),
            np.asarray(pressure_profile["lower_cp"], dtype=np.float32),
        )

    @staticmethod
    def _pressure_title(step, airfoil_angle_deg):
        if airfoil_angle_deg is None:
            return f"Airfoil pressure profile (step {step})"
        return f"Airfoil pressure profile (step {step}, angle {airfoil_angle_deg:.1f}°)"

    def _update_flowfield_range(self, flowfield):
        finite_values = flowfield[np.isfinite(flowfield)]
        if finite_values.size == 0:
            return
        vmax = max(self.flow_scaling[1], float(np.max(finite_values)))
        vmin = min(self.flow_scaling[0], float(np.min(finite_values)))
        self.flow_scaling[0] = vmin * 0.99
        self.flow_scaling[1] = vmax * 0.99
        self.flow_image.setLevels((vmin, vmax))

    def _update_pressure_range(self, upper_cp, lower_cp):
        finite_values = np.concatenate(
            [upper_cp[np.isfinite(upper_cp)], lower_cp[np.isfinite(lower_cp)]]
        )
        if finite_values.size == 0:
            return
        cp_min = float(np.min(finite_values))
        cp_max = float(np.max(finite_values))
        padding = max(0.05 * (cp_max - cp_min), 0.05)
        self.pressure_plot.setYRange(cp_min - padding, cp_max + padding, padding=0.0)

    def _update_coefficient_range(
        self,
        drag_coefficient,
        lift_coefficient,
        step,
    ):
        finite_values = np.asarray(
            [drag_coefficient, lift_coefficient],
            dtype=np.float32,
        )
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return

        self.coefficient_plot.setXRange(0.0, min(step, self.curve_length), padding=0.0)

        coeff_min = float(np.min(finite_values))
        coeff_max = float(np.max(finite_values))
        padding = max(0.05 * (coeff_max - coeff_min), 0.05)
        self.coefficient_plot.setYRange(
            coeff_min - padding,
            coeff_max + padding,
            padding=0.0,
        )


def _dashboard_process_main(message_queue, stop_event, flowfield_shape, voxel_size):
    dashboard = WindTunnelDashboard(
        flowfield_shape=flowfield_shape,
        voxel_size=voxel_size,
    )
    try:
        while not stop_event.is_set():
            try:
                payload = message_queue.get(timeout=0.1)
            except queue_module.Empty:
                dashboard.app.processEvents()
                continue

            if payload is None:
                break

            dashboard.update(**payload)
    finally:
        dashboard.app.processEvents()


class AsyncWindTunnelDashboard:
    def __init__(self, flowfield_shape, voxel_size, flow_vmax=None):
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=1)
        self._stop_event = ctx.Event()
        self._process = ctx.Process(
            target=_dashboard_process_main,
            args=(self._queue, self._stop_event, flowfield_shape, voxel_size),
            daemon=True,
        )
        self._closed = False
        self._process.start()
        atexit.register(self.close)

    def update(
        self,
        step,
        flowfield_np,
        pressure_profile=None,
        boundary_layer_voxels=None,
        airfoil_angle_deg=None,
        drag_coefficient=None,
        lift_coefficient=None,
    ):
        if self._closed:
            return

        payload = {
            "step": step,
            "flowfield_np": flowfield_np,
            "pressure_profile": pressure_profile,
            "boundary_layer_voxels": boundary_layer_voxels,
            "airfoil_angle_deg": airfoil_angle_deg,
            "drag_coefficient": drag_coefficient,
            "lift_coefficient": lift_coefficient,
        }

        try:
            self._queue.put_nowait(payload)
        except queue_module.Full:
            try:
                self._queue.get_nowait()
            except queue_module.Empty:
                pass

            try:
                self._queue.put_nowait(payload)
            except queue_module.Full:
                pass

    def close(self):
        if self._closed:
            return

        self._closed = True
        self._stop_event.set()

        try:
            self._queue.put_nowait(None)
        except queue_module.Full:
            try:
                self._queue.get_nowait()
            except queue_module.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue_module.Full:
                pass

        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)


def _get_dashboard(flowfield_shape, voxel_size, flow_vmax):
    global _WINDTUNNEL_DASHBOARD
    if (
        _WINDTUNNEL_DASHBOARD is None
        or _WINDTUNNEL_DASHBOARD.flowfield_shape != tuple(flowfield_shape)
        or _WINDTUNNEL_DASHBOARD.voxel_size != voxel_size
    ):
        _WINDTUNNEL_DASHBOARD = WindTunnelDashboard(
            flowfield_shape=flowfield_shape,
            voxel_size=voxel_size,
        )
    return _WINDTUNNEL_DASHBOARD


def post_process(
    step,
    f_0,
    f_1,
    macro,
    momentum_transfer,
    missing_mask,
    bc_mask,
    boundary_layer_voxels,
    airfoil_angle_deg,
    units,
    dashboard=None,
):
    flowfield, pressure_profile, force = _extract_windtunnel_diagnostics(
        f_0,
        macro,
        units,
        boundary_layer_voxels=boundary_layer_voxels,
    )

    # lift_coefficient, drag_coefficient = compute_lift_drag_coefficients(force, units)
    lift_coefficient, drag_coefficient = compute_lift_drag_coefficients_from_momentum_transfer(
        f_0, f_1, bc_mask, missing_mask, momentum_transfer, units
    )

    if dashboard is None:
        dashboard = _get_dashboard(
            flowfield_shape=flowfield.shape,
            voxel_size=units.dx,
            flow_vmax=units.wind_speed_mps * 1.5,
        )

    dashboard.update(
        step=step,
        flowfield_np=flowfield,
        pressure_profile=pressure_profile,
        boundary_layer_voxels=boundary_layer_voxels,
        airfoil_angle_deg=airfoil_angle_deg,
        drag_coefficient=drag_coefficient,
        lift_coefficient=lift_coefficient,
    )

    return {
        "flowfield": flowfield,
        "pressure_profile": pressure_profile,
        "drag_coefficient": drag_coefficient,
        "lift_coefficient": lift_coefficient,
    }


def compute_forces_from_boundary_voxels(
    pressure_coefficient_field,
    boundary_layer_voxels,
    units,
    q_inf,
):
    """Compute forces from pressure coefficient field on airfoil surface.

    Uses pressure coefficient (C_p) for proper aerodynamic force calculation:
    F = sum(C_p * q_inf * dA * normal)

    Args:
        pressure_coefficient_field: Dimensionless pressure coefficient field
        boundary_layer_voxels: Boundary voxel coordinates and surface normals
        units: Unit converter with grid spacing dx
        q_inf: Dynamic pressure (0.5 * rho * V^2)

    Returns:
        Force vector [F_x, F_y] in N/m (force per unit depth for 2D simulations)
    """
    if boundary_layer_voxels is None or boundary_layer_voxels.size == 0:
        return np.array([0.0, 0.0])

    sample_cp = pressure_coefficient_field[
        boundary_layer_voxels[:, 0].astype(np.int32),
        boundary_layer_voxels[:, 1].astype(np.int32),
    ]

    # For 2D simulations: force per voxel = C_p * q_inf * ds (per unit depth)
    # ds is the surface element length (arc length / num_voxels), normalized in chord units
    # Multiply by chord_length to get physical ds
    ds_chord = (
        boundary_layer_voxels[:, 7].astype(np.float32)
        if boundary_layer_voxels.shape[1] > 7
        else units.dx
    )
    surface_element_area = (
        ds_chord * units.chord_m
    )  # Convert from normalized to physical length

    force_per_voxel = sample_cp * q_inf * surface_element_area
    normal_x = boundary_layer_voxels[:, 5].astype(np.float32)
    normal_y = boundary_layer_voxels[:, 6].astype(np.float32)

    # Pressure force on body acts opposite to outward surface normal.
    force_x = -np.sum(force_per_voxel * normal_x)
    force_y = -np.sum(force_per_voxel * normal_y)
    force = np.array([force_x, force_y])
    return force


def compute_lift_drag_coefficients(force, units):
    force = np.asarray(force, dtype=np.float32)
    q_inf = 0.5 * units.rho_ref_kgm3 * units.wind_speed_mps**2
    chord_length = units.chord_m

    drag_coefficient = force[0] / (q_inf * chord_length)
    lift_coefficient = force[1] / (q_inf * chord_length)

    return lift_coefficient, drag_coefficient

def compute_lift_drag_coefficients_from_momentum_transfer(
    f_0,
    f_1,
    bc_mask,
    missing_mask,
    momentum_transfer,
    units,
):
    # Compute lift and drag
    boundary_force = momentum_transfer(f_0, f_1, bc_mask, missing_mask)
    drag = boundary_force[0]  # x-direction
    lift = boundary_force[1]
    cd = 2.0 * drag *units.dx / (units.u_lb_inlet**2 * units.chord_m)
    cl = 2.0 * lift *units.dx / (units.u_lb_inlet**2 * units.chord_m)
    return cl, cd
