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
    ZouHeBC,
    ExtrapolationOutflowBC,
    RegularizedBC,
    FullwayBounceBackBC,
)

from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic

from helpers import (
    LBUnitConverter,
    build_airfoil_indices,
    post_process,
)

# Simulation Configuration
grid_shape = (1200, 1200)
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32

velocity_set = D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

domain_length_m = 0.6  # Physical domain size in meters (length x height)
Re = 50000.0
air_kinematic_viscosity_m2ps = 1.511e-5
sim_time = 5.0  # Total simulation time in seconds
print_interval = 10000
output_interval = 100
eval_start_step = 20000

# Airfoil obstacle parameters (NACA 4-digit style)
airfoil_chord_length = 0.13  # m
airfoil_thickness = 0.02 / airfoil_chord_length
airfoil_camber = 0.01
airfoil_camber_position = 0.40
airfoil_angle_deg = -30.0
airfoil_x_position = 0.03
airfoil_y_position = 0.5

units = LBUnitConverter(
    grid_shape=grid_shape,
    domain_length_m=domain_length_m,
    wind_speed_mps=None,
    chord_fraction=airfoil_chord_length / domain_length_m,
    reynolds_number=Re,
    nu_m2ps=air_kinematic_viscosity_m2ps,
)

voxel_size = units.dx
clength = units.chord_m
wind_speed_mps = units.wind_speed_mps
visc = units.nu_m2ps
omega = units.omega

# Print simulation info
print("\n" + "=" * 50 + "\n")
print("Simulation Configuration:")
print(f"Grid size: {grid_shape[0]} x {grid_shape[1]}")
print(f"Grid dimensions: {grid_shape[0]*voxel_size} x {grid_shape[1]*voxel_size}")
print(f"Number of cells: {grid_shape[0]*grid_shape[1]}")
print(f"Backend: {compute_backend}")
print(f"Velocity set: {velocity_set}")
print(f"Precision policy: {precision_policy}")
print(f"Domain length [m]: {domain_length_m}")
print(f"Grid spacing dx [m]: {units.dx}")
print(f"Time step dt [s]: {units.dt}")
print(f"Kinematic viscosity nu [m^2/s]: {visc}")
print(f"Reynolds number: {Re}")
print(f"Computed freestream velocity [m/s]: {wind_speed_mps}")
print(f"Lattice inlet velocity u_lb: {units.u_lb_inlet}")
print(f"Lattice viscosity nu_lb: {units.nu_lb}")
print(f"Relaxation time tau: {units.tau}")
print(f"Relaxation factor omega: {omega}")
print(f"Mach number estimate: {units.ma}")
print(f"Max iterations: {sim_time/units.dt}")
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

obstacle, obstacle_boundary_layer, obstacle_boundary_voxels = build_airfoil_indices(
    grid_shape,
    chord_fraction=airfoil_chord_length / domain_length_m,
    thickness=airfoil_thickness,
    camber=airfoil_camber,
    camber_position=airfoil_camber_position,
    angle_deg=airfoil_angle_deg,
    x_position=airfoil_x_position,
    y_position=airfoil_y_position,
)

# Create a dynamic inlet profile with velocity ramp to suppress acoustic waves
ramp_steps = 50000  # Number of steps to ramp velocity (smooth inlet startup)
max_u_lb = units.u_lb_inlet  # Target inlet velocity

print(
    f"Inlet velocity will ramp from 0 to {max_u_lb} over {ramp_steps} steps ({ramp_steps*units.dt:.2f}s)"
)


def get_inlet_velocity(step):
    """Smooth inlet ramp to suppress acoustic waves from sharp velocity step"""
    if step < ramp_steps:
        ramp_factor = step / ramp_steps
        return max_u_lb * ramp_factor
    return max_u_lb


# Update BC with dynamic velocity based on current step
def update_inlet_bc(bc, step):
    """Update inlet BC velocity based on ramp"""
    current_vel = get_inlet_velocity(step)
    bc.prescribed_value = current_vel
    bc.prescribed_values = jnp.array(
        [current_vel], dtype=bc.precision_policy.store_precision.jax_dtype
    )


bc_inlet = RegularizedBC(
    "velocity",
    prescribed_value=(units.u_lb_inlet, 0.0),
    indices=inlet,
)
bc_outlet = ExtrapolationOutflowBC(indices=outlet)
bc_walls = ExtrapolationOutflowBC(indices=walls)
bc_obstacle = HalfwayBounceBackBC(indices=obstacle)


boundary_conditions = [bc_inlet, bc_outlet, bc_walls, bc_obstacle]

stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
)

f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

# Setup Momentum Transfer for Force Calculation
momentum_transfer = MomentumTransfer(bc_obstacle, compute_backend=compute_backend)
macro = Macroscopic(
    compute_backend=ComputeBackend.JAX,
    precision_policy=precision_policy,
    velocity_set=D2Q9(
        precision_policy=precision_policy, compute_backend=ComputeBackend.JAX
    ),
)

start_time = time.time()

for _T in range(int(sim_time / units.dt)):
    # Update inlet velocity ramp to suppress acoustic waves
    update_inlet_bc(bc_inlet, _T)

    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, _T)
    f_0, f_1 = f_1, f_0

    if _T % print_interval == 0:
        if compute_backend == ComputeBackend.WARP:
            wp.synchronize()
        elapsed_time = time.time() - start_time
        print(
            f"Iteration: {_T*units.dt:0.2f}/{sim_time} | Time elapsed: {elapsed_time:.2f}s"
        )
        start_time = time.time()
    if _T % output_interval == 0:
        post_process(
            step=_T,
            f_0=f_0,
            macro=macro,
            units=units,
            obstacle_indices=obstacle,
            boundary_layer_voxels=obstacle_boundary_voxels,
            airfoil_angle_deg=airfoil_angle_deg,
        )


print("Simulation completed successfully.")
