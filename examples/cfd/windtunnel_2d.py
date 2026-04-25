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

from helpers import LBUnitConverter
from helpers import build_airfoil_indices, plot_simulation_setup, post_process, compute_forces

# Simulation Configuration
grid_shape = (1500, 900)
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32

velocity_set = D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

domain_size_m = (1.5, 0.9)
wind_speed_mps = 10.0
Re = 80000.0
num_steps = 500000
print_interval = 10000
output_interval = 100
eval_interval = 1000

# Airfoil obstacle parameters (NACA 4-digit style)
airfoil_chord_length = 0.30
airfoil_thickness = 0.15
airfoil_camber = 0.05
airfoil_camber_position = 0.40
airfoil_angle_deg = -90.0
airfoil_x_position = 0.25
airfoil_y_position = 0.70

units = LBUnitConverter(
    grid_shape=grid_shape,
    domain_size_m=domain_size_m,
    wind_speed_mps=wind_speed_mps,
    chord_fraction=airfoil_chord_length / domain_size_m[0],
    reynolds_number=Re,
    target_u_lb=0.05,
)

voxel_size = units.dx
clength = units.chord_m
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
print(f"Domain size [m]: {domain_size_m[0]} x {domain_size_m[1]}")
print(f"Grid spacing dx [m]: {units.dx}")
print(f"Time step dt [s]: {units.dt}")
print(f"Prescribed velocity [m/s]: {wind_speed_mps}")
print(f"Reynolds number: {Re}")
print(f"Physical kinematic viscosity nu [m^2/s]: {visc}")
print(f"Lattice inlet velocity u_lb: {units.u_lb_inlet}")
print(f"Lattice viscosity nu_lb: {units.nu_lb}")
print(f"Relaxation time tau: {units.tau}")
print(f"Relaxation factor omega: {omega}")
print(f"Mach number estimate: {units.ma}")
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

obstacle_indices = build_airfoil_indices(
    grid_shape,
    chord_fraction=airfoil_chord_length / domain_size_m[0],
    thickness=airfoil_thickness,
    camber=airfoil_camber,
    camber_position=airfoil_camber_position,
    angle_deg=airfoil_angle_deg,
    x_position=airfoil_x_position,
    y_position=airfoil_y_position,
)

bc_inlet = RegularizedBC(
    "velocity",
    prescribed_value=(units.u_lb_inlet, 0.0),
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
            macro,
            units,
        )
    if step % eval_interval == 0:
        # Perform evaluation or analysis at specified intervals
        compute_forces(
            step,
            f_0,
            f_1,
            bc_mask,
            missing_mask,
            momentum_transfer,
            units,
            airfoil_chord_length,
        )
        pass

print("Simulation completed successfully.")
