import numpy as np
from particle_system import ParticleSystem
from phase_change import PhaseChangeModel
from solver import PhaseChangeSolver

nx, ny = 40, 40
L = 0.1
dx = L / (nx - 1)

ps = ParticleSystem(nx=nx, ny=ny, dx=dx)

phase_model = PhaseChangeModel(
    T_melt=273.15,
    latent_heat=50000.0,
    c_p_solid=2100.0,
    c_p_liquid=4200.0,
    k_solid=2.2,
    k_liquid=0.6,
    rho=1000.0,
    interface_width=2.5 * dx
)

for i in range(ps.n_particles):
    x, y = ps.positions[i]
    if x < L * 0.35:
        ps.temperatures[i] = 260.0
    else:
        ps.temperatures[i] = 286.0

ps.update_phase(phase_model.T_melt, phase_model.interface_width)

solver = PhaseChangeSolver(ps, phase_model)

print("Initial state:")
print(f"  T range: {ps.temperatures.min():.2f} - {ps.temperatures.max():.2f} K")
print(f"  Liquid fraction range: {ps.liquid_fraction.min():.3f} - {ps.liquid_fraction.max():.3f}")

dt = solver.compute_time_step(cfl=0.25)
print(f"\nTime step: {dt:.4f}s")

print("\nBefore step:")
print(f"  T range: {ps.temperatures.min():.2f} - {ps.temperatures.max():.2f} K")

solver.step(dt)

print("\nAfter 1 step:")
print(f"  T range: {ps.temperatures.min():.2f} - {ps.temperatures.max():.2f} K")
print(f"  Liquid fraction range: {ps.liquid_fraction.min():.3f} - {ps.liquid_fraction.max():.3f}")

for _ in range(9):
    solver.step(dt)

print("\nAfter 10 steps:")
print(f"  T range: {ps.temperatures.min():.2f} - {ps.temperatures.max():.2f} K")
print(f"  Liquid fraction range: {ps.liquid_fraction.min():.3f} - {ps.liquid_fraction.max():.3f}")

x_left = ps.positions[:, 0] < L * 0.1
x_right = ps.positions[:, 0] > L * 0.9

print(f"\nLeft boundary (x < 0.01m):")
print(f"  T: {ps.temperatures[x_left].mean():.2f} K")
print(f"Right boundary (x > 0.09m):")
print(f"  T: {ps.temperatures[x_right].mean():.2f} K")
