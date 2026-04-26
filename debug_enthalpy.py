import sys
sys.path.insert(0, 'src')
from particle_system import ParticleSystem
from phase_change import PhaseChangeModel
from solver import PhaseChangeSolver
import numpy as np

nx, ny = 10, 10
L = 0.1
dx = L / (nx - 1)

ps = ParticleSystem(nx=nx, ny=ny, dx=dx)

phase_model = PhaseChangeModel(
    T_melt=273.15,
    latent_heat=10000.0,
    k_solid=5.0,
    k_liquid=2.0,
    rho=1000.0,
    interface_width=1.0
)

for i in range(ps.n_particles):
    x, y = ps.positions[i]
    if x < L * 0.35:
        ps.temperatures[i] = 260.0
    else:
        ps.temperatures[i] = 286.0

ps.update_phase(phase_model.T_melt, phase_model.interface_width)

solver = PhaseChangeSolver(ps, phase_model, T_left=260.0, T_right=286.0)
dt = 0.01

neighbors_list = ps.get_all_neighbors()

H = np.zeros(ps.n_particles)
for i in range(ps.n_particles):
    H[i] = phase_model.temperature_to_enthalpy(ps.temperatures[i], ps.liquid_fraction[i])

print('Initial enthalpy range:', H.min(), '-', H.max(), 'J/kg')
print()

laplacians = solver.operators.compute_all_laplacians(ps.temperatures, neighbors_list)

print('Laplacian statistics:')
print('  Range:', laplacians.min(), '-', laplacians.max(), 'K/m^2')
print('  Non-zero count:', np.count_nonzero(np.abs(laplacians) > 1e-10))
print()

print('Check middle particles:')
for i in [33, 34, 43, 44]:
    x, y = ps.positions[i]
    f = ps.liquid_fraction[i]
    k = phase_model.get_thermal_conductivity(f)
    dH_dt = k / phase_model.rho * laplacians[i]
    dH = dt * dH_dt
    print(f'  Particle {i}: x={x:.4f}m, T={ps.temperatures[i]:.1f}K, lap={laplacians[i]:.6f}, dH/dt={dH_dt:.3f}, dH={dH:.6f}')
