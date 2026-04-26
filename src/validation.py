import numpy as np
import os
from particle_system import ParticleSystem
from phase_change import PhaseChangeModel
from solver import PhaseChangeSolver
from stefan_problem import StefanProblem
from visualize import plot_1d_comparison, plot_convergence
from tqdm import tqdm

def run_stefan_1d_validation(nx: int, t_end: float = 50.0, output_dir: str = 'results'):
    os.makedirs(output_dir, exist_ok=True)

    L = 0.05
    dx = L / (nx - 1)

    ps = ParticleSystem(nx=nx, ny=3, dx=dx)

    phase_model = PhaseChangeModel(
        T_melt=273.15,
        latent_heat=33400.0,
        c_p_solid=2100.0,
        c_p_liquid=4200.0,
        k_solid=2.0,
        k_liquid=0.5,
        rho=1000.0,
        interface_width=1.5 * dx
    )

    stefan = StefanProblem(
        T_cold=263.15,
        T_hot=283.15,
        T_melt=273.15,
        k_solid=2.0,
        k_liquid=0.5,
        c_p_solid=2100.0,
        c_p_liquid=4200.0,
        rho=1000.0,
        latent_heat=33400.0
    )

    for i in range(ps.n_particles):
        x = ps.positions[i, 0]
        if x < L * 0.3:
            ps.temperatures[i] = 263.15
        else:
            ps.temperatures[i] = 283.15

    ps.update_phase(phase_model.T_melt, phase_model.interface_width)

    solver = PhaseChangeSolver(ps, phase_model)

    print(f"Running 1D Stefan validation (nx={nx})...")
    print(f"  Stefan lambda: {stefan.lambda_param:.4f}")
    print(f"  Domain: [0, {L}]m, dx={dx:.6f}m")

    dt_cfl = solver.compute_time_step(cfl=0.3)
    n_steps = max(50, int(t_end / dt_cfl))
    dt = t_end / n_steps

    print(f"  Time step: {dt:.6e}s, Steps: {n_steps}")

    for step in tqdm(range(n_steps), desc="Solving"):
        solver.step(dt)

    x_sim = ps.positions[:, 0]
    T_sim = ps.temperatures

    mid_y = ps.positions[:, 1].mean()
    mask = np.abs(ps.positions[:, 1] - mid_y) < 0.5 * dx
    x_sim_1d = x_sim[mask]
    T_sim_1d = T_sim[mask]

    sort_idx = np.argsort(x_sim_1d)
    x_sim_1d = x_sim_1d[sort_idx]
    T_sim_1d = T_sim_1d[sort_idx]

    x_analytical = np.linspace(0, L, 500)
    T_analytical = stefan.temperature_field(x_analytical, t_end)

    interface_analytical = stefan.interface_position(t_end)

    interface_particles = np.where((ps.liquid_fraction > 0.4) & (ps.liquid_fraction < 0.6))[0]
    if len(interface_particles) > 0:
        interface_sim = ps.positions[interface_particles, 0].mean()
    else:
        interface_sim = L * 0.3

    T_analytical_interp = np.interp(x_sim_1d, x_analytical, T_analytical)
    error = np.sqrt(np.mean((T_sim_1d - T_analytical_interp)**2))

    interface_error = abs(interface_sim - interface_analytical)

    print(f"  L2 Error: {error:.6e}")
    print(f"  Interface position - Analytical: {interface_analytical:.6f}m, Simulated: {interface_sim:.6f}m")
    print(f"  Interface error: {interface_error:.6f}m")

    return {
        'nx': nx,
        'error': error,
        'interface_error': interface_error,
        'x_sim': x_sim_1d,
        'T_sim': T_sim_1d,
        'x_analytical': x_analytical,
        'T_analytical': T_analytical,
        'interface_sim': interface_sim,
        'interface_analytical': interface_analytical,
        'time': t_end
    }

def run_convergence_study(resolutions=[20, 40, 80, 160], t_end=50.0, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    results = []
    errors = []
    orders = []

    print("="*60)
    print("Stefan Problem Convergence Study")
    print("="*60)

    for nx in resolutions:
        result = run_stefan_1d_validation(nx, t_end, output_dir)
        results.append(result)
        errors.append(result['error'])
        print()

    print("="*60)
    print("Convergence Analysis")
    print("="*60)
    print(f"{'Resolution':<12} {'L2 Error':<15} {'Order':<10}")
    print("-"*60)

    for i, (nx, err) in enumerate(zip(resolutions, errors)):
        if i == 0:
            print(f"{nx:<12} {err:<15.6e} {'-':<10}")
        else:
            order = np.log(errors[i-1] / errors[i]) / np.log(resolutions[i] / resolutions[i-1])
            orders.append(order)
            print(f"{nx:<12} {err:<15.6e} {order:<10.2f}")

    print("="*60)

    plot_convergence(resolutions, errors, orders,
                    filename=os.path.join(output_dir, 'convergence_analysis.png'))

    last_result = results[-1]
    plot_1d_comparison(
        last_result['x_sim'],
        last_result['T_sim'],
        last_result['x_analytical'],
        last_result['T_analytical'],
        last_result['time'],
        last_result['interface_sim'],
        last_result['interface_analytical'],
        filename=os.path.join(output_dir, 'stefan_validation.png')
    )

    print(f"\nResults saved to '{output_dir}/' directory")

    return results, errors, orders

if __name__ == '__main__':
    print("Starting Stefan problem validation...\n")
    results, errors, orders = run_convergence_study()
    print("\nValidation complete!")
