import numpy as np
import argparse
import os
from particle_system import ParticleSystem
from phase_change import PhaseChangeModel
from solver import PhaseChangeSolver
from stefan_problem import StefanProblem
from visualize import plot_temperature_field, plot_interface, create_animation
from tqdm import tqdm

def run_stefan_case(nx=50, ny=50, t_end=2.0, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    L = 0.1
    dx = L / (nx - 1)

    print(f"Initializing particle system: {nx}x{ny} particles, dx={dx:.6f}m")
    ps = ParticleSystem(nx=nx, ny=ny, dx=dx)

    phase_model = PhaseChangeModel(
        T_melt=273.15,
        latent_heat=33400.0,
        c_p_solid=2100.0,
        c_p_liquid=4200.0,
        k_solid=2.0,
        k_liquid=0.5,
        rho=1000.0,
        interface_width=2.0 * dx
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
        latent_heat=334000.0
    )

    print("Setting initial conditions...")
    for i in range(ps.n_particles):
        x = ps.positions[i, 0]
        if x < L * 0.3:
            ps.temperatures[i] = 263.15
        else:
            ps.temperatures[i] = 283.15

    ps.update_phase(phase_model.T_melt, phase_model.interface_width)

    solver = PhaseChangeSolver(ps, phase_model)

    print(f"Stefan problem parameters:")
    params = stefan.get_parameters()
    for key, value in params.items():
        print(f"  {key}: {value:.6f}")

    dt = solver.compute_time_step(cfl=0.2)
    n_steps = int(t_end / dt)
    save_interval = max(1, n_steps // 50)

    print(f"\nSimulation parameters:")
    print(f"  Time step: {dt:.6e}s")
    print(f"  Total steps: {n_steps}")
    print(f"  End time: {t_end}s")

    history = []

    print("\nRunning simulation...")
    for step in tqdm(range(n_steps)):
        solver.step(dt)

        if step % save_interval == 0 or step == n_steps - 1:
            state = {
                'time': solver.time,
                'positions': ps.positions.copy(),
                'temperatures': ps.temperatures.copy(),
                'liquid_fraction': ps.liquid_fraction.copy(),
                'phase': ps.phase.copy(),
                'level_set': solver.interface.level_set.copy()
            }
            history.append(state)

    print("\nGenerating visualizations...")

    final_state = history[-1]
    plot_temperature_field(
        final_state['positions'],
        final_state['temperatures'],
        final_state['phase'],
        title=f"Temperature Field at t={final_state['time']:.3f}s",
        filename=os.path.join(output_dir, 'temperature_field.png')
    )

    plot_interface(
        final_state['positions'],
        final_state['level_set'],
        final_state['liquid_fraction'],
        title=f"Phase Interface at t={final_state['time']:.3f}s",
        filename=os.path.join(output_dir, 'interface_evolution.png')
    )

    print("Creating animation...")
    create_animation(history, filename=os.path.join(output_dir, 'animation.gif'), fps=10)

    print(f"\nSimulation complete!")
    print(f"Results saved to '{output_dir}/' directory")

    return history

def run_custom_case(nx=100, ny=100, t_end=10.0, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    L = 0.2
    dx = L / (nx - 1)

    print(f"Initializing custom case: {nx}x{ny} particles")
    ps = ParticleSystem(nx=nx, ny=ny, dx=dx)

    phase_model = PhaseChangeModel(
        T_melt=273.15,
        latent_heat=334000.0,
        c_p_solid=2100.0,
        c_p_liquid=4200.0,
        k_solid=2.0,
        k_liquid=0.5,
        rho=1000.0,
        interface_width=3.0 * dx
    )

    center_x = L / 2
    center_y = L / 2
    radius = L / 4

    for i in range(ps.n_particles):
        x, y = ps.positions[i]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        if dist < radius:
            ps.temperatures[i] = 263.15
        else:
            ps.temperatures[i] = 283.15

    ps.update_phase(phase_model.T_melt, phase_model.interface_width)

    solver = PhaseChangeSolver(ps, phase_model)

    dt = solver.compute_time_step(cfl=0.2)
    n_steps = int(t_end / dt)
    save_interval = max(1, n_steps // 50)

    print(f"Running custom simulation (t_end={t_end}s)...")

    history = []
    for step in tqdm(range(n_steps)):
        solver.step(dt)

        if step % save_interval == 0 or step == n_steps - 1:
            state = {
                'time': solver.time,
                'positions': ps.positions.copy(),
                'temperatures': ps.temperatures.copy(),
                'liquid_fraction': ps.liquid_fraction.copy(),
                'phase': ps.phase.copy(),
                'level_set': solver.interface.level_set.copy()
            }
            history.append(state)

    print("Generating visualizations...")
    final_state = history[-1]

    plot_temperature_field(
        final_state['positions'],
        final_state['temperatures'],
        final_state['phase'],
        title=f"Custom Case: Temperature at t={final_state['time']:.3f}s",
        filename=os.path.join(output_dir, 'custom_temperature.png')
    )

    plot_interface(
        final_state['positions'],
        final_state['level_set'],
        final_state['liquid_fraction'],
        title=f"Custom Case: Interface at t={final_state['time']:.3f}s",
        filename=os.path.join(output_dir, 'custom_interface.png')
    )

    create_animation(history, filename=os.path.join(output_dir, 'custom_animation.gif'), fps=10)

    print(f"Custom simulation complete! Results in '{output_dir}/'")

    return history

def main():
    parser = argparse.ArgumentParser(description='Phase Change Simulation using Particle Method')
    parser.add_argument('--case', type=str, default='stefan', choices=['stefan', 'custom'],
                       help='Simulation case: stefan or custom')
    parser.add_argument('--nx', type=int, default=50, help='Number of particles in x direction')
    parser.add_argument('--ny', type=int, default=50, help='Number of particles in y direction')
    parser.add_argument('--duration', type=float, default=2.0, help='Simulation duration (seconds)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    if args.case == 'stefan':
        run_stefan_case(nx=args.nx, ny=args.ny, t_end=args.duration, output_dir=args.output)
    elif args.case == 'custom':
        run_custom_case(nx=args.nx, ny=args.ny, t_end=args.duration, output_dir=args.output)

if __name__ == '__main__':
    main()
