import numpy as np
import os
from particle_system import ParticleSystem
from phase_change import PhaseChangeModel
from solver import PhaseChangeSolver
from visualize import plot_temperature_field, plot_interface, create_animation
from tqdm import tqdm

def run_demo_simulation(nx=60, ny=60, t_end=100.0, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    L = 0.01
    dx = L / (nx - 1)

    print(f"Initializing demo simulation: {nx}x{ny} particles")
    ps = ParticleSystem(nx=nx, ny=ny, dx=dx)

    phase_model = PhaseChangeModel(
        T_melt=273.15,
        latent_heat=334000.0,
        c_p_solid=2100.0,
        c_p_liquid=4200.0,
        k_solid=2.2,
        k_liquid=0.6,
        rho=1000.0,
        interface_width=1.0
    )

    print("Setting initial conditions...")
    for i in range(ps.n_particles):
        x, y = ps.positions[i]
        if x < L * 0.35:
            ps.temperatures[i] = 260.0
        else:
            ps.temperatures[i] = 286.0

    ps.update_phase(phase_model.T_melt, phase_model.interface_width)

    solver = PhaseChangeSolver(ps, phase_model, T_left=260.0, T_right=286.0)

    dt = solver.compute_time_step(cfl=0.25)
    n_steps = int(t_end / dt)
    save_interval = max(1, n_steps // 40)

    print(f"Simulation parameters:")
    print(f"  Domain: {L}m x {L}m")
    print(f"  Particle spacing: {dx:.6f}m")
    print(f"  Time step: {dt:.4f}s")
    print(f"  Total steps: {n_steps}")
    print(f"  Duration: {t_end}s")

    history = []

    print("\nRunning simulation...")
    for step in tqdm(range(n_steps), desc="Time steps"):
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
        title=f"Temperature Field at t={final_state['time']:.1f}s",
        filename=os.path.join(output_dir, 'temperature_field.png')
    )

    plot_interface(
        final_state['positions'],
        final_state['level_set'],
        final_state['liquid_fraction'],
        title=f"Phase Interface at t={final_state['time']:.1f}s",
        filename=os.path.join(output_dir, 'interface_evolution.png')
    )

    print("Creating animation...")
    create_animation(history, filename=os.path.join(output_dir, 'animation.gif'), fps=8)

    print(f"\nSimulation complete!")
    print(f"Results saved to '{output_dir}/' directory")

    return history

if __name__ == '__main__':
    print("Phase Change Simulation Demo")
    print("="*60)
    run_demo_simulation()
