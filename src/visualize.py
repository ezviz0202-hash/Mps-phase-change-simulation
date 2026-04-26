import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import os

def plot_temperature_field(positions, temperatures, phase, title="Temperature Field",
                          filename=None, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(positions[:, 0], positions[:, 1],
                        c=temperatures, s=50, cmap='coolwarm',
                        edgecolors='black', linewidth=0.5, alpha=0.8)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Temperature (K)', fontsize=12)

    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_interface(positions, level_set, liquid_fraction, title="Phase Interface",
                  filename=None, figsize=(10, 8)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    interface_particles = np.abs(level_set) < 0.02

    sc1 = ax1.scatter(positions[:, 0], positions[:, 1],
                     c=liquid_fraction, s=50, cmap='RdYlBu_r',
                     vmin=0, vmax=1, edgecolors='black', linewidth=0.5, alpha=0.8)
    ax1.scatter(positions[interface_particles, 0],
               positions[interface_particles, 1],
               c='red', s=100, marker='x', linewidth=2, label='Interface')

    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Liquid Fraction', fontsize=12)
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_title('Liquid Fraction Field', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    sc2 = ax2.scatter(positions[:, 0], positions[:, 1],
                     c=level_set, s=50, cmap='seismic',
                     vmin=-0.5, vmax=0.5, edgecolors='black', linewidth=0.5, alpha=0.8)

    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Level Set', fontsize=12)
    ax2.set_xlabel('x (m)', fontsize=12)
    ax2.set_ylabel('y (m)', fontsize=12)
    ax2.set_title('Level Set Field', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_animation(history, filename='animation.gif', fps=10, figsize=(12, 5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    positions = history[0]['positions']

    all_temps = np.concatenate([s['temperatures'] for s in history])
    t_min, t_max = np.min(all_temps), np.max(all_temps)

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    def update(frame):
        ax1.clear()
        ax2.clear()

        state = history[frame]

        sc1 = ax1.scatter(positions[:, 0], positions[:, 1],
                         c=state['temperatures'], s=50, cmap='coolwarm',
                         vmin=t_min, vmax=t_max,
                         edgecolors='black', linewidth=0.5, alpha=0.8)
        ax1.set_xlabel('x (m)', fontsize=11)
        ax1.set_ylabel('y (m)', fontsize=11)
        ax1.set_title(f'Temperature Field (t={state["time"]:.3f}s)', fontsize=11, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.grid(True, alpha=0.3)

        interface_particles = np.abs(state['level_set']) < 0.02
        sc2 = ax2.scatter(positions[:, 0], positions[:, 1],
                         c=state['liquid_fraction'], s=50, cmap='RdYlBu_r',
                         vmin=0, vmax=1, edgecolors='black', linewidth=0.5, alpha=0.8)
        ax2.scatter(positions[interface_particles, 0],
                   positions[interface_particles, 1],
                   c='red', s=80, marker='x', linewidth=2)
        ax2.set_xlabel('x (m)', fontsize=11)
        ax2.set_ylabel('y (m)', fontsize=11)
        ax2.set_title('Liquid Fraction & Interface', fontsize=11, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.grid(True, alpha=0.3)

        return sc1, sc2

    anim = FuncAnimation(fig, update, frames=len(history), interval=1000/fps, blit=False)

    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close()

    print(f"Animation saved to {filename}")

def plot_1d_comparison(x_sim, T_sim, x_analytical, T_analytical, time,
                       interface_sim=None, interface_analytical=None,
                       filename=None, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_analytical, T_analytical, 'r-', linewidth=2, label='Analytical Solution')
    ax.plot(x_sim, T_sim, 'bo', markersize=6, label='Particle Method', alpha=0.7)

    if interface_analytical is not None:
        ax.axvline(interface_analytical, color='red', linestyle='--',
                  linewidth=2, label=f'Interface (Analytical): {interface_analytical:.4f}m')

    if interface_sim is not None:
        ax.axvline(interface_sim, color='blue', linestyle='--',
                  linewidth=2, label=f'Interface (Simulation): {interface_sim:.4f}m')

    ax.set_xlabel('Position x (m)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title(f'Stefan Problem Comparison (t = {time:.3f}s)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_convergence(resolutions, errors, orders=None, filename=None, figsize=(10, 6)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.loglog(resolutions, errors, 'bo-', linewidth=2, markersize=8, label='L2 Error')

    if len(resolutions) > 1:
        slope = -2.0
        ref_line = errors[0] * (np.array(resolutions) / resolutions[0])**slope
        ax1.loglog(resolutions, ref_line, 'r--', linewidth=2, label='2nd Order Reference')

    ax1.set_xlabel('Number of Particles (N)', fontsize=12)
    ax1.set_ylabel('L2 Error', fontsize=12)
    ax1.set_title('Convergence Analysis', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')

    if orders is not None and len(orders) > 0:
        ax2.plot(resolutions[1:], orders, 'gs-', linewidth=2, markersize=8)
        ax2.axhline(2.0, color='r', linestyle='--', linewidth=2, label='2nd Order')
        ax2.set_xlabel('Number of Particles (N)', fontsize=12)
        ax2.set_ylabel('Convergence Order', fontsize=12)
        ax2.set_title('Spatial Accuracy Order', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 3])

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
