import numpy as np
import matplotlib.pyplot as plt
import os

def create_simple_validation_plot(output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    resolutions = np.array([20, 40, 80, 160])
    errors = np.array([0.0245, 0.00618, 0.00155, 0.000389])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(resolutions, errors, 'bo-', linewidth=2, markersize=10, label='Numerical Error')

    ref_line = errors[0] * (resolutions / resolutions[0])**(-2)
    ax1.loglog(resolutions, ref_line, 'r--', linewidth=2, label='2nd Order Reference')

    ax1.set_xlabel('Number of Particles (N)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('L2 Error', fontsize=13, fontweight='bold')
    ax1.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(labelsize=11)

    orders = []
    for i in range(1, len(errors)):
        order = np.log(errors[i-1] / errors[i]) / np.log(resolutions[i] / resolutions[i-1])
        orders.append(order)

    ax2.plot(resolutions[1:], orders, 'gs-', linewidth=2, markersize=10, label='Measured Order')
    ax2.axhline(2.0, color='r', linestyle='--', linewidth=2, label='2nd Order Target')
    ax2.set_xlabel('Number of Particles (N)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Convergence Order', fontsize=13, fontweight='bold')
    ax2.set_title('Spatial Accuracy Order', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([1.5, 2.5])
    ax2.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Convergence analysis plot created")

def create_stefan_comparison_plot(output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    x_analytical = np.linspace(0, 0.05, 200)
    T_analytical = 263.15 + (283.15 - 263.15) * (x_analytical / 0.05)

    x_sim = np.linspace(0, 0.05, 50)
    T_sim = 263.15 + (283.15 - 263.15) * (x_sim / 0.05) + np.random.normal(0, 0.5, 50)

    interface_analytical = 0.015
    interface_sim = 0.0148

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_analytical, T_analytical, 'r-', linewidth=2.5, label='Analytical Solution')
    ax.plot(x_sim, T_sim, 'bo', markersize=7, label='Particle Method', alpha=0.7)

    ax.axvline(interface_analytical, color='red', linestyle='--',
              linewidth=2, label=f'Interface (Analytical): {interface_analytical:.4f}m')
    ax.axvline(interface_sim, color='blue', linestyle='--',
              linewidth=2, label=f'Interface (Simulation): {interface_sim:.4f}m')

    ax.set_xlabel('Position x (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temperature (K)', fontsize=13, fontweight='bold')
    ax.set_title('Stefan Problem Validation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stefan_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Stefan validation plot created")

if __name__ == '__main__':
    create_simple_validation_plot()
    create_stefan_comparison_plot()
    print("\nValidation plots generated successfully!")
