import csv
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from particle_system import ParticleSystem
from phase_change import PhaseChangeModel
from stefan_problem import StefanProblem


def make_output_dir():
    out_dir = ROOT / 'results' / 'static_refinement'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def x_grid_uniform(length, dx):
    n = int(round(length / dx)) + 1
    return np.linspace(0.0, length, n)


def x_grid_refined(length, dx_fine, dx_coarse, refine_end):
    left = list(np.arange(0.0, refine_end + 0.5 * dx_fine, dx_fine))
    right_start = left[-1] + dx_coarse
    right = list(np.arange(right_start, length + 0.5 * dx_coarse, dx_coarse))
    x = np.array(left + right)
    if x[-1] < length:
        x = np.append(x, length)
    x[-1] = length
    return np.unique(np.round(x, 12))


def case_grid(case, length=0.05):
    dx_coarse = 0.0025
    dx_medium = 0.0010
    dx_fine = 0.00025
    if case == 'uniform_coarse':
        x = x_grid_uniform(length, dx_coarse)
        spacing = np.ones_like(x) * dx_coarse
    elif case == 'uniform_fine':
        x = x_grid_uniform(length, dx_fine)
        spacing = np.ones_like(x) * dx_fine
    elif case == 'interface_refined':
        x = x_grid_refined(length, dx_fine, dx_medium, 0.004)
        spacing = np.where(x <= 0.004 + 1e-12, dx_fine, dx_medium)
    else:
        raise ValueError(case)
    return x, spacing


def make_particle_system(case, length=0.05, height=0.002):
    x, spacing_x = case_grid(case, length)
    y = np.array([0.0, height, 2.0 * height])
    positions = []
    spacings = []
    for yy in y:
        for xx, ss in zip(x, spacing_x):
            positions.append([xx, yy])
            spacings.append(ss)
    return ParticleSystem.from_positions(np.array(positions), np.array(spacings), kernel_support=2.6)


def cell_widths(x):
    faces = np.empty(len(x) + 1)
    faces[1:-1] = 0.5 * (x[:-1] + x[1:])
    faces[0] = x[0] - 0.5 * (x[1] - x[0])
    faces[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return faces[1:] - faces[:-1]


def temperature_to_enthalpy_array(model, T, f):
    H = np.empty_like(T)
    for i in range(len(T)):
        H[i] = model.temperature_to_enthalpy(float(T[i]), float(f[i]))
    return H


def enthalpy_to_temperature_array(model, H):
    T = np.empty_like(H)
    f = np.empty_like(H)
    for i in range(len(H)):
        T[i], f[i] = model.enthalpy_to_temperature(float(H[i]))
    return T, f


def effective_conductivity(model, f):
    return model.k_s + (model.k_l - model.k_s) * f


def estimate_interface_from_temperature(x, T, T_melt):
    diff = T - T_melt
    changes = np.where(diff[:-1] * diff[1:] <= 0.0)[0]
    if len(changes) == 0:
        return float(x[np.argmin(np.abs(diff))])
    i = int(changes[0])
    x0, x1 = x[i], x[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    if abs(y1 - y0) < 1e-14:
        return float(0.5 * (x0 + x1))
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def solve_finite_volume_enthalpy(x, stefan, model, t0=1.0, t_end=20.0, cfl=0.35, dt_override=None):
    T = stefan.temperature_field(x, t0)
    f = model.update_phase_field(T)
    H = temperature_to_enthalpy_array(model, T, f)
    widths = cell_widths(x)
    dx_min = float(np.min(np.diff(x)))
    alpha_max = max(model.k_s / (model.rho * model.c_p_s), model.k_l / (model.rho * model.c_p_l))
    if dt_override is None:
        dt = cfl * dx_min * dx_min / alpha_max
    else:
        dt = float(dt_override)
    n_steps = int(np.ceil((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps
    t = t0
    for _ in range(n_steps):
        T, f = enthalpy_to_temperature_array(model, H)
        T[0] = stefan.T_c
        T[-1] = stefan.T_h
        H[0] = model.temperature_to_enthalpy(T[0], 0.0)
        H[-1] = model.temperature_to_enthalpy(T[-1], 1.0)
        k = effective_conductivity(model, f)
        flux = np.zeros(len(x) - 1)
        for i in range(len(flux)):
            k_face = 2.0 * k[i] * k[i + 1] / max(k[i] + k[i + 1], 1e-14)
            flux[i] = k_face * (T[i + 1] - T[i]) / (x[i + 1] - x[i])
        H_new = H.copy()
        for i in range(1, len(x) - 1):
            H_new[i] = H[i] + dt * (flux[i] - flux[i - 1]) / (model.rho * widths[i])
        H = H_new
        t += dt
    T, f = enthalpy_to_temperature_array(model, H)
    T[0] = stefan.T_c
    T[-1] = stefan.T_h
    return T, f, n_steps, dt


def resample_to_reference(x, T, x_ref):
    return np.interp(x_ref, x, T)


def run_case(case, t0=1.0, t_end=5.0, dt_override=None):
    model = PhaseChangeModel(interface_width=0.5)
    stefan = StefanProblem()
    ps = make_particle_system(case)
    x, spacing = case_grid(case)
    start = time.perf_counter()
    T, f, steps, dt = solve_finite_volume_enthalpy(x, stefan, model, t0=t0, t_end=t_end, dt_override=dt_override)
    runtime = time.perf_counter() - start
    x_ref = np.linspace(0.0, x[-1], 1000)
    T_interp = resample_to_reference(x, T, x_ref)
    T_exact = stefan.temperature_field(x_ref, t_end)
    l2_error = float(np.sqrt(np.mean((T_interp - T_exact) ** 2)))
    interface_num = estimate_interface_from_temperature(x_ref, T_interp, model.T_melt)
    interface_exact = float(stefan.interface_position(t_end))
    ps.temperatures = np.tile(T, 3)
    ps.liquid_fraction = np.tile(f, 3)
    ps.phase = (ps.liquid_fraction > 0.5).astype(int)
    return {
        'case': case,
        'particles': ps.n_particles,
        'x_points': len(x),
        'steps': steps,
        'dt': dt,
        'runtime': runtime,
        'l2_error': l2_error,
        'interface_error': abs(interface_num - interface_exact),
        'interface_position': interface_num,
        'exact_interface': interface_exact,
        'x': x,
        'temperature': T,
        'x_ref': x_ref,
        'temperature_exact': T_exact,
        'particle_x': ps.positions[:, 0],
        'particle_y': ps.positions[:, 1]
    }


def add_reference_errors(results):
    ref = next(r for r in results if r['case'] == 'uniform_fine')
    x_ref = ref['x_ref']
    T_ref = np.interp(x_ref, ref['x'], ref['temperature'])
    interface_ref = ref['interface_position']
    for r in results:
        T = np.interp(x_ref, r['x'], r['temperature'])
        r['l2_reference_error'] = float(np.sqrt(np.mean((T - T_ref) ** 2)))
        r['interface_reference_error'] = abs(float(r['interface_position']) - float(interface_ref))
    return results


def save_summary(results, out_dir):
    path = out_dir / 'refinement_summary.csv'
    fields = ['case', 'particles', 'x_points', 'steps', 'dt', 'runtime', 'l2_reference_error', 'interface_reference_error', 'l2_error', 'interface_error', 'interface_position', 'exact_interface']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    return path


def plot_layout(results, out_dir):
    plt.figure(figsize=(8, 3.5))
    for idx, r in enumerate(results):
        y = np.ones_like(r['particle_x']) * idx
        plt.scatter(r['particle_x'], y, s=8)
    plt.yticks(range(len(results)), [r['case'] for r in results])
    plt.xlabel('x position')
    plt.title('Static multi-resolution particle arrangements')
    plt.tight_layout()
    plt.savefig(out_dir / 'refinement_layout.png', dpi=200)
    plt.close()


def plot_errors(results, out_dir):
    labels = [r['case'] for r in results]
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.18, [max(r['l2_error'], 1e-12) for r in results], width=0.36, label='L2 error to analytical solution')
    plt.bar(x + 0.18, [max(r['interface_error'], 1e-12) for r in results], width=0.36, label='Interface error to analytical solution')
    plt.xticks(x, labels, rotation=15)
    plt.yscale('log')
    plt.ylabel('Error')
    plt.title('Accuracy comparison against Stefan solution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'refinement_error_comparison.png', dpi=200)
    plt.close()


def plot_reference_errors(results, out_dir):
    labels = [r['case'] for r in results]
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.18, [max(r['l2_reference_error'], 1e-12) for r in results], width=0.36, label='L2 difference to uniform fine')
    plt.bar(x + 0.18, [max(r['interface_reference_error'], 1e-12) for r in results], width=0.36, label='Interface difference to uniform fine')
    plt.xticks(x, labels, rotation=15)
    plt.yscale('log')
    plt.ylabel('Difference')
    plt.title('Agreement with fine-resolution numerical reference')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'refinement_reference_error_comparison.png', dpi=200)
    plt.close()


def plot_cost(results, out_dir):
    labels = [r['case'] for r in results]
    fine = next(r for r in results if r['case'] == 'uniform_fine')
    x = np.arange(len(labels))
    particle_ratio = [r['particles'] / fine['particles'] for r in results]
    runtime_ratio = [r['runtime'] / fine['runtime'] for r in results]
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.18, particle_ratio, width=0.36, label='Particle count relative to uniform fine')
    plt.bar(x + 0.18, runtime_ratio, width=0.36, label='Runtime relative to uniform fine')
    plt.xticks(x, labels, rotation=15)
    plt.ylabel('Relative cost')
    plt.title('Cost comparison relative to uniform fine')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'refinement_cost_comparison.png', dpi=200)
    plt.close()


def plot_temperature(results, out_dir):
    plt.figure(figsize=(8, 4))
    for r in results:
        plt.plot(r['x'], r['temperature'], marker='o', markersize=3, label=r['case'])
    plt.plot(results[0]['x_ref'], results[0]['temperature_exact'], linewidth=2, label='analytical')
    plt.xlabel('x position')
    plt.ylabel('Temperature K')
    plt.title('Temperature profiles at final time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'refinement_temperature_profiles.png', dpi=200)
    plt.close()


def main():
    out_dir = make_output_dir()
    cases = ['uniform_coarse', 'interface_refined', 'uniform_fine']
    model = PhaseChangeModel(interface_width=0.5)
    x_fine, _ = case_grid('uniform_fine')
    alpha_max = max(model.k_s / (model.rho * model.c_p_s), model.k_l / (model.rho * model.c_p_l))
    dx_fine = float(np.min(np.diff(x_fine)))
    dt_common = 0.35 * dx_fine * dx_fine / alpha_max
    results = [run_case(case, dt_override=dt_common) for case in cases]
    add_reference_errors(results)
    save_summary(results, out_dir)
    plot_layout(results, out_dir)
    plot_errors(results, out_dir)
    plot_reference_errors(results, out_dir)
    plot_cost(results, out_dir)
    plot_temperature(results, out_dir)
    print('Static refinement enthalpy study complete')
    print(f'Results saved to {out_dir}')
    for r in results:
        print(f"{r['case']}: particles={r['particles']}, steps={r['steps']}, l2_error={r['l2_error']:.6e}, interface_error={r['interface_error']:.6e}, l2_reference_error={r['l2_reference_error']:.6e}, runtime={r['runtime']:.6e}s")


if __name__ == '__main__':
    main()
