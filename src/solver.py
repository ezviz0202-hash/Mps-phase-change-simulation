import numpy as np
from typing import List, Tuple
from particle_system import ParticleSystem
from operators import ParticleOperators
from phase_change import PhaseChangeModel
from interface_tracker import InterfaceTracker

class PhaseChangeSolver:
    def __init__(self, particle_system: ParticleSystem, phase_model: PhaseChangeModel,
                 T_left: float = 263.15, T_right: float = 283.15):
        self.ps = particle_system
        self.model = phase_model
        self.operators = ParticleOperators(particle_system)
        self.interface = InterfaceTracker(particle_system)
        self.time = 0.0
        self.dt = 0.0
        self.T_left = T_left
        self.T_right = T_right

    def compute_time_step(self, cfl: float = 0.2) -> float:
        max_alpha = max(self.model.k_s / (self.model.rho * self.model.c_p_s),
                       self.model.k_l / (self.model.rho * self.model.c_p_l))
        dt = cfl * self.ps.particle_spacing**2 / max_alpha
        return dt

    def apply_boundary_conditions(self, T: np.ndarray, bc_type: str = 'stefan'):
        if bc_type == 'stefan':
            x_min = self.ps.positions[:, 0].min()
            x_max = self.ps.positions[:, 0].max()

            for i in range(self.ps.n_particles):
                x = self.ps.positions[i, 0]
                if x <= x_min + 0.5 * self.ps.particle_spacing:
                    T[i] = self.T_left
                elif x >= x_max - 0.5 * self.ps.particle_spacing:
                    T[i] = self.T_right

    def solve_heat_equation(self, dt: float, neighbors_list: List[np.ndarray]) -> np.ndarray:
        T_new = self.ps.temperatures.copy()
        laplacians = self.operators.compute_all_laplacians(self.ps.temperatures, neighbors_list)

        for i in range(self.ps.n_particles):
            f = self.ps.liquid_fraction[i]
            alpha = self.model.get_thermal_diffusivity(f)
            c_p = self.model.get_specific_heat(f)

            dT_dt = alpha * laplacians[i]
            T_new[i] = self.ps.temperatures[i] + dt * dT_dt

        return T_new

    def solve_enthalpy_method(self, dt: float, neighbors_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        H = np.zeros(self.ps.n_particles)
        for i in range(self.ps.n_particles):
            H[i] = self.model.temperature_to_enthalpy(
                self.ps.temperatures[i],
                self.ps.liquid_fraction[i]
            )

        laplacians = self.operators.compute_all_laplacians(self.ps.temperatures, neighbors_list)

        for i in range(self.ps.n_particles):
            f = self.ps.liquid_fraction[i]
            k = self.model.get_thermal_conductivity(f)

            dH_dt = k / self.model.rho * laplacians[i]
            H[i] = H[i] + dt * dH_dt

        T_new = np.zeros(self.ps.n_particles)
        f_new = np.zeros(self.ps.n_particles)
        for i in range(self.ps.n_particles):
            T_new[i], f_new[i] = self.model.enthalpy_to_temperature(H[i])

        T_new = np.clip(T_new, 250.0, 300.0)

        return T_new, f_new

    def step(self, dt: float = None) -> float:
        if dt is None:
            dt = self.compute_time_step()

        self.dt = dt
        neighbors_list = self.ps.get_all_neighbors()

        self.apply_boundary_conditions(self.ps.temperatures)

        T_new, f_new = self.solve_enthalpy_method(dt, neighbors_list)

        self.ps.temperatures = T_new
        self.ps.liquid_fraction = f_new
        self.ps.phase = (self.ps.liquid_fraction > 0.5).astype(int)

        self.apply_boundary_conditions(self.ps.temperatures)

        self.interface.update_from_phase()
        self.interface.compute_interface_normal(neighbors_list)

        self.time += dt
        return dt

    def solve(self, t_end: float, dt: float = None, callback=None) -> List[dict]:
        history = []

        while self.time < t_end:
            if dt is None:
                dt_actual = self.compute_time_step()
            else:
                dt_actual = dt

            if self.time + dt_actual > t_end:
                dt_actual = t_end - self.time

            self.step(dt_actual)

            state = {
                'time': self.time,
                'positions': self.ps.positions.copy(),
                'temperatures': self.ps.temperatures.copy(),
                'liquid_fraction': self.ps.liquid_fraction.copy(),
                'phase': self.ps.phase.copy(),
                'level_set': self.interface.level_set.copy()
            }
            history.append(state)

            if callback is not None:
                callback(state)

        return history
