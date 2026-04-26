import numpy as np
from typing import List
from particle_system import ParticleSystem
from operators import ParticleOperators

class InterfaceTracker:
    def __init__(self, particle_system: ParticleSystem):
        self.ps = particle_system
        self.level_set = np.zeros(particle_system.n_particles)
        self.interface_normal = np.zeros((particle_system.n_particles, 2))
        self.interface_curvature = np.zeros(particle_system.n_particles)

    def initialize_from_temperature(self, T_melt: float):
        self.level_set = self.ps.temperatures - T_melt

    def initialize_from_liquid_fraction(self):
        self.level_set = self.ps.liquid_fraction - 0.5

    def update_from_phase(self):
        self.level_set = self.ps.liquid_fraction - 0.5

    def compute_interface_normal(self, neighbors_list: List[np.ndarray]):
        operators = ParticleOperators(self.ps)
        gradients = operators.compute_all_gradients(self.level_set, neighbors_list)

        for i in range(self.ps.n_particles):
            grad_mag = np.linalg.norm(gradients[i])
            if grad_mag > 1e-10:
                self.interface_normal[i] = gradients[i] / grad_mag
            else:
                self.interface_normal[i] = np.zeros(2)

    def compute_interface_curvature(self, neighbors_list: List[np.ndarray]):
        operators = ParticleOperators(self.ps)

        for i in range(self.ps.n_particles):
            grad_mag = np.linalg.norm(self.interface_normal[i])
            if grad_mag > 1e-10:
                div_n = operators.compute_laplacian(self.level_set, i, neighbors_list[i])
                self.interface_curvature[i] = -div_n / (grad_mag + 1e-10)
            else:
                self.interface_curvature[i] = 0.0

    def get_interface_particles(self, threshold: float = 0.02) -> np.ndarray:
        return np.where(np.abs(self.level_set) < threshold)[0]

    def get_interface_position(self) -> np.ndarray:
        interface_idx = self.get_interface_particles()
        if len(interface_idx) == 0:
            return np.array([])
        return self.ps.positions[interface_idx]

    def reinitialize(self, neighbors_list: List[np.ndarray], n_iterations: int = 5, dt: float = 0.1):
        operators = ParticleOperators(self.ps)

        for _ in range(n_iterations):
            gradients = operators.compute_all_gradients(self.level_set, neighbors_list)

            for i in range(self.ps.n_particles):
                grad_mag = np.linalg.norm(gradients[i])
                sign = np.sign(self.level_set[i])
                if np.abs(self.level_set[i]) < 1e-10:
                    sign = 0.0

                self.level_set[i] -= dt * sign * (grad_mag - 1.0)
