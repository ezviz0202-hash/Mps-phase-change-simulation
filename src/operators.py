import numpy as np
from typing import List
from particle_system import ParticleSystem
from kernel import wendland_c2, wendland_c2_gradient

class ParticleOperators:
    def __init__(self, particle_system: ParticleSystem):
        self.ps = particle_system

    def compute_gradient(self, field: np.ndarray, particle_idx: int, neighbors: np.ndarray) -> np.ndarray:
        if len(neighbors) < 3:
            return np.zeros(2)

        pos_i = self.ps.positions[particle_idx]
        grad = np.zeros(2)
        weight_sum = 0.0

        for j in neighbors:
            if j == particle_idx:
                continue
            r_vec = self.ps.positions[j] - pos_i
            r = np.linalg.norm(r_vec)
            if r < 1e-10:
                continue

            grad_w = wendland_c2_gradient(r_vec, self.ps.kernel_radius)
            volume_j = self.ps.masses[j] / self.ps.densities[j]
            grad += (field[j] - field[particle_idx]) * grad_w * volume_j
            weight_sum += np.linalg.norm(grad_w) * volume_j

        if weight_sum > 1e-10:
            return grad
        return np.zeros(2)

    def compute_laplacian(self, field: np.ndarray, particle_idx: int, neighbors: np.ndarray) -> float:
        if len(neighbors) < 3:
            return 0.0

        pos_i = self.ps.positions[particle_idx]
        laplacian = 0.0
        weight_sum = 0.0

        dim = 2
        for j in neighbors:
            if j == particle_idx:
                continue
            r_vec = self.ps.positions[j] - pos_i
            r = np.linalg.norm(r_vec)
            if r < 1e-10:
                continue

            w = wendland_c2(r, self.ps.kernel_radius)
            volume_j = self.ps.masses[j] / self.ps.densities[j]

            laplacian += 2.0 * dim * (field[j] - field[particle_idx]) * w * volume_j / (r * r + 0.01 * self.ps.particle_spacing**2)
            weight_sum += 2.0 * dim * w * volume_j / (r * r + 0.01 * self.ps.particle_spacing**2)

        if weight_sum > 1e-10:
            return laplacian / weight_sum
        return 0.0

    def compute_all_gradients(self, field: np.ndarray, neighbors_list: List[np.ndarray]) -> np.ndarray:
        gradients = np.zeros((self.ps.n_particles, 2))
        for i in range(self.ps.n_particles):
            gradients[i] = self.compute_gradient(field, i, neighbors_list[i])
        return gradients

    def compute_all_laplacians(self, field: np.ndarray, neighbors_list: List[np.ndarray]) -> np.ndarray:
        laplacians = np.zeros(self.ps.n_particles)
        for i in range(self.ps.n_particles):
            laplacians[i] = self.compute_laplacian(field, i, neighbors_list[i])
        return laplacians
