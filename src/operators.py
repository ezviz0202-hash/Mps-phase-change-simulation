import numpy as np
from typing import List
from particle_system import ParticleSystem
from kernel import wendland_c2, wendland_c2_gradient

class ParticleOperators:
    def __init__(self, particle_system: ParticleSystem):
        self.ps = particle_system

    def _pair_h(self, i: int, j: int) -> float:
        if hasattr(self.ps, 'smoothing_lengths'):
            return 0.5 * (self.ps.smoothing_lengths[i] + self.ps.smoothing_lengths[j])
        return self.ps.kernel_radius

    def _pair_spacing(self, i: int, j: int) -> float:
        if hasattr(self.ps, 'particle_spacings'):
            return 0.5 * (self.ps.particle_spacings[i] + self.ps.particle_spacings[j])
        return self.ps.particle_spacing

    def compute_gradient(self, field: np.ndarray, particle_idx: int, neighbors: np.ndarray) -> np.ndarray:
        if len(neighbors) < 3:
            return np.zeros(2)

        pos_i = self.ps.positions[particle_idx]
        grad = np.zeros(2)
        correction = np.zeros((2, 2))

        for j in neighbors:
            if j == particle_idx:
                continue
            r_vec = self.ps.positions[j] - pos_i
            r = np.linalg.norm(r_vec)
            if r < 1e-10:
                continue

            h = self._pair_h(particle_idx, j)
            grad_w = wendland_c2_gradient(r_vec, h)
            volume_j = self.ps.masses[j] / self.ps.densities[j]
            grad += (field[j] - field[particle_idx]) * grad_w * volume_j
            correction += np.outer(r_vec, grad_w) * volume_j

        try:
            if abs(np.linalg.det(correction)) > 1e-14:
                return np.linalg.solve(correction, grad)
        except np.linalg.LinAlgError:
            pass
        return grad

    def compute_laplacian(self, field: np.ndarray, particle_idx: int, neighbors: np.ndarray) -> float:
        if len(neighbors) < 3:
            return 0.0

        pos_i = self.ps.positions[particle_idx]
        numerator = 0.0
        denominator = 0.0
        dim = 2

        for j in neighbors:
            if j == particle_idx:
                continue
            r_vec = self.ps.positions[j] - pos_i
            r = np.linalg.norm(r_vec)
            if r < 1e-10:
                continue

            h = self._pair_h(particle_idx, j)
            spacing = self._pair_spacing(particle_idx, j)
            w = wendland_c2(r, h)
            volume_j = self.ps.masses[j] / self.ps.densities[j]
            coeff = 2.0 * dim * w * volume_j / (r * r + 0.01 * spacing * spacing)
            numerator += (field[j] - field[particle_idx]) * coeff
            denominator += coeff

        if denominator > 1e-10:
            return numerator / denominator
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
