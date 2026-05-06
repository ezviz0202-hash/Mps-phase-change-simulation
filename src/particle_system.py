import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ParticleSystem:
    positions: np.ndarray
    velocities: np.ndarray
    temperatures: np.ndarray
    phase: np.ndarray
    liquid_fraction: np.ndarray
    masses: np.ndarray
    densities: np.ndarray
    particle_spacing: float
    kernel_radius: float

    def __init__(self, nx: int, ny: int, dx: float, kernel_support: float = 2.1):
        n_particles = nx * ny
        self.particle_spacing = dx
        self.kernel_radius = kernel_support * dx
        self.kernel_support = kernel_support

        x = np.linspace(0, (nx-1)*dx, nx)
        y = np.linspace(0, (ny-1)*dx, ny)
        X, Y = np.meshgrid(x, y)

        self.positions = np.column_stack([X.ravel(), Y.ravel()])
        self.velocities = np.zeros((n_particles, 2))
        self.temperatures = np.zeros(n_particles)
        self.phase = np.zeros(n_particles, dtype=int)
        self.liquid_fraction = np.zeros(n_particles)
        self.masses = np.ones(n_particles) * dx * dx * 1000.0
        self.densities = np.ones(n_particles) * 1000.0
        self.particle_spacings = np.ones(n_particles) * dx
        self.smoothing_lengths = np.ones(n_particles) * self.kernel_radius

    @classmethod
    def from_positions(cls, positions: np.ndarray, spacings: np.ndarray, kernel_support: float = 2.1, rho: float = 1000.0):
        obj = cls.__new__(cls)
        positions = np.asarray(positions, dtype=float)
        spacings = np.asarray(spacings, dtype=float)
        n_particles = len(positions)
        obj.positions = positions
        obj.velocities = np.zeros((n_particles, 2))
        obj.temperatures = np.zeros(n_particles)
        obj.phase = np.zeros(n_particles, dtype=int)
        obj.liquid_fraction = np.zeros(n_particles)
        obj.densities = np.ones(n_particles) * rho
        obj.particle_spacings = spacings
        obj.particle_spacing = float(np.min(spacings))
        obj.kernel_support = kernel_support
        obj.smoothing_lengths = kernel_support * spacings
        obj.kernel_radius = float(np.max(obj.smoothing_lengths))
        obj.masses = rho * spacings * spacings
        return obj

    @property
    def n_particles(self) -> int:
        return len(self.positions)

    def get_neighbors(self, particle_idx: int) -> np.ndarray:
        pos = self.positions[particle_idx]
        distances = np.linalg.norm(self.positions - pos, axis=1)
        if hasattr(self, 'smoothing_lengths'):
            h = np.maximum(self.smoothing_lengths[particle_idx], self.smoothing_lengths)
            return np.where(distances < h)[0]
        return np.where(distances < self.kernel_radius)[0]

    def get_all_neighbors(self) -> List[np.ndarray]:
        neighbors = []
        for i in range(self.n_particles):
            neighbors.append(self.get_neighbors(i))
        return neighbors

    def update_phase(self, T_melt: float, T_width: float = 0.1):
        self.liquid_fraction = np.clip(
            (self.temperatures - (T_melt - T_width/2)) / T_width,
            0.0, 1.0
        )
        self.phase = (self.liquid_fraction > 0.5).astype(int)
