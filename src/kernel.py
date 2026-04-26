import numpy as np
from numba import jit

@jit(nopython=True)
def wendland_c2(r: float, h: float) -> float:
    q = r / h
    if q >= 1.0:
        return 0.0
    factor = 7.0 / (4.0 * np.pi * h * h)
    return factor * (1.0 - q)**4 * (4.0 * q + 1.0)

@jit(nopython=True)
def wendland_c2_gradient(r_vec: np.ndarray, h: float) -> np.ndarray:
    r = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
    if r < 1e-10 or r >= h:
        return np.zeros(2)
    q = r / h
    factor = 7.0 / (4.0 * np.pi * h * h)
    dw_dq = factor * (-4.0 * (1.0 - q)**3 * (4.0 * q + 1.0) + (1.0 - q)**4 * 4.0)
    dq_dr = 1.0 / h
    return (dw_dq * dq_dr / r) * r_vec

def compute_kernel_matrix(positions: np.ndarray, h: float) -> np.ndarray:
    n = len(positions)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            W[i, j] = wendland_c2(r, h)
    return W

def compute_kernel_gradient_matrix(positions: np.ndarray, h: float) -> np.ndarray:
    n = len(positions)
    grad_W = np.zeros((n, n, 2))
    for i in range(n):
        for j in range(n):
            r_vec = positions[j] - positions[i]
            grad_W[i, j] = wendland_c2_gradient(r_vec, h)
    return grad_W
