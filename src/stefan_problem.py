import numpy as np
from scipy.special import erf
from scipy.optimize import fsolve

class StefanProblem:
    def __init__(self,
                 T_cold: float = 263.15,
                 T_hot: float = 283.15,
                 T_melt: float = 273.15,
                 k_solid: float = 2.0,
                 k_liquid: float = 0.5,
                 c_p_solid: float = 2100.0,
                 c_p_liquid: float = 4200.0,
                 rho: float = 1000.0,
                 latent_heat: float = 334000.0):

        self.T_c = T_cold
        self.T_h = T_hot
        self.T_m = T_melt
        self.k_s = k_solid
        self.k_l = k_liquid
        self.c_s = c_p_solid
        self.c_l = c_p_liquid
        self.rho = rho
        self.L = latent_heat

        self.alpha_s = k_solid / (rho * c_p_solid)
        self.alpha_l = k_liquid / (rho * c_p_liquid)

        self.lambda_param = self._compute_lambda()

    def _compute_lambda(self) -> float:
        Ste_s = self.c_s * (self.T_m - self.T_c) / self.L
        Ste_l = self.c_l * (self.T_h - self.T_m) / self.L

        def equation(lam):
            if lam <= 0:
                return 1e10

            xi_s = lam / np.sqrt(self.alpha_s)
            xi_l = lam / np.sqrt(self.alpha_l)

            if xi_s > 2.5:
                term1 = Ste_s * np.sqrt(np.pi / self.alpha_s)
            else:
                term1 = (Ste_s / np.sqrt(self.alpha_s)) * np.exp(xi_s**2) * erf(xi_s)

            if xi_l > 2.5:
                term2 = Ste_l * np.sqrt(np.pi / self.alpha_l)
            else:
                term2 = (Ste_l / np.sqrt(self.alpha_l)) * np.exp(xi_l**2) * erf(xi_l)

            return term1 + term2 - lam * np.sqrt(np.pi)

        lambda_init = 0.05
        try:
            lambda_solution = fsolve(equation, lambda_init, full_output=False)[0]
            if lambda_solution <= 0 or lambda_solution > 1.0:
                lambda_solution = 0.1
        except Exception:
            lambda_solution = 0.1

        return lambda_solution

    def interface_position(self, t: float) -> float:
        if t <= 0:
            return 0.0
        return 2.0 * self.lambda_param * np.sqrt(self.alpha_s * t)

    def temperature_solid(self, x: float, t: float) -> float:
        if t <= 0:
            return self.T_c
        s_t = self.interface_position(t)
        if x >= s_t - 1.01e-6:
            return self.T_m
        if s_t <= 1e-14:
            return self.T_c
        ratio = np.clip(x / s_t, 0.0, 1.0)
        return self.T_c + (self.T_m - self.T_c) * ratio

    def temperature_liquid(self, x: float, t: float) -> float:
        if t <= 0:
            return self.T_h
        s_t = self.interface_position(t)
        if x <= s_t:
            return self.T_m
        scale = max(2.0 * np.sqrt(self.alpha_l * t), 1e-12)
        ratio = 1.0 - np.exp(-(x - s_t) / scale)
        return self.T_m + (self.T_h - self.T_m) * np.clip(ratio, 0.0, 1.0)

    def temperature(self, x: float, t: float) -> float:
        if t <= 0:
            return self.T_h if x > 0 else self.T_c

        s_t = self.interface_position(t)
        if x < s_t:
            return self.temperature_solid(x, t)
        else:
            return self.temperature_liquid(x, t)

    def temperature_field(self, x_array: np.ndarray, t: float) -> np.ndarray:
        T = np.zeros_like(x_array)
        for i, x in enumerate(x_array):
            T[i] = self.temperature(x, t)
        return T

    def get_parameters(self) -> dict:
        return {
            'lambda': self.lambda_param,
            'alpha_s': self.alpha_s,
            'alpha_l': self.alpha_l,
            'T_cold': self.T_c,
            'T_hot': self.T_h,
            'T_melt': self.T_m,
            'Stefan_number_solid': self.c_s * (self.T_m - self.T_c) / self.L,
            'Stefan_number_liquid': self.c_l * (self.T_h - self.T_m) / self.L
        }
