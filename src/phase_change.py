import numpy as np
from typing import Dict

class PhaseChangeModel:
    def __init__(self,
                 T_melt: float = 273.15,
                 latent_heat: float = 334000.0,
                 c_p_solid: float = 2100.0,
                 c_p_liquid: float = 4200.0,
                 k_solid: float = 2.0,
                 k_liquid: float = 0.5,
                 rho: float = 1000.0,
                 interface_width: float = 0.1):

        self.T_melt = T_melt
        self.L = latent_heat
        self.c_p_s = c_p_solid
        self.c_p_l = c_p_liquid
        self.k_s = k_solid
        self.k_l = k_liquid
        self.rho = rho
        self.interface_width = interface_width

    def get_thermal_conductivity(self, liquid_fraction: float) -> float:
        return self.k_s + (self.k_l - self.k_s) * liquid_fraction

    def get_specific_heat(self, liquid_fraction: float) -> float:
        return self.c_p_s + (self.c_p_l - self.c_p_s) * liquid_fraction

    def get_thermal_diffusivity(self, liquid_fraction: float) -> float:
        k = self.get_thermal_conductivity(liquid_fraction)
        c_p = self.get_specific_heat(liquid_fraction)
        return k / (self.rho * c_p)

    def temperature_to_enthalpy(self, T: float, f: float) -> float:
        if f <= 0.0:
            return self.c_p_s * T
        if f >= 1.0:
            return self.c_p_s * self.T_melt + self.L + self.c_p_l * (T - self.T_melt)
        return self.c_p_s * self.T_melt + f * self.L

    def enthalpy_to_temperature(self, H: float) -> tuple:
        H_s = self.c_p_s * self.T_melt
        H_l = H_s + self.L

        if H < H_s:
            return H / self.c_p_s, 0.0
        if H > H_l:
            return self.T_melt + (H - H_l) / self.c_p_l, 1.0
        f = np.clip((H - H_s) / self.L, 0.0, 1.0)
        return self.T_melt, f

    def compute_source_term(self, T: np.ndarray, T_old: np.ndarray,
                           f: np.ndarray, f_old: np.ndarray, dt: float) -> np.ndarray:
        df_dt = (f - f_old) / dt
        source = -self.rho * self.L * df_dt
        return source

    def update_phase_field(self, T: np.ndarray) -> np.ndarray:
        f = np.clip(
            (T - (self.T_melt - self.interface_width / 2)) / self.interface_width,
            0.0, 1.0
        )
        return f

    def get_properties_array(self, liquid_fractions: np.ndarray) -> Dict[str, np.ndarray]:
        n = len(liquid_fractions)
        k = np.zeros(n)
        c_p = np.zeros(n)
        alpha = np.zeros(n)

        for i in range(n):
            k[i] = self.get_thermal_conductivity(liquid_fractions[i])
            c_p[i] = self.get_specific_heat(liquid_fractions[i])
            alpha[i] = self.get_thermal_diffusivity(liquid_fractions[i])

        return {'k': k, 'c_p': c_p, 'alpha': alpha}
