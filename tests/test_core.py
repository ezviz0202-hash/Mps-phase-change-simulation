import unittest
import numpy as np
from particle_system import ParticleSystem
from phase_change import PhaseChangeModel
from stefan_problem import StefanProblem
from kernel import wendland_c2, wendland_c2_gradient

class TestKernel(unittest.TestCase):
    def test_wendland_c2_support(self):
        h = 1.0
        self.assertAlmostEqual(wendland_c2(0.0, h), 7.0 / (4.0 * np.pi), places=5)
        self.assertAlmostEqual(wendland_c2(h, h), 0.0, places=10)
        self.assertAlmostEqual(wendland_c2(2*h, h), 0.0, places=10)

    def test_wendland_gradient(self):
        h = 1.0
        r_vec = np.array([0.0, 0.0])
        grad = wendland_c2_gradient(r_vec, h)
        self.assertTrue(np.allclose(grad, np.zeros(2)))

class TestParticleSystem(unittest.TestCase):
    def test_initialization(self):
        ps = ParticleSystem(nx=10, ny=10, dx=0.01)
        self.assertEqual(ps.n_particles, 100)
        self.assertEqual(ps.positions.shape, (100, 2))
        self.assertEqual(ps.temperatures.shape, (100,))

    def test_phase_update(self):
        ps = ParticleSystem(nx=5, ny=5, dx=0.01)
        ps.temperatures = np.ones(25) * 273.15
        ps.update_phase(T_melt=273.15, T_width=0.1)
        self.assertTrue(np.all(ps.liquid_fraction >= 0.0))
        self.assertTrue(np.all(ps.liquid_fraction <= 1.0))

class TestPhaseChangeModel(unittest.TestCase):
    def test_thermal_properties(self):
        model = PhaseChangeModel()
        k_solid = model.get_thermal_conductivity(0.0)
        k_liquid = model.get_thermal_conductivity(1.0)
        self.assertAlmostEqual(k_solid, model.k_s)
        self.assertAlmostEqual(k_liquid, model.k_l)

    def test_enthalpy_conversion(self):
        model = PhaseChangeModel()
        T = 273.15
        f = 0.5
        H = model.temperature_to_enthalpy(T, f)
        T_back, f_back = model.enthalpy_to_temperature(H)
        self.assertAlmostEqual(T, T_back, places=3)
        self.assertAlmostEqual(f, f_back, places=3)

class TestStefanProblem(unittest.TestCase):
    def test_interface_position(self):
        stefan = StefanProblem()
        s_0 = stefan.interface_position(0.0)
        self.assertEqual(s_0, 0.0)

        s_1 = stefan.interface_position(1.0)
        self.assertGreater(s_1, 0.0)

    def test_temperature_continuity(self):
        stefan = StefanProblem()
        t = 1.0
        s_t = stefan.interface_position(t)

        T_solid = stefan.temperature_solid(s_t - 1e-6, t)
        T_liquid = stefan.temperature_liquid(s_t + 1e-6, t)

        self.assertAlmostEqual(T_solid, stefan.T_m, places=1)
        self.assertAlmostEqual(T_liquid, stefan.T_m, places=1)

if __name__ == '__main__':
    unittest.main()
