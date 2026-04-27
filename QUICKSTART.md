# Quick Start Guide

## Installation

```bash
git clone (https://github.com/ezviz0202-hash/Mps-phase-change-simulation)
cd phase-change-particle-method
pip install -r requirements.txt
```

## Running Simulations

### 1. Stefan Problem Validation

Run the validation study to reproduce convergence results:

```bash
python src/validation.py
```

This will:
- Run simulations at 4 different resolutions (20, 40, 80, 160 particles)
- Compare with analytical Stefan problem solution
- Generate convergence analysis plots
- Save results to `results/` directory

**Output files:**
- `results/stefan_validation.png` - Comparison with analytical solution
- `results/convergence_analysis.png` - Convergence study

**Expected runtime:** ~2-3 minutes

### 2. Stefan Problem Simulation

Run a single Stefan problem simulation with visualization:

```bash
python src/main.py --case stefan --nx 60 --ny 60 --duration 3.0
```

**Parameters:**
- `--case stefan`: Stefan problem case
- `--nx 60`: 60 particles in x direction
- `--ny 60`: 60 particles in y direction
- `--duration 3.0`: Simulate for 3 seconds

**Output files:**
- `results/temperature_field.png` - Final temperature distribution
- `results/interface_evolution.png` - Phase interface visualization
- `results/animation.gif` - Animated phase change process

**Expected runtime:** ~10-20 seconds

### 3. Custom Simulation

Run a custom phase change simulation:

```bash
python src/main.py --case custom --nx 100 --ny 100 --duration 10.0
```

This simulates a circular cold region melting in a hot environment.

**Expected runtime:** ~1-2 minutes

## Understanding the Results

### Temperature Field
Shows the spatial distribution of temperature with:
- Blue colors: Cold regions (solid phase)
- Red colors: Hot regions (liquid phase)
- Transition zone: Phase change interface

### Interface Evolution
Displays:
- **Left panel**: Liquid fraction field (0=solid, 1=liquid)
- **Right panel**: Level-set field with interface markers (red X)

### Stefan Validation
Compares simulation results with analytical solution:
- Blue circles: Particle method results
- Red line: Analytical solution
- Vertical dashed lines: Interface positions

### Convergence Analysis
Shows numerical accuracy:
- **Left panel**: L2 error vs resolution (log-log plot)
- **Right panel**: Convergence order (should approach 2.0)

### Animation
GIF showing time evolution of:
- Temperature field (left)
- Liquid fraction and interface (right)

## Typical Workflow

1. **Start with validation** to verify installation:
   ```bash
   python src/validation.py
   ```

2. **Run a quick test** with coarse resolution:
   ```bash
   python src/main.py --case stefan --nx 30 --ny 30 --duration 1.0
   ```

3. **Increase resolution** for better accuracy:
   ```bash
   python src/main.py --case stefan --nx 100 --ny 100 --duration 5.0
   ```

4. **Analyze results** in `results/` directory

## Parameter Guidelines

### Resolution
- **Coarse (20-40)**: Quick tests, ~seconds
- **Medium (50-80)**: Good balance, ~10-30 seconds
- **Fine (100-200)**: High accuracy, ~1-5 minutes

### Duration
- **Short (1-3s)**: Initial phase change
- **Medium (5-10s)**: Significant melting
- **Long (>10s)**: Complete phase transition

### Physical Properties
Edit in the code:
- `T_melt`: Melting temperature (K)
- `latent_heat`: Latent heat of fusion (J/kg)
- `k_solid`, `k_liquid`: Thermal conductivity (W/m·K)
- `c_p_solid`, `c_p_liquid`: Specific heat (J/kg·K)

## Troubleshooting

### Issue: Simulation is slow
**Solution**: Reduce resolution or duration
```bash
python src/main.py --case stefan --nx 30 --ny 30 --duration 1.0
```

### Issue: Results look wrong
**Solution**: Check boundary conditions and initial conditions in the code

### Issue: Animation not generated
**Solution**: Ensure imageio is installed
```bash
pip install imageio
```

### Issue: Import errors
**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

## Advanced Usage

### Custom Initial Conditions

Edit `src/main.py` to modify initial temperature distribution:

```python
for i in range(ps.n_particles):
    x, y = ps.positions[i]
    if x < 0.025:
        ps.temperatures[i] = 263.15  # Cold
    else:
        ps.temperatures[i] = 283.15  # Hot
```

### Custom Boundary Conditions

Modify `apply_boundary_conditions` in `src/solver.py`:

```python
def apply_boundary_conditions(self, T: np.ndarray, bc_type: str = 'stefan'):
    if bc_type == 'stefan':
        x_min = self.ps.positions[:, 0].min()
        x_max = self.ps.positions[:, 0].max()
        for i in range(self.ps.n_particles):
            x = self.ps.positions[i, 0]
            if x <= x_min + 0.5 * self.ps.particle_spacing:
                T[i] = 263.15  # Left boundary
            elif x >= x_max - 0.5 * self.ps.particle_spacing:
                T[i] = 283.15  # Right boundary
```

### Export Data

Save simulation data for post-processing:

```python
import numpy as np

history = solver.solve(t_end=10.0)
np.savez('simulation_data.npz',
         positions=ps.positions,
         temperatures=ps.temperatures,
         liquid_fraction=ps.liquid_fraction)
```

## Performance Tips

1. **Use appropriate resolution**: Start coarse, refine as needed
2. **Adjust CFL number**: Lower for stability, higher for speed
3. **Reduce save frequency**: Save fewer frames for faster runs
4. **Use Numba**: Already enabled for kernel functions

## Next Steps

1. Read [DOCUMENTATION.md](DOCUMENTATION.md) for mathematical details
2. Check [EXAMPLES.md](EXAMPLES.md) for more usage examples
3. See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
4. Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for overview

## Getting Help

- Check documentation files
- Review example code
- Open GitHub issue for bugs
- Read error messages carefully

---

**Happy Simulating!** 🔥❄️
