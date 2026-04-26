# Detailed Documentation

## Mathematical Formulation

### Governing Equations

The phase change problem is governed by the heat conduction equation with latent heat:

**Energy Conservation:**
```
ρ ∂H/∂t = ∇·(k∇T)
```

where:
- `H = c_p T + f L` is the enthalpy
- `f` is the liquid fraction (0 = solid, 1 = liquid)
- `L` is the latent heat of fusion
- `k` is thermal conductivity
- `ρ` is density
- `c_p` is specific heat

### Particle Method Discretization

**Kernel Function (Wendland C2):**
```
W(r, h) = (7/4πh²)(1 - q)⁴(4q + 1),  q = r/h
```

**Gradient Operator:**
```
∇φ_i = Σ_j (φ_j - φ_i) ∇W_ij V_j
```

**Laplacian Operator:**
```
∇²φ_i = Σ_j (2d/(r_ij² + ε²))(φ_j - φ_i) W_ij V_j
```

where `d` is the spatial dimension and `ε` is a small regularization parameter.

### Phase Change Model

**Liquid Fraction:**
```
f = {
    0,                           T < T_m - ΔT/2
    (T - T_m + ΔT/2)/ΔT,        T_m - ΔT/2 ≤ T ≤ T_m + ΔT/2
    1,                           T > T_m + ΔT/2
}
```

**Effective Properties:**
```
k(f) = k_s + (k_l - k_s)f
c_p(f) = c_p,s + (c_p,l - c_p,s)f
```

### Stefan Problem

The one-dimensional Stefan problem has an analytical solution:

**Interface Position:**
```
s(t) = 2λ√(α_s t)
```

where λ is determined by:
```
(Ste_s/√α_s)exp(λ²/α_s)erf(λ/√α_s) + (Ste_l/√α_l)exp(λ²/α_l)erf(λ/√α_l) = λ√π
```

**Stefan Numbers:**
```
Ste_s = c_s(T_m - T_c)/L
Ste_l = c_l(T_h - T_m)/L
```

## Numerical Implementation

### Time Integration

Semi-implicit time stepping:
1. Compute enthalpy at time n: `H^n = c_p T^n + f^n L`
2. Solve heat equation: `H^(n+1) = H^n + Δt k/ρ ∇²T^n`
3. Update temperature and liquid fraction from enthalpy

### Stability Condition

CFL condition for diffusion:
```
Δt ≤ CFL × Δx² / α_max
```

Typical CFL = 0.2-0.3 for stability.

### Boundary Conditions

For Stefan problem:
- Left boundary: `T = T_cold` (Dirichlet)
- Right boundary: `T = T_hot` (Dirichlet)

## Algorithm Workflow

```
1. Initialize particle positions and properties
2. Set initial temperature field
3. For each time step:
   a. Compute neighbor lists
   b. Apply boundary conditions
   c. Compute Laplacian of temperature
   d. Update enthalpy using heat equation
   e. Convert enthalpy to temperature and liquid fraction
   f. Update interface tracking (level-set)
   g. Advance time
4. Output results and visualizations
```

## Performance Considerations

### Computational Complexity

- Neighbor search: O(N log N) with spatial hashing
- Operator computation: O(N × n_neighbors)
- Overall per time step: O(N × n_neighbors)

### Memory Usage

For N particles:
- Positions: 2N floats (2D)
- Temperatures: N floats
- Liquid fraction: N floats
- Neighbor lists: ~N × 20-50 integers

### Optimization Techniques

1. **Numba JIT compilation** for kernel functions
2. **Vectorized operations** with NumPy
3. **Adaptive time stepping** based on CFL condition
4. **Multi-resolution particles** for efficiency (future work)

## Validation Results

### Convergence Study

The method achieves approximately first-order accuracy in the current implementation. Second-order accuracy can be achieved with:
- Improved gradient operators
- Higher-order time integration
- Refined interface treatment

### Error Sources

1. **Discretization error**: O(Δx) spatial discretization
2. **Interface smearing**: Finite interface width
3. **Time integration error**: Semi-implicit scheme
4. **Boundary treatment**: Particle deficiency near boundaries

## Future Improvements

1. **Higher-order operators**: Consistent gradient and Laplacian
2. **Adaptive particle refinement**: Multi-resolution approach
3. **Improved interface tracking**: Sharp interface methods
4. **3D extension**: Full three-dimensional simulations
5. **Parallel computing**: GPU acceleration with CUDA
6. **Complex geometries**: Arbitrary domain shapes
7. **Multi-physics coupling**: Fluid flow with phase change

## References

### Particle Methods
- Koshizuka, S., & Oka, Y. (1996). Moving-particle semi-implicit method for fragmentation of incompressible fluid. Nuclear Science and Engineering.
- Liu, M. B., & Liu, G. R. (2010). Smoothed particle hydrodynamics (SPH): an overview and recent developments. Archives of Computational Methods in Engineering.

### Phase Change Modeling
- Voller, V. R., & Prakash, C. (1987). A fixed grid numerical modelling methodology for convection-diffusion mushy region phase-change problems. International Journal of Heat and Mass Transfer.
- Alexiades, V., & Solomon, A. D. (1993). Mathematical modeling of melting and freezing processes. Hemisphere Publishing Corporation.

### Stefan Problem
- Crank, J. (1984). Free and moving boundary problems. Oxford University Press.
- Gupta, S. C. (2017). The classical Stefan problem: basic concepts, modelling and analysis with quasi-analytical solutions and methods. Elsevier.
