# spectral-LES

# Pseudo-Spectral Navier-Stokes Solver

This repository implements a parallelized pseudo-spectral solver for simulating incompressible fluid dynamics using various time-integration schemes (RK3, RK4, ETD2). The solver supports 2D and 3D domains with optional Large Eddy Simulation (LES) models.

## Features

- Supports RK3, RK4, and ETD2 timestepping methods
- Modular LES modeling (e.g., Smagorinsky)
- MPI parallelism using `mpi4py` and `mpi4py-fft`
- Energy spectrum plotting and data saving in HDF5
- 2D and 3D compatible

---

## File Overview

| File | Description |
|------|-------------|
| `initialize_domain.py` | Initializes the simulation domain, FFT configuration, and parameters. |
| `initial.py` | Sets initial conditions for velocity or vorticity. |
| `math_formula.py` | Contains core math functions like velocity gradients and vorticity computations. |
| `ETD2_method.py` | Implements ETD2 time integration for the Navier-Stokes solver. |
| `RK4_method.py` | Implements the classical Runge-Kutta 4th-order scheme. |
| `RK3_method.py` | Implements a 3rd-order Runge-Kutta method. |
| `smagorinsky.py` | Contains the Smagorinsky LES model and helper functions for turbulence modeling. |
| `plotting.py` | Provides a utility to generate stylized matplotlib plots. |
| `utilites.py` | Utility functions for managing files, printing status, and data extraction. |
| `SM_solver.py` | Main solver driver that orchestrates time integration, saving output, and energy spectrum plotting. |

---

## Getting Started

### Prerequisites

- Python 3.7+
- MPI
- `numpy`, `matplotlib`, `mpi4py`, `mpi4py-fft`, `pandas`

### Running the Simulation

```bash
mpirun -np <num_procs> python run_simulation.py

---

## References

This solver is adapted from the spectral formulation and implementation presented in:

1. Mortensen, M., & Langtangen, H. P. (2016). High performance Python for direct numerical simulations of turbulent flows. *Computer Physics Communications*, 203, 53–65. https://doi.org/10.1016/j.cpc.2016.02.005

Additional references and background:

2. Pope, S. B. (2000). *Turbulent Flows*. Cambridge University Press.  
3. Canuto, C., Hussaini, M. Y., Quarteroni, A., & Zang, T. A. (2006). *Spectral Methods: Fundamentals in Single Domains*. Springer.  
4. Boyd, J. P. (2001). *Chebyshev and Fourier Spectral Methods*. Dover Publications.  
5. Smagorinsky, J. (1963). General circulation experiments with the primitive equations: I. The basic experiment. *Monthly Weather Review*, 91(3), 99–164.  
6. Chai, J., & Lee, Y. (2020). Efficient parallel pseudo-spectral method using MPI. *Journal of Computational Physics*, 419, 109676.  
7. [mpi4py-fft GitHub Repository](https://github.com/mpi4py/mpi4py-fft): A Python package for FFT using MPI.

---

For algorithmic formulation, numerical stability, and energy spectrum interpretation, please refer to the cited works.

