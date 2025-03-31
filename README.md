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
