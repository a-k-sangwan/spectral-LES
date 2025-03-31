# Pseudo-Spectral Navier-Stokes Solver

This is a high-performance, MPI-parallelized pseudo-spectral solver for the incompressible Navier-Stokes equations in 2D or 3D. It supports multiple time-stepping schemes (RK3, RK4, ETD2), LES turbulence models, and runtime energy spectrum visualization.

> ⚙️ This solver is **adapted from** the implementation described in:  
> **Mortensen, M., & Langtangen, H. P. (2016).**  
> *High performance Python for direct numerical simulations of turbulent flows.*  
> *Computer Physics Communications*, 203, 53–65.  
> [https://doi.org/10.1016/j.cpc.2016.02.005](https://doi.org/10.1016/j.cpc.2016.02.005)

---

## 🧩 Project Structure

| File                  | Description |
|-----------------------|-------------|
| `initialize_domain.py` | Initializes simulation domain, grid, MPI, FFT, and LES support. |
| `initial.py`           | Sets initial conditions for velocity or vorticity fields. |
| `math_formula.py`      | Contains core math routines like curl, velocity gradient, and residual stress. |
| `ETD2_method.py`       | Exponential Time Differencing scheme for time advancement. |
| `RK4_method.py`        | Classical 4th-order Runge-Kutta method. |
| `RK3_method.py`        | 3rd-order Runge-Kutta method for time stepping. |
| `smagorinsky.py`       | Smagorinsky LES turbulence model and stress tensor calculations. |
| `plotting.py`          | Matplotlib-based helper for styling and generating plots. |
| `utilites.py`          | Utilities for file I/O, MPI-based logging, directory setup. |
| `SM_solver.py`         | Main solver routine. Manages timestepping, I/O, and plotting. |

---

## 🚀 Installation

Make sure you have the following dependencies installed:

```bash
pip install numpy matplotlib pandas mpi4py mpi4py-fft h5py
