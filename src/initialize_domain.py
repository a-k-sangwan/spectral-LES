"""
initialize_domain.py
---------------------
Initializes the simulation domain for pseudo-spectral Navier-Stokes solver.
Sets up FFT grid, MPI parallelization, variables, and LES support if enabled.
Used by all other modules to define the working simulation space.
"""

from numpy.fft import fftfreq  # , ifft, irfft2, rfft2
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
from numpy import array, zeros_like, mgrid, meshgrid, empty
from numpy import arange, pi, sum, where, savetxt, random
from .utilites import get_index

__all__ = ["InitializeDomain"]


class InitializeDomain:
    def __init__(
        self,
        nu: float = 0.01,  # Kinematic Viscosity
        final_time: float = 0.0,  # Final time for simulation
        dimension: int = 3,  # 2D or 3D
        time_stepSize: float = 0.01,  # dt or step size in time
        No_of_gridpoint: int = 128,  # Number of grid points in space
        forcing=0.0,  # Forcing value
        forcing_range=[0, 1],  # Range of K to be forced
        method="ETD2",
        LES_ON=0,
        CFL: float=0.5, Cs: float=0.14,
        LES_model=None
    ):
        self.nu = nu
        self.CFL = CFL
        self.T = final_time
        self.dt = time_stepSize
        self.LES_ON = True if LES_ON == 1 else False
        self.NN = 4 if self.LES_ON else 1
        self.N = int(No_of_gridpoint / self.NN)
        self.N_full = No_of_gridpoint if self.LES_ON else None
        self.dx = 2 * pi / self.N
        self.method = method
        # self.trial = random.randint(0, 100, (2, 2))
        self.firstIteration = True  # For keeping track of first Iteration
        if type(self.N) == list:
            if dimension != len(self.N):
                raise Exception("Grid size and dimension is not same")
        self.forcing = forcing
        self.forcing_range = forcing_range
        self.comm = MPI.COMM_WORLD
        self.num_processes = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.Np = self.N // self.num_processes
        self.dim = dimension
        self.shape = (
            [self.N for i in range(self.dim)] if type(self.N) != list else self.N
        )
        self.fft = PFFT(
            self.comm,
            self.shape,
            axes=[i for i in range(self.dim)],
            dtype=float,
            grid=(-1,),
        )
        self.U = [newDistArray(self.fft, False) for i in range(self.dim)]
        self.U_hat = array([newDistArray(self.fft, True) for i in range(self.dim)])
        self.temp1_Img = zeros_like(self.U_hat)
        self.P = newDistArray(self.fft, False)
        self.P_hat = newDistArray(self.fft, True)
        self.dU = zeros_like(self.U_hat)
        self.U_hat0 = zeros_like(self.U_hat)
        self.U_hat1 = zeros_like(self.U_hat)
        self.F0 = zeros_like(self.U_hat) if self.method == "ETD2" else None
        self.F1 = zeros_like(self.U_hat) if self.method == "ETD2" else None
        self.kx = fftfreq(self.N, 1.0 / self.N)
        self.kz = self.kx[: (self.N // 2 + 1)].copy()
        self.kz[-1] *= -1
        self.kmax_dealias = 2.0 / 3.0 * (self.N // 2 + 1)
        if self.dim == 3:
            self.w = newDistArray(self.fft, False)
            index = get_index(self.N, self.num_processes, self.rank)
            self.X = (
                mgrid[index[0] : index[1], : self.N, : self.N].astype(float)
                * 2
                * pi
                / self.N
            )
            if self.LES_ON:
                index_f = get_index(self.N_full, self.num_processes, self.rank)
                self.X_full = (
                    mgrid[index_f[0] : index_f[1], : self.N_full, : self.N_full].astype(
                        float
                    )
                    * 2
                    * pi
                    / self.N_full
                )
            self.K = array(
                meshgrid(
                    self.kx,
                    self.kx[index[0] : index[1]],
                    self.kz,
                    indexing="ij",
                ),
                dtype=int,
            )
            self.curl = zeros_like(array(self.U), dtype=float)
            self.dealias = array(
                (abs(self.K[0]) < self.kmax_dealias)
                * (abs(self.K[1]) < self.kmax_dealias)
                * (abs(self.K[2]) < self.kmax_dealias),
                dtype=bool,
            )
            self.d = (
                arange(self.shape[0], dtype=float) * 2 * pi / self.shape[0],
                arange(self.shape[1], dtype=float) * 2 * pi / self.shape[1],
                arange(self.shape[2], dtype=float) * 2 * pi / self.shape[2],
            )
        else:
            index = get_index(self.N, self.num_processes, self.rank)
            self.X = (
                mgrid[
                    index[0] : index[1],
                    : self.N,
                ].astype(float)
                * 2
                * pi
                / self.N
            )
            if self.LES_ON:
                index = get_index(self.N_full, self.num_processes, self.rank)
                self.X_full = (
                    mgrid[
                        # self.rank * self.Np : (self.rank + 1) * self.Np, : self.N
                        index[0] : index[1],
                        : self.N_full,
                    ].astype(float)
                    * 2
                    * pi
                    / self.N_full
                )
            index = get_index((self.N // 2 + 1), self.num_processes, self.rank)
            self.K = array(
                meshgrid(
                    self.kx,
                    self.kz[index[0] : index[1]],
                    indexing="ij",
                ),
                dtype=int,
            )
            self.curl = zeros_like(self.P)
            self.dealias = array(
                (abs(self.K[0]) < self.kmax_dealias)
                * (abs(self.K[1]) < self.kmax_dealias),
                dtype=bool,
            )
            self.d = (
                arange(self.shape[0], dtype=float) * 2 * pi / self.shape[0],
                arange(self.shape[1], dtype=float) * 2 * pi / self.shape[1],
            )
        self.K2 = sum(self.K * self.K, 0, dtype=int)
        self.K_over_K2 = self.K.astype(float) / where(self.K2 == 0, 1, self.K2).astype(
            float
        )
        self.a = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
        self.b = [0.5, 0.5, 1.0]
        if self.LES_ON:
            if LES_model == None:
                print("\n Please choose LES_model (Smagorinsky or Vreman)")
                exit()
            self.LES_model = LES_model.upper()
            self.Cs = Cs
            self.nuR = zeros_like(self.U[0], dtype=float)
            self.temp_I = zeros_like(self.U_hat)
            self.a11 = zeros_like(self.U[0], dtype=float)
            self.a12 = self.a11.copy()
            self.a21 = self.a11.copy()
            self.a22 = self.a11.copy()
            if self.dim == 3:
                self.a13 = self.a11.copy()
                self.a23 = self.a11.copy()
                self.a31 = self.a11.copy()
                self.a32 = self.a11.copy()
                self.a33 = self.a11.copy()
                self.aTemp = self.a11.copy()
