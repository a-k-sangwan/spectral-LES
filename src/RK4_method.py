# __all__ = ['solveForNextTimestep']
# ----------------------------- RK4_method.py -----------------------------
"""
RK4_method.py
-------------
Contains a classical 4th-order Runge-Kutta method for time integration.
Includes calculation of the RHS of Navier-Stokes and LES model integration.
"""

from .math_formula import *
from .initialize_domain import *
from .smagorinsky import *
from .vreman import *
from numpy import sum, array, sqrt

# , where, absolute, conjugate, pi, mean,  ones_like
from mpi4py import MPI


def solveForNextTimestep(domain: InitializeDomain):
    domain.U_hat0[:] = domain.U_hat[:]
    for rk in range(4):
        domain.dU[:] = computeRHS(domain, domain.dU, rk)
        if rk < 3:
            domain.U_hat[:] = domain.U_hat0 + domain.b[rk] * domain.dt * domain.dU
        domain.U_hat1[:] += domain.a[rk] * domain.dt * domain.dU
    domain.U_hat[:] = domain.U_hat1[:]
    domain.curl[:] = del_cross_U(
        domain.fft, domain.dim, domain.U_hat, domain.curl, domain.K
    )
    for i in range(domain.dim):
        domain.U[i][:] = domain.fft.backward(
            domain.U_hat[i], domain.U[i][:], normalize=True
        )
    return


def computeRHS(domain: InitializeDomain, dU, rk):
    if rk > 0:
        for i in range(domain.dim):
            domain.U[i][:] = domain.fft.backward(
                domain.U_hat[i], domain.U[i][:], normalize=True
            )
        domain.curl[:] = del_cross_U(
            domain.fft, domain.dim, domain.U_hat, domain.curl, domain.K
        )
    dU[:] = U_cross_omega(domain.fft, domain.dim, domain.U, domain.curl, dU)
    dU *= domain.dealias
    if bool(domain.LES_ON):
        model = Smagorinsky_model if domain.LES_model[0] == "S" else Vreman_model
        domain.temp_I[:] = model(domain, dU)
        domain.P_hat[:] = sum(
            (dU + domain.temp_I) * domain.K_over_K2, 0, out=domain.P_hat
        )
    else:
        domain.P_hat[:] = sum(dU * domain.K_over_K2, 0, out=domain.P_hat)
    dU -= domain.P_hat * domain.K
    if domain.forcing > 0.0:
        domain.temp1_Img[:] = domain.forcing * domain.U_hat[:]
        if domain.dim == 3:
            domain.temp1_Img[:, 0, 0, 0] = 0.0
        else:
            domain.temp1_Img[:, 0, 0] = 0.0
        dU += domain.temp1_Img
    if domain.method == "ETD2":
        domain.F1[:] = dU[:]
    if domain.nu > 0.0:
        dU -= (domain.nu) * domain.K2 * (domain.U_hat)
    if bool(domain.LES_ON):
        dU[:] += domain.temp_I[:]
    return dU


##################### Extra Functions ########################


def U_square(domain: InitializeDomain):
    u = (
        sum(array([sum(domain.U[i] ** 2) for i in range(domain.dim)]))
        / domain.N**domain.dim
    )
    U = array([0.0], dtype=float)
    domain.comm.Allreduce([u, MPI.DOUBLE], [U, MPI.DOUBLE], op=MPI.SUM)
    return U[0]


def dissipation_rate(domain: InitializeDomain):
    """This will calulate the energy dissipation due to viscous effects

    Args:
        domain: simulation domain
    """
    eps = 0.0
    for i in range(domain.dim):
        eps += domain.nu * sum(
            domain.U[i]
            * domain.fft.backward(domain.U_hat[i] * domain.K2, normalize=True)
        )
    U = array([0.0], dtype=float)
    domain.comm.Allreduce([eps, MPI.DOUBLE], [U, MPI.DOUBLE], op=MPI.SUM)
    return U[0]
