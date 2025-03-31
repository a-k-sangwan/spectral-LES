# __all__ = ['solveForNextTimestep']
"""
RK3_method.py
-------------
Implements a 3rd-order Runge-Kutta scheme for solving the Navier-Stokes equations.
Supports LES models and handles nonlinear term evaluations.
"""


from .math_formula import *
from .initialize_domain import *
from .smagorinsky import *
from .vreman import *
from numpy import sum, array, sqrt

# , where, absolute, conjugate, pi, mean,  ones_like
from mpi4py import MPI


def solveForNextTimestep(domain: InitializeDomain):
    dt = domain.dt
    U_hat = domain.U_hat
    U0 = domain.U_hat0
    dU = domain.dU

    U0[:] = U_hat[:]
    dU[:] = computeRHS(domain, dU, rk=0)   # k1
    U_hat[:] = U0 + 0.5 * dt * dU
    dU[:] = computeRHS(domain, dU, rk=1)   # k2
    U_hat[:] = U0 - dt * dU + 2.0 * dt * dU  
    dU[:] = computeRHS(domain, dU, rk=2)   # k3
    U_hat[:] = U0 + dt * (1/6 * dU + 2/3 * dU + 1/6 * dU) 

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
