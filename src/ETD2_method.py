from numpy import exp, where, sum, conjugate, absolute
from .math_formula import *
from . import RK4_method as RK4

__all__ = ['solveForNextTimestep']

def solveForNextTimestep(domain):
    if domain.firstIteration:
        RK4.solveForNextTimestep(domain)  # using RK4 for 1st iteration
        domain.firstIteration = False
        del domain.U_hat1, domain.U_hat0
        # domain.U_hat0 = None
        # domain.U_hat1 = None
        return
    computeFn_for_ETD2(domain)
    domain.U_hat[:] *= exp(-domain.dt * domain.nu * domain.K2)
    domain.U_hat[:] += (
        domain.F1
        * (
            (1 - domain.dt * domain.nu * domain.K2)
            + 2 * domain.dt * domain.nu * domain.K2
        )
        + domain.F0
        * (
            -exp(-domain.dt * domain.nu * domain.K2)
            + 1
            - domain.dt * domain.nu * domain.K2
        )
    ) / where(domain.K2 == 0, 1, (domain.dt * (domain.nu * domain.K2) ** 2))
    for i in range(domain.dim):
        domain.U[i][:] = domain.fft.backward(
            domain.U_hat[i], domain.U[i][:], normalize=True
        )
    domain.curl[:] = del_cross_U(
        domain.fft, domain.dim, domain.U_hat, domain.curl, domain.K
    )
    return None

def computeFn_for_ETD2(domain):
    domain.F0[:] = domain.F1[:]
    domain.F1[:] = U_cross_omega(
        domain.fft, domain.dim, domain.U, domain.curl, domain.F1
    )
    domain.F1[:] *= domain.dealias
    domain.P_hat[:] = sum(domain.F1 * domain.K_over_K2, 0, out=domain.P_hat)
    domain.F1[:] -= domain.P_hat * domain.K
    if domain.forcing > 0:
        temp = absolute(domain.U_hat) < 1e-5
        domain.F_hat = (
            domain.epsilon
            * where(temp, 0, 1)
            / where(temp, 1, conjugate(domain.U_hat))
        )
        domain.F1[:] += (
            domain.F_hat - sum(domain.F_hat * domain.K_over_K2, 0) * domain.K
        )
    return 
