from .math_formula import del_cross_U
from numpy import where, array, sum, mean, sqrt
from mpi4py import MPI
from .initialize_domain import *


def initial_velocity(domain: InitializeDomain, u=0, v=0, w=0):
    domain.U[0][:] = u
    domain.U[1][:] = v
    if domain.dim == 3:
        domain.U[2][:] = w
    for i in range(domain.dim):
        domain.U_hat[i] = domain.fft.forward(
            domain.U[i], domain.U_hat[i], normalize=False
        )
    domain.curl[:] = del_cross_U(
        domain.fft, domain.dim, domain.U_hat, domain.curl, domain.K
    )

# def initial_velocity(domain: InitializeDomain, u=0, v=0, w=0):
#     domain.U[0][:] = (
#         filter(u, domain.U[0], domain.NN, domain.dim) if domain.LES_ON else u
#     )
#     domain.U[1][:] = (
#         filter(v, domain.U[1], domain.NN, domain.dim) if domain.LES_ON else v
#     )
#     if domain.dim == 3:
#         domain.U[2][:] = (
#         filter(w, domain.U[2], domain.NN, domain.dim) if domain.LES_ON else w
#     )
#     for i in range(domain.dim):
#         domain.U_hat[i] = domain.fft.forward(
#             domain.U[i], domain.U_hat[i], normalize=False
#         )
#     domain.curl[:] = del_cross_U(
#         domain.fft, domain.dim, domain.U_hat, domain.curl, domain.K
#     )

def initial_vorticity(domain: InitializeDomain, omega):
    phi_hat = domain.fft.forward(omega, normalize=False) / where(
        domain.K2 == 0, 1, domain.K2
    ).astype(float)
    phi_hat[0, 0] = 0
    domain.U_hat[0][:] = 1j * domain.K[1] * phi_hat
    domain.U_hat[1][:] = -1j * domain.K[0] * phi_hat
    if domain.dim == 3:
        domain.U_hat[2] = 0
        domain.curl[2] = omega
    else:
        domain.curl[:] = omega
    for i in range(domain.dim):
        domain.U[i][:] = domain.fft.backward(
            domain.U_hat[i], domain.U[i][:], normalize=True
        )


def filter(Input, Output, NN, dim):
    N = Input.shape[0]
    for i in range(0, Output.shape[0], NN):
        for j in range(0, Output.shape[1], NN):
            if dim == 3:
                for k in range(0, Output.shape[2], NN):
                    Output[i, j, k] = sqrt(mean(
                        Input[
                            NN * i : NN * (i + 1),
                            NN * j : NN * (j + 1),
                            NN * k : NN * (k + 1),
                        ]**2
                    ))
            else:
                Output[i, j] = mean(Input[NN * i : NN * (i + 1), NN * j : NN * (j + 1)])
    return Output
