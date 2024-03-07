from .math_formula import *
from .initialize_domain import *
from numpy import sqrt, max, mean

__all__ = ["Smagorinsky_model"]


def Smagorinsky_model(domain: InitializeDomain, dU):
    """Smagorinsky Model

    Args:
        domain (InitializeDomain): class object which holds all the variables
        dU (4-d numpy array): RHS term

    Returns:
        dU
    """
    velocity_gradient(domain=domain)
    calculate_nuR(domain)
    domain.temp_I[:] = residue_stress(domain=domain)
    return domain.temp_I


def calculate_nuR(domain: InitializeDomain):
    """This will calculate the Eddy viscosity for Smagorinsky Model

    Args:
        domain (InitializeDomain): class object which holds all the variables
    """
    if domain.dim == 3:
        domain.nuR[:] = (
            domain.a11 * domain.a11
            + (0.5 * (domain.a12 + domain.a21) ** 2)
            + domain.a22 * domain.a22
            + (0.5 * (domain.a13 + domain.a31) ** 2)
            + (0.5 * (domain.a23 + domain.a32) ** 2)
            + domain.a33 * domain.a33
        )
        domain.nuR[:] = sqrt(2.0 * domain.nuR) * (domain.Cs * domain.dx) ** 2
    else:
        domain.nuR[:] = (
            (domain.a11 * domain.a11)
            + (0.5 * (domain.a12 + domain.a21) ** 2)
            + (domain.a22 * domain.a22)
        )
        domain.nuR[:] = sqrt(2.0 * domain.nuR) * (domain.Cs * domain.dx) ** 2.0
    return


def Smagorinsky_model2(domain: InitializeDomain, dU, a=False):
    """Smagorinsky Model

    Args:
        domain (InitializeDomain): class object which holds all the variables
        dU (4-d numpy array): RHS term

    Returns:
        dU
    """
    if a:
        domain.S11[:] = domain.fft.backward(
            1j * domain.K[0] * domain.U_hat[0], domain.S11, normalize=True
        )
        domain.S22[:] = domain.fft.backward(
            1j * domain.K[1] * domain.U_hat[1], domain.S22, normalize=True
        )
        domain.S12[:] = domain.fft.backward(
            0.5 * 1j * (domain.K[0] * domain.U_hat[1] + domain.K[1] * domain.U_hat[0]),
            domain.S12,
            normalize=True,
        )
        if domain.dim == 3:
            domain.S13[:] = domain.fft.backward(
                0.5
                * 1j
                * (domain.K[0] * domain.U_hat[2] + domain.K[2] * domain.U_hat[0]),
                domain.S13,
                normalize=True,
            )
            domain.S23[:] = domain.fft.backward(
                0.5
                * 1j
                * (domain.K[1] * domain.U_hat[2] + domain.K[2] * domain.U_hat[1]),
                domain.S23,
                normalize=True,
            )
            domain.S33[:] = domain.fft.backward(
                1j * domain.K[2] * domain.U_hat[2],
                domain.S33,
                normalize=True,
            )
            domain.nuR[:] = (
                domain.S11 * domain.S11
                + 2.0 * domain.S12 * domain.S12
                + domain.S22 * domain.S22
                + 2.0 * domain.S13 * domain.S13
                + 2.0 * domain.S23 * domain.S23
                + domain.S33 * domain.S33
            )
            domain.nuR[:] = sqrt(2.0 * domain.nuR) * (domain.Cs * domain.dx) ** 2
            domain.temp2_I[0] = (
                domain.K[0] * (domain.fft.forward(domain.S11 * domain.nuR))
                + domain.K[1] * domain.fft.forward(domain.S12 * domain.nuR)
                + domain.K[2] * domain.fft.forward(domain.S13 * domain.nuR)
            ) * (2.0 * 1j * domain.dealias)
            domain.temp2_I[1] = (
                domain.K[0] * domain.fft.forward(domain.S12 * domain.nuR)
                + domain.K[1] * domain.fft.forward(domain.S22 * domain.nuR)
                + domain.K[2] * domain.fft.forward(domain.S23 * domain.nuR)
            ) * (2.0 * 1j * domain.dealias)
            domain.temp2_I[2] = (
                domain.K[0] * domain.fft.forward(domain.S12 * domain.nuR)
                + domain.K[1] * domain.fft.forward(domain.S22 * domain.nuR)
                + domain.K[2] * domain.fft.forward(domain.S23 * domain.nuR)
            ) * (2.0 * 1j * domain.dealias)
            return
    else:
        dU[:] -= domain.temp2_I[:]
    # else:
    #     domain.nuR[:] = domain.S11**2.0 + 2.0 * domain.S12**2.0 + domain.S22**2.0
    #     domain.nuR[:] = sqrt(2.0 * domain.nuR) * (domain.Cs * domain.dx) ** 2.0
    #     dU[0] += (
    #         domain.K[0] * (domain.fft.forward(domain.S11 * domain.nuR))
    #         + domain.K[1] * domain.fft.forward(domain.S12 * domain.nuR)
    #     ) * (2.0 * 1j * domain.dealias)
    #     dU[1] += (
    #         domain.K[0] * domain.fft.forward(domain.S12 * domain.nuR)
    #         + domain.K[1] * domain.fft.forward(domain.S22 * domain.nuR)
    #     ) * (2.0 * 1j * domain.dealias)

    return dU


def Smagorinsky_model1(domain: InitializeDomain, dU):
    """Smagorinsky Model

    Args:
        domain (InitializeDomain): class object which holds all the variables
        dU (4-d numpy array): RHS term

    Returns:
        dU
    """
    domain.S11[:] = dudx(U=domain.U[0], input=domain.S11, dx=domain.dx)
    domain.S22[:] = dvdy(U=domain.U[1], input=domain.S22, dx=domain.dx)
    domain.S12[:] = dvdxdudy(
        U=domain.U[0], V=domain.U[1], input=domain.S12, dx=domain.dx
    )
    # domain.nuR[:] = domain.S11**2.0 + 2.0 * domain.S12**2.0 + domain.S22**2.0
    if domain.dim == 3:
        domain.S13[:] = dwdxdudz(
            U=domain.U[0], W=domain.U[2], input=domain.S13, dx=domain.dx
        )
        domain.S23[:] = dwdydvdz(
            V=domain.U[1], W=domain.U[2], input=domain.S23, dx=domain.dx
        )
        domain.S33[:] = dwdz(U=domain.U[2], input=domain.S33, dx=domain.dx)
        domain.nuR[:] = (
            domain.S11**2.0
            + 2.0 * domain.S12**2.0
            + domain.S22**2.0
            + 2.0 * domain.S13**2.0
            + 2.0 * domain.S23**2.0
            + domain.S33**2.0
        )
        domain.nuR[:] = sqrt(2.0 * domain.nuR) * (domain.Cs * domain.dx) ** 2.0
        dU[0] += (
            domain.K[0] * (domain.fft.forward(domain.S11 * domain.nuR))
            + domain.K[1] * domain.fft.forward(domain.S12 * domain.nuR)
            + domain.K[2] * domain.fft.forward(domain.S13 * domain.nuR)
        ) * (2.0 * 1j * domain.dealias)
        dU[1] += (
            domain.K[0] * domain.fft.forward(domain.S12 * domain.nuR)
            + domain.K[1] * domain.fft.forward(domain.S22 * domain.nuR)
            + domain.K[2] * domain.fft.forward(domain.S23 * domain.nuR)
        ) * (2.0 * 1j * domain.dealias)
        dU[2] += (
            domain.K[0] * domain.fft.forward(domain.S13 * domain.nuR)
            + domain.K[1] * domain.fft.forward(domain.S23 * domain.nuR)
            + domain.K[2] * domain.fft.forward(domain.S33 * domain.nuR)
        ) * (2.0 * 1j * domain.dealias)
    else:
        domain.nuR[:] = domain.S11**2.0 + 2.0 * domain.S12**2.0 + domain.S22**2.0
        domain.nuR[:] = sqrt(2.0 * domain.nuR) * (domain.Cs * domain.dx) ** 2.0
        dU[0] += (
            domain.K[0] * (domain.fft.forward(domain.S11 * domain.nuR))
            + domain.K[1] * domain.fft.forward(domain.S12 * domain.nuR)
        ) * (2.0 * 1j * domain.dealias)
        dU[1] += (
            domain.K[0] * domain.fft.forward(domain.S12 * domain.nuR)
            + domain.K[1] * domain.fft.forward(domain.S22 * domain.nuR)
        ) * (2.0 * 1j * domain.dealias)

    return dU


def dudx(U, input, dx):
    input[1:-1, :, :] = (U[2:, :, :] - U[:-2, :, :]) * 0.5 / dx
    input[0, :, :] = (U[1, :, :] - U[-1, :, :]) * 0.5 / dx
    input[-1, :, :] = (U[0, :, :] - U[-2, :, :]) * 0.5 / dx
    return input


def dvdy(U, input, dx):
    input[:, 1:-1, :] = (U[:, 2:, :] - U[:, :-2, :]) * 0.5 / dx
    input[:, 0, :] = (U[:, 1, :] - U[:, -1, :]) * 0.5 / dx
    input[:, -1, :] = (U[:, 0, :] - U[:, -2, :]) * 0.5 / dx
    return input


def dwdz(U, input, dx):
    input[:, :, 1:-1] = (U[:, :, 2:] - U[:, :, :-2]) * 0.5 / dx
    input[:, :, 0] = (U[:, :, 1] - U[:, :, -1]) * 0.5 / dx
    input[:, :, -1] = (U[:, :, 0] - U[:, :, -2]) * 0.5 / dx
    return input


def dvdxdudy(U, V, input, dx):
    input[1:-1, 1:-1, :] = (
        U[1:-1, 2:, :] - U[1:-1, 0:-2, :] + V[2:, 1:-1, :] - V[0:-2, 1:-1, :]
    ) * (0.5 / dx)
    input[0, 0, :] = (U[0, 1, :] - U[0, -1, :] + V[1, 0, :] - V[-1, 0, :]) * (0.5 / dx)
    input[0, -1, :] = (U[0, 0, :] - U[0, -2, :] + V[1, -1, :] - V[-1, -1, :]) * (
        0.5 / dx
    )
    input[-1, 0, :] = (U[-1, 1, :] - U[-1, -1, :] + V[0, 0, :] - V[-2, 0, :]) * (
        0.5 / dx
    )
    input[-1, -1, :] = (U[-1, 0, :] - U[-1, -2, :] + V[0, -1, :] - V[-2, -1, :]) * (
        0.5 / dx
    )
    input[0, 1:-1, :] = (
        U[0, 2:, :] - U[0, 0:-2, :] + V[1, 1:-1, :] - V[-1, 1:-1, :]
    ) * (0.5 / dx)
    input[-1, 1:-1, :] = (
        U[-1, 2:, :] - U[-1, 0:-2, :] + V[0, 1:-1, :] - V[-2, 1:-1, :]
    ) * (0.5 / dx)
    input[1:-1, 0, :] = (
        U[1:-1, 1, :] - U[1:-1, -1, :] + V[2:, 0, :] - V[:-2, 0, :]
    ) * (0.5 / dx)
    input[1:-1, -1, :] = (
        U[1:-1, 0, :] - U[1:-1, -2, :] + V[2:, -1, :] - V[:-2, -1, :]
    ) * (0.5 / dx)
    return input


def dwdxdudz(U, W, input, dx):
    input[1:-1, :, 1:-1] = (
        U[1:-1, :, 2:] - U[1:-1, :, 0:-2] + W[2:, :, 1:-1] - W[0:-2, :, 1:-1]
    ) * (0.5 / dx)
    input[0, :, 0] = (U[0, :, 1] - U[0, :, -1] + W[1, :, 0] - W[-1, :, 0]) * (0.5 / dx)
    input[0, :, -1] = (U[0, :, 0] - U[0, :, -2] + W[1, :, -1] - W[-1, :, -1]) * (
        0.5 / dx
    )
    input[-1, :, 0] = (U[-1, :, 1] - U[-1, :, -1] + W[0, :, 0] - W[-2, :, 0]) * (
        0.5 / dx
    )
    input[-1, :, -1] = (U[-1, :, 0] - U[-1, :, -2] + W[0, :, -1] - W[-2, :, -1]) * (
        0.5 / dx
    )
    input[0, :, 1:-1] = (
        U[0, :, 2:] - U[0, :, 0:-2] + W[1, :, 1:-1] - W[-1, :, 1:-1]
    ) * (0.5 / dx)
    input[-1, :, 1:-1] = (
        U[-1, :, 2:] - U[-1, :, 0:-2] + W[0, :, 1:-1] - W[-2, :, 1:-1]
    ) * (0.5 / dx)
    input[1:-1, :, 0] = (
        U[1:-1, :, 1] - U[1:-1, :, -1] + W[2:, :, 0] - W[:-2, :, 0]
    ) * (0.5 / dx)
    input[1:-1, :, -1] = (
        U[1:-1, :, 0] - U[1:-1, :, -2] + W[2:, :, -1] - W[:-2, :, -1]
    ) * (0.5 / dx)
    return input


def dwdydvdz(V, W, input, dx):
    input[:, 1:-1, 1:-1] = (
        V[:, 1:-1, 2:] - V[:, 1:-1, 0:-2] + W[:, 2:, 1:-1] - W[:, 0:-2, 1:-1]
    ) * (0.5 / dx)
    input[:, 0, 0] = (V[:, 0, 1] - V[:, 0, -1] + W[:, 1, 0] - W[:, -1, 0]) * (0.5 / dx)
    input[:, 0, -1] = (V[:, 0, 0] - V[:, 0, -2] + W[:, 1, -1] - W[:, -1, -1]) * (
        0.5 / dx
    )
    input[:, -1, 0] = (V[:, -1, 1] - V[:, -1, -1] + W[:, 0, 0] - W[:, -2, 0]) * (
        0.5 / dx
    )
    input[:, -1, -1] = (V[:, -1, 0] - V[:, -1, -2] + W[:, 0, -1] - W[:, -2, -1]) * (
        0.5 / dx
    )
    input[:, 0, 1:-1] = (
        V[:, 0, 2:] - V[:, 0, 0:-2] + W[:, 1, 1:-1] - W[:, -1, 1:-1]
    ) * (0.5 / dx)
    input[:, -1, 1:-1] = (
        V[:, -1, 2:] - V[:, -1, 0:-2] + W[:, 0, 1:-1] - W[:, -2, 1:-1]
    ) * (0.5 / dx)
    input[:, 1:-1, 0] = (
        V[:, 1:-1, 1] - V[:, 1:-1, -1] + W[:, 2:, 0] - W[:, :-2, 0]
    ) * (0.5 / dx)
    input[:, 1:-1, -1] = (
        V[:, 1:-1, 0] - V[:, 1:-1, -2] + W[:, 2:, -1] - W[:, :-2, -1]
    ) * (0.5 / dx)
    return input
