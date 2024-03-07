# __all__ = ["U_cross_omega", "del_cross_U"]
from .initialize_domain import *


def U_cross_omega(fft, dim, a, b, c):
    if dim == 2:
        c[0] = fft.forward(a[1][:] * b[:], c[0], normalize=False)
        c[1] = fft.forward(-a[0][:] * b[:], c[1], normalize=False)
    else:
        c[0] = fft.forward(a[1] * b[2] - a[2] * b[1], c[0], normalize=False)
        c[1] = fft.forward(a[2] * b[0] - a[0] * b[2], c[1], normalize=False)
        c[2] = fft.forward(a[0] * b[1] - a[1] * b[0], c[2], normalize=False)
    return c


def del_cross_U(fft, dim, a, c, K):
    if dim == 2:
        c[:] = fft.backward(
            1j * (K[0] * a[1][:] - K[1] * a[0][:]), c[:], normalize=True
        )
    else:
        c[2] = fft.backward(1j * (K[0] * a[1] - K[1] * a[0]), c[2], normalize=True)
        c[1] = fft.backward(1j * (K[2] * a[0] - K[0] * a[2]), c[1], normalize=True)
        c[0] = fft.backward(1j * (K[1] * a[2] - K[2] * a[1]), c[0], normalize=True)
    return c

def velocity_gradient(domain: InitializeDomain):
    """This will calculate the gradient of velocity

    Args:
        domain (InitializeDomain): class object which holds all the variables
    """
    domain.a11[:] = domain.fft.backward(
        1j * domain.K[0] * domain.U_hat[0], domain.a11, normalize=True
    )
    domain.a12[:] = domain.fft.backward(
        1j * domain.K[0] * domain.U_hat[1], domain.a12, normalize=True
    )

    domain.a21[:] = domain.fft.backward(
        1j * domain.K[1] * domain.U_hat[0], domain.a21, normalize=True
    )
    domain.a22[:] = domain.fft.backward(
        1j * domain.K[1] * domain.U_hat[1], domain.a22, normalize=True
    )
    if domain.dim == 3:
        domain.a13[:] = domain.fft.backward(
            1j * domain.K[0] * domain.U_hat[2], domain.a13, normalize=True
        )
        domain.a23[:] = domain.fft.backward(
            1j * domain.K[1] * domain.U_hat[2], domain.a23, normalize=True
        )
        domain.a31[:] = domain.fft.backward(
            1j * domain.K[2] * domain.U_hat[0], domain.a31, normalize=True
        )
        domain.a32[:] = domain.fft.backward(
            1j * domain.K[2] * domain.U_hat[1], domain.a32, normalize=True
        )
        domain.a33[:] = domain.fft.backward(
            1j * domain.K[2] * domain.U_hat[2], domain.a33, normalize=True
        )
    return

def residue_stress(domain: InitializeDomain):
    """This will calculate the residual stress part of RHS of NS equation in fourier space

    Args:
        domain (InitializeDomain): class object which holds all the variables

    Returns:
        the residual stress part of RHS of NS equation in fourier space
    """
    if domain.dim == 3:
        domain.temp_I[0] = (
            (
                domain.K[0]
                * (domain.fft.forward(domain.a11 * domain.nuR, normalize=False))
            )
            + (
                domain.K[1]
                * domain.fft.forward(
                    0.5 * (domain.a12 + domain.a21) * domain.nuR, normalize=False
                )
            )
            + (
                domain.K[2]
                * domain.fft.forward(
                    0.5 * (domain.a13 + domain.a31) * domain.nuR, normalize=False
                )
            )
        ) * (2.0 * 1j * domain.dealias)
        domain.temp_I[1] = (
            (
                domain.K[0]
                * domain.fft.forward(
                    0.5 * (domain.a12 + domain.a21) * domain.nuR, normalize=False
                )
            )
            + (
                domain.K[1]
                * domain.fft.forward(domain.a22 * domain.nuR, normalize=False)
            )
            + (
                domain.K[2]
                * domain.fft.forward(
                    0.5 * (domain.a23 + domain.a32) * domain.nuR, normalize=False
                )
            )
        ) * (2.0 * 1j * domain.dealias)
        domain.temp_I[2] = (
            (
                domain.K[0]
                * domain.fft.forward(
                    0.5 * (domain.a13 + domain.a31) * domain.nuR, normalize=False
                )
            )
            + (
                domain.K[1]
                * domain.fft.forward(
                    0.5 * (domain.a23 + domain.a32) * domain.nuR, normalize=False
                )
            )
            + (
                domain.K[2]
                * domain.fft.forward(domain.a33 * domain.nuR, normalize=False)
            )
        ) * (2.0 * 1j * domain.dealias)
    else:
        domain.temp_I[0] = (
            (
                domain.K[0]
                * (domain.fft.forward(domain.a11 * domain.nuR, normalize=False))
            )
            + (
                domain.K[1]
                * domain.fft.forward(0.5 * (domain.a12 + domain.a21) * domain.nuR, normalize=False)
            )
        ) * (2.0 * 1j * domain.dealias)
        domain.temp_I[0] = (
            (domain.K[0] * domain.fft.forward(0.5 * (domain.a12 + domain.a21) * domain.nuR, normalize=False))
            + (
                domain.K[1]
                * domain.fft.forward(domain.a22 * domain.nuR, normalize=False)
            )
        ) * (2.0 * 1j * domain.dealias)

    return domain.temp_I