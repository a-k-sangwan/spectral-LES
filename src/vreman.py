from .math_formula import *
from .initialize_domain import *
from numpy import sqrt, max, where, min

__all__ = ["Vreman_model"]


def Vreman_model(domain: InitializeDomain, dU):
    """Verman eddy viscosity model

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
    """This will calculate the Eddy viscosity for Model given by Vreman: "http://basilisk.fr/sandbox/Antoonvh/vreman.h"

    Args:
        domain (InitializeDomain): class object which holds all the variables
    """
    if domain.dim == 3:

        domain.nuR[:] = sqrt(
            (
                (
                    (
                        domain.a11 * domain.a11
                        + domain.a21 * domain.a21
                        + domain.a31 * domain.a31
                    )
                    * (
                        domain.a12 * domain.a12
                        + domain.a22 * domain.a22
                        + domain.a32 * domain.a32
                    )
                )
                - (
                    (
                        domain.a11 * domain.a12
                        + domain.a21 * domain.a22
                        + domain.a31 * domain.a32
                    )
                    * (
                        domain.a11 * domain.a12
                        + domain.a21 * domain.a22
                        + domain.a31 * domain.a32
                    )
                )
                + (
                    (
                        domain.a11 * domain.a11
                        + domain.a21 * domain.a21
                        + domain.a31 * domain.a31
                    )
                    * (
                        domain.a13 * domain.a13
                        + domain.a23 * domain.a23
                        + domain.a33 * domain.a33
                    )
                )
                - (
                    (
                        domain.a11 * domain.a13
                        + domain.a21 * domain.a23
                        + domain.a31 * domain.a33
                    )
                    * (
                        domain.a11 * domain.a13
                        + domain.a21 * domain.a23
                        + domain.a31 * domain.a33
                    )
                )
                + (
                    (
                        domain.a12 * domain.a12
                        + domain.a22 * domain.a22
                        + domain.a32 * domain.a32
                    )
                    * (
                        domain.a13 * domain.a13
                        + domain.a23 * domain.a23
                        + domain.a33 * domain.a33
                    )
                )
                - (
                    (
                        domain.a12 * domain.a13
                        + domain.a22 * domain.a23
                        + domain.a32 * domain.a33
                    )
                    * (
                        domain.a12 * domain.a13
                        + domain.a22 * domain.a23
                        + domain.a32 * domain.a33
                    )
                )
            )
        )
        domain.aTemp[:] = (
            domain.a11 * domain.a11
            + domain.a12 * domain.a12
            + domain.a13 * domain.a13
            + domain.a21 * domain.a21
            + domain.a22 * domain.a22
            + domain.a23 * domain.a23
            + domain.a31 * domain.a31
            + domain.a32 * domain.a32
            + domain.a33 * domain.a33
        )
        domain.nuR[:] *= where(
            (domain.aTemp > 10e-5) * (domain.nuR > (domain.aTemp / 10e6)),
            sqrt(1 / domain.aTemp),
            0.0,
        ) * (2.5 * domain.Cs * domain.Cs * domain.dx * domain.dx)
    else:
        domain.nuR[:] = sqrt(
            (
                (
                    (domain.a11 * domain.a11 + domain.a21 * domain.a21)
                    * (domain.a12 * domain.a12 + domain.a22 * domain.a22)
                )
                - (
                    (domain.a11 * domain.a12 + domain.a21 * domain.a22)
                    * (domain.a11 * domain.a12 + domain.a21 * domain.a22)
                )
            )
        )
        domain.aTemp[:] = (
            domain.a11 * domain.a11
            + domain.a12 * domain.a12
            + domain.a21 * domain.a21
            + domain.a22 * domain.a22
        )
        domain.nuR[:] *= where(
            (domain.aTemp > 10e-5) * (domain.nuR > (domain.aTemp / 10e6)),
            sqrt(1 / domain.aTemp),
            0.0,
        ) * (2.5 * domain.Cs * domain.Cs * domain.dx * domain.dx)
    return
