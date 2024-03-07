from .utilites import *
from .plotting import *
from .initialize_domain import *
import time
import os
from pandas import read_csv, DataFrame
from mpi4py import MPI
from mpi4py_fft import HDF5File
from . import RK4_method as RK4
from . import ETD2_method as ETD2
from pandas import DataFrame, read_csv
from matplotlib.pyplot import close
from numpy import array, empty, meshgrid, sum, unique, max, random
from numpy import sqrt, zeros, arange, pi, absolute, any, mean


def spectralMethodSolver(
    domain: InitializeDomain, file_out_time: float = 0.0, No_Ek_plot: int = 5
):
    u_max = find_max_velocity(domain.U, domain.comm)
    domain.dt = compute_dt(
        domain.U, u_max, domain.nu, domain.comm, dx=domain.dx, CFL=0.3
    )
    t = domain.dt
    file_time = 0.0
    Ek_plot_interval = domain.T / No_Ek_plot if No_Ek_plot != 0 else 0
    Ek_plot_counter = 0
    if domain.rank == 0:
        print_initial_decorator(domain)
        start = time.time()
        make_directory(name="dumpfile", delete_old=True)
    domain.U_hat1[:] = domain.U_hat[:]
    if Ek_plot_interval > 0:
        Ek_var = Ek_plot_variables(domain)

    ############## Computation starts from Here ##############
    counter1 = 0
    while True:
        if file_out_time > 0 and t >= file_time:
            output_h5(domain, t, file_out_time)
            file_time += file_out_time
        if Ek_plot_interval > 0 and t >= Ek_plot_counter:
            Ek_plot(domain, Ek_var, t, Ek_plot_counter)
            Ek_plot_counter += Ek_plot_interval
        if (t >= domain.T) or (domain.dt < 1e-8):
            break
        if domain.rank == 0:
            print_time(domain.T, domain.dt, t, start, u_max)
            # print(f'{t:<20g} ',end="")
        t += domain.dt
        if domain.method == "RK4":
            RK4.solveForNextTimestep(domain)
        elif domain.method == "ETD2":
            ETD2.solveForNextTimestep(domain)
        u_max = find_max_velocity(domain.U, domain.comm)
        domain.dt = compute_dt(
            domain.U, u_max, domain.nu, domain.comm, dx=domain.dx, CFL=domain.CFL
        )
        if domain.dt != domain.dt:
            if domain.rank == 0:
                print("\n############ dt = Nan #############\n", counter1)
            break
        counter1 += 1
    if Ek_plot_interval > 0 and domain.rank == 0:
        save_Ek_plot(name="Ek_plot.png", Ek_var=Ek_var)
    if domain.rank == 0:
        end = time.time()
        total_time = end - start
        print(
            f"=>  run time: {(total_time) // 3600} H {((total_time)% 3600) // 60} Min {(((total_time)% 3600) % 60):g} sec\n"
        )
        print(f"=>  run time: {(end-start)} sec")
        print(
            f"Time per gridpoint per timestep: {total_time*domain.num_processes/(domain.N**domain.dim * counter1)}\n"
        )
        print(
            "\n************************** SIMULATION COMPLETE **************************\n"
        )


###############################################################################
################################## Functions ##################################
###############################################################################


class Ek_plot_variables:
    """This class is made to hold the variables required to plot the Total Energy Ek in fourier space"""

    def __init__(self, domain: InitializeDomain):
        self.Ek = empty(array(domain.P_hat).shape, dtype=float)
        domain.k2 = unique(domain.K2)
        if domain.dim == 2:
            K2 = array(meshgrid(domain.kx, domain.kz, indexing="ij"), dtype=int)
        else:
            K2 = array(
                meshgrid(domain.kx, domain.kx.copy(), domain.kz, indexing="ij"),
                dtype=int,
            )
        K2 = sum(K2 * K2, 0, dtype=int)
        k2 = unique(K2)
        self.avgPara = 5
        self.smoothing_para = round(sqrt(max(k2))) // self.avgPara
        self.temp_k = zeros((self.smoothing_para + 1), dtype=float)
        self.temp_Ek = zeros((self.smoothing_para + 1), dtype=float)
        if domain.rank == 0:
            self.temp_Ek2 = zeros((self.smoothing_para + 1), dtype=float)
            self.temp_k2 = zeros((self.smoothing_para + 1), dtype=float)
        del K2
        if domain.rank == 0:
            if os.path.exists("Ek_plot_data.csv"):
                os.remove("Ek_plot_data.csv")
            self.fig, self.ax = gen_plot(xlabel=r"$\kappa$", ylabel=r"$E_k$")


def Ek_plot(domain: InitializeDomain, E: Ek_plot_variables, t, Ek_plot_counter):
    """This will add the plot of the Total Energy Ek in fourier space at time t in Ek Plot

    Args:
        domain (object of class: InitializeDomain): This class object holds all the variables
        E (object of class: Ek_plot_variables): This class object holds all the variables required or used for plotting total energy Ek
        t (float): simulation time
        Ek_plot_counter (int): this will count the number of plot
    """
    calculate_Ek(domain=domain, E=E)
    if domain.rank > 0:
        req1 = domain.comm.Isend([E.temp_k, MPI.FLOAT], dest=0, tag=10)
        req1.wait()
        req2 = domain.comm.Isend([E.temp_Ek, MPI.FLOAT], dest=0, tag=20)
        req2.wait()
    else:
        collect_Ek_fromAll(domain, E)
        if os.path.exists("Ek_plot_data.csv"):
            data = read_csv("Ek_plot_data.csv")
            data[f"k-{t:g}"] = E.temp_k
            data[f"Ek-{t:g}"] = E.temp_Ek
            data.to_csv("Ek_plot_data.csv", mode="w", index=False)
        else:
            DataFrame(data={f"k-{t:g}": E.temp_k, f"Ek-{t:g}": E.temp_Ek}).to_csv(
                "Ek_plot_data.csv", mode="w", index=False
            )
        E.ax.loglog(
            (E.temp_k[(E.temp_Ek >= 1e-20) * (E.temp_k >= 0)]),
            (E.temp_Ek[(E.temp_Ek >= 1e-20) * (E.temp_k >= 0)]),
            linewidth=3,
            label=f"t = {round(Ek_plot_counter,2)}",
        )


def calculate_Ek(domain: InitializeDomain, E: Ek_plot_variables):
    """This will calculate the Total Energy Ek in fourier space.

    Args:
        domain (object of class: InitializeDomain): This class object holds all the variables
        E (object of class: Ek_plot_variables): This class object holds all the variables required or used for plotting total energy Ek
    """
    E.temp_k[:] = 0
    E.temp_Ek[:] = 0
    E.Ek[:] = sum(absolute(domain.U_hat[:]) ** 2, 0)
    if domain.rank == 0:
        if domain.dim == 2:
            E.Ek[:, 1:] *= 2
        else:
            E.Ek[:, :, 1:] *= 2
    else:
        E.Ek[:] *= 2
    for i in range(E.smoothing_para + 1):
        if any(
            domain.k2[
                (domain.k2 >= (i * E.avgPara) ** 2)
                * (domain.k2 < ((i + 1) * E.avgPara) ** 2)
            ]
        ):
            E.temp_k[i] = mean(
                sqrt(
                    domain.K2[
                        (domain.K2 >= (i * E.avgPara) ** 2)
                        * (domain.K2 < ((i + 1) * E.avgPara) ** 2)
                    ]
                )
            )
            E.temp_Ek[i] = mean(
                E.Ek[
                    (domain.K2 >= (i * E.avgPara) ** 2)
                    * (domain.K2 < ((i + 1) * E.avgPara) ** 2)
                ]
            )
    return


def collect_Ek_fromAll(domain: InitializeDomain, E: Ek_plot_variables):
    """This will collect the Total Energy Ek in fourier space from all the processor

    Args:
        domain (object of class: InitializeDomain): This class object holds all the variables
        E (object of class: Ek_plot_variables): This class object holds all the variables required or used for plotting total energy Ek
    """
    for i in range(1, domain.num_processes):
        req1 = domain.comm.Irecv([E.temp_k2, MPI.FLOAT], source=i, tag=10)
        req1.wait()
        req2 = domain.comm.Irecv([E.temp_Ek2, MPI.FLOAT], source=i, tag=20)
        req2.wait()
        E.temp_k[(E.temp_k > 0) * (E.temp_k2 > 0)] = 0.5 * (
            E.temp_k[(E.temp_k > 0) * (E.temp_k2 > 0)]
            + E.temp_k2[(E.temp_k > 0) * (E.temp_k2 > 0)]
        )
        E.temp_k[E.temp_k == 0] = E.temp_k2[E.temp_k == 0]
        E.temp_Ek[(E.temp_Ek > 0) * (E.temp_Ek2 > 0)] = 0.5 * (
            E.temp_Ek[(E.temp_Ek > 0) * (E.temp_Ek2 > 0)]
            + E.temp_Ek2[(E.temp_Ek > 0) * (E.temp_Ek2 > 0)]
        )
        E.temp_Ek[E.temp_Ek == 0] = E.temp_Ek2[E.temp_Ek == 0]


def save_Ek_plot(name, Ek_var: Ek_plot_variables):
    """This will save the Final plot of the Total Energy Ek in fourier space

    Args:
        name (string): Name of the plot
        Ek_var (object of class: Ek_plot_variables): This class object holds all the variables required or used for plotting total energy Ek
    """
    ax, fig = Ek_var.ax, Ek_var.fig
    ax.loglog(
        (Ek_var.temp_k[Ek_var.temp_Ek >= 1e-20]),
        3e6 * (Ek_var.temp_k[Ek_var.temp_Ek >= 1e-20]) ** (-3),
        "--",
        label=r"-3",
        linewidth=3,
    )
    pos = ax.get_position()
    ax.set_position([pos.x0 + pos.width * 0.05, pos.y0, pos.width * 0.8, pos.height])
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=20)
    jj = 1
    while os.path.exists(name):
        name = name[:-4] + f"-{jj}" + ".png"
        jj += 1
    fig.savefig(name)
    close(fig)


def find_max_velocity(U, comm):
    """This will return the maximum Velocity in domian.

    Args:
        U (list): list in which 1st element is u (x-direction velocity), 2nd element is v (y-direction velocity) and w (z-direction velocity).
        comm : MPI COMM_WORLD
    """
    u_max = max(U)
    U_max = array([0.0], dtype=float)
    comm.Allreduce([u_max, MPI.DOUBLE], [U_max, MPI.DOUBLE], op=MPI.MAX)
    return U_max[0]


def compute_dt(U, u_max, nu, comm, dx, CFL=0.5):
    """This will calculate the time step (dt):

    Args:
        U (list): list in which 1st element is u (x-direction velocity), 2nd element is v (y-direction velocity) and w (z-direction velocity).
        nu (float): Kinematic viscosity
        comm : MPI COMM_WORLD
        CFL (float): CFL Number, default 0.5
    Returns:
        dt
    """
    # u_max = find_max_velocity(U, comm)
    dt1 = CFL * dx / u_max
    dt2 = 0.1 * dx**2 / nu if (nu > 0) else 1e6
    dt = min(dt1, dt2)
    if dt < 1e-6:
        if comm.rank == 0:
            print(f"\n\n ###  Time step (dt = {dt:g}) is smaller than 1e-6  ###\n\n")
        exit()
    return dt


def output_h5(domain: InitializeDomain, t, file_out_time):
    """This will write the output (velocities) in the files in h5py format

    Args:
        domain (object of class: InitializeDomain): This class object holds all the variables
        t (float): Simulation time
        file_out_time (float): OUTPUT file interval
    """
    if domain.dim == 2:
        field = {
            "u": [domain.U[0]],
            "v": [domain.U[1]],
            "omega": [domain.curl],
        }

    else:
        field = {
            "u": [domain.U[0]],
            "v": [domain.U[1]],
            "w": [domain.U[2]],
        }
    f0 = HDF5File(f"dump{t//file_out_time:g}.h5", mode="w", domain=domain.d)
    f0.write(round(t // file_out_time), fields=field)
