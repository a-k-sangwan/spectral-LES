import os
import sys
import time
from mpi4py_fft.io import generate_xdmf



def make_directory(name, delete_old=True):
    path = name
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    else:
        if delete_old:
            for file in os.listdir(path):
                os.remove(path + "/" + file)


def get_index(N, n, rank):
    batch_size = N // n
    remaining = N % n
    if rank < remaining:
        return (rank * (batch_size + 1), (rank + 1) * (batch_size + 1))
    else:
        return (rank * batch_size + remaining, (rank + 1) * batch_size + remaining)


def extract_files(path:str = './'):
    for file in os.listdir('./'):
        if file.endswith('.h5'):
            generate_xdmf(file)
            os.rename(file, 'dumpfile/'+file)
            name = os.path.splitext(file)[0] + '.xdmf'
            os.rename(name, 'dumpfile/'+name)
    return


def print_time(final_time, dt, t,start, max_u, diss_rate = None, aa=0.0):
    
    complete = (t * 40 / final_time) if final_time > dt else 40
    if diss_rate != None:
        print("\033[A\033[A\033[A\033[A =>  Time: {0:<10g}, dt: {2:<10g} a: {1:<10g}\n".format(t,max_u, dt), end="")
        print(f"=>  Dissipation Rate: {diss_rate:<15g}, Sim. a: {aa:<g}\n", end="")
    else:
        print("\033[A\033[A\033[A =>  Time: {0:<10g}, dt: {2:<10g}, Max Velocity: {1:<g}\n".format(t,max_u, dt), end="")
    sys.stdout.write("Progress: [%-40s] %d%%\n" % ("=" * round(complete), 2.5 * complete))
    end = time.time()
    total_time = end - start
    print(
        f"""Time elapsed: {(total_time) // 3600 :g} Hours, {((total_time)% 3600) // 60:g} Min, {(((total_time)% 3600) % 60):<8g} sec{"":5s}"""
    )
    sys.stdout.flush()


def print_initial_decorator(domain):
    print(
        """
*************************** SIMULATION STARTS ***************************

{0:}D Pseudo Spectral Solver :-

Simulation Parameters: 
    {1:<30s} |   {2:<20s}
    {4:<30s} |   {3:<20s}
    {5:<30s} |   {6:s}, {7:<10s}
        
        

""".format(
            f'{domain.dim}',
            f'Grid Size = {[domain.N] * domain.dim}',
            f'Final Time  = {domain.T}',
            f'No. of processor = {domain.comm.Get_size()}',
            f'Kinematic Viscosity = {domain.nu}',
            f"Forcing = ON ({domain.forcing:g})" if domain.forcing > 0.0 else "Forcing = OFF",
            f'CFL = {domain.CFL}',
            f'Cs = {domain.Cs}' if domain.LES_ON else "",
        )
    )

########
# def print_initial_decorator(domain):
#     print(
#         """
# *************************** SIMULATION STARTS ***************************

# {}D Pseudo Spectral Solver :-

# Simulation Parameters: 
#         Grid Size   = {:} 
#         Final Time  = {}
#         No. of processor = {}
#         Kinematic Viscosity = {}
#         Forcing = {}
#         CFL = {}
        
        

# """.format(
#             domain.dim,
#             [domain.N] * domain.dim,
#             domain.T,
#             domain.comm.Get_size(),
#             domain.nu,
#             f"ON ({domain.forcing:g})" if domain.forcing > 0.0 else "OFF",
#             domain.CFL,
#         )
#     )