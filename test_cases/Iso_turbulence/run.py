import sys
import atexit
sys.path.append("/home/anil/10TB/Work/spectral_method_python/main_code/")
from numpy import pi, random, exp, sin, cos
from src.utilites import extract_files
from src.initialize_domain import InitializeDomain
from src.initial import initial_vorticity, initial_velocity
from src.SM_solver import spectralMethodSolver

@atexit.register
def exit_hand():
    if my_domain.rank == 0:
        extract_files()


################### Parameters ####################
parameters = {
    "LES ON": True,
    "nu": 0.01,
    "CFL": 0.5,
    "final time": 300.0,
    "h5 file out time": 0.25,
    "No of EK plot": 0,
    "dimension": 3,
    "Time step": 0.001,
    "No of gridpoints": 32*4,
    "Forcing value": 0.1,
    "Range of forced K": [0, 5],
    "Method for time integration": "RK4",
    "Cut-off wavenumber": 35.0,
    "LES_model":["Smagorinsky", "vreman"]
}

###################################################


my_domain = InitializeDomain(
    nu=parameters["nu"],
    final_time=parameters["final time"],
    dimension=parameters["dimension"],
    time_stepSize=parameters["Time step"],
    No_of_gridpoint=parameters["No of gridpoints"],
    forcing=parameters["Forcing value"],
    forcing_range=parameters["Range of forced K"],
    method=parameters["Method for time integration"],
    LES_ON=parameters["LES ON"],
    CFL=parameters["CFL"],
    Cs=0.125,
    LES_model=parameters["LES_model"][1]
)

############### Initial Condition ####################

initial_velocity(
    my_domain,
    cos(my_domain.X[1]) + sin(my_domain.X[2]),
    sin(my_domain.X[0]) + cos(my_domain.X[2]),
    cos(my_domain.X[0]) + sin(my_domain.X[1]),
) #X_full

#######################################################

if (len(sys.argv) >= 2 ):
    if my_domain.rank == 0 and (sys.argv[1]).upper() == "E":
        extract_files()
else:   
    spectralMethodSolver(
        domain=my_domain,
        file_out_time=parameters["h5 file out time"],
        No_Ek_plot=parameters["No of EK plot"],
    )
