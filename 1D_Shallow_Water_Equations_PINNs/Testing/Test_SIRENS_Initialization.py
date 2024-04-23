
# import External Libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

# import Internal Libraries
from functions.LBFGS import FullBatchLBFGS, LBFGS
from PINN_SWE import PINN

solution_directory = "Numerical_Solutions/No_Adv_No_Drag_No_Diff/Gaussian_Bell/"
# domain splitting & separate networks for u and h
learning_rate_annealing = False  # adaptive loss function weights according to "UNDERSTANDING AND MITIGATING GRADIENT PATHOLOGIES IN PHYSICS-INFORMED NEURAL NETWORKS"
mixed_activation_functions = False  # apply relu to half of the inputs and other activation function to the other half
number_of_models = 1  # number of models split over the time domain
train_only_on_first_interval = False  # train network only on the first section of the time interval
split_networks = False  # split network in two networks, one each for u and h
sirens_initialization = True  # initialization for model with sine activation functions

# network options
boundary_condition_transition_function = False  # option for explicitly constraining the boundary conditions
initial_condition_transition_function = False  # option for explicitly constraining the initial conditions
non_dimensionalization = True  # True: network describes solution in dimensionless space
save_output_over_training = True  # True: network output is saved on a discrete grid for later use
save_symbolic_function_over_training = True  # True: PDE Losses are saved on a discrete grid for later use

# initial and boundary condition settings
initial_sea_level = "gaussian"  # "gaussian" or "sine" - see Initial_And_Boundary_Conditions
boundary_conditions = "closed_boundaries"  # "closed_boundaries": u = 0 at boundaries, "periodic_boundaries"
new_initial_conditions = None  # initial conditions from a previous model to be passed to the next model
new_initial_condition_sampling_points = None  # respective sampling points of new_initial_conditions

# select parts of the objective function - every selected (True) part is added to the total MSE
train_on_solution = False  # MSE comparing the error compared to the numerical solution
train_on_PINNs_Loss = True  # MSE term for the initial and boundary conditions and the PDE term
train_on_boundary_condition_loss = True  # True: add BC loss term to total MSE, False: BC term is not considered
train_on_initial_condition_loss = True  # True: add IC loss term to total MSE, False: IC term is not considered


# dimensional parameters
momentum_advection = False  # include or exclude momentum advection from the momentum equation
initial_perturbation_amplitude = 1.0  # amplitude of sine wave, cosine or gaussian bell [m]
average_sea_level = 100.0  # reference sea level from which the elevations are modeled [m]
gravitational_acceleration = 9.81  # [ms^-1]
momentum_dissipation = 1e+4  # [m^2/s]
nonlinear_drag_coefficient = 0.0  # 2e-3  # dimensionless

# reference scales
horizontal_length_scale = 1000000.0  # length scale for horizontal lengths and velocities [m]
time_scale = 86400.0  # time scale for all processes (vertical and horizontal) [s]
vertical_scaling_factor = 1.0  # parameter for adjusting the vertical length scale see Bihlo and Popovych April 2021
vertical_length_scale = (vertical_scaling_factor * horizontal_length_scale ** 2 / (
        time_scale ** 2 * gravitational_acceleration))  # length scale for vertical lengths and gravity [m]

# space for sampling methods -> determines domain of training based on the numerical model
numerical_solution_time_interval = [0.0, 270000.0]  # [0.0, 129000.0]  # time interval in numerical model [m]
numerical_solution_time_step = 60.0  # time step in numerical discretization [s]
numerical_solution_x_interval = [-1000000.0, 1000000.0]  # spatial interval in numerical model [m]
numerical_solution_space_step = 10000.0  # spatial step in numerical discretization [m]
minimum_x = numerical_solution_x_interval[0]  # lower boundary of the spatial interval for training [m]
maximum_x = numerical_solution_x_interval[1]  # upper boundary of the spatial interval for training [m]

if non_dimensionalization == True:
    # All physical parameters and the physical space is scaled by the chosen scale lengths and time
    initial_perturbation_amplitude = initial_perturbation_amplitude / vertical_length_scale
    gravitational_acceleration = gravitational_acceleration * time_scale ** 2 / vertical_length_scale
    average_sea_level = average_sea_level / vertical_length_scale
    momentum_dissipation = momentum_dissipation * time_scale / vertical_length_scale ** 2
    nonlinear_drag_coefficient = nonlinear_drag_coefficient
    minimum_x = minimum_x / horizontal_length_scale
    maximum_x = maximum_x / horizontal_length_scale

# These weights determine the contribution of each Mean Squared Error (MSE) term to the total MSE
boundary_condition_weight = 1  # loss weight for the loss term including all boundary conditions
initial_condition_weight = 1  # loss weight for the loss term including all initial conditions
symbolic_function_weight = 1  # loss weight for the loss term including all PDEs of the system of equations

# The number of sampling points for computing the MSE terms for BC, IC and within the domain for the symbolic function
boundary_condition_batch_size = 1000  # number of sampling points on each boundary (split over the models)
initial_condition_batch_size = 1000  # number of sampling points for each initial condition (split over the models)
symbolic_function_batch_size = 50000  # number of sampling points within the x-t-domain (split over the models)

# set Hyper Parameters
device = "cpu"  # select "cuda" to run all computations in Pytorch on the GPU, otherwise select "cpu"
root_mean_squared_error = False  # if False: mean squared errors are used, if True: root mean squared errors
training_steps = 2000000  # sum of the number of training steps of all models
batch_resampling_period = training_steps  # number of training steps after which new collocation points are selected
console_output_period = 20  # number of training steps after which new collocation points are selected

# optimizer options
optimizer = torch.optim.Adam  # optimizer used for the entire training, if:  projected_gradients = False
learning_rate = 0.0005  # the default learning rates are 0.001 for Adam and 1 for LBFGS
line_search = None  # "strong_wolfe"  # valid options are None, "Armijo" or "strong_wolfe", only used for LBFGS
projected_gradients = True  # When this is True, the optimizer is automatically set to be Adam

# activation function and selection of the depth and width of the neural network
number_of_layers = 4  # number of hidden layers
neurons_per_layer = 100  # number of neurons in each hidden layer
layer_sizes = [2] + number_of_layers * [neurons_per_layer] + [2]  # list of layer sizes including input and output
activation_function = torch.tanh  # activation function applied on all hidden layers, but not on the output layer

if train_only_on_first_interval is True:
    model_range = np.arange(1)
    training_steps = training_steps * number_of_models
    symbolic_function_batch_size = symbolic_function_batch_size * number_of_models
else:
    model_range = np.arange(number_of_models)

start = time.time()

for model_number in model_range:
    # adjust time interval to the number of models
    fraction_of_time_interval = [0.0 + model_number / number_of_models,
                                 (model_number + 1) / number_of_models]
    minimum_time = (numerical_solution_time_interval[0] + fraction_of_time_interval[0]
                    * (numerical_solution_time_interval[1] - numerical_solution_time_interval[0]))
    maximum_time = (numerical_solution_time_interval[0] + fraction_of_time_interval[1]
                    * (numerical_solution_time_interval[1] - numerical_solution_time_interval[0]))  # [s]

    if non_dimensionalization == True:
        minimum_time = minimum_time / time_scale
        maximum_time = maximum_time / time_scale

    minimum_x = numerical_solution_x_interval[0]  # lower boundary of the spatial interval for training [m]
    maximum_x = numerical_solution_x_interval[1]  # upper boundary of the spatial interval for training [m]
    if non_dimensionalization == True:
        minimum_x = minimum_x / horizontal_length_scale
        maximum_x = maximum_x / horizontal_length_scale

    # Initialization of the Physics Informed Neural Network
    Physics_Informed_Neural_Network = PINN(layer_sizes, activation_function, optimizer, learning_rate, line_search,
                                           boundary_condition_weight, initial_condition_weight,
                                           symbolic_function_weight,
                                           int(boundary_condition_batch_size / number_of_models),
                                           int(initial_condition_batch_size),
                                           int(symbolic_function_batch_size / number_of_models),
                                           int(training_steps / number_of_models), batch_resampling_period,
                                           console_output_period, device, gravitational_acceleration,
                                           average_sea_level, momentum_dissipation,
                                           nonlinear_drag_coefficient, initial_sea_level,
                                           initial_perturbation_amplitude, boundary_conditions, non_dimensionalization,
                                           vertical_length_scale, vertical_scaling_factor, horizontal_length_scale,
                                           time_scale, minimum_time, maximum_time, minimum_x, maximum_x,
                                           projected_gradients, root_mean_squared_error,
                                           save_output_over_training, save_symbolic_function_over_training,
                                           numerical_solution_time_interval, numerical_solution_time_step,
                                           numerical_solution_x_interval, numerical_solution_space_step,
                                           fraction_of_time_interval, model_number, number_of_models,
                                           new_initial_conditions, new_initial_condition_sampling_points,
                                           train_on_solution, train_on_PINNs_Loss,
                                           boundary_condition_transition_function, initial_condition_transition_function,
                                           split_networks, train_on_boundary_condition_loss,
                                           train_on_initial_condition_loss, momentum_advection,
                                           mixed_activation_functions, sirens_initialization,
                                           learning_rate_annealing, solution_directory).to(device)

    Physics_Informed_Neural_Network.resample_sampling_points()

    input = torch.cat((Physics_Informed_Neural_Network.time_input_grid,
                        Physics_Informed_Neural_Network.x_input_grid), dim=1)

    output = Physics_Informed_Neural_Network.activation_function(Physics_Informed_Neural_Network.layers[0](input))

    plt.figure()
    plt.plot(Physics_Informed_Neural_Network.x_grid, output)
    plt.title("Structure of first layer")
    plt.savefig("First_Layer_Init.png")
    plt.show()

    plt.figure()
    plt.hist(Physics_Informed_Neural_Network.layers[0](input).cpu().detach().numpy())
    plt.title("Wx in first hidden layer")
    plt.savefig("Wx_layer_1.png")
    plt.show()

    plt.figure()
    plt.hist(output.cpu().detach().numpy())
    plt.title("Activations in first hidden layer")
    plt.savefig("Activations_layer_1.png")
    plt.show()

    for i in range(1, Physics_Informed_Neural_Network.number_of_layers-1):

        plt.figure()
        plt.hist(Physics_Informed_Neural_Network.layers[i](output).cpu().detach().numpy())
        plt.title("Wx in hidden layer no. " + str(i + 1))
        plt.savefig("Wx_layer_" + str(i+1) + ".png")
        plt.show()

        output = Physics_Informed_Neural_Network.activation_function(Physics_Informed_Neural_Network.layers[i](output))

        plt.figure()
        plt.hist(output.cpu().detach().numpy())
        plt.title("Activations in hidden layer no. " + str(i+1))
        plt.savefig("Activations_layer_" + str(i + 1) + ".png")
        plt.show()

    output = Physics_Informed_Neural_Network.layers[Physics_Informed_Neural_Network.number_of_layers - 1](output)

    plt.figure()
    plt.hist(output[:, 0].cpu().detach().numpy())
    plt.title("Output u")
    plt.show()

    plt.figure()
    plt.hist(output[:, 1].cpu().detach().numpy())
    plt.title(r"Output $\zeta$")
    plt.show()
