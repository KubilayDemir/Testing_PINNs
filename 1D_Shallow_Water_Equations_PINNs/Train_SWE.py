# import External Libraries
import json
import os
import time

import numpy as np
import torch

# import Internal Libraries
from functions.LBFGS import LBFGS, FullBatchLBFGS
from PINN_SWE import PINN

numerical_solution_directory = "AH=5e+4"  # "AH=5e+4" or "AH=0"

# training options
minibatch_training = True  # mini batch gradient descent
learning_rate_annealing = False  # adaptive loss function weights
mixed_activation_functions = False  # combination of relu and other activations
number_of_models = 1  # number of models split over the time domain
non_dimensionalization = True  # True: network describes solution in dimensionless space

# model architecture and options
boundary_condition_transition_function = False  # option for explicitly constraining the boundary conditions
initial_condition_transition_function = False  # option for explicitly constraining the initial conditions
split_networks = False  # split model in two networks, one each for u and h
sirens_initialization = False  # initialization for model with sine activation functions
save_output_over_training = True  # True: network output is saved on a discrete grid for later use
save_symbolic_function_over_training = True  # True: PDE Losses are saved on a discrete grid for later use

# select parts of the objective function - every selected (True) part is added to the total MSE
train_on_solution = False  # MSE comparing the error compared to the numerical solution
train_on_PINNs_Loss = True  # MSE term for the initial and boundary conditions and the PDE term
train_on_boundary_condition_loss = True  # True: add BC loss term to total MSE, False: BC term is not considered
train_on_initial_condition_loss = True  # True: add IC loss term to total MSE, False: IC term is not considered

# dimensional parameters
momentum_advection = True  # include or exclude momentum advection from the momentum equation
initial_perturbation_amplitude = 1.0  # amplitude of sine wave, cosine or gaussian bell [m]
average_sea_level = 100.0  # reference sea level from which the elevations are modeled [m]
gravitational_acceleration = 9.81  # [ms^-1]
momentum_dissipation = 5e+4  # (for "AH=5e+4")  # [m^2/s]
nonlinear_drag_coefficient = 0.0  # dimensionless (~2e-3)

# reference scales
horizontal_length_scale = 1000000.0  # [m]
time_scale = 86400.0  # [s]
vertical_scaling_factor = 1.0  # adjusting between vertical and horizontal length scale
vertical_length_scale = (
        vertical_scaling_factor * horizontal_length_scale ** 2 / (time_scale ** 2 * gravitational_acceleration)
)  # length scale for vertical lengths and gravity [m]

# spatio-temporal domain of numerical model and training
numerical_solution_time_interval = [0.0, 270000.0]  # [0.0, 129000.0]  # time domain [s]
numerical_solution_time_step = 60.0  # time step [s]
numerical_solution_x_interval = [-1000000.0, 1000000.0]  # spatial domain [m]
numerical_solution_space_step = 10000.0  # spatial step size [m]
minimum_x = numerical_solution_x_interval[0]  # lower boundary [m]
maximum_x = numerical_solution_x_interval[1]  # upper boundary [m]

# scaling by length and time scales
if non_dimensionalization is True:
    initial_perturbation_amplitude = initial_perturbation_amplitude / vertical_length_scale
    gravitational_acceleration = gravitational_acceleration * time_scale ** 2 / vertical_length_scale
    average_sea_level = average_sea_level / vertical_length_scale
    momentum_dissipation = momentum_dissipation * time_scale / horizontal_length_scale ** 2
    minimum_x = minimum_x / horizontal_length_scale
    maximum_x = maximum_x / horizontal_length_scale

# contribution of each loss term to the total loss function
boundary_condition_weight = 1  # loss weight for boundary conditions
initial_condition_weight = 1  # loss weight for initial conditions
symbolic_function_weight = 1  # loss weight for symbolic form of PDEs

# sampling points for loss terms for BC, IC and within the domain for the symbolic function
boundary_condition_batch_size = 40000  # sampling points on each boundary
initial_condition_batch_size = 40000  # sampling points for each initial condition
symbolic_function_batch_size = 100000  # sampling points within the x-t-domain

# batch sizes for mini batch gradient descent
bc_mini_batch_size = 200
ic_mini_batch_size = 200
pde_mini_batch_size = 500
epochs = 10000
device = "cuda"  # "cuda" for GPU or "cpu"
iterations_per_epoch = int(symbolic_function_batch_size / pde_mini_batch_size)
batch_resampling_period = 1000 * epochs  # number of training steps after which new collocation points are selected
output_period = 1  #

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
model_range = np.arange(number_of_models)

# initialize new initial conditions for use of several models
new_initial_conditions = None  # initial conditions from a previous model to be passed to the next model
new_initial_condition_sampling_points = None  # respective sampling points of new_initial_conditions

start = time.time()

for model_number in model_range:

    # adjust time interval to the number of models
    fraction_of_time_interval = [0.0 + model_number / number_of_models, (model_number + 1) / number_of_models]
    minimum_time = numerical_solution_time_interval[0] + fraction_of_time_interval[0] * (
            numerical_solution_time_interval[1] - numerical_solution_time_interval[0]
    )
    maximum_time = numerical_solution_time_interval[0] + fraction_of_time_interval[1] * (
            numerical_solution_time_interval[1] - numerical_solution_time_interval[0]
    )  # [s]

    if non_dimensionalization is True:
        minimum_time = minimum_time / time_scale
        maximum_time = maximum_time / time_scale

    minimum_x = numerical_solution_x_interval[0]  # lower boundary of the spatial interval for training [m]
    maximum_x = numerical_solution_x_interval[1]  # upper boundary of the spatial interval for training [m]
    if non_dimensionalization is True:
        minimum_x = minimum_x / horizontal_length_scale
        maximum_x = maximum_x / horizontal_length_scale

    # Initialization of the Physics Informed Neural Network
    Physics_Informed_Neural_Network = PINN(
        layer_sizes,
        activation_function,
        optimizer,
        learning_rate,
        line_search,
        boundary_condition_weight,
        initial_condition_weight,
        symbolic_function_weight,
        int(boundary_condition_batch_size / number_of_models),
        int(initial_condition_batch_size),
        int(symbolic_function_batch_size / number_of_models),
        int(epochs / number_of_models),
        batch_resampling_period,
        output_period,
        device,
        gravitational_acceleration,
        average_sea_level,
        momentum_dissipation,
        nonlinear_drag_coefficient,
        initial_perturbation_amplitude,
        non_dimensionalization,
        vertical_length_scale,
        vertical_scaling_factor,
        horizontal_length_scale,
        time_scale,
        minimum_time,
        maximum_time,
        minimum_x,
        maximum_x,
        projected_gradients,
        save_output_over_training,
        save_symbolic_function_over_training,
        numerical_solution_time_interval,
        numerical_solution_time_step,
        numerical_solution_x_interval,
        numerical_solution_space_step,
        fraction_of_time_interval,
        model_number,
        number_of_models,
        new_initial_conditions,
        new_initial_condition_sampling_points,
        train_on_solution,
        train_on_PINNs_Loss,
        boundary_condition_transition_function,
        initial_condition_transition_function,
        split_networks,
        train_on_boundary_condition_loss,
        train_on_initial_condition_loss,
        momentum_advection,
        mixed_activation_functions,
        sirens_initialization,
        learning_rate_annealing,
        minibatch_training,
        pde_mini_batch_size,
        bc_mini_batch_size,
        ic_mini_batch_size,
        iterations_per_epoch,
        numerical_solution_directory,
    ).to(device)

    # Train Model with parameters chosen above -> generate model output and MSE over training
    Physics_Informed_Neural_Network.train_PINN()

    end = time.time()
    computation_time = end - start

    # load best state dict
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    best_step = Physics_Informed_Neural_Network.best_step
    Physics_Informed_Neural_Network.load_state_dict(
        torch.load(("Best_State_Dict_" + str(Physics_Informed_Neural_Network.model_number) + "_" + str(slurm_job_id)))
    )
    if model_number == 0:
        os.chdir(slurm_job_id)
    os.mkdir(str(model_number))
    os.chdir(str(model_number))

    # Save Model Parameters
    torch.save(Physics_Informed_Neural_Network.state_dict(), ("TrainedParameters_SWE_" + str(model_number)))

    # Save new initial conditions and respective sampling points
    [new_initial_conditions, new_initial_condition_sampling_points] = Physics_Informed_Neural_Network.Save_Final_State()

    np.save("improvement_steps", Physics_Informed_Neural_Network.improvement_steps)
    # save different loss terms over training
    np.save("total_MSE_over_training", Physics_Informed_Neural_Network.total_MSE_over_training)
    np.save(
        "MSE_boundary_conditions_over_training", Physics_Informed_Neural_Network.MSE_boundary_conditions_over_training
    )
    np.save(
        "MSE_initial_conditions_over_training", Physics_Informed_Neural_Network.MSE_initial_conditions_over_training
    )
    np.save(
        "MSE_symbolic_functions_over_training", Physics_Informed_Neural_Network.MSE_symbolic_functions_over_training
    )
    np.save("Relative_L2_Error_h_over_training", Physics_Informed_Neural_Network.MSE_numerical_solution_h_over_training)
    np.save("Relative_L2_Error_u_over_training", Physics_Informed_Neural_Network.MSE_numerical_solution_u_over_training)
    np.save(
        "MSE_initial_condition_u_over_training", Physics_Informed_Neural_Network.MSE_initial_condition_u_over_training
    )
    np.save(
        "MSE_initial_condition_h_over_training", Physics_Informed_Neural_Network.MSE_initial_condition_h_over_training
    )
    np.save(
        "MSE_symbolic_function_u_over_training", Physics_Informed_Neural_Network.MSE_symbolic_function_u_over_training
    )
    np.save(
        "MSE_symbolic_function_h_over_training", Physics_Informed_Neural_Network.MSE_symbolic_function_h_over_training
    )
    np.save(
        "MSE_symbolic_function_h_over_training", Physics_Informed_Neural_Network.MSE_symbolic_function_h_over_training
    )

    np.save(
        "MSE_lower_boundary_condition_u_over_training",
        Physics_Informed_Neural_Network.MSE_lower_boundary_condition_u_over_training,
    )
    np.save(
        "MSE_upper_boundary_condition_u_over_training",
        Physics_Informed_Neural_Network.MSE_upper_boundary_condition_u_over_training,
    )

    if Physics_Informed_Neural_Network.learning_rate_annealing is True:
        np.save(
            "symbolic_function_weight_over_training",
            Physics_Informed_Neural_Network.symbolic_function_weight_over_training,
        )
        np.save(
            "initial_condition_weight_over_training",
            Physics_Informed_Neural_Network.initial_condition_weight_over_training,
        )
        np.save(
            "boundary_condition_weight_over_training",
            Physics_Informed_Neural_Network.boundary_condition_weight_over_training,
        )

    # save grids for plots
    if Physics_Informed_Neural_Network.non_dimensionalization is True:
        np.save("dimensional_time_mesh_grid", Physics_Informed_Neural_Network.dimensional_time_mesh_grid)
        np.save("dimensional_x_mesh_grid", Physics_Informed_Neural_Network.dimensional_x_mesh_grid)

    np.save("time_mesh_grid", Physics_Informed_Neural_Network.time_mesh_grid)
    np.save("x_mesh_grid", Physics_Informed_Neural_Network.x_mesh_grid)
    np.save("time_input_grid", Physics_Informed_Neural_Network.time_input_grid.cpu().detach().numpy())
    np.save("x_input_grid", Physics_Informed_Neural_Network.x_input_grid.cpu().detach().numpy())
    np.save("mesh_grid_shape", Physics_Informed_Neural_Network.mesh_grid_shape)
    np.save(
        "zeta_solution_time_input_grid",
        Physics_Informed_Neural_Network.zeta_solution_time_input_grid.cpu().detach().numpy(),
    )
    np.save(
        "u_solution_time_input_grid", Physics_Informed_Neural_Network.u_solution_time_input_grid.cpu().detach().numpy()
    )
    np.save(
        "zeta_solution_x_input_grid", Physics_Informed_Neural_Network.zeta_solution_x_input_grid.cpu().detach().numpy()
    )
    np.save("u_solution_x_input_grid", Physics_Informed_Neural_Network.u_solution_x_input_grid.cpu().detach().numpy())
    np.save(
        "dimensional_zeta_solution_time_mesh_grid",
        Physics_Informed_Neural_Network.dimensional_zeta_solution_time_mesh_grid,
    )
    np.save(
        "dimensional_u_solution_time_mesh_grid", Physics_Informed_Neural_Network.dimensional_u_solution_time_mesh_grid
    )
    np.save(
        "dimensional_zeta_solution_x_mesh_grid", Physics_Informed_Neural_Network.dimensional_zeta_solution_x_mesh_grid
    )
    np.save("dimensional_u_solution_x_mesh_grid", Physics_Informed_Neural_Network.dimensional_u_solution_x_mesh_grid)
    np.save(
        "dimensional_solution_mesh_grid_shape",
        Physics_Informed_Neural_Network.dimensional_zeta_solution_x_mesh_grid.shape,
    )

    # save output
    network_output_h_values = (
        Physics_Informed_Neural_Network(
            Physics_Informed_Neural_Network.zeta_solution_time_input_grid,
            Physics_Informed_Neural_Network.zeta_solution_x_input_grid,
        )[:, 1:2]
        .cpu()
        .detach()
        .numpy()
        .reshape(Physics_Informed_Neural_Network.zeta_solution_mesh_grid_shape)
    )
    network_output_u_values = (
        Physics_Informed_Neural_Network(
            Physics_Informed_Neural_Network.u_solution_time_input_grid,
            Physics_Informed_Neural_Network.u_solution_x_input_grid,
        )[:, 0:1]
        .cpu()
        .detach()
        .numpy()
        .reshape(Physics_Informed_Neural_Network.u_solution_mesh_grid_shape)
    )
    np.save("network_output_h_values", network_output_h_values)
    np.save("network_output_u_values", network_output_u_values)

    if Physics_Informed_Neural_Network.non_dimensionalization is True:
        dimensional_network_output_h_values = (
                network_output_h_values * Physics_Informed_Neural_Network.vertical_length_scale
        )
        dimensional_network_output_u_values = (
                network_output_u_values
                * Physics_Informed_Neural_Network.horizontal_length_scale
                / Physics_Informed_Neural_Network.time_scale
        )
        np.save("dimensional_network_output_h_values", dimensional_network_output_h_values)
        np.save("dimensional_network_output_u_values", dimensional_network_output_u_values)

    # save numerical solution
    exact_solution_u_values = Physics_Informed_Neural_Network.exact_solution_u_values.cpu().detach().numpy()
    exact_solution_h_values = Physics_Informed_Neural_Network.exact_solution_h_values.cpu().detach().numpy()
    np.save("exact_solution_u_values", exact_solution_u_values)
    np.save("exact_solution_h_values", exact_solution_h_values)

    abs_error_h_values = abs(exact_solution_h_values - dimensional_network_output_h_values)
    abs_error_u_values = abs(exact_solution_u_values - dimensional_network_output_u_values)
    np.save("abs_error_h_values", abs_error_h_values)
    np.save("abs_error_u_values", abs_error_u_values)

    # save initial condition output
    minimum_time = Physics_Informed_Neural_Network.minimum_time
    x = (
        torch.FloatTensor(Physics_Informed_Neural_Network.x_grid)
        .unsqueeze(dim=1)
        .to(Physics_Informed_Neural_Network.device)
    )
    t = torch.zeros_like(x).to(Physics_Informed_Neural_Network.device)

    # define network output and target initial condition
    if Physics_Informed_Neural_Network.model_number == 0:
        true_initial_condition_h_values = (
            Physics_Informed_Neural_Network.true_initial_condition_h_function(x).cpu().detach().numpy()
        )
        network_output_h_initial_conditions = Physics_Informed_Neural_Network(t, x)[:, 1:2].cpu().detach().numpy()
        true_initial_condition_u_values = (
            Physics_Informed_Neural_Network.true_initial_condition_u_function(x).cpu().detach().numpy()
        )
        network_output_u_initial_conditions = Physics_Informed_Neural_Network(t, x)[:, 0:1].cpu().detach().numpy()

        if Physics_Informed_Neural_Network.non_dimensionalization is True:
            x = x * Physics_Informed_Neural_Network.horizontal_length_scale
            true_initial_condition_h_values = (
                    true_initial_condition_h_values * Physics_Informed_Neural_Network.vertical_length_scale
            )
            network_output_h_initial_conditions = (
                    network_output_h_initial_conditions * Physics_Informed_Neural_Network.vertical_length_scale
            )
            true_initial_condition_u_values = (
                    true_initial_condition_u_values
                    * Physics_Informed_Neural_Network.horizontal_length_scale
                    / Physics_Informed_Neural_Network.time_scale
            )
            network_output_u_initial_conditions = (
                    network_output_u_initial_conditions
                    * Physics_Informed_Neural_Network.horizontal_length_scale
                    / Physics_Informed_Neural_Network.time_scale
            )
            minimum_time = minimum_time * Physics_Informed_Neural_Network.time_scale

        np.save("true_initial_condition_h_values", true_initial_condition_h_values)
        np.save("network_output_h_initial_conditions", network_output_h_initial_conditions)
        np.save("true_initial_condition_u_values", true_initial_condition_u_values)
        np.save("network_output_u_initial_conditions", network_output_u_initial_conditions)

    # save boundary condition output
    dimensional_minimum_x = Physics_Informed_Neural_Network.minimum_x
    dimensional_maximum_x = Physics_Informed_Neural_Network.maximum_x
    t = (
        torch.FloatTensor(Physics_Informed_Neural_Network.time_grid)
        .unsqueeze(dim=1)
        .to(Physics_Informed_Neural_Network.device)
    )
    x_lower_boundary = -1.0 * torch.ones_like(t).to(Physics_Informed_Neural_Network.device)
    x_upper_boundary = 1.0 * torch.ones_like(t).to(Physics_Informed_Neural_Network.device)

    true_lower_boundary_condition_u_values = (
        Physics_Informed_Neural_Network.true_lower_boundary_condition_u_function(t).cpu().detach().numpy()
    )
    true_upper_boundary_condition_u_values = (
        Physics_Informed_Neural_Network.true_upper_boundary_condition_u_function(t).cpu().detach().numpy()
    )
    network_output_u_lower_boundary_condition = (
        Physics_Informed_Neural_Network(t, x_lower_boundary)[:, 0:1].cpu().detach().numpy()
    )
    network_output_u_upper_boundary_condition = (
        Physics_Informed_Neural_Network(t, x_upper_boundary)[:, 0:1].cpu().detach().numpy()
    )

    # re-dimensionalize
    if Physics_Informed_Neural_Network.non_dimensionalization is True:
        t = t * Physics_Informed_Neural_Network.time_scale
        true_lower_boundary_condition_u_values = (
                true_lower_boundary_condition_u_values
                * Physics_Informed_Neural_Network.horizontal_length_scale
                / Physics_Informed_Neural_Network.time_scale
        )
        network_output_u_lower_boundary_condition = (
                network_output_u_lower_boundary_condition
                * Physics_Informed_Neural_Network.horizontal_length_scale
                / Physics_Informed_Neural_Network.time_scale
        )
        true_upper_boundary_condition_u_values = (
                true_upper_boundary_condition_u_values
                * Physics_Informed_Neural_Network.horizontal_length_scale
                / Physics_Informed_Neural_Network.time_scale
        )
        network_output_u_upper_boundary_condition = (
                network_output_u_upper_boundary_condition
                * Physics_Informed_Neural_Network.horizontal_length_scale
                / Physics_Informed_Neural_Network.time_scale
        )
        dimensional_minimum_x = dimensional_minimum_x * Physics_Informed_Neural_Network.horizontal_length_scale
        dimensional_maximum_x = dimensional_maximum_x * Physics_Informed_Neural_Network.horizontal_length_scale

    np.save("true_lower_boundary_condition_u_values", true_lower_boundary_condition_u_values)
    np.save("network_output_u_lower_boundary_condition", network_output_u_lower_boundary_condition)
    np.save("true_upper_boundary_condition_u_values", true_upper_boundary_condition_u_values)
    np.save("network_output_u_upper_boundary_condition", network_output_u_upper_boundary_condition)

    # save PDE losses over domain
    mesh_grid_shape = Physics_Informed_Neural_Network.mesh_grid_shape
    t = Physics_Informed_Neural_Network.time_input_grid.clone().detach().requires_grad_(True)
    x = Physics_Informed_Neural_Network.x_input_grid.clone().detach().requires_grad_(True)
    Physics_Informed_Neural_Network.symbolic_function_sampling_points = torch.hstack((t, x))
    Physics_Informed_Neural_Network.Physics_Informed_Symbolic_Function()

    time_mesh_grid = Physics_Informed_Neural_Network.time_mesh_grid * Physics_Informed_Neural_Network.time_scale
    x_mesh_grid = Physics_Informed_Neural_Network.x_mesh_grid * Physics_Informed_Neural_Network.horizontal_length_scale
    PDE_Loss_u = Physics_Informed_Neural_Network.symbolic_function_u_values.reshape(shape=mesh_grid_shape)
    PDE_Loss_h = Physics_Informed_Neural_Network.symbolic_function_h_values.reshape(shape=mesh_grid_shape)
    np.save("PDE_Loss_u", PDE_Loss_u.cpu().detach().numpy())
    np.save("PDE_Loss_h", PDE_Loss_h.cpu().detach().numpy())

    # save network output & losses over training
    np.save("network_output_h_over_training", Physics_Informed_Neural_Network.network_output_h_over_training)
    np.save("network_output_u_over_training", Physics_Informed_Neural_Network.network_output_u_over_training)
    np.save("symbolic_function_u_over_training", Physics_Informed_Neural_Network.symbolic_function_u_over_training)
    np.save("symbolic_function_h_over_training", Physics_Informed_Neural_Network.symbolic_function_h_over_training)

    print("Mean Time Per Epoch: " + str(np.mean(Physics_Informed_Neural_Network.time_per_epoch)))
    print("Mean Time Per Iteration: " + str(np.mean(Physics_Informed_Neural_Network.time_per_iteration)))

    np.save("time_per_epoch", Physics_Informed_Neural_Network.time_per_epoch)
    np.save("time_per_iteration", Physics_Informed_Neural_Network.time_per_iteration)

    Hyper_Parameter_Dictionary = {
        "number_of_models": number_of_models,
        "split_networks": split_networks,
        "boundary_condition_transition_function": boundary_condition_transition_function,
        "initial_condition_transition_function": initial_condition_transition_function,
        "non_dimensionalization": non_dimensionalization,
        "save_output_over_training": save_output_over_training,
        "save_symbolic_function_over_training": save_symbolic_function_over_training,
        "train_on_solution": train_on_solution,
        "train_on_PINNs_Loss": train_on_PINNs_Loss,
        "train_on_boundary_condition_loss": train_on_boundary_condition_loss,
        "train_on_initial_condition_loss": train_on_initial_condition_loss,
        "momentum_advection": momentum_advection,
        "initial_perturbation_amplitude": initial_perturbation_amplitude,
        "average_sea_level": average_sea_level,
        "gravitational_acceleration": gravitational_acceleration,
        "momentum_dissipation": momentum_dissipation,
        "nonlinear_drag_coefficient": nonlinear_drag_coefficient,
        "horizontal_length_scale": horizontal_length_scale,
        "time_scale": time_scale,
        "vertical_scaling_factor": vertical_scaling_factor,
        "vertical_length_scale": vertical_length_scale,
        "numerical_solution_time_interval": numerical_solution_time_interval,
        "numerical_solution_time_step": numerical_solution_time_step,
        "numerical_solution_x_interval": numerical_solution_x_interval,
        "numerical_solution_space_step": numerical_solution_space_step,
        "minimum_x": minimum_x,
        "maximum_x": maximum_x,
        "minimum_time": minimum_time,
        "maximum_time": maximum_time,
        "boundary_condition_weight": boundary_condition_weight,
        "initial_condition_weight": initial_condition_weight,
        "symbolic_function_weight": symbolic_function_weight,
        "boundary_condition_batch_size": boundary_condition_batch_size,
        "initial_condition_batch_size": initial_condition_batch_size,
        "symbolic_function_batch_size": symbolic_function_batch_size,
        "device": device,
        "epochs": epochs,
        "batch_resampling_period": batch_resampling_period,
        "console_output_period": output_period,
        "optimizer": str(optimizer),
        "learning_rate": learning_rate,
        "line_search": line_search,
        "projected_gradients": projected_gradients,
        "number_of_layers": number_of_layers,
        "neurons_per_layer": neurons_per_layer,
        "activation_function": str(activation_function),
        "model_number": float(model_number),
        "mixed_activation_functions": mixed_activation_functions,
        "computation_time": computation_time,
        "best_step": best_step,
        "learning_rate_annealing": learning_rate_annealing,
        "pde_mini_batch_size": pde_mini_batch_size,
        "bc_mini_batch_size": bc_mini_batch_size,
        "ic_mini_batch_size": ic_mini_batch_size,
        "iterations_per_epoch": iterations_per_epoch,
    }

    # create json object from dictionary
    json_dict = json.dumps(Hyper_Parameter_Dictionary)
    # open file for writing, "w"
    f = open("Hyper_Parameter_Dictionary.json", "w")
    # write json object to file
    f.write(json_dict)
    # close file
    f.close()
    os.chdir("..")
