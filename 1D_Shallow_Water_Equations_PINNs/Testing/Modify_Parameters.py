# import External Libraries
import matplotlib.pyplot as plt
import numpy as np
import torch

# import Internal Libraries
from PINN_SWE import PINN

# Problem Parameters
gravitational_acceleration = 9.81  # [ms^-2]
average_sea_level = 1.0  # 100.0 # [m]
momentum_dissipation = 0.0
nonlinear_drag_coefficient = 0.0
vertical_length_scale = 1.0  # [m]
horizontal_length_scale = 1.0  # 1e+6 # [m]
time_scale = 1.0  # 2.0 * horizontal_length_scale / np.sqrt(gravitational_acceleration * average_sea_level)

# dimensionless space
minimum_time = 0.0
maximum_time = 1.0
minimum_x = -1.0
maximum_x = 1.0

# These weights determine the contribution of each Mean Squared Error (MSE) term to the total MSE
boundary_condition_weight = 1
initial_condition_weight = 10
symbolic_function_weight = 1

# The number of sampling points for computing the MSE terms for BC, IC and within the domain for the symbolic function
boundary_condition_batch_size = 1000
initial_condition_batch_size = 1000
symbolic_function_batch_size = 50000

# set Training Parameters and Network Architecture
learning_rate = 0.005
training_steps = 200
optimizer = torch.optim.LBFGS
batch_resampling_period = training_steps
training_step_counting_rate = 1
activation_function = torch.tanh
# layer_sizes = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 2]
layer_sizes = [2, 100, 100, 100, 100, 100, 100, 100, 100, 100, 2]
device = "cpu"

# Initialization of the Physics Informed Neural Network
Physics_Informed_Neural_Network = PINN(
    layer_sizes,
    activation_function,
    optimizer,
    learning_rate,
    boundary_condition_weight,
    initial_condition_weight,
    symbolic_function_weight,
    boundary_condition_batch_size,
    initial_condition_batch_size,
    symbolic_function_batch_size,
    training_steps,
    batch_resampling_period,
    training_step_counting_rate,
    device,
    gravitational_acceleration,
    average_sea_level,
    momentum_dissipation,
    nonlinear_drag_coefficient,
    vertical_length_scale,
    horizontal_length_scale,
    time_scale,
    minimum_time,
    maximum_time,
    minimum_x,
    maximum_x,
).to(device)

# Load Model Parameters from previous training (optional -> TrainedParameters must exist)
Physics_Informed_Neural_Network.load_state_dict(
    torch.load("1D_Shallow_Water_Equations/Trained_Parameters/TrainedParameters_SWE")
)
# Physics_Informed_Neural_Network.eval()

Physics_Informed_Neural_Network.resample_sampling_points()

Physics_Informed_Neural_Network.network_output_lower_boundary_conditions = Physics_Informed_Neural_Network.forward(
    Physics_Informed_Neural_Network.lower_boundary_condition_sampling_points[:, 0:1],
    Physics_Informed_Neural_Network.lower_boundary_condition_sampling_points[:, 1:2],
)
Physics_Informed_Neural_Network.network_output_upper_boundary_conditions = Physics_Informed_Neural_Network.forward(
    Physics_Informed_Neural_Network.upper_boundary_condition_sampling_points[:, 0:1],
    Physics_Informed_Neural_Network.upper_boundary_condition_sampling_points[:, 1:2],
)
Physics_Informed_Neural_Network.network_output_initial_conditions = Physics_Informed_Neural_Network.forward(
    Physics_Informed_Neural_Network.initial_condition_sampling_points[:, 0:1],
    Physics_Informed_Neural_Network.initial_condition_sampling_points[:, 1:2],
)

# compute real initial and boundary values
Physics_Informed_Neural_Network.true_upper_boundary_condition_u_values = (
    Physics_Informed_Neural_Network.true_upper_boundary_condition_u_function(
        Physics_Informed_Neural_Network.upper_boundary_condition_sampling_points[:, 0:1]
    )
)
Physics_Informed_Neural_Network.true_lower_boundary_condition_u_values = (
    Physics_Informed_Neural_Network.true_lower_boundary_condition_u_function(
        Physics_Informed_Neural_Network.lower_boundary_condition_sampling_points[:, 0:1]
    )
)
Physics_Informed_Neural_Network.true_initial_condition_u_values = (
    Physics_Informed_Neural_Network.true_initial_condition_u_function(
        Physics_Informed_Neural_Network.initial_condition_sampling_points[:, 1:2]
    )
)
Physics_Informed_Neural_Network.true_initial_condition_h_values = (
    Physics_Informed_Neural_Network.true_initial_condition_h_function(
        Physics_Informed_Neural_Network.initial_condition_sampling_points[:, 1:2]
    )
)

# compute symbolic function from network output
Physics_Informed_Neural_Network.Physics_Informed_Symbolic_Function()

# compute Mean Squared Error (MSE) terms
Physics_Informed_Neural_Network.MSE_boundary_conditions_function()
Physics_Informed_Neural_Network.MSE_initial_conditions_function()
Physics_Informed_Neural_Network.MSE_symbolic_functions()
Physics_Informed_Neural_Network.total_MSE_function()
Physics_Informed_Neural_Network.MSE_numerical_solution_function()

print("Before Modifying")
print(Physics_Informed_Neural_Network.MSE_boundary_conditions_value)
print(Physics_Informed_Neural_Network.MSE_initial_condition_value)
print(Physics_Informed_Neural_Network.MSE_symbolic_function_value)

MSE_boundary_condition_modified = []
MSE_initial_condition_modified = []
MSE_symbolic_function_modified = []

# irange = np.arange(1, 0.995, -0.0001)
irange = np.arange(0, 20)
irange = 10 ** (-1.0 * irange)

initial_weights = Physics_Informed_Neural_Network.layers[-1].weight
initial_biases = Physics_Informed_Neural_Network.layers[-1].bias

for i in irange:
    # print(Physics_Informed_Neural_Network.layers[-1].weight[0])
    Physics_Informed_Neural_Network.layers[-1].weight = torch.nn.Parameter(i * initial_weights)
    Physics_Informed_Neural_Network.layers[-1].bias = torch.nn.Parameter(i * initial_biases)
    # print(Physics_Informed_Neural_Network.layers[-1].weight[0])

    Physics_Informed_Neural_Network.network_output_lower_boundary_conditions = Physics_Informed_Neural_Network.forward(
        Physics_Informed_Neural_Network.lower_boundary_condition_sampling_points[:, 0:1],
        Physics_Informed_Neural_Network.lower_boundary_condition_sampling_points[:, 1:2],
    )
    Physics_Informed_Neural_Network.network_output_upper_boundary_conditions = Physics_Informed_Neural_Network.forward(
        Physics_Informed_Neural_Network.upper_boundary_condition_sampling_points[:, 0:1],
        Physics_Informed_Neural_Network.upper_boundary_condition_sampling_points[:, 1:2],
    )
    Physics_Informed_Neural_Network.network_output_initial_conditions = Physics_Informed_Neural_Network.forward(
        Physics_Informed_Neural_Network.initial_condition_sampling_points[:, 0:1],
        Physics_Informed_Neural_Network.initial_condition_sampling_points[:, 1:2],
    )

    # compute real initial and boundary values
    Physics_Informed_Neural_Network.true_upper_boundary_condition_u_values = (
        Physics_Informed_Neural_Network.true_upper_boundary_condition_u_function(
            Physics_Informed_Neural_Network.upper_boundary_condition_sampling_points[:, 0:1]
        )
    )
    Physics_Informed_Neural_Network.true_lower_boundary_condition_u_values = (
        Physics_Informed_Neural_Network.true_lower_boundary_condition_u_function(
            Physics_Informed_Neural_Network.lower_boundary_condition_sampling_points[:, 0:1]
        )
    )
    Physics_Informed_Neural_Network.true_initial_condition_u_values = (
        Physics_Informed_Neural_Network.true_initial_condition_u_function(
            Physics_Informed_Neural_Network.initial_condition_sampling_points[:, 1:2]
        )
    )
    Physics_Informed_Neural_Network.true_initial_condition_h_values = (
        Physics_Informed_Neural_Network.true_initial_condition_h_function(
            Physics_Informed_Neural_Network.initial_condition_sampling_points[:, 1:2]
        )
    )

    # compute symbolic function from network output
    Physics_Informed_Neural_Network.Physics_Informed_Symbolic_Function()

    # compute Mean Squared Error (MSE) terms
    Physics_Informed_Neural_Network.MSE_boundary_conditions_function()
    Physics_Informed_Neural_Network.MSE_initial_conditions_function()
    Physics_Informed_Neural_Network.MSE_symbolic_functions()
    Physics_Informed_Neural_Network.total_MSE_function()
    Physics_Informed_Neural_Network.MSE_numerical_solution_function()

    print("Parameters in last layer multiplied by " + str(i))
    print("Boundary Condition Loss: ", Physics_Informed_Neural_Network.MSE_boundary_conditions_value)
    print("Initial Condition Loss: ", Physics_Informed_Neural_Network.MSE_initial_condition_value)
    print("Symbolic Function Loss: ", Physics_Informed_Neural_Network.MSE_symbolic_function_value)

    MSE_boundary_condition_modified.append(
        Physics_Informed_Neural_Network.MSE_boundary_conditions_value.cpu().detach().numpy()
    )
    MSE_initial_condition_modified.append(
        Physics_Informed_Neural_Network.MSE_initial_condition_value.cpu().detach().numpy()
    )
    MSE_symbolic_function_modified.append(
        Physics_Informed_Neural_Network.MSE_symbolic_function_value.cpu().detach().numpy()
    )

plt.plot(irange, MSE_boundary_condition_modified, "o", label="BC")
plt.plot(irange, MSE_initial_condition_modified, "o", label="IC")
plt.plot(irange, MSE_symbolic_function_modified, "o", label="PDE")
plt.xlabel("Factor of Modification")
plt.title("Loss Functions for modified parameters and biases in the last layer")
plt.semilogy()
plt.semilogx()
plt.grid()
plt.legend()
plt.show()
