# import External Libraries
import matplotlib.pyplot as plt
import numpy as np
import torch

# import Internal Libraries
from PINN_SWE import PINN
from Plots_SWE import Plot_Learning_Curve, Plot_Results

# Problem Parameters
nondimensionalization = False  # Activation of Nondimensionalization

if nondimensionalization == True:
    # reference scales
    dimensional_average_sea_level = 100.0  # [m]
    dimensional_gravitational_acceleration = 9.81  # [ms^-1]
    vertical_length_scale = 1.0  # [m]
    horizontal_length_scale = 1000000.0  # 1e+6 # [m]
    time_scale = (
        2.0 * horizontal_length_scale / np.sqrt(dimensional_gravitational_acceleration * dimensional_average_sea_level)
    )  # [s]
    # dimensionless parameters
    gravitational_acceleration = (
        dimensional_gravitational_acceleration * time_scale ** 2 / vertical_length_scale
    )  # dimensionless
    average_sea_level = 100.0 / vertical_length_scale  # dimensionless
    momentum_dissipation = 0.0  # dimensionless
    nonlinear_drag_coefficient = 0.0  # dimensionless
else:
    # deactivated
    gravitational_acceleration = 9.81  # [ms^-2]
    average_sea_level = 1.0  # [m]
    momentum_dissipation = 0.0
    nonlinear_drag_coefficient = 0.0
    vertical_length_scale = 1.0  # [m]
    horizontal_length_scale = 1.0  # [m]
    time_scale = 1.0  # [s]

# dimensionless space
minimum_time = 0.0
maximum_time = 1.0
minimum_x = -1.0
maximum_x = 1.0

# These weights determine the contribution of each Mean Squared Error (MSE) term to the total MSE
boundary_condition_weight = 1
initial_condition_weight = 1
symbolic_function_weight = 1

# The number of sampling points for computing the MSE terms for BC, IC and within the domain for the symbolic function
boundary_condition_batch_size = 1000
initial_condition_batch_size = 1000
symbolic_function_batch_size = 50000

# set Training Parameters and Network Architecture
learning_rate = 1  # for LBFGS the learning rate is automatically set to 1
training_steps = 50
optimizer = torch.optim.LBFGS
batch_resampling_period = training_steps
training_step_counting_rate = 1
activation_function = torch.tanh
layer_sizes = [2, 50, 2]
# layer_sizes = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 2]
# layer_sizes = [2, 100, 100, 100, 100, 100, 100, 100, 100, 100, 2]
device = "cuda"

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
# Physics_Informed_Neural_Network.load_state_dict(torch.load("TrainedParameters_SWE"))
# Physics_Informed_Neural_Network.eval()

# Train Model with parameters chosen above -> generate model output and MSE over training
Physics_Informed_Neural_Network.train_PINN()
# Plot Weights and Biases

# input layer
fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[0].weight[:, 0].cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[0].weight[:, 0].cpu().detach().numpy(),
)
plt.title("Weights for first input in the input layer")
plt.show()

fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[0].weight[:, 1].cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[0].weight[:, 1].cpu().detach().numpy(),
)
plt.title("Weights for second input in the input layer")
plt.show()

fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[0].bias.cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[0].bias.cpu().detach().numpy(),
)
plt.title("Biases for the input layer")
plt.show()

# hidden layer
fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[1].weight[0, :].cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[1].weight[0, :].cpu().detach().numpy(),
)
plt.title("Weights in the hidden layer for first output")
plt.show()

fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[1].weight[1, :].cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[1].weight[1, :].cpu().detach().numpy(),
)
plt.title("Weights in the hidden layer for second output")
plt.show()

fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[1].bias.cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[1].bias.cpu().detach().numpy(),
)
plt.title("Biases in the hidden layer")
plt.show()

# plot gradients for input and output weights

loss = Physics_Informed_Neural_Network.closure()

# first layer
fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[0].weight.grad[:, 0].cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[0].weight.grad[:, 0].cpu().detach().numpy(),
)
plt.title("Gradients for first input in the input layer")
plt.show()

fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[0].weight.grad[:, 1].cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[0].weight.grad[:, 1].cpu().detach().numpy(),
)
plt.title("Gradients for second input in the input layer")
plt.show()

fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[0].bias.grad.cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[0].bias.grad.cpu().detach().numpy(),
)
plt.title("Gradients of Biases for the input layer")
plt.show()

# second layer
fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[1].weight.grad[0, :].cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[1].weight.grad[0, :].cpu().detach().numpy(),
)
plt.title("Gradients for first output in the output layer")
plt.show()

fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[1].weight.grad[1, :].cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[1].weight.grad[1, :].cpu().detach().numpy(),
)
plt.title("Gradients for second output in the output layer")
plt.show()

fig = plt.figure()
plt.plot(
    np.arange(np.size(Physics_Informed_Neural_Network.layers[1].bias.grad.cpu().detach().numpy())),
    Physics_Informed_Neural_Network.layers[1].bias.grad.cpu().detach().numpy(),
)
plt.title("Gradients of Biases for the output layer")
plt.show()

# Generate Plots of the learning curve, approximate solution, exact solution and error
Plot_Learning_Curve(Physics_Informed_Neural_Network)
Plot_Results(Physics_Informed_Neural_Network)
