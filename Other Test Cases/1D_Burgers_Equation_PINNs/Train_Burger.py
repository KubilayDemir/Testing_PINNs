# import External Libraries
import torch

# import Internal Libraries
from Animate_Burger import Animate_Results
from PINN_Burger import PINN
from Plots_Burger import Plot_Learning_Curve, Plot_Results

# These weights determine the contribution of each Mean Squared Error (MSE) term to the total MSE
boundary_condition_weight = 1
initial_condition_weight = 1
symbolic_function_weight = 1

# The number of sampling points for computing the MSE terms for BC, IC and within the domain for the symbolic function
boundary_condition_batch_size = 100
initial_condition_batch_size = 100
symbolic_function_batch_size = 10000

# set Training Parameters and Model Architecture
learning_rate = 0.005
optimizer = torch.optim.LBFGS
training_steps = 10
batch_resampling_period = training_steps
training_step_counting_rate = 1
activation_function = torch.tanh
layer_sizes = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# layer_sizes = [2, 50, 50, 50, 50, 50, 50, 50, 50, 1]
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
).to(device)

# Load Model Parameters from previous training (optional -> TrainedParameters must exist)
# Physics_Informed_Neural_Network.load_state_dict(torch.load("TrainedParameters"))
# model.eval()

# Train Model with parameters chosen above -> generate model output and MSE over training
Physics_Informed_Neural_Network.train_PINN()

# Generate Plots of the learning curve, approximate solution, exact solution and error
Plot_Learning_Curve(Physics_Informed_Neural_Network)
Plot_Results(Physics_Informed_Neural_Network)

# Generate Animation showing the Development of the Network Output over Training and the respective learning curve
Animate_Results(Physics_Informed_Neural_Network)

# Save Model Parameters for use as initialization or for generating plots
torch.save(Physics_Informed_Neural_Network.state_dict(), "TrainedParameters")
