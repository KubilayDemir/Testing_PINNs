# import External Libraries
import numpy as np
import torch

# import Internal Libraries
from Initial_And_Boundary_Conditions import (
    true_initial_condition_function,
    true_lower_boundary_condition_function,
    true_upper_boundary_condition_function,
)
from scipy import io
from smt.sampling_methods import LHS
from torch import nn
from torch.autograd import grad


# Define class of Physics Informed Neural Networks for the Burgers Equation
class PINN(nn.Module):
    def __init__(
        self,
        layer_sizes=[2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1],
        activation_function=torch.tanh,
        optimizer=torch.optim.Adam,
        learning_rate=0.005,
        boundary_condition_weight=1,
        initial_condition_weight=1,
        symbolic_function_weight=1,
        boundary_condition_batch_size=100,
        initial_condition_batch_size=100,
        symbolic_function_batch_size=10000,
        training_steps=1500,
        batch_resampling_period=1,
        training_step_counting_rate=1,
        device="cuda",
    ):
        super().__init__()

        # initialize Model Architecture using the given Parameters
        print("Initialize Network.")
        self.number_of_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        self.layers = []
        self.activation_function = activation_function
        for i in np.arange(self.number_of_layers):
            self.layers.append(torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
        self.layers = nn.ModuleList(self.layers)

        # initialize true Initial and Boundary Conditions
        print("Initialize true Initial and Boundary Conditions.")
        self.true_initial_condition_function = true_initial_condition_function
        self.true_upper_boundary_condition_function = true_upper_boundary_condition_function
        self.true_lower_boundary_condition_function = true_lower_boundary_condition_function

        # initialize Problem Parameters as defined
        print("Initialize Problem Parameters.")
        self.pi = 4 * torch.atan(torch.ones(1)).item()

        # initialize Training Parameters
        print("Initialize Training Parameters.")
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.parameters(), lr=self.learning_rate)
        self.training_steps = training_steps
        self.batch_resampling_period = batch_resampling_period
        self.training_step_counting_rate = training_step_counting_rate
        self.boundary_condition_weight = boundary_condition_weight
        self.initial_condition_weight = initial_condition_weight
        self.symbolic_function_weight = symbolic_function_weight
        self.boundary_condition_batch_size = boundary_condition_batch_size
        self.initial_condition_batch_size = initial_condition_batch_size
        self.symbolic_function_batch_size = symbolic_function_batch_size
        self.device = device

        # define Latin Hypercube Sampling (LHS) methods -> define limits for t in [0,1] and x in [-1,1]
        print("Define Latin Hypercube Sampling (LHS) methods.")
        self.lower_boundary_sampling_method = LHS(xlimits=np.array([[0.0, 1.0], [-1.0, -1.0]]))
        self.upper_boundary_sampling_method = LHS(xlimits=np.array([[0.0, 1.0], [1.0, 1.0]]))
        self.initial_condition_sampling_method = LHS(xlimits=np.array([[0.0, 0.0], [-1.0, 1.0]]))
        self.symbolic_function_sampling_method = LHS(xlimits=np.array([[0.0, 1.0], [-1.0, 1.0]]))

        # load Grid, its Shape and the discrete Representation of the Solution (from PINNs Part I Paper)
        print("Load grid and solution.")
        self.imported_solution_and_grid = io.loadmat("burgers_shock.mat")
        self.exact_solution_values = torch.FloatTensor(self.imported_solution_and_grid["usol"]).to(self.device)
        self.time_grid = self.imported_solution_and_grid["t"]
        self.x_grid = self.imported_solution_and_grid["x"]
        [self.time_mesh_grid, self.x_mesh_grid] = np.meshgrid(self.time_grid, self.x_grid)
        self.mesh_grid_shape = self.time_mesh_grid.shape
        self.time_input_grid = torch.FloatTensor(self.time_mesh_grid.reshape(np.size(self.time_mesh_grid), 1)).to(
            self.device
        )
        self.x_input_grid = torch.FloatTensor(self.x_mesh_grid.reshape(np.size(self.x_mesh_grid), 1)).to(self.device)

    def forward(self, t, x):
        input = torch.cat((t, x), dim=1)
        output = self.activation_function(self.layers[0](input))
        for i in np.arange(1, self.number_of_layers):
            output = self.activation_function(self.layers[i](output))
        return output

    def Physics_Informed_Symbolic_Function(self):
        t = self.symbolic_function_sampling_points[:, 0:1]
        x = self.symbolic_function_sampling_points[:, 1:2]
        u = self.forward(t, x)
        u_t = grad(u.sum(), t, create_graph=True)[0].to(self.device)
        u_x = grad(u.sum(), x, create_graph=True)[0].to(self.device)
        u_xx = grad(u_x.sum(), x, create_graph=True)[0].to(self.device)
        self.symbolic_function_values = u_t + u * u_x - (0.01 / self.pi) * u_xx
        return

    def MSE_boundary_conditions_function(self):
        self.MSE_boundary_conditions_value = (
            (self.network_output_lower_boundary_conditions - self.true_lower_boundary_condition_values) ** 2
            / len(self.network_output_lower_boundary_conditions)
            + (self.network_output_upper_boundary_conditions - self.true_upper_boundary_condition_values) ** 2
            / len(self.network_output_upper_boundary_conditions)
        ).sum()
        return

    def MSE_initial_conditions_function(self):
        self.MSE_initial_condition_value = (
            (self.network_output_initial_conditions - self.true_initial_condition_values) ** 2
            / len(self.network_output_initial_conditions)
        ).sum()
        return

    def MSE_symbolic_functions(self):
        self.MSE_symbolic_function_value = (
            self.symbolic_function_values ** 2 / len(self.symbolic_function_values)
        ).sum()
        return

    def total_MSE_function(self):
        self.total_MSE_value = (
            self.boundary_condition_weight * self.MSE_boundary_conditions_value
            + self.initial_condition_weight * self.MSE_initial_condition_value
            + self.symbolic_function_weight * self.MSE_symbolic_function_value
        )
        return

    def Relative_L2_Error(self):
        network_output = self.forward(self.time_input_grid, self.x_input_grid).reshape(self.mesh_grid_shape)
        self.Relative_L2_Error_value = torch.linalg.norm(
            self.exact_solution_values - network_output, 2
        ) / torch.linalg.norm(self.exact_solution_values, 2)
        return

    # generate Random Samples using Latin Hypercube Sampling in the Domain: t in [0,1], x in [-1,1]
    def resample_sampling_points(self):
        # samples for t in [0,1], x = -1
        self.lower_boundary_condition_sampling_points = (
            torch.FloatTensor(self.lower_boundary_sampling_method(self.boundary_condition_batch_size))
            .requires_grad_(True)
            .to(self.device)
        )
        # samples for t in [0,1], x = 1
        self.upper_boundary_condition_sampling_points = (
            torch.FloatTensor(self.upper_boundary_sampling_method(self.boundary_condition_batch_size))
            .requires_grad_(True)
            .to(self.device)
        )
        # samples for t = 0, x  in [-1,1]
        self.initial_condition_sampling_points = (
            torch.FloatTensor(self.initial_condition_sampling_method(self.initial_condition_batch_size))
            .requires_grad_(True)
            .to(self.device)
        )
        # samples for t in [0,1], x in [-1,1]
        self.symbolic_function_sampling_points = (
            torch.FloatTensor(self.symbolic_function_sampling_method(self.symbolic_function_batch_size))
            .requires_grad_(True)
            .to(self.device)
        )
        return

    def closure(self):
        self.optimizer.zero_grad()

        # compute network output for Initial and Boundary Conditions for Model Evaluation
        self.network_output_lower_boundary_conditions = self.forward(
            self.lower_boundary_condition_sampling_points[:, 0:1], self.lower_boundary_condition_sampling_points[:, 1:2]
        )
        self.network_output_upper_boundary_conditions = self.forward(
            self.upper_boundary_condition_sampling_points[:, 0:1], self.upper_boundary_condition_sampling_points[:, 1:2]
        )
        self.network_output_initial_conditions = self.forward(
            self.initial_condition_sampling_points[:, 0:1], self.initial_condition_sampling_points[:, 1:2]
        )

        # compute real Initial and Boundary Values for Model Evaluation
        self.true_upper_boundary_condition_values = self.true_upper_boundary_condition_function(
            self.upper_boundary_condition_sampling_points[:, 0:1]
        )
        self.true_lower_boundary_condition_values = self.true_lower_boundary_condition_function(
            self.lower_boundary_condition_sampling_points[:, 0:1]
        )
        self.true_initial_condition_values = self.true_initial_condition_function(
            self.initial_condition_sampling_points[:, 1:2]
        )

        # compute Symbolic Function from Network Output for Model Evaluation
        self.Physics_Informed_Symbolic_Function()

        # compute individual Mean Squared Error (MSE) terms and total MSE
        self.MSE_boundary_conditions_function()
        self.MSE_initial_conditions_function()
        self.MSE_symbolic_functions()
        self.Relative_L2_Error()
        self.total_MSE_function()

        # compute gradients
        self.total_MSE_value.backward()
        return self.total_MSE_value

    def Save_MSE(self):
        self.MSE_boundary_conditions_over_training.append(self.MSE_boundary_conditions_value.cpu().detach().numpy())
        self.MSE_initial_conditions_over_training.append(self.MSE_initial_condition_value.cpu().detach().numpy())
        self.MSE_symbolic_function_over_training.append(self.MSE_symbolic_function_value.cpu().detach().numpy())
        self.total_MSE_over_training.append(self.total_MSE_value.cpu().detach().numpy())
        self.Relative_L2_Error_over_training.append(self.Relative_L2_Error_value.cpu().detach().numpy())
        return

    def Save_Network_Output(self):
        self.network_output = self.forward(self.time_input_grid, self.x_input_grid)
        self.network_output_over_training.append(
            self.network_output.cpu().detach().numpy().reshape(self.mesh_grid_shape)
        )

    def train_PINN(self):

        # track individual Mean Squared Error (MSE) terms over training
        self.MSE_boundary_conditions_over_training = []
        self.MSE_initial_conditions_over_training = []
        self.MSE_symbolic_function_over_training = []
        self.Relative_L2_Error_over_training = []
        self.total_MSE_over_training = []
        self.network_output_over_training = []

        # evaluate Model before Training
        print("Choose Sampling Points.")
        self.resample_sampling_points()
        print("Evaluate Initial Network Output.")
        self.closure()
        print("Save Initial Network Output and Losses.")
        self.Save_Network_Output()
        self.Save_MSE()

        print("Begin training:")
        for i in range(self.training_steps):

            # Latin Hypercube Sampling
            if (i / self.batch_resampling_period) == int(i / self.batch_resampling_period) and i > 0:
                self.resample_sampling_points()
                print("Change Sampling Points")
            # Model Evaluation, Backward Propagation and Training Step
            self.optimizer.step(self.closure)
            # save MSE Terms over Training Steps
            self.Save_MSE()
            # save Model Output at current training step to track its development over training
            self.Save_Network_Output()
            # print Number of Training Step to track Progress of the Run
            if (i + 1) / self.training_step_counting_rate == int((i + 1) / self.training_step_counting_rate):
                print("Training Step no.: " + str(i + 1) + "/" + str(self.training_steps))

        print("Training completed.")
        return
