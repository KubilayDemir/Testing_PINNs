# external libraries
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from functools import partial
from torch import nn

# internal libraries
from .sampling import (
    FinalStateDataset,
    InitialConditionDataset,
    LeftBoundaryDataset,
    PDEDataset,
    RightBoundaryDataset,
)
from .conditions import (
    initial_zeta,
    initial_u,
    boundary_u_left,
    boundary_u_right,
)
from .optimizers.pcgrad import PCGrad
from .optimizers.lbfgs import FullBatchLBFGS
from .reference import load_reference


def _layer_forward(value, layer, activation):
    value = layer(value)
    return activation(value) if activation is not None else value


def _mixed_activation(activation, value):
    middle = value.shape[1] // 2
    return torch.cat((activation(value[:, :middle]), torch.relu(value[:, middle:])), dim=1)


def _network_forward(layers, value, activation, activate_output=False, checkpointing=False, mixed=False):
    for index, layer in enumerate(layers):
        function = activation if activate_output or index < len(layers) - 1 else None
        if mixed and 0 < index < len(layers) - 1:
            function = partial(_mixed_activation, activation)
        if checkpointing and torch.is_grad_enabled():
            value = checkpoint(_layer_forward, value, layer, function, use_reentrant=False)
        else:
            value = _layer_forward(value, layer, function)
    return value


class PINN(nn.Module):
    """PINN for SWE, heat diffusion or viscous Burgers, with equation-specific physics."""
    def __init__(
            self,
            layer_sizes: list = None,
            activation_function=torch.tanh,
            optimizer=torch.optim.Adam,
            learning_rate=0.0005,
            line_search="strong_wolfe",
            weight_bc=1,
            weight_ic=1,
            weight_pde=1,
            n_bc_points=5000,
            n_ic_points=5000,
            n_pde_points=50000,
            training_steps=1500,
            batch_resampling_period=1,
            console_output_period=1,
            device="cuda",
            gravitational_acceleration=9.81,
            average_sea_level=100.0,
            momentum_dissipation=0,
            nonlinear_drag_coefficient=0,
            initial_perturbation_amplitude=1.0,
            nondimensionalize=True,
            vertical_length_scale=1.0,
            vertical_scaling_factor=1.0,
            horizontal_length_scale=1e6,
            time_scale=5e4,
            t_min=0.0,
            t_max=1.0,
            x_min=-1.0,
            x_max=1.0,
            use_pcgrad=True,
            save_predictions=False,
            save_residuals=True,
            numerical_solution_time_interval=None,
            numerical_solution_time_step=60.0,
            numerical_solution_x_interval=None,
            numerical_solution_space_step=1000.0,
            fraction_of_time_interval=None,
            segment_id=0,
            n_time_segments=1,
            new_initial_conditions=None,
            new_initial_condition_sampling_points=None,
            train_on_solution=False,
            use_pde_loss=True,
            boundary_condition_transition_function=False,
            initial_condition_transition_function=False,
            ic_transition="TF2",
            split_networks=False,
            use_bc_loss=True,
            use_ic_loss=True,
            momentum_advection=False,
            mixed_activation_functions=False,
            sirens_initialization=False,
            adaptive_loss_weights=False,
            minibatch_training=True,
            pde_batch_size=1000,
            bc_batch_size=100,
            ic_batch_size=100,
            iterations_per_epoch=50,
            reference_path=None,
            checkpoint_dir=None,
            snapshot_period=100,
            evaluation_batch_size=8192,
            dataloader_workers=0,
            pin_memory=False,
            resume_period=0,
            stream_history=False,
            batch_diagnostics=False,
            keep_improving_checkpoints=True,
            checkpoint_period=0,
            reference_on_cpu=False,
            equation="swe",
            hidden_layers=None,
            activation="tanh",
            diffusivity=None,
            enforce_bc=False,
            enforce_ic=False,
            activation_checkpointing=False,
    ):

        super().__init__()
        if equation not in ("swe", "heat", "burgers"):
            raise ValueError("equation must be swe, heat or burgers")
        self.equation = equation
        self.activation_checkpointing = activation_checkpointing
        if equation != "swe":
            self.pi = 4 * torch.atan(torch.ones(1)).item() if equation == "burgers" else np.pi
            self.diffusivity = diffusivity if diffusivity is not None else (0.25 if equation == "heat" else 0.01 / self.pi)
            self.enforce_bc, self.enforce_ic = enforce_bc, enforce_ic
            if not hidden_layers:
                raise ValueError("hidden_layers are required for heat and Burgers")
            sizes = [2, *hidden_layers, 1]
            self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])])
            self.activation_function = getattr(torch, activation)
            return

        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        self.snapshot_period = snapshot_period
        self.evaluation_batch_size = evaluation_batch_size
        self.dataloader_workers = dataloader_workers
        self.pin_memory = pin_memory
        self.resume_period = resume_period
        self.stream_history = stream_history
        self.batch_diagnostics = batch_diagnostics
        self.keep_improving_checkpoints = keep_improving_checkpoints
        self.checkpoint_period = checkpoint_period
        self.history_writer = None
        self.ic_transition = ic_transition
        self.reference_device = "cpu" if reference_on_cpu else device
        # initialize Model Architecture using the given Parameters
        print("Initialize Network Architecture.")
        self.minibatch_training = minibatch_training
        self.sirens_initialization = sirens_initialization
        self.adaptive_loss_weights = adaptive_loss_weights
        self.mixed_activation_functions = mixed_activation_functions
        self.split_networks = split_networks
        self.use_bc_loss = use_bc_loss
        self.use_ic_loss = use_ic_loss
        self.improvement_steps = []
        self.best_step = 0
        self.number_of_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.relu = torch.nn.functional.relu

        if self.split_networks is True:
            self.layers_u = []
            self.layers_h = []
            for i in np.arange(self.number_of_layers):
                if i < self.number_of_layers - 1:
                    self.layers_u.append(
                        torch.nn.Linear(
                            self.layer_sizes[i], self.layer_sizes[i + 1]
                        )
                    )
                    self.layers_h.append(
                        torch.nn.Linear(
                            self.layer_sizes[i], self.layer_sizes[i + 1]
                        )
                    )
                else:
                    self.layers_u.append(
                        torch.nn.Linear(self.layer_sizes[i], 1)
                    )
                    self.layers_h.append(
                        torch.nn.Linear(self.layer_sizes[i], 1)
                    )
            self.layers_u = nn.ModuleList(self.layers_u)
            self.layers_h = nn.ModuleList(self.layers_h)
        else:
            self.layers = []
            for i in np.arange(self.number_of_layers):
                self.layers.append(
                    torch.nn.Linear(
                        self.layer_sizes[i], self.layer_sizes[i + 1]
                    )
                )
            self.layers = nn.ModuleList(self.layers)

        if self.sirens_initialization is True:
            c = np.sqrt(6)
            omega_0 = 30
            groups = (self.layers_u, self.layers_h) if self.split_networks else (self.layers,)
            for layers in groups:
                for i in range(self.number_of_layers):
                    input_size = layers[i].weight.shape[0]
                    layers[i].weight = torch.nn.Parameter(
                        2.0
                        * (c / np.sqrt(input_size))
                        * torch.rand(size=layers[i].weight.size())
                        - (c / np.sqrt(input_size))
                    )
                layers[0].weight = torch.nn.Parameter(
                    omega_0 * layers[0].weight
                )

        # initialize true Initial and Boundary Conditions
        print("Initialize true Initial and Boundary Conditions.")
        self.save_predictions = save_predictions
        self.save_residuals = save_residuals
        self.segment_id = segment_id
        self.n_time_segments = n_time_segments

        self.boundary_u_right = boundary_u_right
        self.boundary_u_left = boundary_u_left

        if self.segment_id == 0:
            self.initial_u = initial_u
            self.initial_zeta = initial_zeta(
                initial_perturbation_amplitude,
                nondimensionalize,
                horizontal_length_scale,
            )
        else:
            self.new_initial_conditions = new_initial_conditions
            self.new_initial_condition_sampling_points = (
                new_initial_condition_sampling_points
            )

        # problem parameters
        print("Initialize Problem Parameters.")
        self.nondimensionalize = nondimensionalize
        self.pi = np.pi
        self.momentum_advection = momentum_advection
        self.gravitational_acceleration = gravitational_acceleration
        self.average_sea_level = average_sea_level
        self.momentum_dissipation = momentum_dissipation
        self.nonlinear_drag_coefficient = nonlinear_drag_coefficient
        self.initial_perturbation_amplitude = initial_perturbation_amplitude

        # numerical space and time interval and reference scale lengths and time
        self.vertical_scaling_factor = vertical_scaling_factor
        self.vertical_length_scale = vertical_length_scale
        self.horizontal_length_scale = horizontal_length_scale
        self.time_scale = time_scale
        self.numerical_solution_time_interval = (
            numerical_solution_time_interval
        )
        self.numerical_solution_time_step = numerical_solution_time_step
        self.numerical_solution_x_interval = numerical_solution_x_interval
        self.numerical_solution_space_step = numerical_solution_space_step
        self.fraction_of_time_interval = fraction_of_time_interval

        # Hyper Parameters
        print("Initialize Hyper Parameters.")
        self.boundary_condition_transition_function = (
            boundary_condition_transition_function
        )
        self.initial_condition_transition_function = (
            initial_condition_transition_function
        )
        self.train_on_solution = train_on_solution
        self.use_pde_loss = use_pde_loss
        self.new_sampling_points = False
        self.learning_rate = learning_rate
        self.use_pcgrad = use_pcgrad
        self.line_search = line_search
        if self.use_pcgrad is True:
            self.optimizer = PCGrad(
                torch.optim.Adam(self.parameters(), lr=learning_rate)
            )
        elif optimizer == torch.optim.LBFGS:
            self.optimizer = optimizer(
                self.parameters(),
                lr=self.learning_rate,
                max_iter=20,
                max_eval=None,
                tolerance_grad=1.0 * np.finfo(float).eps,
                tolerance_change=1.0 * np.finfo(float).eps,
                history_size=20,
                line_search_fn=self.line_search,
            )
        elif optimizer == FullBatchLBFGS:
            self.optimizer = optimizer(
                self.parameters(),
                lr=self.learning_rate,
                line_search=self.line_search or "None",
            )
        else:
            self.optimizer = optimizer(
                self.parameters(), lr=self.learning_rate
            )
        self.epochs = training_steps
        self.batch_resampling_period = batch_resampling_period
        self.output_period = console_output_period
        self.weight_bc = weight_bc
        self.weight_ic = weight_ic
        self.weight_pde = weight_pde
        self.n_bc_points = n_bc_points
        self.n_ic_points = n_ic_points
        self.n_pde_points = n_pde_points
        self.pde_batch_size = pde_batch_size
        self.bc_batch_size = bc_batch_size
        self.ic_batch_size = ic_batch_size
        self.iterations_per_epoch = iterations_per_epoch
        self.device = device
        self.reference_path = reference_path

        # define Latin Hypercube Sampling (LHS) methods -> define limits for t in [0,1]
        # and x in [-1,1]
        print("Define Data Sets.")
        self.t_min = t_min
        self.t_max = t_max
        self.x_min = x_min
        self.x_max = x_max
        self.pde_dataset = PDEDataset(
            t_min=self.t_min,
            t_max=self.t_max,
            x_min=self.x_min,
            x_max=self.x_max,
            batch_size=self.n_pde_points,
        )
        self.lbc_dataset = LeftBoundaryDataset(
            t_min=self.t_min,
            t_max=self.t_max,
            x_min=self.x_min,
            batch_size=self.n_bc_points,
        )
        self.ubc_dataset = RightBoundaryDataset(
            t_min=self.t_min,
            t_max=self.t_max,
            x_max=self.x_max,
            batch_size=self.n_bc_points,
        )
        self.ic_dataset = InitialConditionDataset(
            t_min=self.t_min,
            x_min=self.x_min,
            x_max=self.x_max,
            batch_size=self.n_ic_points,
        )
        self.final_state_dataset = FinalStateDataset(
            t_max=self.t_max,
            x_min=self.x_min,
            x_max=self.x_max,
            batch_size=self.n_ic_points,
        )

        # load discrete Representation of the numerical Solution
        print("Create grid and load solution.")
        reference = load_reference(
            reference_path,
            (
                t_min * time_scale if nondimensionalize else t_min,
                t_max * time_scale if nondimensionalize else t_max,
            ),
        )
        self.solution_time_grid = reference.time
        self.u_solution_x_grid = reference.x_u
        self.zeta_solution_x_grid = reference.x_zeta
        self.u_ref = torch.as_tensor(
            reference.u, dtype=torch.float32, device=self.reference_device
        )
        self.zeta_ref = torch.as_tensor(
            reference.zeta, dtype=torch.float32, device=self.reference_device
        )
        [
            self.dimensional_zeta_solution_time_mesh_grid,
            self.dimensional_zeta_solution_x_mesh_grid,
        ] = np.meshgrid(self.solution_time_grid, self.zeta_solution_x_grid)
        [
            self.dimensional_u_solution_time_mesh_grid,
            self.dimensional_u_solution_x_mesh_grid,
        ] = np.meshgrid(self.solution_time_grid, self.u_solution_x_grid)

        if self.nondimensionalize is True:
            self.solution_time_grid = self.solution_time_grid / self.time_scale
            self.zeta_solution_x_grid = (
                    self.zeta_solution_x_grid / self.horizontal_length_scale
            )
            self.u_solution_x_grid = (
                    self.u_solution_x_grid / self.horizontal_length_scale
            )

        [self.zeta_solution_time_mesh_grid, self.zeta_solution_x_mesh_grid] = (
            np.meshgrid(self.solution_time_grid, self.zeta_solution_x_grid)
        )
        [self.u_solution_time_mesh_grid, self.u_solution_x_mesh_grid] = (
            np.meshgrid(self.solution_time_grid, self.u_solution_x_grid)
        )
        self.zeta_solution_mesh_grid_shape = (
            self.zeta_solution_x_mesh_grid.shape
        )
        self.u_solution_mesh_grid_shape = self.u_solution_x_mesh_grid.shape
        self.zeta_solution_time_input_grid = torch.FloatTensor(
            self.zeta_solution_time_mesh_grid.reshape(
                (np.size(self.zeta_solution_time_mesh_grid), 1)
            )
        ).to(self.reference_device)
        self.u_solution_time_input_grid = torch.FloatTensor(
            self.u_solution_time_mesh_grid.reshape(
                (np.size(self.u_solution_time_mesh_grid), 1)
            )
        ).to(self.reference_device)
        self.zeta_solution_x_input_grid = torch.FloatTensor(
            self.zeta_solution_x_mesh_grid.reshape(
                (np.size(self.zeta_solution_x_mesh_grid), 1)
            )
        ).to(self.reference_device)
        self.u_solution_x_input_grid = torch.FloatTensor(
            self.u_solution_x_mesh_grid.reshape(
                (np.size(self.u_solution_x_mesh_grid), 1)
            )
        ).to(self.reference_device)

        # generate grid for tracking model output during training
        print("Create grid for tracking the model output.")
        self.x_grid = np.arange(
            self.x_min,
            self.x_max + (self.x_max - self.x_min) / 100.0,
            (self.x_max - self.x_min) / 100.0,
        )
        self.time_grid = np.arange(
            self.t_min,
            self.t_max + (self.t_max - self.t_min) / 100.0,
            (self.t_max - self.t_min) / 100.0,
        )
        [self.time_mesh_grid, self.x_mesh_grid] = np.meshgrid(
            self.time_grid, self.x_grid
        )
        self.time_input_grid = torch.FloatTensor(
            self.time_mesh_grid.reshape(np.size(self.time_mesh_grid), 1)
        ).to(self.device)
        self.x_input_grid = torch.FloatTensor(
            self.x_mesh_grid.reshape(np.size(self.x_mesh_grid), 1)
        ).to(self.device)
        self.mesh_grid_shape = self.time_mesh_grid.shape

        if self.nondimensionalize is True:
            [self.dimensional_time_mesh_grid, self.dimensional_x_mesh_grid] = (
                np.meshgrid(
                    self.time_grid * self.time_scale,
                    self.x_grid * self.horizontal_length_scale,
                )
            )

    def forward(self, t, x):
        if self.equation != "swe":
            output = _network_forward(self.layers, torch.cat((t, x), dim=1), self.activation_function,
                                      activate_output=True, checkpointing=self.activation_checkpointing)
            if self.enforce_bc:
                output = (1 - torch.exp((x - 1) / 0.1)) * (1 - torch.exp((-x - 1) / 0.1)) * output
            if self.enforce_ic:
                decay = torch.exp(-t / 0.1)
                output = self.initial_condition(x) * decay + output * (1 - decay)
            return output

        tx = torch.cat((t, x), dim=1)

        if self.split_networks is True:
            output_u = _network_forward(self.layers_u, tx, self.activation_function,
                                        checkpointing=self.activation_checkpointing)
            output_zeta = _network_forward(self.layers_h, tx, self.activation_function,
                                           checkpointing=self.activation_checkpointing)
        else:
            output = _network_forward(self.layers, tx, self.activation_function,
                                      checkpointing=self.activation_checkpointing,
                                      mixed=self.mixed_activation_functions)

            output_u = output[:, 0:1]
            output_zeta = output[:, 1:2]

        if self.boundary_condition_transition_function is True:
            tau_x = 0.1 * self.x_max
            output_u = (
                    (1.0 - torch.exp((x - self.x_max) / tau_x))
                    * (1.0 - torch.exp((-x - self.x_max) / tau_x))
                    * output_u
            )

        if self.initial_condition_transition_function is True:
            tau_t = 0.1 * self.t_max
            output_u = output_u * (1.0 - torch.exp(-t / tau_t))
            if self.ic_transition == "TF2":
                output_zeta = self.initial_zeta(x) * torch.exp(-t / tau_t) + output_zeta * (
                        1.0 - torch.exp(-t / tau_t)
                )
            else:
                output_zeta = self.initial_zeta(x) + output_zeta * (1.0 - torch.exp(-t / tau_t))

        output = torch.cat((output_u, output_zeta), dim=1)

        return output

    def initial_condition(self, x):
        if self.equation == "swe":
            return torch.cat((self.initial_u(x), self.initial_zeta(x)), dim=1)
        return (1 if self.equation == "heat" else -1) * torch.sin(np.pi * x)

    def pde_residual(self, points):
        from .physics import benchmark_residual, swe_residuals
        if self.equation == "swe":
            return swe_residuals(self, points[:, :1], points[:, 1:2])
        return benchmark_residual(self, points)

    def losses(self, batches):
        """PDE and condition losses for the heat and Burgers benchmark workflow."""
        if self.equation == "swe":
            raise ValueError("SWE condition targets and segmented losses use closure()")
        pde, left, right, initial = batches
        residual = self.pde_residual(pde)
        eleft = self(left[:, :1], left[:, 1:2])
        eright = self(right[:, :1], right[:, 1:2])
        einitial = self(initial[:, :1], initial[:, 1:2]) - self.initial_condition(initial[:, 1:2])
        return {"loss_pde": (residual ** 2 / len(residual)).sum(),
                "loss_bc": (eleft ** 2 / len(eleft) + eright ** 2 / len(eright)).sum(),
                "loss_ic": (einitial ** 2 / len(einitial)).sum()}

    def compute_pde_residuals(self):
        from .physics import swe_residuals

        self.momentum_residual, self.continuity_residual = swe_residuals(
            self, self.pde_points[:, :1], self.pde_points[:, 1:2]
        )

    def compute_bc_loss(self):

        self.loss_bc_left = (
                (
                        self.left_boundary_prediction[:, 0:1]
                        - self.left_boundary_target
                )
                ** 2
        ).mean()
        self.loss_bc_right = (
                (
                        self.right_boundary_prediction[:, 0:1]
                        - self.right_boundary_target
                )
                ** 2
        ).mean()
        self.loss_bc = self.loss_bc_left + self.loss_bc_right

    def compute_ic_loss(self):
        self.loss_ic_u = (
                (
                        self.network_output_initial_conditions[:, 0:1]
                        - self.initial_velocity_target
                )
                ** 2
        ).mean()
        self.loss_ic_zeta = (
                (
                        self.network_output_initial_conditions[:, 1:2]
                        - self.initial_elevation_target
                )
                ** 2
        ).mean()
        self.loss_ic = self.loss_ic_u + self.loss_ic_zeta

    def compute_pde_loss(self):
        self.loss_momentum = (self.momentum_residual ** 2).mean()
        self.loss_continuity = (self.continuity_residual ** 2).mean()
        self.loss_pde = self.loss_momentum + self.loss_continuity

    def compute_total_loss(self):
        self.loss_total = 0.0

        if self.adaptive_loss_weights is True:
            alpha = 0.9  # weight between new and old weight for update

            layers = list(self.layers_u) + list(self.layers_h) if self.split_networks else self.layers
            number_of_layers = len(layers)

            self.optimizer.zero_grad()
            self.loss_pde.backward(retain_graph=True)
            gradient_tensor = layers[0].weight.grad.flatten()
            for k in range(1, number_of_layers):
                gradient_tensor = torch.cat(
                    (gradient_tensor, layers[k].weight.grad.flatten())
                )
                gradient_tensor = torch.cat(
                    (gradient_tensor, layers[k].bias.grad.flatten())
                )
            max_pde_grad = torch.max(abs(gradient_tensor))

            self.optimizer.zero_grad()
            self.loss_ic.backward(retain_graph=True)
            gradient_tensor = layers[0].weight.grad.flatten()
            for k in range(1, number_of_layers):
                gradient_tensor = torch.cat(
                    (gradient_tensor, layers[k].weight.grad.flatten())
                )
                gradient_tensor = torch.cat(
                    (gradient_tensor, layers[k].bias.grad.flatten())
                )
            mean_initial_condition_grad = torch.mean(abs(gradient_tensor))

            self.optimizer.zero_grad()
            self.loss_bc.backward(retain_graph=True)
            gradient_tensor = layers[0].weight.grad.flatten()
            for k in range(1, number_of_layers):
                gradient_tensor = torch.cat(
                    (gradient_tensor, layers[k].weight.grad.flatten())
                )
                gradient_tensor = torch.cat(
                    (gradient_tensor, layers[k].bias.grad.flatten())
                )
            mean_boundary_condition_grad = torch.mean(abs(gradient_tensor))
            self.optimizer.zero_grad()

            eps = torch.finfo(gradient_tensor.dtype).eps
            new_initial_condition_weight = (
                    max_pde_grad
                    / mean_initial_condition_grad.clamp_min(eps)
            ).detach()
            self.weight_ic = (
                                     1 - alpha
                             ) * self.weight_ic + alpha * new_initial_condition_weight

            new_boundary_condition_weight = (
                    max_pde_grad
                    / mean_boundary_condition_grad.clamp_min(eps)
            ).detach()
            self.weight_bc = (
                                     1 - alpha
                             ) * self.weight_bc + alpha * new_boundary_condition_weight

        # MSE containing the error at the initial and boundary conditions and the PDE
        # loss term
        if self.use_pde_loss is True:
            self.loss_total += self.weight_pde * self.loss_pde

        if (
                self.initial_condition_transition_function is False
                and self.use_ic_loss is True
        ):
            self.loss_total += self.weight_ic * self.loss_ic

        if (
                self.boundary_condition_transition_function is False
                and self.use_bc_loss is True
        ):
            self.loss_total += self.weight_bc * self.loss_bc

        # MSE comparing the network output to the numerical solution
        if self.train_on_solution is True:
            self.loss_total += self.mse_ref_zeta + self.mse_ref_u

    def compute_reference_error(self):
        # Preserve autograd for reference-supervised training; scoring callers use
        # no_grad.
        def predict(t, x, column):
            size = self.evaluation_batch_size
            return torch.cat(
                [
                    self.forward(
                        t[start: start + size].to(self.device),
                        x[start: start + size].to(self.device)
                    )[:, column: column + 1].to(self.reference_device)
                    for start in range(0, len(t), size)
                ]
            )

        self.network_output_u = predict(
            self.u_solution_time_input_grid, self.u_solution_x_input_grid, 0
        )
        self.network_output_zeta = predict(
            self.zeta_solution_time_input_grid,
            self.zeta_solution_x_input_grid,
            1,
        )

        if self.nondimensionalize is True:
            self.network_output_u = (
                    self.network_output_u
                    * self.horizontal_length_scale
                    / self.time_scale
            )
            self.network_output_zeta = (
                    self.network_output_zeta * self.vertical_length_scale
            )

        self.mse_ref_zeta = (
                (
                        self.zeta_ref
                        - self.network_output_zeta.reshape(
                    self.zeta_solution_mesh_grid_shape
                )
                )
                ** 2
        ).mean()
        self.mse_ref_u = (
                (
                        self.u_ref
                        - self.network_output_u.reshape(
                    self.u_solution_mesh_grid_shape
                )
                )
                ** 2
        ).mean()

    # reset datasets and re-sample collocation points
    def resample_sampling_points(self):
        self.pde_dataset = PDEDataset(
            t_min=self.t_min,
            t_max=self.t_max,
            x_min=self.x_min,
            x_max=self.x_max,
            batch_size=self.n_pde_points,
        )
        self.lbc_dataset = LeftBoundaryDataset(
            t_min=self.t_min,
            t_max=self.t_max,
            x_min=self.x_min,
            batch_size=self.n_bc_points,
        )
        self.ubc_dataset = RightBoundaryDataset(
            t_min=self.t_min,
            t_max=self.t_max,
            x_max=self.x_max,
            batch_size=self.n_bc_points,
        )
        self.ic_dataset = InitialConditionDataset(
            t_min=self.t_min,
            x_min=self.x_min,
            x_max=self.x_max,
            batch_size=self.n_ic_points,
        )
        self.final_state_dataset = FinalStateDataset(
            t_max=self.t_max,
            x_min=self.x_min,
            x_max=self.x_max,
            batch_size=self.n_ic_points,
        )
        self.new_sampling_points = True

    def closure(self, backward=True):
        self.optimizer.zero_grad()

        # compute network output for initial and boundary values
        self.left_boundary_prediction = self.forward(
            self.left_boundary_points[:, 0:1],
            self.left_boundary_points[:, 1:2],
        )
        self.right_boundary_prediction = self.forward(
            self.right_boundary_points[:, 0:1],
            self.right_boundary_points[:, 1:2],
        )

        if self.segment_id == 0:
            self.network_output_initial_conditions = self.forward(
                self.initial_condition_sampling_points[:, 0:1],
                self.initial_condition_sampling_points[:, 1:2],
            )
        else:
            self.network_output_initial_conditions = self.forward(
                self.new_initial_condition_sampling_points[:, 0:1],
                self.new_initial_condition_sampling_points[:, 1:2],
            )

        self.right_boundary_target = self.boundary_u_right(
            self.right_boundary_points[:, 0:1]
        )
        self.left_boundary_target = self.boundary_u_left(
            self.left_boundary_points[:, 0:1]
        )

        if self.segment_id == 0:
            self.initial_velocity_target = self.initial_u(
                self.initial_condition_sampling_points[:, 1:2]
            )
            self.initial_elevation_target = self.initial_zeta(
                self.initial_condition_sampling_points[:, 1:2]
            )

        else:
            self.initial_velocity_target = self.new_initial_conditions[
                :, 0:1
            ]
            self.initial_elevation_target = self.new_initial_conditions[
                :, 1:2
            ]

        self.new_sampling_points = False

        # compute PDE residual from network output
        self.compute_pde_residuals()

        # compute Mean Squared Error (MSE) terms
        if self.train_on_solution is True:
            self.compute_reference_error()
        self.compute_bc_loss()
        self.compute_ic_loss()
        self.compute_pde_loss()
        self.compute_total_loss()

        # compute gradients
        if backward and (
                self.use_pcgrad is False or self.train_on_solution is True
        ):
            self.loss_total.backward(
                retain_graph=False
            )

        return self.loss_total

    def save_loss_history(self):
        self.loss_bc_history.append(self.loss_bc.cpu().detach().numpy())
        self.loss_ic_history.append(self.loss_ic.cpu().detach().numpy())
        self.loss_pde_history.append(self.loss_pde.cpu().detach().numpy())
        self.loss_total_history.append(self.loss_total.cpu().detach().numpy())
        self.mse_ref_zeta_history.append(self.mse_ref_zeta.cpu().detach().numpy())
        self.mse_ref_u_history.append(self.mse_ref_u.cpu().detach().numpy())
        self.loss_ic_u_history.append(self.loss_ic_u.cpu().detach().numpy())
        self.loss_ic_zeta_history.append(self.loss_ic_zeta.cpu().detach().numpy())
        self.loss_momentum_history.append(self.loss_momentum.cpu().detach().numpy())
        self.loss_continuity_history.append(self.loss_continuity.cpu().detach().numpy())
        self.loss_bc_left_history.append(
            self.loss_bc_left.cpu().detach().numpy()
        )
        self.loss_bc_right_history.append(
            self.loss_bc_right.cpu().detach().numpy()
        )

    def save_prediction_snapshot(self):
        if self.batch_diagnostics:
            from .diagnostics import prediction_snapshot
            prediction_snapshot(self)
            return
        self.network_output = self.forward(
            self.time_input_grid, self.x_input_grid
        )
        self.velocity_history.append(
            self.network_output[:, 0:1]
            .cpu()
            .detach()
            .numpy()
            .reshape(self.mesh_grid_shape)
        )
        self.elevation_history.append(
            self.network_output[:, 1:2]
            .cpu()
            .detach()
            .numpy()
            .reshape(self.mesh_grid_shape)
        )

    def save_residual_snapshot(self):
        if self.batch_diagnostics:
            from .diagnostics import residual_snapshot
            residual_snapshot(self)
            return
        sampling_points = self.pde_points
        self.pde_points = torch.hstack(
            (
                self.time_input_grid.clone().detach().requires_grad_(True),
                self.x_input_grid.clone().detach().requires_grad_(True),
            )
        )
        self.compute_pde_residuals()
        momentum = self.momentum_residual.reshape(shape=self.mesh_grid_shape)
        continuity = self.continuity_residual.reshape(shape=self.mesh_grid_shape)
        self.momentum_residual_history.append(
            momentum.cpu().detach().numpy().reshape(self.mesh_grid_shape)
        )
        self.continuity_residual_history.append(
            continuity.cpu().detach().numpy().reshape(self.mesh_grid_shape)
        )
        self.pde_points = sampling_points

    def save_weight_history(self):
        self.pde_weight_history.append(self.weight_pde)
        self.ic_weight_history.append(
            float(self.weight_ic)
        )
        self.bc_weight_history.append(
            float(self.weight_bc)
        )

    def final_state(self):
        self.final_state_sampling_points = self.final_state_dataset.tx.to(
            self.device
        )
        new_initial_conditions = self.forward(
            self.final_state_sampling_points[:, 0:1],
            self.final_state_sampling_points[:, 1:2],
        )
        final_state_sampling_points = (
            self.final_state_sampling_points.cpu().detach().numpy()
        )
        return (
            new_initial_conditions.cpu().detach().numpy(),
            final_state_sampling_points,
        )

    def _use_full_pool(self):
        self.pde_points = self.pde_dataset.tx.to(self.device)
        self.left_boundary_points = self.lbc_dataset.tx.to(
            self.device
        )
        self.right_boundary_points = self.ubc_dataset.tx.to(
            self.device
        )
        self.initial_condition_sampling_points = self.ic_dataset.tx.to(
            self.device
        )

    def _save_best(self, epoch, score):
        from .training_checkpoints import save_torch

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # Keep every improving state dict; best.pt is a convenient latest-best copy.
            state = self.state_dict()
            if self.keep_improving_checkpoints:
                save_torch(
                    state, self.checkpoint_dir
                           / f"best_epoch_{epoch:06d}.pt",
                )
            save_torch(state, self.checkpoint_dir / "best.pt")
        self.improvement_steps.append(epoch)
        self.best_step = epoch
        self.best_score = score

    def fit(self, resume=None):
        """Train using the explicit loop in training_loop.py."""
        from .training_loop import fit
        return fit(self, resume)

    def _record_snapshot(self, epoch):
        if self.save_predictions:
            with torch.no_grad():
                self.save_prediction_snapshot()
        if self.save_residuals:
            self.save_residual_snapshot()
        if self.save_predictions or self.save_residuals:
            self.snapshot_epochs.append(epoch)
            if self.history_writer is not None:
                self.history_writer.snapshot(self, epoch)

    def _record_epoch(self, epoch):
        if not self.batch_diagnostics:
            self._use_full_pool()
        # Reporting must not update adaptive weights or leave parameter gradients
        # behind.
        adaptive = self.adaptive_loss_weights
        self.adaptive_loss_weights = False
        try:
            if self.batch_diagnostics:
                from .diagnostics import report_losses
                report_losses(self)
            else:
                self.closure(backward=False)
        finally:
            self.adaptive_loss_weights = adaptive
        with torch.no_grad():
            self.compute_reference_error()
        if not torch.isfinite(self.loss_total) or not torch.isfinite(
                self.mse_ref_u + self.mse_ref_zeta
        ):
            raise FloatingPointError(
                f"Non-finite loss or reference error at epoch {epoch}"
            )
        self.save_loss_history()
        self.history_epochs.append(epoch)
        if self.history_writer is not None:
            self.history_writer.record(self, epoch)
        self.optimizer.zero_grad()
        if epoch == 0 or epoch % self.snapshot_period == 0:
            self._record_snapshot(epoch)
