
# import external libraries
import numpy as np
import torch
from Alternative_LBFGS.LBFGS import FullBatchLBFGS
import hydra
from omegaconf import DictConfig, MissingMandatoryValue, open_dict

# import Internal Libraries
from Animate_SWE import Animate_Solution, Animate_PDE_Losses
from PINN_SWE import PINN
from Plots_SWE import (Plot_Results, Plot_Learning_Curve, Plot_Initial_Conditions, Plot_Boundary_Conditions,
                       Plot_New_Initial_Conditions, Plot_PDE_Losses)

@hydra.main(config_path="config", config_name="default_config")
def my_app(cfg: DictConfig) -> None:

    cfg.vertical_length_scale = (cfg.vertical_scaling_factor * cfg.horizontal_length_scale ** 2. / (
        cfg.time_scale ** 2. * cfg.gravitational_acceleration))  # length scale for vertical lengths and gravity [m]

    cfg.minimum_x = cfg.numerical_solution_x_interval[0]
    cfg.maximum_x = cfg.numerical_solution_x_interval[1]

    if cfg.non_dimensionalization == True:
        # All physical parameters and the physical space is scaled by the chosen scale lengths and time
        cfg.initial_perturbation_amplitude = cfg.initial_perturbation_amplitude / cfg.vertical_length_scale
        cfg.gravitational_acceleration = cfg.gravitational_acceleration * cfg.time_scale ** 2. / cfg.vertical_length_scale
        cfg.average_sea_level = cfg.average_sea_level / cfg.vertical_length_scale
        cfg.momentum_dissipation = 0.0
        cfg.nonlinear_drag_coefficient = 0.0
        cfg.minimum_x = cfg.minimum_x / cfg.horizontal_length_scale
        cfg.maximum_x = cfg.maximum_x / cfg.horizontal_length_scale
        layer_sizes = [2] + cfg.number_of_layers * [cfg.neurons_per_layer] + [2]

    if cfg.train_only_on_first_interval is True:
        model_range = list(range(1))
        cfg.training_steps = cfg.training_steps * cfg.number_of_models
        cfg.symbolic_function_batch_size = cfg.symbolic_function_batch_size * cfg.number_of_models
    else:
        model_range = list(range(cfg.number_of_models))

    for model_number in model_range:
        # adjust time interval to the number of models
        cfg.fraction_of_time_interval = [0.0 + model_number / cfg.number_of_models,
                                         (model_number + 1) / cfg.number_of_models]
        cfg.minimum_time = (cfg.numerical_solution_time_interval[0] + cfg.fraction_of_time_interval[0]
                        * (cfg.numerical_solution_time_interval[1] - cfg.numerical_solution_time_interval[0]))
        cfg.maximum_time = (cfg.numerical_solution_time_interval[0] + cfg.fraction_of_time_interval[1]
                        * (cfg.numerical_solution_time_interval[1] - cfg.numerical_solution_time_interval[0]))  # [s]

        if cfg.non_dimensionalization == True:
            cfg.minimum_time = cfg.minimum_time / cfg.time_scale
            cfg.maximum_time = cfg.maximum_time / cfg.time_scale

        if cfg.optimizer == "FullBatchLBFGS":
            optimizer = FullBatchLBFGS
        else:
            optimizer = FullBatchLBFGS

        if cfg.activation_function == "tanh":
            activation_function = torch.tanh
        else:
            activation_function = torch.tanh

        # Initialization of the Physics Informed Neural Network
        Physics_Informed_Neural_Network = PINN(layer_sizes,
                                               activation_function,
                                               optimizer,
                                               cfg.learning_rate,
                                               cfg.line_search,
                                               cfg.boundary_condition_weight,
                                               cfg.initial_condition_weight,
                                               cfg.symbolic_function_weight,
                                               int(cfg.boundary_condition_batch_size / cfg.number_of_models),
                                               int(cfg.initial_condition_batch_size),
                                               int(cfg.symbolic_function_batch_size / cfg.number_of_models),
                                               int(cfg.training_steps / cfg.number_of_models),
                                               cfg.batch_resampling_period,
                                               cfg.console_output_period,
                                               cfg.device,
                                               cfg.gravitational_acceleration,
                                               cfg.average_sea_level,
                                               cfg.momentum_dissipation,
                                               cfg.nonlinear_drag_coefficient,
                                               cfg.initial_sea_level,
                                               cfg.initial_perturbation_amplitude,
                                               cfg.boundary_conditions,
                                               cfg.non_dimensionalization,
                                               cfg.vertical_length_scale,
                                               cfg.vertical_scaling_factor,
                                               cfg.horizontal_length_scale,
                                               cfg.time_scale,
                                               cfg.minimum_time,
                                               cfg.maximum_time,
                                               cfg.minimum_x,
                                               cfg.maximum_x,
                                               cfg.projected_gradients,
                                               cfg.root_mean_squared_error,
                                               cfg.save_output_over_training,
                                               cfg.save_symbolic_function_over_training,
                                               cfg.numerical_solution_time_interval,
                                               cfg.numerical_solution_time_step,
                                               cfg.numerical_solution_x_interval,
                                               cfg.numerical_solution_space_step,
                                               cfg.fraction_of_time_interval,
                                               model_number,
                                               cfg.number_of_models,
                                               cfg.new_initial_conditions,
                                               cfg.new_initial_condition_sampling_points,
                                               cfg.train_on_solution,
                                               cfg.train_on_PINNs_Loss,
                                               cfg.boundary_condition_transition_function,
                                               cfg.split_networks,
                                               cfg.train_on_boundary_condition_loss,
                                               cfg.train_on_initial_condition_loss,
                                               cfg.momentum_advection).to(cfg.device)

        # Load Model Parameters from previous training (optional -> TrainedParameters must exist)
        if cfg.initial_sea_level == "sine":
            if model_number == 0:
                model_number = 0
                # Physics_Informed_Neural_Network.load_state_dict(
                #    torch.load("1D_Shallow_Water_Equations/TrainedParameters_SWE"))
            else:
                Physics_Informed_Neural_Network.load_state_dict(
                    torch.load("/Users/kubi/Documents/Python/PINNs/Testing-PINNs/1D_Shallow_Water_Equations/Trained_Parameters/TrainedParameters_SWE_" + str(
                        model_number - 1)))

        # Train Model with parameters chosen above -> generate model output and MSE over training
        Physics_Informed_Neural_Network.train_PINN()

        # Save Model Parameters
        torch.save(Physics_Informed_Neural_Network.state_dict(),
                   ("/Users/kubi/Documents/Python/PINNs/Testing-PINNs/1D_Shallow_Water_Equations/Trained_Parameters/TrainedParameters_SWE_" + str(model_number)))

        # Save new initial conditions and respective sampling points
        [new_initial_conditions,
         new_initial_condition_sampling_points] = Physics_Informed_Neural_Network.Save_Final_State()

        # Generate Plots of the learning curve, approximate solution, exact solution and error
        Plot_Learning_Curve(Physics_Informed_Neural_Network)
        Plot_Results(Physics_Informed_Neural_Network)
        Plot_Boundary_Conditions(Physics_Informed_Neural_Network)
        Plot_PDE_Losses(Physics_Informed_Neural_Network)
        if model_number == 0:
            Plot_Initial_Conditions(Physics_Informed_Neural_Network)
        else:
            Plot_New_Initial_Conditions(Physics_Informed_Neural_Network)

        # Generate Animation showing the Development of the Network Output over Training and the respective learning curve
        Animate_PDE_Losses(Physics_Informed_Neural_Network, frames_per_second=10)
        Animate_Solution(Physics_Informed_Neural_Network, frames_per_second=100)

if __name__ == "__main__":
    my_app()