# import External Libraries
import matplotlib.pyplot as plt
import torch
from matplotlib import cm

# import seaborn as sns

# Plot_learning_curve generates a plot showing all Mean Squared Error (MSE) terms over training.
# Additionally, a plot showing individual MSE terms for u and h is generated.


def Plot_Learning_Curve(Physics_Informed_Neural_Network):
    # sns.set()
    plt.figure()
    training_steps = range(Physics_Informed_Neural_Network.epochs + 1)
    plt.plot(training_steps, Physics_Informed_Neural_Network.total_MSE_over_training, label="MSE")
    plt.plot(training_steps, Physics_Informed_Neural_Network.MSE_boundary_conditions_over_training, label=r"$MSE_{BC}$")
    plt.plot(training_steps, Physics_Informed_Neural_Network.MSE_initial_conditions_over_training, label=r"$MSE_{IC}$")
    plt.plot(training_steps, Physics_Informed_Neural_Network.MSE_symbolic_functions_over_training, label=r"$MSE_{PDE}$")
    plt.plot(
        training_steps, Physics_Informed_Neural_Network.MSE_numerical_solution_h_over_training, label=r"$MSE_{SOL,h}$"
    )
    plt.plot(
        training_steps, Physics_Informed_Neural_Network.MSE_numerical_solution_u_over_training, label=r"$MSE_{SOL,u}$"
    )
    plt.title(
        "M"
        + str(Physics_Informed_Neural_Network.model_number)
        + " Training Curve over "
        + str(Physics_Informed_Neural_Network.epochs)
        + " Training Steps"
    )
    plt.ylabel(r"$MSE$ = $w_{PDE}$ * $MES_{PDE}$ + $w_{BC}$ * $MSE_{BC}$ + $w_{IC}$ * $MSE_{IC}$")
    plt.xlabel("iterations")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(
        training_steps, Physics_Informed_Neural_Network.MSE_initial_condition_u_over_training, label=r"$MSE_{IC,u}$"
    )
    plt.plot(
        training_steps, Physics_Informed_Neural_Network.MSE_initial_condition_h_over_training, label=r"$MSE_{IC,h}$"
    )
    plt.plot(
        training_steps, Physics_Informed_Neural_Network.MSE_symbolic_function_u_over_training, label=r"$MSE_{PDE,u}$"
    )
    plt.plot(
        training_steps, Physics_Informed_Neural_Network.MSE_symbolic_function_h_over_training, label=r"$MSE_{PDE,h}$"
    )

    if Physics_Informed_Neural_Network.boundary_conditions == "closed_boundaries":
        plt.plot(
            training_steps,
            Physics_Informed_Neural_Network.MSE_lower_boundary_condition_u_over_training,
            label=r"$MSE_{BC,u,LB}$",
        )
        plt.plot(
            training_steps,
            Physics_Informed_Neural_Network.MSE_upper_boundary_condition_u_over_training,
            label=r"$MSE_{BC,u,UB}$",
        )

    if Physics_Informed_Neural_Network.boundary_conditions == "boundary_forcing":
        plt.plot(
            training_steps,
            Physics_Informed_Neural_Network.MSE_lower_boundary_condition_h_over_training,
            label=r"$MSE_{BC,h,LB}$",
        )
        plt.plot(
            training_steps,
            Physics_Informed_Neural_Network.MSE_upper_boundary_condition_h_over_training,
            label=r"$MSE_{BC,h,UB}$",
        )

    if Physics_Informed_Neural_Network.boundary_conditions == "periodic_boundaries":
        plt.plot(
            training_steps,
            Physics_Informed_Neural_Network.MSE_boundary_condition_h_over_training,
            label=r"$MSE_{BC,u}$",
        )
        plt.plot(
            training_steps,
            Physics_Informed_Neural_Network.MSE_boundary_condition_u_over_training,
            label=r"$MSE_{BC,h}$",
        )

    plt.title(
        "M"
        + str(Physics_Informed_Neural_Network.model_number)
        + " Separate u and h MSEs over "
        + str(Physics_Informed_Neural_Network.epochs)
        + " Training Steps"
    )
    plt.xlabel("iterations")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()


# Plot_Results generates 3D surface plots of the Network Output, the Closed Form Solution and the Approximation Error
def Plot_Results(Physics_Informed_Neural_Network):
    # sns.set()

    # import Test-Grid and Test-Mesh-Grid from Network
    if Physics_Informed_Neural_Network.non_dimensionalization == True:
        time_mesh_grid = Physics_Informed_Neural_Network.dimensional_time_mesh_grid
        x_mesh_grid = Physics_Informed_Neural_Network.dimensional_x_mesh_grid
    else:
        time_mesh_grid = Physics_Informed_Neural_Network.time_mesh_grid
        x_mesh_grid = Physics_Informed_Neural_Network.x_mesh_grid
    time_input_grid = Physics_Informed_Neural_Network.time_input_grid
    x_input_grid = Physics_Informed_Neural_Network.x_input_grid
    mesh_grid_shape = Physics_Informed_Neural_Network.mesh_grid_shape

    solution_time_input_grid = Physics_Informed_Neural_Network.solution_time_input_grid
    solution_x_input_grid = Physics_Informed_Neural_Network.zeta_solution_x_input_grid
    u_solution_x_input_grid = Physics_Informed_Neural_Network.u_solution_x_input_grid

    dimensional_solution_time_mesh_grid = Physics_Informed_Neural_Network.dimensional_solution_time_mesh_grid
    dimensional_solution_x_mesh_grid = Physics_Informed_Neural_Network.dimensional_zeta_solution_x_mesh_grid
    dimensional_u_solution_x_mesh_grid = Physics_Informed_Neural_Network.dimensional_u_solution_x_mesh_grid
    dimensional_solution_mesh_grid_shape = dimensional_solution_x_mesh_grid.shape

    # define boundaries
    minimum_time = Physics_Informed_Neural_Network.minimum_time
    maximum_time = Physics_Informed_Neural_Network.maximum_time
    minimum_x = Physics_Informed_Neural_Network.minimum_x
    maximum_x = Physics_Informed_Neural_Network.maximum_x
    if Physics_Informed_Neural_Network.non_dimensionalization == True:
        minimum_time = minimum_time * Physics_Informed_Neural_Network.time_scale
        maximum_time = maximum_time * Physics_Informed_Neural_Network.time_scale
        minimum_x = minimum_x * Physics_Informed_Neural_Network.horizontal_length_scale
        maximum_x = maximum_x * Physics_Informed_Neural_Network.horizontal_length_scale

    # compute network output
    network_output_h_values = (
        Physics_Informed_Neural_Network(time_input_grid, x_input_grid)[:, 1:2]
        .cpu()
        .detach()
        .numpy()
        .reshape(mesh_grid_shape)
    )
    network_output_u_values = (
        Physics_Informed_Neural_Network(time_input_grid, x_input_grid)[:, 0:1]
        .cpu()
        .detach()
        .numpy()
        .reshape(mesh_grid_shape)
    )

    # re-dimensionalize the output
    if Physics_Informed_Neural_Network.non_dimensionalization == True:
        network_output_h_values = network_output_h_values * Physics_Informed_Neural_Network.vertical_length_scale
        network_output_u_values = (
            network_output_u_values
            * Physics_Informed_Neural_Network.horizontal_length_scale
            / Physics_Informed_Neural_Network.time_scale
        )

    # import solutions for h and u
    exact_solution_u_values = Physics_Informed_Neural_Network.exact_solution_u_values.cpu().detach().numpy()
    exact_solution_h_values = Physics_Informed_Neural_Network.exact_solution_h_values.cpu().detach().numpy()

    # 3D Surface Plot of the Network Output for h after Training
    network_output_h_surface_plot = plt.figure(figsize=(15, 7))
    network_output_axis = network_output_h_surface_plot.gca(projection="3d")
    network_ouput_h_surface = network_output_axis.plot_surface(
        time_mesh_grid, x_mesh_grid, network_output_h_values, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    plt.title(
        "M"
        + str(Physics_Informed_Neural_Network.model_number)
        + " Approximate Solution h(t,x) for the Shallow Water Equation"
    )
    plt.xlabel("t [s]")
    plt.ylabel("x [m]")
    network_output_axis.set_xlim(minimum_time, maximum_time)
    network_output_axis.set_ylim(minimum_x, maximum_x)
    network_output_h_surface_plot.colorbar(network_ouput_h_surface, shrink=0.5, aspect=5, label="h [m]")
    plt.show()

    # 3D Surface Plot of the Network Output u after Training
    network_output_u_surface_plot = plt.figure(figsize=(15, 7))
    network_output_axis = network_output_u_surface_plot.gca(projection="3d")
    network_output_u_surface = network_output_axis.plot_surface(
        time_mesh_grid, x_mesh_grid, network_output_u_values, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    plt.title(
        "M"
        + str(Physics_Informed_Neural_Network.model_number)
        + " Approximate Solution u(t,x) for the Shallow Water Equation"
    )
    plt.xlabel("t [s]")
    plt.ylabel("x [m]")
    network_output_axis.set_xlim(minimum_time, maximum_time)
    network_output_axis.set_ylim(minimum_x, maximum_x)
    network_output_u_surface_plot.colorbar(network_output_u_surface, shrink=0.5, aspect=5, label=r"$u [ms^{-1}]$")
    plt.show()

    # 3D Surface Plot of the exact Solution for h
    exact_solution_surface_plot = plt.figure(figsize=(15, 7))
    exact_solution_axis = exact_solution_surface_plot.gca(projection="3d")
    exact_solution_surface = exact_solution_axis.plot_surface(
        dimensional_solution_time_mesh_grid,
        dimensional_solution_x_mesh_grid,
        exact_solution_h_values,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.title(
        "M" + str(Physics_Informed_Neural_Network.model_number) + " Exact Solution h for the Shallow Water Equation"
    )
    plt.xlabel("t [s]")
    plt.ylabel("x [m]")
    exact_solution_surface_plot.colorbar(exact_solution_surface, shrink=0.5, aspect=5, label="h [m]")
    plt.show()

    # 3D Surface Plot of the exact Solution for h
    exact_solution_surface_plot = plt.figure(figsize=(15, 7))
    exact_solution_axis = exact_solution_surface_plot.gca(projection="3d")
    exact_solution_surface = exact_solution_axis.plot_surface(
        dimensional_solution_time_mesh_grid,
        dimensional_solution_x_mesh_grid,
        exact_solution_u_values,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.title(
        "M" + str(Physics_Informed_Neural_Network.model_number) + " Exact Solution u for the Shallow Water Equation"
    )
    plt.xlabel("t [s]")
    plt.ylabel("x [m]")
    exact_solution_surface_plot.colorbar(exact_solution_surface, shrink=0.5, aspect=5, label="h [m]")
    plt.show()

    # compute Network Output on solution grid
    network_output_h_values = (
        Physics_Informed_Neural_Network(solution_time_input_grid, solution_x_input_grid)[:, 1:2]
        .cpu()
        .detach()
        .numpy()
        .reshape(dimensional_solution_mesh_grid_shape)
    )
    network_output_u_values = (
        Physics_Informed_Neural_Network(solution_time_input_grid, u_solution_x_input_grid)[:, 0:1]
        .cpu()
        .detach()
        .numpy()
        .reshape(dimensional_solution_mesh_grid_shape)
    )

    if Physics_Informed_Neural_Network.non_dimensionalization == True:
        network_output_h_values = network_output_h_values * Physics_Informed_Neural_Network.vertical_length_scale
        network_output_u_values = (
            network_output_u_values
            * Physics_Informed_Neural_Network.horizontal_length_scale
            / Physics_Informed_Neural_Network.time_scale
        )

    abs_error_h_values = abs(exact_solution_h_values - network_output_h_values)
    # 3D Surface Plot of the Absolute Error compared to the exact Solution h
    absolute_error_surface_plot = plt.figure(figsize=(15, 7))
    absolute_error_axis = absolute_error_surface_plot.gca(projection="3d")
    absolute_error_surface = absolute_error_axis.plot_surface(
        dimensional_solution_time_mesh_grid,
        dimensional_solution_x_mesh_grid,
        abs_error_h_values,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.title(
        "M" + str(Physics_Informed_Neural_Network.model_number) + " Absolute error in h for the Shallow Water Equation"
    )
    plt.xlabel("t [s]")
    plt.ylabel("x [m]")
    absolute_error_surface_plot.colorbar(absolute_error_surface, shrink=0.5, aspect=5, label="h [m]")
    plt.show()

    # 3D Surface Plot of the Absolute Error compared to the exact Solution u
    abs_error_u_values = abs(exact_solution_u_values - network_output_u_values)
    absolute_error_surface_plot = plt.figure(figsize=(15, 7))
    absolute_error_axis = absolute_error_surface_plot.gca(projection="3d")
    absolute_error_surface = absolute_error_axis.plot_surface(
        dimensional_solution_time_mesh_grid,
        dimensional_solution_x_mesh_grid,
        abs_error_u_values,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.title(
        "M" + str(Physics_Informed_Neural_Network.model_number) + " Absolute error in u for the Shallow Water Equation"
    )
    plt.xlabel("t [s]")
    plt.ylabel("x [m]")
    absolute_error_surface_plot.colorbar(absolute_error_surface, shrink=0.5, aspect=5, label=r"$u [ms^{-1}]$")
    plt.show()


def Plot_Initial_Conditions(Physics_Informed_Neural_Network):
    # import grid
    minimum_time = Physics_Informed_Neural_Network.minimum_time
    x = (
        torch.FloatTensor(Physics_Informed_Neural_Network.x_grid)
        .unsqueeze(dim=1)
        .to(Physics_Informed_Neural_Network.device)
    )
    t = torch.zeros_like(x).to(Physics_Informed_Neural_Network.device)

    # define network output and target initial condition
    true_initial_condition_h_values = (
        Physics_Informed_Neural_Network.true_initial_condition_h_function(x).cpu().detach().numpy()
    )
    network_output_h_initial_conditions = Physics_Informed_Neural_Network(t, x)[:, 1:2].cpu().detach().numpy()
    true_initial_condition_u_values = (
        Physics_Informed_Neural_Network.true_initial_condition_u_function(x).cpu().detach().numpy()
    )
    network_output_u_initial_conditions = Physics_Informed_Neural_Network(t, x)[:, 0:1].cpu().detach().numpy()

    # re-dimensionalize
    if Physics_Informed_Neural_Network.non_dimensionalization == True:
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

    # Plot for h
    plt.figure()
    plt.plot(x.cpu().detach().numpy(), network_output_h_initial_conditions, "-", label="out")
    plt.plot(x.cpu().detach().numpy(), true_initial_condition_h_values, "-", label="sol")
    plt.title(
        "M"
        + str(Physics_Informed_Neural_Network.model_number)
        + " Target initial condition and respective network output for h"
    )
    plt.xlabel("x [m] for t = " + str(minimum_time) + " s")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot for u
    plt.figure()
    plt.plot(x.cpu().detach().numpy(), network_output_u_initial_conditions, "-", label="out")
    plt.plot(x.cpu().detach().numpy(), true_initial_condition_u_values, "-", label="sol")
    plt.title(
        "M"
        + str(Physics_Informed_Neural_Network.model_number)
        + " Target initial condition and respective network output for u"
    )
    plt.xlabel("x [m] for t = " + str(minimum_time) + " s")
    plt.grid()
    plt.legend()
    plt.show()


def Plot_Boundary_Conditions(Physics_Informed_Neural_Network):
    minimum_x = Physics_Informed_Neural_Network.minimum_x
    maximum_x = Physics_Informed_Neural_Network.maximum_x
    t = (
        torch.FloatTensor(Physics_Informed_Neural_Network.time_grid)
        .unsqueeze(dim=1)
        .to(Physics_Informed_Neural_Network.device)
    )
    x_lower_boundary = -1.0 * torch.ones_like(t).to(Physics_Informed_Neural_Network.device)
    x_upper_boundary = 1.0 * torch.ones_like(t).to(Physics_Informed_Neural_Network.device)

    if Physics_Informed_Neural_Network.boundary_conditions == "closed_boundaries":
        # define network output and target boundary condition
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
        if Physics_Informed_Neural_Network.non_dimensionalization == True:
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
            minimum_x = minimum_x * Physics_Informed_Neural_Network.horizontal_length_scale
            maximum_x = maximum_x * Physics_Informed_Neural_Network.horizontal_length_scale

        # Plot for u at lower boundary
        plt.figure()
        plt.plot(t.cpu().detach().numpy(), network_output_u_lower_boundary_condition, "-", label="out")
        plt.plot(t.cpu().detach().numpy(), true_lower_boundary_condition_u_values, "-", label="sol")
        plt.title(
            "M"
            + str(Physics_Informed_Neural_Network.model_number)
            + " Target lower boundary condition and respective network output for u"
        )
        plt.xlabel("t [s] for x = " + str(minimum_x) + " m")
        plt.grid()
        plt.legend()
        plt.show()

        # Plot for u at lower boundary
        plt.figure()
        plt.plot(t.cpu().detach().numpy(), network_output_u_upper_boundary_condition, "-", label="out")
        plt.plot(t.cpu().detach().numpy(), true_upper_boundary_condition_u_values, "-", label="sol")
        plt.title(
            "M"
            + str(Physics_Informed_Neural_Network.model_number)
            + " Target upper boundary condition and respective network output for u"
        )
        plt.xlabel("t [s] for x = " + str(maximum_x) + " m")
        plt.grid()
        plt.legend()
        plt.show()

    if Physics_Informed_Neural_Network.boundary_conditions == "boundary_forcing":
        # define network output and target boundary condition
        true_lower_boundary_condition_h_values = (
            Physics_Informed_Neural_Network.true_lower_boundary_condition_h_function(t).cpu().detach().numpy()
        )
        true_upper_boundary_condition_h_values = (
            Physics_Informed_Neural_Network.true_upper_boundary_condition_h_function(t).cpu().detach().numpy()
        )
        network_output_h_lower_boundary_condition = (
            Physics_Informed_Neural_Network(t, x_lower_boundary)[:, 1:2].cpu().detach().numpy()
        )
        network_output_h_upper_boundary_condition = (
            Physics_Informed_Neural_Network(t, x_upper_boundary)[:, 1:2].cpu().detach().numpy()
        )

        # re-dimensionalize
        if Physics_Informed_Neural_Network.non_dimensionalization == True:
            t = t * Physics_Informed_Neural_Network.time_scale
            true_lower_boundary_condition_h_values = (
                true_lower_boundary_condition_h_values
                * Physics_Informed_Neural_Network.horizontal_length_scale
                / Physics_Informed_Neural_Network.time_scale
            )
            network_output_h_lower_boundary_condition = (
                network_output_h_lower_boundary_condition
                * Physics_Informed_Neural_Network.horizontal_length_scale
                / Physics_Informed_Neural_Network.time_scale
            )
            true_upper_boundary_condition_h_values = (
                true_upper_boundary_condition_h_values
                * Physics_Informed_Neural_Network.horizontal_length_scale
                / Physics_Informed_Neural_Network.time_scale
            )
            network_output_h_upper_boundary_condition = (
                network_output_h_upper_boundary_condition
                * Physics_Informed_Neural_Network.horizontal_length_scale
                / Physics_Informed_Neural_Network.time_scale
            )
            minimum_x = minimum_x * Physics_Informed_Neural_Network.horizontal_length_scale
            maximum_x = maximum_x * Physics_Informed_Neural_Network.horizontal_length_scale

        # Plot for h at lower boundary
        plt.figure()
        plt.plot(t.cpu().detach().numpy(), network_output_h_lower_boundary_condition, "-", label="out")
        plt.plot(t.cpu().detach().numpy(), true_lower_boundary_condition_h_values, "-", label="sol")
        plt.title(
            "M"
            + str(Physics_Informed_Neural_Network.model_number)
            + " Target lower boundary condition and respective network output for h"
        )
        plt.xlabel("t [s] for x = " + str(minimum_x) + " m")
        plt.grid()
        plt.legend()
        plt.show()

        # Plot for h at lower boundary
        plt.figure()
        plt.plot(t.cpu().detach().numpy(), network_output_h_upper_boundary_condition, "-", label="out")
        plt.plot(t.cpu().detach().numpy(), true_upper_boundary_condition_h_values, "-", label="sol")
        plt.title(
            "M"
            + str(Physics_Informed_Neural_Network.model_number)
            + " Target upper boundary condition and respective network output for h"
        )
        plt.xlabel("t [s] for x = " + str(maximum_x) + " m")
        plt.grid()
        plt.legend()
        plt.show()


def Plot_New_Initial_Conditions(Physics_Informed_Neural_Network):
    # import grid
    minimum_time = Physics_Informed_Neural_Network.minimum_time
    t = Physics_Informed_Neural_Network.new_initial_condition_sampling_points[:, 0:1]
    x = Physics_Informed_Neural_Network.new_initial_condition_sampling_points[:, 1:2]

    # define network output and target initial condition
    true_initial_condition_u_values = (
        Physics_Informed_Neural_Network.new_initial_conditions[:, 0:1].cpu().detach().numpy()
    )
    true_initial_condition_h_values = (
        Physics_Informed_Neural_Network.new_initial_conditions[:, 1:2].cpu().detach().numpy()
    )
    network_output_u_initial_conditions = Physics_Informed_Neural_Network(t, x)[:, 0:1].cpu().detach().numpy()
    network_output_h_initial_conditions = Physics_Informed_Neural_Network(t, x)[:, 1:2].cpu().detach().numpy()

    # re-dimensionalize
    if Physics_Informed_Neural_Network.non_dimensionalization == True:
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

    # Plot for h
    plt.figure()
    plt.plot(x.cpu().detach().numpy(), network_output_h_initial_conditions, "o", label="out")
    plt.plot(x.cpu().detach().numpy(), true_initial_condition_h_values, "o", label="sol")
    plt.title(
        "M"
        + str(Physics_Informed_Neural_Network.model_number)
        + " Target initial condition and respective network output for h"
    )
    plt.xlabel("x [m] for t = " + str(minimum_time) + " s")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot for u
    plt.figure()
    plt.plot(x.cpu().detach().numpy(), network_output_u_initial_conditions, "o", label="out")
    plt.plot(x.cpu().detach().numpy(), true_initial_condition_u_values, "o", label="sol")
    plt.title(
        "M"
        + str(Physics_Informed_Neural_Network.model_number)
        + " Target initial condition and respective network output for u"
    )
    plt.xlabel("x [m] for t = " + str(minimum_time) + " s")
    plt.grid()
    plt.legend()
    plt.show()


def Plot_PDE_Losses(Physics_Informed_Neural_Network):
    mesh_grid_shape = Physics_Informed_Neural_Network.mesh_grid_shape
    t = Physics_Informed_Neural_Network.time_input_grid.clone().detach().requires_grad_(True)
    x = Physics_Informed_Neural_Network.x_input_grid.clone().detach().requires_grad_(True)
    Physics_Informed_Neural_Network.symbolic_function_sampling_points = torch.hstack((t, x))
    Physics_Informed_Neural_Network.Physics_Informed_Symbolic_Function()

    time_mesh_grid = Physics_Informed_Neural_Network.time_mesh_grid * Physics_Informed_Neural_Network.time_scale
    x_mesh_grid = Physics_Informed_Neural_Network.x_mesh_grid * Physics_Informed_Neural_Network.horizontal_length_scale
    PDE_Loss_u = Physics_Informed_Neural_Network.symbolic_function_u_values.reshape(shape=mesh_grid_shape)
    PDE_Loss_h = Physics_Informed_Neural_Network.symbolic_function_h_values.reshape(shape=mesh_grid_shape)

    PDE_Loss_u_figure = plt.figure()
    PDE_Loss_u_axis = PDE_Loss_u_figure.gca(projection="3d")
    PDE_Loss_u_axis.plot_surface(
        time_mesh_grid / 60,
        0.001 * x_mesh_grid,
        PDE_Loss_u.cpu().detach().numpy(),
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
    )
    plt.title("PDE Loss Term for u")
    plt.xlabel("t [min]")
    plt.ylabel("x [km]")
    plt.show()

    PDE_Loss_h_figure = plt.figure()
    PDE_Loss_h_axis = PDE_Loss_h_figure.gca(projection="3d")
    PDE_Loss_h_axis.plot_surface(
        time_mesh_grid / 60,
        0.001 * x_mesh_grid,
        PDE_Loss_h.cpu().detach().numpy(),
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
    )
    plt.title("PDE Loss Term for h")
    plt.xlabel("t [min]")
    plt.ylabel("x [km]")
    plt.show()
