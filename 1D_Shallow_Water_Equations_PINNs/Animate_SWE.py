# import External Libraries
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# Animate_Results generates an animation showing the Development of the Network Output over Training,
# the development of the approximation error and the respective learning curve
def Animate_Results(Physics_Informed_Neural_Network, frames_per_second=5):
    # import Test-Grid from Network
    time_input_grid = Physics_Informed_Neural_Network.time_input_grid
    x_input_grid = Physics_Informed_Neural_Network.x_input_grid
    time_mesh_grid = Physics_Informed_Neural_Network.time_mesh_grid
    x_mesh_grid = Physics_Informed_Neural_Network.x_mesh_grid
    training_steps = range(Physics_Informed_Neural_Network.epochs + 1)

    # compute exact Solution, convert it into a Numpy Array and reshape it
    Physics_Informed_Neural_Network.exact_solution_h_function(time_input_grid, x_input_grid)
    exact_solution_h_values = (
        Physics_Informed_Neural_Network.exact_solution_h_values.cpu()
        .detach()
        .numpy()
        .reshape(Physics_Informed_Neural_Network.mesh_grid_shape)
    )

    # initialize Figure and Axes for Animation
    training_animation_figure = plt.figure(figsize=(20, 25))
    plt.rcParams.update({"font.size": 30})
    network_output_axis = training_animation_figure.add_subplot(221, projection="3d")
    error_surface_plot_axis = training_animation_figure.add_subplot(222, projection="3d")
    learning_curve_axis = training_animation_figure.add_subplot(212)

    # 3D Surface Plot of Network Output
    def update_network_output(frame, network_output_axis):
        network_output_axis.cla()
        network_output_axis.plot_surface(
            time_mesh_grid,
            x_mesh_grid,
            Physics_Informed_Neural_Network.network_output_h_over_training[frame],
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        network_output_axis.set_xlim(0, 1)
        network_output_axis.set_ylim(-1, 1)
        network_output_axis.set_zlim(-1, 1)
        network_output_axis.set_xlabel("\n t", linespacing=5)
        network_output_axis.set_ylabel("\n x", linespacing=5)
        network_output_axis.set_title(
            "Approximate Solution h(t,x) for the \n Shallow Water Equations at the " + str(frame) + "-th iteration",
            fontsize=27,
        )
        network_output_axis.tick_params(axis="x", pad=15)
        network_output_axis.tick_params(axis="y", pad=10)
        network_output_axis.tick_params(axis="z", pad=15)

    # 3D Surface Plot of Approximation Error in Network Output compared to exact Solution
    def update_approximation_error(frame, error_surface_plot_axis):
        error_surface_plot_axis.cla()
        error_surface_plot_axis.plot_surface(
            time_mesh_grid,
            x_mesh_grid,
            abs(exact_solution_h_values - Physics_Informed_Neural_Network.network_output_h_over_training[frame]),
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        error_surface_plot_axis.set_xlim(0, 1)
        error_surface_plot_axis.set_ylim(-1, 1)
        error_surface_plot_axis.set_zlim(0, 1)
        error_surface_plot_axis.set_title(
            "Absolute error in the Solution for the \n Shallow Water Equation at the " + str(frame) + "-th iteration",
            fontsize=27,
        )
        error_surface_plot_axis.set_xlabel("\n t", linespacing=5)
        error_surface_plot_axis.set_ylabel("\n x", linespacing=5)
        error_surface_plot_axis.tick_params(axis="x", pad=15)
        error_surface_plot_axis.tick_params(axis="y", pad=10)
        error_surface_plot_axis.tick_params(axis="z", pad=15)

    # Learning Curve with all individual Mean Squared Error (MSE) terms over training
    def update_MSE_terms(frame, learning_curve_axis):
        learning_curve_axis.cla()
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.total_MSE_over_training, label=r"$MSE$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.MSE_boundary_conditions_over_training, label=r"$MSE_{BC}$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.MSE_initial_conditions_over_training, label=r"$MSE_{IC}$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.MSE_symbolic_functions_over_training, label=r"$MSE_{PDE}$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.MSE_numerical_solution_h_over_training, label=r"$MSE_{SOL}$"
        )
        learning_curve_axis.axvline(x=frame)
        learning_curve_axis.set_title(
            "Loss Functions at the "
            + str(frame)
            + "-th of "
            + str(Physics_Informed_Neural_Network.epochs)
            + " Iterations"
        )
        learning_curve_axis.set_xlabel("number of iterations")
        learning_curve_axis.set_ylabel(r"$MSE$ = $MES_{PDE}$ + $MSE_{BC}$ + $MSE_{IC}$")
        learning_curve_axis.legend()
        learning_curve_axis.grid()

    # Function calling all three Plot-Update Functions
    def update_all_axes(frame, network_output_axis, error_surface_plot_axis, learning_curve_axis):
        network_output_surface_plot = update_network_output(frame, network_output_axis)
        error_surface_plot = update_approximation_error(frame, error_surface_plot_axis)
        learning_curve_plot = update_MSE_terms(frame, learning_curve_axis)
        return network_output_surface_plot, error_surface_plot, learning_curve_plot

    # generate and save Animation
    number_of_frames = len(Physics_Informed_Neural_Network.network_output_h_over_training)
    training_animation = animation.FuncAnimation(
        training_animation_figure,
        update_all_axes,
        fargs=[network_output_axis, error_surface_plot_axis, learning_curve_axis],
        frames=np.arange(number_of_frames),
        init_func=None,
        interval=1000 / frames_per_second,
    )
    training_animation.save("1D_Shallow_Water_Equations/Results.gif", writer="Pillow", fps=int(frames_per_second))


def Animate_Solution(Physics_Informed_Neural_Network, frames_per_second=100):
    # import grid
    time_input_grid = Physics_Informed_Neural_Network.solution_time_input_grid
    x_input_grid = Physics_Informed_Neural_Network.zeta_solution_x_input_grid
    u_solution_x_input_grid = Physics_Informed_Neural_Network.u_solution_x_input_grid
    mesh_grid_shape = Physics_Informed_Neural_Network.solution_mesh_grid_shape
    x_grid = Physics_Informed_Neural_Network.zeta_solution_x_grid
    u_solution_x_grid = Physics_Informed_Neural_Network.u_solution_x_grid
    minimum_x = Physics_Informed_Neural_Network.minimum_x
    maximum_x = Physics_Informed_Neural_Network.maximum_x

    # compute network output
    network_output_u = Physics_Informed_Neural_Network.forward(time_input_grid, u_solution_x_input_grid)[:, 0:1]
    network_output_h = Physics_Informed_Neural_Network.forward(time_input_grid, x_input_grid)[:, 1:2]

    # import numerical solution
    numerical_solution_u = Physics_Informed_Neural_Network.exact_solution_u_values.cpu().detach().numpy()
    numerical_solution_h = Physics_Informed_Neural_Network.exact_solution_h_values.cpu().detach().numpy()

    if Physics_Informed_Neural_Network.non_dimensionalization == True:
        minimum_x = minimum_x * Physics_Informed_Neural_Network.horizontal_length_scale
        maximum_x = maximum_x * Physics_Informed_Neural_Network.horizontal_length_scale
        x_grid = x_grid * Physics_Informed_Neural_Network.horizontal_length_scale
        u_solution_x_grid = u_solution_x_grid * Physics_Informed_Neural_Network.horizontal_length_scale
        network_output_u = (
            (
                network_output_u
                * Physics_Informed_Neural_Network.horizontal_length_scale
                / Physics_Informed_Neural_Network.time_scale
            )
            .cpu()
            .detach()
            .numpy()
            .reshape(mesh_grid_shape)
        )
        network_output_h = (
            (network_output_h * Physics_Informed_Neural_Network.vertical_length_scale)
            .cpu()
            .detach()
            .numpy()
            .reshape(mesh_grid_shape)
        )
    else:
        network_output_u.cpu().detach().numpy().reshape(mesh_grid_shape)
        network_output_h.cpu().detach().numpy().reshape(mesh_grid_shape)

    # Generate Animation for h
    sea_level_figure = plt.figure()

    def update_plot(frame):
        plt.cla()
        plt.plot(0.001 * x_grid, network_output_h[:, frame], label="out")
        plt.plot(0.001 * x_grid, numerical_solution_h[:, frame], label="sol")
        plt.plot(0.001 * x_grid, numerical_solution_h[:, frame] - network_output_h[:, frame], label="sol-out")
        plt.xlim(0.001 * minimum_x, 0.001 * maximum_x)
        plt.ylim(-1, 1)
        time = (
            frame
            + Physics_Informed_Neural_Network.model_number
            * Physics_Informed_Neural_Network.numerical_solution_time_interval[1]
            / 60.0
            / Physics_Informed_Neural_Network.number_of_models
        )
        plt.title("Network Output for Sea Level Elevation at t = " + str(time) + " minutes")
        plt.xlabel("x [km]")
        plt.ylabel("h [m]")
        plt.legend()
        return

    # generate and save Animation
    number_of_frames = np.shape(network_output_h)[1]
    sea_level_animation = animation.FuncAnimation(
        sea_level_figure, update_plot, frames=np.arange(0, number_of_frames, 50), init_func=None
    )
    sea_level_animation.save(
        "1D_Shallow_Water_Equations/sea_level_" + str(Physics_Informed_Neural_Network.model_number) + ".gif",
        writer="Pillow",
        fps=int(frames_per_second),
    )

    # Generate Animation for u
    zonal_velocity_figure = plt.figure()

    def update_plot(frame):
        plt.cla()
        plt.plot(0.001 * u_solution_x_grid, network_output_u[:, frame], label="out")
        plt.plot(0.001 * u_solution_x_grid, numerical_solution_u[:, frame], label="sol")
        plt.plot(
            0.001 * u_solution_x_grid, numerical_solution_u[:, frame] - network_output_u[:, frame], label="sol-out"
        )
        plt.xlim(0.001 * minimum_x, 0.001 * maximum_x)
        plt.ylim(-0.4, 0.4)
        time = (
            frame
            + Physics_Informed_Neural_Network.model_number
            * Physics_Informed_Neural_Network.numerical_solution_time_interval[1]
            / 60.0
            / Physics_Informed_Neural_Network.number_of_models
        )
        plt.title("Network Output for Zonal Velocity at t = " + str(time) + " minutes")
        plt.xlabel("x [km]")
        plt.ylabel("u [m/s]")
        plt.legend()
        return

    # generate and save Animation
    number_of_frames = np.shape(network_output_u)[1]
    zonal_velocity_animation = animation.FuncAnimation(
        zonal_velocity_figure, update_plot, frames=np.arange(0, number_of_frames, 50), init_func=None
    )
    zonal_velocity_animation.save(
        "1D_Shallow_Water_Equations/zonal_velocity_" + str(Physics_Informed_Neural_Network.model_number) + ".gif",
        writer="Pillow",
        fps=int(frames_per_second),
    )


def Animate_PDE_Losses(Physics_Informed_Neural_Network, frames_per_second=10):

    # animation parameters
    number_of_frames = len(Physics_Informed_Neural_Network.symbolic_function_u_over_training)
    if number_of_frames >= 100:
        frame_step = int(number_of_frames / 100)
    else:
        frame_step = 1

    # import grid
    x_mesh_grid = Physics_Informed_Neural_Network.x_mesh_grid
    time_mesh_grid = Physics_Informed_Neural_Network.time_mesh_grid
    training_steps = range(Physics_Informed_Neural_Network.epochs + 1)
    maximum_x = Physics_Informed_Neural_Network.maximum_x
    minimum_x = Physics_Informed_Neural_Network.minimum_x
    maximum_time = Physics_Informed_Neural_Network.maximum_time
    minimum_time = Physics_Informed_Neural_Network.minimum_time

    # initialize Figure and Axes for Animation
    animation_figure = plt.figure(figsize=(14, 21))
    symbolic_function_u_axis = animation_figure.add_subplot(325)
    symbolic_function_h_axis = animation_figure.add_subplot(326)
    learning_curve_axis = animation_figure.add_subplot(312)
    network_output_u_axis = animation_figure.add_subplot(321)
    network_output_h_axis = animation_figure.add_subplot(322)

    # 3D Surface Plot of the Network Output for u
    def update_network_output_u(frame, network_output_u_axis):
        network_output_u_axis.cla()
        if Physics_Informed_Neural_Network.non_dimensionalization == True:
            contour_u = network_output_u_axis.contourf(
                time_mesh_grid,
                x_mesh_grid,
                (
                    Physics_Informed_Neural_Network.network_output_u_over_training[frame]
                    * Physics_Informed_Neural_Network.horizontal_length_scale
                    / Physics_Informed_Neural_Network.time_scale
                ),
                cmap="gist_rainbow",
                levels=np.arange(-0.3, 0.3, 0.01),
            )
        else:
            contour_u = network_output_u_axis.contourf(
                time_mesh_grid,
                x_mesh_grid,
                Physics_Informed_Neural_Network.network_output_u_over_training[frame],
                cmap="gist_rainbow",
                levels=np.arange(-0.3, 0.3, 0.01),
            )
        network_output_u_axis.set_xlim(minimum_time, maximum_time)
        network_output_u_axis.set_ylim(minimum_x, maximum_x)
        if frame == frame_step:
            animation_figure.colorbar(contour_u, ax=network_output_u_axis)
        network_output_u_axis.set_title("Network Output for u at " + str(frame) + "-th iteration")

    # 3D Surface Plot of the Network Output for h
    def update_network_output_h(frame, network_output_h_axis):
        network_output_h_axis.cla()
        if Physics_Informed_Neural_Network.non_dimensionalization == True:
            contour_h = network_output_h_axis.contourf(
                time_mesh_grid,
                x_mesh_grid,
                (
                    Physics_Informed_Neural_Network.network_output_h_over_training[frame]
                    * Physics_Informed_Neural_Network.vertical_length_scale
                ),
                cmap="gist_rainbow",
                levels=np.arange(-1, 1, 0.01),
            )
        else:
            contour_h = network_output_h_axis.contourf(
                time_mesh_grid,
                x_mesh_grid,
                Physics_Informed_Neural_Network.network_output_h_over_training[frame],
                cmap="gist_rainbow",
                levels=np.arange(-1, 1, 0.01),
            )
        network_output_h_axis.set_xlim(minimum_time, maximum_time)
        network_output_h_axis.set_ylim(minimum_x, maximum_x)
        if frame == frame_step:
            animation_figure.colorbar(contour_h, ax=network_output_h_axis)
        network_output_h_axis.set_title("Network Output for h at " + str(frame) + "-th iteration")

    # 3D Surface Plot of PDE Loss of u
    def update_symbolic_function_u(frame, symbolic_function_u_axis):
        symbolic_function_u_axis.cla()
        contour_u = symbolic_function_u_axis.contourf(
            time_mesh_grid,
            x_mesh_grid,
            np.log10(abs(Physics_Informed_Neural_Network.symbolic_function_u_over_training[frame])),
            cmap="gist_rainbow",
            levels=np.arange(-6, 1, 0.1),
        )
        symbolic_function_u_axis.set_xlim(minimum_time, maximum_time)
        symbolic_function_u_axis.set_ylim(minimum_x, maximum_x)
        if frame == frame_step:
            animation_figure.colorbar(contour_u, ax=symbolic_function_u_axis)
        symbolic_function_u_axis.set_title("PDE Loss for u at " + str(frame) + "-th iteration")

        # 3D Surface Plot of PDE Loss for h

    def update_symbolic_function_h(frame, symbolic_function_h_axis):
        symbolic_function_h_axis.cla()
        contour_h = symbolic_function_h_axis.contourf(
            time_mesh_grid,
            x_mesh_grid,
            np.log10(abs(Physics_Informed_Neural_Network.symbolic_function_h_over_training[frame])),
            cmap="gist_rainbow",
            levels=np.arange(-6, 1, 0.1),
        )
        symbolic_function_h_axis.set_xlim(minimum_time, maximum_time)
        symbolic_function_h_axis.set_ylim(minimum_x, maximum_x)
        if frame == frame_step:
            animation_figure.colorbar(contour_h, ax=symbolic_function_h_axis)
        symbolic_function_h_axis.set_title("PDE Loss for h at " + str(frame) + "-th iteration")

    def update_MSE_terms(frame, learning_curve_axis):
        learning_curve_axis.cla()
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.total_MSE_over_training, label=r"$MSE$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.MSE_boundary_conditions_over_training, label=r"$MSE_{BC}$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.MSE_initial_conditions_over_training, label=r"$MSE_{IC}$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.MSE_symbolic_functions_over_training, label=r"$MSE_{PDE}$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.MSE_numerical_solution_h_over_training, label=r"$MSE_{SOL}$"
        )
        learning_curve_axis.axvline(x=frame)
        learning_curve_axis.set_title(
            "Loss Functions at the "
            + str(frame)
            + "-th of "
            + str(Physics_Informed_Neural_Network.epochs)
            + " Iterations"
        )
        learning_curve_axis.set_xlabel("number of iterations")
        learning_curve_axis.set_ylabel(r"$MSE$ = $MES_{PDE}$ + $MSE_{BC}$ + $MSE_{IC}$")
        learning_curve_axis.legend()
        learning_curve_axis.grid()

    # Function calling all three Plot-Update Functions
    def update_all_axes(
        frame,
        symbolic_function_u_axis,
        symbolic_function_h_axis,
        network_output_u_axis,
        network_output_h_axis,
        learning_curve_axis,
    ):
        symbolic_function_u_plot = update_symbolic_function_u(frame, symbolic_function_u_axis)
        symbolic_function_h_plot = update_symbolic_function_h(frame, symbolic_function_h_axis)
        network_output_u_plot = update_network_output_u(frame, network_output_u_axis)
        network_output_h_plot = update_network_output_h(frame, network_output_h_axis)
        learning_curve_plot = update_MSE_terms(frame, learning_curve_axis)
        return (
            symbolic_function_u_plot,
            symbolic_function_h_plot,
            network_output_u_plot,
            network_output_h_plot,
            learning_curve_plot,
        )

    # generate and save Animation
    training_animation = animation.FuncAnimation(
        animation_figure,
        update_all_axes,
        fargs=[
            symbolic_function_u_axis,
            symbolic_function_h_axis,
            network_output_u_axis,
            network_output_h_axis,
            learning_curve_axis,
        ],
        frames=np.arange(0, number_of_frames, frame_step),
        init_func=None,
        interval=1000 / frames_per_second,
    )
    training_animation.save("1D_Shallow_Water_Equations/PDE_Losses.gif", writer="Pillow", fps=int(frames_per_second))

    return
