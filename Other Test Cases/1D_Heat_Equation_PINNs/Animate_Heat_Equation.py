# import External Libraries
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Animate_Results generates an animation showing the Development of the Network Output over Training,
# the development of the approximation error and the respective learning curve


def Animate_Results(Physics_Informed_Neural_Network, frames_per_second=5):

    # import grid for computing solution
    time_input_grid = Physics_Informed_Neural_Network.time_input_grid
    x_input_grid = Physics_Informed_Neural_Network.x_input_grid
    time_mesh_grid = Physics_Informed_Neural_Network.time_mesh_grid
    x_mesh_grid = Physics_Informed_Neural_Network.x_mesh_grid
    training_steps = range(Physics_Informed_Neural_Network.epochs + 1)

    # compute and reshape exact solution
    Physics_Informed_Neural_Network.closed_form_solution(time_input_grid, x_input_grid)
    closed_form_solution_values = (
        Physics_Informed_Neural_Network.closed_form_solution_values.cpu()
        .detach()
        .numpy()
        .reshape(Physics_Informed_Neural_Network.mesh_grid_shape)
    )

    # initialize figure
    training_animation_figure = plt.figure(figsize=(20, 25))
    plt.rcParams.update({"font.size": 30})
    network_output_axis = training_animation_figure.add_subplot(221, projection="3d")
    error_surface_plot_axis = training_animation_figure.add_subplot(222, projection="3d")
    learning_curve_axis = training_animation_figure.add_subplot(212)

    def update_network_output(frame, network_output_axis):  # 3D Plot of Network Output
        network_output_axis.cla()
        network_output_axis.plot_surface(
            time_mesh_grid,
            x_mesh_grid,
            Physics_Informed_Neural_Network.network_output_over_training[frame],
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
            "Approximate Solution T for the \n Heat Equation at the " + str(frame) + "-th iteration", fontsize=27
        )
        network_output_axis.tick_params(axis="x", pad=15)
        network_output_axis.tick_params(axis="y", pad=10)
        network_output_axis.tick_params(axis="z", pad=15)

    def update_approximation_error(frame, error_surface_plot_axis):  # 3D Plot of u_sol - u
        error_surface_plot_axis.cla()
        error_surface_plot_axis.plot_surface(
            time_mesh_grid,
            x_mesh_grid,
            abs(closed_form_solution_values - Physics_Informed_Neural_Network.network_output_over_training[frame]),
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        error_surface_plot_axis.set_xlim(0, 1)
        error_surface_plot_axis.set_ylim(-1, 1)
        error_surface_plot_axis.set_zlim(0, 1)
        error_surface_plot_axis.set_title(
            "Error e in the Solution for the \n Heat Equation at the " + str(frame) + "-th iteration", fontsize=27
        )
        error_surface_plot_axis.set_xlabel("\n t", linespacing=5)
        error_surface_plot_axis.set_ylabel("\n x", linespacing=5)
        error_surface_plot_axis.tick_params(axis="x", pad=15)
        error_surface_plot_axis.tick_params(axis="y", pad=10)
        error_surface_plot_axis.tick_params(axis="z", pad=15)

    def update_MSE_terms(frame, learning_curve_axis):  # Plot losses
        learning_curve_axis.cla()
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.total_MSE_over_training, label=r"$MSE$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.MSE_boundary_conditions_over_training, label=r"$MSE_u$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.MSE_initial_conditions_over_training, label=r"$MSE_0$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.MSE_symbolic_function_over_training, label=r"$MSE_f$"
        )
        learning_curve_axis.semilogy(
            training_steps, Physics_Informed_Neural_Network.Relative_L2_Error_over_training, label=r"$MSE_{sol}$"
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
        learning_curve_axis.set_ylabel(r"$MSE$ = $MES_{u}$ + $MSE_{f}$ + $MSE_{0}$")
        learning_curve_axis.legend()
        learning_curve_axis.grid()

    # Function calling all three Plot-Update Functions
    def update_all_axes(frame, network_output_axis, error_surface_plot_axis, learning_curve_axis):
        network_output_surface_plot = update_network_output(frame, network_output_axis)
        error_surface_plot = update_approximation_error(frame, error_surface_plot_axis)
        learning_curve_plot = update_MSE_terms(frame, learning_curve_axis)
        return network_output_surface_plot, error_surface_plot, learning_curve_plot

    # generate and save Animation
    number_of_frames = len(Physics_Informed_Neural_Network.network_output_over_training)
    training_animation = animation.FuncAnimation(
        training_animation_figure,
        update_all_axes,
        fargs=[network_output_axis, error_surface_plot_axis, learning_curve_axis],
        frames=np.arange(number_of_frames),
        init_func=None,
        interval=1000 / frames_per_second,
    )
    training_animation.save("Heat_Equation.gif", writer="Pillow", fps=int(frames_per_second))
