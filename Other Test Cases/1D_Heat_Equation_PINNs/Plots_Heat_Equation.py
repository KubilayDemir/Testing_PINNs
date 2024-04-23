# import External Libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# Plot_learning_curve generates a plot showing all Mean Squared Error (MSE) terms over training
def Plot_Learning_Curve(Physics_Informed_Neural_Network):
    training_steps = range(Physics_Informed_Neural_Network.epochs + 1)
    plt.plot(training_steps, Physics_Informed_Neural_Network.total_MSE_over_training, label=r"$MSE$")
    plt.plot(training_steps, Physics_Informed_Neural_Network.MSE_boundary_conditions_over_training, label=r"$MSE_{BC}$")
    plt.plot(training_steps, Physics_Informed_Neural_Network.MSE_initial_conditions_over_training, label=r"$MSE_{IC}$")
    plt.plot(training_steps, Physics_Informed_Neural_Network.MSE_symbolic_function_over_training, label=r"$MSE_{PDE}$")
    plt.plot(training_steps, Physics_Informed_Neural_Network.Relative_L2_Error_over_training, label=r"$MSE_{sol}$")
    plt.title("Loss Function Over " + str(Physics_Informed_Neural_Network.epochs) + " Iterations")
    plt.ylabel(r"$MSE$ = $w_{PDE}$ * $MES_{PDE}$ + $w_{BC}$ * $MSE_{BC}$ + $w_{IC}$ * $MSE_{IC}$")
    plt.xlabel("iterations")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()


# Plot_Results generates 3D surface plots of the Network Output, the Closed Form Solution and the Approximation error
def Plot_Results(Physics_Informed_Neural_Network):
    # import the test-grid from model
    time_input_grid = Physics_Informed_Neural_Network.time_input_grid
    x_input_grid = Physics_Informed_Neural_Network.x_input_grid
    time_mesh_grid = Physics_Informed_Neural_Network.time_mesh_grid
    x_mesh_grid = Physics_Informed_Neural_Network.x_mesh_grid

    # compute Network Output, convert it into a Numpy Array and reshape it
    network_output = (
        Physics_Informed_Neural_Network(time_input_grid, x_input_grid)
        .cpu()
        .detach()
        .numpy()
        .reshape(Physics_Informed_Neural_Network.mesh_grid_shape)
    )

    # compute Closed Form Solution, convert it into a Numpy Array and reshape it
    Physics_Informed_Neural_Network.closed_form_solution(time_input_grid, x_input_grid)
    closed_form_solution_values = (
        Physics_Informed_Neural_Network.closed_form_solution_values.cpu()
        .detach()
        .numpy()
        .reshape(Physics_Informed_Neural_Network.mesh_grid_shape)
    )

    # 3D Surface Plot of the Network Output after Training
    Network_Output_Surface_Plot = plt.figure(figsize=(15, 7))
    Network_Output_Axis = Network_Output_Surface_Plot.gca(projection="3d")
    surf = Network_Output_Axis.plot_surface(
        time_mesh_grid, x_mesh_grid, network_output, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    plt.title("Approximate Solution to the Heat Equation")
    plt.xlabel("t")
    plt.ylabel("x")
    Network_Output_Axis.set_xlim(0, 1)
    Network_Output_Axis.set_ylim(-1, 1)
    Network_Output_Axis.set_zlim(-1, 1)
    Network_Output_Surface_Plot.colorbar(surf, shrink=0.5, aspect=5, label="T")
    plt.show()

    # 3D Surface Plot of the Close Form Solution
    Solution_Surface_Plot = plt.figure(figsize=(15, 7))
    Solution_Axis = Solution_Surface_Plot.gca(projection="3d")
    surf = Solution_Axis.plot_surface(
        time_mesh_grid, x_mesh_grid, closed_form_solution_values, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    plt.title("Exact Solution to the Heat Equation")
    plt.xlabel("t")
    plt.ylabel("x")
    Solution_Axis.set_xlim(0, 1)
    Solution_Axis.set_ylim(-1, 1)
    Solution_Axis.set_zlim(-1, 1)
    Solution_Surface_Plot.colorbar(surf, shrink=0.5, aspect=5, label="T")
    plt.show()

    # 3D Surface Plot of the Absolute Error compared to the Closed Form Solution
    Error_Surface_Plot = plt.figure(figsize=(15, 7))
    Error_Axis = Error_Surface_Plot.gca(projection="3d")
    surf = Error_Axis.plot_surface(
        time_mesh_grid,
        x_mesh_grid,
        abs(closed_form_solution_values - network_output),
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    plt.title("Absolute Error in the Approximate Solution to th  Heat Equation")
    plt.xlabel("t")
    plt.ylabel("x")
    Error_Axis.set_xlim(0, 1)
    Error_Axis.set_ylim(-1, 1)
    # Error_Axis.set_zlim(0, 1)
    Error_Surface_Plot.colorbar(surf, shrink=0.5, aspect=5, label="T")
    plt.show()

    # print the Mean Absolute Error on the chosen Grid Points
    print(
        "The Mean Absolute Error on the chosen test-grid is: "
        + str(abs(closed_form_solution_values - network_output).sum().sum() / np.size(network_output))
    )
