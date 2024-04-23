# %% import packages
import matplotlib.pyplot as plt
import numpy as np


# Plot_learning_curve generates a plot showing all Mean Squared Error (MSE) terms over training
def Plot_Learning_Curve(Physics_Informed_Neural_Network):
    training_steps = range(Physics_Informed_Neural_Network.epochs + 1)
    plt.plot(training_steps, Physics_Informed_Neural_Network.total_MSE_over_training, label="MSE")
    plt.plot(training_steps, Physics_Informed_Neural_Network.MSE_boundary_conditions_over_training, label=r"MSE_{BC}")
    plt.plot(training_steps, Physics_Informed_Neural_Network.MSE_initial_conditions_over_training, label=r"MSE_{IC}")
    plt.plot(training_steps, Physics_Informed_Neural_Network.MSE_symbolic_function_over_training, label=r"MSE_{PDE}")
    plt.plot(training_steps, Physics_Informed_Neural_Network.Relative_L2_Error_over_training, label=r"L2")
    plt.title("Learning Curve Over " + str(Physics_Informed_Neural_Network.epochs) + " Training Steps")
    plt.ylabel(r"$MSE$ = $w_{PDE}$ * $MES_{PDE}$ + $w_{BC}$ * $MSE_{BC}$ + $w_{IC}$ * $MSE_{IC}$")
    plt.xlabel("Training Steps")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()


# Plot_Results generates 3D surface plots of the Network Output, the Closed Form Solution and the Approximation Error
def Plot_Results(Physics_Informed_Neural_Network):
    # import solution and grid from Network
    exact_solution_values = Physics_Informed_Neural_Network.exact_solution_values
    time_mesh_grid = Physics_Informed_Neural_Network.time_mesh_grid
    x_mesh_grid = Physics_Informed_Neural_Network.x_mesh_grid
    time_input_grid = Physics_Informed_Neural_Network.time_input_grid
    x_input_grid = Physics_Informed_Neural_Network.x_input_grid

    # compute Network Output, reshape it and convert it into a Numpy Array
    network_output = (
        Physics_Informed_Neural_Network(time_input_grid, x_input_grid)
        .cpu()
        .detach()
        .numpy()
        .reshape(Physics_Informed_Neural_Network.mesh_grid_shape)
    )

    # 3D Surface Plot of the Network Output after Training
    plt.figure(figsize=(10, 5))
    plt.title("Approximate Solution to the Burgers Equation")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.pcolormesh(time_mesh_grid, x_mesh_grid, network_output, vmin=-1, vmax=1, cmap="jet", shading="nearest")
    plt.colorbar(label="u")

    # 3D Surface Plot of the exact Solution
    plt.figure(figsize=(10, 5))
    plt.title("Exact Solution to the Burgers Equation")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.pcolormesh(
        time_mesh_grid,
        x_mesh_grid,
        exact_solution_values.cpu().detach().numpy(),
        vmin=-1,
        vmax=1,
        cmap="jet",
        shading="nearest",
    )
    plt.colorbar(label="u")

    # 3D Surface Plot of the Absolute Error compared to the exact Solution
    plt.figure(figsize=(10, 5))
    plt.title("Error in Approximate Solution to the Burgers Equation")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.pcolormesh(
        time_mesh_grid,
        x_mesh_grid,
        abs(exact_solution_values.cpu().detach().numpy() - network_output),
        vmin=0,
        vmax=0.2,
        cmap="jet",
        shading="nearest",
    )
    plt.colorbar(label="u")

    # print the Mean Absolute Error on the chosen Grid Points
    print(
        "The relative error in the L2-norm is: "
        + str(
            np.linalg.norm(exact_solution_values.cpu().detach().numpy() - network_output, 2)
            / np.linalg.norm(exact_solution_values.cpu().detach().numpy())
        )
    )
