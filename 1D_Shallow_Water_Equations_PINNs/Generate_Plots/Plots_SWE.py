# import External Libraries
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Plot_learning_curve generates a plot showing all Mean Squared Error (MSE) terms over training.
# Additionally, a plot showing individual MSE terms for u and h is generated.

def Plot_Learning_Curve():

    # Opening JSON file
    with open("Output_Data/Hyper_Parameter_Dictionary.json") as json_file:
        Hyper_Parameter_Dictionary = json.load(json_file)

    if Hyper_Parameter_Dictionary["number_of_models"] == 1:
        total_MSE_over_training = np.load("Output_Data/total_MSE_over_training.npy")
        MSE_boundary_conditions_over_training = np.load("Output_Data/MSE_boundary_conditions_over_training.npy")
        MSE_initial_conditions_over_training = np.load("Output_Data/MSE_initial_conditions_over_training.npy")
        MSE_symbolic_functions_over_training = np.load("Output_Data/MSE_symbolic_functions_over_training.npy")
        Relative_L2_Error_h_over_training = np.load("Output_Data/Relative_L2_Error_h_over_training.npy")
        Relative_L2_Error_u_over_training = np.load("Output_Data/Relative_L2_Error_u_over_training.npy")
        MSE_initial_condition_u_over_training = np.load("Output_Data/MSE_initial_condition_u_over_training.npy")
        MSE_initial_condition_h_over_training = np.load("Output_Data/MSE_initial_condition_h_over_training.npy")
        MSE_symbolic_function_u_over_training = np.load("Output_Data/MSE_symbolic_function_u_over_training.npy")
        MSE_symbolic_function_h_over_training = np.load("Output_Data/MSE_symbolic_function_h_over_training.npy")
        MSE_lower_boundary_condition_u_over_training = np.load(
            "Output_Data/MSE_lower_boundary_condition_u_over_training.npy")
        MSE_upper_boundary_condition_u_over_training = np.load(
            "Output_Data/MSE_upper_boundary_condition_u_over_training.npy")

    else:

        with open("Output_Data/0/Hyper_Parameter_Dictionary.json") as json_file:
            Hyper_Parameter_Dictionary = json.load(json_file)
        total_MSE_over_training = np.load("Output_Data/0/total_MSE_over_training.npy")
        MSE_boundary_conditions_over_training = np.load("Output_Data/0/MSE_boundary_conditions_over_training.npy")
        MSE_initial_conditions_over_training = np.load("Output_Data/0/MSE_initial_conditions_over_training.npy")
        MSE_symbolic_functions_over_training = np.load("Output_Data/0/MSE_symbolic_functions_over_training.npy")
        Relative_L2_Error_h_over_training = np.load("Output_Data/0/Relative_L2_Error_h_over_training.npy")
        Relative_L2_Error_u_over_training = np.load("Output_Data/0/Relative_L2_Error_u_over_training.npy")
        MSE_initial_condition_u_over_training = np.load("Output_Data/0/MSE_initial_condition_u_over_training.npy")
        MSE_initial_condition_h_over_training = np.load("Output_Data/0/MSE_initial_condition_h_over_training.npy")
        MSE_symbolic_function_u_over_training = np.load("Output_Data/0/MSE_symbolic_function_u_over_training.npy")
        MSE_symbolic_function_h_over_training = np.load("Output_Data/0/MSE_symbolic_function_h_over_training.npy")

        MSE_lower_boundary_condition_u_over_training = np.load(
            "Output_Data/0/MSE_lower_boundary_condition_u_over_training.npy")
        MSE_upper_boundary_condition_u_over_training = np.load(
            "Output_Data/0/MSE_upper_boundary_condition_u_over_training.npy")

        for i in range(1, Hyper_Parameter_Dictionary["number_of_models"]):
            total_MSE_over_training = np.concatenate(
                (total_MSE_over_training, np.load("Output_Data/" + str(i) + "/total_MSE_over_training.npy")), axis=0)
            MSE_boundary_conditions_over_training = np.concatenate(
                (MSE_boundary_conditions_over_training,
                 np.load("Output_Data/" + str(i) + "/MSE_boundary_conditions_over_training.npy")), axis=0)
            MSE_initial_conditions_over_training = np.concatenate(
                (MSE_initial_conditions_over_training,
                 np.load("Output_Data/" + str(i) + "/MSE_initial_conditions_over_training.npy")), axis=0)
            MSE_symbolic_functions_over_training = np.concatenate(
                (MSE_symbolic_functions_over_training,
                 np.load("Output_Data/" + str(i) + "/MSE_symbolic_functions_over_training.npy")), axis=0)
            Relative_L2_Error_h_over_training = np.concatenate(
                (Relative_L2_Error_h_over_training,
                 np.load("Output_Data/" + str(i) + "/Relative_L2_Error_h_over_training.npy")), axis=0)
            Relative_L2_Error_u_over_training = np.concatenate(
                (Relative_L2_Error_u_over_training,
                 np.load("Output_Data/" + str(i) + "/Relative_L2_Error_u_over_training.npy")), axis=0)
            MSE_initial_condition_u_over_training = np.concatenate(
                (MSE_initial_condition_u_over_training,
                 np.load("Output_Data/" + str(i) + "/MSE_initial_condition_u_over_training.npy")), axis=0)
            MSE_initial_condition_h_over_training = np.concatenate(
                (MSE_initial_condition_h_over_training,
                 np.load("Output_Data/" + str(i) + "/MSE_initial_condition_h_over_training.npy")), axis=0)
            MSE_symbolic_function_u_over_training = np.concatenate(
                (MSE_symbolic_function_u_over_training,
                 np.load("Output_Data/" + str(i) + "/MSE_symbolic_function_u_over_training.npy")), axis=0)
            MSE_symbolic_function_h_over_training = np.concatenate(
                (MSE_symbolic_function_h_over_training,
                 np.load("Output_Data/" + str(i) + "/MSE_symbolic_function_h_over_training.npy")), axis=0)

            MSE_lower_boundary_condition_u_over_training = np.concatenate(
                (MSE_lower_boundary_condition_u_over_training,
                 np.load("Output_Data/" + str(i) + "/MSE_lower_boundary_condition_u_over_training.npy")), axis=0)
            MSE_upper_boundary_condition_u_over_training = np.concatenate(
                (MSE_upper_boundary_condition_u_over_training,
                 np.load("Output_Data/" + str(i) + "/MSE_upper_boundary_condition_u_over_training.npy")), axis=0)


    # sns.set()
    plt.figure()
    training_steps = range(len(total_MSE_over_training))
    plt.plot(training_steps, total_MSE_over_training, label=r"$\mathcal{L}_{total}$", linewidth=0.08,
                     alpha=0.7)
    plt.plot(training_steps, MSE_boundary_conditions_over_training, label=r"$\mathcal{L}_{BC}$", linewidth=0.08,
             alpha=0.7)
    plt.plot(training_steps, MSE_initial_conditions_over_training, label=r"$\mathcal{L}_{IC}$", linewidth=0.08,
             alpha=0.7)
    plt.plot(training_steps, MSE_symbolic_functions_over_training, label=r"$\mathcal{L}_{PDE}$", linewidth=0.08,
             alpha=0.7)
    plt.plot(training_steps, Relative_L2_Error_h_over_training, label=r"$\mathcal{L}_{num,\zeta}$", linewidth=0.08,
             alpha=0.7)
    plt.plot(training_steps, Relative_L2_Error_u_over_training, label=r"$\mathcal{L}_{num,u}$", linewidth=0.08,
             alpha=0.7)
    # plt.title("M" + str(model_number) + " Training Curve over " + str(
    #    training_steps[-1]) + " epochs")
    # plt.ylabel(r"$MSE$ = $w_{PDE}$ * $MES_{PDE}$ + $w_{BC}$ * $MSE_{BC}$ + $w_{IC}$ * $MSE_{IC}$")
    plt.xlabel("epochs")
    plt.yscale("log")
    leg = plt.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    plt.grid()
    plt.savefig("Training_Curve.png", format="png")
    plt.show()

    training_steps_ind = np.arange(len(MSE_initial_condition_u_over_training))
    plt.plot(training_steps_ind, MSE_initial_condition_u_over_training,
             label=r"$\mathcal{L}_{IC,u}$", linewidth=0.08, alpha=0.7)
    #plt.plot(training_steps_ind, MSE_initial_condition_h_over_training,
    #         label=r"$\mathcal{L}_{IC,\zeta}$", linewidth=0.08, alpha=0.7)
    plt.plot(training_steps_ind, MSE_symbolic_function_u_over_training,
             label=r"$\mathcal{L}_{PDE,u}$", linewidth=0.08, alpha=0.7)
    plt.plot(training_steps_ind, MSE_symbolic_function_h_over_training,
             label=r"$\mathcal{L}_{PDE,\zeta}$", linewidth=0.08, alpha=0.7)

    plt.plot(training_steps_ind, MSE_lower_boundary_condition_u_over_training,
             label=r"$\mathcal{L}_{BC1}$", linewidth=0.08, alpha=0.7)
    plt.plot(training_steps_ind, MSE_upper_boundary_condition_u_over_training,
             label=r"$\mathcal{L}_{BC2}$", linewidth=0.08, alpha=0.7)

    # plt.title("M" + str(model_number) + " Separate u and h MSEs over " + str(training_steps[-1]) + " epochs")
    plt.xlabel("epochs")
    plt.yscale("log")
    leg = plt.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    plt.grid()
    plt.savefig("Separate_Losses.png", format="png")
    plt.show()


# Plot_Results generates 3D surface plots of the Network Output, the Closed Form Solution and the Approximation Error
def Plot_Results():

    with open("Output_Data/Hyper_Parameter_Dictionary.json") as json_file:
        Hyper_Parameter_Dictionary = json.load(json_file)
    non_dimensionalization = Hyper_Parameter_Dictionary["non_dimensionalization"]

    # define boundaries
    minimum_time = Hyper_Parameter_Dictionary["minimum_time"]
    maximum_time = Hyper_Parameter_Dictionary["maximum_time"]
    minimum_x = Hyper_Parameter_Dictionary["minimum_x"]
    maximum_x = Hyper_Parameter_Dictionary["maximum_x"]

    if Hyper_Parameter_Dictionary["non_dimensionalization"] is True:
        minimum_time = minimum_time * Hyper_Parameter_Dictionary["time_scale"]
        maximum_time = maximum_time * Hyper_Parameter_Dictionary["time_scale"]
        minimum_x = minimum_x * Hyper_Parameter_Dictionary["horizontal_length_scale"]
        maximum_x = maximum_x * Hyper_Parameter_Dictionary["horizontal_length_scale"]

    if Hyper_Parameter_Dictionary["number_of_models"] == 1:

        # import Test-Grid and Test-Mesh-Grid from Network
        if non_dimensionalization == True:
            time_mesh_grid = np.load("Output_Data/dimensional_time_mesh_grid.npy")
            x_mesh_grid = np.load("Output_Data/dimensional_x_mesh_grid.npy")
        else:
            time_mesh_grid = np.load("Output_Data/time_mesh_grid.npy")
            x_mesh_grid = np.load("Output_Data/x_mesh_grid.npy")

        dimensional_zeta_solution_time_mesh_grid = np.load("Output_Data/dimensional_zeta_solution_time_mesh_grid.npy")
        dimensional_zeta_solution_x_mesh_grid = np.load("Output_Data/dimensional_zeta_solution_x_mesh_grid.npy")
        dimensional_u_solution_time_mesh_grid = np.load("Output_Data/dimensional_u_solution_time_mesh_grid.npy")
        dimensional_u_solution_x_mesh_grid = np.load("Output_Data/dimensional_u_solution_x_mesh_grid.npy")

        # compute network output
        dimensional_network_output_h_values = np.load("Output_Data/dimensional_network_output_h_values.npy")
        dimensional_network_output_u_values = np.load("Output_Data/dimensional_network_output_u_values.npy")

        # import solutions for h and u
        exact_solution_u_values = np.load("Output_Data/exact_solution_u_values.npy")
        exact_solution_h_values = np.load("Output_Data/exact_solution_h_values.npy")

        abs_error_h_values = np.load("Output_Data/abs_error_h_values.npy")
        abs_error_u_values = np.load("Output_Data/abs_error_u_values.npy")

    else:

        # import Test-Grid and Test-Mesh-Grid from Network
        if non_dimensionalization == True:
            time_mesh_grid = np.load("Output_Data/0/dimensional_time_mesh_grid.npy")
            x_mesh_grid = np.load("Output_Data/0/dimensional_x_mesh_grid.npy")
        else:
            time_mesh_grid = np.load("Output_Data/0/time_mesh_grid.npy")
            x_mesh_grid = np.load("Output_Data/0/x_mesh_grid.npy")

        dimensional_zeta_solution_time_mesh_grid = np.load("Output_Data/0/dimensional_zeta_solution_time_mesh_grid.npy")
        dimensional_zeta_solution_x_mesh_grid = np.load("Output_Data/0/dimensional_zeta_solution_x_mesh_grid.npy")
        dimensional_u_solution_time_mesh_grid = np.load("Output_Data/0/dimensional_u_solution_time_mesh_grid.npy")
        dimensional_u_solution_x_mesh_grid = np.load("Output_Data/0/dimensional_u_solution_x_mesh_grid.npy")

        # compute network output
        dimensional_network_output_h_values = np.load("Output_Data/0/dimensional_network_output_h_values.npy")
        dimensional_network_output_u_values = np.load("Output_Data/0/dimensional_network_output_u_values.npy")

        # import solutions for h and u
        exact_solution_u_values = np.load("Output_Data/0/exact_solution_u_values.npy")
        exact_solution_h_values = np.load("Output_Data/0/exact_solution_h_values.npy")

        abs_error_h_values = np.load("Output_Data/0/abs_error_h_values.npy")
        abs_error_u_values = np.load("Output_Data/0/abs_error_u_values.npy")

        for i in range(1, Hyper_Parameter_Dictionary["number_of_models"]):

            # import Test-Grid and Test-Mesh-Grid from Network
            if non_dimensionalization == True:
                time_mesh_grid = np.concatenate((time_mesh_grid,
                     np.load("Output_Data/" + str(i) + "/dimensional_time_mesh_grid.npy")), axis=1)
                x_mesh_grid = np.concatenate((x_mesh_grid,
                    np.load("Output_Data/" + str(i) + "/dimensional_x_mesh_grid.npy")), axis=1)
            else:
                time_mesh_grid = np.concatenate((time_mesh_grid,
                                                 np.load("Output_Data/" + str(i) + "/time_mesh_grid.npy")),
                                                axis=1)
                x_mesh_grid = np.concatenate((x_mesh_grid,
                                              np.load("Output_Data/" + str(i) + "/x_mesh_grid.npy")),
                                             axis=1)

            dimensional_zeta_solution_time_mesh_grid = np.concatenate((dimensional_zeta_solution_time_mesh_grid,
                np.load("Output_Data/" + str(i) + "/dimensional_zeta_solution_time_mesh_grid.npy")), axis=1)
            dimensional_zeta_solution_x_mesh_grid = np.concatenate((dimensional_zeta_solution_x_mesh_grid,
                np.load("Output_Data/" + str(i) + "/dimensional_zeta_solution_x_mesh_grid.npy")), axis=1)
            dimensional_u_solution_time_mesh_grid = np.concatenate((dimensional_u_solution_time_mesh_grid,
                np.load("Output_Data/" + str(i) + "/dimensional_u_solution_time_mesh_grid.npy")), axis=1)
            dimensional_u_solution_x_mesh_grid = np.concatenate((dimensional_u_solution_x_mesh_grid,
                np.load("Output_Data/" + str(i) + "/dimensional_u_solution_x_mesh_grid.npy")), axis=1)

            # compute network output
            dimensional_network_output_h_values = np.concatenate((dimensional_network_output_h_values,
                                        np.load("Output_Data/" + str(i) + "/dimensional_network_output_h_values.npy")),
                                        axis=1)
            dimensional_network_output_u_values = np.concatenate((dimensional_network_output_u_values,
                                                                  np.load("Output_Data/" + str(
                                                                      i) + "/dimensional_network_output_u_values.npy")),
                                                                 axis=1)

            # import solutions for h and u
            exact_solution_u_values = np.concatenate((exact_solution_u_values,
                                                                  np.load("Output_Data/" + str(
                                                                      i) + "/exact_solution_u_values.npy")),
                                                                 axis=1)
            exact_solution_h_values = np.concatenate((exact_solution_h_values,
                                                                  np.load("Output_Data/" + str(
                                                                      i) + "/exact_solution_h_values.npy")),
                                                                 axis=1)
            abs_error_u_values = np.concatenate((abs_error_u_values,
                                                      np.load("Output_Data/" + str(
                                                          i) + "/abs_error_u_values.npy")),
                                                     axis=1)
            abs_error_h_values = np.concatenate((abs_error_h_values,
                                                      np.load("Output_Data/" + str(
                                                          i) + "/abs_error_h_values.npy")),
                                                     axis=1)

    # 3D Surface Plot of the Network Output for h after Training
    plt.figure()
    plt.pcolor(dimensional_zeta_solution_time_mesh_grid / 60. / 60.,
               0.001 * dimensional_zeta_solution_x_mesh_grid,
               dimensional_network_output_h_values,
               cmap="coolwarm", linewidth=0, antialiased=False)
    plt.title(r"Network Output $\zeta(x,t)$")
    plt.xlabel("t [h]")
    plt.ylabel("x [km]")
    # plt.xlim(minimum_time, maximum_time)
    # plt.ylim(minimum_x, maximum_x)
    plt.colorbar(label=r"$\zeta$ [m]")
    plt.savefig("Network_Output_zeta.png", format="png")
    plt.show()

    # 3D Surface Plot of the Network Output for u after Training
    plt.figure()
    plt.pcolor(dimensional_u_solution_time_mesh_grid / 60. / 60.,
               0.001 * dimensional_u_solution_x_mesh_grid,
               dimensional_network_output_u_values,
               cmap="coolwarm", linewidth=0, antialiased=False)
    plt.title(r"Network Output $u(x,t)$")
    plt.xlabel("t [h]")
    plt.ylabel("x [km]")
    # plt.xlim(minimum_time, maximum_time)
    # plt.ylim(minimum_x, maximum_x)
    plt.colorbar(label="u [m/s]")
    plt.savefig("Network_Output_u.png", format="png")
    plt.show()

    # 3D Surface Plot of the Numerical Solution for h
    plt.figure()
    plt.pcolor(dimensional_zeta_solution_time_mesh_grid / 60. / 60.,
               0.001 * dimensional_zeta_solution_x_mesh_grid,
               exact_solution_h_values,
               cmap="coolwarm", linewidth=0, antialiased=False)
    plt.title(r"Numerical Solution $\zeta(x,t)$")
    plt.xlabel("t [h]")
    plt.ylabel("x [km]")
    # plt.xlim(minimum_time, maximum_time)
    # plt.ylim(minimum_x, maximum_x)
    plt.colorbar(label=r"$\zeta$ [m]")
    plt.savefig("Numerical_Solution_zeta.png", format="png")
    plt.show()

    ## 3D Surface Plot of the Numerical Solution for u
    plt.figure()
    plt.pcolor(dimensional_u_solution_time_mesh_grid / 60. / 60.,
               0.001 * dimensional_u_solution_x_mesh_grid,
               exact_solution_u_values,
               cmap="coolwarm", linewidth=0, antialiased=False)
    plt.title(r"Numerical Solution $u(x,t)$")
    plt.xlabel("t [h]")
    plt.ylabel("x [km]")
    # plt.xlim(minimum_time, maximum_time)
    # plt.ylim(minimum_x, maximum_x)
    plt.colorbar(label="u [m/s]")
    plt.savefig("Numerical_Solution_u.png", format="png")
    plt.show()

    ## 3D Surface Plot of the Abs Error in h
    plt.figure()
    plt.pcolor(dimensional_zeta_solution_time_mesh_grid / 60. / 60.,
               0.001 * dimensional_zeta_solution_x_mesh_grid,
               abs_error_h_values,
               cmap="coolwarm", linewidth=0, antialiased=False)
    plt.title(r"Absolute Error in $\zeta(x,t)$")
    plt.xlabel("t [h]")
    plt.ylabel("x [km]")
    # plt.xlim(minimum_time, maximum_time)
    # plt.ylim(minimum_x, maximum_x)
    plt.colorbar(label=r"$\zeta$ [m]")
    plt.savefig("Absolute_Error_zeta.png", format="png")
    plt.show()

    ## 3D Surface Plot of the Abs Error in u
    plt.figure()
    plt.pcolor(dimensional_u_solution_time_mesh_grid / 60. / 60.,
               0.001 * dimensional_u_solution_x_mesh_grid,
               abs_error_u_values,
               cmap="coolwarm", linewidth=0, antialiased=False)
    plt.title(r"Absolute Error in $u(x,t)$")
    plt.xlabel("t [h]")
    plt.ylabel("x [km]")
    # plt.xlim(minimum_time, maximum_time)
    # plt.ylim(minimum_x, maximum_x)
    plt.colorbar(label="u [m/s]")
    plt.savefig("Absolute_Error_u.png", format="png")
    plt.show()


def Plot_Initial_Conditions():
    with open("Output_Data/Hyper_Parameter_Dictionary.json") as json_file:
        Hyper_Parameter_Dictionary = json.load(json_file)

    if Hyper_Parameter_Dictionary["number_of_models"] == 1:
        x = np.load("Output_Data/dimensional_x_mesh_grid.npy")[:, 0]
        true_initial_condition_h_values = np.load("Output_Data/true_initial_condition_h_values.npy")
        network_output_h_initial_conditions = np.load("Output_Data/network_output_h_initial_conditions.npy")
        true_initial_condition_u_values = np.load("Output_Data/true_initial_condition_u_values.npy")
        network_output_u_initial_conditions = np.load("Output_Data/network_output_u_initial_conditions.npy")
    else:
        x = np.load("Output_Data/0/dimensional_x_mesh_grid.npy")[:, 0]
        true_initial_condition_h_values = np.load("Output_Data/0/true_initial_condition_h_values.npy")
        network_output_h_initial_conditions = np.load("Output_Data/0/network_output_h_initial_conditions.npy")
        true_initial_condition_u_values = np.load("Output_Data/0/true_initial_condition_u_values.npy")
        network_output_u_initial_conditions = np.load("Output_Data/0/network_output_u_initial_conditions.npy")

    # Plot for h
    plt.figure()
    plt.plot(0.001 * x, network_output_h_initial_conditions, "-", label=r"$\zeta_{NN}$")
    plt.plot(0.001 * x, true_initial_condition_h_values, "-", label=r"$\zeta_{num}$")
    # plt.title("M" + str(Hyper_Parameter_Dictionary["model_number"]) + " Target initial condition and respective network output for h")
    plt.xlabel("x [km] for t = 0")
    plt.grid()
    plt.legend()
    plt.savefig("Initial_Condition_zeta.png", format="png")
    plt.show()

    # Plot for u
    plt.figure()
    plt.plot(0.001 * x, network_output_u_initial_conditions, "-", label=r"$u_{NN}$")
    plt.plot(0.001 * x, true_initial_condition_u_values, "-", label=r"$u_{num}$")
    # plt.title("M" + str(Hyper_Parameter_Dictionary["model_number"]) + " Target initial condition and respective network output for u")
    plt.xlabel("x [km] for t = 0")
    plt.grid()
    plt.legend()
    plt.savefig("Initial_Condition_u.png", format="png")
    plt.show()


def Plot_Boundary_Conditions():

    with open("Output_Data/Hyper_Parameter_Dictionary.json") as json_file:
        Hyper_Parameter_Dictionary = json.load(json_file)

    minimum_x = Hyper_Parameter_Dictionary["minimum_x"]
    maximum_x = Hyper_Parameter_Dictionary["maximum_x"]

    if Hyper_Parameter_Dictionary["non_dimensionalization"] == True:
        minimum_x = minimum_x * Hyper_Parameter_Dictionary["horizontal_length_scale"]
        maximum_x = maximum_x * Hyper_Parameter_Dictionary["horizontal_length_scale"]

    if Hyper_Parameter_Dictionary["number_of_models"] == 1:
        t = np.load("Output_Data/dimensional_time_mesh_grid.npy")[0, :]
        # define network output and target boundary condition
        true_lower_boundary_condition_u_values = np.load("Output_Data/true_lower_boundary_condition_u_values.npy")
        true_upper_boundary_condition_u_values = np.load("Output_Data/true_upper_boundary_condition_u_values.npy")
        network_output_u_lower_boundary_condition = np.load("Output_Data/network_output_u_lower_boundary_condition.npy")
        network_output_u_upper_boundary_condition = np.load("Output_Data/network_output_u_upper_boundary_condition.npy")
    else:
        t = np.load("Output_Data/0/dimensional_time_mesh_grid.npy")[0, :]
        # define network output and target boundary condition
        true_lower_boundary_condition_u_values = np.load("Output_Data/0/true_lower_boundary_condition_u_values.npy")
        true_upper_boundary_condition_u_values = np.load("Output_Data/0/true_upper_boundary_condition_u_values.npy")
        network_output_u_lower_boundary_condition = np.load(
            "Output_Data/0/network_output_u_lower_boundary_condition.npy")
        network_output_u_upper_boundary_condition = np.load(
            "Output_Data/0/network_output_u_upper_boundary_condition.npy")

        for i in range(1, Hyper_Parameter_Dictionary["number_of_models"]):
            t = np.concatenate((t, np.load("Output_Data/" + str(i) + "/dimensional_time_mesh_grid.npy")[0, :]), axis=0)
            true_lower_boundary_condition_u_values = np.concatenate(
                (true_lower_boundary_condition_u_values,
                 np.load("Output_Data/" + str(i) + "/true_lower_boundary_condition_u_values.npy")), axis=0)
            true_upper_boundary_condition_u_values = np.concatenate(
                (true_upper_boundary_condition_u_values,
                 np.load("Output_Data/" + str(i) + "/true_upper_boundary_condition_u_values.npy")), axis=0)
            network_output_u_lower_boundary_condition = np.concatenate(
                (network_output_u_lower_boundary_condition,
                 np.load("Output_Data/" + str(i) + "/network_output_u_lower_boundary_condition.npy")), axis=0)
            network_output_u_upper_boundary_condition = np.concatenate(
                (network_output_u_upper_boundary_condition,
                 np.load("Output_Data/" + str(i) + "/network_output_u_upper_boundary_condition.npy")), axis=0)


        # Plot for u at lower boundary
        plt.figure()
        plt.plot(t / 60. / 60., network_output_u_lower_boundary_condition, "-", label=r"$u_{NN}$")
        plt.plot(t / 60. / 60., true_lower_boundary_condition_u_values, "-", label=r"$u_{num}$")
        # plt.title("M" + str(Hyper_Parameter_Dictionary["model_number"])
        #          + " Target lower boundary condition and respective network output for u")
        plt.xlabel("t [h] for x = -1000 km")
        plt.grid()
        plt.legend()
        plt.savefig("Lower_Boundary_Condition_u.png", format="png")
        plt.show()

        # Plot for u at lower boundary
        plt.figure()
        plt.plot(t / 60. / 60., network_output_u_upper_boundary_condition, "-", label=r"$u_{NN}$")
        plt.plot(t / 60. / 60., true_upper_boundary_condition_u_values, "-", label=r"$u_{num}$")
        # plt.title("M" + str(Hyper_Parameter_Dictionary["model_number"])
        #          + " Target upper boundary condition and respective network output for u")
        plt.xlabel("t [h] for x = 1000 km")
        plt.grid()
        plt.legend()
        plt.savefig("Upper_Boundary_Condition_u.png", format="png")
        plt.show()


def Plot_PDE_Losses():

    with open("Output_Data/Hyper_Parameter_Dictionary.json") as json_file:
        Hyper_Parameter_Dictionary = json.load(json_file)

    if Hyper_Parameter_Dictionary["number_of_models"] == 1:
        time_mesh_grid = np.load("Output_Data/dimensional_time_mesh_grid.npy")
        x_mesh_grid = np.load("Output_Data/dimensional_x_mesh_grid.npy")
        PDE_Loss_u = np.load("Output_Data/PDE_Loss_u.npy")
        PDE_Loss_h = np.load("Output_Data/PDE_Loss_h.npy")

    else:
        time_mesh_grid = np.load("Output_Data/0/dimensional_time_mesh_grid.npy")
        x_mesh_grid = np.load("Output_Data/0/dimensional_x_mesh_grid.npy")
        PDE_Loss_u = np.load("Output_Data/0/PDE_Loss_u.npy")
        PDE_Loss_h = np.load("Output_Data/0/PDE_Loss_h.npy")
        for i in range(1, Hyper_Parameter_Dictionary["number_of_models"]):
            time_mesh_grid = np.concatenate(
                (time_mesh_grid, np.load("Output_Data/" + str(i) + "/dimensional_time_mesh_grid.npy")), axis=1)
            x_mesh_grid = np.concatenate(
                (x_mesh_grid, np.load("Output_Data/" + str(0) + "/dimensional_x_mesh_grid.npy")), axis=1)
            PDE_Loss_u = np.concatenate(
                (PDE_Loss_u, np.load("Output_Data/" + str(i) + "/PDE_Loss_u.npy")), axis=1)
            PDE_Loss_h = np.concatenate(
                (PDE_Loss_h, np.load("Output_Data/" + str(i) + "/PDE_Loss_h.npy")), axis=1)

    ## 3D Surface Plot of the PDE Loss in h
    plt.figure()
    plt.pcolor(time_mesh_grid / 60. / 60., 0.001 * x_mesh_grid, PDE_Loss_h,
               cmap="coolwarm", linewidth=0, antialiased=False)
    plt.title(r"PDE Loss In Continuity Equation")
    plt.xlabel("t [h]")
    plt.ylabel("x [km]")
    # plt.xlim(minimum_time, maximum_time)
    # plt.ylim(minimum_x, maximum_x)
    plt.colorbar(label=r"$\mathcal{L}_{PDE,\zeta}$")
    plt.savefig("PDE_Loss_Cont_Eq.png", format="png")
    plt.show()

    ## 3D Surface Plot of the PDE Loss in u
    plt.figure()
    plt.pcolor(time_mesh_grid / 60. / 60., 0.001 * x_mesh_grid, PDE_Loss_u,
               cmap="coolwarm", linewidth=0, antialiased=False)
    plt.title(r"PDE Loss In Momentum Equation")
    plt.xlabel("t [h]")
    plt.ylabel("x [km]")
    # plt.xlim(minimum_time, maximum_time)
    # plt.ylim(minimum_x, maximum_x)
    plt.colorbar(label=r"$\mathcal{L}_{PDE,u}$")
    plt.savefig("PDE_Loss_Mom_Eq.png", format="png")
    plt.show()

def Plot_Solution_Over_Training(epochs=10000):

    with open("Output_Data/Hyper_Parameter_Dictionary.json") as json_file:
        Hyper_Parameter_Dictionary = json.load(json_file)

    network_output_h_over_training = (np.load("Output_Data/network_output_h_over_training.npy")
                                     * Hyper_Parameter_Dictionary["vertical_length_scale"])
    network_output_u_over_training = (np.load("Output_Data/network_output_u_over_training.npy")
                                      * Hyper_Parameter_Dictionary["horizontal_length_scale"]
                                      / Hyper_Parameter_Dictionary["time_scale"])
    dimensional_x_mesh_grid = np.load("Output_Data/dimensional_x_mesh_grid.npy")
    dimensional_time_mesh_grid = np.load("Output_Data/dimensional_time_mesh_grid.npy")

    os.mkdir("Output_Over_Training")
    os.chdir("Output_Over_Training")
    
    os.mkdir("zeta")
    os.chdir("zeta")
    
    for i in range(0, int(epochs/100. + 10), 10):
        network_output_h = network_output_h_over_training[i]
        # 3D Surface Plot of the Network Output for h after Training
        plt.figure()
        plt.pcolor(dimensional_time_mesh_grid / 60. / 60., 0.001 * dimensional_x_mesh_grid,
                   network_output_h, cmap="coolwarm", linewidth=0, antialiased=False, vmax=1.0, vmin=0.0)
        plt.title(r"Network Output $\zeta(x,t)$ after " + str(int(i * epochs / 100.)) + " epochs")
        plt.xlabel("t [h]")
        plt.ylabel("x [km]")
        plt.colorbar(label=r"$\zeta$ [m]")
        plt.savefig("Network_Output_zeta_" + str(int(i * epochs / 100.)) + ".png", format="png")
        plt.close()
    
    os.chdir("..")
    os.mkdir("u")
    os.chdir("u")
    
    for i in range(0, int(epochs/100.) + 10, 10):
        network_output_u = network_output_u_over_training[i]
        # 3D Surface Plot of the Network Output for u after Training
        plt.figure()
        plt.pcolor(dimensional_time_mesh_grid / 60. / 60., 0.001 * dimensional_x_mesh_grid,
                   network_output_u, cmap="coolwarm", linewidth=0, antialiased=False, vmax=0.15, vmin=-0.15)
        plt.title(r"Network Output $u(x,t)$ after " + str(int(i * epochs / 100.)) + " epochs")
        plt.xlabel("t [h]")
        plt.ylabel("x [km]")
        plt.colorbar(label="u [m/s]")
        plt.savefig("Network_Output_zeta_" + str(int(i * epochs / 100.)) + ".png", format="png")
        plt.close()


def Plot_PDE_Loss_Over_Training(epochs=10000):
    with open("Output_Data/Hyper_Parameter_Dictionary.json") as json_file:
        Hyper_Parameter_Dictionary = json.load(json_file)

    symbolic_function_h_over_training = (np.load("Output_Data/symbolic_function_h_over_training.npy")
                                      * Hyper_Parameter_Dictionary["vertical_length_scale"])
    symbolic_function_u_over_training = (np.load("Output_Data/symbolic_function_u_over_training.npy")
                                      * Hyper_Parameter_Dictionary["horizontal_length_scale"]
                                      / Hyper_Parameter_Dictionary["time_scale"])
    dimensional_x_mesh_grid = np.load("Output_Data/dimensional_x_mesh_grid.npy")
    dimensional_time_mesh_grid = np.load("Output_Data/dimensional_time_mesh_grid.npy")

    os.mkdir("Symbolic_Function_Over_Training")
    os.chdir("Symbolic_Function_Over_Training")

    os.mkdir("zeta")
    os.chdir("zeta")

    for i in range(0, int(epochs / 100. + 10), 10):
        symbolic_function_h = np.log10(abs(symbolic_function_h_over_training[i]))
        # 3D Surface Plot of the Network Output for h after Training
        plt.figure()
        plt.pcolor(dimensional_time_mesh_grid / 60. / 60., 0.001 * dimensional_x_mesh_grid,
                   symbolic_function_h, cmap="viridis", linewidth=0, antialiased=False, vmax=-2.0, vmin=-6.0)
        plt.title(r"Log10 of absolute continuum equation loss after " + str(int(i * epochs / 100.)) + " epochs")
        plt.xlabel("t [h]")
        plt.ylabel("x [km]")
        plt.colorbar(label=r"$\zeta$ [m]")
        plt.savefig("Symbolic_Function_zeta_" + str(int(i * epochs / 100.)) + ".png", format="png")
        plt.close()

    os.chdir("..")
    os.mkdir("u")
    os.chdir("u")

    for i in range(0, int(epochs / 100.) + 10, 10):
        symbolic_function_u = np.log10(abs(symbolic_function_u_over_training[i]))
        # 3D Surface Plot of the Network Output for u after Training
        plt.figure()
        plt.pcolor(dimensional_time_mesh_grid / 60. / 60., 0.001 * dimensional_x_mesh_grid,
                   symbolic_function_u, cmap="viridis", linewidth=0, antialiased=False, vmax=-2.0, vmin=-6.0)
        plt.title(r"Log10 of absolute momentum equation loss after " + str(int(i * epochs / 100.)) + " epochs")
        plt.xlabel("t [h]")
        plt.ylabel("x [km]")
        plt.colorbar(label="u [m/s]")
        plt.savefig("Symbolic_Function_zeta_" + str(int(i * epochs / 100.)) + ".png", format="png")
        plt.close()
