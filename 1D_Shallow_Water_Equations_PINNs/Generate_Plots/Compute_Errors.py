# import external libraries
import json
import numpy as np

# model information
with open("Hyper_Parameter_Dictionary.json") as json_file:
    Hyper_Parameter_Dictionary = json.load(json_file)

print(str(Hyper_Parameter_Dictionary["number_of_models"]) + " Model(s) With Model Size: "
      + str(Hyper_Parameter_Dictionary["number_of_layers"]) + "x"
      + str(Hyper_Parameter_Dictionary["neurons_per_layer"]))

print("Sampling Points: [" + str(Hyper_Parameter_Dictionary["boundary_condition_batch_size"]) + ", "
      + str(Hyper_Parameter_Dictionary["initial_condition_batch_size"]) + ", "
      + str(Hyper_Parameter_Dictionary["symbolic_function_batch_size"]) + "]")

# third of Time Interval
if Hyper_Parameter_Dictionary["number_of_models"] == 1:
    Abs_Error_u = np.load("abs_error_u_values.npy")
    Abs_Error_h = np.load("abs_error_h_values.npy")
    Numerical_Sol_u = np.load("exact_solution_u_values.npy")
    Numerical_Sol_h = np.load("exact_solution_h_values.npy")
else:
    Abs_Error_u = np.load("0/abs_error_u_values.npy")
    Abs_Error_h = np.load("0/abs_error_h_values.npy")
    Numerical_Sol_u = np.load("0/exact_solution_u_values.npy")
    Numerical_Sol_h = np.load("0/exact_solution_h_values.npy")
    for i in range(1, Hyper_Parameter_Dictionary["number_of_models"]):
        Abs_Error_u = np.concatenate((Abs_Error_u, np.load(str(i) + "/abs_error_u_values.npy")), axis=1)
        Abs_Error_h = np.concatenate((Abs_Error_h, np.load(str(i) + "/abs_error_h_values.npy")), axis=1)
        Numerical_Sol_u = np.concatenate((Numerical_Sol_u, np.load(str(i) + "/exact_solution_u_values.npy")), axis=1)
        Numerical_Sol_h = np.concatenate((Numerical_Sol_h, np.load(str(i) + "/exact_solution_h_values.npy")), axis=1)
    print("Size Abs_Error_u: ", Abs_Error_u.shape)
    print("Size Abs_Error_zeta: ", Abs_Error_h.shape)
    print("Size Numerical_Sol_u: ", Numerical_Sol_u.shape)
    print("Size Numerical_Sol_zeta: ", Numerical_Sol_h.shape)

Relative_L2_Error_u = np.sqrt((Abs_Error_u ** 2.).sum()) / np.sqrt((Numerical_Sol_u ** 2.).sum())
Relative_L2_Error_h = np.sqrt((Abs_Error_h ** 2.).sum()) / np.sqrt((Numerical_Sol_h ** 2.).sum())

Relative_Max_Error_u = np.max(Abs_Error_u) / np.max(abs(Numerical_Sol_u))
Relative_Max_Error_h = np.max(Abs_Error_h) / np.max(abs(Numerical_Sol_h))

print("Errors on Full Time Interval:")
print("Rel L2(zeta): " + str(np.round(100.0 * Relative_L2_Error_h, 2)) + "%")
print("Rel L2(u): " + str(np.round(100.0 * Relative_L2_Error_u, 2)))
print("Rel L-infty(zeta): " + str(np.round(100.0 * Relative_Max_Error_h, 2)) + "%")
print("Rel L-infty(u): " + str(np.round(100.0 * Relative_Max_Error_u, 2)) + "%")

if Hyper_Parameter_Dictionary["number_of_models"] == 1:
    computation_time = Hyper_Parameter_Dictionary["computation_time"]
    print("Computation Time: " + str(np.round(computation_time / 60. / 60., 2)) + " h")

else:
    computation_time = 0.
    for i in range(Hyper_Parameter_Dictionary["number_of_models"]):
        with open(str(i) + "/Hyper_Parameter_Dictionary.json") as json_file:
            Hyper_Parameter_Dictionary = json.load(json_file)
        computation_time += Hyper_Parameter_Dictionary["computation_time"]
    print("Computation Time: " + str(np.round(computation_time / 60. / 60., 2)) + " h")
