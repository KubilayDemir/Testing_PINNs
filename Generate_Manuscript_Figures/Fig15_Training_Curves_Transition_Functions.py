import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_directories = ["Train_On_PINNs_Loss/AH=0/", "Train_On_PINNs_Loss/BC_Only/",
                    "Train_On_PINNs_Loss/BC_IC_opt1/", "Train_On_PINNs_Loss/BC_IC_opt2/"]

general_font_size = 18
fig, axs = plt.subplots(2, 2, figsize=(20, 13), gridspec_kw={'hspace': 0.4})
plt.rcParams.update({'font.size': general_font_size})
plt.subplots_adjust(left=0.08, right=0.95,
                    bottom=0.12, top=0.88)

plt.xticks(fontsize=general_font_size)
plt.yticks(fontsize=general_font_size)

with open(data_directories[0] + "Hyper_Parameter_Dictionary.json") as json_file:
    Hyper_Parameter_Dictionary = json.load(json_file)
non_dimensionalization = Hyper_Parameter_Dictionary["non_dimensionalization"]

total_MSE_over_training = np.load(data_directories[0] + "total_MSE_over_training.npy")
MSE_boundary_conditions_over_training = np.load(data_directories[0] + "MSE_boundary_conditions_over_training.npy")
MSE_initial_conditions_over_training = np.load(data_directories[0] + "MSE_initial_conditions_over_training.npy")
MSE_symbolic_functions_over_training = np.load(data_directories[0] + "MSE_symbolic_functions_over_training.npy")
Relative_L2_Error_h_over_training = np.load(data_directories[0] + "Relative_L2_Error_h_over_training.npy")
Relative_L2_Error_u_over_training = np.load(data_directories[0] + "Relative_L2_Error_u_over_training.npy")
MSE_initial_condition_u_over_training = np.load(data_directories[0] + "MSE_initial_condition_u_over_training.npy")
MSE_initial_condition_h_over_training = np.load(data_directories[0] + "MSE_initial_condition_h_over_training.npy")
MSE_symbolic_function_u_over_training = np.load(data_directories[0] + "MSE_symbolic_function_u_over_training.npy")
MSE_symbolic_function_h_over_training = np.load(data_directories[0] + "MSE_symbolic_function_h_over_training.npy")
MSE_lower_boundary_condition_u_over_training = np.load(
    data_directories[0] + "MSE_lower_boundary_condition_u_over_training.npy")
MSE_upper_boundary_condition_u_over_training = np.load(
    data_directories[0] + "MSE_upper_boundary_condition_u_over_training.npy")

training_steps = range(len(total_MSE_over_training))

axs[0, 0].plot(training_steps[0], pd.Series(total_MSE_over_training[0]), label=r"$\mathcal{L}_{total}$",
               linewidth=4, alpha=1.0, color="C0")
axs[0, 0].plot(training_steps[0], pd.Series(MSE_boundary_conditions_over_training[0]),
               label=r"$\mathcal{L}_{BC}$", linewidth=4, alpha=1.0, color="C1")
axs[0, 0].plot(training_steps[0], pd.Series(MSE_initial_conditions_over_training[0]),
               label=r"$\mathcal{L}_{IC}$", linewidth=4,
               alpha=1.0, color="C2")
axs[0, 0].plot(training_steps[0], pd.Series(MSE_symbolic_functions_over_training[0]),
               label=r"$\mathcal{L}_{PDE}$", linewidth=4,
               alpha=1.0, color="C3")
axs[0, 0].plot(training_steps[0], pd.Series(Relative_L2_Error_h_over_training[0]),
               label=r"$\mathcal{L}_{num,\zeta}$", linewidth=4,
               alpha=1.0, color="C4")
axs[0, 0].plot(training_steps[0], pd.Series(Relative_L2_Error_u_over_training[0]),
               label=r"$\mathcal{L}_{num,u}$", linewidth=4,
               alpha=1.0, color="C5")

axs[0, 0].plot(training_steps, total_MSE_over_training, linewidth=0.20, alpha=0.35, color="C0")
axs[0, 0].plot(training_steps, MSE_boundary_conditions_over_training, linewidth=0.20, alpha=0.35, color="C1")
axs[0, 0].plot(training_steps, MSE_initial_conditions_over_training, linewidth=0.20, alpha=0.35, color="C2")
axs[0, 0].plot(training_steps, MSE_symbolic_functions_over_training, linewidth=0.20, alpha=0.35, color="C3")
axs[0, 0].plot(training_steps, Relative_L2_Error_h_over_training, linewidth=0.20, alpha=0.35, color="C4")
axs[0, 0].plot(training_steps, Relative_L2_Error_u_over_training, linewidth=0.20, alpha=0.35, color="C5")

axs[0, 0].plot(training_steps, pd.Series(total_MSE_over_training).rolling(100).mean(), label=r"$\mathcal{L}_{total}$",
               linewidth=1.5, alpha=1.0, color="C0")
axs[0, 0].plot(training_steps, pd.Series(MSE_boundary_conditions_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{BC}$", linewidth=1.5, alpha=1.0, color="C1")
axs[0, 0].plot(training_steps, pd.Series(MSE_initial_conditions_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{IC}$", linewidth=1.5,
               alpha=1.0, color="C2")
axs[0, 0].plot(training_steps, pd.Series(MSE_symbolic_functions_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{PDE}$", linewidth=1.5,
               alpha=1.0, color="C3")
axs[0, 0].plot(training_steps, pd.Series(Relative_L2_Error_h_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{num,\zeta}$", linewidth=1.5,
               alpha=1.0, color="C4")
axs[0, 0].plot(training_steps, pd.Series(Relative_L2_Error_u_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{num,u}$", linewidth=1.5,
               alpha=1.0, color="C5")

axs[0, 0].set_xlabel("epochs", fontsize=general_font_size)
axs[0, 0].set_yscale("log")
axs[0, 0].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[0, 0].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)
axs[0, 0].set_xlim(0, 10000)
axs[0, 0].grid(alpha=0.35)

with open(data_directories[1] + "Hyper_Parameter_Dictionary.json") as json_file:
    Hyper_Parameter_Dictionary = json.load(json_file)
non_dimensionalization = Hyper_Parameter_Dictionary["non_dimensionalization"]

total_MSE_over_training = np.load(data_directories[1] + "total_MSE_over_training.npy")
MSE_boundary_conditions_over_training = np.load(data_directories[1] + "MSE_boundary_conditions_over_training.npy")
MSE_initial_conditions_over_training = np.load(data_directories[1] + "MSE_initial_conditions_over_training.npy")
MSE_symbolic_functions_over_training = np.load(data_directories[1] + "MSE_symbolic_functions_over_training.npy")
Relative_L2_Error_h_over_training = np.load(data_directories[1] + "Relative_L2_Error_h_over_training.npy")
Relative_L2_Error_u_over_training = np.load(data_directories[1] + "Relative_L2_Error_u_over_training.npy")
MSE_initial_condition_u_over_training = np.load(data_directories[1] + "MSE_initial_condition_u_over_training.npy")
MSE_initial_condition_h_over_training = np.load(data_directories[1] + "MSE_initial_condition_h_over_training.npy")
MSE_symbolic_function_u_over_training = np.load(data_directories[1] + "MSE_symbolic_function_u_over_training.npy")
MSE_symbolic_function_h_over_training = np.load(data_directories[1] + "MSE_symbolic_function_h_over_training.npy")
MSE_lower_boundary_condition_u_over_training = np.load(
    data_directories[1] + "MSE_lower_boundary_condition_u_over_training.npy")
MSE_upper_boundary_condition_u_over_training = np.load(
    data_directories[1] + "MSE_upper_boundary_condition_u_over_training.npy")

training_steps = range(len(total_MSE_over_training))
axs[0, 1].plot(training_steps, total_MSE_over_training, linewidth=0.20, alpha=0.35, color="C0")
#axs[0, 1].plot(training_steps, MSE_boundary_conditions_over_training, linewidth=0.20, alpha=0.35, color="C1")
axs[0, 1].plot(training_steps, MSE_initial_conditions_over_training, linewidth=0.20, alpha=0.35, color="C2")
axs[0, 1].plot(training_steps, MSE_symbolic_functions_over_training, linewidth=0.20, alpha=0.35, color="C3")
axs[0, 1].plot(training_steps, Relative_L2_Error_h_over_training, linewidth=0.20, alpha=0.35, color="C4")
axs[0, 1].plot(training_steps, Relative_L2_Error_u_over_training, linewidth=0.20, alpha=0.35, color="C5")

axs[0, 1].plot(training_steps, pd.Series(total_MSE_over_training).rolling(100).mean(), label=r"$\mathcal{L}_{total}$",
               linewidth=1.5, alpha=1.0, color="C0")
axs[0, 1].plot(training_steps, pd.Series(MSE_initial_conditions_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{IC}$", linewidth=1.5,
               alpha=1.0, color="C2")
axs[0, 1].plot(training_steps, pd.Series(MSE_symbolic_functions_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{PDE}$", linewidth=1.5,
               alpha=1.0, color="C3")
axs[0, 1].plot(training_steps, pd.Series(Relative_L2_Error_h_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{num,\zeta}$", linewidth=1.5,
               alpha=1.0, color="C4")
axs[0, 1].plot(training_steps, pd.Series(Relative_L2_Error_u_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{num,u}$", linewidth=1.5,
               alpha=1.0, color="C5")

axs[0, 1].set_xlabel("epochs", fontsize=general_font_size)
axs[0, 1].set_yscale("log")
axs[0, 1].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[0, 1].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)
axs[0, 1].set_xlim(0, 10000)
axs[0, 1].grid(alpha=0.35)

with open(data_directories[2] + "Hyper_Parameter_Dictionary.json") as json_file:
    Hyper_Parameter_Dictionary = json.load(json_file)
non_dimensionalization = Hyper_Parameter_Dictionary["non_dimensionalization"]

total_MSE_over_training = np.load(data_directories[2] + "total_MSE_over_training.npy")
MSE_boundary_conditions_over_training = np.load(data_directories[2] + "MSE_boundary_conditions_over_training.npy")
MSE_initial_conditions_over_training = np.load(data_directories[2] + "MSE_initial_conditions_over_training.npy")
MSE_symbolic_functions_over_training = np.load(data_directories[2] + "MSE_symbolic_functions_over_training.npy")
Relative_L2_Error_h_over_training = np.load(data_directories[2] + "Relative_L2_Error_h_over_training.npy")
Relative_L2_Error_u_over_training = np.load(data_directories[2] + "Relative_L2_Error_u_over_training.npy")
MSE_initial_condition_u_over_training = np.load(data_directories[2] + "MSE_initial_condition_u_over_training.npy")
MSE_initial_condition_h_over_training = np.load(data_directories[2] + "MSE_initial_condition_h_over_training.npy")
MSE_symbolic_function_u_over_training = np.load(data_directories[2] + "MSE_symbolic_function_u_over_training.npy")
MSE_symbolic_function_h_over_training = np.load(data_directories[2] + "MSE_symbolic_function_h_over_training.npy")
MSE_lower_boundary_condition_u_over_training = np.load(
    data_directories[2] + "MSE_lower_boundary_condition_u_over_training.npy")
MSE_upper_boundary_condition_u_over_training = np.load(
    data_directories[2] + "MSE_upper_boundary_condition_u_over_training.npy")

total_MSE_over_training = np.load(data_directories[2] + "total_MSE_over_training.npy")
#axs[1, 0].plot(training_steps, total_MSE_over_training, linewidth=0.20, alpha=0.35, color="C0")
axs[1, 0].plot(training_steps, MSE_symbolic_functions_over_training, linewidth=0.20, alpha=0.35, color="C3")
axs[1, 0].plot(training_steps, Relative_L2_Error_h_over_training, linewidth=0.20, alpha=0.35, color="C4")
axs[1, 0].plot(training_steps, Relative_L2_Error_u_over_training, linewidth=0.20, alpha=0.35, color="C5")

#axs[1, 0].plot(training_steps, pd.Series(total_MSE_over_training).rolling(100).mean(), label=r"$\mathcal{L}_{total}$",
#               linewidth=1.5, alpha=1.0, color="C0")
axs[1, 0].plot(training_steps, pd.Series(MSE_symbolic_functions_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{PDE}$", linewidth=1.5,
               alpha=1.0, color="C3")
axs[1, 0].plot(training_steps, pd.Series(Relative_L2_Error_h_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{num,\zeta}$", linewidth=1.5,
               alpha=1.0, color="C4")
axs[1, 0].plot(training_steps, pd.Series(Relative_L2_Error_u_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{num,u}$", linewidth=1.5,
               alpha=1.0, color="C5")

axs[1, 0].set_xlabel("epochs", fontsize=general_font_size)
axs[1, 0].set_yscale("log")
axs[1, 0].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[1, 0].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)
axs[1, 0].set_xlim(0, 10000)
axs[1, 0].grid(alpha=0.35)

with open(data_directories[3] + "Hyper_Parameter_Dictionary.json") as json_file:
    Hyper_Parameter_Dictionary = json.load(json_file)
non_dimensionalization = Hyper_Parameter_Dictionary["non_dimensionalization"]

total_MSE_over_training = np.load(data_directories[3] + "total_MSE_over_training.npy")
MSE_boundary_conditions_over_training = np.load(data_directories[3] + "MSE_boundary_conditions_over_training.npy")
MSE_initial_conditions_over_training = np.load(data_directories[3] + "MSE_initial_conditions_over_training.npy")
MSE_symbolic_functions_over_training = np.load(data_directories[3] + "MSE_symbolic_functions_over_training.npy")
Relative_L2_Error_h_over_training = np.load(data_directories[3] + "Relative_L2_Error_h_over_training.npy")
Relative_L2_Error_u_over_training = np.load(data_directories[3] + "Relative_L2_Error_u_over_training.npy")
MSE_initial_condition_u_over_training = np.load(data_directories[3] + "MSE_initial_condition_u_over_training.npy")
MSE_initial_condition_h_over_training = np.load(data_directories[3] + "MSE_initial_condition_h_over_training.npy")
MSE_symbolic_function_u_over_training = np.load(data_directories[3] + "MSE_symbolic_function_u_over_training.npy")
MSE_symbolic_function_h_over_training = np.load(data_directories[3] + "MSE_symbolic_function_h_over_training.npy")
MSE_lower_boundary_condition_u_over_training = np.load(
    data_directories[3] + "MSE_lower_boundary_condition_u_over_training.npy")
MSE_upper_boundary_condition_u_over_training = np.load(
    data_directories[3] + "MSE_upper_boundary_condition_u_over_training.npy")

total_MSE_over_training = np.load(data_directories[3] + "total_MSE_over_training.npy")
axs[1, 1].plot(training_steps, MSE_symbolic_functions_over_training, linewidth=0.20, alpha=0.35,
               color="C3")
axs[1, 1].plot(training_steps, Relative_L2_Error_h_over_training, linewidth=0.20, alpha=0.35, color="C4")
axs[1, 1].plot(training_steps, Relative_L2_Error_u_over_training, linewidth=0.20, alpha=0.35, color="C5")
axs[1, 1].plot(training_steps, pd.Series(MSE_symbolic_functions_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{PDE}$", linewidth=1.5,
               alpha=1.0, color="C3")
axs[1, 1].plot(training_steps, pd.Series(Relative_L2_Error_h_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{num,\zeta}$", linewidth=1.5,
               alpha=1.0, color="C4")
axs[1, 1].plot(training_steps, pd.Series(Relative_L2_Error_u_over_training).rolling(100).mean(),
               label=r"$\mathcal{L}_{num,u}$", linewidth=1.5,
               alpha=1.0, color="C5")

axs[1, 1].set_xlabel("epochs", fontsize=general_font_size)
axs[1, 1].set_yscale("log")
axs[1, 1].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[1, 1].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)
axs[1, 1].set_xlim(0, 10000)
axs[1, 1].grid(alpha=0.35)

def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[0] * scale - 0.2, ax.get_ylim()[1] * scale + 0.2

texts = [r"(a) Default", r"(b) Transition Function $\mathcal{T}_{1}$", r"(c) Transition Function $\mathcal{T}_{2}$",
         r"(d) Transition Function $\mathcal{T}_{3}$"]

axes = fig.get_axes()
for axis_number, axis_label in zip(axes, texts):
    axis_number.annotate(axis_label, xy=(-0.1, 1.1), xycoords="axes fraction", fontsize=general_font_size,
                         weight="bold")

labels = [r"$\mathcal{L}_{total}$", r"$\mathcal{L}_{BC}$", r"$\mathcal{L}_{IC}$",
          r"$\mathcal{L}_{PDE}$", r"$\mathcal{L}_{num,\zeta}$", r"$\mathcal{L}_{num,u}$"]

lgd = axs[0, 0].legend(labels, loc='center', bbox_to_anchor=(0.502, 0.03), ncol=len(labels),
                 bbox_transform=fig.transFigure, frameon=False)

plt.savefig("TIFF/Fig15.tiff", bbox_extra_artists=(lgd, ), dpi=95)
plt.show()
