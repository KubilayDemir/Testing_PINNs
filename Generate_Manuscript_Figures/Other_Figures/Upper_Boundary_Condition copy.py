import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_directories = ["Train_On_PINNs_Loss/AH=0/", "Train_On_PINNs_Loss/AH=5e+4/", "Train_On_Validation_Loss/AH=0/", "Train_On_Validation_Loss/AH=5e+4/"]

general_font_size = 13
fig, axs = plt.subplots(2, 2, figsize=(20, 10))
plt.rcParams.update({'font.size': general_font_size})

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.25,
                    hspace=0.35)

plt.xticks(fontsize=general_font_size)
plt.yticks(fontsize=general_font_size)

# Train_On_PINNs_Loss AH=0
with open(data_directories[0] + "Hyper_Parameter_Dictionary.json") as json_file:
    Hyper_Parameter_Dictionary = json.load(json_file)
non_dimensionalization = Hyper_Parameter_Dictionary["non_dimensionalization"]

t = np.load(data_directories[0] + "dimensional_time_mesh_grid.npy")[0, :]
true_lower_boundary_condition_u_values = np.load(data_directories[0] + "true_lower_boundary_condition_u_values.npy")
true_upper_boundary_condition_u_values = np.load(data_directories[0] + "true_upper_boundary_condition_u_values.npy")
network_output_u_lower_boundary_condition = np.load(data_directories[0] + "network_output_u_lower_boundary_condition.npy")
network_output_u_upper_boundary_condition = np.load(data_directories[0] + "network_output_u_upper_boundary_condition.npy")

axs[0, 0].plot(t / 60. / 60., network_output_u_upper_boundary_condition, "-", label=r"$u_{NN}$")
axs[0, 0].plot(t / 60. / 60., true_upper_boundary_condition_u_values, "-", label=r"$u_{NUM}$")
axs[0, 0].set_xlabel("time [h]")
axs[0, 0].set_title(r"$u(t,x=1000km)$")
leg = axs[1, 1].legend()
axs[0, 0].grid()
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)

# Train_On_PINNs_Loss AH=5e+4
with open(data_directories[1] + "Hyper_Parameter_Dictionary.json") as json_file:
    Hyper_Parameter_Dictionary = json.load(json_file)
non_dimensionalization = Hyper_Parameter_Dictionary["non_dimensionalization"]

t = np.load(data_directories[1] + "dimensional_time_mesh_grid.npy")[1, :]
true_lower_boundary_condition_u_values = np.load(data_directories[1] + "true_lower_boundary_condition_u_values.npy")
true_upper_boundary_condition_u_values = np.load(data_directories[1] + "true_upper_boundary_condition_u_values.npy")
network_output_u_lower_boundary_condition = np.load(data_directories[1] + "network_output_u_lower_boundary_condition.npy")
network_output_u_upper_boundary_condition = np.load(data_directories[1] + "network_output_u_upper_boundary_condition.npy")

axs[0, 1].plot(t / 60. / 60., network_output_u_upper_boundary_condition, "-", label=r"$u_{NN}$")
axs[0, 1].plot(t / 60. / 60., true_upper_boundary_condition_u_values, "-", label=r"$u_{NUM}$")
axs[0, 1].set_xlabel("time [h]")
axs[0, 1].set_title(r"$u(t,x=1000km)$")
leg = axs[0, 1].legend()
axs[0, 1].grid()
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)

# Train_On_Validation_Loss AH=0
with open(data_directories[2] + "Hyper_Parameter_Dictionary.json") as json_file:
    Hyper_Parameter_Dictionary = json.load(json_file)
non_dimensionalization = Hyper_Parameter_Dictionary["non_dimensionalization"]

t = np.load(data_directories[2] + "dimensional_time_mesh_grid.npy")[0, :]
true_lower_boundary_condition_u_values = np.load(data_directories[2] + "true_lower_boundary_condition_u_values.npy")
true_upper_boundary_condition_u_values = np.load(data_directories[2] + "true_upper_boundary_condition_u_values.npy")
network_output_u_lower_boundary_condition = np.load(data_directories[2] + "network_output_u_lower_boundary_condition.npy")
network_output_u_upper_boundary_condition = np.load(data_directories[2] + "network_output_u_upper_boundary_condition.npy")

axs[1, 0].plot(t / 60. / 60., network_output_u_upper_boundary_condition, "-", label=r"$u_{NN}$")
axs[1, 0].plot(t / 60. / 60., true_upper_boundary_condition_u_values, "-", label=r"$u_{NUM}$")
axs[1, 0].set_xlabel("time [h]")
axs[1, 0].set_title(r"$u(t,x=1000km)$")
leg = axs[1, 0].legend()
axs[1, 0].grid()
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)

# Train_On_Validation_Loss AH=5e+4
with open(data_directories[3] + "Hyper_Parameter_Dictionary.json") as json_file:
    Hyper_Parameter_Dictionary = json.load(json_file)
non_dimensionalization = Hyper_Parameter_Dictionary["non_dimensionalization"]

t = np.load(data_directories[3] + "dimensional_time_mesh_grid.npy")[0, :]
true_lower_boundary_condition_u_values = np.load(data_directories[3] + "true_lower_boundary_condition_u_values.npy")
true_upper_boundary_condition_u_values = np.load(data_directories[3] + "true_upper_boundary_condition_u_values.npy")
network_output_u_lower_boundary_condition = np.load(data_directories[3] + "network_output_u_lower_boundary_condition.npy")
network_output_u_upper_boundary_condition = np.load(data_directories[3] + "network_output_u_upper_boundary_condition.npy")

axs[1, 1].plot(t / 60. / 60., network_output_u_upper_boundary_condition, "-", label=r"$u_{NN}$")
axs[1, 1].plot(t / 60. / 60., true_upper_boundary_condition_u_values, "-", label=r"$u_{NUM}$")
axs[1, 1].set_xlabel("time [h]")
axs[1, 1].set_title(r"$u(t,x=1000km)$")
leg = axs[1, 1].legend()
axs[1, 1].grid()
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)


def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[0] * scale - 0.2, ax.get_ylim()[1] * scale + 0.2


texts = [r"A: Train_On_PINNs_Loss Loss $A_{H}=0$", r"B: Train_On_PINNs_Loss Loss $A_{H}=5 \cdot 10^{4}$", r"C: Numerical Solution Loss $A_{H}=0$",
         r"D: Numerical Solution Loss $A_{H}=5 \cdot 10^{4}$"]

axes = fig.get_axes()
for axis_number, axis_label in zip(axes, texts):
    axis_number.annotate(axis_label, xy=(-0.1, 1.1), xycoords="axes fraction", fontsize=general_font_size,
                         weight="bold")

plt.savefig("fig10.pdf", format="pdf")
plt.show()
