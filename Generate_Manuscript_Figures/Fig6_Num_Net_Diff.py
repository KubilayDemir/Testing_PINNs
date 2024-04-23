import json
import cmocean
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

data_directory = "Train_On_PINNs_Loss/AH=0/"

with open(data_directory + "Hyper_Parameter_Dictionary.json") as json_file:
    Hyper_Parameter_Dictionary = json.load(json_file)
non_dimensionalization = Hyper_Parameter_Dictionary["non_dimensionalization"]

# Domain Size and Grids
minimum_time = Hyper_Parameter_Dictionary["minimum_time"] * Hyper_Parameter_Dictionary["time_scale"]
maximum_time = Hyper_Parameter_Dictionary["maximum_time"] * Hyper_Parameter_Dictionary["time_scale"]
minimum_x = Hyper_Parameter_Dictionary["minimum_x"] * Hyper_Parameter_Dictionary["horizontal_length_scale"]
maximum_x = Hyper_Parameter_Dictionary["maximum_x"] * Hyper_Parameter_Dictionary["horizontal_length_scale"]
time_mesh_grid = np.load(data_directory + "dimensional_time_mesh_grid.npy")
x_mesh_grid = np.load(data_directory + "dimensional_x_mesh_grid.npy")
dimensional_zeta_solution_time_mesh_grid = np.load(data_directory + "dimensional_zeta_solution_time_mesh_grid.npy")
dimensional_zeta_solution_x_mesh_grid = np.load(data_directory + "dimensional_zeta_solution_x_mesh_grid.npy")
dimensional_u_solution_time_mesh_grid = np.load(data_directory + "dimensional_u_solution_time_mesh_grid.npy")
dimensional_u_solution_x_mesh_grid = np.load(data_directory + "dimensional_u_solution_x_mesh_grid.npy")

# Network Output, Numerical Solution and Absolute Difference
dimensional_network_output_h_values = np.load(data_directory + "dimensional_network_output_h_values.npy")
dimensional_network_output_u_values = np.load(data_directory + "dimensional_network_output_u_values.npy")
exact_solution_u_values = np.load(data_directory + "exact_solution_u_values.npy")
exact_solution_h_values = np.load(data_directory + "exact_solution_h_values.npy")
abs_error_h_values = np.load(data_directory + "abs_error_h_values.npy")
abs_error_u_values = np.load(data_directory + "abs_error_u_values.npy")

sea_level_colormap = cmocean.cm.haline
velocity_colormap = 'coolwarm'
difference_colormap = 'plasma'
general_font_size = 16
yticks = np.arange(-1000, 1250, 250)
xticks = np.arange(0, 75, 10)

fig, axs = plt.subplots(2, 3, figsize=(20, 10))
plt.rcParams.update({'font.size': general_font_size})

plt.subplots_adjust(left=0.065,
                    bottom=0.1,
                    right=0.96,
                    top=0.9,
                    wspace=0.25,
                    hspace=0.25)

im00 = axs[0, 0].imshow(np.flipud(exact_solution_h_values), cmap=sea_level_colormap, aspect='auto',
                        extent=[0, 75, -1000, 1000], vmin=0.0, vmax=1.0)
axs[0, 0].set_xticks(xticks)
axs[0, 0].set_yticks(yticks)
axs[0, 0].set_title(r"$\zeta_{NUM}(x,t)$ [m]")
axs[0, 0].set_ylabel("x [km]", fontsize=general_font_size-2.5)
axs[0, 0].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[0, 0].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)
fig.colorbar(mappable=im00, ax=axs[0, 0])

im01 = axs[0, 1].imshow(np.flipud(dimensional_network_output_h_values), cmap=sea_level_colormap, aspect='auto',
                        extent=[0, 75, -1000, 1000], vmin=0.0, vmax=1.0)
axs[0, 1].set_xticks(xticks)
axs[0, 1].set_yticks(yticks)
axs[0, 1].set_title(r"$\zeta_{NN}(x,t)$ [m]")
fig.colorbar(mappable=im01, ax=axs[0, 1])
axs[0, 1].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[0, 1].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)

im02 = axs[0, 2].imshow(np.flipud(abs_error_h_values), cmap=difference_colormap, aspect='auto',
                        extent=[0, 75, -1000, 1000], vmin=0.000, vmax=0.04)
axs[0, 2].set_xticks(xticks)
axs[0, 2].set_yticks(yticks)
axs[0, 2].set_title("$|\zeta_{NUM} - \zeta_{NN}|$ [m]")
axs[0, 2].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[0, 2].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)
cbar = fig.colorbar(mappable=im02, ax=axs[0, 2])
tick_locator = ticker.MaxNLocator(nbins=6)
cbar.locator = tick_locator
cbar.update_ticks()

im10 = axs[1, 0].imshow(np.flipud(exact_solution_u_values), cmap=velocity_colormap, aspect='auto',
                        extent=[0, 75, -1000, 1000], vmin=-0.15, vmax=0.15)
axs[1, 0].set_xticks(xticks)
axs[1, 0].set_yticks(yticks)
axs[1, 0].set_title(r"$u_{NUM}(x,t)$ [m/s]")
axs[1, 0].set_xlabel("t [h]", fontsize=general_font_size-2.5)
axs[1, 0].set_ylabel("x [km]", fontsize=general_font_size-2.5)
axs[1, 0].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[1, 0].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)
fig.colorbar(mappable=im10, ax=axs[1, 0])

im11 = axs[1, 1].imshow(np.flipud(dimensional_network_output_u_values), cmap=velocity_colormap, aspect='auto',
                        extent=[0, 75, -1000, 1000], vmin=-0.15, vmax=0.15)
axs[1, 1].set_xticks(xticks)
axs[1, 1].set_yticks(yticks)
axs[1, 1].set_title(r"$u_{NN}(x,t)$ [m/s]")
axs[1, 1].set_xlabel("t [h]", fontsize=general_font_size-2.5)
axs[1, 1].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[1, 1].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)
fig.colorbar(mappable=im11, ax=axs[1, 1])

im12 = axs[1, 2].imshow(np.flipud(abs_error_u_values), cmap=difference_colormap, aspect='auto',
                        extent=[0, 75, -1000, 1000], vmin=0.000, vmax=0.01)
axs[1, 2].set_xticks(xticks)
axs[1, 2].set_yticks(yticks)
axs[1, 2].set_title(r"$|u_{NUM} - u_{NN}|$ [m/s]")
axs[1, 2].set_xlabel("t [h]", fontsize=general_font_size-2.5)
axs[1, 2].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[1, 2].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)
cbar2 = fig.colorbar(mappable=im12, ax=axs[1, 2])
tick_locator = ticker.MaxNLocator(nbins=5)
cbar2.locator = tick_locator
cbar2.update_ticks()

def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[0] * scale - 0.2, ax.get_ylim()[1] * scale + 0.2

texts = ["(a): Numerical Solution", "(b): Network Output", "(c): Absolute Difference"]
axes = fig.get_axes()
for axis_number, axis_label in zip(axes, texts):
    axis_number.annotate(axis_label, xy=(-0.1, 1.15), xycoords="axes fraction",
                         weight="bold")

plt.savefig("TIFF/Fig6.tiff", dpi=95)
# plt.show()
