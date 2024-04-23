import json
import matplotlib.pyplot as plt
import numpy as np

data_directory = "Train_On_PINNs_Loss/AH=0/"

with open(data_directory + "Hyper_Parameter_Dictionary.json") as json_file:
    Hyper_Parameter_Dictionary = json.load(json_file)

symbolic_function_h_over_training = (np.load(data_directory + "symbolic_function_h_over_training.npy")
                                     * Hyper_Parameter_Dictionary["vertical_length_scale"])
network_output_u_over_training = (np.load(data_directory + "network_output_u_over_training.npy")
                                  * Hyper_Parameter_Dictionary["horizontal_length_scale"]
                                  / Hyper_Parameter_Dictionary["time_scale"])
dimensional_x_mesh_grid = np.load(data_directory + "dimensional_x_mesh_grid.npy")
dimensional_time_mesh_grid = np.load(data_directory + "dimensional_time_mesh_grid.npy")

continuity_residual_colormap = 'inferno'
momentum_residual_colormap = 'viridis'
general_font_size = 16
yticks = np.arange(-1000, 1250, 250)
xticks = np.arange(0, 75, 10)

fig, axs = plt.subplots(2, 5, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.16]})
plt.rcParams.update({'font.size': general_font_size})

plt.subplots_adjust(left=0.08,
                    bottom=0.1,
                    right=0.94,
                    top=0.9,
                    wspace=0.45,
                    hspace=0.25)

plt.xticks(fontsize=general_font_size)
plt.yticks(fontsize=general_font_size)

im00 = axs[0, 0].imshow(np.flipud(np.log10(abs(symbolic_function_h_over_training[5]))),
                          cmap=continuity_residual_colormap, aspect='auto', extent=[0, 75, -1000, 1000],
                        vmin=-6, vmax=-1)
axs[0, 0].set_xticks(xticks)
axs[0, 0].set_yticks(yticks)
axs[0, 0].set_title(r"$\log_{10}|f_{\zeta}|$")
axs[0, 0].set_ylabel("x [km]", fontsize=general_font_size)
axs[0, 0].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[0, 0].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)

im01 = axs[0, 1].imshow(np.flipud(np.log10(abs(symbolic_function_h_over_training[10]))),
                          cmap=continuity_residual_colormap, aspect='auto', extent=[0, 75, -1000, 1000],
                        vmin=-6, vmax=-1)
axs[0, 1].set_xticks(xticks)
axs[0, 1].set_yticks(yticks)
axs[0, 1].set_title(r"$\log_{10}|f_{\zeta}|$")
axs[0, 1].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[0, 1].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)

im02 = axs[0, 2].imshow(np.flipud(np.log10(abs(symbolic_function_h_over_training[20]))),
                          cmap=continuity_residual_colormap, aspect='auto', extent=[0, 75, -1000, 1000],
                        vmin=-6, vmax=-1)
axs[0, 2].set_xticks(xticks)
axs[0, 2].set_yticks(yticks)
axs[0, 2].set_title(r"$\log_{10}|f_{\zeta}|$")
axs[0, 2].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[0, 2].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)

im03 = axs[0, 3].imshow(np.flipud(np.log10(abs(symbolic_function_h_over_training[30]))),
                          cmap=continuity_residual_colormap, aspect='auto', extent=[0, 75, -1000, 1000],
                        vmin=-6, vmax=-1)
axs[0, 3].set_xticks(xticks)
axs[0, 3].set_yticks(yticks)
axs[0, 3].set_title(r"$\log_{10}|f_{\zeta}|$")
axs[0, 3].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[0, 3].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)

im04 = axs[0, 4].imshow(np.flipud(np.log10(abs(symbolic_function_h_over_training[40]))),
                          cmap=continuity_residual_colormap, aspect='auto', extent=[0, 75, -1000, 1000],
                        vmin=-6, vmax=-1)
axs[0, 4].set_xticks(xticks)
axs[0, 4].set_yticks(yticks)
axs[0, 4].set_title(r"$\log_{10}|f_{\zeta}|$")
axs[0, 4].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[0, 4].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)
fig.colorbar(mappable=im04, ax=axs[0, 4])

im10 = axs[1, 0].imshow(np.flipud(np.log10(abs(network_output_u_over_training[1]))),
                        cmap=momentum_residual_colormap, aspect='auto', extent=[0, 75, -1000, 1000],
                        vmin=-6, vmax=-1)
axs[1, 0].set_xticks(xticks)
axs[1, 0].set_yticks(yticks)
axs[1, 0].set_xlabel("t [h]", fontsize=general_font_size)
axs[1, 0].set_ylabel("x [km]", fontsize=general_font_size)
axs[1, 0].set_title(r"$\log_{10}|f_{u}|$")
axs[1, 0].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[1, 0].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)

im11 = axs[1, 1].imshow(np.flipud(np.log10(abs(network_output_u_over_training[10]))),
                        cmap=momentum_residual_colormap, aspect='auto', extent=[0, 75, -1000, 1000],
                        vmin=-6, vmax=-1)
axs[1, 1].set_xticks(xticks)
axs[1, 1].set_yticks(yticks)
axs[1, 1].set_xlabel("t [h]", fontsize=general_font_size)
axs[1, 1].set_title(r"$\log_{10}|f_{u}|$")
axs[1, 1].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[1, 1].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)

im12 = axs[1, 2].imshow(np.flipud(np.log10(abs(network_output_u_over_training[20]))),
                        cmap=momentum_residual_colormap, aspect='auto', extent=[0, 75, -1000, 1000],
                        vmin=-6, vmax=-1)
axs[1, 2].set_xticks(xticks)
axs[1, 2].set_yticks(yticks)
axs[1, 2].set_xlabel("t [h]", fontsize=general_font_size)
axs[1, 2].set_title(r"$\log_{10}|f_{u}|$")
axs[1, 2].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[1, 2].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)

im13 = axs[1, 3].imshow(np.flipud(np.log10(abs(network_output_u_over_training[30]))),
                        cmap=momentum_residual_colormap, aspect='auto', extent=[0, 75, -1000, 1000],
                        vmin=-6, vmax=-1)
axs[1, 3].set_xticks(xticks)
axs[1, 3].set_yticks(yticks)
axs[1, 3].set_xlabel("t [h]", fontsize=general_font_size)
axs[1, 3].set_title(r"$\log_{10}|f_{u}|$")
axs[1, 3].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[1, 3].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)

im14 = axs[1, 4].imshow(np.flipud(np.log10(abs(network_output_u_over_training[40]))),
                        cmap=momentum_residual_colormap, aspect='auto', extent=[0, 75, -1000, 1000],
                        vmin=-6, vmax=-1)
axs[1, 4].set_xticks(xticks)
axs[1, 4].set_yticks(yticks)
axs[1, 4].set_title(r"$\log_{10}|f_{u}|$")
axs[1, 4].set_xlabel("t [h]", fontsize=general_font_size)
axs[1, 4].tick_params(axis='both', which='major', labelsize=general_font_size-2.5)
axs[1, 4].tick_params(axis='both', which='minor', labelsize=general_font_size-2.5)
fig.colorbar(mappable=im14, ax=axs[1, 4])

def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[0] * scale - 0.2, ax.get_ylim()[1] * scale + 0.2


texts = ["(a): 500 epochs", "(b): 1000 epochs", "(c): 2000 epochs", "(d): 3000 epochs", "(e): 4000 epochs"]
axes = fig.get_axes()
for axis_number, axis_label in zip(axes, texts):
    axis_number.annotate(axis_label, xy=(-0.1, 1.15), xycoords="axes fraction", fontsize=general_font_size,
                         weight="bold")

plt.savefig("TIFF/Fig11.tiff", dpi=95)
plt.show()
