# import External Libraries
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

tmax = 9000
xmin = -1000

xmax = 1010

x_grid = np.arange(xmin, xmax, 10)
time_grid = np.arange(0, tmax, 1)
zeta_x_grid = np.arange(-995, 1005, 10)
[x_mesh_grid, time_mesh_grid] = np.meshgrid(x_grid, time_grid)
[zeta_x_mesh_grid, zeta_time_mesh_grid] = np.meshgrid(zeta_x_grid, time_grid)

# load solutions with different resolutions
zeta = np.load("Examples/Adv_No_Diff_dx=400m_dt=1s/Sea_Level_Elevation.npy")[:-1, 13::25]
u = np.load("Examples/Adv_No_Diff_dx=400m_dt=1s/Zonal_Velocity.npy")[:-1, ::25]

zeta_diff = np.load("Examples/Adv_AH=1e+4_dx=400m_dt=1s/Sea_Level_Elevation.npy")[:-1, 13::25]
u_diff = np.load("Examples/Adv_AH=1e+4_dx=400m_dt=1s/Zonal_Velocity.npy")[:-1, ::25]

zeta_diff2 = np.load("Examples/Adv_AH=5e+4_dx=400m_dt=1s/Sea_Level_Elevation.npy")[:-1, 13::25]
u_diff2 = np.load("Examples/Adv_AH=5e+4_dx=400m_dt=1s/Zonal_Velocity.npy")[:-1, ::25]

zeta_no_adv_no_diff = np.load("Examples/no_adv_no_diff/Sea_Level_Elevation.npy")
u_no_adv_no_diff = np.load("Examples/no_adv_no_diff/Zonal_Velocity.npy")

# zeta
plt.figure()
plt.contourf(zeta_time_mesh_grid[:4500, :], zeta_x_mesh_grid[:4500, :], zeta[:4500, :],
                                            cmap=cm.coolwarm,
                                            linewidth=0, antialiased=False)
plt.title(r"$\zeta$ No Diffusion")
plt.xlabel("t [min]")
plt.ylabel("x [km]")
plt.colorbar()
plt.show()

# zeta diff
plt.figure()
plt.contourf(zeta_time_mesh_grid[:4500, :], zeta_x_mesh_grid[:4500, :], zeta_diff[:4500, :],
                                            cmap=cm.coolwarm,
                                            linewidth=0, antialiased=False)
plt.title(r"$\zeta_{diff}$")
plt.xlabel("t [min]")
plt.ylabel("x [km]")
plt.colorbar()
plt.show()

# zeta diff2
plt.figure()
plt.contourf(zeta_time_mesh_grid[:4500, :], zeta_x_mesh_grid[:4500, :], zeta_diff2[:4500, :],
                                            cmap=cm.coolwarm,
                                            linewidth=0, antialiased=False)
plt.title(r"$\zeta_{diff2}$")
plt.xlabel("t [min]")
plt.ylabel("x [km]")
plt.colorbar()
plt.show()

# zeta - zeta diff
plt.figure()
plt.contourf(zeta_time_mesh_grid[:4500, :], zeta_x_mesh_grid[:4500, :], abs(zeta_diff[:4500, :] - zeta[:4500, :]),
                                            cmap=cm.coolwarm,
                                            linewidth=0, antialiased=False)
plt.title(r"$|\zeta_{diff}-\zeta|$")
plt.xlabel("t [min]")
plt.ylabel("x [km]")
plt.colorbar()
plt.show()

# zeta - zeta diff2
plt.figure()
plt.contourf(zeta_time_mesh_grid[:4500, :], zeta_x_mesh_grid[:4500, :], abs(zeta_diff2[:4500, :] - zeta[:4500, :]),
                                            cmap=cm.coolwarm,
                                            linewidth=0, antialiased=False)
plt.title(r"$|\zeta_{diff2}-\zeta|$")
plt.xlabel("t [min]")
plt.ylabel("x [km]")
plt.colorbar()
plt.show()



