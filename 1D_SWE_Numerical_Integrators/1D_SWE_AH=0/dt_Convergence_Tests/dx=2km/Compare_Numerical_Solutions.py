# import External Libraries
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

tmax = 4500
xmin = -1000

xmax = 1010

x_grid = np.arange(xmin, xmax, 10)
time_grid = np.arange(0, tmax, 1)
zeta_x_grid = np.arange(-995, 1005, 10)
[x_mesh_grid, time_mesh_grid] = np.meshgrid(x_grid, time_grid)
[zeta_x_mesh_grid, zeta_time_mesh_grid] = np.meshgrid(zeta_x_grid, time_grid)

# load solutions with different resolutions
zeta4 = np.load("Examples/dx=2km_dt=1s/Sea_Level_Elevation.npy")[:-1, 3::5]
u4 = np.load("Examples/dx=2km_dt=1s/Zonal_Velocity.npy")[:-1, ::5]

zeta3 = np.load("Examples/dx=2km_dt=5s/Sea_Level_Elevation.npy")[:-1, 3::5]
u3 = np.load("Examples/dx=2km_dt=5s/Zonal_Velocity.npy")[:-1, ::5]

zeta2 = np.load("Examples/dx=2km_dt=10s/Sea_Level_Elevation.npy")[:-1, 3::5]
u2 = np.load("Examples/dx=2km_dt=10s/Zonal_Velocity.npy")[:-1, ::5]

zeta1 = np.load("Examples/dx=2km_dt=30s/Sea_Level_Elevation.npy")[:-1, 3::5]
u1 = np.load("Examples/dx=2km_dt=30s/Zonal_Velocity.npy")[:-1, ::5]

zeta0 = np.load("Examples/dx=2km_dt=60s/Sea_Level_Elevation.npy")[:-1, 3::5]
u0 = np.load("Examples/dx=2km_dt=60s/Zonal_Velocity.npy")[:-1, ::5]

# zeta
plt.figure()
plt.contourf(zeta_time_mesh_grid[:4500, :], zeta_x_mesh_grid[:4500, :], zeta0[:4500, :], cmap=cm.coolwarm,
             antialiased=False, vmin=0.0, vmax=1.0)
plt.title(r"$\zeta_{0}$")
plt.xlabel("t [min]")
plt.ylabel("x [km]")
plt.colorbar()
plt.show()

print("Relative L2-Norm(zeta): " + str(
    round(np.linalg.norm(abs(zeta1 - zeta0)) / np.linalg.norm(abs(zeta0)) * 100, 2)) + "%")

print("Relative max-Norm(zeta): " + str(
    round(np.max(abs(zeta1 - zeta0)) / np.max(abs(zeta0)) * 100, 2)) + "%")

print("Relative L2-Norm(u): " + str(round(np.linalg.norm(abs(u1 - u0)) / np.linalg.norm(abs(u0)) * 100, 2)) + "%")

print("Relative max-Norm(u): " + str(
    round(np.max(abs(u1 - u0)) / np.max(abs(u0)) * 100, 2)) + "%")
