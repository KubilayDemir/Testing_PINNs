# import External Libraries
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


tmax = 2150
xmin = -1000
xmax = 1000

x_grid = np.arange(xmin, xmax, 10)
time_grid = np.arange(0, tmax+1, 1)
[x_mesh_grid, time_mesh_grid] = np.meshgrid(x_grid, time_grid)

# Load Data for u and h
u_data = np.load("Zonal_Velocity.npy")
h_data = np.load("Sea_Level_Elevation.npy")
print(np.shape(u_data))
print(np.shape(h_data))

# Plot u
Numerical_Solution_Surface_Plot = plt.figure(figsize=(15, 7))
Numerical_Solution_Axis = Numerical_Solution_Surface_Plot.gca(projection='3d')
surf = Numerical_Solution_Axis.plot_surface(time_mesh_grid, x_mesh_grid, u_data,
                                            cmap=cm.coolwarm,
                                            linewidth=0, antialiased=False)
plt.title("Numerical Solution for u")
plt.xlabel("t [min]")
plt.ylabel("x [km]")
Numerical_Solution_Surface_Plot.colorbar(surf, shrink=0.5, aspect=5, label="T")
plt.show()

# Plot h
Numerical_Solution_Surface_Plot = plt.figure(figsize=(15, 7))
Numerical_Solution_Axis = Numerical_Solution_Surface_Plot.gca(projection='3d')
surf = Numerical_Solution_Axis.plot_surface(time_mesh_grid, x_mesh_grid, h_data,
                                            cmap=cm.coolwarm,
                                            linewidth=0, antialiased=False)
plt.title("Numerical Solution for h")
plt.xlabel("t [min]")
plt.ylabel("x [km]")
Numerical_Solution_Surface_Plot.colorbar(surf, shrink=0.5, aspect=5, label="T")
plt.show()

# Generate Animation for h
sea_level_figure = plt.figure()

def update_plot_h(frame):
    plt.cla()
    plt.plot(x_grid, h_data[frame, :])
    plt.xlim(xmin, xmax)
    plt.ylim(-1, 1)
    plt.title("Numerical Solution for Sea Level Elevation h at time t = " + str(frame) + " minutes")
    plt.xlabel("x [km]")
    plt.ylabel("h [m]")
    return

# generate and save Animation
number_of_frames = np.shape(h_data)[0]
frames_per_second = 20
sea_level_animation = animation.FuncAnimation(sea_level_figure, update_plot_h,
                                             frames=np.arange(0, number_of_frames, 10),
                                             init_func=None) #, interval=1000 / frames_per_second)
sea_level_animation.save('sea_level.gif', writer='Pillow', fps=int(frames_per_second))

# Generate Animation for u
zonal_velocity_figure = plt.figure()

def update_plot_u(frame):
    plt.cla()
    plt.plot(x_grid, u_data[frame, :])
    plt.xlim(xmin, xmax)
    plt.ylim(-0.25, 0.25)
    plt.title("Numerical Solution for Zonal Velocity u at time t = " + str(frame) + " minutes")
    plt.xlabel("x [km]")
    plt.ylabel("h [m]")
    return

# generate and save Animation
number_of_frames = np.shape(h_data)[0]
frames_per_second = 20
zonal_velocity_animation = animation.FuncAnimation(zonal_velocity_figure, update_plot_u,
                                             frames=np.arange(0, number_of_frames, 10),
                                             init_func=None) #, interval=1000 / frames_per_second)
zonal_velocity_animation.save('zonal_velocity.gif', writer='Pillow', fps=int(frames_per_second))