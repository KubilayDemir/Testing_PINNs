# import external libraries
import matplotlib.pyplot as plt
import numpy as np
import torch

# test derivative of cosine
print("##################")
print("Test1")
print("##################")
A = torch.FloatTensor(np.arange(0, 2.2 * np.pi, 0.2 * np.pi))
A.requires_grad_()
B = torch.sin(A)
C = torch.autograd.grad(B.sum(), A, create_graph=True)[0]
D = torch.cos(A)
print("sin(x): ", B)
print("dsin(x)/dx: ", C)
print("cos(x): ", D)
print("(dsin(x)/dx - cos(x)): ", C - D)

# test derivative of cosine
print("##################")
print("Test2")
print("##################")
x = torch.FloatTensor(np.arange(0, 2.2 * np.pi, 0.2 * np.pi))
y = torch.FloatTensor(np.arange(0, 2.2 * np.pi, 0.2 * np.pi))
[x_mesh_grid, y_mesh_grid] = torch.meshgrid(x, y)
x_input_grid = x_mesh_grid.reshape(x_mesh_grid.size())
y_input_grid = y_mesh_grid.reshape(y_mesh_grid.size())
x_input_grid.requires_grad_()
y_input_grid.requires_grad_()
u = torch.sin(x_input_grid) * torch.sin(y_input_grid)
u_x = torch.autograd.grad(u.sum(), x_input_grid, create_graph=True)[0]
u_x = u_x.reshape(shape=(len(x), len(y)))

figure = plt.figure()
axis = figure.gca(projection="3d")
axis.plot_surface(x_mesh_grid.detach().numpy(), y_mesh_grid.detach().numpy(), u_x.detach().numpy())
plt.show()

# test second derivative
print("##################")
print("Test3")
print("##################")
C.requires_grad_()
E = torch.autograd.grad(C.sum(), A, create_graph=True)[0]
print("(d^2 sin(x)/dx^2 + sin(x)): ", E + B)
