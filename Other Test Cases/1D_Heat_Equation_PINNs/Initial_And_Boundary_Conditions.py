# import External Libraries
import numpy as np
import torch


# Define the true Initial and Boundary Conditions
def true_initial_condition_function(x):
    true_initial_condition_values = torch.sin(np.pi * x)
    return true_initial_condition_values


def true_upper_boundary_condition_function(x):
    true_upper_boundary_condition_values = 0 * x
    return true_upper_boundary_condition_values


def true_lower_boundary_condition_function(x):
    true_lower_boundary_condition_values = 0 * x
    return true_lower_boundary_condition_values
