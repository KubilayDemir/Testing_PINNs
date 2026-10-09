import numpy as np
import torch
from smt.sampling_methods import LHS
from torch.utils.data import Dataset


class PDEDataset(Dataset):
    def __init__(self, t_min, t_max, x_min, x_max, batch_size):
        self.t_min = t_min
        self.t_max = t_max
        self.x_min = x_min
        self.x_max = x_max
        self.n_pde_points = batch_size

        self.pde_sampler = LHS(
            xlimits=np.array(
                [[self.t_min, self.t_max], [self.x_min, self.x_max]]
            ),
            seed=int(np.random.randint(0, 2**31 - 1)),
        )

        self.tx = torch.FloatTensor(
            self.pde_sampler(self.n_pde_points)
        ).requires_grad_(True)
        self.t = self.tx[:, 0:1]
        self.x = self.tx[:, 1:2]
        self.n_samples = self.tx.shape[0]

    def __getitem__(self, index):
        return self.tx[index, :]

    def __len__(self):
        return self.n_samples


class LeftBoundaryDataset(Dataset):
    def __init__(self, t_min, t_max, x_min, batch_size):
        self.t_min = t_min
        self.t_max = t_max
        self.x_min = x_min
        self.lower_boundary_batch_size = batch_size

        self.lower_boundary_sampling_method = LHS(
            xlimits=np.array(
                [[self.t_min, self.t_max], [self.x_min, self.x_min]]
            ),
            seed=int(np.random.randint(0, 2**31 - 1)),
        )

        self.tx = torch.FloatTensor(
            self.lower_boundary_sampling_method(self.lower_boundary_batch_size)
        ).requires_grad_(True)
        self.t = self.tx[:, 0:1]
        self.x = self.tx[:, 1:2]
        self.n_samples = self.tx.shape[0]

    def __getitem__(self, index):
        return self.tx[index, :]

    def __len__(self):
        return self.n_samples


class RightBoundaryDataset(Dataset):
    def __init__(self, t_min, t_max, x_max, batch_size):
        self.t_min = t_min
        self.t_max = t_max
        self.x_max = x_max
        self.upper_boundary_batch_size = batch_size

        self.upper_boundary_sampling_method = LHS(
            xlimits=np.array(
                [[self.t_min, self.t_max], [self.x_max, self.x_max]]
            ),
            seed=int(np.random.randint(0, 2**31 - 1)),
        )

        self.tx = torch.FloatTensor(
            self.upper_boundary_sampling_method(self.upper_boundary_batch_size)
        ).requires_grad_(True)
        self.t = self.tx[:, 0:1]
        self.x = self.tx[:, 1:2]
        self.n_samples = self.tx.shape[0]

    def __getitem__(self, index):
        return self.tx[index, :]

    def __len__(self):
        return self.n_samples


class InitialConditionDataset(Dataset):
    def __init__(self, t_min, x_min, x_max, batch_size):
        self.t_min = t_min
        self.x_min = x_min
        self.x_max = x_max
        self.n_ic_points = batch_size

        self.initial_condition_sampling_method = LHS(
            xlimits=np.array(
                [[self.t_min, self.t_min], [self.x_min, self.x_max]]
            ),
            seed=int(np.random.randint(0, 2**31 - 1)),
        )

        self.tx = torch.FloatTensor(
            self.initial_condition_sampling_method(self.n_ic_points)
        ).requires_grad_(True)
        self.t = self.tx[:, 0:1]
        self.x = self.tx[:, 1:2]
        self.n_samples = self.tx.shape[0]

    def __getitem__(self, index):
        return self.tx[index, :]

    def __len__(self):
        return self.n_samples


class FinalStateDataset(Dataset):
    def __init__(self, t_max, x_min, x_max, batch_size):
        self.t_max = t_max
        self.x_min = x_min
        self.x_max = x_max
        self.n_ic_points = batch_size

        self.final_state_sampling_method = LHS(
            xlimits=np.array(
                [[self.t_max, self.t_max], [self.x_min, self.x_max]]
            ),
            seed=int(np.random.randint(0, 2**31 - 1)),
        )

        self.tx = torch.FloatTensor(
            self.final_state_sampling_method(self.n_ic_points)
        ).requires_grad_(True)
        self.t = self.tx[:, 0:1]
        self.x = self.tx[:, 1:2]
        self.n_samples = self.tx.shape[0]

    def __getitem__(self, index):
        return self.tx[index, :]

    def __len__(self):
        return self.n_samples
