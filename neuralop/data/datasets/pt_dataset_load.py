from pathlib import Path

import numpy as np
import torch

from neuralop.utils import get_project_root


def numerical_solution():
    # Parameters for the problem
    N = 15  # Number of intervals in each dimension
    h = 1 / N  # Grid spacing
    x1 = np.linspace(0, 1, N + 1)
    x2 = np.linspace(0, 1, N + 1)
    X1, X2 = np.meshgrid(x1, x2)

    # Coefficient function a(x1, x2): Exponential Decay
    a_0, alpha = 1, 1
    a = a_0 * np.exp(-alpha * (X1 ** 2 + X2 ** 2))

    # Source function f(x1, x2): Gaussian Distribution
    q, sigma = 1, 1
    x1_0, x2_0 = 0.5, 0.5
    f = q * np.exp(-((X1 - x1_0) ** 2 + (X2 - x2_0) ** 2) / (2 * sigma ** 2))

    return a, f


# Подгружаю дата сеты
example_data_root = get_project_root() / "neuralop/data/datasets/data"
data_root = example_data_root
root_dir = data_root
root_dir = Path(root_dir)
dataset_name = "darcy"
train_resolution = 16

# data = torch.load(
#     Path(root_dir).joinpath(f"{dataset_name}_train_{train_resolution}.pt").as_posix()
# )
data = torch.load(
    Path("/home/dimitri/neuraloperator_Bolt/darcyFlow/").joinpath(f"{dataset_name}_train_{train_resolution}.pt").as_posix()
)

a, f = numerical_solution()
var1 = torch.tensor(a, dtype=torch.float32)
# data["y"] = torch.tensor(a)

pass
