import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from torch.utils.data import Dataset, DataLoader


# Part I: Numerical Solution of Darcy Flow
def numerical_solution():
    # Parameters for the problem
    N = 16  # Number of intervals in each dimension
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

    # Initialize sparse matrix and right-hand side vector
    grid_points = (N + 1) ** 2
    A = sp.lil_matrix((grid_points, grid_points))
    b = f.flatten()

    # Indexing function
    def index(i, j):
        return i * (N + 1) + j

    # Fill the sparse matrix using 5-point stencil
    for i in range(1, N):
        for j in range(1, N):
            idx = index(i, j)
            a_center = a[i, j]
            a_east = 0.5 * (a[i, j] + a[i, j + 1])
            a_west = 0.5 * (a[i, j] + a[i, j - 1])
            a_north = 0.5 * (a[i, j] + a[i - 1, j])
            a_south = 0.5 * (a[i, j] + a[i + 1, j])

            A[idx, idx] = -(a_east + a_west + a_north + a_south) / h ** 2
            A[idx, index(i, j + 1)] = a_east / h ** 2
            A[idx, index(i, j - 1)] = a_west / h ** 2
            A[idx, index(i - 1, j)] = a_north / h ** 2
            A[idx, index(i + 1, j)] = a_south / h ** 2

    # Apply boundary conditions (u = 0 on boundaries)
    for i in range(N + 1):
        for j in [0, N]:
            idx = index(i, j)
            A[idx, idx] = 1
            b[idx] = 0
        for j in range(N + 1):
            idx = index(0, j)
            A[idx, idx] = 1
            b[idx] = 0
            idx = index(N, j)
            A[idx, idx] = 1
            b[idx] = 0

    # Solve the linear system
    A = A.tocsr()
    u = spla.spsolve(A, b)

    # Reshape solution for visualization
    u = u.reshape((N + 1, N + 1))

    # Save arrays to npy archives
    np.save("X1.npy", X1)
    np.save("X2.npy", X2)
    np.save("u.npy", u)

    return X1, X2, u, a


# Part II: Custom Dataset and DataLoader
class DarcyDataset(Dataset):
    def __init__(self, X1, X2, u):
        self.data = np.column_stack((X1.flatten(), X2.flatten()))
        self.targets = u.flatten()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Part III: Neural Network Model
def neural_network_solution(X1, X2, u):
    dataset = DarcyDataset(X1, X2, u)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Define neural network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 500
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x_batch, y_batch = batch
            optimizer.zero_grad()
            y_pred = model(x_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader))

    # Plot the loss function over epochs (log scale)
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), losses, label="Loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Epoch (log scale)")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss over Epochs (Log Scale)")
    plt.legend()
    plt.grid()
    plt.show()

    # Predict on the training data
    x_train = np.column_stack((X1.flatten(), X2.flatten()))
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    u_pred = model(x_train_tensor).detach().numpy().reshape(X1.shape)

    return u_pred


# Part IV: Visualization
def visualize_3d_gif(X1, X2, Z, title, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="k", alpha=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Z")
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    def update(frame):
        ax.view_init(elev=30, azim=frame)

    ani = FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50)
    ani.save(filename, writer="imagemagick")


if __name__ == "__main__":
    # Numerical solution
    X1, X2, u, a = numerical_solution()

    # Visualize coefficient and solution as GIFs
    visualize_3d_gif(X1, X2, a, "Exponential Decay Coefficient Function", "coefficient.gif")
    visualize_3d_gif(X1, X2, u, "Solution u(x1, x2)", "solution.gif")

    # Train and visualize neural network solution
    u_pred = neural_network_solution(X1, X2, u)
    visualize_3d_gif(X1, X2, u_pred, "NN Predicted Solution", "nn_solution.gif")
