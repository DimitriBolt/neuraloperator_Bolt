import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

# Parameters
grid_size = 32
x1 = np.linspace(0, 1, grid_size)
x2 = np.linspace(0, 1, grid_size)
X1, X2 = np.meshgrid(x1, x2)
dx = 1 / (grid_size - 1)
Stop_a_0 = 5  # Default value for a_0 stop
alpha_range = range(1, 10 + 1)  # Alpha range


# Discretization and stencil for boundary conditions
def solve_darcy_with_boundary(a, f, dx, grid_size):
    '''
    Solves the Darcy Flow PDE using finite difference method with enforced boundary conditions.
    Ensures u = 0 on the boundary.
    '''
    n = grid_size
    u = np.zeros((n, n))  # Initialize solution array with boundary condition u=0
    a_flat = a.flatten()
    f_flat = f.flatten()

    # Coefficient matrix and right-hand side vector
    A = np.zeros((n ** 2, n ** 2))
    b = f_flat.copy()

    for i in range(n):
        for j in range(n):
            index = i * n + j
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                A[index, index] = 1  # Boundary condition
                b[index] = 0  # Enforce u=0 on the boundary
            else:
                # Fill stencil for inner grid points
                A[index, index] = -4 * a[i, j] / dx ** 2
                A[index, index - 1] = a[i, j - 1] / dx ** 2  # Left neighbor
                A[index, index + 1] = a[i, j + 1] / dx ** 2  # Right neighbor
                A[index, index - n] = a[i - 1, j] / dx ** 2  # Top neighbor
                A[index, index + n] = a[i + 1, j] / dx ** 2  # Bottom neighbor

    # Solve the linear system
    u_flat = np.linalg.solve(A, b)
    return u_flat.reshape((n, n))


# Generate and solve for all combinations of a_0 and alpha
a_values = []
u_values = []
f = np.ones((grid_size, grid_size))  # Source term

for a0 in range(1, Stop_a_0 + 1):
    for alpha in alpha_range:
        a = a0 * np.exp(-alpha * (X1 ** 2 + X2 ** 2))
        u = solve_darcy_with_boundary(a, f, dx, grid_size)
        a_values.append(a)
        u_values.append(u)

# Convert to PyTorch tensors
a_tensor = torch.tensor(a_values, dtype=torch.float32)
u_tensor = torch.tensor(u_values, dtype=torch.float32)

# Save tensors in a dictionary
data = {"x": a_tensor, "y": u_tensor}
torch.save(data, "darcy_test_32.pt")
print("Tensors saved to 'darcy_test_32.pt'")

# Visualization for a0=5 and alpha=1
a0_vis = 5
alpha_vis = 1
a_vis = a0_vis * np.exp(-alpha_vis * (X1 ** 2 + X2 ** 2))
u_vis = solve_darcy_with_boundary(a_vis, f, dx, grid_size)

# Create figures
fig_a = plt.figure()
ax_a = fig_a.add_subplot(111, projection='3d')
ax_a.plot_surface(X1, X2, a_vis, cmap='viridis')
ax_a.set_title("Diffusion Coefficient a(x)")
ax_a.set_xlabel("x1")
ax_a.set_ylabel("x2")
ax_a.set_zlabel("a(x)")

fig_u = plt.figure()
ax_u = fig_u.add_subplot(111, projection='3d')
ax_u.plot_surface(X1, X2, u_vis, cmap='viridis')
ax_u.set_title("Solution u(x)")
ax_u.set_xlabel("x1")
ax_u.set_ylabel("x2")
ax_u.set_zlabel("u(x)")


# Animation functions for spinning charts
def update_a(angle):
    ax_a.view_init(elev=30, azim=angle)


def update_u(angle):
    ax_u.view_init(elev=30, azim=angle)


# Create animations
ani_a = FuncAnimation(fig_a, update_a, frames=360, interval=50)
ani_u = FuncAnimation(fig_u, update_u, frames=360, interval=50)

# Save animations to files
ani_a.save("diffusion_coefficient_rotation.gif", writer='pillow', fps=20)
ani_u.save("solution_rotation.gif", writer='pillow', fps=20)

print("Animations saved as 'diffusion_coefficient_rotation.gif' and 'solution_rotation.gif'")
