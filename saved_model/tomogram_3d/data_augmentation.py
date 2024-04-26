import numpy as np
from scipy.ndimage import rotate, zoom

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def random_flip_3d(data, seed=123):
    # Generate random integers to determine flip axes
    
    np.random.seed(seed)
    flip_axes = np.random.randint(0, 2, 3) * 2 - 1  # Either -1 or 1

    # Apply flips
    flipped_data = np.copy(data)
    for axis in range(3):
        if flip_axes[axis] == -1:
            flipped_data = np.flip(flipped_data, axis=axis)
    return flipped_data


def random_zoom_3d(data, zoom_factor=0.9):
    
    # Zoom the data to the specified zoom factor
    zoomed_data = zoom(data, zoom_factor, mode='nearest')

    # Compute the scaling factors to resize back to the original dimensions
    scaling_factors = [1 / factor for factor in zoom_factor]

    # Resize the zoomed data to match the original dimensions
    resized_data = zoom(zoomed_data, scaling_factors, mode='nearest')

    return resized_data


def random_rotation_3d(data, angle_range=(-10, 10), axes=None, seed=123):
    # Generate random rotation angles for each axis
    if axes is None:
        axes = [(1, 0), (0, 1), (0, 2)]  # Rotation can occur around any axis by default
    
    rotation_angles = np.random.uniform(angle_range[0], angle_range[1], size=(3,))

    # Perform random rotation around each axis
    rotated_data = np.copy(data)
    for axis, angle in zip(axes, rotation_angles):
        rotated_data = rotate(rotated_data, angle, axes=axis, reshape=False, mode='nearest')

    return rotated_data



shape = (3, 3, 3)
arr = np.zeros(shape=shape)

for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            arr[i, j, k] = 9*i + 3*j + k

# print(arr)

zoomed_arr = random_zoom_3d(arr)
print(zoomed_arr)

# rotated_arr = random_rotation_3d(arr)
# print(rotated_arr)

# Generate a sample 3D numpy array
data = zoomed_arr

# Create a figure and a 3D Axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Get the shape of the data
x_size, y_size, z_size = data.shape

# Create arrays for the x, y, and z coordinates
x = np.arange(0, x_size)
y = np.arange(0, y_size)
z = np.arange(0, z_size)

# Create a meshgrid from the x, y, and z coordinates
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Plot the 3D array
ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=data.flatten(), cmap='viridis')

# Add labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
