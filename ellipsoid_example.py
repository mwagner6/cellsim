"""
Example script demonstrating ellipsoid geometry with arbitrary orientations.

Shows how to create axis-aligned and rotated ellipsoids with different shapes.
"""

from simulation import Simulation
import numpy as np

# Configuration
N = 128
wavelength = 0.65

# Create simulation
sim = Simulation(N=N, wavelength=wavelength, n_photons=100000, batch_size=100000)

# Initialize material types
types = [
    {"name": "nucleus", "rel_scale": 1.0, "rel_density": 0.6, "rel_pigment": 0.05, "r_index": 1.38},
    {"name": "cytoplasm", "rel_scale": 1.2, "rel_density": 0.4, "rel_pigment": 0.02, "r_index": 1.36},
    {"name": "membrane", "rel_scale": 0.8, "rel_density": 0.8, "rel_pigment": 0.1, "r_index": 1.40},
]
sim.init_types(types, scatter_prec=10000)

# Initialize ellipsoids with various orientations

# Example 1: Axis-aligned ellipsoid (default orientation)
ellipsoid_aligned = {
    "center": [30, 30, 30],
    "axes": [[20, 15, 10]],  # Semi-axes along x, y, z
    "types": ["cytoplasm"]
}

# Example 2: Ellipsoid rotated 45° in XY plane
ellipsoid_rotated_xy = {
    "center": [64, 64, 64],
    "axes": [[30, 15, 10]],  # Major axis along axis1_dir
    "types": ["cytoplasm"],
    "axis1_dir": [1, 1, 0],   # 45° in XY plane
    "axis2_dir": [-1, 1, 0]   # Perpendicular in XY plane
}

# Example 3: Ellipsoid tilted in XZ plane
ellipsoid_tilted_xz = {
    "center": [30, 64, 64],
    "axes": [[25, 12, 8]],
    "types": ["cytoplasm"],
    "axis1_dir": [1, 0, 1],   # Tilted in XZ plane
    "axis2_dir": [0, 1, 0]    # Along Y axis
}

# Example 4: Fiber-like ellipsoid along arbitrary direction
# Pointing from [64,64,20] to [64,64,108] (along Z axis)
ellipsoid_fiber_z = {
    "center": [64, 30, 64],
    "axes": [[40, 5, 5]],  # Very elongated
    "types": ["cytoplasm"],
    "axis1_dir": [0, 0, 1],   # Along Z
    "axis2_dir": [1, 0, 0]    # Along X
}

# Example 5: Fiber pointing along diagonal
ellipsoid_fiber_diagonal = {
    "center": [98, 64, 64],
    "axes": [[35, 5, 5]],
    "types": ["cytoplasm"],
    "axis1_dir": [1, 1, 1],   # Along space diagonal
    "axis2_dir": [1, -1, 0]   # Perpendicular
}

# Example 6: Multi-layer cell with rotation
ellipsoid_multilayer = {
    "center": [64, 98, 64],
    "axes": [
        [8, 6, 6],         # Nucleus (slightly elongated)
        [15, 12, 10],      # Cytoplasm
        [15.5, 12.5, 10.5] # Membrane
    ],
    "types": ["nucleus", "cytoplasm", "membrane"],
    "axis1_dir": [1, 0.5, 0],  # Tilted orientation
    "axis2_dir": [0, 1, 0]
}

# Initialize all ellipsoids
sim.initEllipsoids([
    ellipsoid_aligned,
    ellipsoid_rotated_xy,
    ellipsoid_tilted_xz,
    ellipsoid_fiber_z,
    ellipsoid_fiber_diagonal,
    ellipsoid_multilayer
])

# Compare with spheres (you can mix both!)
spheres = [
    {
        "center": [20, 20, 20],
        "radii": [10],
        "types": ["nucleus"]
    }
]
sim.initSpheres(spheres)

# Visualize the volume
sim.volume_to_np()
print("\n=== Volume Statistics ===")
print(f"Non-empty voxels: {(sim.volume_np > 0).sum()}")
print(f"Material type distribution:")
for i, type_info in enumerate(types):
    count = (sim.volume_np == i + 1).sum()
    if count > 0:
        print(f"  {type_info['name']}: {count} voxels")

# You can project to see cross-sections
projections = sim.project_volume(["x", "y", "z"])

print("\n=== Projection Shapes ===")
for axis, proj in projections.items():
    print(f"{axis}-axis projection: {proj.shape}")

# To visualize, you could use napari or matplotlib:
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(projections['z'][0, :, :], cmap='viridis')
# axes[0].set_title('Z-projection (XY view)')
# axes[1].imshow(projections['y'][:, 0, :], cmap='viridis')
# axes[1].set_title('Y-projection (XZ view)')
# axes[2].imshow(projections['x'][:, :, 0], cmap='viridis')
# axes[2].set_title('X-projection (YZ view)')
# plt.show()
