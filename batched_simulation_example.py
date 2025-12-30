"""
Example script demonstrating batched photon simulation with microscope image accumulation.

This shows the proper workflow:
1. Initialize simulation and microscopes BEFORE running simulation
2. Call simulation_loop() which automatically accumulates photons to microscope images
3. Retrieve final images after all batches complete
"""

from simulation import Simulation

# Configuration
N = 128
wavelength = 0.65
total_photons = 1000000  # Total photons to simulate
batch_size = 100000      # Photons per batch (to avoid VRAM overflow)

# Create simulation
sim = Simulation(N=N, wavelength=wavelength, n_photons=total_photons, batch_size=batch_size)

# Initialize material types
types = [
    {"name": "cell", "rel_scale": 1.0, "rel_density": 0.5, "rel_pigment": 0.1, "r_index": 1.37},
]
sim.init_types(types, scatter_prec=10000)

# Initialize geometry (example: single sphere)
spheres = [
    {"center": [64, 64, 64], "radii": [30], "types": ["cell"]}
]
sim.initSpheres(spheres)

# Initialize beam generator
sim.init_beam_generator(
    central_point=[64, 64, 0],
    distance=100,
    theta=0,
    phi=0,
    mask_diameter=100,
    sigma_a=20,
    sigma_b=20,
    divergence_a=0,
    divergence_b=0,
    divergence_sigma_a=0.01,
    divergence_sigma_b=0.01
)

# IMPORTANT: Initialize microscopes BEFORE running simulation
print("\n=== Initializing Microscopes ===")
sim.init_microscope(
    observation_face='+z',
    volume_size=N,
    voxel_size=1.0,
    NA=0.65,
    magnification=20.0,
    n_medium=1.33,
    sensor_size=(128, 128),
    pixel_size=6.5,
    wavelength=wavelength * 1000,  # Convert to nm
    focal_depth=64.0
)

# Run batched simulation
# This will automatically:
# - Reset microscope images
# - Run multiple batches
# - Accumulate photons to microscope images after each batch
print("\n=== Running Batched Simulation ===")
sim.simulation_loop(
    max_steps=1000,
    step_length=0.1,
    track=False,
    n_tracked=0
)

# Get final accumulated images
print("\n=== Retrieving Final Images ===")
image = sim.defocus_image(0)
print(f"Final image shape: {image.shape}")
print(f"Final image intensity range: [{image.min():.2e}, {image.max():.2e}]")

# You can now save or display the image
# import matplotlib.pyplot as plt
# plt.imshow(image, cmap='hot')
# plt.colorbar()
# plt.title('Accumulated Defocused Microscope Image')
# plt.show()
