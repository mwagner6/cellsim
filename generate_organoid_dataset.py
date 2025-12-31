"""
Generate a dataset of organoid images from multiple random viewing angles.

For each organoid:
- Generate random geometric parameters (axes, scales for lumen/necrotic/fringe)
- For each angle, randomize orientation and capture -z microscope image
- Save images and metadata to organized directory structure

Usage:
    python generate_organoid_dataset.py --n_cells 10 --n_angles 5 --output_dir organoid_dataset
"""

import numpy as np
import argparse
import os
from pathlib import Path
import csv
from PIL import Image

from simulation import Simulation


def random_axes_within_bounds(N=128, max_scale_fringe=0.5):
    """
    Generate random ellipsoid axes that fit within the volume regardless of rotation.

    To ensure the ellipsoid fits when rotated arbitrarily, we need the largest axis
    (including fringe) to fit within a sphere inscribed in the cube.

    Args:
        N: Volume size
        max_scale_fringe: Maximum fringe scale (e.g., 0.5 means fringe adds up to 50% to axes)

    Returns:
        axes: [ax, ay, az] semi-axis lengths
    """
    # Radius of inscribed sphere in cube of size N
    max_radius = N / 2

    # Account for potential fringe expansion
    effective_max_radius = max_radius / (1 + max_scale_fringe)

    # Generate random axes, ensuring largest axis fits
    # Use a reasonable range for organoid-like shapes
    min_axis = 10  # Minimum axis length
    max_axis = effective_max_radius * 0.9  # Leave some margin

    # Generate three axes with some variation
    # Make sure they're reasonable organoid proportions
    base_size = np.random.uniform(min_axis, max_axis)

    # Random aspect ratios (between 0.3 and 1.0 for each axis relative to base)
    ratios = np.random.uniform(0.4, 1.0, size=3)
    axes = base_size * ratios

    return axes.tolist()


def random_organoid_params():
    """
    Generate random parameters for an organoid.

    Returns:
        dict with keys: axes, scale_lumen, scale_necrotic, scale_fringe
    """
    # At least one of lumen or necrotic must be > 0
    use_lumen = np.random.choice([True, False])

    if use_lumen:
        scale_lumen = np.random.uniform(0.1, 0.6)
        scale_necrotic = 0.0 if np.random.random() < 0.5 else np.random.uniform(0.1, 0.4)
    else:
        scale_lumen = 0.0
        scale_necrotic = np.random.uniform(0.1, 0.6)

    # Fringe is optional
    scale_fringe = np.random.uniform(0.0, 0.3)

    # Generate axes that will fit in volume
    axes = random_axes_within_bounds(N=128, max_scale_fringe=scale_fringe)

    return {
        "axes": axes,
        "scale_lumen": scale_lumen,
        "scale_necrotic": scale_necrotic,
        "scale_fringe": scale_fringe
    }


def random_orientation():
    """
    Generate random orientation vectors for the ellipsoid.

    Returns:
        axis1_dir, axis2_dir: Two random orthogonal direction vectors
    """
    # Generate random axis1_dir (normalized)
    axis1_dir = np.random.randn(3)
    axis1_dir = axis1_dir / np.linalg.norm(axis1_dir)

    # Generate random axis2_dir and orthogonalize
    axis2_dir = np.random.randn(3)
    axis2_dir = axis2_dir - np.dot(axis2_dir, axis1_dir) * axis1_dir
    axis2_dir = axis2_dir / np.linalg.norm(axis2_dir)

    return axis1_dir.tolist(), axis2_dir.tolist()


def setup_simulation_and_microscope(N=128, wavelength=0.65, n_photons=100000000, batch_size=50000000):
    """
    Set up simulation with beam generator and microscope matching taichi_napari.ipynb.

    Returns:
        sim: Configured Simulation object (without organoid)
    """
    sim = Simulation(N=N, wavelength=wavelength, n_photons=n_photons, batch_size=batch_size)

    # Beam parameters from notebook
    central_point = (64, 64, 64)
    distance = np.sqrt(2) * 64 + 1
    theta = 0
    phi = np.pi / 2
    mask_diameter = 1.5 * np.sqrt(128**2)
    sigma_a = 256
    sigma_b = 256
    divergence_a = -0.02
    divergence_b = -0.02
    divergence_sigma_a = 0.005
    divergence_sigma_b = 0.005

    sim.init_beam_generator(
        central_point, distance, theta, phi, mask_diameter,
        sigma_a, sigma_b, divergence_a, divergence_b,
        divergence_sigma_a, divergence_sigma_b
    )

    # Initialize -z microscope (from notebook)
    sim.init_microscope(
        observation_face="-z",
        volume_size=N,
        voxel_size=1.0,
        NA=0.15,
        magnification=1,
        n_medium=1.33,
        sensor_size=(N, N),
        pixel_size=1.0,
        wavelength=550,
        focal_depth=64.0
    )

    return sim


def save_organoid_metadata(cell_dir, cell_params, cell_id):
    """Save organoid parameters to CSV file."""
    metadata_path = cell_dir / "metadata.csv"

    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["cell_id", cell_id])
        writer.writerow(["axis_x", cell_params["axes"][0]])
        writer.writerow(["axis_y", cell_params["axes"][1]])
        writer.writerow(["axis_z", cell_params["axes"][2]])
        writer.writerow(["scale_lumen", cell_params["scale_lumen"]])
        writer.writerow(["scale_necrotic", cell_params["scale_necrotic"]])
        writer.writerow(["scale_fringe", cell_params["scale_fringe"]])

    print(f"  Saved metadata to {metadata_path}")


def save_angle_metadata(cell_dir, angle_id, axis1_dir, axis2_dir):
    """Save orientation parameters for a specific angle."""
    angles_file = cell_dir / "angles.csv"

    # Create file with header if it doesn't exist
    file_exists = angles_file.exists()

    with open(angles_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["angle_id", "axis1_x", "axis1_y", "axis1_z",
                           "axis2_x", "axis2_y", "axis2_z"])
        writer.writerow([
            angle_id,
            axis1_dir[0], axis1_dir[1], axis1_dir[2],
            axis2_dir[0], axis2_dir[1], axis2_dir[2]
        ])


def main():
    parser = argparse.ArgumentParser(
        description='Generate organoid image dataset from multiple viewing angles'
    )
    parser.add_argument('--n_cells', type=int, default=10,
                        help='Number of different organoids to generate (default: 10)')
    parser.add_argument('--n_angles', type=int, default=5,
                        help='Number of random angles per organoid (default: 5)')
    parser.add_argument('--output_dir', type=str, default='organoid_dataset',
                        help='Output directory for dataset (default: organoid_dataset)')
    parser.add_argument('--n_photons', type=int, default=100000000,
                        help='Total number of photons (default: 100000000)')
    parser.add_argument('--batch_size', type=int, default=50000000,
                        help='Photon batch size (default: 50000000)')
    parser.add_argument('--N', type=int, default=128,
                        help='Volume size (default: 128)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Generating organoid dataset:")
    print(f"  {args.n_cells} organoids")
    print(f"  {args.n_angles} angles per organoid")
    print(f"  Output directory: {output_dir}")
    print(f"  Total images: {args.n_cells * args.n_angles}")
    print()

    # Simulation parameters
    MAX_STEPS = round((np.sqrt(2) * 64 + 1) * 10)
    STEP_LENGTH = 0.5

    # Generate dataset
    for cell_id in range(args.n_cells):
        print(f"=== Organoid {cell_id + 1}/{args.n_cells} ===")

        # Create directory for this cell
        cell_dir = output_dir / f"cell_{cell_id:03d}"
        cell_dir.mkdir(exist_ok=True)

        # Generate random organoid parameters
        cell_params = random_organoid_params()
        print(f"  Axes: [{cell_params['axes'][0]:.1f}, {cell_params['axes'][1]:.1f}, {cell_params['axes'][2]:.1f}]")
        print(f"  Lumen: {cell_params['scale_lumen']:.2f}, Necrotic: {cell_params['scale_necrotic']:.2f}, Fringe: {cell_params['scale_fringe']:.2f}")

        # Save metadata
        save_organoid_metadata(cell_dir, cell_params, cell_id)

        # Generate images from different angles
        for angle_id in range(args.n_angles):
            print(f"  Angle {angle_id + 1}/{args.n_angles}...", end=" ")

            # Create new simulation for this angle
            sim = setup_simulation_and_microscope(
                N=args.N,
                n_photons=args.n_photons,
                batch_size=args.batch_size
            )

            # Generate random orientation
            axis1_dir, axis2_dir = random_orientation()

            # Create organoid with random orientation
            sim.organoidGen(
                center=[64, 64, 64],
                axes=cell_params["axes"],
                axis1_dir=axis1_dir,
                axis2_dir=axis2_dir,
                scale_lumen=cell_params["scale_lumen"],
                scale_necrotic=cell_params["scale_necrotic"],
                scale_fringe=cell_params["scale_fringe"]
            )

            # Run simulation
            sim.simulation_loop(MAX_STEPS, STEP_LENGTH, track=False, n_tracked=0)

            # Get microscope image (-z face, index 0)
            image = sim.defocus_image(0)

            # Normalize and save as 16-bit grayscale PNG
            if image.max() > 0:
                # Normalize to 16-bit range
                image_normalized = (image / image.max() * 65535).astype(np.uint16)
            else:
                image_normalized = np.zeros_like(image, dtype=np.uint16)

            # Save image
            image_path = cell_dir / f"angle_{angle_id:03d}.png"
            Image.fromarray(image_normalized).save(image_path)

            # Save angle metadata
            save_angle_metadata(cell_dir, angle_id, axis1_dir, axis2_dir)

            print(f"Saved (intensity range: [{image.min():.2e}, {image.max():.2e}])")

        print()

    print(f"Dataset generation complete!")
    print(f"Total images generated: {args.n_cells * args.n_angles}")
    print(f"Saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
