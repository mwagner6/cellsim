"""
Generate test images of circles and ellipses for DINOv3 testing.

Usage:
    python generate_test_images.py --n_circles 50 --n_ellipses 50 --axis_ratio 0.6
"""

import numpy as np
from PIL import Image, ImageDraw
import argparse
import os


def generate_circle_image(image_size=(512, 512), min_radius=30, max_radius=150):
    """
    Generate an image with a single circle at random position and size.

    Args:
        image_size: Size of output image (width, height)
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius

    Returns:
        PIL Image
    """
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)

    # Random radius
    radius = np.random.uniform(min_radius, max_radius)

    # Random center position (ensure circle fits in image)
    cx = np.random.uniform(radius, image_size[0] - radius)
    cy = np.random.uniform(radius, image_size[1] - radius)

    # Draw circle
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(bbox, fill='black', outline='black')

    return img


def generate_ellipse_image(image_size=(512, 512), axis_ratio=0.6,
                          min_major_axis=60, max_major_axis=300):
    """
    Generate an image with a single ellipse at random position, size, and rotation.
    Ensures the entire ellipse is always within image bounds.

    Args:
        image_size: Size of output image (width, height)
        axis_ratio: Ratio of minor axis to major axis (e.g., 0.6 means minor = 0.6 * major)
        min_major_axis: Minimum major axis length
        max_major_axis: Maximum major axis length

    Returns:
        PIL Image
    """
    img = Image.new('RGB', image_size, color='white')

    # Random major axis length
    major_axis = np.random.uniform(min_major_axis, max_major_axis)
    minor_axis = major_axis * axis_ratio

    # Random rotation angle (0 to 360 degrees)
    angle = np.random.uniform(0, 360)

    # Calculate the bounding box of the rotated ellipse
    # The maximum extent is when the ellipse is at 45 degrees
    angle_rad = np.radians(angle)
    cos_a = np.abs(np.cos(angle_rad))
    sin_a = np.abs(np.sin(angle_rad))

    # Axis-aligned bounding box of rotated ellipse
    half_width = np.sqrt((major_axis/2 * cos_a)**2 + (minor_axis/2 * sin_a)**2)
    half_height = np.sqrt((major_axis/2 * sin_a)**2 + (minor_axis/2 * cos_a)**2)

    # Random center position ensuring the rotated ellipse fits
    cx = np.random.uniform(half_width, image_size[0] - half_width)
    cy = np.random.uniform(half_height, image_size[1] - half_height)

    # Create temporary image for rotation
    temp_size = (int(image_size[0] * 2), int(image_size[1] * 2))
    temp_img = Image.new('RGB', temp_size, color='white')
    temp_draw = ImageDraw.Draw(temp_img)

    # Draw ellipse at center of temp image
    temp_cx = temp_size[0] / 2
    temp_cy = temp_size[1] / 2
    bbox = [temp_cx - major_axis/2, temp_cy - minor_axis/2,
            temp_cx + major_axis/2, temp_cy + minor_axis/2]
    temp_draw.ellipse(bbox, fill='black', outline='black')

    # Rotate the temporary image
    rotated = temp_img.rotate(angle, expand=False, fillcolor='white')

    # Crop to final size, centered at the desired position
    left = temp_size[0] / 2 - cx
    top = temp_size[1] / 2 - cy
    right = left + image_size[0]
    bottom = top + image_size[1]

    final_img = rotated.crop((left, top, right, bottom))

    return final_img


def main():
    parser = argparse.ArgumentParser(description='Generate circle and ellipse test images')
    parser.add_argument('--n_circles', type=int, default=50,
                        help='Number of circle images to generate')
    parser.add_argument('--n_ellipses', type=int, default=50,
                        help='Number of ellipse images to generate')
    parser.add_argument('--axis_ratio', type=float, default=0.6,
                        help='Ratio of minor to major axis for ellipses (default: 0.6)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Size of output images (default: 512)')
    parser.add_argument('--output_dir', type=str, default='testimages',
                        help='Output directory (default: testimages)')
    parser.add_argument('--min_radius', type=int, default=30,
                        help='Minimum radius/axis length (default: 30)')
    parser.add_argument('--max_radius', type=int, default=150,
                        help='Maximum radius/major axis length (default: 150)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    image_size = (args.image_size, args.image_size)

    print(f"Generating {args.n_circles} circle images...")
    for i in range(args.n_circles):
        img = generate_circle_image(
            image_size=image_size,
            min_radius=args.min_radius,
            max_radius=args.max_radius
        )
        filepath = os.path.join(args.output_dir, f'circle_{i}.jpg')
        img.save(filepath, quality=95)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{args.n_circles} circles")

    print(f"\nGenerating {args.n_ellipses} ellipse images (axis ratio: {args.axis_ratio})...")
    for i in range(args.n_ellipses):
        img = generate_ellipse_image(
            image_size=image_size,
            axis_ratio=args.axis_ratio,
            min_major_axis=args.min_radius * 2,
            max_major_axis=args.max_radius * 2
        )
        filepath = os.path.join(args.output_dir, f'ellipse_{i}.jpg')
        img.save(filepath, quality=95)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{args.n_ellipses} ellipses")

    print(f"\nDone! Generated {args.n_circles + args.n_ellipses} images in '{args.output_dir}/'")
    print(f"  - {args.n_circles} circles")
    print(f"  - {args.n_ellipses} ellipses (axis ratio: {args.axis_ratio})")


if __name__ == "__main__":
    main()
