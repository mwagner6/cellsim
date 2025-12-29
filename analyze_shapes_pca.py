"""
Analyze circle vs ellipse discrimination using DINOv3 embeddings and PCA.

Usage:
    python analyze_shapes_pca.py --image_dir testimages --model facebook/dinov3-vitl16-pretrain-lvd1689m
"""

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import argparse
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


def load_model(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"):
    """Load DINOv3 model and processor."""
    print(f"Loading model: {model_name}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map="auto")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on: {device}")
    return processor, model, device


def extract_cls_token(image_path, processor, model, device):
    """Extract CLS token embedding from an image."""
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    cls_token = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_token.squeeze()


def process_image_directory(image_dir, processor, model, device):
    """
    Process all images in directory and extract CLS tokens.

    Returns:
        embeddings: numpy array of shape (n_images, embedding_dim)
        labels: list of labels ('circle' or 'ellipse')
        filenames: list of image filenames
    """
    image_dir = Path(image_dir)

    # Get all image files
    circle_files = sorted(image_dir.glob('circle_*.jpg'))
    ellipse_files = sorted(image_dir.glob('ellipse_*.jpg'))

    print(f"Found {len(circle_files)} circle images")
    print(f"Found {len(ellipse_files)} ellipse images")

    all_files = list(circle_files) + list(ellipse_files)
    labels = ['circle'] * len(circle_files) + ['ellipse'] * len(ellipse_files)

    embeddings = []
    filenames = []

    print("\nExtracting embeddings...")
    for img_path in tqdm(all_files):
        cls_token = extract_cls_token(str(img_path), processor, model, device)
        embeddings.append(cls_token)
        filenames.append(img_path.name)

    embeddings = np.array(embeddings)

    return embeddings, labels, filenames


def perform_pca_analysis(embeddings, labels, n_components=2):
    """
    Perform PCA on embeddings and return transformed data.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        labels: list of labels for each sample
        n_components: number of PCA components (default: 2 for visualization)

    Returns:
        pca_embeddings: transformed embeddings
        pca: fitted PCA object
    """
    print(f"\nPerforming PCA (reducing from {embeddings.shape[1]} to {n_components} dimensions)...")

    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)

    explained_var = pca.explained_variance_ratio_
    print(f"Explained variance ratio: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}")
    print(f"Total variance explained: {explained_var.sum():.4f}")

    return pca_embeddings, pca


def plot_pca_results(pca_embeddings, labels, save_path=None):
    """
    Plot PCA results with different colors for circles and ellipses.

    Args:
        pca_embeddings: numpy array of shape (n_samples, 2)
        labels: list of labels ('circle' or 'ellipse')
        save_path: path to save the plot (optional)
    """
    # Convert labels to array for easy indexing
    labels = np.array(labels)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot circles
    circle_mask = labels == 'circle'
    ax.scatter(pca_embeddings[circle_mask, 0],
               pca_embeddings[circle_mask, 1],
               c='blue', label='Circle', alpha=0.6, s=50)

    # Plot ellipses
    ellipse_mask = labels == 'ellipse'
    ax.scatter(pca_embeddings[ellipse_mask, 0],
               pca_embeddings[ellipse_mask, 1],
               c='red', label='Ellipse', alpha=0.6, s=50)

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('DINOv3 Embeddings: Circles vs Ellipses (PCA)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def compute_distance_matrix(embeddings, labels):
    """
    Compute L2 distance matrix between all embeddings.
    Arrange so circles come first, then ellipses.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        labels: list of labels ('circle' or 'ellipse')

    Returns:
        distance_matrix: pairwise L2 distances
        n_circles: number of circle samples
    """
    labels = np.array(labels)

    # Already ordered: circles first, then ellipses
    # Just compute the distance matrix
    distance_matrix = euclidean_distances(embeddings, embeddings)

    n_circles = (labels == 'circle').sum()

    return distance_matrix, n_circles


def plot_distance_heatmap(distance_matrix, n_circles, save_path=None):
    """
    Plot heatmap of L2 distances between embeddings.
    Circles are in the first n_circles rows/cols.

    Args:
        distance_matrix: pairwise distance matrix
        n_circles: number of circle samples (rest are ellipses)
        save_path: path to save the plot (optional)
    """
    n_total = distance_matrix.shape[0]
    n_ellipses = n_total - n_circles

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('L2 Distance', fontsize=12)

    # Add lines to separate circles from ellipses
    ax.axhline(y=n_circles - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=n_circles - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Add labels
    ax.set_xlabel('Image Index', fontsize=12)
    ax.set_ylabel('Image Index', fontsize=12)
    ax.set_title('L2 Distance Matrix (Circles first, then Ellipses)', fontsize=14)

    # Add text annotations for regions
    ax.text(n_circles / 2, -5, 'Circles', ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(n_circles + n_ellipses / 2, -5, 'Ellipses', ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(-5, n_circles / 2, 'Circles', ha='right', va='center', fontsize=12, fontweight='bold', rotation=90)
    ax.text(-5, n_circles + n_ellipses / 2, 'Ellipses', ha='right', va='center', fontsize=12, fontweight='bold', rotation=90)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nDistance heatmap saved to: {save_path}")

    plt.show()


def compute_separation_metrics(pca_embeddings, labels):
    """Compute metrics to quantify how well circles and ellipses are separated."""
    labels = np.array(labels)

    # Get centroids for each class
    circle_centroid = pca_embeddings[labels == 'circle'].mean(axis=0)
    ellipse_centroid = pca_embeddings[labels == 'ellipse'].mean(axis=0)

    # Distance between centroids
    centroid_distance = np.linalg.norm(circle_centroid - ellipse_centroid)

    # Within-class variance
    circle_var = np.var(pca_embeddings[labels == 'circle'], axis=0).mean()
    ellipse_var = np.var(pca_embeddings[labels == 'ellipse'], axis=0).mean()
    avg_within_var = (circle_var + ellipse_var) / 2

    # Separation ratio (higher is better)
    separation_ratio = centroid_distance / np.sqrt(avg_within_var)

    print("\n=== Separation Metrics ===")
    print(f"Centroid distance: {centroid_distance:.4f}")
    print(f"Average within-class variance: {avg_within_var:.4f}")
    print(f"Separation ratio: {separation_ratio:.4f}")
    print(f"Circle centroid (PC1, PC2): ({circle_centroid[0]:.4f}, {circle_centroid[1]:.4f})")
    print(f"Ellipse centroid (PC1, PC2): ({ellipse_centroid[0]:.4f}, {ellipse_centroid[1]:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze shape discrimination using DINOv3 embeddings and PCA'
    )
    parser.add_argument('--image_dir', type=str, default='testimages',
                        help='Directory containing test images (default: testimages)')
    parser.add_argument('--model', type=str, default='facebook/dinov3-vitl16-pretrain-lvd1689m',
                        help='DINOv3 model to use')
    parser.add_argument('--output_plot', type=str, default='pca_analysis.png',
                        help='Output path for PCA plot (default: pca_analysis.png)')
    parser.add_argument('--output_heatmap', type=str, default='distance_heatmap.png',
                        help='Output path for distance heatmap (default: distance_heatmap.png)')
    parser.add_argument('--save_embeddings', type=str, default=None,
                        help='Path to save embeddings and labels (optional)')

    args = parser.parse_args()

    # Load model
    processor, model, device = load_model(args.model)

    # Process all images
    embeddings, labels, filenames = process_image_directory(
        args.image_dir, processor, model, device
    )

    print(f"\nExtracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Compute distance matrix
    distance_matrix, n_circles = compute_distance_matrix(embeddings, labels)

    print(f"\n=== Distance Matrix Stats ===")
    print(f"Matrix shape: {distance_matrix.shape}")
    print(f"Number of circles: {n_circles}")
    print(f"Number of ellipses: {distance_matrix.shape[0] - n_circles}")
    print(f"Min distance: {distance_matrix[distance_matrix > 0].min():.4f}")
    print(f"Max distance: {distance_matrix.max():.4f}")
    print(f"Mean distance: {distance_matrix[distance_matrix > 0].mean():.4f}")

    # Plot distance heatmap
    plot_distance_heatmap(distance_matrix, n_circles, save_path=args.output_heatmap)

    # Perform PCA
    pca_embeddings, pca = perform_pca_analysis(embeddings, labels, n_components=2)

    # Compute separation metrics
    compute_separation_metrics(pca_embeddings, labels)

    # Plot PCA results
    plot_pca_results(pca_embeddings, labels, save_path=args.output_plot)

    # Save embeddings if requested
    if args.save_embeddings:
        np.savez(args.save_embeddings,
                 embeddings=embeddings,
                 pca_embeddings=pca_embeddings,
                 labels=labels,
                 filenames=filenames,
                 distance_matrix=distance_matrix)
        print(f"\nEmbeddings saved to: {args.save_embeddings}")


if __name__ == "__main__":
    main()
