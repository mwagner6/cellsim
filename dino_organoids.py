"""
Analyze organoid dataset using DINOv3 embeddings.

For each organoid:
- Extract DINOv3 CLS token embeddings for all angle images
- Save embeddings to CSV in organoid's folder

Generate visualizations:
- 2D PCA plot with different color per organoid
- Shows whether DINOv3 can distinguish different organoids

Usage:
    python dino_organoids.py --dataset_dir organoid_dataset --output_plot organoid_pca.png
"""

import numpy as np
import argparse
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from tqdm import tqdm


def load_model(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"):
    """Load DINOv3 model and processor."""
    print(f"Loading model: {model_name}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")
    return processor, model, device


def extract_embedding(image_path, processor, model, device):
    """
    Extract CLS token embedding from a single image.

    Args:
        image_path: Path to image file
        processor: DINOv3 image processor
        model: DINOv3 model
        device: torch device

    Returns:
        embedding: numpy array of shape (embedding_dim,)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)

    # Get CLS token (first token)
    cls_token = outputs.last_hidden_state[:, 0, :]

    # Convert to numpy
    embedding = cls_token.cpu().numpy().squeeze()

    return embedding


def process_organoid_dataset(dataset_dir, processor, model, device):
    """
    Process all organoids in the dataset directory.

    Args:
        dataset_dir: Path to dataset directory containing cell_XXX folders
        processor: DINOv3 processor
        model: DINOv3 model
        device: torch device

    Returns:
        all_embeddings: list of embeddings for all images
        organoid_ids: list of organoid IDs corresponding to each embedding
        cell_dirs: list of cell directory paths
    """
    dataset_path = Path(dataset_dir)

    # Find all cell directories
    cell_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('cell_')])

    if len(cell_dirs) == 0:
        raise ValueError(f"No cell directories found in {dataset_dir}")

    print(f"\nFound {len(cell_dirs)} organoids in dataset")

    all_embeddings = []
    organoid_ids = []

    # Process each organoid
    for cell_dir in tqdm(cell_dirs, desc="Processing organoids"):
        cell_id = cell_dir.name

        # Find all angle images
        image_files = sorted(cell_dir.glob("angle_*.png"))

        if len(image_files) == 0:
            print(f"Warning: No angle images found in {cell_dir}")
            continue

        # Extract embeddings for all angles
        embeddings = []
        angle_ids = []

        for image_path in image_files:
            # Extract angle ID from filename (e.g., "angle_003.png" -> 3)
            angle_id = int(image_path.stem.split('_')[1])

            # Get embedding
            embedding = extract_embedding(image_path, processor, model, device)
            embeddings.append(embedding)
            angle_ids.append(angle_id)

            # Track for global analysis
            all_embeddings.append(embedding)
            organoid_ids.append(cell_id)

        # Save embeddings to CSV in organoid's folder
        embeddings_array = np.array(embeddings)
        save_embeddings_csv(cell_dir, embeddings_array, angle_ids)

        print(f"  {cell_id}: Processed {len(embeddings)} angles")

    return np.array(all_embeddings), organoid_ids, cell_dirs


def save_embeddings_csv(cell_dir, embeddings, angle_ids):
    """
    Save embeddings to CSV file in organoid's directory.

    Args:
        cell_dir: Path to cell directory
        embeddings: numpy array of shape (n_angles, embedding_dim)
        angle_ids: list of angle IDs
    """
    csv_path = cell_dir / "embeddings.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ["angle_id"] + [f"dim_{i}" for i in range(embeddings.shape[1])]
        writer.writerow(header)

        # Data rows
        for angle_id, embedding in zip(angle_ids, embeddings):
            row = [angle_id] + embedding.tolist()
            writer.writerow(row)


def plot_organoid_pca(embeddings, organoid_ids, save_path=None):
    """
    Plot 2D PCA of embeddings with different color per organoid.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        organoid_ids: list of organoid IDs (e.g., ['cell_000', 'cell_000', 'cell_001', ...])
        save_path: path to save plot (optional)
    """
    # Perform PCA
    print("\nPerforming PCA...")
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)

    print(f"PCA variance explained: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # Get unique organoids
    unique_organoids = sorted(list(set(organoid_ids)))
    n_organoids = len(unique_organoids)

    # Create color map
    cmap = plt.cm.get_cmap('tab20' if n_organoids <= 20 else 'hsv')
    colors = [cmap(i / n_organoids) for i in range(n_organoids)]
    organoid_to_color = {org: colors[i] for i, org in enumerate(unique_organoids)}

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each organoid with different color
    for organoid_id in unique_organoids:
        mask = np.array([oid == organoid_id for oid in organoid_ids])
        ax.scatter(
            pca_embeddings[mask, 0],
            pca_embeddings[mask, 1],
            c=[organoid_to_color[organoid_id]],
            label=organoid_id,
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidths=0.5
        )

    # Add labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('DINOv3 Embeddings: Organoid Discrimination\n(Each color = different organoid, each point = viewing angle)',
                 fontsize=14, fontweight='bold')

    # Add legend (might be large if many organoids)
    if n_organoids <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        # Too many organoids for readable legend
        ax.text(1.02, 0.5, f'{n_organoids} organoids\n(legend omitted)',
                transform=ax.transAxes, va='center', fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPCA plot saved to: {save_path}")

    plt.show()


def compute_organoid_separation_metrics(embeddings, organoid_ids):
    """
    Compute metrics to quantify how well different organoids are separated.

    Args:
        embeddings: numpy array of shape (n_samples, n_dims)
        organoid_ids: list of organoid IDs
    """
    from sklearn.metrics.pairwise import euclidean_distances

    unique_organoids = sorted(list(set(organoid_ids)))
    n_organoids = len(unique_organoids)

    # Compute centroids for each organoid
    centroids = {}
    for organoid_id in unique_organoids:
        mask = np.array([oid == organoid_id for oid in organoid_ids])
        centroids[organoid_id] = embeddings[mask].mean(axis=0)

    # Compute inter-organoid distances (between centroids)
    centroid_matrix = np.array([centroids[oid] for oid in unique_organoids])
    inter_distances = euclidean_distances(centroid_matrix, centroid_matrix)

    # Compute intra-organoid distances (within same organoid)
    intra_distances = []
    for organoid_id in unique_organoids:
        mask = np.array([oid == organoid_id for oid in organoid_ids])
        organoid_embeddings = embeddings[mask]
        if len(organoid_embeddings) > 1:
            dists = euclidean_distances(organoid_embeddings, organoid_embeddings)
            # Get upper triangle (excluding diagonal)
            intra_distances.extend(dists[np.triu_indices_from(dists, k=1)])

    # Get inter-organoid distances (excluding diagonal)
    inter_distances_flat = inter_distances[np.triu_indices_from(inter_distances, k=1)]

    print("\n=== Organoid Separation Metrics ===")
    print(f"Number of organoids: {n_organoids}")
    print(f"\nIntra-organoid distances (same organoid, different angles):")
    print(f"  Mean: {np.mean(intra_distances):.4f}")
    print(f"  Std:  {np.std(intra_distances):.4f}")
    print(f"  Min:  {np.min(intra_distances):.4f}")
    print(f"  Max:  {np.max(intra_distances):.4f}")

    print(f"\nInter-organoid distances (different organoids):")
    print(f"  Mean: {np.mean(inter_distances_flat):.4f}")
    print(f"  Std:  {np.std(inter_distances_flat):.4f}")
    print(f"  Min:  {np.min(inter_distances_flat):.4f}")
    print(f"  Max:  {np.max(inter_distances_flat):.4f}")

    # Compute separation ratio (higher is better)
    separation_ratio = np.mean(inter_distances_flat) / np.mean(intra_distances) if len(intra_distances) > 0 else float('inf')
    print(f"\nSeparation ratio (inter/intra): {separation_ratio:.4f}")
    print("  (Higher values indicate better organoid discrimination)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze organoid dataset using DINOv3 embeddings'
    )
    parser.add_argument('--dataset_dir', type=str, default='organoid_dataset',
                        help='Directory containing organoid dataset (default: organoid_dataset)')
    parser.add_argument('--model', type=str, default='facebook/dinov3-vitl16-pretrain-lvd1689m',
                        help='DINOv3 model to use')
    parser.add_argument('--output_plot', type=str, default='organoid_pca.png',
                        help='Output path for PCA plot (default: organoid_pca.png)')
    parser.add_argument('--save_global', type=str, default=None,
                        help='Path to save global embeddings array (optional, .npz format)')

    args = parser.parse_args()

    # Load model
    processor, model, device = load_model(args.model)

    # Process dataset
    embeddings, organoid_ids, cell_dirs = process_organoid_dataset(
        args.dataset_dir, processor, model, device
    )

    print(f"\n=== Processing Complete ===")
    print(f"Total embeddings extracted: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of organoids: {len(set(organoid_ids))}")

    # Compute separation metrics
    compute_organoid_separation_metrics(embeddings, organoid_ids)

    # Plot PCA
    plot_organoid_pca(embeddings, organoid_ids, save_path=args.output_plot)

    # Save global embeddings if requested
    if args.save_global:
        np.savez(args.save_global,
                 embeddings=embeddings,
                 organoid_ids=organoid_ids,
                 cell_dirs=[str(d) for d in cell_dirs])
        print(f"\nGlobal embeddings saved to: {args.save_global}")


if __name__ == "__main__":
    main()
