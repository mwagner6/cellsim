"""
Train a CNN to predict organoid geometric parameters from microscopy images.

The model predicts:
- axis_major: Largest semi-axis length
- axis_middle: Middle semi-axis length
- axis_minor: Smallest semi-axis length
- scale_lumen: Lumen scale parameter
- scale_necrotic: Necrotic core scale parameter
- scale_fringe: Fringe scale parameter

Usage:
    # Train model
    python train_organoid_predictor.py --dataset_dir organoid_dataset --mode train --epochs 100

    # Test model
    python train_organoid_predictor.py --dataset_dir organoid_dataset --mode test --model_path organoid_predictor.pth

    # Train and test
    python train_organoid_predictor.py --dataset_dir organoid_dataset --mode both --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import csv
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


class OrganoidDataset(Dataset):
    """
    Dataset for organoid microscopy images and their parameters.

    Each sample is a pair (image, parameters) where:
    - image: microscopy image of organoid from some angle
    - parameters: [axis_major, axis_middle, axis_minor, scale_lumen, scale_necrotic, scale_fringe]
    """

    def __init__(self, dataset_dir, transform=None, normalize_params=True):
        """
        Args:
            dataset_dir: Path to dataset directory containing cell_XXX folders
            transform: Optional image transforms
            normalize_params: Whether to normalize parameters to [0, 1]
        """
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.normalize_params = normalize_params

        # Collect all samples (image_path, parameters)
        self.samples = []
        self.param_stats = None

        # Find all cell directories
        cell_dirs = sorted([d for d in self.dataset_dir.iterdir()
                          if d.is_dir() and d.name.startswith('cell_')])

        for cell_dir in cell_dirs:
            # Read metadata
            metadata_path = cell_dir / "metadata.csv"
            if not metadata_path.exists():
                print(f"Warning: No metadata found for {cell_dir.name}, skipping")
                continue

            params = self._read_metadata(metadata_path)

            # Find all angle images
            image_files = sorted(cell_dir.glob("angle_*.png"))

            for image_path in image_files:
                self.samples.append((image_path, params))

        print(f"Loaded {len(self.samples)} samples from {len(cell_dirs)} organoids")

        # Compute parameter statistics for normalization
        if normalize_params:
            self._compute_param_stats()

    def _read_metadata(self, metadata_path):
        """
        Read metadata CSV and extract parameters.
        Returns parameters sorted as [major, middle, minor, lumen, necrotic, fringe]
        """
        params = {}
        with open(metadata_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    params[row[0]] = float(row[1])

        # Get axes and sort them (major > middle > minor)
        axes = sorted([params['axis_x'], params['axis_y'], params['axis_z']], reverse=True)

        # Return as [major, middle, minor, lumen, necrotic, fringe]
        return np.array([
            axes[0],  # major
            axes[1],  # middle
            axes[2],  # minor
            params['scale_lumen'],
            params['scale_necrotic'],
            params['scale_fringe']
        ], dtype=np.float32)

    def _compute_param_stats(self):
        """Compute mean and std for each parameter for normalization."""
        all_params = np.array([params for _, params in self.samples])
        self.param_stats = {
            'mean': all_params.mean(axis=0),
            'std': all_params.std(axis=0),
            'min': all_params.min(axis=0),
            'max': all_params.max(axis=0)
        }
        print("\n=== Parameter Statistics ===")
        param_names = ['axis_major', 'axis_middle', 'axis_minor',
                      'scale_lumen', 'scale_necrotic', 'scale_fringe']
        for i, name in enumerate(param_names):
            print(f"{name:15s}: mean={self.param_stats['mean'][i]:.3f}, "
                  f"std={self.param_stats['std'][i]:.3f}, "
                  f"min={self.param_stats['min'][i]:.3f}, "
                  f"max={self.param_stats['max'][i]:.3f}")

    def _normalize_params(self, params):
        """Normalize parameters using computed statistics."""
        if self.param_stats is None:
            return params
        # Z-score normalization
        return (params - self.param_stats['mean']) / (self.param_stats['std'] + 1e-8)

    def _denormalize_params(self, params_normalized):
        """Denormalize parameters back to original scale."""
        if self.param_stats is None:
            return params_normalized
        return params_normalized * (self.param_stats['std'] + 1e-8) + self.param_stats['mean']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, params = self.samples[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Normalize parameters if requested
        if self.normalize_params:
            params = self._normalize_params(params)

        return image, torch.tensor(params, dtype=torch.float32)


class OrganoidPredictor(nn.Module):
    """
    CNN regression model to predict organoid parameters from microscopy images.

    Architecture:
    - Multiple convolutional blocks with batch norm and pooling
    - Global average pooling
    - Fully connected layers for regression
    - Output: 6 parameters [major, middle, minor, lumen, necrotic, fringe]
    """

    def __init__(self, input_channels=3, output_dim=6):
        super(OrganoidPredictor, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # Extract features
        x = self.features(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Regression
        x = self.regressor(x)

        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def test_model(model, test_loader, dataset, device, save_dir=None):
    """
    Test the model and generate visualizations.

    Args:
        model: Trained model
        test_loader: Test data loader
        dataset: Dataset object (for denormalization)
        device: torch device
        save_dir: Directory to save plots
    """
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Denormalize if needed
    if dataset.normalize_params:
        predictions = np.array([dataset._denormalize_params(p) for p in predictions])
        targets = np.array([dataset._denormalize_params(t) for t in targets])

    # Compute metrics
    param_names = ['Axis Major', 'Axis Middle', 'Axis Minor',
                  'Scale Lumen', 'Scale Necrotic', 'Scale Fringe']

    print("\n=== Test Results ===")
    print(f"Total samples: {len(predictions)}")
    print("\nPer-parameter metrics:")
    print(f"{'Parameter':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 52)

    overall_metrics = []
    for i, name in enumerate(param_names):
        pred = predictions[:, i]
        true = targets[:, i]

        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true)**2))

        # R² score
        ss_res = np.sum((true - pred)**2)
        ss_tot = np.sum((true - np.mean(true))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        print(f"{name:<20} {mae:>10.4f} {rmse:>10.4f} {r2:>10.4f}")
        overall_metrics.append({'name': name, 'mae': mae, 'rmse': rmse, 'r2': r2})

    # Plot predictions vs ground truth
    plot_predictions(predictions, targets, param_names, save_dir)

    return predictions, targets, overall_metrics


def plot_predictions(predictions, targets, param_names, save_dir=None):
    """
    Plot predicted vs actual values for all parameters.

    Args:
        predictions: Predicted parameters (n_samples, 6)
        targets: Ground truth parameters (n_samples, 6)
        param_names: List of parameter names
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        pred = predictions[:, i]
        true = targets[:, i]

        # Scatter plot
        ax.scatter(true, pred, alpha=0.5, s=20, edgecolors='black', linewidths=0.5)

        # Perfect prediction line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

        # Compute R²
        ss_res = np.sum((true - pred)**2)
        ss_tot = np.sum((true - np.mean(true))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        mae = np.mean(np.abs(pred - true))

        ax.set_xlabel(f'True {name}', fontsize=10)
        ax.set_ylabel(f'Predicted {name}', fontsize=10)
        ax.set_title(f'{name}\nR² = {r2:.3f}, MAE = {mae:.3f}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / 'prediction_scatter.png'
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPrediction scatter plot saved to: {save_path}")

    plt.show()


def plot_training_history(train_losses, val_losses, save_dir=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / 'training_history.png'
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN to predict organoid parameters from microscopy images'
    )
    parser.add_argument('--dataset_dir', type=str, default='organoid_dataset',
                        help='Directory containing organoid dataset')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                        help='Mode: train, test, or both')
    parser.add_argument('--model_path', type=str, default='organoid_predictor.pth',
                        help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data for training (rest for validation/test)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='training_results',
                        help='Directory to save results and plots')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_dir}...")
    full_dataset = OrganoidDataset(args.dataset_dir, transform=transform, normalize_params=True)

    # Split dataset
    train_size = int(args.train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Further split train into train/val (80/20 of train set)
    val_size = int(0.2 * len(train_dataset))
    train_size_final = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size_final, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = OrganoidPredictor(input_channels=3, output_dim=6).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Training
    if args.mode in ['train', 'both']:
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}")

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)

            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'param_stats': full_dataset.param_stats
                }, args.model_path)
                print(f"✓ Best model saved (val_loss: {val_loss:.6f})")

        # Plot training history
        plot_training_history(train_losses, val_losses, save_dir=args.output_dir)

        # Save training history
        history_path = Path(args.output_dir) / 'training_history.json'
        history_path.parent.mkdir(exist_ok=True, parents=True)
        with open(history_path, 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, f, indent=2)

    # Testing
    if args.mode in ['test', 'both']:
        print(f"\n{'='*60}")
        print("Testing model...")
        print(f"{'='*60}")

        # Load best model
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model_path} (epoch {checkpoint['epoch']})")

        # Test
        predictions, targets, metrics = test_model(model, test_loader, full_dataset, device,
                                                   save_dir=args.output_dir)

        # Save test results
        results_path = Path(args.output_dir) / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nTest results saved to: {results_path}")


if __name__ == "__main__":
    main()
