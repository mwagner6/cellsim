"""
Usage:
    python run_dinov3.py --image_path path/to/image.jpg
"""

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from PIL import Image
import argparse
import numpy as np


def load_model(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"):
    print(f"Loading model: {model_name}")

    # Load image processor and model
    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
    model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", device_map="auto")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on: {device}")
    return processor, model, device


def process_image(image_path, processor, model, device):
    """
    Args:
        image_path: Path to input image
        processor: Image processor
        model: DINOv3 model
        device: torch device

    Returns:
        features: Image features (embeddings)
        cls_token: CLS token embedding
    """
    # Load and preprocess image
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model outputs
    with torch.inference_mode():
        outputs = model(**inputs)

    # Extract features
    # Last hidden state has shape [batch_size, num_patches + 1, hidden_size]
    # First token is CLS token, rest are patch tokens
    last_hidden_state = outputs.last_hidden_state

    cls_token = last_hidden_state[:, 0, :]  # CLS token
    patch_tokens = last_hidden_state[:, 1:, :]  # Patch tokens

    print(f"CLS token shape: {cls_token.shape}")
    print(f"Patch tokens shape: {patch_tokens.shape}")

    return {
        'cls_token': cls_token.cpu().numpy(),
        'patch_tokens': patch_tokens.cpu().numpy(),
        'full_embedding': last_hidden_state.cpu().numpy()
    }


def main():
    parser = argparse.ArgumentParser(description='Run DINOv3 or DINOv2 on an image')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='facebook/dinov3-vitl16-pretrain-lvd1689m',
                        help='DINOv3/v2 model to use (default: facebook/dinov3-vitl16-pretrain-lvd1689m)')
    parser.add_argument('--save_embeddings', type=str, default=None,
                        help='Path to save embeddings (optional)')

    args = parser.parse_args()

    # Load model
    processor, model, device = load_model(args.model)

    # Process image
    print(f"\nProcessing image: {args.image_path}")
    features = process_image(args.image_path, processor, model, device)

    # Print feature statistics
    print("\n=== Feature Statistics ===")
    print(f"CLS token mean: {features['cls_token'].mean():.4f}")
    print(f"CLS token std: {features['cls_token'].std():.4f}")
    print(f"CLS token norm: {np.linalg.norm(features['cls_token']):.4f}")

    # Save embeddings if requested
    if args.save_embeddings:
        np.savez(args.save_embeddings,
                 cls_token=features['cls_token'],
                 patch_tokens=features['patch_tokens'])
        print(f"\nEmbeddings saved to: {args.save_embeddings}")

    return features


if __name__ == "__main__":
    main()
