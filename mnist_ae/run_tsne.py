#!/usr/bin/env python3
"""
Run t-SNE visualization on trained autoencoder latent features
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from tqdm import tqdm
import argparse

from common.data import HF_MNIST, make_mnist_loader
import model as model_mlp
import model_conv


def load_autoencoder(checkpoint_path, device):
    """Load autoencoder from checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    # Handle both dict and object args
    if isinstance(args, dict):
        latent_dim = args["latent_dim"]
        model_type = args["model"]
    else:
        latent_dim = args.latent_dim
        model_type = args.model

    print(f"Model type: {model_type}")
    print(f"Latent dimension: {latent_dim}")

    # Create and load model based on type
    if model_type == "conv":
        autoencoder = model_conv.Autoencoder(latent_dim=latent_dim, use_sigmoid=True).to(device)
    elif model_type == "mlp":
        autoencoder = model_mlp.Autoencoder(latent_dim=latent_dim, use_sigmoid=True).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    autoencoder.load_state_dict(ckpt["model"])
    autoencoder.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    return autoencoder, latent_dim, model_type


def extract_features(autoencoder, test_loader, device):
    """Extract latent features from all test samples"""
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Extracting features"):
            images = images.to(device)
            latents = autoencoder.encoder(images)
            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.numpy())

    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Extracted latent features: {latents.shape}")
    print(f"Labels: {labels.shape}")

    return latents, labels


def run_tsne(latents, perplexity=30, n_iter=1000):
    """Apply t-SNE dimensionality reduction"""
    print("Running t-SNE (this may take a few minutes)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=n_iter, verbose=1)
    latents_2d = tsne.fit_transform(latents)
    print(f"t-SNE embedding: {latents_2d.shape}")
    return latents_2d


def plot_tsne_combined(latents_2d, labels, output_path):
    """Create combined scatter plot colored by digit class"""
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=10,
        edgecolors='none'
    )
    plt.colorbar(scatter, label='Digit', ticks=range(10))
    plt.title('t-SNE Visualization of Autoencoder Latent Space', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {output_path}")


def plot_tsne_by_digit(latents_2d, labels, output_path):
    """Create subplots for each digit"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for digit in range(10):
        ax = axes[digit]
        mask = labels == digit
        ax.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            c=f'C{digit}',
            alpha=0.6,
            s=10,
            edgecolors='none',
            label=f'Digit {digit}'
        )
        ax.set_title(f'Digit {digit}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.suptitle('t-SNE Visualization by Digit Class', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved by-digit plot to {output_path}")


def compute_clustering_metrics(latents, labels, latent_dim):
    """Compute clustering quality metrics"""
    silhouette = silhouette_score(latents, labels)
    calinski = calinski_harabasz_score(latents, labels)

    print(f"\nClustering Quality Metrics (on {latent_dim}D latent space):")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Calinski-Harabasz Score: {calinski:.2f}")
    print(f"\nNote: Higher values indicate better-separated clusters")


def plot_reconstructions(autoencoder, test_loader, device, output_path, n_samples=10):
    """Visualize sample reconstructions"""
    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images[:n_samples].to(device)
    sample_labels = sample_labels[:n_samples]

    with torch.no_grad():
        reconstructed, _ = autoencoder(sample_images)

    fig, axes = plt.subplots(2, n_samples, figsize=(20, 4))
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(sample_images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original\nDigit: {sample_labels[i]}')
        axes[0, i].axis('off')

        # Reconstructed
        axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')

    plt.suptitle('Original vs Reconstructed Images', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstructions to {output_path}")


def main():
    parser = argparse.ArgumentParser("t-SNE Visualization of Autoencoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("SLURM_TMPDIR", "./hf_cache"),
                        help="HuggingFace cache directory")
    parser.add_argument("--output-dir", type=str, default="./tsne_outputs", help="Output directory for plots")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--n-iter", type=int, default=1000, help="t-SNE iterations")
    parser.add_argument("--device", type=str, default=None, help="Device (mps/cuda/cpu)")
    args = parser.parse_args()

    # Setup device
    if args.device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load autoencoder
    autoencoder, latent_dim, model_type = load_autoencoder(args.checkpoint, device)

    # Load MNIST test set
    print(f"\nLoading MNIST test set from {args.cache_dir}...")
    mnist_test = HF_MNIST(split="test", only_digit=None, cache_dir=args.cache_dir)
    test_loader, n_test = make_mnist_loader(
        mnist_dataset=mnist_test,
        batch_size=256,
        num_workers=0,
        shuffle=False
    )
    print(f"Loaded {n_test} test samples")

    # Extract latent features
    latents, labels = extract_features(autoencoder, test_loader, device)

    # Run t-SNE
    latents_2d = run_tsne(latents, perplexity=args.perplexity, n_iter=args.n_iter)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_tsne_combined(latents_2d, labels, output_dir / "tsne_latent_space.png")
    plot_tsne_by_digit(latents_2d, labels, output_dir / "tsne_by_digit.png")
    plot_reconstructions(autoencoder, test_loader, device, output_dir / "reconstructions.png")

    # Compute clustering metrics
    compute_clustering_metrics(latents, labels, latent_dim)

    print(f"\n✅ Done! All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
