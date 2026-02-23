"""
Initialize combined Tracks-to-Histogram model from pretrained weights.

This script combines pretrained Tracks-to-KDE and KDE-to-Histogram models
into a single end-to-end Tracks-to-Histogram model.
"""

import argparse
from pathlib import Path

import torch


def initialize_combined_model(
    model,
    tracks2kde_weights_path,
    kde2hist_weights_path,
    n_latent_channels=1,
    n_output_bins=1000,
    verbose=True,
):
    """
    Initialize a combined Tracks-to-Histogram model from pretrained weights.

    Args:
        model: trackstoHists_UNet_1000 model instance
        tracks2kde_weights_path: Path to Tracks-to-KDE model weights (.pyt or .pth)
        kde2hist_weights_path: Path to KDE-to-Histogram model weights (.pyt)
        n_latent_channels: Number of latent KDE channels (1 for KDE-A-z only)
        n_output_bins: Number of output bins (1000 for sub-events)
        verbose: Print progress messages

    Returns:
        model: Initialized model with combined weights
    """

    if verbose:
        print("=" * 80)
        print("Initializing Combined Tracks-to-Histogram Model")
        print("=" * 80)

    # Load pretrained weights
    if verbose:
        print(f"\nLoading Tracks-to-KDE weights from: {tracks2kde_weights_path}")

    if tracks2kde_weights_path.endswith(".pth"):
        # Checkpoint format with 'model_state'
        t2kde_checkpoint = torch.load(tracks2kde_weights_path, map_location="cpu")
        t2kde_weights = (
            t2kde_checkpoint["model_state"]
            if "model_state" in t2kde_checkpoint
            else t2kde_checkpoint
        )
    else:
        # Direct model save
        t2kde_model = torch.load(tracks2kde_weights_path, map_location="cpu")
        t2kde_weights = t2kde_model.state_dict()

    if verbose:
        print(f"Loading KDE-to-Histogram weights from: {kde2hist_weights_path}")

    kde2hist_model = torch.load(kde2hist_weights_path, map_location="cpu")
    kde2hist_weights = kde2hist_model.state_dict()

    # Get model state dict
    model_dict = model.state_dict()

    if verbose:
        print("\n" + "=" * 80)
        print("Step 1: Loading Fully Connected Network (Tracks-to-KDE) weights")
        print("=" * 80)

    # Copy FCN weights (layers 1-5), expanding layer1 input if needed
    old_linear1_weight = t2kde_weights["linear1.weight"].cpu()
    old_linear1_bias = t2kde_weights["linear1.bias"].cpu()
    old_n_features = old_linear1_weight.shape[1]
    new_n_features = model_dict["layer1.weight"].shape[1]

    if old_n_features != new_n_features:
        if verbose:
            print(
                f"  ⚠️  Input feature mismatch: old={old_n_features}, new={new_n_features}"
            )
            print(
                f"  → Zero-padding extra {new_n_features - old_n_features} input features"
            )

        # Expand layer1 input by zero-padding (following Qi Bin's approach in Cell 51-54)
        new_linear1_weight = torch.zeros(100, new_n_features - old_n_features)
        updated_linear1_weight = torch.cat(
            (old_linear1_weight, new_linear1_weight), dim=1
        )
        model_dict["layer1.weight"] = updated_linear1_weight
        model_dict["layer1.bias"] = old_linear1_bias
        if verbose:
            print(
                f"  ✓ Expanded linear1 -> layer1 ({old_n_features} → {new_n_features} features)"
            )
    else:
        model_dict["layer1.weight"] = old_linear1_weight
        model_dict["layer1.bias"] = old_linear1_bias
        if verbose:
            print("  ✓ Copied linear1 -> layer1")

    # Copy layers 2-5 directly
    for old_key, new_key in [
        ("linear2", "layer2"),
        ("linear3", "layer3"),
        ("linear4", "layer4"),
        ("linear5", "layer5"),
    ]:
        model_dict[f"{new_key}.weight"] = t2kde_weights[f"{old_key}.weight"]
        model_dict[f"{new_key}.bias"] = t2kde_weights[f"{old_key}.bias"]
        if verbose:
            print(f"  ✓ Copied {old_key} -> {new_key}")

    if verbose:
        print("\n" + "=" * 80)
        print("Step 2: Expanding output layer (linear6 -> layer6A)")
        print("=" * 80)

    # Handle layer6 expansion if needed
    old_linear6_weight = t2kde_weights["linear6.weight"].cpu()
    old_linear6_bias = t2kde_weights["linear6.bias"].cpu()

    old_output_size = old_linear6_weight.shape[0]
    new_output_size = n_latent_channels * n_output_bins

    if verbose:
        print(f"  Old output size: {old_output_size}")
        print(
            f"  New output size: {new_output_size} ({n_latent_channels} channels × {n_output_bins} bins)"
        )

    if old_output_size != new_output_size:
        if verbose:
            print("  → Expanding output layer...")

        # Create expanded weights
        # Keep old weights, initialize rest randomly
        new_linear6_weight = torch.randn(new_output_size - old_output_size, 100) * 0.02
        new_linear6_bias = torch.zeros(new_output_size - old_output_size)

        updated_linear6_weight = torch.cat(
            (old_linear6_weight, new_linear6_weight), dim=0
        )
        updated_linear6_bias = torch.cat((old_linear6_bias, new_linear6_bias), dim=0)

        model_dict["layer6A.weight"] = updated_linear6_weight
        model_dict["layer6A.bias"] = updated_linear6_bias

        if verbose:
            print(f"  ✓ Expanded layer6A: {updated_linear6_weight.shape}")
    else:
        # Output size already matches, just copy directly
        model_dict["layer6A.weight"] = old_linear6_weight
        model_dict["layer6A.bias"] = old_linear6_bias
        if verbose:
            print("  ✓ Copied linear6 -> layer6A (no expansion needed)")

    if verbose:
        print("\n" + "=" * 80)
        print("Step 3: Loading UNet (KDE-to-Histogram) weights")
        print("=" * 80)

    # Copy UNet weights, handling rcbn1 channel expansion
    old_rcbn1_weight = kde2hist_weights["rcbn1.0.weight"]  # Shape: (64, 1, 25)
    old_rcbn1_bias = kde2hist_weights["rcbn1.0.bias"]

    if verbose:
        print(f"  Old rcbn1 input channels: {old_rcbn1_weight.shape[1]}")
        print(f"  New rcbn1 input channels: {n_latent_channels}")

    with torch.no_grad():
        # Initialize rcbn1 with pretrained weights for first channel
        model_dict["rcbn1.0.weight"][:, :1, :] = (
            old_rcbn1_weight  # First channel from pretrained
        )
        model_dict["rcbn1.0.bias"] = old_rcbn1_bias

        # Initialize remaining channels randomly if n_latent_channels > 1
        if n_latent_channels > 1:
            model_dict["rcbn1.0.weight"][:, 1:, :] = torch.randn_like(
                model_dict["rcbn1.0.weight"][:, 1:, :]
            )
            model_dict["rcbn1.0.bias"] = torch.zeros_like(model_dict["rcbn1.0.bias"])
            if verbose:
                print(
                    f"  ✓ Expanded rcbn1 input channels (channels 2-{n_latent_channels} randomly initialized)"
                )

    # Copy rest of UNet weights
    unet_layers_copied = 0
    for key in kde2hist_weights.keys():
        if key in model_dict:
            # Skip rcbn1 weights as we handled them manually
            if "rcbn1.0.weight" in key or "rcbn1.0.bias" in key:
                continue
            model_dict[key] = kde2hist_weights[key]
            unet_layers_copied += 1

    if verbose:
        print(f"  ✓ Copied {unet_layers_copied} UNet layer weights")

    # Load updated state dict into model
    model.load_state_dict(model_dict)

    if verbose:
        print("\n" + "=" * 80)
        print("✅ Successfully initialized combined model!")
        print("=" * 80)
        print("\nModel ready for training with:")
        print("  - Pretrained Tracks-to-KDE (FCN layers)")
        print("  - Pretrained KDE-to-Histogram (UNet layers)")
        print(f"  - {n_latent_channels} latent channel(s)")
        print(f"  - {n_output_bins} output bins per channel")
        print("=" * 80)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Initialize combined Tracks-to-Histogram model from pretrained weights"
    )
    parser.add_argument(
        "--t2kde",
        type=str,
        required=True,
        help="Path to Tracks-to-KDE model weights (.pyt or .pth)",
    )
    parser.add_argument(
        "--kde2hist",
        type=str,
        required=True,
        help="Path to KDE-to-Histogram model weights (.pyt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save initialized combined model weights (.pth)",
    )
    parser.add_argument(
        "--n-latent",
        type=int,
        default=1,
        help="Number of latent KDE channels (default: 1 for KDE-A-z only)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=9,
        help="Number of input track features (default: 9)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Dropout probability (default: 0.25)",
    )

    args = parser.parse_args()

    # Import model
    from pv_finder.models.autoencoder_models import trackstoHists_UNet_1000

    # Create model
    print("\nCreating trackstoHists_UNet_1000 model...")
    model = trackstoHists_UNet_1000(
        n_InputFeatures=args.n_features,
        n_LatentChannels=args.n_latent,
        dropout=args.dropout,
    )

    # Initialize weights
    model = initialize_combined_model(
        model=model,
        tracks2kde_weights_path=args.t2kde,
        kde2hist_weights_path=args.kde2hist,
        n_latent_channels=args.n_latent,
        n_output_bins=1000,
        verbose=True,
    )

    # Save initialized model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving initialized model to: {output_path}")
    torch.save(model.state_dict(), output_path)
    print("✅ Done!")


if __name__ == "__main__":
    main()
