"""
Training script for end-to-end Tracks-to-Histogram model.

This script trains a combined model that goes directly from track features
to histogram predictions, bypassing the intermediate KDE representation.

Based on Qi Bin's TrainATLAS_script_1000bins_tracksHists_UNet_8latent.py
"""

import argparse
import sys
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml

from pv_finder.data.collectdata_poca_KDE import (
    collect_data_poca_ATLAS as collect_data_poca,
)
from pv_finder.models.alt_loss_A import Loss
from pv_finder.models.autoencoder_models import trackstoHists_UNet_1000
from pv_finder.training.initialize_combined_weights import initialize_combined_model
from pv_finder.training.training import trainNet
from pv_finder.utils.utilities import (
    count_parameters,
    load_checkpoint,
    save_checkpoint,
    save_to_mlflow,
    set_seed,
)


def main(configs, resume_checkpoint=None, resume_epoch=None):
    # Set seed for reproducibility
    set_seed()

    # Set GPU
    device_id = configs["device_id"]
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\n" + "=" * 80)
    print("Loading training data...")
    print("=" * 80)

    train_loader, val_loader, _ = collect_data_poca(
        filepath=configs["data_file"],
        data_pipeline="tracks-to-hist",
        batch_size=configs["batch_size"],
        masking=False,
        num_workers=configs["num_workers"],
        prefetch_factor=2,
        train_split=configs["train_split"],
    )

    print(f"✓ Loaded {len(train_loader.dataset)} training samples")
    print(f"✓ Loaded {len(val_loader.dataset)} validation samples")

    # Set up MLflow tracking
    # Use the same mlruns directory as other training scripts for consistency
    mlflow.set_tracking_uri("file:/data/home/matmauro/codice/PV-Finder/mlruns")
    mlflow.set_experiment(configs["experimentname"])

    # Create model
    print("\n" + "=" * 80)
    print("Creating model...")
    print("=" * 80)

    # Create new model (checkpoint loading happens after optimizer creation)
    model = trackstoHists_UNet_1000(
        n_InputFeatures=configs["models_config"]["n_input_features"],
        n_LatentChannels=configs["models_config"]["n_latent_channels"],
        dropout=configs["models_config"]["dropout"],
    )

    # Initialize with pretrained weights if provided
    if "pretrained_weights" in configs and configs["pretrained_weights"] is not None:
        print("\n" + "=" * 80)
        print("Loading pretrained weights...")
        print("=" * 80)

        pretrained_path = configs["pretrained_weights"]

        if pretrained_path.endswith(".pth"):
            # Load combined model weights
            print(f"Loading combined model weights from: {pretrained_path}")
            pretrained_weights = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(pretrained_weights)
            print("✓ Loaded pretrained combined model weights")
        else:
            # Initialize from separate Tracks-to-KDE and KDE-to-Histogram models
            tracks2kde_path = configs["pretrained_tracks2kde"]
            kde2hist_path = configs["pretrained_kde2hist"]

            model = initialize_combined_model(
                model=model,
                tracks2kde_weights_path=tracks2kde_path,
                kde2hist_weights_path=kde2hist_path,
                n_latent_channels=configs["models_config"]["n_latent_channels"],
                n_output_bins=1000,
                verbose=True,
            )

    model.to(device)

    # Output index for Target_Y
    out_idx = 0

    # Set up optimizer (must be created before resume logic for full checkpoint loading)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs["learning_rate"], betas=(0.9, 0.999)
    )

    # Optional resume from checkpoint (after optimizer is created)
    # - If resume_checkpoint is a .pth full checkpoint, restore model + optimizer and start from loaded_epoch + 1
    # - If it's a legacy .pyt model object, restore weights only and use provided resume_epoch
    epoch_start = 0
    if resume_checkpoint is not None:
        print(f"\n{'=' * 80}")
        print(f"RESUMING FROM CHECKPOINT: {resume_checkpoint}")
        print(f"{'=' * 80}\n")

        if str(resume_checkpoint).endswith(".pth"):
            model, loaded_epoch = load_checkpoint(
                model, optimizer, path=resume_checkpoint
            )
            model.to(device)
            epoch_start = int(loaded_epoch) + 1
            print(
                f"✓ Loaded full checkpoint with optimizer state from epoch {loaded_epoch}"
            )
            print(f"✓ Training will continue from epoch {epoch_start}\n")
        else:
            # Legacy .pyt format - load model weights only
            loaded_model = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(loaded_model.state_dict(), strict=False)
            model.to(device)
            epoch_start = int(resume_epoch) + 1 if resume_epoch is not None else 0
            print("✓ Loaded model weights (no optimizer state)")
            print(f"✓ Training will continue from epoch {epoch_start}\n")

    # Set up loss function
    if configs.get("use_mse_loss", False):
        loss = nn.MSELoss()
        print("Using MSE loss")
    else:
        loss = Loss(epsilon=1e-5, coefficient=configs["asymmetry_param"])
        print(f"Using asymmetric loss (coefficient={configs['asymmetry_param']})")

    # Count parameters
    parameters = count_parameters(model)
    print(f"\nTotal trainable parameters: {parameters:,}")

    # Get run name
    runname = configs["runname"]

    # Start training
    print("\n" + "=" * 80)
    print(f"Starting training: {runname}")
    print("=" * 80)
    print(f"Epochs: {configs['n_epochs']}")
    print(f"Batch size: {configs['batch_size']}")
    print(f"Learning rate: {configs['learning_rate']}")
    print("=" * 80 + "\n")

    train_iter = enumerate(
        trainNet(
            model,
            optimizer,
            loss,
            train_loader,
            val_loader,
            configs["n_epochs"],
            out_idx,
            notebook=False,
            epoch_start=epoch_start,
        )
    )

    with mlflow.start_run(run_name=runname) as run:  # noqa: F841
        # Log training script and config
        mlflow.log_artifact(str(Path(__file__)))
        if "config_file" in configs:
            mlflow.log_artifact(configs["config_file"])

        for i, result in train_iter:
            epoch = result.epoch
            train_loss = result.cost
            val_loss = result.val
            efficiency = result.eff_val.eff_rate
            fpr = result.eff_val.fp_rate

            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | Eff: {efficiency:.4f} | FPR: {fpr:.4f}"
            )

            # Save current model
            torch.save(model, "run_stats.pyt")
            mlflow.log_artifact("run_stats.pyt")

            # Save model weights at specified frequency
            if i % configs["save_frequency"] == 0 and configs["save_weights"]:
                save_dir = Path(configs["save_folder"])
                save_dir.mkdir(parents=True, exist_ok=True)

                # Save legacy .pyt format (model only)
                output_path = save_dir / f"{runname}_epoch_{epoch}.pyt"
                torch.save(model, output_path)
                mlflow.log_artifact(str(output_path))
                print(f"  ✓ Saved model: {output_path.name}")

                # Save full checkpoint (model + optimizer state)
                full_ckpt_path = save_dir / f"{runname}_epoch_{epoch}_fullstate.pth"
                save_checkpoint(
                    model, optimizer, epoch, val_loss, path=str(full_ckpt_path)
                )
                mlflow.log_artifact(str(full_ckpt_path))
                print(f"  ✓ Saved full checkpoint: {full_ckpt_path.name}")

            # Log metrics to MLflow
            save_to_mlflow(
                {
                    "Metric: Training loss": train_loss,
                    "Metric: Validation loss": val_loss,
                    "Metric: Efficiency": efficiency,
                    "Metric: False positive rate": fpr,
                    "Param: Parameters": parameters,
                    "Param: Asymmetry": configs.get("asymmetry_param", 0),
                    "Param: Epochs": configs["n_epochs"],
                    "Param: Learning rate": configs["learning_rate"],
                    "Param: Batch size": configs["batch_size"],
                },
                step=i,
            )

    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train end-to-end Tracks-to-Histogram PV-Finder model for ATLAS data"
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to configuration YAML file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        help="Path to checkpoint to resume training from (e.g., model_epoch_90.pyt)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resume_epoch",
        help="Epoch number to resume from (used with --resume_checkpoint)",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    # Load config
    with open(args.config_file) as file:
        try:
            configs = yaml.safe_load(file)
            configs["config_file"] = args.config_file
        except yaml.YAMLError as exc:
            print(f"Error loading config file: {exc}")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("Configuration:")
    print("=" * 80)
    for key, value in configs.items():
        if key != "config_file":
            print(f"  {key}: {value}")
    print("=" * 80 + "\n")

    main(
        configs,
        resume_checkpoint=args.resume_checkpoint,
        resume_epoch=args.resume_epoch,
    )
