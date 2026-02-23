"""
Attempt 4 Training Script

Phase 1: Train only the MLP on histogram supervision (UNet disabled).
Phase 2: Fine-tune the full MLP + UNet model end-to-end on histograms.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import yaml

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pv_finder.models.autoencoder_models import trackstoHists_UNet_1000
from pv_finder.models.alt_loss_A import Loss
from pv_finder.data.collectdata_poca_KDE import collect_data_poca_ATLAS as collect_data_poca
from pv_finder.training.training import trainNet
from pv_finder.utils.utilities import (
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    save_to_mlflow,
    set_seed,
)

LOGGER = logging.getLogger("attempt4")


def setup_logging(log_folder: Path, runname: str) -> Path:
    """Configure logging to both console and file, returning the log file path."""
    log_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_folder / f"{runname}_{timestamp}.log"

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    return log_path


def get_parameter_groups(model: nn.Module):
    """Separate model parameters into MLP and UNet groups."""
    mlp_params = []
    unet_params = []

    for name, param in model.named_parameters():
        if "layer" in name:
            mlp_params.append(param)
        else:
            unet_params.append(param)

    return mlp_params, unet_params


def forward_mlp_hist(model: trackstoHists_UNet_1000, inputs: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the MLP stack only, returning aggregated histogram predictions.
    """
    leaky_relu = nn.LeakyReLU(model.LeakyReLU_param)

    n_events, _, n_tracks = inputs.shape

    mask = inputs[:, 1, :] > model.maskVal
    filt = mask.float()
    f1 = filt.unsqueeze(2)
    f2 = f1.expand(-1, -1, model.n_OutputFeatures)

    x = inputs.transpose(1, 2)
    x = leaky_relu(model.layer1(x))
    x = leaky_relu(model.layer2(x))
    x = leaky_relu(model.layer3(x))
    x = leaky_relu(model.layer4(x))
    x = leaky_relu(model.layer5(x))
    x = F.softplus(model.layer6A(x))

    x = x.view(n_events, n_tracks, model.n_LatentChannels, model.n_OutputFeatures)

    f2 = torch.unsqueeze(f2, 2)
    x = torch.mul(f2, x)
    output = torch.sum(x, dim=1)
    output = torch.mul(output, float(model.predScaleFactor))

    return output


def _squeeze_hist(predicted: torch.Tensor) -> torch.Tensor:
    """
    Reduce predicted histogram tensor to (batch, n_bins).
    Handles cases with >1 latent channels by summing over that dimension.
    """
    if predicted.dim() == 3:
        if predicted.size(1) == 1:
            return predicted.squeeze(1)
        return predicted.sum(dim=1)
    return predicted


def train_phase1_epoch(
    model: trackstoHists_UNet_1000,
    train_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch_idx: int,
) -> float:
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Phase 1 Epoch {epoch_idx}", leave=False, ncols=100)

    for batch_idx, (inputs, target_split) in enumerate(pbar):
        inputs = inputs.to(device).float()
        target_split = target_split.to(device).float()
        target_hist = target_split[:, 0, :]

        optimizer.zero_grad()

        predicted = forward_mlp_hist(model, inputs)
        predicted_hist = _squeeze_hist(predicted)

        loss = loss_fn(predicted_hist, target_hist)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

    return total_loss / len(train_loader)


@torch.no_grad()
def validate_phase1_epoch(
    model: trackstoHists_UNet_1000,
    val_loader: Iterable,
    loss_fn: nn.Module,
    device: torch.device,
    epoch_idx: int,
) -> float:
    model.eval()
    total_loss = 0.0

    pbar = tqdm(val_loader, desc=f"Phase 1 Epoch {epoch_idx} (val)", leave=False, ncols=100)

    for batch_idx, (inputs, target_split) in enumerate(pbar):
        inputs = inputs.to(device).float()
        target_split = target_split.to(device).float()
        target_hist = target_split[:, 0, :]

        predicted = forward_mlp_hist(model, inputs)
        predicted_hist = _squeeze_hist(predicted)

        loss = loss_fn(predicted_hist, target_hist)
        total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

    return total_loss / len(val_loader)


def train_phase1(
    model: trackstoHists_UNet_1000,
    train_loader: Iterable,
    val_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    n_epochs: int,
    save_folder: Path,
    runname: str,
    save_frequency: int,
) -> Tuple[float, int]:
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("PHASE 1: MLP-only training (histogram supervision)")
    LOGGER.info("=" * 80)

    save_folder.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(n_epochs):
        epoch_idx = epoch + 1
        train_loss = train_phase1_epoch(model, train_loader, optimizer, loss_fn, device, epoch_idx)
        val_loss = validate_phase1_epoch(model, val_loader, loss_fn, device, epoch_idx)

        LOGGER.info(
            "Phase 1 Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f",
            epoch_idx,
            n_epochs,
            train_loss,
            val_loss,
        )

        save_to_mlflow(
            {
                "Metric: Phase1 Training loss": train_loss,
                "Metric: Phase1 Validation loss": val_loss,
            },
            step=epoch_idx,
        )

        if (epoch_idx % save_frequency == 0) or (epoch_idx == n_epochs):
            checkpoint_path = save_folder / f"{runname}_phase1_epoch_{epoch_idx}_fullstate.pth"
            save_checkpoint(model, optimizer, epoch_idx, val_loss, path=str(checkpoint_path))
            mlflow.log_artifact(str(checkpoint_path))
            LOGGER.info("  ✓ Saved checkpoint: %s", checkpoint_path.name)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_idx

    save_to_mlflow(
        {
            "Metric: Phase1 Best Val Loss": best_val_loss,
            "Metric: Phase1 Best Epoch": best_epoch,
        },
        step=n_epochs,
    )

    LOGGER.info(
        "\n✓ Phase 1 complete! Best val loss: %.6f at epoch %d",
        best_val_loss,
        best_epoch,
    )

    return best_val_loss, best_epoch


def main(configs):
    log_path = setup_logging(Path(configs["log_folder"]), configs["runname"])

    set_seed()

    device_id = configs["device_id"]
    if torch.cuda.is_available() and torch.cuda.device_count() > device_id:
        torch.cuda.set_device(device_id)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            LOGGER.warning(
                "Requested CUDA device %d not available. Falling back to CPU.", device_id
            )
        else:
            LOGGER.warning("CUDA not available. Training will run on CPU.")

    LOGGER.info("Using device: %s", device)

    mlflow.set_tracking_uri("file:/data/home/matmauro/codice/PV-Finder/mlruns")
    mlflow.set_experiment(configs["experimentname"])

    save_folder = Path(configs["save_folder"])
    save_folder.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=configs["runname"]):
        # Log script and configuration
        mlflow.log_artifact(str(SCRIPT_PATH))
        if "config_file" in configs:
            mlflow.log_artifact(configs["config_file"])

        # Loggers also to MLflow at the end
        mlflow.log_param("phase1_epochs", configs["phase1_epochs"])
        mlflow.log_param("phase2_epochs", configs["phase2_epochs"])
        mlflow.log_param("batch_size", configs["batch_size"])
        mlflow.log_param("phase1_learning_rate", configs["phase1_learning_rate"])
        mlflow.log_param("phase2_learning_rate", configs["phase2_learning_rate"])

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("Creating model...")
        LOGGER.info("=" * 80)

        model = trackstoHists_UNet_1000(
            n_InputFeatures=configs["models_config"]["n_input_features"],
            n_LatentChannels=configs["models_config"]["n_latent_channels"],
            dropout=configs["models_config"]["dropout"],
        )
        model.to(device)

        mlp_params, unet_params = get_parameter_groups(model)
        mlp_param_count = sum(p.numel() for p in mlp_params)
        unet_param_count = sum(p.numel() for p in unet_params)

        LOGGER.info("MLP parameters: %d tensors (%s)", len(mlp_params), f"{mlp_param_count:,}")
        LOGGER.info("UNet parameters: %d tensors (%s)", len(unet_params), f"{unet_param_count:,}")
        LOGGER.info("Total trainable parameters: %s", f"{count_parameters(model):,}")

        save_to_mlflow({"Param: Total trainable parameters": count_parameters(model)}, step=0)

        # Phase 1 -----------------------------------------------------------------
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("PHASE 1: Loading data (tracks-to-hist)")
        LOGGER.info("=" * 80)

        train_loader_phase1, val_loader_phase1, _ = collect_data_poca(
            filepath=configs["data_file"],
            data_pipeline="tracks-to-hist",
            batch_size=configs["batch_size"],
            masking=False,
            num_workers=configs["num_workers"],
            prefetch_factor=2,
            train_split=configs["train_split"],
        )

        LOGGER.info("✓ Loaded %d training samples", len(train_loader_phase1.dataset))
        LOGGER.info("✓ Loaded %d validation samples", len(val_loader_phase1.dataset))

        for param in unet_params:
            param.requires_grad = False

        optimizer_phase1 = torch.optim.Adam(
            mlp_params,
            lr=configs["phase1_learning_rate"],
            betas=(0.9, 0.999),
        )
        loss_phase1 = nn.MSELoss()

        best_val_loss_phase1, best_epoch_phase1 = train_phase1(
            model=model,
            train_loader=train_loader_phase1,
            val_loader=val_loader_phase1,
            optimizer=optimizer_phase1,
            loss_fn=loss_phase1,
            device=device,
            n_epochs=configs["phase1_epochs"],
            save_folder=save_folder,
            runname=configs["runname"],
            save_frequency=configs["save_frequency"],
        )

        phase1_checkpoint_path = save_folder / f"{configs['runname']}_phase1_epoch_{configs['phase1_epochs']}_fullstate.pth"
        if not phase1_checkpoint_path.exists():
            raise FileNotFoundError(f"Expected Phase 1 checkpoint not found: {phase1_checkpoint_path}")

        model, loaded_epoch = load_checkpoint(model, optimizer_phase1, path=str(phase1_checkpoint_path))
        model.to(device)
        LOGGER.info("✓ Loaded Phase 1 checkpoint from epoch %d", loaded_epoch)

        # Phase 2 -----------------------------------------------------------------
        for param in model.parameters():
            param.requires_grad = True

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("PHASE 2: Loading data (tracks-to-hist)")
        LOGGER.info("=" * 80)

        train_loader_phase2, val_loader_phase2, _ = collect_data_poca(
            filepath=configs["data_file"],
            data_pipeline="tracks-to-hist",
            batch_size=configs["batch_size"],
            masking=False,
            num_workers=configs["num_workers"],
            prefetch_factor=2,
            train_split=configs["train_split"],
        )

        LOGGER.info("✓ Loaded %d training samples", len(train_loader_phase2.dataset))
        LOGGER.info("✓ Loaded %d validation samples", len(val_loader_phase2.dataset))

        if configs.get("use_mse_loss", False):
            loss_phase2 = nn.MSELoss()
            LOGGER.info("Using MSE loss for Phase 2")
        else:
            loss_phase2 = Loss(epsilon=1e-5, coefficient=configs["asymmetry_param"])
            LOGGER.info(
                "Using asymmetric loss for Phase 2 (coefficient=%s)",
                configs["asymmetry_param"],
            )

        optimizer_phase2 = torch.optim.Adam(
            model.parameters(),
            lr=configs["phase2_learning_rate"],
            betas=(0.9, 0.999),
        )

        out_idx = 0
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("PHASE 2: End-to-End Training")
        LOGGER.info("=" * 80)
        LOGGER.info("Epochs: %d", configs["phase2_epochs"])
        LOGGER.info("Batch size: %d", configs["batch_size"])
        LOGGER.info("Learning rate: %.6f", configs["phase2_learning_rate"])
        LOGGER.info("=" * 80 + "\n")

        train_iter = enumerate(
            trainNet(
                model,
                optimizer_phase2,
                loss_phase2,
                train_loader_phase2,
                val_loader_phase2,
                configs["phase2_epochs"],
                out_idx,
                notebook=None,
                epoch_start=0,
            )
        )

        best_val_loss_phase2 = float("inf")
        best_epoch_phase2 = 0

        for i, result in train_iter:
            epoch = result.epoch
            train_loss = result.cost
            val_loss = result.val
            efficiency_val = result.eff_val.eff_rate
            fpr_val = result.eff_val.fp_rate

            LOGGER.info(
                "Phase 2 Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f | Eff: %.4f | FPR: %.4f",
                epoch + 1,
                configs["phase2_epochs"],
                train_loss,
                val_loss,
                efficiency_val,
                fpr_val,
            )

            save_to_mlflow(
                {
                    "Metric: Phase2 Training loss": train_loss,
                    "Metric: Phase2 Validation loss": val_loss,
                    "Metric: Phase2 Efficiency": efficiency_val,
                    "Metric: Phase2 False positive rate": fpr_val,
                },
                step=configs["phase1_epochs"] + epoch + 1,
            )

            if ((epoch + 1) % configs["save_frequency"] == 0) or (epoch + 1 == configs["phase2_epochs"]):
                save_dir = save_folder
                save_dir.mkdir(parents=True, exist_ok=True)

                pyt_path = save_dir / f"{configs['runname']}_phase2_epoch_{epoch + 1}.pyt"
                torch.save(model, pyt_path)
                mlflow.log_artifact(str(pyt_path))

                full_ckpt_path = save_dir / f"{configs['runname']}_phase2_epoch_{epoch + 1}_fullstate.pth"
                save_checkpoint(model, optimizer_phase2, epoch + 1, val_loss, path=str(full_ckpt_path))
                mlflow.log_artifact(str(full_ckpt_path))

                LOGGER.info("  ✓ Saved checkpoints for epoch %d", epoch + 1)

            if val_loss < best_val_loss_phase2:
                best_val_loss_phase2 = val_loss
                best_epoch_phase2 = epoch + 1

        save_to_mlflow(
            {
                "Metric: Phase2 Best Val Loss": best_val_loss_phase2,
                "Metric: Phase2 Best Epoch": best_epoch_phase2,
            },
            step=configs["phase1_epochs"] + configs["phase2_epochs"],
        )

        LOGGER.info(
            "\n✓ Phase 2 complete! Best val loss: %.6f at epoch %d",
            best_val_loss_phase2,
            best_epoch_phase2,
        )

        mlflow.log_artifact(str(log_path))

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("✅ Training complete!")
        LOGGER.info("   Phase 1 best val loss: %.6f (epoch %d)", best_val_loss_phase1, best_epoch_phase1)
        LOGGER.info("   Phase 2 best val loss: %.6f (epoch %d)", best_val_loss_phase2, best_epoch_phase2)
        LOGGER.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train tracks-to-histograms with MLP-only warmup followed by end-to-end optimization."
    )
    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        type=str,
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    with open(args.config_file, "r") as file:
        configs = yaml.safe_load(file)
        configs["config_file"] = args.config_file

    print("\n" + "=" * 80)
    print("Configuration:")
    print("=" * 80)
    for key, value in configs.items():
        if key != "config_file":
            print(f"  {key}: {value}")
    print("=" * 80 + "\n")

    main(configs)

