"""HLLHC PU200-tuned E2E training.

MLP warmup (Phase 1) + full E2E (Phase 2) with LR warmup, cosine decay, and
gradient clipping. Designed to avoid the mid-training divergence observed in
the plain `lr=1e-3` recipe at μ≈200 (the Phase 2 LR was effectively too hot
once the per-event loss scale scales up with pileup).

Shares the Phase 1 MLP-only forward pass with `train_mlp_hist_then_e2e.py`,
but Phase 2 is a custom loop so we can inject gradient clipping and an LR
schedule without modifying the shared `trainNet` helper.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from pv_finder.data.collectdata_poca_KDE import (
    collect_data_poca_ATLAS as collect_data_poca,
)
from pv_finder.models.autoencoder_models import MaskedDNN, trackstoHists_UNet_1000
from pv_finder.models.unet_v2 import TracksToHist_v2, UNet_1000_v2
from pv_finder.training.train_mlp_hist_then_e2e import (
    _squeeze_hist,
    forward_mlp_hist,
    get_parameter_groups,
)
from pv_finder.training.training import validate as _validate_full
from pv_finder.utils.utilities import (
    count_parameters,
    save_checkpoint,
    save_to_mlflow,
    set_seed,
)

LOGGER = logging.getLogger("hllhc_e2e")
SCRIPT_PATH = Path(__file__)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_logging(log_folder: Path, runname: str) -> Path:
    log_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_folder / f"{runname}_{timestamp}.log"

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)
    return log_path


def build_model(configs: dict, device: torch.device) -> nn.Module:
    """Instantiate model from config. Supports v1 (trackstoHists_UNet_1000) and v2."""
    mc = configs["models_config"]
    model_type = mc.get("model_type", "v1")
    if model_type == "v2":
        n_latent = mc.get("n_latent_channels", 1)
        t2kde = MaskedDNN(
            input_size=mc["n_input_features"],
            hidden_nodes=list(mc.get("l_hidden_nodes", [100] * 5)),
            output_size=1000 * n_latent,
            leaky_param=0.01,
            maskVal=-240.0,
            predScaleFactor=0.001,
        )
        k2h = UNet_1000_v2(
            n=mc.get("n_unet_channels", 64),
            n_features=n_latent,
            dropout_p=mc["dropout"],
        )
        model = TracksToHist_v2(t2kde, k2h)
    else:
        model = trackstoHists_UNet_1000(
            n_InputFeatures=mc["n_input_features"],
            n_LatentChannels=mc["n_latent_channels"],
            dropout=mc["dropout"],
            n_UNetChannels=mc.get("n_unet_channels", 64),
            l_HiddenNodes=list(mc.get("l_hidden_nodes", [100] * 5)),
        )
    return model.to(device)


def build_phase2_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    eta_min_frac: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup 0.01·lr → lr over `warmup_epochs`, then cosine to `eta_min`."""
    eta_min = base_lr * eta_min_frac
    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)


# ---------------------------------------------------------------------------
# Shared step helpers
# ---------------------------------------------------------------------------
def _clip_and_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, max_norm: float | None
) -> None:
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()


def _noop_progress(iterable, **_kwargs):
    return iterable


def _is_v2(model: nn.Module) -> bool:
    return isinstance(model, TracksToHist_v2)


def _get_param_groups_v2(model: TracksToHist_v2):
    """Separate MLP (t2kde) and UNet (k2h) parameters for v2 models."""
    return list(model.t2kde.parameters()), list(model.k2h.parameters())


def _forward_mlp_v2(model: TracksToHist_v2, inputs: torch.Tensor) -> torch.Tensor:
    """Phase 1 forward: run only the MLP (t2kde) part of a v2 model."""
    kde = model.t2kde(inputs)  # (B, n_latent * 1000)
    return kde.view(kde.shape[0], model.n_latent, 1000)  # (B, n_latent, 1000)


def _load_model_state(path: str, model: nn.Module) -> int:
    """Load only the model_state from a fullstate checkpoint.

    Skips optimizer_state. Used to resume Phase 2 from a Phase 1 checkpoint
    whose optimizer covers a different set of parameters (MLP-only) than the
    Phase 2 optimizer (full model).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return int(ckpt.get("epoch", -1))


# ---------------------------------------------------------------------------
# Phase 1: MLP warmup (UNet frozen)
# ---------------------------------------------------------------------------
def _phase1_forward(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Phase 1 MLP-only forward, dispatching by model type."""
    if _is_v2(model):
        return _squeeze_hist(_forward_mlp_v2(model, inputs))
    return _squeeze_hist(forward_mlp_hist(model, inputs))


def train_phase1_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch_idx: int,
    max_norm: float | None,
) -> float:
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc=f"P1 E{epoch_idx}", leave=False, ncols=100)
    for step, (inputs, target_split) in enumerate(pbar):
        inputs = inputs.to(device).float()
        target_hist = target_split.to(device).float()[:, 0, :]
        optimizer.zero_grad()
        pred = _phase1_forward(model, inputs)
        loss = loss_fn(pred, target_hist)
        loss.backward()
        _clip_and_step(model, optimizer, max_norm)
        total += loss.item()
        pbar.set_postfix({"loss": f"{total / (step + 1):.6f}"})
    return total / len(loader)


@torch.no_grad()
def validate_phase1_epoch(
    model: nn.Module, loader, loss_fn: nn.Module, device: torch.device
) -> float:
    model.eval()
    total = 0.0
    for inputs, target_split in loader:
        inputs = inputs.to(device).float()
        target_hist = target_split.to(device).float()[:, 0, :]
        pred = _phase1_forward(model, inputs)
        total += loss_fn(pred, target_hist).item()
    return total / len(loader)


def run_phase1(
    model: nn.Module, data, configs: dict, device: torch.device, save_folder: Path
) -> None:
    train_loader, val_loader = data
    if _is_v2(model):
        mlp_params, unet_params = _get_param_groups_v2(model)
    else:
        mlp_params, unet_params = get_parameter_groups(model)
    for p in unet_params:
        p.requires_grad = False
    optimizer = torch.optim.Adam(
        mlp_params, lr=configs["phase1_learning_rate"], betas=(0.9, 0.999)
    )
    loss_fn = nn.MSELoss()
    max_norm = configs.get("max_grad_norm")
    n_epochs = configs["phase1_epochs"]
    save_freq = configs["save_frequency"]
    runname = configs["runname"]

    LOGGER.info("=" * 80)
    LOGGER.info(
        "PHASE 1 — MLP warmup | %d epochs | lr=%g | grad_clip=%s",
        n_epochs,
        configs["phase1_learning_rate"],
        max_norm,
    )
    LOGGER.info("=" * 80)

    best_val, best_ep = float("inf"), 0
    for ep in range(1, n_epochs + 1):
        tr = train_phase1_epoch(
            model, train_loader, optimizer, loss_fn, device, ep, max_norm
        )
        va = validate_phase1_epoch(model, val_loader, loss_fn, device)
        LOGGER.info("P1 Ep %d/%d | train=%.6f | val=%.6f", ep, n_epochs, tr, va)
        save_to_mlflow(
            {
                "Metric: Phase1 Training loss": tr,
                "Metric: Phase1 Validation loss": va,
            },
            step=ep,
        )
        if ep % save_freq == 0 or ep == n_epochs:
            path = save_folder / f"{runname}_phase1_epoch_{ep}_fullstate.pth"
            save_checkpoint(model, optimizer, ep, va, path=str(path))
            mlflow.log_artifact(str(path))
            LOGGER.info("  ✓ Saved %s", path.name)
        if va < best_val:
            best_val, best_ep = va, ep

    LOGGER.info("✓ Phase 1 complete. best val=%.6f at ep %d", best_val, best_ep)
    for p in model.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------------------
# Phase 2: full E2E with LR schedule + grad clipping
# ---------------------------------------------------------------------------
def train_phase2_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch_idx: int,
    max_norm: float | None,
) -> float:
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc=f"P2 E{epoch_idx}", leave=False, ncols=100)
    for step, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out, labels[:, 0, :])
        loss.backward()
        _clip_and_step(model, optimizer, max_norm)
        total += loss.item()
        pbar.set_postfix({"loss": f"{total / (step + 1):.6f}"})
    return total / len(loader)


@torch.no_grad()
def validate_phase2_epoch(
    model: nn.Module, loader, loss_fn: nn.Module, device: torch.device
):
    total, eff = _validate_full(
        model, loss_fn, loader, device, out_idx=0, progress=_noop_progress
    )
    return total / len(loader), eff


def run_phase2(
    model: nn.Module, data, configs: dict, device: torch.device, save_folder: Path
) -> None:
    train_loader, val_loader = data
    base_lr = configs["phase2_learning_rate"]
    n_epochs = configs["phase2_epochs"]
    warmup = configs.get("phase2_warmup_epochs", 0)
    eta_min_frac = configs.get("phase2_cosine_eta_min_frac", 0.01)
    max_norm = configs.get("max_grad_norm")
    save_freq = configs["save_frequency"]
    runname = configs["runname"]

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()
    scheduler = build_phase2_scheduler(
        optimizer, warmup, n_epochs, base_lr, eta_min_frac
    )

    LOGGER.info("=" * 80)
    LOGGER.info(
        "PHASE 2 — E2E | %d epochs | lr=%g → %g | warmup=%d | cosine | grad_clip=%s",
        n_epochs,
        base_lr,
        base_lr * eta_min_frac,
        warmup,
        max_norm,
    )
    LOGGER.info("=" * 80)

    best_val, best_ep = float("inf"), 0
    for ep in range(1, n_epochs + 1):
        tr = train_phase2_epoch(
            model, train_loader, optimizer, loss_fn, device, ep, max_norm
        )
        va, eff = validate_phase2_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        LOGGER.info(
            "P2 Ep %d/%d | train=%.6f | val=%.6f | eff=%.4f | fpr=%.4f | lr=%.2e",
            ep,
            n_epochs,
            tr,
            va,
            eff.eff_rate,
            eff.fp_rate,
            cur_lr,
        )
        save_to_mlflow(
            {
                "Metric: Phase2 Training loss": tr,
                "Metric: Phase2 Validation loss": va,
                "Metric: Phase2 Efficiency": eff.eff_rate,
                "Metric: Phase2 FPR": eff.fp_rate,
                "Metric: Phase2 LR": cur_lr,
            },
            step=configs["phase1_epochs"] + ep,
        )
        if ep % save_freq == 0 or ep == n_epochs:
            ckpt = save_folder / f"{runname}_phase2_epoch_{ep}_fullstate.pth"
            save_checkpoint(model, optimizer, ep, va, path=str(ckpt))
            mlflow.log_artifact(str(ckpt))
            pyt_path = save_folder / f"{runname}_phase2_epoch_{ep}.pyt"
            torch.save(model, pyt_path)
            mlflow.log_artifact(str(pyt_path))
            LOGGER.info("  ✓ Saved %s", ckpt.name)
        if va < best_val:
            best_val, best_ep = va, ep

    LOGGER.info("✓ Phase 2 complete. best val=%.6f at ep %d", best_val, best_ep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(configs: dict, phase1_checkpoint: str | None = None) -> None:
    log_path = setup_logging(Path(configs["log_folder"]), configs["runname"])
    set_seed()
    if torch.cuda.is_available():
        torch.cuda.set_device(configs["device_id"])
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        LOGGER.warning("CUDA not available. Training will run on CPU.")
    LOGGER.info("Device: %s", device)

    mlflow.set_tracking_uri("file:/data/home/matmauro/codice/PV-Finder/mlruns")
    mlflow.set_experiment(configs["experimentname"])

    save_folder = Path(configs["save_folder"])
    save_folder.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=configs["runname"]):
        mlflow.log_artifact(str(SCRIPT_PATH))
        if "config_file" in configs:
            mlflow.log_artifact(configs["config_file"])
        mlflow.log_param("phase1_epochs", configs["phase1_epochs"])
        mlflow.log_param("phase2_epochs", configs["phase2_epochs"])
        mlflow.log_param("phase1_lr", configs["phase1_learning_rate"])
        mlflow.log_param("phase2_lr", configs["phase2_learning_rate"])
        mlflow.log_param("phase2_warmup_epochs", configs.get("phase2_warmup_epochs", 0))
        mlflow.log_param("max_grad_norm", configs.get("max_grad_norm"))
        mlflow.log_param("batch_size", configs["batch_size"])

        model = build_model(configs, device)
        n_params = count_parameters(model)
        LOGGER.info("Total trainable parameters: %s", f"{n_params:,}")
        save_to_mlflow({"Param: Total trainable parameters": n_params}, step=0)

        train_loader, val_loader, _ = collect_data_poca(
            filepath=configs["data_file"],
            data_pipeline="tracks-to-hist",
            batch_size=configs["batch_size"],
            masking=False,
            num_workers=configs["num_workers"],
            prefetch_factor=2,
            train_split=configs["train_split"],
        )
        LOGGER.info("✓ %d train samples, %d val samples",
                    len(train_loader.dataset), len(val_loader.dataset))  # fmt: skip
        data = (train_loader, val_loader)

        if phase1_checkpoint is not None:
            ep = _load_model_state(phase1_checkpoint, model)
            model.to(device)
            for p in model.parameters():
                p.requires_grad = True
            LOGGER.info(
                "✓ Skipping Phase 1 — loaded weights from %s (Phase 1 epoch %d)",
                Path(phase1_checkpoint).name,
                ep,
            )
        else:
            run_phase1(model, data, configs, device, save_folder)
            # run_phase1 leaves the trained model in memory and unfreezes all
            # params; no reload needed.

        run_phase2(model, data, configs, device, save_folder)
        mlflow.log_artifact(str(log_path))

    LOGGER.info("=" * 80)
    LOGGER.info("✅ Training complete")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HLLHC PU200 E2E training (LR warmup + cosine + grad clip)"
    )
    parser.add_argument("-c", "--config_file", required=True, type=str)
    parser.add_argument(
        "--phase1-checkpoint",
        default=None,
        dest="phase1_checkpoint",
        help="Path to a Phase-1 fullstate.pth. Skips Phase 1 entirely and "
        "runs Phase 2 from the loaded weights (model_state only).",
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        configs = yaml.safe_load(f)
        configs["config_file"] = args.config_file

    print("=" * 80)
    print("Configuration:")
    for key, value in configs.items():
        if key != "config_file":
            print(f"  {key}: {value}")
    print("=" * 80)
    main(configs, phase1_checkpoint=args.phase1_checkpoint)
