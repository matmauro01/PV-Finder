"""HL-LHC PU200 end-to-end training driver.

The recipe runs in two phases:

Phase 1 — MLP warm-up
    Only the tracks-to-KDE MLP is trained; the UNet (KDE-to-hist) is frozen.
    Plain ``Adam`` with a fixed learning rate. This stabilises the MLP
    output distribution before letting gradients flow through the UNet.

Phase 2 — full end-to-end
    All parameters are unfrozen. Optimiser (``adam`` or ``sgd``) and a
    cosine LR schedule (optionally with linear warmup, or with warm
    restarts) drive the joint optimisation. Gradient clipping is applied
    if ``max_grad_norm`` is set, and an optional 1-D TV penalty
    (``tv_lambda``) regularises the predicted histogram.

Both v1 (``trackstoHists_UNet_1000``) and v2 (``TracksToHist_v2``)
architectures are supported, selected from the YAML config via
``models_config.model_type``. MLflow captures per-epoch metrics plus
optional mid-epoch mini-validation telemetry (see ``mini_val.py``).

Public entry points: ``main``, ``run_phase1``, ``run_phase2``,
``train_phase1_epoch``, ``train_phase2_epoch``.
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
from tqdm import tqdm

from pv_finder.data.collectdata_poca_KDE import (
    collect_data_poca_ATLAS as collect_data_poca,
)
from pv_finder.training.hllhc_helpers import (
    build_model,
    build_phase2_scheduler,
    clip_and_step,
    get_param_groups_v2,
    is_v2,
    load_model_state,
    phase1_forward,
    tv_loss,
)
from pv_finder.training.mini_val import (
    mini_val_full,
    mini_val_mse,
    prefetch_val_batches,
)
from pv_finder.training.train_mlp_hist_then_e2e import get_parameter_groups
from pv_finder.training.training import validate as _validate_full
from pv_finder.utils.utilities import (
    count_parameters,
    save_checkpoint,
    save_to_mlflow,
    set_seed,
)

LOGGER = logging.getLogger("hllhc_e2e")
SCRIPT_PATH = Path(__file__)
SEPARATOR = "=" * 80


# --- Setup ---
def setup_logging(log_folder: Path, runname: str) -> Path:
    """Attach a stdout + file handler to ``LOGGER`` and return the log path."""
    log_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_folder / f"{runname}_{timestamp}.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        handler.setFormatter(fmt)
        LOGGER.addHandler(handler)
    return log_path


# --- Phase 1: MLP warmup (UNet frozen) ---
def train_phase1_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch_idx: int,
    max_norm: float | None,
    log_every: int = 500,
) -> float:
    """Run one Phase 1 training epoch and return the mean batch loss.

    Only the MLP path is exercised; the UNet must already have been frozen
    by the caller (``run_phase1``).
    """
    model.train()
    total = 0.0
    win_sum, win_n = 0.0, 0
    n_per = len(loader)
    pbar = tqdm(
        loader, desc=f"P1 E{epoch_idx}", leave=False, ncols=100, mininterval=2.0
    )
    for step, (inputs, target_split) in enumerate(pbar):
        inputs = inputs.to(device).float()
        target_hist = target_split.to(device).float()[:, 0, :]
        optimizer.zero_grad()
        pred = phase1_forward(model, inputs)
        loss = loss_fn(pred, target_hist)
        loss.backward()
        clip_and_step(model, optimizer, max_norm)
        batch_loss = loss.item()
        total += batch_loss
        win_sum += batch_loss
        win_n += 1
        gstep = (epoch_idx - 1) * n_per + step + 1
        if log_every > 0 and gstep % log_every == 0:
            mlflow.log_metric("p1_train_loss_step", win_sum / win_n, step=gstep)
            mlflow.log_metric("p1_lr", optimizer.param_groups[0]["lr"], step=gstep)
            win_sum, win_n = 0.0, 0
        pbar.set_postfix({"loss": f"{total / (step + 1):.6f}"})
    return total / len(loader)


@torch.no_grad()
def validate_phase1_epoch(
    model: nn.Module, loader, loss_fn: nn.Module, device: torch.device
) -> float:
    """Validation pass through the MLP-only Phase 1 forward."""
    model.eval()
    total = 0.0
    for inputs, target_split in loader:
        inputs = inputs.to(device).float()
        target_hist = target_split.to(device).float()[:, 0, :]
        pred = phase1_forward(model, inputs)
        total += loss_fn(pred, target_hist).item()
    return total / len(loader)


def run_phase1(
    model: nn.Module,
    data: tuple,
    configs: dict,
    device: torch.device,
    save_folder: Path,
) -> None:
    """Drive the Phase 1 (MLP warmup) loop: freeze UNet, train MLP, checkpoint."""
    train_loader, val_loader = data
    if is_v2(model):
        mlp_params, unet_params = get_param_groups_v2(model)
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

    LOGGER.info(SEPARATOR)
    LOGGER.info(
        "PHASE 1 — MLP warmup | %d epochs | lr=%g | grad_clip=%s",
        n_epochs,
        configs["phase1_learning_rate"],
        max_norm,
    )
    LOGGER.info(SEPARATOR)

    log_every = int(configs.get("log_every_n_steps", 500))
    best_val, best_ep = float("inf"), 0
    for ep in range(1, n_epochs + 1):
        tr = train_phase1_epoch(
            model, train_loader, optimizer, loss_fn, device, ep, max_norm, log_every
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


# --- Phase 2: full E2E with LR schedule + grad clipping ---
def train_phase2_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch_idx: int,
    max_norm: float | None,
    tv_lambda: float = 0.0,
    log_every: int = 500,
    mini_val_cache: list | None = None,
    mini_val_every: int = 0,
    mini_val_eff_every: int = 0,
) -> float:
    """Run one Phase 2 (full E2E) training epoch.

    Returns the mean batch loss. Mid-epoch MLflow telemetry (train loss
    windowed mean, LR, optional mini-val MSE / efficiency) is logged based
    on the ``log_every`` / ``mini_val_*`` cadences.
    """
    model.train()
    total = 0.0
    win_sum, win_n = 0.0, 0
    n_per = len(loader)
    pbar = tqdm(
        loader, desc=f"P2 E{epoch_idx}", leave=False, ncols=100, mininterval=2.0
    )
    for step, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out, labels[:, 0, :])
        if tv_lambda > 0:
            loss = loss + tv_lambda * tv_loss(out)
        loss.backward()
        clip_and_step(model, optimizer, max_norm)
        batch_loss = loss.item()
        total += batch_loss
        win_sum += batch_loss
        win_n += 1
        gstep = (epoch_idx - 1) * n_per + step + 1
        if log_every > 0 and gstep % log_every == 0:
            mlflow.log_metric("p2_train_loss_step", win_sum / win_n, step=gstep)
            mlflow.log_metric("p2_lr", optimizer.param_groups[0]["lr"], step=gstep)
            win_sum, win_n = 0.0, 0
        if (
            mini_val_cache
            and mini_val_eff_every > 0
            and gstep % mini_val_eff_every == 0
        ):
            vl, eff = mini_val_full(model, mini_val_cache, loss_fn, device)
            mlflow.log_metric("p2_val_loss_step", vl, step=gstep)
            mlflow.log_metric("p2_eff_step", eff.eff_rate, step=gstep)
            mlflow.log_metric("p2_fpr_step", eff.fp_rate, step=gstep)
        elif mini_val_cache and mini_val_every > 0 and gstep % mini_val_every == 0:
            mlflow.log_metric(
                "p2_val_loss_step",
                mini_val_mse(model, mini_val_cache, loss_fn, device),
                step=gstep,
            )
        pbar.set_postfix({"loss": f"{total / (step + 1):.6f}"})
    return total / len(loader)


@torch.no_grad()
def validate_phase2_epoch(
    model: nn.Module, loader, loss_fn: nn.Module, device: torch.device
):
    """Full validation: MSE + per-item efficiency. Returns ``(mean_loss, eff)``."""
    total, eff = _validate_full(
        model, loss_fn, loader, device, out_idx=0, progress=lambda x, **_: x
    )
    return total / len(loader), eff


def run_phase2(
    model: nn.Module,
    data: tuple,
    configs: dict,
    device: torch.device,
    save_folder: Path,
) -> None:
    """Drive the Phase 2 (full E2E) loop: build optimiser + scheduler, train, save."""
    train_loader, val_loader = data
    base_lr = configs["phase2_learning_rate"]
    n_epochs = configs["phase2_epochs"]
    warmup = configs.get("phase2_warmup_epochs", 0)
    eta_min_frac = configs.get("phase2_cosine_eta_min_frac", 0.01)
    max_norm = configs.get("max_grad_norm")
    save_freq = configs["save_frequency"]
    runname = configs["runname"]
    tv_lambda = configs.get("tv_lambda", 0.0)
    warm_restarts_t0 = configs.get("warm_restarts_t0", 0)

    opt_name = configs.get("optimizer", "adam")
    if opt_name == "sgd":
        wd = configs.get("weight_decay", 1e-4)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=base_lr, momentum=0.9, weight_decay=wd
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()
    scheduler = build_phase2_scheduler(
        optimizer, warmup, n_epochs, base_lr, eta_min_frac, warm_restarts_t0
    )

    sched = (
        f"wr(T0={warm_restarts_t0})" if warm_restarts_t0 > 0 else f"warmup={warmup}+cos"
    )
    LOGGER.info(SEPARATOR)
    LOGGER.info(
        "PHASE 2 — full E2E | %d epochs | lr=%g | opt=%s | sched=%s | clip=%s | tv=%g",
        n_epochs,
        base_lr,
        opt_name,
        sched,
        max_norm,
        tv_lambda,
    )
    LOGGER.info(SEPARATOR)

    log_every = int(configs.get("log_every_n_steps", 500))
    mini_val_every = int(configs.get("mini_val_every_n_steps", 0))
    mini_val_eff_every = int(configs.get("mini_val_eff_every_n_steps", 0))
    mini_val_n_batches = int(configs.get("mini_val_n_batches", 30))
    mini_val_cache: list = []
    if mini_val_every > 0 or mini_val_eff_every > 0:
        mini_val_cache = prefetch_val_batches(val_loader, mini_val_n_batches)
        LOGGER.info(
            "Mini-val cache: %d batches | MSE every %d steps | full eff every %d steps",
            len(mini_val_cache),
            mini_val_every,
            mini_val_eff_every,
        )
    best_val, best_ep = float("inf"), 0
    for ep in range(1, n_epochs + 1):
        tr = train_phase2_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            ep,
            max_norm,
            tv_lambda,
            log_every,
            mini_val_cache,
            mini_val_every,
            mini_val_eff_every,
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


# --- Entry point ---
def main(configs: dict, phase1_checkpoint: str | None = None) -> None:
    """Run Phase 1 (or skip via ``phase1_checkpoint``) then Phase 2.

    Side effects: configures logging, sets seeds, starts an MLflow run,
    builds the data loaders + model, and writes checkpoints/artifacts to
    ``configs["save_folder"]``.
    """
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
            prefetch_factor=configs.get("prefetch_factor", 4),
            train_split=configs["train_split"],
        )
        LOGGER.info("✓ %d train samples, %d val samples",
                    len(train_loader.dataset), len(val_loader.dataset))  # fmt: skip
        data = (train_loader, val_loader)

        if phase1_checkpoint is not None:
            ep = load_model_state(phase1_checkpoint, model)
            model.to(device)
            for p in model.parameters():
                p.requires_grad = True
            LOGGER.info(
                "✓ Skipping Phase 1 — loaded weights from %s (P1 ep %d)",
                Path(phase1_checkpoint).name,
                ep,
            )
        else:
            run_phase1(model, data, configs, device, save_folder)

        run_phase2(model, data, configs, device, save_folder)
        mlflow.log_artifact(str(log_path))

    LOGGER.info(SEPARATOR)
    LOGGER.info("✅ Training complete")
    LOGGER.info(SEPARATOR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HLLHC PU200 E2E training")
    parser.add_argument("-c", "--config_file", required=True, type=str)
    parser.add_argument(
        "--phase1-checkpoint",
        default=None,
        dest="phase1_checkpoint",
        help="Phase-1 fullstate.pth; if set, skips Phase 1 (model_state only).",
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        configs = yaml.safe_load(f)
        configs["config_file"] = args.config_file

    print(SEPARATOR)
    print("Configuration:")
    for key, value in configs.items():
        if key != "config_file":
            print(f"  {key}: {value}")
    print(SEPARATOR)
    main(configs, phase1_checkpoint=args.phase1_checkpoint)
