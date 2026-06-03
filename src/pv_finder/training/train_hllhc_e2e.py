"""HLLHC PU200 E2E training: Phase 1 (MLP warmup) + Phase 2 (full E2E).

Supports v1/v2 archs, Adam/SGD, cosine/warm-restart, optional TV loss.
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
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from tqdm import tqdm

from pv_finder.data.collectdata_poca_KDE import (
    collect_data_poca_ATLAS as collect_data_poca,
)
from pv_finder.models.autoencoder_models import MaskedDNN, trackstoHists_UNet_1000
from pv_finder.models.unet_v2 import TracksToHist_v2, UNet_1000_v2
from pv_finder.training.mini_val import (
    mini_val_full,
    mini_val_mse,
    prefetch_val_batches,
)
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


# --- Setup ---
def setup_logging(log_folder: Path, runname: str) -> Path:
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


def _tv_loss(pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))


def build_phase2_scheduler(
    optimizer, warmup_epochs, total_epochs, base_lr, eta_min_frac, warm_restarts_t0=0
):
    """Linear warmup -> cosine decay. Optionally cosine with warm restarts."""
    eta_min = base_lr * eta_min_frac
    if warm_restarts_t0 > 0:
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=warm_restarts_t0, T_mult=1, eta_min=eta_min
        )
    if warmup_epochs > 0:
        w = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        c = CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min
        )
        return SequentialLR(optimizer, schedulers=[w, c], milestones=[warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)


# --- Shared step helpers ---
def _clip_and_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, max_norm: float | None
) -> None:
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()


def _is_v2(model: nn.Module) -> bool:
    return isinstance(model, TracksToHist_v2)


def _get_param_groups_v2(model: TracksToHist_v2):
    return list(model.t2kde.parameters()), list(model.k2h.parameters())


def _forward_mlp_v2(model: TracksToHist_v2, inputs: torch.Tensor) -> torch.Tensor:
    kde = model.t2kde(inputs)  # (B, n_latent * 1000)
    return kde.view(kde.shape[0], model.n_latent, 1000)  # (B, n_latent, 1000)


def _load_model_state(path: str, model: nn.Module) -> int:
    """Load model_state only (skip optimizer_state); resumes P2 from P1 ckpt."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return int(ckpt.get("epoch", -1))


# --- Phase 1: MLP warmup (UNet frozen) ---
def _phase1_forward(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Phase 1 MLP-only forward, dispatching by model type."""
    if _is_v2(model):
        return _squeeze_hist(_forward_mlp_v2(model, inputs))
    return _squeeze_hist(forward_mlp_hist(model, inputs))


def train_phase1_epoch(
    model, loader, optimizer, loss_fn, device, epoch_idx, max_norm, log_every=500
):
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
        pred = _phase1_forward(model, inputs)
        loss = loss_fn(pred, target_hist)
        loss.backward()
        _clip_and_step(model, optimizer, max_norm)
        L = loss.item()
        total += L
        win_sum += L
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

    LOGGER.info(
        "=" * 80
        + "\nPHASE 1 — MLP warmup | %d epochs | lr=%g | grad_clip=%s\n"
        + "=" * 80,
        n_epochs,
        configs["phase1_learning_rate"],
        max_norm,
    )
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
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    epoch_idx,
    max_norm,
    tv_lambda=0.0,
    log_every=500,
    mini_val_cache=None,
    mini_val_every=0,
    mini_val_eff_every=0,
):
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
            loss = loss + tv_lambda * _tv_loss(out)
        loss.backward()
        _clip_and_step(model, optimizer, max_norm)
        L = loss.item()
        total += L
        win_sum += L
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
    total, eff = _validate_full(
        model, loss_fn, loader, device, out_idx=0, progress=lambda x, **_: x
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
    LOGGER.info("P2 | %d ep | lr=%g | %s | %s | clip=%s | tv=%g",
                n_epochs, base_lr, opt_name, sched, max_norm, tv_lambda)  # fmt: skip
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
            ep = _load_model_state(phase1_checkpoint, model)
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

    LOGGER.info("=" * 80 + "\n✅ Training complete\n" + "=" * 80)


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

    print("=" * 80)
    print("Configuration:")
    for key, value in configs.items():
        if key != "config_file":
            print(f"  {key}: {value}")
    print("=" * 80)
    main(configs, phase1_checkpoint=args.phase1_checkpoint)
