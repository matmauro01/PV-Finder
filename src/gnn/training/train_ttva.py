"""Training script for GNN Track-to-Vertex Association model.

Trains a heterogeneous GAT on bipartite track-vertex graphs, predicting
which tracks belong to which primary vertex.
Based on atlas_pvfinder/tracks_to_vertex/TrainATLAS_script_GATConv_edgeattr_TTVA.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import torch
import yaml

from gnn.models.ttva_gat import TTVAGATModel
from gnn.training.training_loop import trainNet
from pv_finder.utils.utilities import (
    count_parameters,
    load_checkpoint,
    save_checkpoint,
    save_to_mlflow,
    set_seed,
)


def main(
    configs: dict,
    resume_checkpoint: str | None = None,
    resume_epoch: int | None = None,
) -> None:
    """Run GNN TTVA training."""
    set_seed()

    device_id = configs["device_id"]
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-built graph data (list of HeteroData objects)
    event_data_list = torch.load(configs["data_file"], weights_only=False)

    # Train / val split
    idx_train = int(len(event_data_list) * configs["train_split"][0])
    idx_val = idx_train + int(len(event_data_list) * configs["train_split"][1])
    train_data = event_data_list[:idx_train]
    val_data = event_data_list[idx_train:idx_val]

    from torch_geometric.loader import DataLoader

    loader_kw = dict(
        batch_size=configs["batch_size"],
        shuffle=False,
        num_workers=configs.get("num_workers", 2),
        prefetch_factor=2,
    )
    train_loader = DataLoader(train_data, **loader_kw)
    val_loader = DataLoader(val_data, **loader_kw)
    print(f"{len(train_data)} training graphs, {len(val_data)} validation graphs")

    # MLflow
    mlflow.set_tracking_uri("file:/data/home/matmauro/codice/PV-Finder/mlruns")
    mlflow.set_experiment(configs["experimentname"])

    # Model
    model_cfg = configs.get("model_config", {})
    model = TTVAGATModel(
        track_input_size=model_cfg.get("track_input_size", 8),
        pv_input_size=model_cfg.get("pv_input_size", 2),
        hidden_dim=model_cfg.get("hidden_dim", 32),
        num_heads=model_cfg.get("num_heads", 4),
        edge_attr_dim=model_cfg.get("edge_attr_dim", 3),
        dropout=model_cfg.get("dropout", 0.25),
    )
    model.to(device)

    # Materialize lazy GATConv parameters (in_channels=(-1, -1)) with a dummy
    # forward pass — required before the optimizer sees model.parameters().
    model.eval()
    with torch.no_grad():
        model(train_data[0].clone().to(device))
    print(f"GNN model:\n{model}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs["learning_rate"], betas=(0.9, 0.999)
    )

    # Optional epoch-stepped LR schedule + gradient clipping (stability)
    scheduler = None
    sched_cfg = configs.get("lr_schedule")
    if sched_cfg:
        if sched_cfg.get("type") != "cosine":
            msg = f"Unsupported lr_schedule type: {sched_cfg.get('type')}"
            raise ValueError(msg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=configs["n_epochs"],
            eta_min=sched_cfg.get("eta_min", 1e-5),
        )
        print(f"LR schedule: cosine -> {sched_cfg.get('eta_min', 1e-5)}")
    grad_clip = configs.get("grad_clip")
    if grad_clip:
        print(f"Gradient clipping: max norm {grad_clip}")

    # Resume from checkpoint
    epoch_start = 0
    if resume_checkpoint is not None:
        print(f"Resuming from: {resume_checkpoint}")
        if str(resume_checkpoint).endswith(".pth"):
            model, loaded_epoch = load_checkpoint(
                model, optimizer, path=resume_checkpoint
            )
            model.to(device)
            epoch_start = int(loaded_epoch) + 1
        else:
            loaded = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(loaded.state_dict(), strict=False)
            model.to(device)
            epoch_start = int(resume_epoch) + 1 if resume_epoch is not None else 0
        print(f"  Continuing from epoch {epoch_start}")

    parameters = count_parameters(model)
    runname = configs["runname"]
    print(f"Trainable parameters: {parameters:,}")
    print(
        f"Starting {runname}: {configs['n_epochs']} epochs, "
        f"bs={configs['batch_size']}, lr={configs['learning_rate']}"
    )

    train_iter = enumerate(
        trainNet(
            model,
            optimizer,
            train_loader,
            val_loader,
            configs["n_epochs"],
            notebook=False,
            epoch_start=epoch_start,
            scheduler=scheduler,
            grad_clip=grad_clip,
        )
    )

    with mlflow.start_run(run_name=runname):
        mlflow.log_artifact(str(Path(__file__)))
        if "config_file" in configs:
            mlflow.log_artifact(configs["config_file"])

        for i, result in train_iter:
            torch.save(model, "run_stats.pyt")
            mlflow.log_artifact("run_stats.pyt")

            if i % configs["save_frequency"] == 0 and configs["save_weights"]:
                save_dir = Path(configs["save_folder"])
                save_dir.mkdir(parents=True, exist_ok=True)
                ckpt = save_dir / f"{runname}_epoch_{result.epoch}.pyt"
                torch.save(model.state_dict(), ckpt)
                mlflow.log_artifact(str(ckpt))
                full = save_dir / f"{runname}_epoch_{result.epoch}_fullstate.pth"
                save_checkpoint(
                    model, optimizer, result.epoch, result.val, path=str(full)
                )
                mlflow.log_artifact(str(full))

            save_to_mlflow(
                {
                    "Metric: Training loss": result.cost,
                    "Metric: Validation loss": result.val,
                    "Param: Parameters": parameters,
                    "Param: Epochs": configs["n_epochs"],
                    "Param: Learning rate": configs["learning_rate"],
                    "Param: Batch size": configs["batch_size"],
                },
                step=i,
            )

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GNN Track-to-Vertex Association model for ATLAS data"
    )
    parser.add_argument(
        "-c", "--config_file", help="Path to YAML config", type=str, required=True
    )
    parser.add_argument(
        "-r", "--resume_checkpoint", help="Checkpoint to resume from", type=str
    )
    parser.add_argument(
        "--resume_epoch", help="Epoch for legacy .pyt checkpoints", type=int
    )

    args = parser.parse_args()

    with open(args.config_file) as f:
        try:
            configs = yaml.safe_load(f)
            configs["config_file"] = args.config_file
        except yaml.YAMLError as exc:
            print(f"Error loading config: {exc}")
            sys.exit(1)

    print("Configuration:")
    for key, value in configs.items():
        if key != "config_file":
            print(f"  {key}: {value}")

    main(configs, args.resume_checkpoint, args.resume_epoch)
