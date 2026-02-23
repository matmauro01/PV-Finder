import argparse

import mlflow
import torch
import torch.nn as nn
import yaml

from pv_finder.data.collectdata_poca_KDE import (
    collect_data_poca_ATLAS as collect_data_poca,
)
from pv_finder.models.autoencoder_models import MaskedDNN
from pv_finder.training.training import trainNet
from pv_finder.utils.utilities import (
    load_checkpoint,
    save_checkpoint,
    save_to_mlflow,
    set_seed,
)

# traintracksKDE()


def main(configs):
    # Set seed for reproducability
    set_seed()

    # Set GPU
    device_id = configs["device_id"]
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, _ = collect_data_poca(
        configs["data_file"],
        data_pipeline=configs["model_class"],
        batch_size=configs["batch_size"],
        masking=False,
        num_workers=configs.get("num_workers", 16),
        prefetch_factor=configs.get("prefetch_factor", 2),
        train_split=configs["train_split"],
    )

    # Set where you want the mlflow tracking to go
    mlflow.tracking.set_tracking_uri("file:/data/home/matmauro/codice/PV-Finder/mlruns")
    mlflow.set_experiment(configs["experimentname"])

    # Define model
    model = MaskedDNN(
        input_size=configs["models_config"]["MaskedDNN"]["input_size"],
        hidden_nodes=configs["models_config"]["MaskedDNN"]["hidden_nodes"],
        output_size=configs["models_config"]["MaskedDNN"]["output_size"],
        leaky_param=configs["models_config"]["MaskedDNN"]["leaky_param"],
        use_bn=configs["models_config"]["MaskedDNN"]["use_bn"],
        use_drop=configs["models_config"]["MaskedDNN"]["use_drop"],
        maskVal=configs["models_config"]["MaskedDNN"]["maskVal"],
        predScaleFactor=configs["models_config"]["MaskedDNN"]["predScaleFactor"],
        allow_negative_output=configs["models_config"]["MaskedDNN"][
            "allow_negative_output"
        ],
    )

    print("This is the selected neural network model: ", model)
    out_idx = configs["models_config"]["MaskedDNN"]["out_idx"]
    print("Your out index: ", out_idx)

    model.to(device)

    loss = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=configs["learning_rate"])

    # Optional resume from checkpoint (after optimizer is created)
    # Priority: resume_checkpoint takes precedence over continue_prior
    epoch_start = 0
    resume_path = configs.get("resume_checkpoint", None)

    if resume_path:
        # New resume mechanism - recommended for resuming interrupted training
        # - If resume_checkpoint is a .pth full checkpoint, restore model + optimizer and start from loaded_epoch + 1
        # - If it's a legacy .pyt model object, restore weights only and use provided resume_epoch
        print(f"\n{'=' * 80}")
        print(f"RESUMING FROM CHECKPOINT: {resume_path}")
        print(f"{'=' * 80}\n")

        if str(resume_path).endswith(".pth"):
            model, loaded_epoch = load_checkpoint(model, optimizer, path=resume_path)
            model.to(device)
            epoch_start = int(loaded_epoch) + 1
            print(
                f"✓ Loaded full checkpoint with optimizer state from epoch {loaded_epoch}"
            )
            print(f"✓ Training will continue from epoch {epoch_start}\n")
        else:
            loaded_model = torch.load(resume_path, map_location=device)
            model.load_state_dict(loaded_model.state_dict(), strict=False)
            model.to(device)
            epoch_start = int(configs.get("resume_epoch", 0))
            print("✓ Loaded model weights (no optimizer state)")
            print(f"✓ Training will continue from epoch {epoch_start}\n")

    elif configs.get("continue_prior", False):
        # Legacy mechanism - for loading pretrained weights from older training runs
        print(f"\n{'=' * 80}")
        print("LOADING PRIOR WEIGHTS (legacy continue_prior)")
        print(f"{'=' * 80}\n")

        if out_idx == 0:
            prior_weights = configs["DNN_KDE_A_z_prior_weights"]
        elif out_idx == 1:
            prior_weights = configs["DNN_KDE_B_z_prior_weights"]
        elif out_idx == 2:
            prior_weights = configs["DNN_KDE_A_x_prior_weights"]
        elif out_idx == 3:
            prior_weights = configs["DNN_KDE_A_y_prior_weights"]
        else:
            raise ValueError(f"Invalid out_idx {out_idx} for continue_prior")

        model, loaded_epoch = load_checkpoint(model, optimizer, path=prior_weights)
        epoch_start = int(loaded_epoch) + 1
        print(f"✓ Loaded prior weights from epoch {loaded_epoch}")
        print(f"✓ Training will continue from epoch {epoch_start}\n")

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Train
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
    with mlflow.start_run(run_name=configs["runname"]) as run:  # noqa: F841
        mlflow.log_artifact("training/train_tracks_to_kde.py")
        for i, result in train_iter:
            print(result.cost)
            torch.save(model, "run_stats.pyt")
            mlflow.log_artifact("run_stats.pyt")

            ## save each epoch's model state dictionary to separate folder
            ## use to load weights from specific epoch (choose using mlflow)
            if i % configs["save_frequency"] == 0 and configs["save_weights"]:
                output = (
                    configs["save_folder"]
                    + configs["runname"]
                    + "_epoch_"
                    + str(result.epoch)
                    + ".pyt"
                )
                torch.save(model, output)
                mlflow.log_artifact(output)
                full_ckpt = (
                    configs["save_folder"]
                    + configs["runname"]
                    + "_epoch_"
                    + str(result.epoch)
                    + "_fullstate.pth"
                )
                save_checkpoint(
                    model, optimizer, result.epoch, result.val, path=full_ckpt
                )
                mlflow.log_artifact(full_ckpt)

            save_to_mlflow(
                {
                    "Metric: Training loss": result.cost,
                    "Metric: Validation loss": result.val,
                    "Metric: Efficiency": result.eff_val.eff_rate,
                    "Metric: False positive rate": result.eff_val.fp_rate,
                    "Param: Parameters": parameters,
                    "Param: Asymmetry": configs["asymmetry_param"],
                    "Param: Epochs": configs["n_epochs"],
                },
                step=i,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program trains a PV-Finder UNet model for ATLAS data."
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help="Path to configuration YAML file",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    # Open and print yaml file
    with open(args.config_file) as file:
        try:
            configs = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print("These are your yaml file configs: ", configs)

    main(configs)
