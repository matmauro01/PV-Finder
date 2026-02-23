##### IMPORTS #####
import argparse

import mlflow
import torch
import yaml

from pv_finder.data.collectdata_poca_KDE import (
    collect_data_poca_ATLAS as collect_data_poca,
)
from pv_finder.models.alt_loss_A import Loss

# from pv_finder.models.autoencoder_models import trackstoHists_UNet_400, UNet_400
from pv_finder.models.autoencoder_models import UNet_1000
from pv_finder.training.training import trainNet
from pv_finder.utils.utilities import (
    load_checkpoint,
    save_checkpoint,
    save_to_mlflow,
    set_seed,
)


def main(configs):
    # Set seed for reproducability
    set_seed()

    # Set GPU
    device_id = configs["device_id"]
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = collect_data_poca(
        filepath=configs["data_file"],
        data_pipeline=configs["model_class"],
        batch_size=configs["batch_size"],
        masking=False,
        num_workers=configs.get("num_workers", 8),
        prefetch_factor=configs.get("prefetch_factor", 2),
        train_split=configs["train_split"],
    )

    mlflow.tracking.set_tracking_uri("file:/data/home/matmauro/codice/PV-Finder/mlruns")
    mlflow.set_experiment(configs["experimentname"])

    # print("Loaded model weights dictionary: ", best_model_state)

    model = UNet_1000(
        dropout_p=configs["models_config"]["UNet"].get("dropout_p", 0.25),
        n_features=configs["models_config"]["UNet"].get("n_features", 1),
    )

    # print("Model dict before: ", model.state_dict())

    # pretrained_model = torch.load("model_weights/initialized_weights_incbounds_t2h_KDEAUNet_400bins.pth")
    # pretrained_model = torch.load("model_weights/initialized_weights_fixindices_incbounds_t2h_KDEAUNet_400bins.pth")

    # HARD CODED: model_weights/initialized_weights_t2h_KDEAUNet_400bins.pth
    # model.load_state_dict(pretrained_model)

    # print("Model dict after: ", model.state_dict())

    model.to(device)

    # print("This is the selected neural network model: ", model)
    # Output index for Target_Y
    out_idx = configs["models_config"]["UNet"].get("out_idx", 0)
    # print("Your out index: ", out_idx)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs["learning_rate"], betas=(0.9, 0.999)
    )
    # Reduce LR on Plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # noqa: F841
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )
    loss = Loss(epsilon=1e-5, coefficient=configs["asymmetry_param"])

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_dict = model.state_dict()  # noqa: F841

    runname = configs["runname"]

    # Optional resume from checkpoint (after optimizer is created)
    epoch_start = 0
    resume_path = configs.get("resume_checkpoint", None)

    if resume_path:
        # Resume mechanism for interrupted training
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
        mlflow.log_artifact("training/train_kde_to_hist.py")
        for i, result in train_iter:
            print(result.cost)
            torch.save(model, "run_stats.pyt")
            mlflow.log_artifact("run_stats.pyt")

            ## save each epoch's model state dictionary to separate folder
            ## use to load weights from specific epoch (choose using mlflow)
            if i % configs["save_frequency"] == 0 and configs["save_weights"]:
                output = (
                    configs["save_folder"]
                    + runname
                    + "_epoch_"
                    + str(result.epoch)
                    + ".pyt"
                )
                torch.save(model, output)
                mlflow.log_artifact(output)
                full_ckpt = (
                    configs["save_folder"]
                    + runname
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

            # scheduler.step(result.val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program trains an end-to-end tracks-to-hist PV-Finder UNet model for ATLAS data."
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
