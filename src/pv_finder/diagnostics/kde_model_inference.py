"""
Load the T2KDE (tracks-to-KDE) neural network and run inference on MC / Run 3 data.

The model is a MaskedDNN saved via ``torch.save(model, path)`` (full pickle).
It was originally saved with the class at ``model.autoencoder_models.MaskedDNN``,
so we install a sys.modules alias before loading.

CRITICAL padding note
---------------------
The model masks tracks using ``z0 > maskVal`` where ``maskVal = -240.0``.
Feature-loading code pads with ``MASK_VAL = -999999``.  Before feeding tensors
to the model we must **re-pad with -240.0** so the mask works correctly.
"""

import sys
import types
import warnings

import numpy as np
import torch
from tqdm import tqdm

import pv_finder.models.autoencoder_models as am
from pv_finder.data.feature_loading import (
    MASK_VAL,
    N_SUBEVENTS,
    build_run3_subevent_tensor,
)

# Padding value expected by the model (NOT the -999999 from feature_loading).
# MaskedDNN.forward() masks via ``z0 > self.maskVal`` with maskVal = -240.0,
# so columns padded with -240.0 are correctly excluded.
MODEL_PAD_VAL = -240.0


# ============================================================================
# Model loading
# ============================================================================


def load_t2kde_model(model_path, device="cpu"):
    """Load a pickled MaskedDNN model, fixing the old module path.

    The model was saved when the class lived at ``model.autoencoder_models``.
    We create module aliases so ``torch.load`` can resolve the old path to
    ``pv_finder.models.autoencoder_models``.
    """
    # Pickle alias: old save used 'model.autoencoder_models'
    sys.modules["model"] = types.ModuleType("model")
    sys.modules["model.autoencoder_models"] = am

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    return model


# ============================================================================
# Tensor helpers
# ============================================================================


def _pad_to_length(tensor, length, pad_val=MODEL_PAD_VAL):
    """Pad or truncate a (7, N) numpy array to (7, length).

    Parameters
    ----------
    tensor : np.ndarray
        Shape ``(7, N)``.
    length : int
        Desired second-axis size.
    pad_val : float
        Value used for padding columns (default ``-240.0``).

    Returns
    -------
    np.ndarray
        Shape ``(7, length)``.
    """
    n_feat, n_tracks = tensor.shape
    if n_tracks == length:
        return tensor
    if n_tracks > length:
        warnings.warn(
            f"Truncating tensor from {n_tracks} to {length} tracks",
            stacklevel=2,
        )
        return tensor[:, :length]
    padded = np.full((n_feat, length), pad_val, dtype=np.float32)
    padded[:, :n_tracks] = tensor
    return padded


def _repad(tensor, old_val=MASK_VAL, new_val=MODEL_PAD_VAL):
    """Replace the feature-loading padding sentinel with the model's value.

    A track column is considered padded when its z0 (channel 1)
    satisfies ``z0 <= old_val + 1``.  All 7 channels of that column
    are set to *new_val*.

    Returns a copy; the input is never modified.
    """
    out = np.array(tensor, dtype=np.float32, copy=True)
    padded_mask = out[1, :] <= (old_val + 1)
    out[:, padded_mask] = new_val
    return out


# ============================================================================
# Per-subevent input builders
# ============================================================================


def build_mc_subevent_input(tracks_subevent):
    """Prepare one MC sub-event tensor for model inference.

    Parameters
    ----------
    tracks_subevent : np.ndarray
        Shape ``(7, N_max)`` straight from the H5 file, padded with
        ``MASK_VAL = -999999``.

    Returns
    -------
    np.ndarray
        Shape ``(7, N_max)`` with padding replaced by ``-240.0``.
    """
    return _repad(tracks_subevent)


def build_run3_subevent_input(event_dict, sub_idx):
    """Build one Run 3 sub-event tensor ready for model inference.

    Parameters
    ----------
    event_dict : dict
        Run 3 event with keys ``z0, d0, d0_err, z0_err, d0_z0_cov``.
    sub_idx : int
        Sub-event index (0..11).

    Returns
    -------
    np.ndarray
        Shape ``(7, N_tracks)`` with padding at ``-240.0``.
    """
    tensor, _ = build_run3_subevent_tensor(
        event_dict["z0"],
        event_dict["d0"],
        event_dict["d0_err"],
        event_dict["z0_err"],
        event_dict["d0_z0_cov"],
        sub_idx,
    )
    return _repad(tensor)


# ============================================================================
# Batch inference
# ============================================================================


def run_model_on_events(model, events, dataset_type, device="cpu", batch_size=64):
    """Run the T2KDE model on a list of events.

    For each event, 12 subevent tensors are built, re-padded from
    ``-999999`` to ``-240.0``, batched, and fed through the model.

    Parameters
    ----------
    model : torch.nn.Module
        A ``MaskedDNN`` model in eval mode.
    events : list of dict
        MC events (``"tracks"`` key with shape ``(12, 7, 695)``) or
        Run 3 events (dicts with ``z0, d0, ...`` arrays).
    dataset_type : str
        ``"mc"`` or ``"run3"``.
    device : str
        Torch device string.
    batch_size : int
        Number of subevents per forward pass.

    Returns
    -------
    list of np.ndarray
        One ``(12, 1000)`` array per event.
    """
    if dataset_type not in ("mc", "run3"):
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    # -- collect all subevent tensors (already re-padded to -240.0) --------
    subevent_tensors = []
    for evt in events:
        for si in range(N_SUBEVENTS):
            if dataset_type == "mc":
                tensor = build_mc_subevent_input(evt["tracks"][si])
            else:
                tensor = build_run3_subevent_input(evt, si)
            subevent_tensors.append(tensor)

    n_total = len(subevent_tensors)
    outputs = []

    with torch.no_grad():
        for start in tqdm(
            range(0, n_total, batch_size), desc="T2KDE inference", leave=False
        ):
            batch_np = subevent_tensors[start : start + batch_size]

            # Pad every tensor in the mini-batch to the same track count
            max_tracks = max(t.shape[1] for t in batch_np)
            padded = np.stack(
                [_pad_to_length(t, max_tracks) for t in batch_np]
            )  # (B, 7, max_tracks)

            batch_t = torch.from_numpy(padded).float().to(device)
            preds = model(batch_t)  # (B, 1000)
            outputs.append(preds.cpu().numpy())

    all_outputs = np.concatenate(outputs, axis=0)  # (n_total, 1000)

    # -- reshape back to per-event (12, 1000) ------------------------------
    results = []
    for i in range(len(events)):
        start = i * N_SUBEVENTS
        results.append(all_outputs[start : start + N_SUBEVENTS])

    return results
