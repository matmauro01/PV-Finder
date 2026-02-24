"""Load and run the e2e model on MC and Run 3 events.

Feeds ALL tracks per subevent without truncation.  Padding uses -240.0 so
that padded entries are masked out by the model's z0 threshold (maskVal =
-240.0).  The model class name in the .pyt file may differ from the weight
filename (e.g. trackstoHists_UNet_1000 vs e2e_mlpHist50_...).
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from pv_finder.data.feature_loading import (
    MASK_VAL,
    N_SUBEVENTS,
    build_run3_subevent_tensor,
)

# Pad value fed to the model — model masks tracks with z0 <= -240.0
_MODEL_PAD = -240.0


def load_e2e_model(model_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load trackstoHists_UNet_1000 model from a .pyt file.

    The file was saved with torch.save(model, path) so we reload the full
    object directly.
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def _repad(tensor: np.ndarray) -> np.ndarray:
    """Replace MASK_VAL (-999999) padding with model maskVal (-240.0).

    Parameters
    ----------
    tensor:
        Shape (7, N_tracks). MC H5 uses MASK_VAL = -999999 for padding;
        the model masks tracks whose z0 (channel 1) <= -240.0, so any
        value << -240 is already masked but we normalise for cleanliness.

    Returns
    -------
    np.ndarray
        New array with padded-track slots set to -240.0 on all channels.
    """
    out = tensor.copy()
    # A track is padded if z0 is near MASK_VAL
    padded = out[1, :] <= (MASK_VAL + 1.0)  # MASK_VAL = -999999
    out[:, padded] = _MODEL_PAD
    return out


def run_e2e_on_events(
    model: torch.nn.Module,
    events: list[dict],
    dataset_type: str,
    device: str = "cpu",
    batch_size: int = 32,
) -> list[np.ndarray]:
    """Run the e2e model on all events, returning per-event (12, 1000) arrays.

    Parameters
    ----------
    model:
        Loaded trackstoHists_UNet_1000, already in eval mode.
    events:
        List of event dicts as returned by load_mc_data or load_run3_data.
    dataset_type:
        ``"mc"`` or ``"run3"``.
    device:
        PyTorch device string.
    batch_size:
        Number of subevents per inference batch.

    Returns
    -------
    list of np.ndarray
        One array per event, shape (12, 1000), dtype float32.
        Empty subevents (0 real tracks in Run 3) are filled with zeros.
    """
    if dataset_type not in ("mc", "run3"):
        raise ValueError(f"dataset_type must be 'mc' or 'run3', got {dataset_type!r}")

    # ------------------------------------------------------------------
    # Collect all (event_idx, sub_idx, tensor_7xN) triples
    # ------------------------------------------------------------------
    triples: list[tuple[int, int, np.ndarray]] = []
    # Track which (event_idx, sub_idx) pairs are empty (Run 3 only)
    empty_pairs: set[tuple[int, int]] = set()

    for ev_i, evt in enumerate(events):
        if dataset_type == "mc":
            tracks = evt["tracks"]  # (12, 7, 695)
            for si in range(N_SUBEVENTS):
                t = _repad(tracks[si])  # (7, 695)
                triples.append((ev_i, si, t))
        else:  # run3
            z0 = evt["z0"]
            d0 = evt["d0"]
            d0_err = evt["d0_err"]
            z0_err = evt["z0_err"]
            d0_z0_cov = evt["d0_z0_cov"]
            for si in range(N_SUBEVENTS):
                t, n_trk = build_run3_subevent_tensor(
                    z0, d0, d0_err, z0_err, d0_z0_cov, si
                )
                if n_trk == 0:
                    empty_pairs.add((ev_i, si))
                else:
                    triples.append((ev_i, si, t))

    # ------------------------------------------------------------------
    # Allocate output arrays
    # ------------------------------------------------------------------
    n_events = len(events)
    results: list[np.ndarray] = [
        np.zeros((N_SUBEVENTS, 1000), dtype=np.float32) for _ in range(n_events)
    ]

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------
    n_batches = (len(triples) + batch_size - 1) // batch_size

    with torch.no_grad():
        for b_start in tqdm(
            range(0, len(triples), batch_size), total=n_batches, desc="e2e inference"
        ):
            batch = triples[b_start : b_start + batch_size]

            # Pad all tensors in this batch to the same N_tracks
            max_n = max(t.shape[1] for _, _, t in batch)
            padded_list = []
            for _, _, t in batch:
                n = t.shape[1]
                if n < max_n:
                    pad = np.full((7, max_n - n), _MODEL_PAD, dtype=np.float32)
                    t = np.concatenate([t, pad], axis=1)
                padded_list.append(t)

            # Stack to (B, 7, max_N) and run model
            x = torch.tensor(np.stack(padded_list, axis=0), dtype=torch.float32).to(
                device
            )

            out = model(x)  # (B, 1000) or (1000,) when B=1

            # Re-attach the batch dimension if squeeze() collapsed it
            if out.dim() == 1:
                out = out.unsqueeze(0)

            out_np = out.cpu().numpy().astype(np.float32)  # (B, 1000)

            for bi, (ev_i, si, _) in enumerate(batch):
                results[ev_i][si] = out_np[bi]

    return results
