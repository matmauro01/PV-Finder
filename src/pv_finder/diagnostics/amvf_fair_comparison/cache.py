"""Build the compact position cache the fair comparison runs on.

Pulls PV-Finder's peak list out of an existing ``eval_results.pkl`` (so no
inference is re-run) and the AMVF vertices plus the *unfiltered* truth vertices
out of the source ROOT file.  Both are needed because the pkl stores neither:
``run_eval_pvf_run3.py`` saves the AMVF *categories* but not the AMVF positions,
and ``run3_io.py`` drops every truth vertex with nTracks < 2 at load time, which
is precisely the population the second asymmetry is about.

Alignment between the two sources is *proved*, not assumed: the per-event truth
multiplicity, AMVF multiplicity and mu read back from ROOT must equal what the
pkl recorded for the same event index, for every event.  A mismatch raises.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from pv_finder.utils.pairwise_dz import in_summary_window

TRUTH_BRANCHES = ("TruthVertex_z", "TruthVertex_nTracks")
RECO_BRANCHES = ("RecoVertex_z", "RecoVertex_nTracks")
EVENT_BRANCHES = ("ActualNumOfInt", "BeamPosZ")


def _flatten(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack a ragged list of 1-d arrays into (values, offsets)."""
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([len(a) for a in arrays], out=offsets[1:])
    if offsets[-1] == 0:
        return np.zeros(0, dtype=np.float32), offsets
    return np.concatenate(arrays).astype(np.float32), offsets


def unflatten(values: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """Inverse of :func:`_flatten`."""
    return [values[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def build_cache(  # noqa: PLR0913, PLR0915
    pkl_path: str | Path,
    root_path: str | Path,
    out_path: str | Path,
    *,
    mu_min: float = 185.0,
    mu_max: float = 215.0,
    tag: str = "",
) -> dict[str, Any]:
    """Extract aligned PV-Finder / AMVF / truth positions for the mu window.

    Returns a dict of summary metadata (also written into the npz as JSON).
    """
    import uproot  # lazy — only this entry point needs it

    pkl_path, root_path, out_path = Path(pkl_path), Path(root_path), Path(out_path)
    print(f"[cache] pkl : {pkl_path}")
    with open(pkl_path, "rb") as fh:
        res = pickle.load(fh)
    per_event = res["per_event"]
    pred = res["pred_pvs_mm"]
    pred_h = res["pred_heights"]
    truth_pkl = res["truth_pvs_mm"]
    if not (len(per_event) == len(pred) == len(truth_pkl)):
        raise ValueError(
            f"pkl arrays disagree in length: per_event={len(per_event)} "
            f"pred={len(pred)} truth={len(truth_pkl)}. The eval skips events "
            "with zero truth vertices when filling per_event, so this cache "
            "cannot assume index alignment."
        )
    n_read = len(per_event)
    print(f"[cache]   {n_read} events stored (every event read, not the mu subset)")

    # The pkl stores every event read, not the mu subset; select on the stored
    # per-event mu using the *same* predicate the eval summary uses, so this
    # cache and the published summary cannot describe different populations.
    ev_idx = np.array([r["event_idx"] for r in per_event], dtype=np.int64)
    sel = np.array(
        [
            k
            for k, r in enumerate(per_event)
            if in_summary_window(r["mu"], r["n_truth"], mu_min, mu_max)
        ],
        dtype=np.int64,
    )
    print(f"[cache]   mu in [{mu_min:g},{mu_max:g}] -> {len(sel)}/{n_read} events")
    if len(sel) == 0:
        raise ValueError("mu window selects no event")

    entry_stop = int(ev_idx.max()) + 1
    print(f"[cache] root: {root_path}  (reading entries 0..{entry_stop})")
    tree = uproot.open(root_path)["PVFinderData"]
    arrays = tree.arrays(
        list(TRUTH_BRANCHES + RECO_BRANCHES + EVENT_BRANCHES),
        entry_stop=entry_stop,
        library="np",
    )

    beam_z = np.asarray(arrays["BeamPosZ"], dtype=np.float64)
    n_beam_nonzero = int(np.count_nonzero(beam_z))
    if n_beam_nonzero:
        raise ValueError(
            f"{n_beam_nonzero} events have BeamPosZ != 0. Truth, AMVF and "
            "PV-Finder peaks are compared in the detector frame with no beam "
            "correction (as run_eval_pvf_run3.py does); that is only safe when "
            "the beam is at the origin. Add an explicit frame correction first."
        )

    pvf_list, pvf_h_list, amvf_list, t2_list, t1_list = [], [], [], [], []
    mu_out, idx_out = [], []
    for k in sel.tolist():
        e = int(ev_idx[k])
        t_ntrk = np.asarray(arrays["TruthVertex_nTracks"][e], dtype=np.float32)
        t_z = np.asarray(arrays["TruthVertex_z"][e], dtype=np.float32)
        a_ntrk = np.asarray(arrays["RecoVertex_nTracks"][e], dtype=np.float32)
        a_z = np.asarray(arrays["RecoVertex_z"][e], dtype=np.float32)
        mu_root = float(arrays["ActualNumOfInt"][e])

        t2 = t_z[t_ntrk >= 2]
        t1 = t_z[(t_ntrk >= 1) & (t_ntrk < 2)]
        a2 = a_z[a_ntrk >= 2]

        rec = per_event[k]
        if len(t2) != rec["n_truth"]:
            raise ValueError(
                f"event {e}: ROOT nTrk>=2 truth count {len(t2)} != pkl "
                f"n_truth {rec['n_truth']}. pkl and ROOT are not aligned."
            )
        if rec["n_amvf"] is not None and len(a2) != rec["n_amvf"]:
            raise ValueError(
                f"event {e}: ROOT AMVF count {len(a2)} != pkl n_amvf "
                f"{rec['n_amvf']}. pkl and ROOT are not aligned."
            )
        if mu_root != rec["mu"]:
            raise ValueError(f"event {e}: ROOT mu {mu_root} != pkl mu {rec['mu']}.")
        # The eval's own truth array must also agree, element for element.
        if not np.array_equal(np.sort(t2), np.sort(np.asarray(truth_pkl[k]))):
            raise ValueError(f"event {e}: truth positions differ between pkl and ROOT.")

        pk_z = np.asarray(pred[k], dtype=np.float32)
        pk_h = np.asarray(pred_h[k], dtype=np.float32)
        if len(pk_z) != len(pk_h):
            raise ValueError(
                f"event {e}: {len(pk_z)} peak positions but {len(pk_h)} heights."
            )
        pvf_list.append(pk_z)
        pvf_h_list.append(pk_h)
        amvf_list.append(a2)
        t2_list.append(t2)
        t1_list.append(t1)
        mu_out.append(mu_root)
        idx_out.append(e)

    n_evt = len(idx_out)
    pvf_v, pvf_o = _flatten(pvf_list)
    pvf_hv, _ = _flatten(pvf_h_list)
    amvf_v, amvf_o = _flatten(amvf_list)
    t2_v, t2_o = _flatten(t2_list)
    t1_v, t1_o = _flatten(t1_list)

    meta = {
        "tag": tag or root_path.stem,
        "pkl": str(pkl_path),
        "root": str(root_path),
        "mu_min": mu_min,
        "mu_max": mu_max,
        "n_events": n_evt,
        "mu_mean": float(np.mean(mu_out)),
        "pvf_per_evt": len(pvf_v) / n_evt,
        "amvf_per_evt": len(amvf_v) / n_evt,
        "truth_ge2_per_evt": len(t2_v) / n_evt,
        "truth_eq1_per_evt": len(t1_v) / n_evt,
        "sigma_pvf_eval_mm": float(res.get("sigma_vtx_vtx_mm", float("nan"))),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        event_idx=np.asarray(idx_out, dtype=np.int64),
        mu=np.asarray(mu_out, dtype=np.float32),
        pvf_z=pvf_v, pvf_off=pvf_o, pvf_h=pvf_hv,
        amvf_z=amvf_v, amvf_off=amvf_o,
        truth_ge2_z=t2_v, truth_ge2_off=t2_o,
        truth_eq1_z=t1_v, truth_eq1_off=t1_o,
        meta=np.array(json.dumps(meta)),
    )  # fmt: skip
    print(f"[cache]   alignment verified on all {n_evt} events")
    print(
        f"[cache]   per event: PVF {meta['pvf_per_evt']:.2f}  "
        f"AMVF {meta['amvf_per_evt']:.2f}  truth(nTrk>=2) "
        f"{meta['truth_ge2_per_evt']:.2f}  truth(nTrk==1) "
        f"{meta['truth_eq1_per_evt']:.2f}"
    )
    print(f"[cache]   wrote {out_path}")
    return meta


def load_cache(path: str | Path) -> dict[str, Any]:
    """Read a cache written by :func:`build_cache` back into ragged lists."""
    z = np.load(path, allow_pickle=False)
    return {
        "meta": json.loads(str(z["meta"])),
        "event_idx": z["event_idx"],
        "mu": z["mu"],
        "pvf": unflatten(z["pvf_z"], z["pvf_off"]),
        "pvf_h": unflatten(z["pvf_h"], z["pvf_off"]),
        "amvf": unflatten(z["amvf_z"], z["amvf_off"]),
        "truth_ge2": unflatten(z["truth_ge2_z"], z["truth_ge2_off"]),
        "truth_eq1": unflatten(z["truth_eq1_z"], z["truth_eq1_off"]),
    }
