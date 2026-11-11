"""Build the aligned per-event caches for the Run 2 vs HL-LHC comparison.

Run 2 MC
    PV-Finder peaks from the 2026-04-16 reproduction eval pkl (integral 0.5,
    amplitude 0.01, no height floor, full-region weighted mean).  The pkl's
    per-event order is the h5 test order; the ROOT entry of each event is given
    by ``configs/qibin_test_main_indices_v2.p``, which the eval itself used.
    The mapping is *proved* here against the alternative identity map, using
    four independent per-event observables.
HL-LHC
    Two evals of the same v6 checkpoint on the same 25 000 r16443 entries:
    ``eval_v6_mu_window`` (local centroid, height floor 0.03, counting
    integral 0.30) and ``eval_v6_heldout`` (full-region mean, no floor,
    counting integral 0.20).  The integral-0.5 lists of both are recovered from
    the stored pairwise arrays (see ``recover``).  pkl event_idx == ROOT entry,
    verified event by event.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from pv_finder.diagnostics.amvf_fair_comparison.cache import _flatten, unflatten
from pv_finder.diagnostics.run2_vs_run4.recover import (
    check_window_subarray,
    recover_subsets,
)

TREE = "PVFinderData"
VTX_BRANCHES = (
    "TruthVertex_z", "TruthVertex_nTracks", "TruthVertex_type",
    "RecoVertex_z", "RecoVertex_nTracks", "RecoVertex_ErrZ",
    "ActualNumOfInt", "BeamPosZ", "BeamPosSigmaZ",
)  # fmt: skip
TRK_BRANCHES = ("RecoTrack_ErrZ0", "RecoTrack_pT", "RecoTrack_eta")
RAGGED = ("truth_z", "truth_ntrk", "truth_type", "amvf_z", "amvf_ntrk", "amvf_errz")


def _read_root(path: str | Path, entries: np.ndarray) -> dict[str, Any]:
    """Read the vertex branches for the given entries (sorted range read)."""
    import uproot

    tree = uproot.open(path)[TREE]
    stop = int(entries.max()) + 1
    arr = tree.arrays(list(VTX_BRANCHES), entry_stop=stop, library="np")
    if np.any(np.asarray(arr["BeamPosZ"])[entries] != 0):
        raise ValueError("BeamPosZ != 0 in the selection; a frame correction is needed")
    out: dict[str, Any] = {
        "mu": np.asarray(arr["ActualNumOfInt"], dtype=np.float64)[entries],
        "beam_sigma_z": np.asarray(arr["BeamPosSigmaZ"], dtype=np.float64)[entries],
    }
    keymap = {
        "truth_z": "TruthVertex_z", "truth_ntrk": "TruthVertex_nTracks",
        "truth_type": "TruthVertex_type", "amvf_z": "RecoVertex_z",
        "amvf_ntrk": "RecoVertex_nTracks", "amvf_errz": "RecoVertex_ErrZ",
    }  # fmt: skip
    for k, b in keymap.items():
        out[k] = [np.asarray(arr[b][e], dtype=np.float32) for e in entries.tolist()]
    return out


def prove_run2_alignment(
    per_event: list[dict[str, Any]],
    truth_pkl: list[np.ndarray],
    root_path: str | Path,
    qibin: list[int],
) -> dict[str, dict[str, int]]:
    """Score two candidate h5 -> ROOT maps on four per-event observables.

    Observables: nTrk >= 2 truth count (pkl ``n_truth``), AMVF nTrk >= 2 count
    (``n_amvf``), ``mu``, and the full truth position list (the pkl stores
    every truth vertex, which in this ntuple all have nTracks >= 1).
    """
    import uproot

    tree = uproot.open(root_path)[TREE]
    a = tree.arrays(
        [
            "TruthVertex_z",
            "TruthVertex_nTracks",
            "RecoVertex_nTracks",
            "ActualNumOfInt",
        ],
        library="np",
    )
    maps = {
        "identity (ROOT entry = h5 event index)": [r["event_idx"] for r in per_event],
        "qibin map (configs/qibin_test_main_indices_v2.p)": qibin,
    }
    res: dict[str, dict[str, int]] = {}
    for name, mp in maps.items():
        c = {"n_truth": 0, "n_amvf": 0, "mu": 0, "truth_positions": 0}
        for i, rec in enumerate(per_event):
            e = int(mp[i])
            z, n = a["TruthVertex_z"][e], a["TruthVertex_nTracks"][e]
            c["n_truth"] += int((n >= 2).sum()) == rec["n_truth"]
            c["n_amvf"] += int((a["RecoVertex_nTracks"][e] >= 2).sum()) == rec["n_amvf"]
            c["mu"] += float(a["ActualNumOfInt"][e]) == rec["mu"]
            tp = np.sort(np.asarray(truth_pkl[i], dtype=np.float32))
            zz = np.sort(np.asarray(z, dtype=np.float32))
            c["truth_positions"] += bool(len(tp) == len(zz) and np.array_equal(tp, zz))
        c["n_events"] = len(per_event)
        res[name] = c
    return res


def build_run2(pkl: str, root: str, qibin_path: str, out: Path) -> dict[str, Any]:
    """Run 2 MC cache: PV-Finder (2550 events), AMVF and truth on the same entries."""
    with open(pkl, "rb") as fh:
        r = pickle.load(fh)
    with open(qibin_path, "rb") as fh:
        qibin = [int(x) for x in pickle.load(fh)]
    per_event = r["per_event"]
    if not (len(per_event) == len(r["pred_pvs_mm"]) == len(qibin)):
        raise ValueError("pkl per_event / pred / qibin lengths disagree")
    proof = prove_run2_alignment(per_event, r["truth_pvs_mm"], root, qibin)
    qi = proof["qibin map (configs/qibin_test_main_indices_v2.p)"]
    if min(qi[k] for k in ("n_truth", "n_amvf", "mu", "truth_positions")) != len(
        per_event
    ):
        raise ValueError(f"qibin alignment fails: {qi}")
    entries = np.asarray(qibin, dtype=np.int64)
    d = _read_root(root, entries)
    d["pvf_int050_h000_full"] = [
        np.asarray(p, dtype=np.float32) for p in r["pred_pvs_mm"]
    ]
    # The pkl's own pairwise array must be exactly these lists' pairs.
    n_pairs = sum(len(p) * (len(p) - 1) // 2 for p in d["pvf_int050_h000_full"])
    if n_pairs != len(r["pairwise_dz_mm"]):
        raise ValueError("Run 2 pkl pred lists do not reproduce pairwise_dz_mm")
    meta = {
        "sample": "run2", "pkl": pkl, "root": root, "qibin": qibin_path,
        "alignment_proof": proof, "sigma_pkl_mm": float(r["sigma_vtx_vtx_mm"]),
        "fit_params_pkl": list(map(float, r["fit_params"])),
        "n_pairs_pkl": int(len(r["pairwise_dz_mm"])),
    }  # fmt: skip
    d["event_idx"] = np.array([x["event_idx"] for x in per_event], dtype=np.int64)
    d["root_entry"] = entries
    # AMVF on the ROOT entries 0..2499 used for the published 0.793 mm.
    d_pub = _read_root(root, np.arange(2500, dtype=np.int64))
    d["pub_amvf_z"], d["pub_amvf_ntrk"] = d_pub["amvf_z"], d_pub["amvf_ntrk"]
    meta["pub_entries_overlap_with_pvf"] = int(np.isin(entries, np.arange(2500)).sum())
    _save(out, d, meta, ("pvf_int050_h000_full", "pub_amvf_z", "pub_amvf_ntrk"))
    return meta


def _check_hl_alignment(r: dict[str, Any], d: dict[str, Any]) -> int:
    """pkl event_idx == ROOT entry: counts, mu and truth lists must agree."""
    bad = 0
    for k, rec in enumerate(r["per_event"]):
        t2 = d["truth_z"][k][d["truth_ntrk"][k] >= 2]
        a2 = d["amvf_ntrk"][k] >= 2
        ok = (
            rec["event_idx"] == k
            and len(t2) == rec["n_truth"]
            and int(a2.sum()) == rec["n_amvf"]
            and float(d["mu"][k]) == float(rec["mu"])
            and np.array_equal(np.sort(t2), np.sort(np.asarray(r["truth_pvs_mm"][k])))
        )
        bad += not ok
    return bad


def _heights_of(subset: np.ndarray | None, pk: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Look up each recovered position's stored peak height (exact position match)."""
    if subset is None:
        return np.zeros(0, dtype=np.float32)
    idx = np.searchsorted(np.sort(pk), subset)
    order = np.argsort(pk)
    return np.asarray(h, dtype=np.float32)[order][idx]


def build_hl(
    pkl_centroid: str, pkl_fullmean: str, root: str, n_events: int, out: Path
) -> dict[str, Any]:
    """HL-LHC cache: 25 000 flat-mu events, four PV-Finder configurations."""
    entries = np.arange(n_events, dtype=np.int64)
    d = _read_root(root, entries)
    meta: dict[str, Any] = {"sample": "hl", "root": root, "recovery": {}}
    configs = (
        (pkl_centroid, "int030_h003_cen", "int050_h003_cen"),
        (pkl_fullmean, "int020_h000_full", "int050_h000_full"),
    )
    for pkl, lo_name, hi_name in configs:
        print(f"[hl] loading {pkl}", flush=True)
        with open(pkl, "rb") as fh:
            r = pickle.load(fh)
        if len(r["per_event"]) != n_events:
            raise ValueError(
                f"{pkl}: {len(r['per_event'])} events, expected {n_events}"
            )
        bad = _check_hl_alignment(r, d)
        if bad:
            raise ValueError(f"{pkl}: {bad} events disagree with ROOT")
        pk = [np.asarray(p, dtype=np.float32) for p in r["pred_pvs_mm"]]
        hs = [np.asarray(h, dtype=np.float32) for h in r["pred_heights"]]
        dz_all = r.get("pairwise_dz_mm_all_events")
        if dz_all is None:
            dz_all = r["pairwise_dz_mm"]  # pre-2026-08-05: flat over every event
        subsets, st = recover_subsets(pk, dz_all)
        if "in_mu_window" in r and r.get("pairwise_dz_mm_all_events") is not None:
            st["mu_window_subarray_ok"] = check_window_subarray(
                subsets, np.asarray(r["in_mu_window"]), r["pairwise_dz_mm"]
            )
            if not st["mu_window_subarray_ok"]:
                raise ValueError(f"{pkl}: mu-window sub-array check failed")
        st["pkl"] = pkl
        st["sigma_pkl_mm"] = float(r["sigma_vtx_vtx_mm"])
        meta["recovery"][hi_name] = st
        d[f"pvf_{lo_name}"], d[f"h_{lo_name}"] = pk, hs
        d[f"pvf_{hi_name}"] = [
            s if s is not None else np.zeros(0, np.float32) for s in subsets
        ]
        d[f"h_{hi_name}"] = [_heights_of(s, p, h) for s, p, h in zip(subsets, pk, hs)]
        d[f"ok_{hi_name}"] = np.array([s is not None for s in subsets])
        del r
    names = tuple(k for k in d if k.startswith(("pvf_", "h_")))
    _save(out, d, meta, names)
    return meta


def read_tracks(root: str | Path, n_entries: int) -> dict[str, np.ndarray]:
    """Flat track arrays (z0 error, pT in GeV, eta) and per-track event mu.

    Reads the first ``n_entries`` entries: track properties do not depend on
    which events PV-Finder was evaluated on, only on the sample.
    """
    import uproot

    tree = uproot.open(root)[TREE]
    a = tree.arrays(
        list(TRK_BRANCHES) + ["ActualNumOfInt"], entry_stop=n_entries, library="np"
    )
    ntrk = np.array([len(x) for x in a["RecoTrack_eta"]])
    return {
        "errz0": np.concatenate(a["RecoTrack_ErrZ0"]).astype(np.float32),
        "pt": np.concatenate(a["RecoTrack_pT"]).astype(np.float32) / 1000.0,
        "eta": np.concatenate(a["RecoTrack_eta"]).astype(np.float32),
        "mu": np.repeat(np.asarray(a["ActualNumOfInt"], dtype=np.float32), ntrk),
    }


def _save(out: Path, d: dict[str, Any], meta: dict[str, Any], extra: tuple) -> None:
    """Flatten every ragged list into (values, offsets) and write one npz."""
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"meta": np.array(json.dumps(meta))}
    for k, v in d.items():
        if k in RAGGED or k in extra:
            vals, offs = _flatten(list(v))
            payload[f"{k}__v"], payload[f"{k}__o"] = vals, offs
        else:
            payload[k] = np.asarray(v)
    np.savez_compressed(out, **payload)
    print(f"[cache] wrote {out}", flush=True)


def load(path: str | Path) -> dict[str, Any]:
    """Inverse of ``_save``: ragged lists back, plus ``meta``."""
    z = np.load(path, allow_pickle=False)
    d: dict[str, Any] = {"meta": json.loads(str(z["meta"]))}
    for k in z.files:
        if k.endswith("__v"):
            b = k[:-3]
            d[b] = unflatten(z[k], z[b + "__o"])
        elif not k.endswith("__o") and k != "meta":
            d[k] = z[k]
    return d
