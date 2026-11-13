#!/usr/bin/env python3
"""GNN score-cut rescan under the >= 2-track vertex rule (validation split).

t = 0.97 was chosen with drop-empty (>= 1 track) vertices. This replays the
``chain_t_metrics`` record made with ``--hs-min-tracks 2`` on the GNN
validation entries (s8 peaks, GNN v2 e527) and tabulates, for each t, the
metrics with vertices that have >= 2 assigned tracks (AMVF: >= 2 associated
tracks): Clean/truth, positional efficiency (truth within 0.17 mm of a
surviving vertex), Clean/reco, Split/reco, track efficiency / purity / F1,
HS selection (Athena rule) and HS resolution, in the inclusive, PU200 and
mu < 20 bins. Ratios of sums, 200-replica event bootstrap.

    PYTHONPATH=src venv/bin/python -u scripts/hllhc_t_rescan_min2.py \
        --metrics outputs/11_12_2026_output/hllhc_final/t_rescan_min2/val_s8_v2e527_hsmin2.npz \
        --out outputs/11_12_2026_output/hllhc_final/t_rescan_min2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from hllhc_final_chain import sha256, subset  # noqa: E402

from gnn.evaluation.chain_t_summary import load, summarize  # noqa: E402
from pv_finder.utils.working_points import get_regime  # noqa: E402

T_SCAN = (0.0, 0.5, 0.8, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98)
BINS = {"inclusive": (0, 1e9), "PU200 (185-215)": (185, 215.5), "mu < 20": (0, 20)}
COLS = (
    ("clean_per_truth", "Clean/truth"), ("pos_eff", "pos. eff"),
    ("clean_per_reco", "Clean/reco"), ("split_per_reco", "Split/reco"),
    ("trk_eff", "trk eff"), ("trk_pur", "trk pur"),
)  # fmt: skip


def table(s: dict) -> str:
    """Markdown, one block per bin, one row per t and AMVF."""
    out = []
    n_t = len(s["arms"]) - 1
    for lab, b in s["bins"].items():
        r = b["ratios_min2"]
        out.append(f"### {lab}: {b['n_events']} events, mean mu {b['mean_mu']:.1f}\n")
        head = " | ".join(c for _, c in COLS)
        out.append(f"| arm | {head} | trk F1 | HS sel (rule) | HS res [um] (AMVF) |")
        out.append("|---|" + "---:|" * (len(COLS) + 3))
        for a, arm in enumerate(s["arms"]):
            cells = [f"{100 * r[k]['value'][a]:.2f} ± {100 * r[k]['err'][a]:.2f}" for k, _ in COLS]  # fmt: skip
            e, p = r["trk_eff"]["value"][a], r["trk_pur"]["value"][a]
            sel = b["hs_selection"]
            si = a if a < n_t else n_t  # AMVF row shows the stored primary
            hs = f"{100 * sel['value'][si]:.2f} ± {100 * sel['err'][si]:.2f}"
            if a == n_t:
                hs += f" (rule {100 * sel['value'][n_t + 1]:.2f})"
            hr = b["hs_resolution"]
            res = "" if hr["rms_um"][a] is None else f"{hr['rms_um'][a]:.2f} ± {hr['err_um'][a]:.2f} ({hr['amvf_um'][a]:.2f})"  # fmt: skip
            out.append(f"| {arm} | " + " | ".join(cells) + f" | {100 * 2 * e * p / (e + p):.2f} | {hs} | {res} |")  # fmt: skip
        out.append("")
    return "\n".join(out)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rec = load(a.metrics)  # asserts the bookkeeping identities
    meta = rec.meta
    if meta.get("hs_min_tracks") != 2:
        raise RuntimeError("expected a --hs-min-tracks 2 record")
    regime = get_regime("hllhc")
    dens = meta["infer_metadata"]["density"]
    if dens["peaks"] != regime.to_dict()["density"]["peaks"]:
        raise RuntimeError("inference was not made with the frozen s8 cuts")
    s = summarize(subset(rec, T_SCAN), bins=BINS)
    s["provenance"] = {
        "metrics": str(a.metrics), "metrics_sha256": sha256(a.metrics),
        "entries": meta["infer_metadata"]["entries_spec"],
        "entries_sha256": meta["infer_metadata"]["entries_sha256"],
        "metrics_git": meta["git"], "density_sha256": dens["sha256"],
        "gnn_sha256": meta["infer_metadata"]["gnn"][meta["gnn"]]["sha256"],
        "regime": regime.to_dict(), "vertex_min_tracks": 2,
    }  # fmt: skip
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "t_rescan_min2.json").write_text(json.dumps(s, indent=1) + "\n")
    md = "# t rescan, >= 2-track vertices, GNN validation split (s8, v2 e527)\n\n" + table(s)  # fmt: skip
    (a.out / "t_rescan_min2.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
