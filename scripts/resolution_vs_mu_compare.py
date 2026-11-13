"""Paired comparison of two resolution-versus-mu studies on identical events.

Typical use: the frozen density model (s8) against the v6 reference.

  python -u scripts/resolution_vs_mu_compare.py --new NEW_DIR --ref REF_DIR

Each study's peaks are found again from its own histograms at the cuts its
results.json records.  The script asserts that the two studies hold the same
events in the same order (selection file, event ids, mu, truth and AMVF vertex
lists), that the AMVF rows of the two results.json are identical (AMVF does not
depend on the density model), and that the recomputed PV-Finder numbers
reproduce each results.json.  It then writes, per mu bin, new minus ref in
sigma_vtx-vtx and in the 68% residual with paired event-bootstrap errors:
comparison_<tag>.json and comparison_<tag>.md under --new.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from resolution_vs_mu import load_study, sha256  # noqa: E402

from pv_finder.diagnostics.resolution_vs_mu import (  # noqa: E402
    mu_bin_index,
    paired_bin_diff,
)
from pv_finder.diagnostics.run2_vs_run4.metrics import (  # noqa: E402
    event_hists,
    per_event_residuals,
)

# AMVF fields that must agree exactly between the two studies.
AMVF_KEYS = ("sigma_mm", "boot_err_mm", "fit_err_mm", "merge_len_mm", "n_pairs",
             "n_events", "p68_um", "p68_err_um", "n_matched", "chi2", "popt")  # fmt: skip


def _selection_digest(d: Path) -> str:
    return sha256(d / "selection.json")


def check_same_events(new: dict, ref: dict) -> None:
    """Raise unless both studies hold identical events, truth and AMVF vertices."""
    if new["id"] != ref["id"]:
        raise AssertionError("event ids (or their order) differ between the studies")
    if not np.array_equal(new["mu"], ref["mu"]):
        raise AssertionError("mu differs between the studies")
    for key in ("truth", "amvf"):
        bad = [i for i, (x, y) in enumerate(zip(new[key], ref[key]))
               if not np.array_equal(x, y)]  # fmt: skip
        if bad:
            raise AssertionError(f"{key} vertices differ in {len(bad)} events")


def same(x: Any, y: Any) -> bool:
    """Exact equality of numbers or lists of numbers, with nan equal to nan."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return xa.shape == ya.shape and bool(
        np.all((xa == ya) | (np.isnan(xa) & np.isnan(ya)))
    )


def check_amvf_identical(new_rows: list[dict], ref_rows: list[dict]) -> int:
    """Raise unless every AMVF number of every bin is identical; return #values."""
    if len(new_rows) != len(ref_rows):
        raise AssertionError("the studies have different numbers of mu bins")
    n = 0
    for a, b in zip(new_rows, ref_rows):
        if (a["mu_lo"], a["mu_hi"], a["n_events"]) != (
            b["mu_lo"],
            b["mu_hi"],
            b["n_events"],
        ):
            raise AssertionError(f"bin layout differs at mu {a['mu_lo']}")
        for k in AMVF_KEYS:
            if not same(a["amvf"][k], b["amvf"][k]):
                raise AssertionError(
                    f"AMVF {k} differs at mu {a['mu_lo']}: {a['amvf'][k]} vs {b['amvf'][k]}"
                )
            n += 1
    return n


def _check_reproduces(rows: list[dict], hz: np.ndarray, res: list, k: np.ndarray,
                      edges: np.ndarray, name: str) -> None:  # fmt: skip
    """The recomputed PV-Finder sigma and p68 must equal the study's results.json."""
    from pv_finder.diagnostics.run2_vs_run4.metrics import fit_counts

    for b, r in enumerate(rows):
        sel = np.flatnonzero(k == b)
        s = fit_counts(edges, hz[sel].sum(0))[0]
        p = 1e3 * float(np.percentile(np.concatenate([res[i] for i in sel]), 68.0))
        if not (same(s, r["pvf"]["sigma_mm"]) and same(p, r["pvf"]["p68_um"])):
            raise AssertionError(
                f"{name}: recomputed bin {b} ({s}, {p}) != results.json"
            )


def compare(a: argparse.Namespace) -> dict[str, Any]:
    """Run all checks and the paired per-bin differences."""
    res_new = json.loads((a.new / "results.json").read_text())
    res_ref = json.loads((a.ref / "results.json").read_text())
    if _selection_digest(a.new) != _selection_digest(a.ref):
        raise AssertionError("selection.json differs between the studies")
    sel = json.loads((a.new / "selection.json").read_text())
    edges = np.asarray(sel["mu_edges"])
    n_amvf = check_amvf_identical(res_new["bins"], res_ref["bins"])
    print(
        f"AMVF identical: {n_amvf} values over {len(res_new['bins'])} bins", flush=True
    )
    new = load_study(a.new, res_new["peaks"], a.workers)
    ref = load_study(a.ref, res_ref["peaks"], a.workers)
    check_same_events(new, ref)
    print(f"same events: {len(new['id'])} ids, mu, truth and AMVF lists", flush=True)
    k = mu_bin_index(new["mu"], edges)
    h_new, hz = event_hists(new["pvf"])
    h_ref, _ = event_hists(ref["pvf"])
    r_new = per_event_residuals(new["truth"], new["pvf"])
    r_ref = per_event_residuals(ref["truth"], ref["pvf"])
    _check_reproduces(res_new["bins"], h_new, r_new, k, hz, "new")
    _check_reproduces(res_ref["bins"], h_ref, r_ref, k, hz, "ref")
    print("recomputed PV-Finder sigma and p68 reproduce both results.json", flush=True)
    # One process per bin, each given only its own events.  Positional
    # resampling of arange(n) on the bin's slice draws the same events as
    # resampling the bin's global indices, so the result does not depend on this.
    jobs = []
    for b in range(len(res_new["bins"])):
        s = np.flatnonzero(k == b)
        jobs.append((h_new[s], h_ref[s], hz, [r_new[i] for i in s],
                     [r_ref[i] for i in s], np.arange(len(s))))  # fmt: skip
    with Pool(a.workers) as pool:
        diffs = pool.starmap(partial(paired_bin_diff, n_boot=a.n_boot), jobs)
    bins = []
    for b, (rn, rr, d) in enumerate(zip(res_new["bins"], res_ref["bins"], diffs)):
        d.update(mu_lo=rn["mu_lo"], mu_hi=rn["mu_hi"],
                 sigma_new_um=1e3 * rn["pvf"]["sigma_mm"],
                 sigma_ref_um=1e3 * rr["pvf"]["sigma_mm"],
                 p68_new_um=rn["pvf"]["p68_um"], p68_ref_um=rr["pvf"]["p68_um"],
                 n_pvf_new=sum(len(new["pvf"][i]) for i in np.flatnonzero(k == b)),
                 n_pvf_ref=sum(len(ref["pvf"][i]) for i in np.flatnonzero(k == b)))  # fmt: skip
        bins.append(d)
    return {"new": str(a.new), "ref": str(a.ref),
            "new_model_sha256": res_new.get("model_sha256",
                                            res_new["provenance"].get("model_sha256")),
            "ref_model_sha256": res_ref["provenance"].get("model_sha256"),
            "new_peaks": res_new["peaks"], "ref_peaks": res_ref["peaks"],
            "selection_sha256": _selection_digest(a.new), "n_events": len(new["id"]),
            "amvf_identical_values": n_amvf, "n_boot": a.n_boot, "bins": bins}  # fmt: skip


def table(out: dict[str, Any], tag: str) -> str:
    """Markdown table of the paired differences (new minus ref, um)."""
    lines = [
        f"| mu | events | sigma {tag} | sigma new | d sigma | p68 {tag} | p68 new "
        f"| d p68 | PVF vtx/evt {tag} / new |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in out["bins"]:
        n = d["n_events"]
        lines.append(
            f"| {d['mu_lo']:.0f}-{d['mu_hi']:.0f} | {n} | {d['sigma_ref_um']:.1f} "
            f"| {d['sigma_new_um']:.1f} | {d['d_sigma_um']:+.1f} ± {d['d_sigma_err_um']:.1f} "
            f"| {d['p68_ref_um']:.2f} | {d['p68_new_um']:.2f} "
            f"| {d['d_p68_um']:+.2f} ± {d['d_p68_err_um']:.2f} "
            f"| {d['n_pvf_ref'] / n:.2f} / {d['n_pvf_new'] / n:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--new", type=Path, required=True, help="study under test")
    p.add_argument("--ref", type=Path, required=True, help="reference study")
    p.add_argument("--tag", default="v6", help="short name of the reference")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--workers", type=int, default=8, help="peak-finding processes")
    a = p.parse_args()
    out = compare(a)
    (a.new / f"comparison_{a.tag}.json").write_text(
        json.dumps(out, indent=1, default=float)
    )
    md = table(out, a.tag)
    (a.new / f"comparison_{a.tag}.md").write_text(md + "\n")
    print(md, flush=True)
    print("ref model sha256", out["ref_model_sha256"], "new", out["new_model_sha256"])


if __name__ == "__main__":
    main()
