"""The working points are defined once (pv_finder.utils.working_points).

Checks the decided values, that the sigma_vtx-vtx fit uses the counting
threshold unless explicitly overridden, that overrides are recorded, and that
no tracked file under src/ or scripts/ re-introduces a hard-coded matching
window outside a documented allowlist.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src/pv_finder/evaluation/vertex_finding"))

from pv_finder.evaluation.vertex_finding.regime_cli import (  # noqa: E402
    PEAK_KEYS,
    add_regime_args,
    resolve_regime_args,
)
from pv_finder.utils import constants  # noqa: E402
from pv_finder.utils import working_points as wp  # noqa: E402
from pv_finder.utils.pairwise_dz import DEFAULT_PAIRWISE_BINS  # noqa: E402


def test_decided_values() -> None:
    assert wp.MATCH_WINDOW_MM == 0.17
    assert wp.HS_MATCH_WINDOW_MM == 0.1
    assert constants.MATCH_WINDOW_MM == wp.MATCH_WINDOW_MM
    assert constants.HS_MATCH_WINDOW_MM == wp.HS_MATCH_WINDOW_MM
    assert wp.PAIRWISE_BINS == DEFAULT_PAIRWISE_BINS
    for regime in wp.REGIMES.values():
        assert regime.match_window_mm == 0.17
        assert regime.hs_match_window_mm == 0.1
        assert regime.peaks.centroid_halfwidth == 3  # one estimator everywhere
    h, low = wp.HLLHC.peaks, wp.LOWPU.peaks
    assert (h.amplitude, h.min_width, h.integral, h.min_height) == (
        0.01,
        3,
        0.184,
        0.0184,
    )
    assert wp.HLLHC_PEAKS.integral == 0.30  # v6 reference cuts
    assert (low.amplitude, low.min_width, low.integral, low.min_height) == (
        0.01,
        3,
        0.50,
        0.0,
    )
    assert wp.LOWPU.density.name == "e400"
    assert wp.window_tag(0.17) == "0p17"


def test_unfrozen_regime_raises() -> None:
    with pytest.raises(RuntimeError, match="not frozen"):
        wp.LOWPU.require_frozen()
    with pytest.raises(ValueError):
        wp.get_regime("run4")


def test_regime_dict_is_plain() -> None:
    d = wp.HLLHC.to_dict()
    assert d["density"]["peaks"]["integral"] == 0.184
    assert d["match_window_mm"] == 0.17


def _parse(argv: list[str], keys: tuple[str, ...] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--e2e-model", default=None, dest="e2e_model")
    add_regime_args(p, keys)
    args = p.parse_args(argv)
    resolve_regime_args(args)
    return args


def test_sigma_fit_uses_counting_threshold_by_default() -> None:
    a = _parse(["--regime", "hllhc"])
    assert a.integral_threshold == 0.184
    assert a.integral_threshold_res == a.integral_threshold
    assert a.regime_overrides == {}
    assert a.density_from_regime and a.e2e_model == wp.HLLHC.density.checkpoint
    b = _parse(["--regime", "hllhc", "--integral-threshold", "0.4"])
    assert b.integral_threshold_res == 0.4
    assert b.regime_overrides == {"integral_threshold": 0.4}


def test_explicit_overrides_are_recorded() -> None:
    a = _parse(
        ["--regime", "hllhc", "--integral-threshold-res", "0.5", "--e2e-model", "m.pth"]
    )
    assert a.integral_threshold == 0.184 and a.integral_threshold_res == 0.5
    assert a.regime_overrides["integral_threshold_res"] == 0.5
    assert a.regime_overrides["density_model"] == "m.pth"
    assert not a.density_from_regime
    same = _parse(["--regime", "hllhc", "--e2e-model", wp.HLLHC.density.checkpoint])
    assert same.density_from_regime and "density_model" not in same.regime_overrides
    low = _parse(["--regime", "lowpu", "--match-window-mm", "0.17"])
    assert low.regime_overrides == {}  # equal to the regime: not an override
    assert low.integral_threshold == 0.50 and low.centroid_halfwidth == 3


def test_peak_subset() -> None:
    a = _parse(["--regime", "lowpu"], PEAK_KEYS)
    assert not hasattr(a, "match_window_mm")
    assert (a.peak_threshold, a.integral_threshold) == (0.01, 0.50)


def test_eval_clis_require_and_use_regime(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_eval_pvf_cli
    import run_eval_pvf_run3

    monkeypatch.setattr(sys, "argv", ["x", "--root", "f.root"])
    with pytest.raises(SystemExit):  # --regime is required
        run_eval_pvf_run3._cli()
    monkeypatch.setattr(sys, "argv", ["x", "--root", "f.root", "--regime", "lowpu"])
    a = run_eval_pvf_run3._cli()
    resolve_regime_args(a)
    assert (a.e2e_type, a.e2e_unet_channels, a.e2e_hidden) == ("v1", 64, [100] * 5)
    assert a.integral_threshold_res == 0.50 and a.pairwise_bins == 300
    p = run_eval_pvf_cli.build_parser("h5", "", "q")
    b = p.parse_args(["--regime", "lowpu"])
    resolve_regime_args(b)
    assert (b.peak_threshold, b.integral_threshold, b.min_height) == (0.01, 0.5, 0.0)
    assert b.centroid_halfwidth == 3 and b.match_window_mm == 0.17


# --- Repository scan --------------------------------------------------------
# A matching window must come from working_points. These patterns catch a
# hard-coded window value; the allowlist names files whose literals are a
# documented different use. Each entry gives the reason.
WINDOW_PATTERNS = [
    re.compile(r"0\.1694"),
    re.compile(r"\bwindow[_a-z]*\s*(?::\s*float\s*)?=\s*[0-9]"),
    re.compile(r"\b[A-Z_]*WINDOW[_A-Z]*\s*=\s*[0-9]"),
]
ALLOWLIST: dict[str, str] = {
    "src/pv_finder/utils/working_points.py": "the definitions",
    "src/pv_finder/utils/constants.py": "comment on the 0.1694 -> 0.17 rounding",
    "src/pv_finder/diagnostics/amvf_fair_comparison/scan.py": "comment: derivation",
    "scripts/fixed_window_replay.py": "docstring: history of the window",
    "src/gnn/diagnostics/chain_gap_decomposition.py": "augmentation calibration "
    "window (GNN training recipe, not an evaluation window)",
    "src/gnn/diagnostics/hs_chain_study.py": "0.5 mm arm reproduces the legacy "
    "note Fig. 7.2 for comparison only",
    "scripts/target_readout.py": "documented pre-freeze study (11 Nov 2026)",
    "scripts/eb_numbers.py": "EB deck source keys; repointed in the recompute",
    "src/pv_finder/diagnostics/amvf_fairness_audit.py": "printed text only",
    "src/pv_finder/diagnostics/metric_worked_example.py": "worked example",
    "src/pv_finder/diagnostics/satellite_mechanism.py": "mechanism study",
    "src/pv_finder/diagnostics/amvf_resolution_vs_ntracks.py": "target-width study",
    "src/pv_finder/diagnostics/visualize_fakes.py": "visualisation",
    "src/pv_finder/evaluation/vertex_finding/nms_diagnostic.py": "visualisation",
    "src/pv_finder/diagnostics/domain_shift_investigation/kde_study/"
    "kde_comparison_plots.py": "visualisation",
    "src/pv_finder/diagnostics/per_vertex_visualization/run_per_vertex.py": "plots",
    "src/pv_finder/diagnostics/per_vertex_visualization/run_per_vertex_run4.py": "plots",
    "src/pv_finder/diagnostics/per_vertex_visualization/vertex_plots.py": "plots",
    "src/pv_finder/diagnostics/per_vertex_visualization/peak_matching.py": "plots",
}


def _tracked_python() -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "src", "scripts"], cwd=REPO, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    return [f for f in out.split() if f.endswith(".py")]


def test_no_new_hard_coded_window() -> None:
    offenders = []
    for rel in _tracked_python():
        if rel in ALLOWLIST:
            continue
        for n, line in enumerate((REPO / rel).read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if any(p.search(line) for p in WINDOW_PATTERNS):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert not offenders, "hard-coded window; use working_points:\n" + "\n".join(
        offenders
    )


def test_allowlist_is_current() -> None:
    tracked = set(_tracked_python())
    stale = [f for f in ALLOWLIST if f not in tracked]
    assert not stale, f"allowlisted files no longer tracked: {stale}"


def test_hllhc_frozen_12nov2026() -> None:
    """Freeze decided by Mattia on 12 Nov 2026: s8 density, GNN v2 e527, t = 0.97."""
    wp.HLLHC.require_frozen()
    assert wp.HLLHC.density.name == "s8" and wp.HLLHC.density.frozen
    assert wp.HLLHC.peaks.integral == 0.184 and wp.HLLHC.peaks.min_height == 0.0184
    assert wp.HLLHC.gnn.t == 0.97 and wp.HLLHC.gnn.frozen
    assert wp.HLLHC.gnn.checkpoint.endswith("_epoch_527.pyt")
