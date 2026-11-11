"""Command-line interface of run_eval_pvf.py (split out for the 500-line limit)."""

from __future__ import annotations

import argparse

from pv_finder.utils.constants import MATCH_WINDOW_MM


def build_parser(
    _DEFAULT_H5: str,  # noqa: N803 - names kept from run_eval_pvf.py
    _DEFAULT_ROOT: str,  # noqa: N803
    _DEFAULT_QIBIN: str,  # noqa: N803
    INTEGRAL_THRESHOLD: float,  # noqa: N803
) -> argparse.ArgumentParser:
    """Build the run_eval_pvf.py argument parser (defaults from the caller)."""
    parser = argparse.ArgumentParser(
        description="PV-Finder MC evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--h5",
        default=_DEFAULT_H5,
        help="HDF5 file path",
    )
    parser.add_argument("--k2h-model", default=None, dest="k2h_model")
    parser.add_argument("--k2h-type", default="v1", choices=["v1", "v2"],
                        dest="k2h_type", help="K2H model class (v1=UNet_1000, v2=UNet_1000_v2)")  # fmt: skip
    parser.add_argument("--e2e-model", default=None, dest="e2e_model")
    parser.add_argument("--e2e-type", default="v1", choices=["v1", "v2"],
                        dest="e2e_type", help="E2E model class (v1=trackstoHists_UNet_1000, v2=TracksToHist_v2)")  # fmt: skip
    parser.add_argument("--root-truth", default=_DEFAULT_ROOT, dest="root_truth",
        help=f"ROOT truth for nTracks>=2 filter + mu + AMVF reco count. "
             f"Default: {_DEFAULT_ROOT}. Pass empty string to disable. "
             f"qibin map auto-loaded from {_DEFAULT_QIBIN}.")  # fmt: skip
    parser.add_argument("--indices", default="configs/test_main_indices_2550evt.p",
                        help="Test event indices pickle")  # fmt: skip
    parser.add_argument("--output-dir", default="outputs/eval_pvf", dest="output_dir")
    parser.add_argument("--device", type=int, default=0, help="CUDA device (-1=CPU)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-2,
        help="Peak-finder threshold (default 1e-2)",
    )
    parser.add_argument(
        "--gbt-filter-model",
        default=None,
        help="Path to peak_classifier_results.pkl to apply GBT hist-only filter",
    )
    parser.add_argument(
        "--gbt-threshold",
        type=float,
        default=0.7,
        help="Threshold for GBT filter (default 0.7)",
    )
    parser.add_argument("--match-window-mm", type=float, default=MATCH_WINDOW_MM,
                        dest="match_window_mm",
                        help="Fixed truth-matching window in mm; <= 0 reuses the "
                             "fitted sigma_vtx-vtx (legacy)")  # fmt: skip
    parser.add_argument("--integral-threshold", type=float, default=INTEGRAL_THRESHOLD,
                        dest="integral_threshold",
                        help=f"Peak-finder integral threshold (default {INTEGRAL_THRESHOLD})")  # fmt: skip
    parser.add_argument("--title", default="", help="Plot title (used for all plots)")
    parser.add_argument("--dataset-name", default="", dest="dataset_name")
    return parser
