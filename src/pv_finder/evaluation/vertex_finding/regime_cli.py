"""Shared ``--regime`` handling for the evaluation scripts.

Every working-point argument defaults to ``None`` and is filled from
``pv_finder.utils.working_points`` after parsing. A value given explicitly on
the command line is kept and recorded as an override, so the result metadata
always says which values came from the regime and which did not.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pv_finder.utils.working_points import (
    REGIMES,
    Regime,
    get_regime,
    record_overrides,
    regime_defaults,
)

# CLI dest -> (flag, type, help). Defaults come from the regime.
REGIME_ARGS: dict[str, tuple[str, type, str]] = {
    "match_window_mm": ("--match-window-mm", float,
        "fixed truth-matching window (mm), same for every algorithm; "
        "<= 0 reuses the fitted sigma_vtx-vtx (legacy diagnostic only)"),
    "peak_threshold": ("--peak-threshold", float, "peak-finder amplitude threshold"),
    "integral_threshold": ("--integral-threshold", float,
        "minimum region integral"),
    "integral_threshold_res": ("--integral-threshold-res", float,
        "integral threshold of the peaks the sigma_vtx-vtx fit uses; defaults "
        "to --integral-threshold (one threshold). Diagnostic override only"),
    "min_height": ("--min-height", float, "minimum peak amplitude (0 disables)"),
    "centroid_halfwidth": ("--centroid-halfwidth", int,
        "local-centroid half-width in bins (0 = full-region mean)"),
    "pairwise_bins": ("--pairwise-bins", int,
        "bins over +-6 mm for the sigma_vtx-vtx fit"),
}  # fmt: skip


PEAK_KEYS: tuple[str, ...] = (
    "peak_threshold",
    "integral_threshold",
    "min_height",
    "centroid_halfwidth",
)


def add_regime_args(
    parser: argparse.ArgumentParser, keys: tuple[str, ...] | None = None
) -> None:
    """Add ``--regime`` and working-point arguments (default: from the regime).

    ``keys`` selects a subset of ``REGIME_ARGS`` (default: all of them), for
    scripts that only do peak finding (``PEAK_KEYS``).
    """
    parser.add_argument(
        "--regime",
        required=True,
        choices=sorted(REGIMES),
        help="working point (pv_finder.utils.working_points); explicit "
        "flags below override it and are recorded",
    )
    for dest, (flag, typ, hlp) in REGIME_ARGS.items():
        if keys is not None and dest not in keys:
            continue
        parser.add_argument(flag, type=typ, default=None, dest=dest,
                            help=f"{hlp} (default: from --regime)")  # fmt: skip


def resolve_regime_args(args: argparse.Namespace) -> Regime:
    """Fill unset working-point arguments from the regime; record overrides.

    Sets ``args.regime_dict``, ``args.regime_overrides`` and
    ``args.density_from_regime``. If no density model was given on the command
    line (no ``--e2e-model`` and no ``--t2kde-model``/``--k2h-model``), the
    regime's density model and architecture are used.
    """
    regime = get_regime(args.regime)
    defaults = regime_defaults(regime)
    present = [d for d in REGIME_ARGS if hasattr(args, d)]
    explicit = {d: getattr(args, d) for d in present if getattr(args, d) is not None}
    for dest in present:
        if getattr(args, dest) is None:
            value = (
                args.integral_threshold
                if dest == "integral_threshold_res"
                else defaults[dest]
            )
            setattr(args, dest, value)
    overrides = record_overrides(regime, explicit)
    model_given = bool(
        getattr(args, "e2e_model", None)
        or getattr(args, "t2kde_model", None)
        or getattr(args, "k2h_model", None)
    )
    args.density_from_regime = not model_given
    if not model_given and hasattr(args, "e2e_model"):
        dm = regime.density
        args.e2e_model = dm.checkpoint
        args.e2e_type = dm.model_type
        if hasattr(args, "e2e_unet_channels"):
            args.e2e_unet_channels = dm.unet_channels
            args.e2e_latent_channels = dm.latent_channels
            args.e2e_hidden = list(dm.hidden)
    elif model_given:
        e2e = getattr(args, "e2e_model", None)
        if e2e and Path(e2e).resolve() == Path(regime.density.checkpoint).resolve():
            args.density_from_regime = True  # the regime's model, named explicitly
        else:
            overrides["density_model"] = (
                e2e
                or f"{getattr(args, 't2kde_model', None)} + "
                f"{getattr(args, 'k2h_model', None)}"
            )
    args.regime_dict = regime.to_dict()
    args.regime_overrides = overrides
    _print_banner(regime, overrides)
    return regime


def _print_banner(regime: Regime, overrides: dict[str, Any]) -> None:
    p = regime.peaks
    print(f"\n  Working point: regime '{regime.name}' ({regime.description})")
    print(f"    window {regime.match_window_mm} mm (HS {regime.hs_match_window_mm} mm), "
          f"peaks amp {p.amplitude} / width {p.min_width} / integral {p.integral} / "
          f"height {p.min_height} / centroid +-{p.centroid_halfwidth}, "
          f"density model {regime.density.name}")  # fmt: skip
    if regime.pending:
        print(f"    NOT FROZEN: {', '.join(regime.pending)}")
    if overrides:
        print(f"    OVERRIDES: {overrides}")


def git_hash() -> str:
    """Current git HEAD (or 'unknown')."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty_eval() -> bool:
    """True if src/ has uncommitted changes (results may not be reproducible)."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.strip())


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Metadata block stored in every result file."""
    return {
        "regime": args.regime_dict,
        "regime_overrides": args.regime_overrides,
        "density_from_regime": args.density_from_regime,
        "git_hash": git_hash(),
        "src_dirty": git_dirty_eval(),
        "argv": list(sys.argv),
    }


def write_invocation(outdir: str | Path, args: argparse.Namespace) -> Path:
    """Write INVOCATION.txt (command, git hash, full regime) to ``outdir``."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance(args)
    path = out / "INVOCATION.txt"
    path.write_text(
        " ".join(prov["argv"])
        + f"\n# git {prov['git_hash']} (src dirty: {prov['src_dirty']})\n"
        + "# regime:\n"
        + json.dumps(
            {k: prov[k] for k in ("regime", "regime_overrides", "density_from_regime")},
            indent=2,
            default=str,
        )
        + "\n"
    )
    return path
