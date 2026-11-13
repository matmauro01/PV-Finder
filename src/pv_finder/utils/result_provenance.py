"""Provenance block for result JSON files.

Every frozen-working-point result file records the code version, the density
checkpoint it was produced with (by content hash, not only by path) and the
full regime of ``pv_finder.utils.working_points``. :func:`result_provenance`
returns that block; callers merge it into their JSON payload.
"""

from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from pv_finder.utils.working_points import get_regime

REPO = Path(__file__).resolve().parents[3]


def git_head() -> str:
    """``git rev-parse HEAD`` of the repository, or "unknown"."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_src_dirty() -> bool:
    """True if tracked or untracked files under src/ or scripts/ differ from HEAD."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "scripts"], cwd=REPO,
            text=True, stderr=subprocess.DEVNULL,
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.strip())


@lru_cache(maxsize=8)
def file_sha256(path: str) -> str:
    """sha256 of a file's content, read in 16 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def result_provenance(
    checkpoint: str | Path | None = None, regime: str = "hllhc"
) -> dict[str, Any]:
    """Git hash, density-checkpoint sha256 and the regime, for one result file.

    ``checkpoint`` defaults to the regime's density checkpoint; a relative
    path is taken relative to the repository root.
    """
    reg = get_regime(regime)
    ckpt = Path(checkpoint if checkpoint is not None else reg.density.checkpoint)
    path = ckpt if ckpt.is_absolute() else REPO / ckpt
    return {
        "git_hash": git_head(),
        "src_or_scripts_dirty": git_src_dirty(),
        "density_checkpoint": str(ckpt),
        "density_checkpoint_sha256": file_sha256(str(path)),
        "regime": reg.to_dict(),
    }
