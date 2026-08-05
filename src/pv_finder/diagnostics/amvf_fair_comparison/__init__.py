"""Apples-to-apples PV-Finder vs AMVF comparison.

The production eval (``run_eval_pvf_run3.py``) compares the two algorithms in a
way that is not symmetric, in two independent respects:

1. **The matching window is PV-Finder's own fitted sigma_vtx-vtx**, applied
   unchanged to AMVF (``run_eval_pvf_run3.py`` lines 492/518/527/551).  AMVF is
   therefore judged with a window derived from a resolution that is not its own.
   Improving PV-Finder's resolution mechanically tightens AMVF's window and
   raises AMVF's measured fake rate without AMVF changing at all.
2. **Truth is filtered to nTracks >= 2** (``run3_io.py``), so a reconstructed
   vertex landing on a real nTrk == 1 interaction counts as a fake.  This
   penalises PV-Finder, which works from a track density and can resolve such an
   interaction, far more than AMVF, which fits vertices and needs >= 2 tracks.

The two effects run in opposite directions.  This package measures both on the
same events and produces the window scan that makes the comparison ungameable.

Modules
-------
``cache``     build a compact npz of peak / vertex / truth positions
``matching``  vertex matcher, verified bit-for-bit against the production one
``scan``      the window scan, the four-cell table, and the bootstrap
``plots``     the window-scan figure
"""

from pv_finder.diagnostics.amvf_fair_comparison.matching import (
    MatchCounts,
    excused_by_low_ntrk,
    match_vertices,
)

__all__ = ["MatchCounts", "excused_by_low_ntrk", "match_vertices"]
