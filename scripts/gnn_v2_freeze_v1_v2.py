#!/usr/bin/env python3
"""GNN v1 (e213) vs v2 (e527, e539) on identical chain events (v6 peaks).

Writes ``v1_vs_v2.md`` from the ``summary/*.json`` files: track F1,
efficiency, purity, Clean/truth and HS selection per mu bin and t.

    venv/bin/python scripts/gnn_v2_freeze_v1_v2.py --dir outputs/11_12_2026_output/gnn_v2_freeze
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TS = ("t0", "t0.9", "t0.95", "t0.98", "t0.99")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    md = ["# GNN v1 vs v2 on identical chain events (v6 peaks, reco >= 1 track)\n"]
    for sset, names in (
        ("val", ("v1e213", "v2e527", "v2e539")),
        ("test", ("v1e213", "v2e527")),
    ):
        s = {
            n: json.loads((a.dir / f"summary/{sset}_v6_{n}.json").read_text())
            for n in names
        }
        arms = s[names[0]]["arms"]
        md.append(f"\n## {sset}: track F1 / Clean/truth [%] ({' / '.join(names)})\n")
        md.append("| mu | " + " | ".join(TS) + " |\n|---|" + "---|" * len(TS))
        for b in s[names[0]]["bins"]:
            cells = []
            for t in TS:
                i = arms.index(t)
                f1s, cts = [], []
                for n in names:
                    r = s[n]["bins"][b]["ratios_min1"]
                    e, p = r["trk_eff"]["value"][i], r["trk_pur"]["value"][i]
                    f1s.append(f"{200 * e * p / (e + p):.2f}")
                    cts.append(f"{100 * r['clean_per_truth']['value'][i]:.2f}")
                cells.append("/".join(f1s) + " ; " + "/".join(cts))
            md.append(f"| {b} | " + " | ".join(cells) + " |")
        if all("hs_selection" in s[n]["bins"]["inclusive"] for n in names):
            md.append(f"\n### {sset}: HS selection [%] (Athena rule)\n")
            md.append("| mu | " + " | ".join(TS) + " |\n|---|" + "---|" * len(TS))
            for b in s[names[0]]["bins"]:
                cells = ["/".join(f"{100 * s[n]['bins'][b]['hs_selection']['value'][arms.index(t)]:.2f}" for n in names) for t in TS]  # fmt: skip
                md.append(f"| {b} | " + " | ".join(cells) + " |")
    (a.dir / "v1_vs_v2.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
