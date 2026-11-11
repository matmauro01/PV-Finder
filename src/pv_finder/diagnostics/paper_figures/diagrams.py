"""Compile the TikZ algorithm diagrams (Figures 1 and 2) to PDF and PNG.

The sources live in ``diagrams/*.tex`` as standalone documents, so they can
also be pasted into the paper as TikZ. pdflatex from TeX Live 2022 is used when
present (the note's build), otherwise the system pdflatex.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

DIAGRAM_DIR = Path(__file__).resolve().parent / "diagrams"
DIAGRAMS = {
    "fig1_pvfinder_diagram": DIAGRAM_DIR / "fig1_pvfinder_diagram.tex",
    "fig2_gnn_diagram": DIAGRAM_DIR / "fig2_gnn_diagram.tex",
}
TEXLIVE_BIN = Path("/data/apps/texlive/bin/x86_64-linux")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    if TEXLIVE_BIN.is_dir():
        env["PATH"] = f"{TEXLIVE_BIN}:{env.get('PATH', '')}"
    return env


def compile_diagram(
    source: Path, out_dir: Path, name: str, dpi: int = 300
) -> list[Path]:
    """pdflatex ``source`` in a temporary directory; copy PDF and a PNG render."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _env()
    with tempfile.TemporaryDirectory(prefix="paperfig-tex-") as tmp:
        work = Path(tmp)
        shutil.copyfile(source, work / f"{name}.tex")
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
            cwd=work, env=env, capture_output=True, text=True, check=False,
        )  # fmt: skip
        pdf = work / f"{name}.pdf"
        if proc.returncode != 0 or not pdf.exists():
            tail = "\n".join(proc.stdout.splitlines()[-20:])
            raise RuntimeError(f"pdflatex failed for {source.name}:\n{tail}")
        shutil.copyfile(pdf, out_dir / f"{name}.pdf")
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf),
             str(out_dir / name)],
            check=True, env=env,
        )  # fmt: skip
    return [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]


def make_diagrams(out_dir: Path) -> list[Path]:
    """Compile every diagram into ``out_dir``."""
    paths: list[Path] = []
    for name, source in DIAGRAMS.items():
        paths += compile_diagram(source, out_dir, name)
        shutil.copyfile(source, out_dir / f"{name}.tex")
        paths.append(out_dir / f"{name}.tex")
    return paths
