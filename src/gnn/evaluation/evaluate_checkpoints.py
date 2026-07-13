"""Learning-curve evaluation: run many checkpoints over one labelled graph set.

Loads the graphs ONCE (the PU200 set is ~28 GB on disk) and loops over all
``*_epoch_<N>.pyt`` checkpoints in a directory, running the same per-graph
evaluation as gnn.evaluation.evaluate_ttva_graphs. Saves one summary.json
per checkpoint plus a combined learning_curve.json.

Usage:
    python -u -m gnn.evaluation.evaluate_checkpoints \\
        -r data/run4/ttva_graphs/pu200_truth_k20_30k.pt \\
        -w model_weights/ttva_gnn_hllhc \\
        --extra zeroshot_mu60=model_weights/gnn_ttva_epoch100.pyt \\
        -e MaxScore -t 0.5 -d 0 --first-event 28500 \\
        -o outputs/<date>_ttva_hllhc_eval/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from gnn.evaluation.evaluate_ttva_graphs import evaluate_graph
from gnn.models.ttva_gat import TTVAGATModel
from pv_finder.utils.constants import GNN_SCORE_THRESHOLD

_EPOCH_RE = re.compile(r"epoch_(\d+)\.pyt$")


def find_checkpoints(weights_dir: str | Path) -> list[tuple[int, Path]]:
    """Return (epoch, path) for every ``*epoch_<N>.pyt`` file, sorted by epoch."""
    found = []
    for path in Path(weights_dir).glob("*.pyt"):
        match = _EPOCH_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def evaluate_checkpoint(
    weights_path: str | Path,
    graphs: list[HeteroData],
    eval_method: str,
    threshold: float,
    device: torch.device,
) -> dict:
    """Evaluate one checkpoint over all graphs; return the summary dict."""
    model = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    totals = np.zeros(6, dtype=np.int64)
    edge_totals: dict[str, float] = {}
    for graph in tqdm(graphs, desc=Path(weights_path).stem, leave=False):
        results, _info, edge_metrics = evaluate_graph(
            model, graph, eval_method, threshold, device
        )
        totals += np.array(results, dtype=np.int64)
        for key, val in edge_metrics.items():
            edge_totals[key] = edge_totals.get(key, 0.0) + val

    clean, merged, split, fake, n_reco, n_truth = (int(x) for x in totals)
    summary = {
        "weights": str(weights_path),
        "clean": clean,
        "merged": merged,
        "split": split,
        "fake": fake,
        "reco_pvs": n_reco,
        "truth_pvs": n_truth,
        "edge_totals": edge_totals,
        "eval_method": eval_method,
        "threshold": threshold,
    }
    if n_reco > 0:
        summary["rates"] = {
            "clean": clean / n_reco,
            "merged": merged / n_reco,
            "split": split / n_reco,
            "fake": fake / n_reco,
        }
        summary["clean_per_truth"] = clean / n_truth if n_truth else 0.0
    if edge_totals.get("n_selected"):
        summary["edge_purity"] = (
            edge_totals["n_correct_selected"] / edge_totals["n_selected"]
        )
        summary["edge_efficiency"] = (
            edge_totals["n_correct_selected"] / edge_totals["n_true_edges"]
        )
    return summary


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a directory of checkpoints on labelled graphs"
    )
    parser.add_argument(
        "-r", "--graphs", required=True, type=str, help="Labelled graphs (.pt)"
    )
    parser.add_argument(
        "-w",
        "--weights-dir",
        required=True,
        type=str,
        help="Directory containing *epoch_<N>.pyt checkpoints",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Extra single checkpoints to include (repeatable)",
    )
    parser.add_argument(
        "-e",
        "--eval-method",
        default="MaxScore",
        choices=["MaxScore", "Threshold"],
        help="Edge selection method (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=GNN_SCORE_THRESHOLD,
        type=float,
        help="Score threshold (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--device-id", required=True, type=int, help="CUDA device (-1 = CPU)"
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument(
        "--first-event",
        default=0,
        type=int,
        help="Start at this graph index (e.g. the test-split boundary)",
    )
    parser.add_argument(
        "-n", "--max-events", default=None, type=int, help="Max events to evaluate"
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    if args.device_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoints: list[tuple[str, Path]] = [
        (f"epoch_{epoch}", path) for epoch, path in find_checkpoints(args.weights_dir)
    ]
    for extra in args.extra:
        label, _, path = extra.partition("=")
        checkpoints.append((label, Path(path)))
    if not checkpoints:
        msg = f"No *epoch_<N>.pyt checkpoints found in {args.weights_dir}"
        raise FileNotFoundError(msg)
    print(f"Checkpoints to evaluate: {[label for label, _ in checkpoints]}")

    print(f"Loading graphs from {args.graphs} ...")
    graphs: list[HeteroData] = torch.load(args.graphs, weights_only=False)
    graphs = graphs[args.first_event :]
    if args.max_events is not None:
        graphs = graphs[: args.max_events]
    print(f"Evaluating {len(graphs)} graphs from index {args.first_event}")

    out_dir = Path(args.output_dir)
    curve: list[dict] = []
    for label, weights_path in checkpoints:
        summary = evaluate_checkpoint(
            weights_path, graphs, args.eval_method, args.threshold, device
        )
        summary["label"] = label
        summary["first_event"] = args.first_event
        summary["graphs"] = args.graphs

        ckpt_dir = out_dir / label
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        curve.append(summary)

        rates = summary.get("rates", {})
        print(
            f"{label}: clean {rates.get('clean', 0):.4f} "
            f"merged {rates.get('merged', 0):.4f} "
            f"split {rates.get('split', 0):.4f} "
            f"fake {rates.get('fake', 0):.4f} "
            f"| clean/truth {summary.get('clean_per_truth', 0):.4f} "
            f"| edge purity {summary.get('edge_purity', 0):.4f} "
            f"eff {summary.get('edge_efficiency', 0):.4f}"
        )

    with open(out_dir / "learning_curve.json", "w") as f:
        json.dump(curve, f, indent=2)
    print(f"Saved learning curve for {len(curve)} checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
