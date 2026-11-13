"""Training script for the GNN v3 TTVA networks (``gnn.models.ttva_v3``).

Shard-cycling training as v1/v2 (one shard resident; one "epoch" = one shard),
with shuffling inside the shard, a cosine learning-rate schedule stepped per
shard epoch, gradient clipping and either loss of ``TTVAv3``. After every
shard epoch:

* validation loss on ``val_file`` (graphs of the run's own candidate type);
* on ``monitor_file`` (s8-candidate graphs with truth labels, the deployment
  distribution, the same for every run): loss and top-1 track efficiency,
  purity and F1 at a few score cuts (edge level: a track counts as correct
  when its kept top-1 edge is its labelled edge).

Everything is appended to ``<save_folder>/learning_curve.jsonl``. Each epoch
saves ``<runname>_epoch_<k>.pt`` (state dict) and overwrites ``last_fullstate.pth``
(model, optimizer, scheduler, epoch) for resumption; ``model_config.json``
next to them makes every checkpoint loadable by ``load_v3``.

``model_config: {legacy_v2: true}`` trains the unchanged v1/v2
``TTVAGATModel`` (hidden 32, 2 GAT layers) with BCE, as the control.

    PYTHONPATH=src python -u -m gnn.training.train_ttva_v3 -c configs/gnn/<cfg>.yml
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from gnn.models.ttva_gat import TTVAGATModel
from gnn.models.ttva_v3 import (
    E_KEY,
    TTVAv3,
    bce_loss,
    count_parameters,
    softmax_null_loss,
)

MONITOR_T = (0.5, 0.9, 0.94, 0.97)


class LegacyV2(torch.nn.Module):
    """The v1/v2 network behind the v3 interface (BCE, sigmoid scores)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = TTVAGATModel(
            track_input_size=8, pv_input_size=2, hidden_dim=32, num_heads=4,
            dropout=0.25, edge_attr_dim=3,
        )  # fmt: skip
        self.loss = "bce"
        self.config = {"legacy_v2": True}

    def forward(self, data):  # noqa: ANN001, ANN201
        return self.net(data).reshape(-1), None

    def scores(self, data):  # noqa: ANN001, ANN201
        return torch.sigmoid(self.forward(data)[0])

    def scores_from(self, logit, null, src, n_t):  # noqa: ANN001, ANN201
        return torch.sigmoid(logit)


def build_model(cfg: dict) -> torch.nn.Module:
    """``TTVAv3(**cfg)`` or the legacy v2 network."""
    if cfg.get("legacy_v2"):
        return LegacyV2()
    return TTVAv3(**cfg)


def load_v3(path: str | Path, device: torch.device) -> torch.nn.Module:
    """Load a v3-campaign checkpoint (state dict + sibling model_config.json)."""
    path = Path(path)
    cfg = json.loads((path.parent / "model_config.json").read_text())
    model = build_model(cfg)
    # lazy GATConv parameters of the legacy network accept a state dict as is
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def batch_loss(model: torch.nn.Module, batch) -> torch.Tensor:  # noqa: ANN001
    """The model's own loss on a batch."""
    y = batch[E_KEY].y.float()
    logit, null = model(batch)
    if model.loss == "bce":
        return bce_loss(logit, y)
    return softmax_null_loss(logit, null, batch[E_KEY].edge_index[0], y, batch["track"].x.shape[0])  # fmt: skip


def top1_counts(scores: torch.Tensor, batch) -> np.ndarray:  # noqa: ANN001
    """Per MONITOR_T: (correct kept, kept, tracks with a labelled edge)."""
    src = batch[E_KEY].edge_index[0]
    y = batch[E_KEY].y > 0.5
    n_t = batch["track"].x.shape[0]
    best = torch.full((n_t,), -1.0, device=scores.device).scatter_reduce(
        0, src, scores, reduce="amax"
    )
    is_best = scores >= best[src]
    # first best edge per track (edges are track-major)
    pos = torch.arange(len(src), device=src.device)
    first = torch.full((n_t,), len(src), device=src.device).scatter_reduce(
        0, src[is_best], pos[is_best], reduce="amin"
    )
    has_edge = first < len(src)
    top_y = torch.zeros(n_t, dtype=torch.bool, device=src.device)
    top_y[has_edge] = y[first[has_edge]]
    labelled = torch.zeros(n_t, dtype=torch.bool, device=src.device)
    labelled[src[y]] = True
    out = []
    for t in MONITOR_T:
        kept = has_edge & (best >= t)
        out.append([int((kept & top_y).sum()), int(kept.sum()), int(labelled.sum())])
    return np.array(out, dtype=np.int64)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, top1: bool) -> dict:  # fmt: skip
    """Mean batch loss (and top-1 counts) over a loader."""
    model.eval()
    tot, nb = 0.0, 0
    counts = np.zeros((len(MONITOR_T), 3), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            tot += float(batch_loss(model, batch))
            nb += 1
            if top1:
                counts += top1_counts(model.scores(batch), batch)
    out: dict = {"loss": tot / max(nb, 1)}
    if top1:
        for t, (c, k, lab) in zip(MONITOR_T, counts):
            eff, pur = c / max(lab, 1), c / max(k, 1)
            out[f"t{t:g}"] = {"eff": eff, "pur": pur, "f1": 2 * eff * pur / max(eff + pur, 1e-12)}  # fmt: skip
    return out


def main(cfg: dict, resume: bool) -> None:
    """Run training."""
    seed = int(cfg.get("seed", 20261113))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(f"cuda:{cfg['device_id']}")
    torch.cuda.set_device(device)
    save = Path(cfg["save_folder"])
    save.mkdir(parents=True, exist_ok=True)
    shards = [Path(p) for p in cfg["data_files"]]
    for p in shards + [Path(cfg["val_file"]), Path(cfg["monitor_file"])]:
        if not p.exists():
            raise FileNotFoundError(p)
    bs, nw = int(cfg["batch_size"]), int(cfg.get("num_workers", 2))
    kw = dict(batch_size=bs, num_workers=nw, prefetch_factor=2 if nw else None)
    val = DataLoader(
        torch.load(cfg["val_file"], weights_only=False), shuffle=False, **kw
    )
    same = Path(cfg["monitor_file"]).resolve() == Path(cfg["val_file"]).resolve()
    mon = val if same else DataLoader(torch.load(cfg["monitor_file"], weights_only=False), shuffle=False, **kw)  # fmt: skip
    model = build_model(cfg["model_config"]).to(device)
    first = next(iter(val)).to(device)
    with torch.no_grad():
        model(first)  # materializes lazy layers of the legacy network
    (save / "model_config.json").write_text(json.dumps(model.config, indent=1) + "\n")
    n_params = count_parameters(model)
    print(f"model {model.config}: {n_params:,} parameters", flush=True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )  # fmt: skip
    n_epochs = int(cfg["n_epochs"])
    warm = int(cfg.get("warmup_epochs", 0))
    eta_min = float(cfg.get("eta_min", 1e-5)) / float(cfg["learning_rate"])

    def lr_factor(ep: int) -> float:
        if ep < warm:
            return (ep + 1) / warm
        x = (ep - warm) / max(n_epochs - warm, 1)
        return eta_min + (1 - eta_min) * 0.5 * (1 + np.cos(np.pi * x))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    start = 0
    full = save / "last_fullstate.pth"
    if resume and full.exists():
        st = torch.load(full, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start = st["epoch"] + 1
        print(f"resumed at epoch {start}", flush=True)
    clip = cfg.get("grad_clip")
    gen = torch.Generator()
    curve = save / "learning_curve.jsonl"
    for ep in range(start, n_epochs):
        t0 = time.time()
        shard = shards[ep % len(shards)]
        gen.manual_seed(seed + ep)
        data = torch.load(shard, weights_only=False)
        t_load = time.time() - t0
        loader = DataLoader(data, shuffle=True, generator=gen, **kw)
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep}")
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            opt.step()
            tot += float(loss)
            nb += 1
        del loader, data
        lr = opt.param_groups[0]["lr"]
        sched.step()
        t_train = time.time() - t0 - t_load
        v = evaluate(model, val, device, top1=same)
        m = v if same else evaluate(model, mon, device, top1=True)
        rec = {
            "epoch": ep, "shard": shard.name, "train_loss": tot / max(nb, 1),
            "val_loss": v["loss"], "monitor": m, "lr": lr, "params": n_params,
            "t_load_s": round(t_load, 1), "t_train_s": round(t_train, 1),
            "t_total_s": round(time.time() - t0, 1),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }  # fmt: skip
        with curve.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        f94 = m.get("t0.94", {}).get("f1", float("nan"))
        print(f"epoch {ep}: train {rec['train_loss']:.5f} val {v['loss']:.5f} "
              f"mon {m['loss']:.5f} F1@0.94 {f94:.4f} lr {lr:.2e} "
              f"({t_load:.0f}+{t_train:.0f}+{rec['t_total_s'] - t_load - t_train:.0f} s)", flush=True)  # fmt: skip
        torch.save(model.state_dict(), save / f"{cfg['runname']}_epoch_{ep}.pt")
        tmp = save / "last_fullstate.tmp"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": ep}, tmp)  # fmt: skip
        tmp.replace(full)
    (save / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
    print("training complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    with open(a.config) as fh:
        configs = yaml.safe_load(fh)
    print(json.dumps(configs, indent=1), flush=True)
    main(configs, a.resume)
