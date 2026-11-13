"""Configurable TTVA networks for the GNN v3 campaign.

Two families on the same bipartite track <-> candidate graph (the inputs of
``ttva_gat.TTVAGATModel``: track.x (8), pv.x = [z, height], edge_attr =
[long_sig, horiz_sig, |dz|]):

* ``arch="gat"``: the v2 design (track / PV encoders, hetero GATConv layers,
  residual, edge MLP) with a configurable number of layers, LayerNorm and
  the input transform below.
* ``arch="ign"``: an interaction network. Each layer updates the edge state
  from its two end nodes, then aggregates messages into the PV nodes and
  back into the tracks with several aggregators at once: multi-head
  attention, a gated mean, max and the log of the gated sum of gates (a soft
  track count, so a vertex knows its multiplicity, which attention-softmax
  and max cannot provide).

Common options:

* inputs: ``asinh(x / s)`` with fixed physical scales (linear below one
  unit, logarithmic above), so heavy-tailed significances and errors do not
  dominate. ``pv_counts`` adds, per PV, log(1 + n) of the incident edges with
  |long_sig| < 1 and < 3 (multiplicity seen before any learning);
  ``pv_sigma`` adds the candidate width (``pv.sigma``, s8 graphs only).
* ``loss="bce"``: one logit per edge, score = sigmoid (as v1/v2).
  ``loss="softmax_null"``: per track, a softmax over its edges plus a learned
  "no vertex" logit from the track embedding; the edge score is that
  probability. This is the decision the chain makes (top-1 edge, kept if its
  score >= t).

``scores(data)`` returns per-edge scores in [0, 1] for either loss; the chain
replays top-1 + cut on them unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.utils import scatter, softmax

# asinh(x / scale) per input column (mm, rad, pT / 50 GeV)
TRACK_SCALES = (0.05, 50.0, 0.05, 0.1, 0.005, 1.0, 1.0, 0.02)
PV_SCALES = (50.0, 0.1)
EDGE_SCALES = (1.0, 1.0, 0.1)
SIGMA_SCALE = 0.05
E_KEY = ("track", "to", "pv")


def _mlp(n_in: int, n_hidden: int, n_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, n_hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(n_hidden, n_out)
    )  # fmt: skip


class _Aggregate(nn.Module):
    """Attention + gated mean + max + log soft count of messages into ``index``."""

    def __init__(self, h: int, he: int, heads: int) -> None:
        super().__init__()
        if h % heads:
            raise ValueError("hidden must be divisible by heads")
        self.heads = heads
        self.att = nn.Linear(he, heads)
        self.gate = nn.Linear(he, 1)

    def forward(
        self, m: torch.Tensor, e: torch.Tensor, index: torch.Tensor, n: int
    ) -> torch.Tensor:
        n_e, h = m.shape
        a = softmax(self.att(e), index, num_nodes=n)  # (E, heads)
        mh = (a.unsqueeze(-1) * m.view(n_e, self.heads, h // self.heads)).view(n_e, h)
        agg_att = scatter(mh, index, dim=0, dim_size=n, reduce="sum")
        g = torch.sigmoid(self.gate(e))
        gs = scatter(g, index, dim=0, dim_size=n, reduce="sum")
        agg_g = scatter(g * m, index, dim=0, dim_size=n, reduce="sum") / (gs + 1e-6)
        agg_max = scatter(m, index, dim=0, dim_size=n, reduce="max")
        return torch.cat([agg_att, agg_g, agg_max, torch.log1p(gs)], dim=1)


class _IGNLayer(nn.Module):
    """Edge update, track -> PV, PV -> track (residual + optional LayerNorm)."""

    def __init__(self, h: int, he: int, heads: int, dropout: float, norm: bool) -> None:
        super().__init__()
        self.e_t = nn.Linear(h, he, bias=False)
        self.e_p = nn.Linear(h, he, bias=False)
        self.e_e = nn.Linear(he, he)
        self.e_out = nn.Linear(he, he)
        self.m_tp_x, self.m_tp_e = nn.Linear(h, h, bias=False), nn.Linear(he, h)
        self.m_pt_x, self.m_pt_e = nn.Linear(h, h, bias=False), nn.Linear(he, h)
        self.agg_p, self.agg_t = _Aggregate(h, he, heads), _Aggregate(h, he, heads)
        self.up = _mlp(4 * h + 1, h, h, dropout)
        self.ut = _mlp(4 * h + 1, h, h, dropout)
        mk = (lambda d: nn.LayerNorm(d)) if norm else (lambda d: nn.Identity())
        self.n_e, self.n_p, self.n_t = mk(he), mk(h), mk(h)

    def forward(self, xt, xp, e, src, dst):  # noqa: ANN001, ANN201
        e = self.n_e(e + self.e_out(F.silu(self.e_t(xt)[src] + self.e_p(xp)[dst] + self.e_e(e))))  # fmt: skip
        m = F.silu(self.m_tp_x(xt)[src] + self.m_tp_e(e))
        xp = self.n_p(xp + self.up(torch.cat([xp, self.agg_p(m, e, dst, xp.shape[0])], 1)))  # fmt: skip
        m = F.silu(self.m_pt_x(xp)[dst] + self.m_pt_e(e))
        xt = self.n_t(xt + self.ut(torch.cat([xt, self.agg_t(m, e, src, xt.shape[0])], 1)))  # fmt: skip
        return xt, xp, e


class TTVAv3(nn.Module):
    """See module docstring. All constructor arguments are saved in ``config``."""

    def __init__(
        self,
        arch: str = "ign",
        hidden: int = 64,
        edge_hidden: int | None = None,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        norm: bool = True,
        pv_counts: bool = True,
        pv_sigma: bool = False,
        loss: str = "bce",
    ) -> None:
        super().__init__()
        if arch not in ("gat", "ign") or loss not in ("bce", "softmax_null"):
            raise ValueError(f"unknown arch/loss {arch}/{loss}")
        self.config = dict(
            arch=arch, hidden=hidden, edge_hidden=edge_hidden, layers=layers,
            heads=heads, dropout=dropout, norm=norm, pv_counts=pv_counts,
            pv_sigma=pv_sigma, loss=loss,
        )  # fmt: skip
        self.arch, self.loss, self.pv_counts, self.pv_sigma = (
            arch,
            loss,
            pv_counts,
            pv_sigma,
        )
        he = edge_hidden or hidden
        h = hidden
        self.register_buffer("s_trk", torch.tensor(TRACK_SCALES))
        self.register_buffer("s_pv", torch.tensor(PV_SCALES))
        self.register_buffer("s_edge", torch.tensor(EDGE_SCALES))
        n_pv = 2 + 2 * pv_counts + pv_sigma
        self.enc_t = _mlp(8, h, h, 0.0)
        self.enc_p = _mlp(n_pv, h, h, 0.0)
        mk = (lambda d: nn.LayerNorm(d)) if norm else (lambda d: nn.Identity())
        if arch == "ign":
            self.enc_e = _mlp(3, he, he, 0.0)
            self.blocks = nn.ModuleList(
                _IGNLayer(h, he, heads, dropout, norm) for _ in range(layers)
            )
            n_edge_in = he
        else:
            self.blocks = nn.ModuleList(
                HeteroConv(
                    {
                        E_KEY: GATConv(
                            (h, h),
                            h,
                            heads=heads,
                            concat=False,
                            edge_dim=3,
                            add_self_loops=False,
                        ),  # fmt: skip
                        ("pv", "rev_to", "track"): GATConv(
                            (h, h),
                            h,
                            heads=heads,
                            concat=False,
                            edge_dim=3,
                            add_self_loops=False,
                        ),  # fmt: skip
                    },
                    aggr="max",
                )
                for _ in range(layers)
            )
            self.gat_norm = nn.ModuleList(
                nn.ModuleDict({"track": mk(h), "pv": mk(h)}) for _ in range(layers)
            )
            n_edge_in = 3
        self.dec_t = nn.Linear(h, h, bias=False)
        self.dec_p = nn.Linear(h, h, bias=False)
        self.dec_e = nn.Linear(n_edge_in, h)
        self.dec = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        if loss == "softmax_null":
            self.null = _mlp(h, h, 1, 0.0)

    def _inputs(self, data: HeteroData):  # noqa: ANN202
        xt = torch.asinh(data["track"].x.float() / self.s_trk)
        xp = torch.asinh(data["pv"].x.float() / self.s_pv)
        ea = data[E_KEY].edge_attr.float()
        e = torch.asinh(ea / self.s_edge)
        src, dst = data[E_KEY].edge_index
        n_p = xp.shape[0]
        extra = []
        if self.pv_counts:
            ls = ea[:, 0].abs()
            for cut in (1.0, 3.0):
                c = scatter((ls < cut).float(), dst, dim=0, dim_size=n_p, reduce="sum")
                extra.append(torch.log1p(c).unsqueeze(1))
        if self.pv_sigma:
            extra.append(
                torch.asinh(data["pv"].sigma.float() / SIGMA_SCALE).unsqueeze(1)
            )
        if extra:
            xp = torch.cat([xp, *extra], dim=1)
        return xt, xp, e, src, dst

    def forward(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(edge logits (E,), null logits (n_tracks,) or None)."""
        xt, xp, e, src, dst = self._inputs(data)
        xt, xp = self.enc_t(xt), self.enc_p(xp)
        if self.arch == "ign":
            e = self.enc_e(e)
            for blk in self.blocks:
                xt, xp, e = blk(xt, xp, e, src, dst)
        else:
            ei = {E_KEY: data[E_KEY].edge_index, ("pv", "rev_to", "track"): data[E_KEY].edge_index.flip(0)}  # fmt: skip
            ea = {E_KEY: e, ("pv", "rev_to", "track"): e}
            x = {"track": xt, "pv": xp}
            for conv, nrm in zip(self.blocks, self.gat_norm):
                new = conv(x, ei, edge_attr_dict=ea)
                x = {k: nrm[k](F.silu(new[k]) + x[k]) for k in x}
            xt, xp = x["track"], x["pv"]
        logit = self.dec(self.dec_t(xt)[src] + self.dec_p(xp)[dst] + self.dec_e(e)).squeeze(-1)  # fmt: skip
        null = self.null(xt).squeeze(-1) if self.loss == "softmax_null" else None
        return logit, null

    def scores_from(
        self,
        logit: torch.Tensor,
        null: torch.Tensor | None,
        src: torch.Tensor,
        n_t: int,
    ) -> torch.Tensor:
        """Per-edge score in [0, 1] (sigmoid, or the per-track softmax with null)."""
        if self.loss == "bce":
            return torch.sigmoid(logit)
        lse = _track_logsumexp(logit, null, src, n_t)
        return torch.exp(logit - lse[src])

    def scores(self, data: HeteroData) -> torch.Tensor:
        """Per-edge scores of a graph (or batch)."""
        logit, null = self(data)
        return self.scores_from(logit, null, data[E_KEY].edge_index[0], data["track"].x.shape[0])  # fmt: skip


def _track_logsumexp(
    logit: torch.Tensor, null: torch.Tensor, src: torch.Tensor, n_t: int
) -> torch.Tensor:
    """log(sum_j exp(logit_ij) + exp(null_i)) per track, numerically stable."""
    mx = torch.maximum(scatter(logit, src, dim=0, dim_size=n_t, reduce="max"), null)
    mx = mx.detach()
    s = scatter(torch.exp(logit - mx[src]), src, dim=0, dim_size=n_t, reduce="sum")
    return mx + torch.log(s + torch.exp(null - mx))


def softmax_null_loss(
    logit: torch.Tensor,
    null: torch.Tensor,
    src: torch.Tensor,
    y: torch.Tensor,
    n_t: int,
) -> torch.Tensor:
    """Mean over tracks of -log p(target); target = the positive edge or "no vertex"."""
    lse = _track_logsumexp(logit, null, src, n_t)
    pos = scatter(logit * y, src, dim=0, dim_size=n_t, reduce="sum")
    has = scatter(y, src, dim=0, dim_size=n_t, reduce="sum")
    if torch.any(has > 1.5):
        raise ValueError("a track has more than one positive edge")
    target = torch.where(has > 0.5, pos, null)
    return (lse - target).mean()


def bce_loss(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-edge BCE with the batch pos_weight of v1/v2 (n_neg / n_pos)."""
    n_pos = y.sum().clamp(min=1.0)
    pw = ((y.numel() - n_pos) / n_pos).detach()
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
