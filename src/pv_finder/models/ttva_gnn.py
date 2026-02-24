"""Heterogeneous GAT model for Track-to-Vertex Association (TTVA).

Bipartite graph attention network that classifies track-vertex edges.
Given a set of tracks and candidate primary vertices (from PVF peak finding
or MC truth), predicts which tracks belong to which vertex.

Architecture:
    - Track encoder: Linear(track_input_size -> hidden_dim) + LeakyReLU
    - PV encoder: Linear(pv_input_size -> hidden_dim) + LeakyReLU
    - N HeteroConv layers with GATConv (multi-head, no concat) + residual
    - Edge predictor: 3-layer MLP (2*hidden -> hidden -> hidden -> 1)

Input graph (HeteroData):
    - track.x: (n_tracks, 8) — [d0, z0, d0_err, z0_err, cov, theta, phi, pt_scaled]
    - pv.x: (n_pvs, 2) — [z_position, peak_height]
    - (track, to, pv).edge_index: fully connected bipartite edges
    - (track, to, pv).edge_attr: (n_edges, 3) — [long_sig, horiz_sig, |dz|]

Output:
    - edge_logits: (n_edges,) — apply sigmoid for association scores

Migrated from atlas_pvfinder/tracks_to_vertex/model/autoencoder_models.py
(class TTVA_GATGraphConv_Model, lines 1471-1525).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, Linear


class TTVAGATModel(torch.nn.Module):
    """Heterogeneous Graph Attention Network for track-vertex association.

    State dict keys are preserved from the original TTVA_GATGraphConv_Model
    for weight compatibility: track_encoder, pv_encoder, convs,
    intermediary_layer_1, intermediary_layer_2, edge_predictor.
    """

    def __init__(
        self,
        track_input_size: int = 8,
        pv_input_size: int = 2,
        hidden_dim: int = 32,
        leaky_param: float = 0.01,
        dropout: float = 0.25,
        num_heads: int = 4,
        edge_attr_dim: int = 3,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.LeakyReLU_param = leaky_param
        self.track_encoder = Linear(track_input_size, hidden_dim)
        self.pv_encoder = Linear(pv_input_size, hidden_dim)
        self.num_heads = num_heads
        self.edge_attr_dim = edge_attr_dim

        self.num_layers = 2
        self.convs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        ("track", "to", "pv"): GATConv(
                            in_channels=(-1, -1),
                            out_channels=hidden_dim,
                            heads=self.num_heads,
                            concat=False,
                            edge_dim=self.edge_attr_dim,
                            add_self_loops=False,
                        ),
                        ("pv", "rev_to", "track"): GATConv(
                            in_channels=(-1, -1),
                            out_channels=hidden_dim,
                            heads=self.num_heads,
                            concat=False,
                            edge_dim=self.edge_attr_dim,
                            add_self_loops=False,
                        ),
                    },
                    aggr="max",
                )
            )

        # Final edge prediction layers
        self.intermediary_layer_1 = Linear(2 * hidden_dim, hidden_dim)
        self.intermediary_layer_2 = Linear(hidden_dim, hidden_dim)
        self.edge_predictor = Linear(hidden_dim, 1)

        # LeakyReLU as module (no learnable params, no state_dict impact)
        self._leaky = nn.LeakyReLU(leaky_param)

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Forward pass: encode nodes, propagate, predict edge logits.

        Args:
            data: HeteroData with track/pv nodes, edges, and edge attributes.

        Returns:
            Edge logits of shape (n_edges,). Apply sigmoid for scores.
        """
        leaky = self._leaky

        # Encode initial features
        track_init = leaky(self.track_encoder(data["track"].x.float()))
        pv_init = leaky(self.pv_encoder(data["pv"].x.float()))

        x_dict = {"track": track_init, "pv": pv_init}

        # Propagate through GNN layers with residual connections
        for conv in self.convs:
            x_dict_new = conv(
                x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict
            )
            x_dict = {
                node_type: F.relu(x_dict_new[node_type] + x_dict[node_type])
                for node_type in x_dict
            }

        # Edge prediction: concatenate source (track) and destination (pv) embeddings
        src, dst = data[("track", "to", "pv")].edge_index
        edge_feat = torch.cat([x_dict["track"][src], x_dict["pv"][dst]], dim=1)
        intermed_1 = leaky(self.intermediary_layer_1(edge_feat))
        intermed_2 = leaky(self.intermediary_layer_2(intermed_1))
        edge_logits = self.edge_predictor(intermed_2).squeeze()

        return edge_logits
