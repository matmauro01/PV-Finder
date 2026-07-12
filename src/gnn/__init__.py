"""GNN Track-to-Vertex Association (TTVA) for ATLAS PV-Finder.

Assigns reconstructed tracks to primary vertices (from PVF peak finding or
MC truth) via binary edge classification on bipartite track-PV graphs.

Depends one-directionally on pv_finder for shared utilities (constants,
peak finding, training diagnostics); pv_finder never imports from gnn.
"""
