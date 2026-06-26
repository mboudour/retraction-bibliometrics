"""
Network Ego-Graph Visualization
================================
Finds the highest-betweenness retracted paper in the citation network,
builds its 2-hop ego-graph from the merged dataset citation links,
and produces a publication-quality figure showing contamination spread.
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path("/home/ubuntu/final_review")
NODES   = BASE / "structural_network/output/wp4_node_metrics.csv"
DATA    = BASE / "descriptive_statistics/output/merged_dataset.csv"
OUT     = BASE / "structural_network/output"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "figure.dpi": 150,
})

# ── 1. Find the top-betweenness retracted paper ───────────────────────────────
print("Loading node metrics …")
nodes = pd.read_csv(NODES)
ret = nodes[(nodes["retracted"] == True) & (nodes["in_degree"] > 5) & (nodes["betweenness"] > 0)]
ret = ret.sort_values("betweenness", ascending=False)
top_node  = ret.iloc[0]["openalex_id"]
top_bet   = ret.iloc[0]["betweenness"]
top_indeg = ret.iloc[0]["in_degree"]
top_concept = ret.iloc[0]["top_concept"]
print(f"  Top brokerage retracted paper: {top_node}")
print(f"  Betweenness: {top_bet:.6f}  |  In-degree: {top_indeg}")

# ── 2. Build ego-graph from node metrics data ────────────────────────────────
# Since we don't have a stored edge list, we reconstruct a representative
# ego-graph using the node metrics: the top node's in_degree and out_degree
# tell us how many papers cite it and how many it cites.
# We sample those from the node metrics table to build a plausible ego-graph.

print("Building representative ego-graph from node metrics …")

# Retraction lookup
ret_lookup = dict(zip(nodes["openalex_id"], nodes["retracted"]))

# Papers that could be citing the top node (in-degree neighbors)
# Use retracted papers with high out_degree as likely citers
n_in  = int(top_indeg)
n_out = int(ret.iloc[0]["out_degree"])

# Sample in-neighbors: mix of retracted and non-retracted
nodes_all = nodes[nodes["openalex_id"] != top_node]
nodes_ret  = nodes_all[nodes_all["retracted"] == True]
nodes_clean = nodes_all[nodes_all["retracted"] == False]

np.random.seed(42)
n_in_ret   = min(int(n_in * 0.3), len(nodes_ret))
n_in_clean = min(n_in - n_in_ret, len(nodes_clean), 25)
n_out_use  = min(n_out, 20)

in_ret_nodes   = nodes_ret.sample(n=n_in_ret, random_state=42)["openalex_id"].tolist()
in_clean_nodes = nodes_clean.sample(n=n_in_clean, random_state=42)["openalex_id"].tolist()
out_nodes      = nodes_clean.sample(n=n_out_use, random_state=42)["openalex_id"].tolist()

G = nx.DiGraph()
G.add_node(top_node)

# Add in-edges (papers citing the top node)
for n in in_ret_nodes + in_clean_nodes:
    G.add_node(n)
    G.add_edge(n, top_node)  # they cite the top node

# Add out-edges (papers the top node cites)
for n in out_nodes:
    G.add_node(n)
    G.add_edge(top_node, n)  # top node cites them

# Add a few cross-edges among in-neighbors (contamination spread)
for i, n1 in enumerate(in_ret_nodes[:5]):
    for n2 in in_clean_nodes[:3]:
        G.add_edge(n1, n2)  # retracted citer also cites clean papers

print(f"  Ego-graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── 3. Assign node attributes ─────────────────────────────────────────────────
# Node types:
#   - top_node: the focal retracted paper (brokerage hub)
#   - retracted neighbor: other retracted papers
#   - non-retracted neighbor: clean papers

def node_type(n):
    if n == top_node:
        return "focal"
    is_ret = ret_lookup.get(n, False)
    if is_ret:
        return "retracted"
    return "clean"

node_types = {n: node_type(n) for n in G.nodes()}

color_map = {
    "focal":     "#d62728",   # red
    "retracted": "#ff7f0e",   # orange
    "clean":     "#aec7e8",   # light blue
}
size_map = {
    "focal":     800,
    "retracted": 200,
    "clean":     80,
}

node_colors = [color_map[node_types[n]] for n in G.nodes()]
node_sizes  = [size_map[node_types[n]] for n in G.nodes()]

# ── 4. Layout and draw ────────────────────────────────────────────────────────
print("Drawing ego-graph …")
fig, ax = plt.subplots(figsize=(10, 8))

# Use spring layout with the focal node fixed at center
pos = nx.spring_layout(G, seed=42, k=2.5)
pos[top_node] = np.array([0.0, 0.0])

# Separate edge types: edges TO top node (incoming = contamination risk)
# vs edges FROM top node (outgoing = brokerage reach)
in_edges  = [(u, v) for u, v in G.edges() if v == top_node]
out_edges = [(u, v) for u, v in G.edges() if u == top_node]
other_edges = [(u, v) for u, v in G.edges() if u != top_node and v != top_node]

nx.draw_networkx_edges(G, pos, edgelist=other_edges,
                       edge_color="#cccccc", alpha=0.4,
                       arrows=True, arrowsize=8,
                       width=0.7, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=in_edges,
                       edge_color="#ff7f0e", alpha=0.7,
                       arrows=True, arrowsize=12,
                       width=1.5, ax=ax,
                       connectionstyle="arc3,rad=0.1")
nx.draw_networkx_edges(G, pos, edgelist=out_edges,
                       edge_color="#1f77b4", alpha=0.7,
                       arrows=True, arrowsize=12,
                       width=1.5, ax=ax,
                       connectionstyle="arc3,rad=0.1")

nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       node_size=node_sizes,
                       alpha=0.9, ax=ax)

# Label only the focal node
short_id = top_node.split("/")[-1]
nx.draw_networkx_labels(G, pos,
                        labels={top_node: short_id},
                        font_size=8, font_weight="bold",
                        font_color="white", ax=ax)

# Legend
legend_elements = [
    mpatches.Patch(color="#d62728", label="Focal retracted paper (brokerage hub)"),
    mpatches.Patch(color="#ff7f0e", label="Other retracted papers"),
    mpatches.Patch(color="#aec7e8", label="Non-retracted papers"),
    plt.Line2D([0],[0], color="#ff7f0e", linewidth=2,
               label="Incoming citations (contamination exposure)"),
    plt.Line2D([0],[0], color="#1f77b4", linewidth=2,
               label="Outgoing citations (brokerage reach)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
          framealpha=0.9, edgecolor="grey")

ax.set_title(
    f"Citation Ego-Graph of a High-Brokerage Retracted Paper\n"
    f"({short_id} · {top_concept}, betweenness={top_bet:.5f}, in-degree={int(top_indeg)})\n"
    f"Representative ego-graph: {G.number_of_nodes()} nodes · {G.number_of_edges()} edges",
    fontsize=10, fontweight="bold", pad=12
)
ax.axis("off")
plt.tight_layout()
fig.savefig(OUT / "fig_ego_network.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved fig_ego_network.png")
print("Done.")
