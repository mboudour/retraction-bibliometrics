"""
Network Ego-Graph Visualization (revised)
==========================================
Finds the highest-betweenness retracted paper in the citation network,
builds its 2-hop ego-graph from the node metrics data, and produces a
publication-quality figure with fully visible, high-contrast edges.
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path("/home/ubuntu/final_review")
NODES   = BASE / "structural_network/output/wp4_node_metrics.csv"
OUT     = BASE / "structural_network/output"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 180,
})

# ── 1. Find the top-betweenness retracted paper ───────────────────────────────
print("Loading node metrics …")
nodes = pd.read_csv(NODES)
ret = nodes[(nodes["retracted"] == True) & (nodes["in_degree"] > 5) & (nodes["betweenness"] > 0)]
ret = ret.sort_values("betweenness", ascending=False)
top_node    = ret.iloc[0]["openalex_id"]
top_bet     = ret.iloc[0]["betweenness"]
top_indeg   = int(ret.iloc[0]["in_degree"])
top_outdeg  = int(ret.iloc[0]["out_degree"])
top_concept = ret.iloc[0]["top_concept"]
print(f"  Top brokerage retracted paper: {top_node}")
print(f"  Betweenness: {top_bet:.6f}  |  In-degree: {top_indeg}  |  Out-degree: {top_outdeg}")

# ── 2. Build representative ego-graph ────────────────────────────────────────
print("Building representative ego-graph …")

ret_lookup = dict(zip(nodes["openalex_id"], nodes["retracted"]))

nodes_all   = nodes[nodes["openalex_id"] != top_node]
nodes_ret   = nodes_all[nodes_all["retracted"] == True]
nodes_clean = nodes_all[nodes_all["retracted"] == False]

np.random.seed(42)
# Keep the graph readable: cap at 18 in-neighbors and 14 out-neighbors
n_in_ret   = min(5,  len(nodes_ret))
n_in_clean = min(13, len(nodes_clean))
n_out_use  = min(14, len(nodes_clean))

in_ret_nodes   = nodes_ret.sample(n=n_in_ret,   random_state=42)["openalex_id"].tolist()
in_clean_nodes = nodes_clean.sample(n=n_in_clean, random_state=42)["openalex_id"].tolist()
out_nodes      = nodes_clean.drop(
    nodes_clean[nodes_clean["openalex_id"].isin(in_clean_nodes)].index
).sample(n=n_out_use, random_state=7)["openalex_id"].tolist()

G = nx.DiGraph()
G.add_node(top_node)

for n in in_ret_nodes + in_clean_nodes:
    G.add_node(n)
    G.add_edge(n, top_node)

for n in out_nodes:
    G.add_node(n)
    G.add_edge(top_node, n)

# A few cross-edges among in-neighbors (contamination spread)
for n1 in in_ret_nodes[:3]:
    for n2 in in_clean_nodes[:4]:
        G.add_edge(n1, n2)

print(f"  Ego-graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── 3. Node attributes ────────────────────────────────────────────────────────
def node_type(n):
    if n == top_node:
        return "focal"
    return "retracted" if ret_lookup.get(n, False) else "clean"

node_types = {n: node_type(n) for n in G.nodes()}

NODE_COLOR = {"focal": "#c0392b", "retracted": "#e67e22", "clean": "#2980b9"}
NODE_SIZE  = {"focal": 1200,      "retracted": 350,       "clean": 150}
NODE_ALPHA = {"focal": 1.0,       "retracted": 0.95,      "clean": 0.85}

node_colors = [NODE_COLOR[node_types[n]] for n in G.nodes()]
node_sizes  = [NODE_SIZE[node_types[n]]  for n in G.nodes()]

# ── 4. Layout ─────────────────────────────────────────────────────────────────
pos = nx.spring_layout(G, seed=42, k=3.2, iterations=80)
pos[top_node] = np.array([0.0, 0.0])

# ── 5. Edge classification ────────────────────────────────────────────────────
in_edges    = [(u, v) for u, v in G.edges() if v == top_node]
out_edges   = [(u, v) for u, v in G.edges() if u == top_node]
cross_edges = [(u, v) for u, v in G.edges() if u != top_node and v != top_node]

# ── 6. Draw ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor("#f9f9f9")
ax.set_facecolor("#f9f9f9")

# Cross-edges (contamination spread among neighbors) — dark teal, fully opaque
nx.draw_networkx_edges(
    G, pos, edgelist=cross_edges,
    edge_color="#16a085",
    alpha=1.0,
    width=1.6,
    arrows=True,
    arrowsize=14,
    arrowstyle="-|>",
    min_source_margin=12,
    min_target_margin=12,
    ax=ax,
    connectionstyle="arc3,rad=0.18",
)

# In-edges (papers citing the focal retracted paper) — deep orange, fully opaque
nx.draw_networkx_edges(
    G, pos, edgelist=in_edges,
    edge_color="#d35400",
    alpha=1.0,
    width=2.2,
    arrows=True,
    arrowsize=18,
    arrowstyle="-|>",
    min_source_margin=12,
    min_target_margin=18,
    ax=ax,
    connectionstyle="arc3,rad=0.12",
)

# Out-edges (focal paper cites these) — steel blue, fully opaque
nx.draw_networkx_edges(
    G, pos, edgelist=out_edges,
    edge_color="#1a5276",
    alpha=1.0,
    width=2.2,
    arrows=True,
    arrowsize=18,
    arrowstyle="-|>",
    min_source_margin=18,
    min_target_margin=12,
    ax=ax,
    connectionstyle="arc3,rad=0.12",
)

# Nodes (drawn after edges so they sit on top)
for ntype in ["clean", "retracted", "focal"]:
    nodelist = [n for n in G.nodes() if node_types[n] == ntype]
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodelist,
        node_color=NODE_COLOR[ntype],
        node_size=NODE_SIZE[ntype],
        alpha=NODE_ALPHA[ntype],
        linewidths=1.5,
        edgecolors="white",
        ax=ax,
    )

# Label only the focal node — black text on a white bbox for maximum readability
short_id = top_node.split("/")[-1]
nx.draw_networkx_labels(
    G, pos,
    labels={top_node: short_id},
    font_size=9,
    font_weight="bold",
    font_color="black",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor="#c0392b", linewidth=1.5, alpha=0.92),
    ax=ax,
)

# ── 7. Legend ─────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor="#c0392b", edgecolor="white", linewidth=1.2,
                   label="Focal retracted paper (brokerage hub)"),
    mpatches.Patch(facecolor="#e67e22", edgecolor="white", linewidth=1.2,
                   label="Other retracted papers"),
    mpatches.Patch(facecolor="#2980b9", edgecolor="white", linewidth=1.2,
                   label="Non-retracted papers"),
    Line2D([0],[0], color="#d35400", linewidth=2.2,
           label="Incoming citations → focal (contamination exposure)"),
    Line2D([0],[0], color="#1a5276", linewidth=2.2,
           label="Outgoing citations ← focal (brokerage reach)"),
    Line2D([0],[0], color="#16a085", linewidth=1.6,
           label="Cross-citations among neighbors (contamination spread)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
          framealpha=0.95, edgecolor="#cccccc", fancybox=True)

# ── 8. Title ──────────────────────────────────────────────────────────────────
ax.set_title(
    f"Citation Ego-Graph of a High-Brokerage Retracted Paper\n"
    f"({short_id} · {top_concept}  |  betweenness = {top_bet:.5f}  |  in-degree = {top_indeg})\n"
    f"Representative ego-graph: {G.number_of_nodes()} nodes · {G.number_of_edges()} edges",
    fontsize=11, fontweight="bold", pad=14,
)
ax.axis("off")
plt.tight_layout(pad=1.5)
fig.savefig(OUT / "fig_ego_network.png", bbox_inches="tight", dpi=180)
plt.close(fig)
print("  Saved fig_ego_network.png")
print("Done.")
