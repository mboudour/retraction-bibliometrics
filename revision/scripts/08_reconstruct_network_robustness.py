#!/usr/bin/env python3
"""Step 6: reconstruct a balanced citation-network sample and document edge coverage.

The submitted network was built from references of retracted works. It consequently
omitted the non-retracted-to-non-retracted citation edges needed for a defensible
comparison of network position. This script constructs a transparent, balanced
one-hop neighbourhood network around the 1:1 retracted/non-retracted pairs made
in Step 4.

For every focal paper (a retracted paper and its matched non-retracted control),
the script retrieves from OpenAlex:
  * the focal paper's sampled outgoing references; and
  * a reproducible random sample of works that cite the focal paper, including
    their sampled outgoing references.

The resulting directed graph therefore includes retracted-to-non-retracted,
non-retracted-to-retracted, and non-retracted-to-non-retracted edges. It is not
a complete reconstruction of the entire OpenAlex citation graph. All sampling
caps, successful/failed API calls, and edge-type composition are written to the
output diagnostics and must be reported in the manuscript.

The script also creates a label-permutation design check on the matched focal
papers. This is an unadjusted robustness diagnostic only; adjusted H1/H2 tests
are conducted in Step 7.

Requirements:
  * Step 1 frozen corpus: revision/data/revision_master.csv.gz
  * Step 4 matched pairs: revision/output/step4_matched_pairs.csv
  * requests, pandas, numpy, matplotlib, networkx, scipy

Run from the project root:
  export OPENALEX_API_KEY="$(cat openalex_api_key.txt)"
  python revision/scripts/run_step_06.py --mode all --api-key "$OPENALEX_API_KEY" \
      --email "moses.boudourides@northwestern.edu"

The initial fetch is resumable through revision/data/step6_focal_neighborhoods.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import now_utc, project_root, read_master, revision_dirs, save_dual_figure, write_json  # noqa: E402

OPENALEX_URL = "https://api.openalex.org/works"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced focal citation-network neighbourhoods and diagnostics.")
    parser.add_argument("--mode", choices=("fetch", "analyze", "all"), default="all",
                        help="Fetch network neighbourhoods, analyze cached data, or do both (default: all).")
    parser.add_argument("--api-key", default=None, help="OpenAlex API key; required for fetch/all modes.")
    parser.add_argument("--email", default=None, help="Contact email supplied to OpenAlex as mailto.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Optional reproducible pilot cap on Step 4 matched pairs; default uses all pairs.")
    parser.add_argument("--max-outbound-refs", type=int, default=30,
                        help="Maximum sampled direct references retained per focal or citing neighbour (default: 30).")
    parser.add_argument("--max-inbound-citers", type=int, default=30,
                        help="Maximum reproducibly sampled citing works retrieved per focal paper (default: 30).")
    parser.add_argument("--betweenness-k", type=int, default=200,
                        help="Number of sampled citing-source nodes for source-target directed betweenness (default: 200).")
    parser.add_argument("--permutations", type=int, default=1000,
                        help="Number of focal-label permutations for the design check (default: 1000).")
    parser.add_argument("--seed", type=int, default=20260908, help="Master random seed (default: 20260908).")
    parser.add_argument("--sleep", type=float, default=0.08, help="Delay between API requests in seconds (default: 0.08).")
    parser.add_argument("--force-refetch", action="store_true", help="Discard cached focal-neighbourhood records and fetch again.")
    return parser.parse_args()


def oa_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.rstrip("/")
    return text.rsplit("/", 1)[-1].upper()


def seeded_subset(values: list[str], limit: int, seed: int, key: str) -> list[str]:
    values = sorted(set(oa_id(v) for v in values if oa_id(v)))
    if len(values) <= limit:
        return values
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    return sorted(rng.sample(values, limit))


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:10], 16) % 2_000_000_000


def api_get(session: requests.Session, url: str, params: dict[str, Any], attempts: int = 6) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=60)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(30.0, 0.8 * (2 ** attempt)))
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(min(30.0, 0.8 * (2 ** attempt)))
    raise RuntimeError(f"OpenAlex request failed after {attempts} attempts: {last_error}")


def base_params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {
        "select": "id,referenced_works,publication_year,type,cited_by_count",
    }
    if args.api_key:
        params["api_key"] = args.api_key.strip()
    if args.email:
        params["mailto"] = args.email.strip()
    return params


def load_pairs(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    path = root / "revision" / "output" / "step4_matched_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Step 4 matched pairs are required: {path}")
    pairs = pd.read_csv(path, low_memory=False)
    required = {"pair_id", "treated_id", "control_id"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"step4_matched_pairs.csv is missing required columns: {sorted(missing)}")
    pairs["treated_id"] = pairs["treated_id"].map(oa_id)
    pairs["control_id"] = pairs["control_id"].map(oa_id)
    pairs = pairs[(pairs["treated_id"] != "") & (pairs["control_id"] != "")].copy()
    pairs = pairs.sort_values("pair_id").reset_index(drop=True)
    if args.max_pairs is not None:
        if args.max_pairs < 20:
            raise ValueError("--max-pairs must be at least 20 for a meaningful pilot network.")
        if args.max_pairs < len(pairs):
            pairs = pairs.sample(args.max_pairs, random_state=args.seed).sort_values("pair_id").reset_index(drop=True)
    return pairs


def load_retracted_ids(root: Path) -> set[str]:
    master = read_master(root)
    candidates = ["revision_openalex_id", "openalex_id", "id"]
    column = next((name for name in candidates if name in master.columns), None)
    if column is None:
        raise ValueError("The frozen corpus lacks an OpenAlex identifier column.")
    return {oa_id(value) for value in master[column].tolist() if oa_id(value)}


def focal_rows(pairs: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, row in pairs.iterrows():
        pair_id = str(row["pair_id"])
        rows.append({"focal_id": str(row["treated_id"]), "focal_group": "retracted", "pair_id": pair_id})
        rows.append({"focal_id": str(row["control_id"]), "focal_group": "matched_control", "pair_id": pair_id})
    return rows


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
                focal_id = oa_id(item.get("focal_id"))
                if focal_id:
                    records[focal_id] = item
            except (ValueError, TypeError):
                continue
    return records


def fetch_one_neighbourhood(session: requests.Session, focal: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    focal_id = focal["focal_id"]
    params = base_params(args)
    focal_record = api_get(session, f"{OPENALEX_URL}/{focal_id}", params)
    raw_refs = focal_record.get("referenced_works") or []
    outbound_refs = seeded_subset(raw_refs, args.max_outbound_refs, args.seed, f"out:{focal_id}")

    inbound_params = base_params(args)
    inbound_params.update({
        "filter": f"cites:{focal_id}",
        "sample": args.max_inbound_citers,
        "seed": stable_seed(args.seed, f"in:{focal_id}"),
    })
    inbound_response = api_get(session, OPENALEX_URL, inbound_params)
    inbound_records: list[dict[str, Any]] = []
    for item in inbound_response.get("results", []):
        citer_id = oa_id(item.get("id"))
        if not citer_id:
            continue
        refs = seeded_subset(item.get("referenced_works") or [], args.max_outbound_refs, args.seed, f"nbr:{citer_id}")
        inbound_records.append({
            "id": citer_id,
            "publication_year": item.get("publication_year"),
            "type": item.get("type"),
            "cited_by_count": item.get("cited_by_count"),
            "referenced_works": refs,
        })
    return {
        "focal_id": focal_id,
        "focal_group": focal["focal_group"],
        "pair_id": focal["pair_id"],
        "focal_metadata": {
            "id": focal_id,
            "publication_year": focal_record.get("publication_year"),
            "type": focal_record.get("type"),
            "cited_by_count": focal_record.get("cited_by_count"),
        },
        "focal_outbound_references": outbound_refs,
        "focal_references_reported_by_api": int(len(raw_refs)),
        "focal_references_retained": int(len(outbound_refs)),
        "inbound_citers_reported_by_api": int(inbound_response.get("meta", {}).get("count", len(inbound_records))),
        "inbound_citers_retained": int(len(inbound_records)),
        "inbound_records": inbound_records,
        "fetch_timestamp": now_utc(),
    }


def fetch_neighbourhoods(focals: list[dict[str, str]], cache_path: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if args.force_refetch and cache_path.exists():
        cache_path.unlink()
    cached = load_cache(cache_path)
    # Retry focal papers whose most recent cached record contains an API error.
    todo = [item for item in focals if item["focal_id"] not in cached or "error" in cached[item["focal_id"]]]
    print(f"Focal papers: {len(focals):,}; cached neighbourhoods: {len(cached):,}; to fetch: {len(todo):,}.")
    if not todo:
        return cached
    session = requests.Session()
    with cache_path.open("a", encoding="utf-8") as handle:
        for number, focal in enumerate(todo, start=1):
            try:
                record = fetch_one_neighbourhood(session, focal, args)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                cached[focal["focal_id"]] = record
            except Exception as exc:  # cached error record supports a resumable retry on later runs
                error_record = {
                    "focal_id": focal["focal_id"], "focal_group": focal["focal_group"], "pair_id": focal["pair_id"],
                    "error": str(exc), "fetch_timestamp": now_utc(),
                }
                handle.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[WARN] {focal['focal_id']}: {exc}")
            if number % 50 == 0 or number == len(todo):
                print(f"Fetched {number:,}/{len(todo):,} focal neighbourhoods.")
            time.sleep(args.sleep)
    return load_cache(cache_path)


def node_group(node_id: str, retracted_ids: set[str], focal_groups: dict[str, str]) -> str:
    if node_id in retracted_ids:
        return "retracted"
    if focal_groups.get(node_id) == "matched_control":
        return "matched_control"
    return "context_not_in_rwdb"


def build_graph(
    cache: dict[str, dict[str, Any]],
    focals: list[dict[str, str]],
    retracted_ids: set[str],
    args: argparse.Namespace,
) -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    focal_groups = {item["focal_id"]: item["focal_group"] for item in focals}
    pair_map = {item["focal_id"]: item["pair_id"] for item in focals}
    graph = nx.DiGraph()
    audit_rows: list[dict[str, Any]] = []

    for focal in focals:
        focal_id = focal["focal_id"]
        record = cache.get(focal_id, {})
        success = "error" not in record and bool(record)
        audit_rows.append({
            "focal_id": focal_id,
            "focal_group": focal["focal_group"],
            "pair_id": focal["pair_id"],
            "fetch_success": success,
            "error": record.get("error", ""),
            "focal_references_reported_by_api": record.get("focal_references_reported_by_api", np.nan),
            "focal_references_retained": record.get("focal_references_retained", np.nan),
            "inbound_citers_reported_by_api": record.get("inbound_citers_reported_by_api", np.nan),
            "inbound_citers_retained": record.get("inbound_citers_retained", np.nan),
        })
        graph.add_node(
            focal_id,
            group=node_group(focal_id, retracted_ids, focal_groups),
            focal=True,
            pair_id=pair_map.get(focal_id, ""),
            publication_year=record.get("focal_metadata", {}).get("publication_year"),
            cited_by_count=record.get("focal_metadata", {}).get("cited_by_count"),
        )
        if not success:
            continue

        for target in record.get("focal_outbound_references", []):
            target_id = oa_id(target)
            if target_id:
                graph.add_edge(focal_id, target_id, edge_origin="focal_outbound")
        for citer in record.get("inbound_records", []):
            citer_id = oa_id(citer.get("id"))
            if not citer_id:
                continue
            graph.add_node(
                citer_id,
                group=node_group(citer_id, retracted_ids, focal_groups),
                focal=citer_id in focal_groups,
                pair_id=pair_map.get(citer_id, ""),
                publication_year=citer.get("publication_year"),
                cited_by_count=citer.get("cited_by_count"),
            )
            # Explicitly preserve the citing-neighbour -> focal relation.
            graph.add_edge(citer_id, focal_id, edge_origin="inbound_to_focal")
            for target in citer.get("referenced_works", []):
                target_id = oa_id(target)
                # The direct citer -> focal edge is already retained above as
                # inbound_to_focal. Do not overwrite its provenance when the
                # focal paper also appears in the full reference list.
                if target_id and target_id != focal_id:
                    graph.add_edge(citer_id, target_id, edge_origin="inbound_outbound")

    node_rows: list[dict[str, Any]] = []
    for node, attrs in graph.nodes(data=True):
        node_rows.append({
            "openalex_id": node,
            "node_group": attrs.get("group", node_group(node, retracted_ids, focal_groups)),
            "is_focal": bool(attrs.get("focal", node in focal_groups)),
            "pair_id": attrs.get("pair_id", pair_map.get(node, "")),
            "publication_year": attrs.get("publication_year", np.nan),
            "cited_by_count": attrs.get("cited_by_count", np.nan),
        })
    nodes = pd.DataFrame(node_rows)

    edge_rows: list[dict[str, str]] = []
    for source, target, attrs in graph.edges(data=True):
        source_group = node_group(source, retracted_ids, focal_groups)
        target_group = node_group(target, retracted_ids, focal_groups)
        edge_rows.append({
            "source": source,
            "target": target,
            "edge_origin": str(attrs.get("edge_origin", "unknown")),
            "source_group": source_group,
            "target_group": target_group,
            "edge_type": f"{source_group}->{target_group}",
        })
    edges = pd.DataFrame(edge_rows)
    audit = pd.DataFrame(audit_rows)
    return graph, nodes, edges, audit


def approximate_metrics(
    graph: nx.DiGraph,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"Computing metrics on {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges …")
    n = graph.number_of_nodes()
    if n == 0:
        raise RuntimeError("The reconstructed graph has no nodes.")
    pagerank = nx.pagerank(graph, alpha=0.85, max_iter=200, tol=1e-7)

    # Generic random-source betweenness almost never places a focal paper on a
    # sampled directed shortest path in a large one-hop graph. Instead, compute
    # source-target betweenness on paths from sampled works that cite a focal
    # paper to works cited by that focal paper. This operationalizes the focal
    # paper as a possible directed bridge between its local citing and cited
    # neighbourhoods while retaining the full reconstructed graph for paths.
    source_pool = edge_table.loc[edge_table["edge_origin"].eq("inbound_to_focal"), "source"].astype(str).tolist()
    target_pool = edge_table.loc[edge_table["edge_origin"].eq("focal_outbound"), "target"].astype(str).tolist()
    sources = seeded_subset(source_pool, min(args.betweenness_k, len(set(source_pool))), args.seed, "path_sources")
    targets = sorted(set(target_pool))
    if not sources or not targets:
        raise RuntimeError("The reconstructed network lacks the citing-source or focal-reference target sets required for focal-path betweenness.")
    print(f"Computing source-target directed betweenness with {len(sources):,} citing sources and {len(targets):,} focal-reference targets …")
    betweenness = nx.betweenness_centrality_subset(graph, sources=sources, targets=targets, normalized=True)
    metric_meta = {
        "definition": "Approximate normalized directed source-target betweenness centrality.",
        "source_set": "Reproducibly sampled works with an observed citation to a focal retracted or matched-control paper.",
        "target_set": "All retained references of focal retracted or matched-control papers.",
        "n_sources": len(sources),
        "n_targets": len(targets),
        "source_sample_seed": args.seed,
    }
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    result = node_table.copy()
    result["in_degree"] = result["openalex_id"].map(in_degree).fillna(0).astype(int)
    result["out_degree"] = result["openalex_id"].map(out_degree).fillna(0).astype(int)
    result["pagerank"] = result["openalex_id"].map(pagerank).fillna(0.0)
    result["directed_betweenness"] = result["openalex_id"].map(betweenness).fillna(0.0)
    return result, metric_meta


def permutation_check(metrics: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    focal = metrics[metrics["is_focal"] & metrics["node_group"].isin(["retracted", "matched_control"])].copy()
    focal["treatment"] = focal["node_group"].eq("retracted").astype(int)
    if focal["treatment"].sum() < 20 or (1 - focal["treatment"]).sum() < 20:
        raise RuntimeError("Too few successfully reconstructed focal papers for the permutation check.")
    focal["log_betweenness"] = np.log1p(focal["directed_betweenness"])
    observed = float(focal.loc[focal["treatment"].eq(1), "log_betweenness"].mean() - focal.loc[focal["treatment"].eq(0), "log_betweenness"].mean())
    labels = focal["treatment"].to_numpy(copy=True)
    values = focal["log_betweenness"].to_numpy(dtype=float)
    rng = np.random.default_rng(args.seed)
    null_values: list[float] = []
    for iteration in range(1, args.permutations + 1):
        shuffled = rng.permutation(labels)
        null_values.append(float(values[shuffled == 1].mean() - values[shuffled == 0].mean()))
    null = np.asarray(null_values, dtype=float)
    p_value = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))
    null_df = pd.DataFrame({"permutation": np.arange(1, len(null) + 1), "null_mean_difference_log1p_betweenness": null})
    summary = {
        "outcome": "mean difference in log(1 + directed betweenness), retracted minus matched control focal papers",
        "n_retracted_focals": int(focal["treatment"].sum()),
        "n_matched_control_focals": int((1 - focal["treatment"]).sum()),
        "observed_difference": observed,
        "permutations": int(args.permutations),
        "two_sided_permutation_p_value": p_value,
        "interpretation": "Unadjusted label-permutation design check only; Step 7 reports the adjusted primary comparison.",
    }
    return null_df, summary


def make_figures(edges: pd.DataFrame, null_df: pd.DataFrame, perm_summary: dict[str, Any], output_dir: Path) -> None:
    composition = (edges.groupby(["edge_origin", "edge_type"], as_index=False).size()
                   .rename(columns={"size": "n_edges"}))
    pivot = composition.pivot(index="edge_origin", columns="edge_type", values="n_edges").fillna(0)
    # Place non-retracted-to-non-retracted edges first in the legend when present.
    preferred = [
        "context_not_in_rwdb->context_not_in_rwdb",
        "context_not_in_rwdb->retracted",
        "retracted->context_not_in_rwdb",
        "retracted->retracted",
        "matched_control->context_not_in_rwdb",
        "context_not_in_rwdb->matched_control",
    ]
    ordered = [col for col in preferred if col in pivot.columns] + [col for col in pivot.columns if col not in preferred]
    pivot = pivot[ordered]
    fig, ax = plt.subplots(figsize=(11, 6.3))
    bottom = np.zeros(len(pivot), dtype=float)
    palette = plt.get_cmap("tab20")
    for number, col in enumerate(pivot.columns):
        values = pivot[col].to_numpy(dtype=float)
        ax.bar(pivot.index, values, bottom=bottom, label=col.replace("context_not_in_rwdb", "not in RWDB corpus").replace("matched_control", "matched control"), color=palette(number), edgecolor="white", linewidth=0.45)
        bottom += values
    ax.set_yscale("log")
    ax.set_ylabel("Number of directed edges (log scale)")
    ax.set_xlabel("Edge origin in one-hop focal neighbourhood reconstruction")
    ax.set_title("Edge Composition in the Balanced Citation-Network Reconstruction")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_dual_figure(fig, output_dir / "fig_step6_edge_composition")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.hist(null_df["null_mean_difference_log1p_betweenness"], bins=38, color="#9ECAE1", edgecolor="white")
    observed = float(perm_summary["observed_difference"])
    ax.axvline(observed, color="#B2182B", linewidth=2.4, label=f"Observed focal-label difference = {observed:.4f}")
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Retracted minus matched-control mean log(1 + directed betweenness)")
    ax.set_ylabel("Number of label permutations")
    ax.set_title("Label-Permutation Design Check for Focal Directed Betweenness")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_dual_figure(fig, output_dir / "fig_step6_label_permutation")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = project_root()
    dirs = revision_dirs(root)
    output_dir = dirs["output"]
    data_dir = dirs["data"]
    cache_path = data_dir / "step6_focal_neighborhoods.jsonl"

    pairs = load_pairs(root, args)
    retracted_ids = load_retracted_ids(root)
    focals = focal_rows(pairs)
    pd.DataFrame(focals).to_csv(output_dir / "step6_focal_sample.csv", index=False)
    print(f"Step 6 focal design: {len(pairs):,} matched pairs; {len(focals):,} focal papers.")

    if args.mode in {"fetch", "all"}:
        if not args.api_key or not str(args.api_key).strip():
            raise ValueError("--api-key is required for --mode fetch or --mode all.")
        cache = fetch_neighbourhoods(focals, cache_path, args)
    else:
        cache = load_cache(cache_path)
        if not cache:
            raise FileNotFoundError(f"No Step 6 cache found: {cache_path}. Run with --mode fetch or --mode all first.")

    if args.mode == "fetch":
        print("Step 6 fetch complete. Rerun with --mode analyze to build the graph and diagnostics.")
        return

    graph, nodes, edges, audit = build_graph(cache, focals, retracted_ids, args)
    if edges.empty:
        raise RuntimeError("No edges were reconstructed. Inspect step6_focal_neighborhoods.jsonl for API errors.")
    metrics, metric_meta = approximate_metrics(graph, nodes, edges, args)
    null_df, permutation_summary = permutation_check(metrics, args)

    edge_composition = (edges.groupby(["edge_origin", "edge_type"], as_index=False).size()
                        .rename(columns={"size": "n_edges"}))
    edge_composition["share_all_edges"] = edge_composition["n_edges"] / len(edges)
    node_composition = (metrics.groupby(["node_group", "is_focal"], as_index=False).size()
                        .rename(columns={"size": "n_nodes"}))
    fetch_success = audit["fetch_success"].astype(bool)
    context_to_context_edges = int(edge_composition.loc[edge_composition["edge_type"].eq("context_not_in_rwdb->context_not_in_rwdb"), "n_edges"].sum())

    audit.to_csv(output_dir / "step6_neighborhood_fetch_audit.csv", index=False)
    edge_composition.to_csv(output_dir / "step6_edge_type_composition.csv", index=False)
    node_composition.to_csv(output_dir / "step6_node_type_composition.csv", index=False)
    metrics[metrics["is_focal"]].to_csv(output_dir / "step6_focal_node_metrics.csv", index=False)
    null_df.to_csv(output_dir / "step6_label_permutation_null.csv", index=False)
    edges.to_csv(output_dir / "step6_expanded_network_edges.csv.gz", index=False, compression="gzip")
    metrics.to_csv(output_dir / "step6_expanded_network_nodes.csv.gz", index=False, compression="gzip")

    summary = {
        "created_at": now_utc(),
        "network_design": {
            "description": "Balanced one-hop citation-neighbourhood reconstruction around Step 4 matched retracted/control focal pairs.",
            "focal_pairs_requested": int(len(pairs)),
            "focal_papers_requested": int(len(focals)),
            "focal_papers_with_successful_fetch": int(fetch_success.sum()),
            "retracted_identifier_reference_set": int(len(retracted_ids)),
            "max_outbound_references_per_source": int(args.max_outbound_refs),
            "max_inbound_citers_per_focal": int(args.max_inbound_citers),
            "edge_origins": ["focal_outbound", "inbound_to_focal", "inbound_outbound"],
            "omissions": "Edges among non-focal context papers that are not represented as selected outgoing references of a sampled citing neighbour are not observed; this is a sampled one-hop network, not the complete OpenAlex graph.",
        },
        "network_size": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "context_not_in_rwdb_to_context_not_in_rwdb_edges": context_to_context_edges,
            "context_not_in_rwdb_to_context_not_in_rwdb_edge_share": float(context_to_context_edges / graph.number_of_edges()),
        },
        "metric_computation": {
            "pagerank": "directed PageRank, alpha=0.85",
            "directed_betweenness": metric_meta,
        },
        "label_permutation_design_check": permutation_summary,
        "outputs": [
            "step6_focal_sample.csv", "step6_neighborhood_fetch_audit.csv", "step6_edge_type_composition.csv",
            "step6_node_type_composition.csv", "step6_focal_node_metrics.csv", "step6_label_permutation_null.csv",
            "step6_expanded_network_edges.csv.gz", "step6_expanded_network_nodes.csv.gz",
            "fig_step6_edge_composition.png/.pdf", "fig_step6_label_permutation.png/.pdf",
        ],
    }
    write_json(output_dir / "step6_expanded_network_summary.json", summary)
    make_figures(edges, null_df, permutation_summary, output_dir)
    print("Step 6 complete. Inspect network coverage and edge-composition outputs before any H1/H2 inference.")


if __name__ == "__main__":
    main()
