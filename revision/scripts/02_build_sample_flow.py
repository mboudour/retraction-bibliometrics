#!/usr/bin/env python3
"""Step 1.4: document every analysis cohort and reconcile sample sizes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import (
    find_column, latex_escape, parse_dates, project_root, read_master,
    revision_dirs, source_paths,
)


def ids_in_file(path: Path, candidates: list[str]) -> set[str]:
    if not path.exists():
        return set()
    header = pd.read_csv(path, nrows=0)
    col = find_column(header.columns, candidates)
    if not col:
        return set()
    values = pd.read_csv(path, usecols=[col], low_memory=False)[col]
    return set(values.dropna().astype(str).str.strip())


def latex_table(flow: pd.DataFrame) -> str:
    newline = r"\\"
    rows = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Analytical sample flow. Counts are generated from the frozen revision dataset; later analytical cohorts are reported only when the required source file is available.}",
        r"\label{tab:sample_flow}",
        r"\begin{tabular}{p{0.62\linewidth}r}",
        r"\toprule",
        "Cohort & $N$ " + newline,
        r"\midrule",
    ]
    for _, row in flow.iterrows():
        rows.append(f"{latex_escape(row['cohort'])} & {int(row['n']):,} " + newline)
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(rows)


def main() -> None:
    root = project_root()
    dirs = revision_dirs(root)
    paths = source_paths(root)
    df = read_master(root)
    n = len(df)

    pub = parse_dates(df["revision_publication_date"])
    ret = parse_dates(df["revision_retraction_date"])
    valid_dates = pub.notna() & ret.notna() & (pub <= ret)
    valid_id = df["revision_openalex_id"].notna()

    flow = [
        {"order": 1, "cohort": "Retraction Watch records matched to OpenAlex (frozen master corpus)", "n": n, "rule": "All rows retained by Step 1.2"},
        {"order": 2, "cohort": "Records with an OpenAlex work identifier", "n": int(valid_id.sum()), "rule": "revision_openalex_id is non-missing"},
        {"order": 3, "cohort": "Records with valid chronological publication and retraction dates", "n": int(valid_dates.sum()), "rule": "both dates parsed and publication date <= retraction date"},
    ]

    node_ids = ids_in_file(paths["node_metrics"], ["openalex_id", "node_id", "id", "work_id"])
    if node_ids:
        networked = valid_id & df["revision_openalex_id"].astype(str).isin(node_ids)
        flow.append({"order": 4, "cohort": "Retracted works represented in the original structural-network node file", "n": int(networked.sum()), "rule": f"identifier present in {paths['node_metrics'].name}"})
    else:
        networked = pd.Series(False, index=df.index)
        flow.append({"order": 4, "cohort": "Retracted works represented in the original structural-network node file", "n": 0, "rule": "Node file absent or no identifier column detected; not calculated"})

    has_series = df.get("counts_by_year_json", pd.Series(pd.NA, index=df.index)).fillna("").astype(str).str.strip().ne("")
    has_series &= ~df.get("counts_by_year_json", pd.Series("", index=df.index)).fillna("").astype(str).isin(["[]", "{}", "nan"])
    flow.append({"order": 5, "cohort": "Records with a non-empty OpenAlex annual citation series", "n": int(has_series.sum()), "rule": "counts_by_year_json is populated and not an empty list/object"})

    citation_ids = ids_in_file(paths["citation_long"], ["openalex_id", "work_id", "paper_id", "id"])
    if citation_ids:
        citation_long = valid_id & df["revision_openalex_id"].astype(str).isin(citation_ids)
        flow.append({"order": 6, "cohort": "Records represented in the original long-format citation file", "n": int(citation_long.sum()), "rule": f"identifier present in {paths['citation_long'].name}"})

    flow_df = pd.DataFrame(flow)
    flow_df.to_csv(dirs["output"] / "sample_flow.csv", index=False)
    flow_df.to_csv(dirs["output"] / "sample_flow_definitions.csv", index=False)
    (dirs["output"] / "table_sample_flow.tex").write_text(latex_table(flow_df), encoding="utf-8")

    # Transparent record-level eligibility flags for later event-study and network scripts.
    eligibility = df[["revision_openalex_id", "revision_doi", "revision_publication_date", "revision_retraction_date"]].copy()
    eligibility["valid_openalex_id"] = valid_id
    eligibility["valid_chronology"] = valid_dates
    eligibility["has_annual_citation_series"] = has_series
    eligibility["in_original_network"] = networked
    eligibility.to_csv(dirs["output"] / "record_eligibility_flags.csv.gz", index=False, compression="gzip")

    print(flow_df[["order", "cohort", "n"]].to_string(index=False))
    print(f"Wrote sample-flow outputs to {dirs['output']}")


if __name__ == "__main__":
    main()
