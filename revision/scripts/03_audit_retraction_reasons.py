#!/usr/bin/env python3
"""Step 1.5: audit the RWDB multi-label reason field and broad-category mapping."""
from __future__ import annotations

import re

import pandas as pd

from common import find_column, latex_escape, project_root, read_master, revision_dirs, split_reason_field


def load_mapping(path):
    mapping = pd.read_csv(path)
    required = {"broad_category", "pattern", "active"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping file lacks columns: {sorted(missing)}")
    mapping = mapping[mapping["active"].astype(str).str.upper().eq("TRUE")].copy()
    return mapping


def categories_for_reason(reason: str, mapping: pd.DataFrame) -> list[str]:
    # Fallback is applied only when no non-fallback pattern matches.
    specific = mapping[mapping["broad_category"].ne("Other/Unclear")]
    hits = [row.broad_category for row in specific.itertuples() if re.search(row.pattern, reason, flags=re.I)]
    return hits if hits else ["Other/Unclear"]


def latex_mapping_table(mapping: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Broad Retraction Watch reason categories used in the revised analysis. The mapping is multi-label: one paper may be assigned to more than one category.}",
        r"\label{tab:reason_mapping}",
        r"\begin{tabular}{p{0.28\linewidth}p{0.62\linewidth}}",
        r"\toprule",
        r"Broad category & Mapping rule \\",
        r"\midrule",
    ]
    for _, r in mapping.iterrows():
        if r["broad_category"] == "Other/Unclear":
            rule = "Fallback when no preceding rule matches."
        else:
            rule = r.get("mapping_rationale", r["pattern"])
        lines.append(f"{latex_escape(r['broad_category'])} & {latex_escape(rule)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    root = project_root()
    dirs = revision_dirs(root)
    df = read_master(root)
    mapping_path = dirs["config"] / "reason_mapping.csv"
    if not mapping_path.exists():
        # Allows the package to be used without manually moving config first.
        fallback = root / "revision" / "config" / "reason_mapping.csv"
        raise FileNotFoundError(f"Missing editable mapping file: {fallback}")
    mapping = load_mapping(mapping_path)

    reason_col = find_column(df.columns, ["Reason", "reason", "RetractionReason"])
    if not reason_col:
        raise KeyError("No RWDB reason field found. Expected Reason, reason, or RetractionReason.")

    record_rows, raw_rows = [], []
    for idx, value in df[reason_col].items():
        raw_reasons = split_reason_field(value)
        categories = []
        for raw in raw_reasons:
            assigned = categories_for_reason(raw, mapping)
            categories.extend(assigned)
            for cat in assigned:
                raw_rows.append({"raw_reason": raw, "broad_category": cat, "record_index": idx})
        categories = sorted(set(categories))
        record_rows.append({
            "record_index": idx,
            "raw_reason_string": value,
            "n_raw_reason_codes": len(raw_reasons),
            "broad_categories": "; ".join(categories),
            "n_broad_categories": len(categories),
            "has_no_reason": len(raw_reasons) == 0,
        })

    record_map = pd.DataFrame(record_rows)
    raw_map = pd.DataFrame(raw_rows)
    record_map.to_csv(dirs["output"] / "reason_record_assignments.csv.gz", index=False, compression="gzip")

    n = len(df)
    prevalence = (
        raw_map.drop_duplicates(["record_index", "broad_category"])
        .groupby("broad_category")["record_index"].nunique()
        .rename("n_papers")
        .reset_index()
    )
    all_categories = mapping["broad_category"].drop_duplicates().tolist()
    prevalence = pd.DataFrame({"broad_category": all_categories}).merge(prevalence, how="left", on="broad_category").fillna({"n_papers": 0})
    prevalence["n_papers"] = prevalence["n_papers"].astype(int)
    prevalence["percent_of_all_retracted_papers"] = 100 * prevalence["n_papers"] / n
    prevalence["denominator"] = n
    prevalence["multi_label_note"] = "Percentages need not sum to 100 because one paper may have multiple reasons."
    prevalence = prevalence.sort_values("n_papers", ascending=False)
    prevalence.to_csv(dirs["output"] / "reason_category_multilabel_prevalence.csv", index=False)

    raw_summary = (
        raw_map.groupby(["raw_reason", "broad_category"])["record_index"].nunique()
        .rename("n_papers")
        .reset_index()
        .sort_values(["n_papers", "raw_reason"], ascending=[False, True])
    )
    raw_summary.to_csv(dirs["output"] / "reason_code_mapping_audit.csv", index=False)

    no_reason = record_map[record_map["has_no_reason"]]
    no_reason.to_csv(dirs["output"] / "reason_missing_or_blank.csv", index=False)
    overlap = record_map[record_map["n_broad_categories"] > 1]
    overlap.to_csv(dirs["output"] / "reason_multilabel_overlap.csv", index=False)
    (dirs["output"] / "table_reason_mapping.tex").write_text(latex_mapping_table(mapping), encoding="utf-8")

    print(prevalence.to_string(index=False))
    print(f"Papers with blank reason field: {len(no_reason):,}")
    print(f"Papers assigned to >1 broad category: {len(overlap):,}")


if __name__ == "__main__":
    main()
