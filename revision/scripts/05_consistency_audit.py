#!/usr/bin/env python3
"""Step 1.7: audit data/output consistency and create a claim register template."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from common import project_root, read_master, revision_dirs


def locate_tex(root: Path, supplied: str | None) -> Path | None:
    if supplied:
        p = Path(supplied).expanduser().resolve()
        return p if p.exists() else None
    candidates = [root / "manuscript_revision.tex", root / "paper.tex", root / "manuscript" / "manuscript_revision.tex"]
    return next((p for p in candidates if p.exists()), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", help="Optional path to manuscript_revision.tex")
    args = parser.parse_args()

    root = project_root()
    dirs = revision_dirs(root)
    df = read_master(root)
    out = dirs["output"]
    checks = []

    expected = [
        out / "revision_manifest.json",
        out / "data_quality_audit.csv",
        out / "sample_flow.csv",
        out / "reason_mapping_review.csv",
        out / "reason_primary_category_counts.csv",
        out / "reason_multilabel_prevalence_supplement.csv",
        out / "fig_retraction_reasons_primary.pdf",
        out / "fig_retraction_reasons_primary.png",
    ]
    for p in expected:
        checks.append({"domain": "required_output", "item": p.name, "status": "PASS" if p.exists() else "FAIL", "detail": str(p)})

    flow_path = out / "sample_flow.csv"
    if flow_path.exists():
        flow = pd.read_csv(flow_path)
        master_n = int(flow.iloc[0]["n"])
        checks.append({"domain": "counts", "item": "frozen_master_matches_sample_flow", "status": "PASS" if master_n == len(df) else "FAIL", "detail": f"master={len(df):,}; sample_flow={master_n:,}"})

    primary_path = out / "reason_primary_category_counts.csv"
    if primary_path.exists():
        primary = pd.read_csv(primary_path)
        denom_ok = primary["denominator"].eq(len(df)).all()
        count_ok = int(primary["n_papers"].sum()) == len(df)
        checks.append({"domain": "reason_denominator", "item": "all_primary_categories_use_frozen_master_denominator", "status": "PASS" if denom_ok else "FAIL", "detail": f"expected denominator={len(df):,}"})
        checks.append({"domain": "reason_primary_assignment", "item": "one_primary_category_per_record", "status": "PASS" if count_ok else "FAIL", "detail": f"primary-count sum={int(primary['n_papers'].sum()):,}; master={len(df):,}"})
        checks.append({"domain": "reason_labels", "item": "six_primary_categories_present", "status": "PASS" if primary["primary_reason_category"].nunique() == 6 else "CHECK", "detail": f"found={primary['primary_reason_category'].nunique()}"})

    tex = locate_tex(root, args.tex)
    if tex:
        text = tex.read_text(encoding="utf-8", errors="replace")
        graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
        new_figure_called = any("fig_retraction_reasons_primary" in g for g in graphics)
        checks.append({"domain": "manuscript_asset", "item": "corrected_primary_reason_figure_called", "status": "PASS" if new_figure_called else "CHECK", "detail": f"tex={tex}"})
        checks.append({"domain": "manuscript_asset", "item": "figure_calls_found", "status": "PASS", "detail": f"n={len(graphics)}"})
    else:
        checks.append({"domain": "manuscript_asset", "item": "manuscript_tex_located", "status": "CHECK", "detail": "Pass --tex /path/to/manuscript_revision.tex to audit figure calls."})

    audit = pd.DataFrame(checks)
    audit.to_csv(out / "consistency_audit.csv", index=False)
    lines = ["# Step 1 consistency audit", "", f"Frozen master rows: **{len(df):,}**", "", "| Domain | Item | Status | Detail |", "|---|---|---|---|"]
    for _, r in audit.iterrows():
        lines.append(f"| {r['domain']} | {r['item']} | {r['status']} | {r['detail']} |")
    (out / "consistency_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    claims = pd.DataFrame([
        {"claim_id": "C01", "manuscript_location": "Abstract / §3.1", "claim": "Size of matched frozen corpus", "source_file": "sample_flow.csv", "source_field": "row 1 / n", "verified_value": len(df), "status": "TO REVIEW"},
        {"claim_id": "C02", "manuscript_location": "§3.1", "claim": "Valid chronological date cohort", "source_file": "sample_flow.csv", "source_field": "row 3 / n", "verified_value": "", "status": "TO REVIEW"},
        {"claim_id": "C03", "manuscript_location": "§3.2 / Figure 4", "claim": "Reason-category percentages", "source_file": "reason_category_multilabel_prevalence.csv", "source_field": "all rows", "verified_value": "", "status": "TO REVIEW"},
        {"claim_id": "C04", "manuscript_location": "Figure 4 caption", "claim": "Multi-label denominator statement", "source_file": "reason_category_multilabel_prevalence.csv", "source_field": "multi_label_note", "verified_value": "Required", "status": "TO INSERT"},
    ])
    claims.to_csv(out / "manuscript_claim_register.csv", index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
