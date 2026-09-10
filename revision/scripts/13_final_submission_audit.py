#!/usr/bin/env python3
r"""Step 13: final revision and submission audit for the JOI manuscript.

This script never alters the manuscript, bibliography, tables, or figures. It
recursively reads LaTeX \input files and checks the exact files supplied via
--manuscript and --bib. Use explicit paths so that duplicate manuscript or
bibliography copies elsewhere in the project cannot be selected accidentally.

Run from the computations directory:
  python revision/scripts/13_final_submission_audit.py \
    --manuscript ../revision/manuscript_revision.tex \
    --bib ../revision/paper.bib

It writes final_submission_audit.csv, final_submission_audit.json, and
final_submission_audit.md to revision/output/ by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

GRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
INPUTS = re.compile(r"\\input\{([^}]+)\}")
LABELS = re.compile(r"\\label\{([^}]+)\}")
REFS = re.compile(r"\\(?:ref|autoref|pageref)\{([^}]+)\}")
CITATIONS = re.compile(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}")
BIBKEYS = re.compile(r"@\w+\s*\{\s*([^,\s]+)")

REQUIRED_AUTHOR_STRINGS = (
    r"\author[1]{Moses Boudourides}",
    r"\author[2]{Emanuela Chiriac}",
    "School of Professional Studies, Northwestern University, Evanston, IL, USA.",
    "Université du Québec en Outaouais, Gatineau, QC, Canada.",
    "moses.boudourides@northwestern.edu",
    "emanuela.chiriac@uqo.ca",
    r"\date{}",
)

SUPERSEDED_PHRASES = {
    "serial offenders": "Use the neutral term 'authors with multiple retracted publications'.",
    "co-authorship ripple effect": "This superseded caption must not remain.",
    "citation trajectories of papers that cite retracted works": "This superseded caption must not remain.",
    "the confirmation of h1": "H1 is not supported in the revised matched analysis.",
    "the confirmation of h2": "H2 is not supported in the revised adjusted analysis.",
    "concept drift": "Do not claim temporal concept drift from the revised ML evaluation.",
    "xgboost": "The final Step 9 workflow evaluates Gradient Boosting, not XGBoost.",
    "shap": "The final Step 9 workflow uses permutation importance, not SHAP.",
    "reflects the intensification of paper mill investigations": "Use neutral descriptive wording; the figure does not identify a cause of the acceleration.",
    "reflects the large-scale retraction campaigns": "Use neutral descriptive wording; journal counts do not establish a causal mechanism.",
    "landmark in the history of research misconduct": "This evaluative language is unnecessary for the descriptive author table.",
    "misconduct is discovered only through retrospective audits or whistleblower disclosures": "The retraction-delay distribution alone does not establish this explanation.",
}

REQUIRED_MAIN_ASSETS = {
    "figures/fig_retractions_per_year.png",
    "figures/fig_retraction_reasons_primary.png",
    "figures/fig_step4_matched_event_study.png",
    "figures/fig_step5_indicator_window_sensitivity.png",
    "tables/table_sample_flow.tex",
    "tables/table_step4_matching_balance.tex",
    "tables/table_step4_event_study.tex",
    "tables/table_step5_indicator_sensitivity.tex",
    "tables/table_step6_network_reconstruction.tex",
    "tables/table_step7_h1_boundary.tex",
    "tables/table_step8_h2_brokerage_persistence.tex",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a final LaTeX submission package without modifying it.")
    parser.add_argument("--manuscript", required=True, help="Exact path to manuscript_revision.tex")
    parser.add_argument("--bib", required=True, help="Exact path to paper.bib")
    parser.add_argument("--output-dir", default="", help="Directory for audit files; default is <manuscript>/../output")
    return parser.parse_args()


def resolve_tex(base: Path, raw: str) -> Path:
    path = (base / raw).resolve()
    return path if path.suffix == ".tex" else path.with_suffix(".tex")


def collect_tex(path: Path, seen: set[Path] | None = None) -> tuple[str, list[Path], list[dict[str, str]]]:
    seen = set() if seen is None else seen
    records: list[dict[str, str]] = []
    if path in seen:
        return "", [], records
    seen.add(path)
    if not path.exists():
        records.append({"kind": "input", "item": str(path), "status": "ERROR", "detail": "Missing LaTeX input file."})
        return "", [], records
    text = path.read_text(encoding="utf-8", errors="replace")
    combined = "\n" + text
    sources = [path]
    for raw in INPUTS.findall(text):
        child = resolve_tex(path.parent, raw)
        child_text, child_sources, child_records = collect_tex(child, seen)
        combined += "\n" + child_text
        sources.extend(child_sources)
        records.extend(child_records)
    return combined, sources, records


def asset_exists(base: Path, raw: str) -> bool:
    path = (base / raw).resolve()
    if path.exists():
        return True
    if not path.suffix:
        return any(path.with_suffix(ext).exists() for ext in (".pdf", ".png", ".jpg", ".jpeg"))
    return False


def add(records: list[dict[str, str]], kind: str, item: str, status: str, detail: str) -> None:
    records.append({"kind": kind, "item": item, "status": status, "detail": detail})


def main() -> None:
    args = parse_args()
    manuscript = Path(args.manuscript).expanduser().resolve()
    bib = Path(args.bib).expanduser().resolve()
    if not manuscript.is_file():
        raise FileNotFoundError(f"Manuscript not found: {manuscript}")
    if not bib.is_file():
        raise FileNotFoundError(f"Bibliography not found: {bib}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else manuscript.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    text, sources, records = collect_tex(manuscript)
    base = manuscript.parent

    for asset in GRAPHICS.findall(text):
        if asset_exists(base, asset):
            add(records, "figure", asset, "PASS", "Referenced graphic exists.")
        else:
            add(records, "figure", asset, "ERROR", "Referenced graphic does not exist.")

    labels = LABELS.findall(text)
    refs = REFS.findall(text)
    for label, count in sorted(Counter(labels).items()):
        if count > 1:
            add(records, "label", label, "ERROR", f"Duplicate label appears {count} times.")
    for label in sorted(set(refs) - set(labels)):
        add(records, "reference", label, "ERROR", "Undefined LaTeX cross-reference.")
    for label in sorted(set(labels) - set(refs)):
        if label.startswith(("fig:", "tab:")):
            add(records, "label", label, "WARN", "Display label is not cited in manuscript text.")

    bib_text = bib.read_text(encoding="utf-8", errors="replace")
    bib_keys = set(BIBKEYS.findall(bib_text))
    cited = {key.strip() for group in CITATIONS.findall(text) for key in group.split(",") if key.strip()}
    for key in sorted(cited - bib_keys):
        add(records, "citation", key, "ERROR", "Cited bibliography key is absent from the supplied .bib file.")
    for key in sorted(bib_keys - cited):
        add(records, "bibliography", key, "WARN", "Bibliography entry is not cited; this is normally harmless with BibTeX styles that print cited entries only.")

    flat_text = re.sub(r"\s+", " ", text.lower())
    for phrase, detail in SUPERSEDED_PHRASES.items():
        if phrase in flat_text:
            add(records, "wording", phrase, "ERROR", detail)

    for item in REQUIRED_AUTHOR_STRINGS:
        if item not in text:
            add(records, "author_block", item, "ERROR", "Required final author-block element is absent.")

    if "109 distinct raw RWDB reason codes" not in text:
        add(records, "taxonomy", "109 distinct raw RWDB reason codes", "WARN", "Confirm the manuscript describes the approved mapping as 109 distinct raw codes.")
    if r"\citep{brandes2001faster}" not in text:
        add(records, "citation", "brandes2001faster", "ERROR", "The brokerage method must cite Brandes (2001).")

    referenced_items = set(GRAPHICS.findall(text)) | set(INPUTS.findall(manuscript.read_text(encoding="utf-8", errors="replace")))
    for item in sorted(REQUIRED_MAIN_ASSETS - referenced_items):
        add(records, "required_asset", item, "WARN", "Expected final main-text asset is not referenced.")

    fieldnames = ["kind", "item", "status", "detail"]
    csv_path = output_dir / "final_submission_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    counts = Counter(record["status"] for record in records)
    summary = {
        "manuscript": str(manuscript),
        "bibliography": str(bib),
        "tex_files_read": [str(path) for path in sources],
        "errors": counts["ERROR"],
        "warnings": counts["WARN"],
        "passes": counts["PASS"],
        "csv": str(csv_path),
    }
    json_path = output_dir / "final_submission_audit.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Final submission audit", "", f"- Manuscript: `{manuscript}`", f"- Bibliography: `{bib}`", f"- Errors: **{counts['ERROR']}**", f"- Warnings: **{counts['WARN']}**", f"- Passes: **{counts['PASS']}**", ""]
    if records:
        for status in ("ERROR", "WARN", "PASS"):
            subset = [row for row in records if row["status"] == status]
            if not subset:
                continue
            lines.extend([f"## {status}", ""])
            lines.extend(f"- **{row['kind']}** `{row['item']}` — {row['detail']}" for row in subset)
            lines.append("")
    else:
        lines.extend(["## Result", "", "No audit findings.", ""])
    md_path = output_dir / "final_submission_audit.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Final submission audit complete: {counts['ERROR']} errors, {counts['WARN']} warnings, {counts['PASS']} passes.")
    print(f"Audit report: {md_path}")


if __name__ == "__main__":
    main()
