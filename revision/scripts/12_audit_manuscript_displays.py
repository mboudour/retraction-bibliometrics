#!/usr/bin/env python3
r"""Step 10: audit a revised LaTeX manuscript's figures, tables, labels, and claims.

This script does not alter the manuscript. It recursively follows \input{...}
files, checks that all referenced assets exist, finds undefined or duplicate
labels, identifies orphaned display assets, and flags known superseded wording
and display filenames from the submitted version.

Run from project root, for example:
  python revision/scripts/12_audit_manuscript_displays.py \
      --manuscript "/path/to/manuscript_revision.tex"
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import now_utc, project_root, revision_dirs, write_json  # noqa: E402

GRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
INPUTS = re.compile(r"\\input\{([^}]+)\}")
LABELS = re.compile(r"\\label\{([^}]+)\}")
REFS = re.compile(r"\\(?:ref|autoref|pageref)\{([^}]+)\}")
CITATIONS = re.compile(r"\\(?:cite|citet|citep|citeauthor|citeyear)(?:\[[^\]]*\])?\{([^}]+)\}")
FIGURE_ENV = re.compile(r"\\begin\{figure\*?\}.*?\\caption\{(.*?)\}.*?\\end\{figure\*?\}", re.DOTALL)
TABLE_ENV = re.compile(r"\\begin\{table\*?\}.*?\\caption\{(.*?)\}.*?\\end\{table\*?\}", re.DOTALL)

SUPERSEDED = {
    "Serial Offenders": "Replace with 'authors with multiple retracted publications'.",
    "Co-authorship ripple effect": "The displayed figure is a distribution of retracted papers per author; replace this caption.",
    "citation trajectories of papers that cite retracted works": "The displayed figure is not a citation-trajectory figure; replace this caption.",
    "XGBoost": "The revised Step 9 workflow evaluates Gradient Boosting, not XGBoost.",
    "SHAP": "Use model-agnostic permutation importance; do not claim SHAP results.",
    "fig_event_study_plot": "Replace with fig_step4_matched_event_study.",
    "fig_event_study_normalised": "Remove; the matched-control event-study figure replaces it.",
    "fig_feature_importance": "Use the revised Step 9 figure only in the supplement, if retained.",
    "fig_roc_curves": "Replace with fig_step9_protocol_auc.",
    "fig_degree_distribution": "Remove; Step 7 replaces the prior unadjusted degree/brokerage analysis.",
    "fig_h2_propagation_analysis": "Remove; Step 8 replaces the prior H2 analysis.",
    "fig_ego_network": "Remove from main text; retain only as a supplementary design illustration if desired.",
    "The confirmation of H1": "H1 is not supported in the revised matched analysis.",
    "The confirmation of H2": "H2 is not supported in the revised adjusted analysis.",
}

REQUIRED_MAIN_ASSETS = {
    "figures/fig_retractions_per_year.png",
    "figures/fig_top_countries.png",
    "figures/fig_top_institutions.png",
    "figures/fig_retraction_reasons_primary.png",
    "figures/fig_step4_matched_event_study.png",
    "figures/fig_step5_indicator_window_sensitivity.png",
    "figures/fig_coauthor_ripple.png",
    "figures/fig_step9_protocol_auc.png",
    "tables/table_sample_flow.tex",
    "tables/table_step6_network_reconstruction.tex",
    "tables/table_step4_matching_balance.tex",
    "tables/table_step4_event_study.tex",
    "tables/table_step5_indicator_sensitivity.tex",
    "tables/table_step7_h1_boundary.tex",
    "tables/table_step8_h2_brokerage_persistence.tex",
    "tables/table_step9_complete_ml_results.tex",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manuscript assets and cross-references.")
    parser.add_argument("--manuscript", required=True, help="Path to the manuscript .tex file.")
    parser.add_argument("--bib", default="", help="Optional path to paper.bib for citation-key validation.")
    return parser.parse_args()


def resolve_tex(base: Path, name: str) -> Path:
    path = (base / name).resolve()
    if path.suffix != ".tex":
        path = path.with_suffix(".tex")
    return path


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
    sources = [path]
    combined = "\n" + text
    for raw in INPUTS.findall(text):
        child = resolve_tex(path.parent, raw)
        child_text, child_sources, child_records = collect_tex(child, seen)
        combined += "\n" + child_text
        sources.extend(child_sources)
        records.extend(child_records)
    return combined, sources, records


def asset_status(base: Path, raw: str) -> tuple[str, str]:
    path = (base / raw).resolve()
    if path.exists():
        return "PASS", "Asset exists."
    # LaTeX permits omitted extension; test standard figure endings.
    if not path.suffix:
        for ext in (".pdf", ".png", ".jpg", ".jpeg"):
            if path.with_suffix(ext).exists():
                return "PASS", f"Asset exists as {path.with_suffix(ext).name}."
    return "ERROR", "Referenced asset does not exist."


def read_bib_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", path.read_text(encoding="utf-8", errors="replace")))


def main() -> None:
    args = parse_args()
    root = project_root()
    output = revision_dirs(root)["output"]
    manuscript = Path(args.manuscript).expanduser().resolve()
    text, sources, records = collect_tex(manuscript)
    base = manuscript.parent

    graphics = GRAPHICS.findall(text)
    inputs = INPUTS.findall(manuscript.read_text(encoding="utf-8", errors="replace"))
    for asset in graphics:
        status, detail = asset_status(base, asset)
        records.append({"kind": "figure", "item": asset, "status": status, "detail": detail})
    labels = LABELS.findall(text)
    refs = REFS.findall(text)
    counts = Counter(labels)
    for label, count in sorted(counts.items()):
        if count > 1:
            records.append({"kind": "label", "item": label, "status": "ERROR", "detail": f"Duplicate label appears {count} times."})
    for ref in sorted(set(refs) - set(labels)):
        records.append({"kind": "reference", "item": ref, "status": "ERROR", "detail": "Undefined cross-reference label."})
    for label in sorted(set(labels) - set(refs)):
        if label.startswith(("fig:", "tab:")):
            records.append({"kind": "label", "item": label, "status": "WARN", "detail": "Display label is not cited in manuscript text."})

    for phrase, detail in SUPERSEDED.items():
        if phrase.lower() in text.lower():
            records.append({"kind": "superseded_content", "item": phrase, "status": "ERROR", "detail": detail})

    for asset in sorted(REQUIRED_MAIN_ASSETS):
        if asset not in graphics and asset not in inputs:
            records.append({"kind": "required_revised_asset", "item": asset, "status": "WARN", "detail": "Expected revised asset is not currently referenced."})

    bib = Path(args.bib).expanduser().resolve() if args.bib else None
    if bib:
        keys = read_bib_keys(bib)
        cited = set()
        for value in CITATIONS.findall(text):
            cited.update(x.strip() for x in value.split(",") if x.strip())
        for key in sorted(cited - keys):
            records.append({"kind": "citation", "item": key, "status": "ERROR", "detail": "Cited bibliography key is absent from the supplied .bib file."})
        for key in sorted(keys - cited):
            records.append({"kind": "bibliography", "item": key, "status": "WARN", "detail": "Bibliography entry is not cited in the manuscript."})

    # Caption audit: flag displays whose captions are suspiciously brief or absent.
    for kind, regex in (("figure_caption", FIGURE_ENV), ("table_caption", TABLE_ENV)):
        for index, caption in enumerate(regex.findall(text), 1):
            caption = re.sub(r"\s+", " ", caption).strip()
            if len(caption) < 45:
                records.append({"kind": kind, "item": str(index), "status": "WARN", "detail": f"Brief caption ({len(caption)} characters): {caption}"})

    frame = __import__('pandas').DataFrame(records, columns=["kind", "item", "status", "detail"])
    if frame.empty:
        frame = __import__('pandas').DataFrame(columns=["kind", "item", "status", "detail"])
    frame.to_csv(output / "step10_display_audit.csv", index=False)
    summary = {
        "created_at": now_utc(), "manuscript": str(manuscript), "tex_files_read": [str(x) for x in sources],
        "figures_referenced": graphics, "top_level_inputs": inputs,
        "labels": len(labels), "references": len(refs),
        "errors": int((frame.status == "ERROR").sum()), "warnings": int((frame.status == "WARN").sum()),
        "passes": int((frame.status == "PASS").sum()),
    }
    write_json(output / "step10_display_audit_summary.json", summary)
    lines = ["# Step 10 display audit", "", f"- Manuscript: `{manuscript}`", f"- Errors: **{summary['errors']}**", f"- Warnings: **{summary['warnings']}**", f"- Passes: **{summary['passes']}**", "", "## Findings", ""]
    if frame.empty:
        lines.append("No findings.")
    else:
        for status in ("ERROR", "WARN", "PASS"):
            subset = frame[frame.status.eq(status)]
            if subset.empty:
                continue
            lines += [f"### {status}", ""]
            for _, row in subset.iterrows():
                lines.append(f"- **{row['kind']}** `{row['item']}` — {row['detail']}")
            lines.append("")
    (output / "step10_display_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Step 10 audit complete: {summary['errors']} errors, {summary['warnings']} warnings, {summary['passes']} passes.")

if __name__ == "__main__":
    main()
