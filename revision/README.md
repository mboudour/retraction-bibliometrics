# Step 1 local revision package — corrected workflow

Copy `scripts/` and `config/` into:

```text
/Users/moses/WorkPlaces/Sharebox3/WorkingProjects/Retraction Study/computations/revision/
```

Required local inputs already present in your project:

```text
computations/data_collection/raw_data/merged_dataset.csv
computations/wp4_structural/output/wp4_node_metrics.csv
computations/wp2_citation_decay/output/wp2_citation_long.csv
```

## Dependencies

```bash
conda activate base
pip install pandas matplotlib
```

## Why this corrected version uses two passes

The first version of the reason figure used broad regular-expression rules and sent too many important RWDB codes into `Other/Unclear`. In particular, `Investigation by Third Party`, peer-review concerns, data concerns, and attribution concerns should not be left as an unexamined residual category.

This corrected package therefore requires an auditable human review of each distinct RWDB reason code before it makes a primary-category Figure 4.

## Pass A — create the review sheet

```bash
cd "/Users/moses/WorkPlaces/Sharebox3/WorkingProjects/Retraction Study/computations"
python revision/scripts/run_step_01.py
```

This creates:

```text
revision/output/reason_mapping_review.csv
```

Open that CSV in Excel. For each row:

1. Verify or amend `approved_primary_category`.
2. Set `review_status` to `APPROVED`.

The permitted categories are:

```text
Paper Mill
Misconduct/Fraud
Plagiarism/Duplication
Error/Unreliable Results
Publication Process/Editorial
Other/Unclear
```

The category priority for records with multiple RWDB reason codes is documented in:

```text
revision/config/reason_category_priority.csv
```

## Pass B — create the defensible Figure 4

After every review row has status `APPROVED`, rerun the same command:

```bash
python revision/scripts/run_step_01.py
```

It then generates:

```text
revision/output/reason_primary_category_counts.csv
revision/output/reason_multilabel_prevalence_supplement.csv
revision/output/fig_retraction_reasons_primary.pdf
revision/output/fig_retraction_reasons_primary.png
revision/output/sample_flow.csv
revision/output/table_sample_flow.tex
revision/output/consistency_audit.csv
```

## Do not use

Do **not** use the previous `fig_retraction_reasons_multilabel.*` as manuscript Figure 4. Keep it only as a diagnostic/supplementary product if useful.

No raw data are modified. No file is uploaded to GitHub by these scripts.
