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

## Step 4 — matched-control citation event study

Step 4 replaces the prior within-treated-paper comparison with an explicit,
matched non-retracted control design. It uses the frozen `revision_master.csv.gz`
from Step 1 as the treatment cohort and retrieves candidate non-retracted OpenAlex
works matched exactly on **OpenAlex work type (article)**, **venue ISSN-L**, and
**publication year**. Within each venue-year stratum, it selects one nearest-neighbour control without replacement
on mean `log(1 + annual citations)` in the three years before the treated paper's
retraction (`t=-3,-2,-1`). Candidate controls are obtained through reproducible
random OpenAlex samples within each matching stratum, rather than from the API's
first result page. The matched control is assigned its treated paper's retraction
year as a pseudo-event year. The default event window is five years
before and four years after the event. It uses only records whose entire
2016--2025 window is observed in OpenAlex's recent annual citation history,
which preserves two pre-trend years ($t=-5,-4$) before the three-year reference
period ($t=-3,-2,-1$).

Run it from the project root:

```bash
python revision/scripts/run_step_04.py --api-key "$OPENALEX_API_KEY" \
  --email "your.email@university.edu"
```

The first full run fetches and caches candidate controls in
`revision/data/step4_nonretracted_control_cache_v2.jsonl`; it may take considerable
time and can be safely rerun. The default baseline caliper is 0.10 log-citation
units, and the script stops before estimating the event study unless the absolute
baseline-citation standardized mean difference is at most 0.10. Once the cache
exists, run the analysis alone:

```bash
python revision/scripts/run_step_04.py --mode analyze
```

Do not use `--max-treated` for final results; it is only for a small local test.

The final analysis outputs are written to `revision/output/`:

- `step4_matched_event_study_summary.json` — analysis manifest and pre-trend test;
- `step4_treated_eligibility.csv` and `step4_unmatched_treated.csv` — cohort and
  matching-attrition audit;
- `step4_matching_balance.csv` — exact-match and baseline-citation balance;
- `step4_matched_event_study_estimates.csv` — coefficient, pair-clustered SE,
  95% CI, p-value, and matched-pair count at each event time;
- `fig_step4_matched_event_study.png` and `.pdf` — replacement event-study figure;
- `table_step4_matching_balance.tex` and `table_step4_event_study.tex` —
  submission-ready tables.

The retraction-year (`t=0`) estimate is reported but must not be interpreted as a
clean post-notice effect because annual OpenAlex citation counts cannot separate
citations made before and after the precise retraction date. Interpret all results
in light of the matching balance, pre-trend diagnostic, cohort attrition, and
right-censoring.
