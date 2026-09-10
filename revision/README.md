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


## Step 5 — Sensitivity analysis for post-retraction citation-persistence indicators

Run from the project root after Step 1:

```bash
python revision/scripts/run_step_05.py
```

This calculation uses only the frozen Step 1 corpus and makes no network or API
requests. It recalculates the four indicator-family measures using **common
symmetric follow-up windows** of $H=1,2,3,4$ years. A paper contributes to a
specific $H$ only if annual OpenAlex citation counts are observable from
$r_p-H$ through $r_p+H$, avoiding the unequal follow-up time that otherwise
advantages older retractions. The script also varies the minimum scholarly-unit
size (10, 30, and 50 papers) and the minimum event-time citation denominator
(1 and 10 citations).

The reference specification is $H=4$, at least 30 papers per unit, and a minimum
event-time denominator of 10 citations. It outputs the global indicator
sensitivity, unit-level estimates for every specification, cohort eligibility,
rank-stability statistics, two LaTeX tables, and two figures. The definitions
are window-specific: Exposure is the pooled post-event citation share over the
common window; Contamination is the mean post-versus-pre event-time ratio;
Persistence is the last event year with any post-event citation; and Recovery is
the OLS slope of that ratio on event time.

Key outputs in `revision/output/`:

```text
step5_indicator_sensitivity_manifest.json
step5_indicator_global_sensitivity.csv
step5_indicator_eligibility.csv
step5_indicator_estimates_long.csv
step5_indicator_ranking_stability.csv
table_step5_indicator_sensitivity.tex
table_step5_indicator_stability.tex
fig_step5_indicator_window_sensitivity.png
fig_step5_indicator_ranking_stability.png
```

Do not report indicator values until inspecting the eligibility and global
sensitivity outputs. The common-follow-up restriction can materially change the
eligible cohort as $H$ increases.


## Step 6 — Balanced citation-network reconstruction and robustness diagnostics

Run from the project root after the completed Step 4 matched-control analysis:

```bash
export OPENALEX_API_KEY="$(cat openalex_api_key.txt)"
python revision/scripts/run_step_06.py \
  --mode all \
  --api-key "$OPENALEX_API_KEY" \
  --email "moses.boudourides@northwestern.edu"
```

The script uses the Step 4 matched pairs as a balanced focal design: each
retracted article and its matched non-retracted control is a focal paper. For
every focal paper, it retrieves a reproducible sample of up to 30 citing works
and up to 30 outgoing references. The selected citing works contribute their own
sampled outgoing references. Therefore, the reconstructed directed graph includes
retracted-to-non-retracted, non-retracted-to-retracted, and context-to-context
citation edges, unlike the earlier retraction-centered reference graph.

This is a **sampled one-hop neighbourhood reconstruction**, not the entire
OpenAlex graph. It has explicit caps and omits citation edges among context works
that are not selected citing neighbours. The output diagnostics report the precise
network size, fetch success, edge-type composition, retained/API-reported
neighbour counts, and the context-to-context edge share. The brokerage metric is
the \textbf{unnormalized focal-path brokerage count}: the number of sampled
shortest directed paths from a reproducible sample of works that cite a focal
paper to retained references of focal papers that pass through each node. This
focuses the metric on the only directed paths for which the sampled one-hop graph
can identify focal mediation. It avoids the near-zero focal values produced by
generic normalized random-source approximation in a large one-hop graph. The
script also runs a focal-label permutation design check for this metric; that
check is unadjusted and is not the primary H1/H2 inference, which follows in
Step 7.

The cache `revision/data/step6_focal_neighborhoods.jsonl` permits an interrupted
fetch to resume. Use `--mode analyze` to recompute outputs from a completed cache
without API calls. Use `--force-refetch` only to deliberately replace all cached
neighbourhood records.

Key outputs in `revision/output/`:

```text
step6_expanded_network_summary.json
step6_neighborhood_fetch_audit.csv
step6_edge_type_composition.csv
step6_node_type_composition.csv
step6_focal_node_metrics.csv
step6_label_permutation_null.csv
step6_expanded_network_edges.csv.gz
step6_expanded_network_nodes.csv.gz
fig_step6_edge_composition.png
fig_step6_label_permutation.png
```


## Step 7 — Matched and adjusted test of H1 (boundary positioning)

After Steps 4 and 6, run:

```bash
python revision/scripts/run_step_07.py
```

The script consumes `step4_matched_pairs.csv`, the Step 4 control cache, and
`step6_focal_node_metrics.csv`. It tests H1 using the unnormalized focal-path
brokerage count in three complementary ways: a paired sign-flip test of the mean
within-pair difference in `log(1 + brokerage)`, a Wilcoxon signed-rank check,
and an exact McNemar comparison of non-zero brokerage prevalence. It also fits
an individual-paper model with matched-pair fixed effects and pair-clustered
standard errors. Pair effects absorb article type, journal ISSN-L, publication
year, and paper age at the pseudo-event; the model adjusts for individual
pre-event citation level and a retracted-paper-by-topic-mismatch term.

Outputs include `step7_h1_summary.json`, a complete inference table, model
coefficients, pair-level data, a manuscript-ready LaTeX table, and a two-panel
matched-comparison figure. Holm-adjusted p-values are reported across the four
confirmatory/secondary H1 tests. No API call is made.


## Step 8 — Direct test of H2 (brokerage and post-retraction citation persistence)

Run after Step 6:

```bash
python revision/scripts/run_step_08.py
```

The script narrows H2 to an observable associational claim: among retracted
papers, is higher pre-retraction local focal-path brokerage associated with
greater post-retraction citation persistence? It recalculates a time-respecting
brokerage count using only selected citing works published before each focal
paper's retraction year. The primary outcome is `log(1 + post-retraction
citations)` in a common four-year follow-up window. The adjusted OLS model
controls for pre-retraction citations and paper age at retraction, with HC3
standard errors. Spearman rank association is tested by label permutation and
bootstrapped confidence intervals. The script also reports H=1--4 sensitivity.

The analysis does not measure citation stance, knowledge transmission, or
multi-step propagation through the literature. Interpret results as associations
within the sampled one-hop citation neighbourhood.


## Step 9 — Complete predictive-model evaluation

Run from the project root after Step 1:

```bash
export OPENALEX_API_KEY="$(cat openalex_api_key.txt)"
python revision/scripts/run_step_09.py --mode all --api-key "$OPENALEX_API_KEY" --email "your.email@institution.edu"
```

The first run creates a cached sample of non-retracted OpenAlex research articles
in `revision/data/step9_nonretracted_candidate_cache.jsonl`. It is resumable. The
analysis selects one control per retracted paper without replacement using
**exact publication-year and broad-field matches only**. Retraction cases without
a strict match are excluded and recorded in `step9_control_matching_audit.csv`;
there is no year-only fallback because it would confound classification with field
composition.

Two feature sets are reported separately: **at-publication** (title length,
author count, reference count, open-access status, and abstract availability) and
**one-year early warning** (the same variables plus citations in the first full
year after publication). The latter is explicitly post-publication. Publication
year, total citations, retraction variables, author-retraction histories, and
network variables are excluded.

Four prespecified classifiers (logistic regression, random forest, gradient
boosting, and RBF SVM) are evaluated under random 80/20, temporal, and
cross-disciplinary validation. Output includes accuracy, precision, recall, F1,
ROC-AUC, PR-AUC, log loss, MCC, Brier score, 10-bin ECE, confusion-matrix
counts, calibration-bin data, and relative permutation importance. The analyses
are retrospective; they are not a deployment-ready retraction-screening system.


### Bounded local run

For a computationally bounded local rerun, the script defaults to a reproducible
proportional subsample of **2,500 strict year-and-field matched pairs**. This cap
is applied only after strict matching and preserves the broad-field composition
of the strict-match cohort. It makes the four-model, three-protocol evaluation
feasible on a laptop without weakening the matching rule. Set
`--max-matched-pairs 0` to analyse every strict match when sufficient compute is
available.


## Step 10 — Display, caption, and cross-reference audit

After integrating all revised figures and LaTeX tables into the manuscript folder,
run:

```bash
python revision/scripts/run_step_10.py \
  --manuscript "/absolute/path/to/manuscript_revision.tex" \
  --bib "/absolute/path/to/paper.bib"
```

The audit does not modify the manuscript. It writes
`step10_display_audit.csv`, `step10_display_audit.md`, and
`step10_display_audit_summary.json` to `revision/output/`. It checks missing
figure/table assets, undefined and duplicate labels, unused display labels,
missing bibliography keys, stale filenames/terminology from the submitted
version, and short captions requiring editorial review. Run it again only after
all manuscript display changes have been made. A submission-ready manuscript
must have zero `ERROR` findings.
