# Retraction Bibliometric Study: A Computational Analysis of Scientific Retractions

**Moses Boudourides and Emanuela Chiriac**

This repository contains the complete computational pipeline, data collection scripts, and analysis code for a comprehensive bibliometric study of scientific retractions. The project investigates the scale, causes, and consequences of retracted literature using a corpus of over 58,000 retracted papers.

## Project Overview

Scientific retractions have grown exponentially over the past two decades. While retractions are a necessary self-correcting mechanism for the scientific record, the continued citation of retracted work (citation contagion) poses a systemic threat to knowledge integrity. 

This project operationalises a "temporal network ecology" approach to study retractions across five distinct work packages:
1. **Descriptive Mapping:** Baseline trends, disciplinary distribution, and taxonomy of retraction reasons.
2. **Citation Decay Event Study:** Difference-in-differences (DiD) analysis of citation trajectories before and after retraction.
3. **Contagion & Ripple Effects:** The spread of citations to retracted work and the impact on co-author networks.
4. **Structural Vulnerability:** Network analysis of the citation graph to identify structural predictors of citation accumulation.
5. **Predictive Modelling:** Machine learning models to identify early-warning signals of retraction using pre-publication features.

## Data Sources

The study integrates two open-access, large-scale data sources:
* **Retraction Watch Database:** The authoritative registry of scientific retractions, hosted by Crossref.
* **OpenAlex:** The open catalog of the global research system, providing rich metadata, citation counts, and reference graphs.

**Note:** Raw data files are excluded from this repository due to size constraints. To reproduce the dataset from scratch, run the scripts in `data_collection/scripts/`.

## Key Findings

### 1. The Exponential Growth of Retractions
The volume of retractions has grown dramatically, peaking at over 13,000 notices in 2023. Analysis of retraction reasons reveals that **Misconduct/Fraud (23.8%)** and **Paper Mills (22.2%)** together account for nearly half of all retractions, with Peer Review Manipulation (15.5%) also representing a substantial share. Genuine errors account for only 16.5%.

### 2. The Inverted-V Citation Trajectory
A field-year normalised event study demonstrates a striking "inverted-V" pattern in citation dynamics. Problematic papers accumulate disproportionate attention (rising steadily above the field-year baseline) in the years *before* retraction. Following the retraction notice (t=0), citations fall persistently and significantly below the baseline.

### 3. Misconduct Papers Drive Citation Contagion
Papers retracted for Misconduct/Fraud accumulated a median of 19 citations prior to retraction — nearly 5× higher than papers retracted for Paper Mill activity or Peer Review Manipulation (median 4). This indicates that deliberate fraud often occurs in highly visible work, representing the highest-risk contagion events in the literature.

### 4. Structural Bridging Predicts Citation Accumulation
Network analysis of the citation graph (1.3M nodes, 1.6M edges) reveals that **betweenness centrality** is the dominant predictor of citation accumulation among retracted papers. Retracted papers that act as structural bridges between different research communities accumulate far more citations than those that are merely locally influential (high PageRank).

### 5. Early-Warning Signals are Detectable
Using strictly pre-publication features (author count, reference count, open access status, abstract presence, title length) and early citations (years 1-2), a Gradient Boosting classifier achieves an **AUC of 0.952** in distinguishing retracted from non-retracted papers. The length of the title and the volume of early citations are the strongest predictive features.

## Repository Structure

```
retraction-bibliometrics/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── run_all.sh                 # Master execution script
├── shared/                    # Shared utilities and data loaders
│   └── utils.py
├── data_collection/           # Data fetching scripts (Crossref & OpenAlex)
├── wp1_descriptive/           # WP1: Descriptive mapping and taxonomy
├── wp2_citation_decay/        # WP2: Difference-in-differences event study
├── wp3_contagion/             # WP3: Contagion proxy and co-author networks
├── wp4_structural/            # WP4: Citation graph and structural metrics
└── wp5_prediction/            # WP5: Machine learning early-warning models
```
*(Note: Each WP directory contains a `scripts/` folder with the source code and an `output/` folder containing the generated figures and CSV results.)*

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/mosesboudourides/retraction-bibliometrics.git
   cd retraction-bibliometrics
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Fetch the raw data:
   ```bash
   python data_collection/scripts/01_fetch_retraction_watch.py
   python data_collection/scripts/02_fetch_openalex_metadata.py
   ```

4. Run the full computational pipeline:
   ```bash
   bash run_all.sh
   ```

## Authors & Acknowledgements
This computational pipeline was developed for the Retraction Bibliometric Study. Data provided by Retraction Watch (via Crossref) and OpenAlex.

## Copyright

© 2026 Moses Boudourides. All rights reserved.

This repository and its contents are made available for academic peer review purposes only. No part of this work may be reproduced, distributed, or used in any form without the express written permission of the authors.
