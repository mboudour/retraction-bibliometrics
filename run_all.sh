#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Master pipeline for the Retraction Bibliometric Study
#
# Runs all work packages in order. Each script is self-contained and reads
# from data_collection/raw_data/. Outputs go into each WP's output/ directory.
#
# Usage:
#   bash run_all.sh [--year-min YYYY] [--year-max YYYY] [--email you@example.com]
#
# Prerequisites:
#   pip install -r requirements.txt
# =============================================================================

set -euo pipefail

YEAR_MIN="${YEAR_MIN:-2000}"
YEAR_MAX="${YEAR_MAX:-2022}"
EMAIL="${EMAIL:-researcher@example.com}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "============================================================"
echo " Retraction Bibliometric Study — Full Pipeline"
echo " Year range: ${YEAR_MIN}–${YEAR_MAX}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Data collection (skip if raw data already present)
# ---------------------------------------------------------------------------
RAW="data_collection/raw_data"
if [ ! -f "${RAW}/retraction_watch_clean.csv" ]; then
    echo "[Step 0a] Downloading Retraction Watch data …"
    python3 data_collection/scripts/01_fetch_retraction_watch.py \
        --year-min "${YEAR_MIN}" --year-max "${YEAR_MAX}"
else
    echo "[Step 0a] retraction_watch_clean.csv already present — skipping download."
fi

if [ ! -f "${RAW}/openalex_metadata.jsonl" ]; then
    echo "[Step 0b] Fetching OpenAlex metadata …"
    python3 data_collection/scripts/02_fetch_openalex_metadata.py \
        --email "${EMAIL}"
else
    echo "[Step 0b] openalex_metadata.jsonl already present — skipping fetch."
fi

echo ""

# ---------------------------------------------------------------------------
# WP1: Descriptive Mapping
# ---------------------------------------------------------------------------
echo "[WP1] Descriptive mapping …"
python3 wp1_descriptive/scripts/wp1_descriptive_mapping.py
echo ""

# ---------------------------------------------------------------------------
# WP2: Citation Decay (Event Study)
# ---------------------------------------------------------------------------
echo "[WP2] Citation decay event study …"
python3 wp2_citation_decay/scripts/wp2_citation_decay.py
echo ""

# ---------------------------------------------------------------------------
# WP3: Retraction Contagion
# ---------------------------------------------------------------------------
echo "[WP3] Retraction contagion and collaboration ripple …"
python3 wp3_contagion/scripts/wp3_contagion.py
echo ""

# ---------------------------------------------------------------------------
# WP4: Structural Vulnerability
# ---------------------------------------------------------------------------
echo "[WP4] Network construction and structural vulnerability …"
python3 wp4_structural/scripts/wp4_structural_vulnerability.py
echo ""

# ---------------------------------------------------------------------------
# WP5: Predictive Modelling
# ---------------------------------------------------------------------------
echo "[WP5] Predictive machine-learning pipeline …"
python3 wp5_prediction/scripts/wp5_prediction.py
echo ""

echo "============================================================"
echo " All work packages complete."
echo " Results are in each WP's output/ directory."
echo "============================================================"
