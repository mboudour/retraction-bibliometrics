#!/usr/bin/env python3
"""Run Step 4: fetch matched non-retracted controls and estimate the event-study DiD.

Examples
--------
Full first run:
    python revision/scripts/run_step_04.py --api-key "$OPENALEX_API_KEY" \
      --email you@university.edu

Resume/analysis-only run after a completed control fetch:
    python revision/scripts/run_step_04.py --mode analyze

Small smoke test on 100 treated papers only (never use for final results):
    python revision/scripts/run_step_04.py --max-treated 100 --api-key "$OPENALEX_API_KEY"
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().with_name("06_matched_event_study_did.py")
    command = [sys.executable, str(script)] + sys.argv[1:]
    result = subprocess.run(command, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
