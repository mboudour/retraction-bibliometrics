#!/usr/bin/env python3
"""Run Step 5: common-follow-up sensitivity analysis for the indicator family.

Run from the project root:
    python revision/scripts/run_step_05.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().with_name("07_indicator_sensitivity.py")
    raise SystemExit(subprocess.run([sys.executable, str(script)] + sys.argv[1:], check=False).returncode)


if __name__ == "__main__":
    main()
