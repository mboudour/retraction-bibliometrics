#!/usr/bin/env python3
"""One-command runner for Step 7: adjusted matched-pair H1 analysis."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

script = Path(__file__).with_name("09_test_h1_boundary_positioning.py")
command = [sys.executable, str(script), *sys.argv[1:]]
raise SystemExit(subprocess.call(command))
