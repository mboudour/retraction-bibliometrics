#!/usr/bin/env python3
"""One-command runner for Step 8: direct H2 brokerage-persistence analysis."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

script = Path(__file__).with_name("10_test_h2_brokerage_persistence.py")
raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))
