#!/usr/bin/env python3
"""One-command runner for Step 9: complete predictive-model evaluation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

script = Path(__file__).with_name("11_complete_ml_evaluation.py")
raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))
