#!/usr/bin/env python3
"""One-command runner for Step 10: manuscript display and cross-reference audit."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

script = Path(__file__).with_name("12_audit_manuscript_displays.py")
raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))
