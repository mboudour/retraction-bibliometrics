#!/usr/bin/env python3
"""Run Step 6: balanced citation-network reconstruction and robustness diagnostics.

Run from the project root:
    python revision/scripts/run_step_06.py --mode all --api-key "$OPENALEX_API_KEY" \
        --email "moses.boudourides@northwestern.edu"
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().with_name("08_reconstruct_network_robustness.py")
    raise SystemExit(subprocess.run([sys.executable, str(script)] + sys.argv[1:], check=False).returncode)


if __name__ == "__main__":
    main()
