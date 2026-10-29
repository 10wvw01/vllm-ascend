# SPDX-License-Identifier: Apache-2.0
"""Make sibling helper modules (unified_metrics) importable under --import-mode=importlib.

quick.py/unified.py gate every device session on `pytest -q <test> --import-mode=importlib`;
that mode skips pytest's sys.path insertion, so test_unified.py's plain
`from unified_metrics import ...` failed collection and STOPped all sessions at the CPU gate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
