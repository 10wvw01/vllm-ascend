#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unified standalone Stock, all-version L1 benchmark and diagnostic entry."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from L1_test.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
