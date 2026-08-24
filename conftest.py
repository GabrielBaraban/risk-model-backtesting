"""Pytest configuration: put the repo root on sys.path.

Lets `tests/` do `from src.models import ...` without installing the
project as a package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
