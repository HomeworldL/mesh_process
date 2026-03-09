#!/usr/bin/env python3
"""Compatibility launcher for tests/print.py."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "tests" / "print.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
