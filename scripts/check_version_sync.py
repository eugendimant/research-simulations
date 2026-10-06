#!/usr/bin/env python3
"""Verify every version location listed in CLAUDE.md carries the SAME version.

A mismatch between ``REQUIRED_UTILS_VERSION`` (app.py) and ``utils.__version__``
shows every user a warning banner, so CI runs this on every push.

Usage:
    python scripts/check_version_sync.py          # exits 0 if in sync, 1 otherwise
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "simulation_app"

VERSION_RE = r"(\d+(?:\.\d+){2,3})"

# BUILD_ID in app.py is a free-form cache-busting label, not a plain version string, so it is
# updated by hand per CLAUDE.md and deliberately not checked here.
# (label, file, regex whose first group is the version)
LOCATIONS: List[Tuple[str, Path, str]] = [
    ("app.py REQUIRED_UTILS_VERSION", APP / "app.py",
     rf'^REQUIRED_UTILS_VERSION\s*=\s*["\']{VERSION_RE}["\']'),
    ("app.py APP_VERSION", APP / "app.py",
     rf'^APP_VERSION\s*=\s*["\']{VERSION_RE}["\']'),
    ("utils/__init__.py __version__", APP / "utils" / "__init__.py",
     rf'^__version__\s*=\s*["\']{VERSION_RE}["\']'),
    ("utils/__init__.py docstring 'Version:'", APP / "utils" / "__init__.py",
     rf'^Version:\s*{VERSION_RE}\b'),
    ("utils/qsf_preview.py __version__", APP / "utils" / "qsf_preview.py",
     rf'^__version__\s*=\s*["\']{VERSION_RE}["\']'),
    ("utils/response_library.py __version__", APP / "utils" / "response_library.py",
     rf'^__version__\s*=\s*["\']{VERSION_RE}["\']'),
    ("utils/instructor_report.py __version__", APP / "utils" / "instructor_report.py",
     rf'^__version__\s*=\s*["\']{VERSION_RE}["\']'),
    ("README.md header '**Version X**'", APP / "README.md",
     rf'^\*\*Version\s+{VERSION_RE}\*\*'),
    ("README.md '## The behavioral engine (vX)'", APP / "README.md",
     rf'^##\s+The behavioral engine\s+\(v{VERSION_RE}\)'),
    ("README.md citation '(Version X)'", APP / "README.md",
     rf'\(Version\s+{VERSION_RE}\)\s*\[Computer software\]'),
]

def read_versions() -> List[Tuple[str, Optional[str], str]]:
    """Return (label, version-or-None, note) for every location."""
    rows: List[Tuple[str, Optional[str], str]] = []
    cache: Dict[Path, str] = {}
    for label, path, pattern in LOCATIONS:
        if path not in cache:
            try:
                cache[path] = path.read_text(encoding="utf-8")
            except OSError as exc:
                rows.append((label, None, f"cannot read {path}: {exc}"))
                continue
        match = re.search(pattern, cache[path], re.MULTILINE)
        if match:
            rows.append((label, match.group(1), ""))
        else:
            rows.append((label, None, "pattern not found"))
    return rows


def main() -> int:
    rows = read_versions()
    found = {v for _, v, _ in rows if v}
    missing = [r for r in rows if r[1] is None]
    in_sync = len(found) == 1 and not missing

    width = max(len(label) for label, _, _ in rows)
    print(f"{'Location'.ljust(width)}  Version")
    print(f"{'-' * width}  -------")
    for label, version, note in rows:
        suffix = f"  ({note})" if note else ""
        print(f"{label.ljust(width)}  {version or 'MISSING'}{suffix}")

    if not in_sync:
        print("\nVERSION MISMATCH: all locations must contain the exact same version.")
        if len(found) > 1:
            print(f"  distinct versions found: {sorted(found)}")
        if missing:
            print(f"  unreadable/missing locations: {[m[0] for m in missing]}")
        return 1

    version = next(iter(found))
    print(f"\nOK: all {len(rows)} version locations are {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
