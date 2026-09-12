#!/usr/bin/env python3
"""Fail explicitly on an incomplete or changed pinned numerical runtime."""
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from runtime_contract import locked_versions, actual_versions, runtime_identity

def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    if sys.version_info[:2] != (3, 11):
        errors.append("CPython 3.11 is the release baseline")
    versions = actual_versions()
    for name, expected in locked_versions().items():
        if versions[name] != expected:
            errors.append(f"{name}: expected {expected}, found {versions[name]}")
    for name in ("numpy", "scipy", "matplotlib", "pyimpspec", "cvxopt"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"import {name}: {exc}")
    print(json.dumps({"status": "FAIL" if errors else "PASS", "errors": errors,
                      "python": sys.version, "runtime": runtime_identity(), "packages": versions}, indent=2))
    return bool(errors)

if __name__ == "__main__":
    raise SystemExit(main())
