#!/bin/sh
set -eu
DRT_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DRT_RUNTIME_PYTHON=${DRT_PYTHON:-"$DRT_SCRIPT_DIR/../.venv/bin/python"}
if [ ! -x "$DRT_RUNTIME_PYTHON" ]; then
  echo "No validated DRT environment. Run: python3.11 scripts/bootstrap.py" >&2
  echo "Or set DRT_PYTHON to an explicit environment's Python." >&2
  exit 2
fi
exec "$DRT_RUNTIME_PYTHON" "$DRT_SCRIPT_DIR/batch_drt.py" "$@"
