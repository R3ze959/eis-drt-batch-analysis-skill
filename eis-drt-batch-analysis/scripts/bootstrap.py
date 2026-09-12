#!/usr/bin/env python3
"""Install into a new isolated Python 3.11 environment; never modify a user's existing venv."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv


def installation_environment():
    # Python -I does not isolate pip configuration. Preserve explicit transport
    # settings only, and disable all on-disk pip configuration including globals.
    transport = {"PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_NO_INDEX", "PIP_FIND_LINKS",
                 "PIP_TRUSTED_HOST", "PIP_CERT", "PIP_CLIENT_CERT", "PIP_PROXY",
                 "PIP_TIMEOUT", "PIP_RETRIES", "PIP_CACHE_DIR"}
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith("PIP_") or k.upper() in transport}
    for key in list(env):
        if key.upper() in {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"}:
            env.pop(key)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_CONFIG_FILE"] = os.devnull
    return env

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--venv", type=Path, help="New environment; default is .venv beside SKILL.md")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = (a.venv or root / ".venv").expanduser().resolve()
    if sys.version_info[:2] != (3, 11):
        p.error("This release pins and tests CPython 3.11. Invoke with python3.11 (Windows: py -3.11).")
    if target.exists():
        p.error("Environment already exists. Choose a NEW --venv; no overwrite or in-place upgrade is performed.")
    if target == root or root.is_relative_to(target):
        p.error("Environment must not be the package or an ancestor of it.")
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(target)
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = installation_environment()
    subprocess.run([str(python), "-I", "-m", "pip", "install", "--disable-pip-version-check",
                    "-r", str(root / "requirements.lock.txt")], env=env, check=True)
    subprocess.run([str(python), "-I", "-m", "pip", "check"], env=env, check=True)
    subprocess.run([str(python), str(root / "scripts/check_runtime.py")], env=env, check=True)
    (target / "drt-installation.json").write_text(json.dumps({
        "version": (root / "VERSION").read_text().strip(), "python": str(python),
        "platform": platform.platform(), "isolated": True,
        "note": "Installation metadata only; scientific acceptance is per spectrum."
    }, indent=2), encoding="utf-8")
    print("Installed. Run:", python, root / "scripts/batch_drt.py", "--help")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
