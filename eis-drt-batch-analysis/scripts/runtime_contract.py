"""Actual locked dependency closure and interpreter identity, without GUI coupling."""
import importlib.metadata
from pathlib import Path
import platform
import sys

LOCK = Path(__file__).resolve().parents[1] / "requirements.lock.txt"


def locked_versions(environment=None):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    pins = {}
    for raw in LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        specifiers = list(requirement.specifier)
        if requirement.url or requirement.extras or len(specifiers) != 1 or specifiers[0].operator != "==" or "*" in specifiers[0].version:
            raise ValueError("Expected an exact dependency pin: " + line)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        name = canonicalize_name(requirement.name)
        if name in pins:
            raise ValueError("Duplicate active dependency pin: " + name)
        pins[name] = specifiers[0].version
    return pins


def actual_versions():
    actual = {}
    for name in locked_versions():
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = "not-installed"
    return actual


def runtime_identity():
    return {"python_version": platform.python_version(), "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag, "system": platform.system(), "machine": platform.machine()}
