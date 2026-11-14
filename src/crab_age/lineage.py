"""Lineage: which code and which data produced a model.

The training run gets these values as MLflow tags, so every registered model
version can be traced back to a git commit and a DVC data version.
"""

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

UNKNOWN = "unknown"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd, check=True).stdout.strip()


def git_sha(cwd: Path = Path(".")) -> str:
    """Return the current commit, with '-dirty' if there are uncommitted changes.

    Inside a Docker image there is no .git folder; then the CRAB_GIT_SHA
    environment variable (set at build time) is used, else 'unknown'.
    Lineage metadata must never break training, so errors are not raised.
    """
    try:
        sha = _git("rev-parse", "HEAD", cwd=cwd)
        dirty = bool(_git("status", "--porcelain", cwd=cwd))
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("CRAB_GIT_SHA", UNKNOWN)
    return f"{sha}-dirty" if dirty else sha


def dvc_hashes(lock_path: Path = Path("dvc.lock")) -> dict[str, str]:
    """Return the DVC md5 of the raw data and of the processed data (from dvc.lock)."""
    hashes = {"dvc_raw_hash": UNKNOWN, "dvc_data_hash": UNKNOWN}
    if not lock_path.exists():
        return hashes
    lock: dict[str, Any] = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    stage: dict[str, Any] = lock.get("stages", {}).get("prepare", {})
    for dep in stage.get("deps", []):
        if str(dep.get("path", "")).startswith("data/raw/"):
            hashes["dvc_raw_hash"] = str(dep.get("md5", UNKNOWN))
    for out in stage.get("outs", []):
        if out.get("path") == "data/processed":
            hashes["dvc_data_hash"] = str(out.get("md5", UNKNOWN))
    return hashes


def lineage_tags(cwd: Path = Path("."), lock_path: Path = Path("dvc.lock")) -> dict[str, str]:
    """Return the MLflow tags that connect a model to its code and data."""
    return {"git_sha": git_sha(cwd), **dvc_hashes(lock_path)}
