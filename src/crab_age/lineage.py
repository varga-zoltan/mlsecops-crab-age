"""Lineage: which code and which data produced a model.

The training run gets these values as MLflow tags, so every registered model
version can be traced back to a git commit and a DVC data version.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

UNKNOWN = "unknown"
_SHA = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")  # SHA-1 (40) or SHA-256 (64) object id


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd, check=True).stdout.strip()


def _valid_sha(value: str) -> str | None:
    return value if _SHA.match(value) else None


def _packed_ref(git_dir: Path, ref: str) -> str | None:
    """Look a ref up in .git/packed-refs (fresh clones and 'git gc' keep refs there)."""
    packed = git_dir / "packed-refs"
    if not packed.is_file():
        return None
    for line in packed.read_text(encoding="utf-8").splitlines():
        # '#' starts the header, '^' marks the peeled commit of the tag above it.
        if line.startswith(("#", "^")):
            continue
        sha, _, name = line.partition(" ")
        if name.strip() == ref:
            return _valid_sha(sha.strip())
    return None


def read_git_dir_sha(git_dir: Path) -> str | None:
    """Read the commit id of HEAD straight from a .git folder, without the git program.

    Handles a branch HEAD (``ref: refs/heads/main``) whose ref is a loose file or
    sits in ``packed-refs``, and a detached HEAD (HEAD holds the commit id itself).
    Returns None if the folder is missing or anything looks wrong.
    """
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return _valid_sha(head)  # detached HEAD
        ref = head.removeprefix("ref:").strip()
        loose = git_dir / ref
        if loose.is_file():
            return _valid_sha(loose.read_text(encoding="utf-8").strip())
        return _packed_ref(git_dir, ref)
    except (OSError, UnicodeDecodeError):
        return None


def git_sha(cwd: Path = Path(".")) -> str:
    """Return the current commit id; lineage errors never break training.

    Sources, in order:

    1. the git program: also adds '-dirty' if there are uncommitted changes;
    2. the .git folder read directly (in Docker it is mounted read-only, the image
       has no git program); uncommitted changes cannot be detected this way;
    3. the CRAB_GIT_SHA environment variable (a build argument of the image);
    4. 'unknown'.
    """
    try:
        sha = _git("rev-parse", "HEAD", cwd=cwd)
        dirty = bool(_git("status", "--porcelain", cwd=cwd))
    except (OSError, subprocess.CalledProcessError):
        from_folder = read_git_dir_sha(cwd / ".git")
        if from_folder is not None:
            return from_folder
        return os.environ.get("CRAB_GIT_SHA", UNKNOWN) or UNKNOWN
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
