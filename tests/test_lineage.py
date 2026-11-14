"""Tests for the lineage helpers (git commit and DVC data hashes)."""

import subprocess
from pathlib import Path

import pytest
import yaml
from pytest_mock import MockerFixture

from crab_age.lineage import UNKNOWN, dvc_hashes, git_sha, lineage_tags


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repository with one commit."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_git_sha_clean_repo(repo: Path) -> None:
    assert git_sha(repo) == _git(repo, "rev-parse", "HEAD")


def test_git_sha_dirty_repo(repo: Path) -> None:
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    assert git_sha(repo).endswith("-dirty")


def test_git_sha_outside_repo_uses_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAB_GIT_SHA", "abc123")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_sha(tmp_path) == "abc123"


def test_git_sha_without_git_binary(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRAB_GIT_SHA", raising=False)
    mocker.patch("crab_age.lineage.subprocess.run", side_effect=FileNotFoundError("git"))
    assert git_sha() == UNKNOWN


def _write_lock(path: Path, content: object) -> Path:
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_dvc_hashes_from_lock(tmp_path: Path) -> None:
    lock = _write_lock(
        tmp_path / "dvc.lock",
        {
            "stages": {
                "prepare": {
                    "deps": [{"path": "data/raw/x.csv", "md5": "raw1"}, {"path": "src/crab_age/data", "md5": "code"}],
                    "outs": [{"path": "data/processed", "md5": "proc1.dir"}, {"path": "metrics/p.json", "md5": "m"}],
                }
            }
        },
    )
    assert dvc_hashes(lock) == {"dvc_raw_hash": "raw1", "dvc_data_hash": "proc1.dir"}


@pytest.mark.parametrize("content", [None, {}, {"stages": {}}, {"stages": {"prepare": {}}}])
def test_dvc_hashes_incomplete_lock(tmp_path: Path, content: object) -> None:
    lock = _write_lock(tmp_path / "dvc.lock", content)
    assert dvc_hashes(lock) == {"dvc_raw_hash": UNKNOWN, "dvc_data_hash": UNKNOWN}


def test_dvc_hashes_missing_lock(tmp_path: Path) -> None:
    assert dvc_hashes(tmp_path / "missing.lock") == {"dvc_raw_hash": UNKNOWN, "dvc_data_hash": UNKNOWN}


def test_project_lineage_tags() -> None:
    tags = lineage_tags()
    assert set(tags) == {"git_sha", "dvc_raw_hash", "dvc_data_hash"}
    assert tags["dvc_data_hash"].endswith(".dir")
