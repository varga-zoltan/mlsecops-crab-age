"""Tests for the lineage helpers (git commit and DVC data hashes)."""

import subprocess
from pathlib import Path

import pytest
import yaml
from pytest_mock import MockerFixture

from crab_age.lineage import UNKNOWN, dvc_hashes, git_sha, lineage_tags, read_git_dir_sha


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


SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture
def no_git_program(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate the Docker image: no git program and no CRAB_GIT_SHA."""
    monkeypatch.delenv("CRAB_GIT_SHA", raising=False)
    mocker.patch("crab_age.lineage.subprocess.run", side_effect=FileNotFoundError("git"))


def _git_dir(root: Path, head: str, files: dict[str, str]) -> Path:
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(head + "\n", encoding="utf-8")
    for name, content in files.items():
        (git_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (git_dir / name).write_text(content, encoding="utf-8")
    return git_dir


def test_without_git_program_and_folder_is_unknown(tmp_path: Path, no_git_program: None) -> None:
    assert git_sha(tmp_path) == UNKNOWN


def test_without_git_program_reads_loose_ref(tmp_path: Path, no_git_program: None) -> None:
    _git_dir(tmp_path, "ref: refs/heads/main", {"refs/heads/main": SHA_A + "\n"})
    assert git_sha(tmp_path) == SHA_A


def test_without_git_program_reads_packed_ref(tmp_path: Path, no_git_program: None) -> None:
    packed = (
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{SHA_B} refs/heads/feature\n"
        f"{SHA_A} refs/heads/main\n"
        f"{SHA_B} refs/tags/v1\n"
        f"^{SHA_A}\n"
    )
    _git_dir(tmp_path, "ref: refs/heads/main", {"packed-refs": packed})
    assert git_sha(tmp_path) == SHA_A


def test_without_git_program_reads_detached_head(tmp_path: Path, no_git_program: None) -> None:
    _git_dir(tmp_path, SHA_B, {})
    assert git_sha(tmp_path) == SHA_B


def test_folder_beats_environment(
    tmp_path: Path, no_git_program: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRAB_GIT_SHA", "from-build-arg")
    _git_dir(tmp_path, SHA_A, {})
    assert git_sha(tmp_path) == SHA_A


@pytest.mark.parametrize(
    ("head", "files"),
    [
        ("ref: refs/heads/main", {}),
        ("ref: refs/heads/main", {"packed-refs": f"{SHA_A} refs/heads/other\n"}),
        ("ref: refs/heads/main", {"refs/heads/main": "not-a-sha\n"}),
        ("ref: refs/heads/main", {"packed-refs": "zzz refs/heads/main\n"}),
        ("1234", {}),
    ],
)
def test_broken_git_folder_falls_back(
    tmp_path: Path, no_git_program: None, monkeypatch: pytest.MonkeyPatch, head: str, files: dict[str, str]
) -> None:
    monkeypatch.setenv("CRAB_GIT_SHA", "from-build-arg")
    _git_dir(tmp_path, head, files)
    assert read_git_dir_sha(tmp_path / ".git") is None
    assert git_sha(tmp_path) == "from-build-arg"


def test_unreadable_head_is_none(tmp_path: Path) -> None:
    git_dir = _git_dir(tmp_path, SHA_A, {})
    (git_dir / "HEAD").write_bytes(b"\xff\xfe\x00")
    assert read_git_dir_sha(git_dir) is None


def test_empty_build_arg_means_unknown(tmp_path: Path, no_git_program: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAB_GIT_SHA", "")
    assert git_sha(tmp_path) == UNKNOWN


def test_reader_agrees_with_git_on_real_repo(repo: Path) -> None:
    assert read_git_dir_sha(repo / ".git") == _git(repo, "rev-parse", "HEAD")
    branch_ref = (repo / ".git" / "HEAD").read_text(encoding="utf-8").removeprefix("ref:").strip()
    _git(repo, "pack-refs", "--all")
    assert not (repo / ".git" / branch_ref).exists()  # the ref now lives only in packed-refs
    assert read_git_dir_sha(repo / ".git") == _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach")
    assert read_git_dir_sha(repo / ".git") == _git(repo, "rev-parse", "HEAD")


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
