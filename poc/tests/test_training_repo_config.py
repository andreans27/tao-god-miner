"""Tests for miner training_repo_config JSON helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "original"))

from miner.training_repo_config import update_commit_hash
from miner.training_repo_config import validate_commit_hash


def test_validate_commit_hash_accepts_40_hex():
    validate_commit_hash("b4c468c8b5f5dcda596e229d4ef492c47a149680")


def test_validate_commit_hash_rejects_branch_name():
    with pytest.raises(ValueError, match="40-character hex"):
        validate_commit_hash("main")


def test_update_commit_hash(tmp_path: Path):
    config = {
        "text": {"github_repo": "https://github.com/u/r", "commit_hash": "a" * 40},
        "image": {"github_repo": "https://github.com/u/r", "commit_hash": "a" * 40},
    }
    path = tmp_path / "training_repo.json"
    path.write_text(json.dumps(config))

    new_hash = "b" * 40
    update_commit_hash(new_hash, path)
    updated = json.loads(path.read_text())
    assert updated["text"]["commit_hash"] == new_hash
    assert updated["image"]["commit_hash"] == new_hash
