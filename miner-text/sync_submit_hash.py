#!/usr/bin/env python3
"""Sync miner-text HEAD commit into original/miner/training_repo.json."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MINER_TEXT = Path(__file__).resolve().parent
TRAINING_REPO_JSON = REPO_ROOT / "original" / "miner" / "training_repo.json"

sys.path.insert(0, str(REPO_ROOT / "original"))

from miner.training_repo_config import update_commit_hash


def main() -> None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=MINER_TEXT,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = result.stdout.strip()
    update_commit_hash(commit, TRAINING_REPO_JSON)
    print(f"Updated {TRAINING_REPO_JSON}")
    print(f"  commit_hash: {commit}")
    print("Remember to push miner-text to GitHub and set github_repo in training_repo.json")


if __name__ == "__main__":
    main()
