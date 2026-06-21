"""Load tournament training-repo submission settings from training_repo.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models.payload_models import TrainingRepoResponse

_CONFIG_PATH = Path(__file__).resolve().parent / "training_repo.json"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _config_path() -> Path:
    return _CONFIG_PATH


def load_training_repo_config(path: Path | None = None) -> dict[str, dict]:
    config_path = path or _config_path()
    with open(config_path) as f:
        return json.load(f)


def validate_commit_hash(commit_hash: str) -> None:
    if not _COMMIT_RE.match(commit_hash):
        raise ValueError(
            f"commit_hash must be a 40-character hex SHA, got: {commit_hash!r}"
        )


def get_training_repo_response(
    task_type: str,
    path: Path | None = None,
) -> TrainingRepoResponse:
    from core.models.payload_models import TrainingRepoResponse

    config = load_training_repo_config(path)
    key = task_type.value if hasattr(task_type, "value") else task_type
    if key not in config:
        raise KeyError(f"No training repo config for tournament type {key!r}")

    entry = config[key]
    commit_hash = entry["commit_hash"]
    validate_commit_hash(commit_hash)

    github_repo = entry["github_repo"]
    if "YOUR_USERNAME" in github_repo:
        raise ValueError(
            "Update miner/training_repo.json with your GitHub repo URL "
            "(replace YOUR_USERNAME) before tournament entry."
        )

    return TrainingRepoResponse(
        github_repo=github_repo,
        commit_hash=commit_hash,
        github_token=entry.get("github_token"),
        requested_datasets=entry.get("requested_datasets"),
    )


def update_commit_hash(commit_hash: str, path: Path | None = None) -> None:
    """Write the same commit hash for all tournament types in training_repo.json."""
    validate_commit_hash(commit_hash)
    config_path = path or _config_path()
    config = load_training_repo_config(config_path)
    for entry in config.values():
        entry["commit_hash"] = commit_hash
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
