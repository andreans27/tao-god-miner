"""Unit tests for train_runner helpers (no Docker/GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from poc.manifest import Manifest
from poc.train_runner import TrainConfig
from poc.train_runner import eval_repo_id
from poc.train_runner import expected_repo_name
from poc.train_runner import local_checkpoint_dir
from poc.train_runner import stage_checkpoint_for_eval


@pytest.fixture
def manifest() -> Manifest:
    return Manifest.load(Path(__file__).parent.parent / "manifest.json")


@pytest.fixture
def train_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        trainer_repo=REPO_ROOT / "miner-text",
        cache_dir=tmp_path / "cache",
        outputs_dir=tmp_path / "outputs",
        expected_repo_prefix="poc-test",
    )


def test_expected_repo_name_uses_task_prefix(manifest: Manifest, train_config: TrainConfig):
    entry = manifest.boss_battery[0]
    name = expected_repo_name(entry, train_config)
    assert name == f"poc-test-{entry.task_id[:8]}"


def test_local_checkpoint_path(manifest: Manifest, train_config: TrainConfig):
    entry = manifest.boss_battery[0]
    path = local_checkpoint_dir(entry, train_config)
    assert path == train_config.outputs_dir / entry.task_id / expected_repo_name(entry, train_config)


def test_eval_repo_id_format(manifest: Manifest, train_config: TrainConfig):
    entry = manifest.boss_battery[0]
    train_config.hf_username = "myuser"
    repo = eval_repo_id(entry, train_config)
    assert repo == f"myuser/{expected_repo_name(entry, train_config)}"


def test_stage_checkpoint_for_eval(tmp_path: Path, manifest: Manifest, train_config: TrainConfig):
    entry = manifest.boss_battery[0]
    checkpoint = local_checkpoint_dir(entry, train_config)
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}")
    (checkpoint / "success.txt").write_text("ok")

    repo_id = "poc-local/test-model"
    stage_checkpoint_for_eval(checkpoint, repo_id)

    cache_path = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--poc-local--test-model"
        / "snapshots"
        / "poc-local"
    )
    assert cache_path.exists()
    assert (cache_path / "success.txt").exists()
