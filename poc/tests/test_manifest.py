"""Unit tests for manifest loading (real manifest.json, no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from poc.manifest import Manifest


MANIFEST_PATH = Path(__file__).resolve().parent.parent / "manifest.json"


@pytest.fixture
def manifest() -> Manifest:
    assert MANIFEST_PATH.exists(), "Run poc/fetch_manifest.py first"
    return Manifest.load(MANIFEST_PATH)


def test_manifest_has_real_tournament_id(manifest: Manifest):
    assert manifest.tournament_id.startswith("tourn_")
    assert manifest.source == "https://api.gradients.io"


def test_boss_battery_has_six_real_tasks(manifest: Manifest):
    assert len(manifest.boss_battery) == 6
    types = {t.task_type for t in manifest.boss_battery}
    assert types == {"InstructTextTask", "DpoTask", "GrpoTask"}
    for entry in manifest.boss_battery:
        assert entry.task_id
        assert entry.model_id
        assert entry.test_data_url
        assert entry.winner_repo
        assert entry.challenger_score is not None


def test_regression_suite_covers_all_types(manifest: Manifest):
    assert len(manifest.regression_suite) == 3
    types = {t["task_type"] for t in manifest.regression_suite}
    assert types == {"InstructTextTask", "DpoTask", "GrpoTask"}


def test_smoke_uses_smallest_grpo_task(manifest: Manifest):
    assert manifest.smoke["task_type"] == "GrpoTask"
    assert manifest.smoke["model_id"]
    assert manifest.smoke["winner_repo"]


def test_manifest_json_roundtrip(manifest: Manifest, tmp_path: Path):
    out = tmp_path / "manifest_copy.json"
    manifest.save(out)
    reloaded = Manifest.load(out)
    assert reloaded.tournament_id == manifest.tournament_id
    assert len(reloaded.boss_battery) == len(manifest.boss_battery)
