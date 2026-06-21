"""Tests for dataset_utils using real manifest entries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "original"))

from poc.dataset_utils import build_dataset_type
from poc.dataset_utils import build_dataset_type_json
from poc.manifest import Manifest


@pytest.fixture
def manifest() -> Manifest:
    return Manifest.load(Path(__file__).parent.parent / "manifest.json")


def test_instruct_dataset_type_from_real_task(manifest: Manifest):
    entry = manifest.get_boss_task("d573960c-4c46-4d22-9469-55c8c20ddf00")
    dt = build_dataset_type(entry)
    assert dt.field_instruction == "instruct"
    assert dt.field_output == "output"


def test_grpo_dataset_type_has_reward_functions(manifest: Manifest):
    entry = manifest.get_boss_task("d5f8b49d-8bda-4e10-8c47-d452747a0210")
    dt = build_dataset_type(entry)
    assert len(dt.reward_functions) == 4
    payload = json.loads(build_dataset_type_json(entry))
    assert payload["field_prompt"] == "prompt"
    assert len(payload["reward_functions"]) == 4


def test_dpo_dataset_type_from_real_task(manifest: Manifest):
    entry = manifest.get_boss_task("c0d6068c-ed7a-4f41-ac54-7a95cea9b247")
    dt = build_dataset_type(entry)
    assert dt.field_prompt == "prompt"
    assert dt.field_chosen == "chosen"
    assert dt.field_rejected == "rejected"
