"""Shared dataset-type helpers for train and eval."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_ROOT = REPO_ROOT / "original"
if str(ORIGINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_ROOT))

from core.models.utility_models import DpoDatasetType
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import RewardFunction
from core.models.utility_models import TaskType

from poc.manifest import BossTaskEntry


def build_dataset_type(entry: BossTaskEntry):
    if entry.task_type == TaskType.INSTRUCTTEXTTASK.value:
        return InstructTextDatasetType(
            field_instruction=entry.field_instruction,
            field_input=entry.field_input,
            field_output=entry.field_output,
            field_system=entry.field_system,
            format=None,
            no_input_format=None,
        )
    if entry.task_type == TaskType.DPOTASK.value:
        return DpoDatasetType(
            field_prompt=entry.field_prompt,
            field_system=entry.field_system,
            field_chosen=entry.field_chosen,
            field_rejected=entry.field_rejected,
            prompt_format=None,
            chosen_format=None,
            rejected_format=None,
        )
    if entry.task_type == TaskType.GRPOTASK.value:
        rewards = []
        for rf in entry.reward_functions or []:
            rewards.append(
                RewardFunction(
                    reward_func=rf["reward_func"],
                    reward_weight=rf.get("reward_weight", 1.0),
                    func_hash=rf.get("func_hash"),
                    is_generic=rf.get("is_generic"),
                )
            )
        return GrpoDatasetType(field_prompt=entry.field_prompt, reward_functions=rewards)
    raise ValueError(f"Unsupported task type: {entry.task_type}")


def build_dataset_type_json(entry: BossTaskEntry) -> str:
    return json.dumps(build_dataset_type(entry).model_dump())


def training_env_vars(entry: BossTaskEntry) -> dict[str, str]:
    env: dict[str, str] = {}
    if entry.use_kl:
        env["USE_KL"] = "1"
        if entry.kl_coef is not None:
            env["KL_COEF"] = str(entry.kl_coef)
    return env
