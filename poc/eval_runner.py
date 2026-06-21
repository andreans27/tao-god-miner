"""Run validator-equivalent text evaluation for POC tasks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# POC lives beside original/; add repo root for validator imports.
REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_ROOT = REPO_ROOT / "original"
if str(ORIGINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_ROOT))

from core.models.payload_models import EvaluationResultText
from core.models.utility_models import FileFormat
from core.utils import download_s3_file
from validator.evaluation.local_evaluation import run_evaluation_docker_text

from poc.dataset_utils import build_dataset_type
from poc.manifest import BossTaskEntry
from poc.manifest import Manifest


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass
class ModelEvalResult:
    model: str
    eval_loss: float
    is_finetune: bool
    error: str | None = None


@dataclass
class TaskEvalResult:
    task_id: str
    task_type: str
    model_id: str
    results: dict[str, ModelEvalResult]


def _build_dataset_type(entry: BossTaskEntry):
    return build_dataset_type(entry)


async def _resolve_test_data_path(entry: BossTaskEntry, use_cache: bool) -> str:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURES_DIR / f"{entry.task_id}_test_data.json"

    if use_cache and fixture_path.exists():
        return str(fixture_path)

    if not entry.test_data_url:
        raise ValueError(f"Task {entry.task_id} has no test_data_url in manifest")

    downloaded = await download_s3_file(entry.test_data_url)
    if use_cache:
        fixture_path.write_bytes(Path(downloaded).read_bytes())
        os.remove(downloaded)
        return str(fixture_path)
    return downloaded


async def evaluate_models_on_task(
    entry: BossTaskEntry,
    models: list[str],
    gpu_ids: list[int],
    *,
    use_cache: bool = True,
) -> TaskEvalResult:
    """Evaluate one or more HF repos on a task's test split (validator docker)."""
    test_path = await _resolve_test_data_path(entry, use_cache=use_cache)
    dataset_type = _build_dataset_type(entry)

    try:
        docker_results = await run_evaluation_docker_text(
            dataset=test_path,
            models=models,
            original_model=entry.model_id,
            dataset_type=dataset_type,
            file_format=FileFormat.JSON,
            gpu_ids=gpu_ids,
        )
    finally:
        if not use_cache and os.path.exists(test_path):
            os.remove(test_path)

    parsed: dict[str, ModelEvalResult] = {}
    for model_name, raw in docker_results.results.items():
        if isinstance(raw, Exception):
            parsed[model_name] = ModelEvalResult(
                model=model_name,
                eval_loss=float("nan"),
                is_finetune=False,
                error=str(raw),
            )
            continue
        if not isinstance(raw, EvaluationResultText):
            parsed[model_name] = ModelEvalResult(
                model=model_name,
                eval_loss=float("nan"),
                is_finetune=False,
                error=f"Unexpected result type: {type(raw)}",
            )
            continue
        parsed[model_name] = ModelEvalResult(
            model=model_name,
            eval_loss=raw.eval_loss,
            is_finetune=raw.is_finetune,
        )

    return TaskEvalResult(
        task_id=entry.task_id,
        task_type=entry.task_type,
        model_id=entry.model_id,
        results=parsed,
    )


async def cache_all_fixtures(manifest: Manifest) -> dict[str, str]:
    """Download and cache test fixtures for every boss-battery task."""
    paths: dict[str, str] = {}
    for entry in manifest.boss_battery:
        path = await _resolve_test_data_path(entry, use_cache=True)
        paths[entry.task_id] = path
    return paths


def check_validator_image() -> None:
    """Warn if validator docker image missing (eval will fail)."""
    import subprocess

    result = subprocess.run(
        ["docker", "images", "-q", "weightswandering/tuning_vali:latest"],
        capture_output=True,
        text=True,
        check=False,
    )
    if not result.stdout.strip():
        raise RuntimeError(
            "Validator image not found. Build it first:\n"
            "  cd original && docker build -f dockerfiles/validator.dockerfile "
            "-t weightswandering/tuning_vali:latest ."
        )
