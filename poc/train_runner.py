"""Docker-based training loop mirroring validator tournament execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_ROOT = REPO_ROOT / "original"
POC_ROOT = Path(__file__).resolve().parent

if str(ORIGINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_ROOT))

from poc.dataset_utils import build_dataset_type_json
from poc.dataset_utils import training_env_vars
from poc.manifest import BossTaskEntry

IMAGE_DOWNLOADER = "poc-trainer-downloader:latest"
IMAGE_TRAINER = "poc-standalone-text-trainer:latest"
IMAGE_UPLOADER = "poc-hf-uploader:latest"

DEFAULT_CACHE_DIR = POC_ROOT / "cache"
DEFAULT_OUTPUTS_DIR = POC_ROOT / "outputs"


@dataclass
class TrainConfig:
    trainer_repo: Path
    trainer_dockerfile: str = "dockerfiles/standalone-text-trainer.dockerfile"
    cache_dir: Path = DEFAULT_CACHE_DIR
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR
    gpu_ids: list[int] | None = None
    memory_gb: int = 64
    cpus: int = 8
    skip_build: bool = False
    hf_username: str | None = None
    hf_token: str | None = None
    wandb_token: str | None = None
    hours_override: float | None = None
    expected_repo_prefix: str = "poc"


@dataclass
class TrainResult:
    task_id: str
    task_type: str
    model_id: str
    expected_repo_name: str
    local_checkpoint_dir: Path
    success: bool
    eval_repo_id: str
    hf_repo_id: str | None = None
    error: str | None = None


def _run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env, check=check)


def _docker_image_exists(tag: str) -> bool:
    result = subprocess.run(["docker", "images", "-q", tag], capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())


def _gpu_device_arg(gpu_ids: list[int]) -> str:
    return ",".join(str(i) for i in gpu_ids)


def build_train_images(config: TrainConfig) -> None:
    if config.skip_build:
        return

    _run(
        [
            "docker", "build",
            "-f", str(ORIGINAL_ROOT / "dockerfiles/trainer-downloader.dockerfile"),
            "-t", IMAGE_DOWNLOADER,
            str(ORIGINAL_ROOT),
        ]
    )
    _run(
        [
            "docker", "build",
            "-f", str(config.trainer_repo / config.trainer_dockerfile),
            "-t", IMAGE_TRAINER,
            str(config.trainer_repo),
        ]
    )
    if config.hf_token and config.hf_username:
        _run(
            [
                "docker", "build",
                "-f", str(ORIGINAL_ROOT / "dockerfiles/hf-uploader.dockerfile"),
                "-t", IMAGE_UPLOADER,
                str(ORIGINAL_ROOT),
            ]
        )


def ensure_train_dirs(config: TrainConfig) -> None:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(config.cache_dir, 0o700)
    os.chmod(config.outputs_dir, 0o700)


def expected_repo_name(entry: BossTaskEntry, config: TrainConfig) -> str:
    return f"{config.expected_repo_prefix}-{entry.task_id[:8]}"


def local_checkpoint_dir(entry: BossTaskEntry, config: TrainConfig) -> Path:
    return config.outputs_dir / entry.task_id / expected_repo_name(entry, config)


def eval_repo_id(entry: BossTaskEntry, config: TrainConfig) -> str:
    username = config.hf_username or "poc-local"
    return f"{username}/{expected_repo_name(entry, config)}"


def stage_checkpoint_for_eval(local_dir: Path, repo_id: str) -> None:
    """Stage a local checkpoint into the HF hub cache layout for validator eval."""
    if not local_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {local_dir}")

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_root / f"models--{repo_id.replace('/', '--')}"
    snapshot_dir = model_cache / "snapshots" / "poc-local"
    refs_main = model_cache / "refs" / "main"

    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    refs_main.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(local_dir, snapshot_dir, dirs_exist_ok=True)
    refs_main.write_text("poc-local\n")
    print(f"Staged checkpoint for eval: {local_dir} -> {snapshot_dir} (repo_id={repo_id})")


def _download_task_assets(entry: BossTaskEntry, config: TrainConfig) -> None:
    if not entry.training_data_url:
        raise ValueError(f"Task {entry.task_id} missing training_data_url")

    _run(
        [
            "docker", "run", "--rm",
            "--volume", f"{config.cache_dir.resolve()}:/cache:rw",
            IMAGE_DOWNLOADER,
            "--task-id", entry.task_id,
            "--model", entry.model_id,
            "--dataset", entry.training_data_url,
            "--file-format", "s3",
            "--task-type", entry.task_type,
        ]
    )


def _run_training_container(entry: BossTaskEntry, config: TrainConfig, repo_name: str) -> None:
    hours = config.hours_override if config.hours_override is not None else entry.hours_to_complete
    gpu_ids = config.gpu_ids or [0]

    dataset_type_json = build_dataset_type_json(entry)
    env = os.environ.copy()
    env.update(training_env_vars(entry))

    cmd = [
        "docker", "run", "--rm",
        "--gpus", f"device={_gpu_device_arg(gpu_ids)}",
        "--security-opt=no-new-privileges",
        "--cap-drop=ALL",
        "--memory", f"{config.memory_gb}g",
        "--cpus", str(config.cpus),
        "--network", "none",
        "--volume", f"{config.cache_dir.resolve()}:/cache:ro",
        "--volume", f"{config.outputs_dir.resolve()}:/app/checkpoints:rw",
        "--name", f"poc-trainer-{entry.task_id[:8]}-{uuid.uuid4().hex[:6]}",
    ]
    for key, value in training_env_vars(entry).items():
        cmd.extend(["--env", f"{key}={value}"])

    cmd.extend(
        [
            IMAGE_TRAINER,
            "--task-id", entry.task_id,
            "--model", entry.model_id,
            "--dataset", entry.training_data_url,
            "--dataset-type", dataset_type_json,
            "--task-type", entry.task_type,
            "--file-format", "s3",
            "--expected-repo-name", repo_name,
            "--hours-to-complete", str(hours),
        ]
    )
    _run(cmd, env=env)


def _upload_to_hf(entry: BossTaskEntry, config: TrainConfig, repo_name: str) -> str:
    if not config.hf_token or not config.hf_username:
        raise ValueError("HF upload requires --hf-token and --hf-username")

    local_folder = f"/app/checkpoints/{entry.task_id}/{repo_name}"
    hf_repo = f"{config.hf_username}/{repo_name}"

    _run(
        [
            "docker", "run", "--rm",
            "--volume", f"{config.outputs_dir.resolve()}:/app/checkpoints:rw",
            "--env", f"HUGGINGFACE_TOKEN={config.hf_token}",
            "--env", f"HUGGINGFACE_USERNAME={config.hf_username}",
            "--env", f"WANDB_TOKEN={config.wandb_token or ''}",
            "--env", f"TASK_ID={entry.task_id}",
            "--env", f"EXPECTED_REPO_NAME={repo_name}",
            "--env", f"LOCAL_FOLDER={local_folder}",
            IMAGE_UPLOADER,
        ]
    )
    return hf_repo


def train_task(
    entry: BossTaskEntry,
    config: TrainConfig,
    *,
    upload_hf: bool = False,
) -> TrainResult:
    """Download assets, train, optionally upload, stage for local eval."""
    ensure_train_dirs(config)
    if not config.skip_build and not _docker_image_exists(IMAGE_TRAINER):
        build_train_images(config)

    repo_name = expected_repo_name(entry, config)
    checkpoint = local_checkpoint_dir(entry, config)
    eval_id = eval_repo_id(entry, config)

    try:
        print(f"\n▶ Training task {entry.task_id} ({entry.task_type})")
        print(f"  Base model: {entry.model_id}")
        print(f"  Hours: {config.hours_override or entry.hours_to_complete}")
        print(f"  Trainer repo: {config.trainer_repo}")

        _download_task_assets(entry, config)
        _run_training_container(entry, config, repo_name)

        success_file = checkpoint / "success.txt"
        success = success_file.exists() and any(checkpoint.iterdir())
        if not success:
            return TrainResult(
                task_id=entry.task_id,
                task_type=entry.task_type,
                model_id=entry.model_id,
                expected_repo_name=repo_name,
                local_checkpoint_dir=checkpoint,
                success=False,
                eval_repo_id=eval_id,
                error=f"Training finished without success.txt in {checkpoint}",
            )

        hf_repo_id = None
        if upload_hf:
            hf_repo_id = _upload_to_hf(entry, config, repo_name)
            eval_id = hf_repo_id
        else:
            stage_checkpoint_for_eval(checkpoint, eval_id)

        return TrainResult(
            task_id=entry.task_id,
            task_type=entry.task_type,
            model_id=entry.model_id,
            expected_repo_name=repo_name,
            local_checkpoint_dir=checkpoint,
            success=True,
            eval_repo_id=eval_id,
            hf_repo_id=hf_repo_id,
        )
    except subprocess.CalledProcessError as exc:
        return TrainResult(
            task_id=entry.task_id,
            task_type=entry.task_type,
            model_id=entry.model_id,
            expected_repo_name=repo_name,
            local_checkpoint_dir=checkpoint,
            success=False,
            eval_repo_id=eval_id,
            error=str(exc),
        )


def write_train_report(results: list[TrainResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [
            {
                "task_id": r.task_id,
                "task_type": r.task_type,
                "model_id": r.model_id,
                "expected_repo_name": r.expected_repo_name,
                "local_checkpoint_dir": str(r.local_checkpoint_dir),
                "success": r.success,
                "eval_repo_id": r.eval_repo_id,
                "hf_repo_id": r.hf_repo_id,
                "error": r.error,
            }
            for r in results
        ]
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
