#!/usr/bin/env python3
"""Text tournament POC harness — train, eval, and compare vs last tournament winner."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.eval_runner import cache_all_fixtures
from poc.eval_runner import check_validator_image
from poc.eval_runner import evaluate_models_on_task
from poc.manifest import Manifest
from poc.manifest import refresh_manifest
from poc.scoring import build_battery_report
from poc.scoring import compare_scores
from poc.train_runner import TrainConfig
from poc.train_runner import TrainResult
from poc.train_runner import build_train_images
from poc.train_runner import train_task
from poc.train_runner import write_train_report


RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINE_TOLERANCE = 0.02
DEFAULT_TRAINER_REPO = REPO_ROOT / "miner-text"

TRAIN_MODES = frozenset({
    "build-trainer",
    "train-smoke",
    "train-regression",
    "train-boss",
    "train-eval-smoke",
    "train-eval-regression",
    "train-eval-boss",
})

EVAL_MODES = frozenset({
    "cache-fixtures",
    "verify-baseline",
    "smoke",
    "regression",
    "boss-battery",
})


def _resolve_opponent_repo(entry, opponent: str) -> str | None:
    if opponent == "winner":
        return entry.winner_repo
    if opponent == "boss":
        return entry.boss_repo
    if opponent == "api_score":
        return None
    raise ValueError(f"Unknown opponent mode: {opponent}")


def _task_entries_for_mode(manifest: Manifest, mode: str) -> list:
    if mode in {"train-smoke", "train-eval-smoke", "smoke"}:
        return [manifest.get_boss_task(manifest.smoke["task_id"])]
    if mode in {"train-regression", "train-eval-regression", "regression"}:
        return [manifest.get_boss_task(t["task_id"]) for t in manifest.regression_suite]
    if mode in {"train-boss", "train-eval-boss", "boss-battery", "verify-baseline"}:
        return list(manifest.boss_battery)
    if mode == "cache-fixtures":
        return list(manifest.boss_battery)
    raise ValueError(f"No task list for mode={mode}")


def _build_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        trainer_repo=args.trainer_repo,
        trainer_dockerfile=args.trainer_dockerfile,
        gpu_ids=args.gpu_ids,
        skip_build=args.skip_build,
        hf_username=args.hf_username or os.environ.get("HUGGINGFACE_USERNAME"),
        hf_token=args.hf_token or os.environ.get("HUGGINGFACE_TOKEN"),
        wandb_token=args.wandb_token or os.environ.get("WANDB_TOKEN"),
        hours_override=args.hours_override,
        expected_repo_prefix=args.repo_prefix,
    )


async def _eval_comparisons(
    entries,
    your_models: dict[str, str],
    gpu_ids: list[int],
    use_cache: bool,
    opponent: str,
    manifest: Manifest,
) -> list:
    comparisons = []
    for entry in entries:
        your_model = your_models.get(entry.task_id)
        if not your_model:
            continue

        opponent_repo = _resolve_opponent_repo(entry, opponent)
        models = [your_model]
        if opponent_repo:
            models.append(opponent_repo)

        print(f"\n▶ Evaluating task {entry.task_id} ({entry.task_type})")
        print(f"  Your model: {your_model}")
        if opponent_repo:
            print(f"  Opponent:   {opponent_repo}")

        task_result = await evaluate_models_on_task(
            entry, models=models, gpu_ids=gpu_ids, use_cache=use_cache
        )

        your = task_result.results.get(your_model)
        if your is None or your.error:
            comparisons.append(
                compare_scores(
                    entry.task_id,
                    entry.task_type,
                    float("nan"),
                    float("nan"),
                    opponent_label=opponent,
                    threshold=entry.threshold_used,
                )
            )
            continue

        if opponent_repo:
            opp = task_result.results.get(opponent_repo)
            opp_score = opp.eval_loss if opp and not opp.error else float("nan")
        else:
            opp_score = entry.challenger_score or float("nan")

        comparison = compare_scores(
            entry.task_id,
            entry.task_type,
            your.eval_loss,
            opp_score,
            opponent_label=opponent,
            threshold=entry.threshold_used,
        )
        comparison.your_is_finetune = your.is_finetune
        comparisons.append(comparison)

    return comparisons


async def _run_train_mode(
    mode: str,
    manifest: Manifest,
    train_config: TrainConfig,
    upload_hf: bool,
    task_id: str | None,
) -> tuple[list[TrainResult], int]:
    if mode == "build-trainer":
        build_train_images(train_config)
        print("Trainer images built.")
        return [], 0

    entries = _task_entries_for_mode(manifest, mode.replace("train-eval-", "train-"))
    if task_id:
        entries = [manifest.get_boss_task(task_id)]

    results: list[TrainResult] = []
    for entry in entries:
        result = await asyncio.to_thread(
            train_task, entry, train_config, upload_hf=upload_hf
        )
        results.append(result)
        status = "OK" if result.success else f"FAIL: {result.error}"
        print(f"  Train {entry.task_id[:8]}… -> {status}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_train_report(results, RESULTS_DIR / f"train_{mode}_{ts}.json")

    failures = sum(1 for r in results if not r.success)
    return results, failures


async def _run_eval_mode(
    mode: str,
    manifest: Manifest,
    your_model: str | None,
    gpu_ids: list[int],
    use_cache: bool,
    opponent: str,
) -> int:
    if mode == "cache-fixtures":
        paths = await cache_all_fixtures(manifest)
        print(f"Cached {len(paths)} test fixtures to poc/fixtures/")
        for tid, path in paths.items():
            print(f"  {tid} -> {path}")
        return 0

    check_validator_image()
    entries = _task_entries_for_mode(manifest, mode)

    if mode == "verify-baseline":
        return await _verify_baseline(entries, gpu_ids, use_cache)

    if mode == "smoke":
        return await _run_smoke(entries[0], your_model, gpu_ids, use_cache)

    if not your_model:
        raise ValueError(f"--model required for mode={mode}")

    comparisons = await _eval_comparisons(
        entries, {e.task_id: your_model for e in entries},
        gpu_ids, use_cache, opponent, manifest,
    )

    wins_required = (
        manifest.boss_round_wins_required if mode == "boss-battery" else len(comparisons)
    )
    report = build_battery_report(mode, manifest.tournament_id, comparisons, wins_required)
    for line in report.summary_lines():
        print(line)
    _write_report(report, your_model, opponent)
    return 0 if report.passed else 1


async def _run_train_eval_mode(
    mode: str,
    manifest: Manifest,
    train_config: TrainConfig,
    gpu_ids: list[int],
    use_cache: bool,
    opponent: str,
    upload_hf: bool,
    task_id: str | None,
) -> int:
    train_mode = mode.replace("train-eval-", "train-")
    train_results, train_failures = await _run_train_mode(
        train_mode, manifest, train_config, upload_hf, task_id
    )
    if train_failures:
        print(f"\nTraining failed on {train_failures} task(s); skipping eval.")
        return 1

    check_validator_image()
    your_models = {
        r.task_id: r.eval_repo_id for r in train_results if r.success
    }
    eval_mode = mode.replace("train-eval-", "")
    entries = _task_entries_for_mode(manifest, eval_mode)

    comparisons = await _eval_comparisons(
        entries, your_models, gpu_ids, use_cache, opponent, manifest,
    )

    wins_required = (
        manifest.boss_round_wins_required if "boss" in mode else len(comparisons)
    )
    report = build_battery_report(eval_mode, manifest.tournament_id, comparisons, wins_required)
    for line in report.summary_lines():
        print(line)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _write_report(report, "trained-local", opponent, suffix=f"train_eval_{ts}")
    return 0 if report.passed else 1


async def _verify_baseline(entries, gpu_ids, use_cache) -> int:
    print("Verifying harness: re-eval winner repos vs API challenger_score\n")
    failures = 0
    for entry in entries:
        if not entry.winner_repo:
            print(f"  SKIP {entry.task_id}: no winner_repo")
            failures += 1
            continue
        task_result = await evaluate_models_on_task(
            entry, models=[entry.winner_repo], gpu_ids=gpu_ids, use_cache=use_cache
        )
        result = task_result.results.get(entry.winner_repo)
        if result is None or result.error:
            print(f"  FAIL {entry.task_id}: {result.error if result else 'missing'}")
            failures += 1
            continue
        api_score = entry.challenger_score
        rel_diff = abs(result.eval_loss - api_score) / abs(api_score) if api_score else 0
        ok = rel_diff <= BASELINE_TOLERANCE
        print(
            f"  {'OK' if ok else 'DRIFT'} {entry.task_type:18} {entry.task_id[:8]}…  "
            f"eval={result.eval_loss:.6f} api={api_score:.6f} drift={rel_diff * 100:.2f}%"
        )
        if not ok:
            failures += 1
    return 0 if failures == 0 else 1


async def _run_smoke(entry, your_model, gpu_ids, use_cache) -> int:
    models = []
    if entry.winner_repo:
        models.append(entry.winner_repo)
    if your_model:
        models.append(your_model)
    if not models:
        raise ValueError("Smoke needs winner_repo or --model")

    task_result = await evaluate_models_on_task(
        entry, models=models, gpu_ids=gpu_ids, use_cache=use_cache
    )
    ok = True
    for name, result in task_result.results.items():
        if result.error:
            print(f"  FAIL {name}: {result.error}")
            ok = False
        else:
            print(f"  OK   {name}: loss={result.eval_loss:.6f} finetune={result.is_finetune}")
    return 0 if ok else 1


def _write_report(report, your_model, opponent, suffix: str | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = suffix or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"report_{report.mode}_{ts}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "your_model": your_model,
        "opponent": opponent,
        "mode": report.mode,
        "tournament_id": report.tournament_id,
        "wins": report.wins,
        "losses": report.losses,
        "wins_required": report.wins_required,
        "passed": report.passed,
        "confidence": report.confidence,
        "comparisons": [
            {
                "task_id": c.task_id,
                "task_type": c.task_type,
                "your_score": c.your_score,
                "opponent_score": c.opponent_score,
                "you_won": c.you_won,
                "margin_pct": c.margin_pct,
            }
            for c in report.comparisons
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nReport saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text tournament POC: train + eval vs last tournament winner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Train modes (need GPU + Docker):
  build-trainer         Build downloader + trainer docker images
  train-smoke           Train on smallest GRPO boss task
  train-regression      Train on 1 instruct + 1 DPO + 1 GRPO
  train-boss            Train on all 6 boss-round tasks
  train-eval-smoke      Train smoke task then eval vs winner
  train-eval-regression Train 3 tasks then eval vs winner
  train-eval-boss       Train 6 tasks then eval vs winner (full gate)

Eval-only modes:
  verify-baseline, smoke, regression, boss-battery, cache-fixtures

Examples:
  python poc/run_text_poc.py --mode build-trainer
  python poc/run_text_poc.py --mode train-eval-smoke --gpu-ids 0
  python poc/run_text_poc.py --mode train-eval-boss --gpu-ids 0 1 --trainer-repo winning_repos/text-winner-latest
        """,
    )
    all_modes = sorted(TRAIN_MODES | EVAL_MODES | {"refresh-manifest"})
    parser.add_argument("--mode", required=True, choices=all_modes)
    parser.add_argument("--model", help="HF repo for eval-only modes")
    parser.add_argument("--task-id", help="Train/eval a single manifest task")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifest.json")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--opponent", choices=["winner", "boss", "api_score"], default="winner")
    parser.add_argument("--trainer-repo", type=Path, default=DEFAULT_TRAINER_REPO)
    parser.add_argument("--trainer-dockerfile", default="dockerfiles/standalone-text-trainer.dockerfile")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--upload-hf", action="store_true", help="Upload checkpoints to HF before eval")
    parser.add_argument("--hf-username", help="HF username (or HUGGINGFACE_USERNAME env)")
    parser.add_argument("--hf-token", help="HF token (or HUGGINGFACE_TOKEN env)")
    parser.add_argument("--wandb-token", help="WandB token (or WANDB_TOKEN env)")
    parser.add_argument("--hours-override", type=float, help="Override task hours_to_complete")
    parser.add_argument("--repo-prefix", default="poc", help="Prefix for expected_repo_name")
    args = parser.parse_args()

    if args.mode == "refresh-manifest":
        asyncio.run(refresh_manifest(args.manifest))
        print(f"Manifest refreshed: {args.manifest}")
        sys.exit(0)

    manifest = Manifest.load(args.manifest)
    print(f"Manifest: {manifest.tournament_id} (generated {manifest.generated_at})")

    train_config = _build_train_config(args)
    use_cache = not args.no_cache

    if args.mode in TRAIN_MODES:
        if args.mode.startswith("train-eval-"):
            code = asyncio.run(
                _run_train_eval_mode(
                    args.mode, manifest, train_config, args.gpu_ids,
                    use_cache, args.opponent, args.upload_hf, args.task_id,
                )
            )
        else:
            _, failures = asyncio.run(
                _run_train_mode(
                    args.mode, manifest, train_config, args.upload_hf, args.task_id
                )
            )
            code = 0 if failures == 0 else 1
        sys.exit(code)

    code = asyncio.run(
        _run_eval_mode(
            args.mode, manifest, args.model, args.gpu_ids, use_cache, args.opponent
        )
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
