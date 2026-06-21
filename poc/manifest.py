"""Load and refresh POC manifest from Gradients API (real tournament data)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

API_BASE = "https://api.gradients.io"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"


@dataclass
class BossTaskEntry:
    task_id: str
    task_type: str
    model_id: str
    hours_to_complete: float
    test_data_url: str | None
    training_data_url: str | None
    winner_hotkey: str
    winner_repo: str | None
    boss_repo: str | None
    boss_score: float | None
    challenger_score: float | None
    threshold_used: float
    challenger_won: bool | None
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None
    field_system: str | None = None
    field_prompt: str | None = None
    field_chosen: str | None = None
    field_rejected: str | None = None
    reward_functions: list[dict[str, Any]] | None = None
    use_kl: bool | None = None
    kl_coef: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BossTaskEntry:
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass
class Manifest:
    generated_at: str
    source: str
    tournament_id: str
    winner_hotkey: str
    base_winner_hotkey: str
    boss_round_wins_required: int
    boss_round_total_tasks: int
    boss_battery: list[BossTaskEntry]
    regression_suite: list[dict[str, str]]
    smoke: dict[str, Any]

    @classmethod
    def load(cls, path: Path | None = None) -> Manifest:
        path = path or DEFAULT_MANIFEST_PATH
        with open(path) as f:
            raw = json.load(f)
        battery = [BossTaskEntry.from_dict(item) for item in raw["boss_battery"]]
        return cls(
            generated_at=raw["generated_at"],
            source=raw["source"],
            tournament_id=raw["tournament_id"],
            winner_hotkey=raw["winner_hotkey"],
            base_winner_hotkey=raw["base_winner_hotkey"],
            boss_round_wins_required=raw["boss_round_wins_required"],
            boss_round_total_tasks=raw["boss_round_total_tasks"],
            boss_battery=battery,
            regression_suite=raw["regression_suite"],
            smoke=raw["smoke"],
        )

    def save(self, path: Path | None = None) -> None:
        path = path or DEFAULT_MANIFEST_PATH
        payload = {
            "generated_at": self.generated_at,
            "source": self.source,
            "tournament_id": self.tournament_id,
            "winner_hotkey": self.winner_hotkey,
            "base_winner_hotkey": self.base_winner_hotkey,
            "boss_round_wins_required": self.boss_round_wins_required,
            "boss_round_total_tasks": self.boss_round_total_tasks,
            "boss_battery": [entry.__dict__ for entry in self.boss_battery],
            "regression_suite": self.regression_suite,
            "smoke": self.smoke,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    def task_ids_for_mode(self, mode: str) -> list[str]:
        if mode == "boss-battery":
            return [t.task_id for t in self.boss_battery]
        if mode == "regression":
            return [t["task_id"] for t in self.regression_suite]
        if mode == "smoke":
            return [self.smoke["task_id"]]
        if mode == "verify-baseline":
            return [t.task_id for t in self.boss_battery]
        raise ValueError(f"Unknown mode: {mode}")

    def get_boss_task(self, task_id: str) -> BossTaskEntry:
        for entry in self.boss_battery:
            if entry.task_id == task_id:
                return entry
        raise KeyError(f"Task {task_id} not in manifest boss_battery")


async def fetch_task_details(client: httpx.AsyncClient, task_id: str) -> dict[str, Any]:
    response = await client.get(f"{API_BASE}/auditing/tasks/{task_id}")
    response.raise_for_status()
    return response.json()


async def refresh_manifest(path: Path | None = None) -> Manifest:
    """Pull last text boss-round tasks from API and write manifest.json."""
    path = path or DEFAULT_MANIFEST_PATH

    async with httpx.AsyncClient(timeout=120) as client:
        boss_resp = await client.get(f"{API_BASE}/v1/performance/last-boss-battle")
        boss_resp.raise_for_status()
        boss_data = boss_resp.json()

        tour_resp = await client.get(f"{API_BASE}/tournament/latest/details")
        tour_resp.raise_for_status()
        tour_data = tour_resp.json()["text"]

        winner_hotkey = tour_data["winner_hotkey"]
        base_winner_hotkey = tour_data["base_winner_hotkey"]
        tournament_id = boss_data["text_tournament_id"]

        boss_battery: list[BossTaskEntry] = []
        for perf in boss_data["text_performance_differences"]:
            task_id = perf["task_id"]
            task = await fetch_task_details(client, task_id)

            winner_repo = None
            boss_repo = None
            for hd in task.get("hotkey_details") or []:
                if hd["hotkey"] == winner_hotkey:
                    winner_repo = hd.get("repo")
                elif hd["hotkey"] != winner_hotkey:
                    boss_repo = hd.get("repo")

            boss_battery.append(
                BossTaskEntry(
                    task_id=task_id,
                    task_type=perf["task_type"],
                    model_id=task["model_id"],
                    hours_to_complete=task["hours_to_complete"],
                    test_data_url=task.get("test_data"),
                    training_data_url=task.get("training_data"),
                    winner_hotkey=winner_hotkey,
                    winner_repo=winner_repo,
                    boss_repo=boss_repo,
                    boss_score=perf.get("boss_score"),
                    challenger_score=perf.get("challenger_score"),
                    threshold_used=perf.get("threshold_used", 0.0),
                    challenger_won=perf.get("challenger_won"),
                    field_instruction=task.get("field_instruction"),
                    field_input=task.get("field_input"),
                    field_output=task.get("field_output"),
                    field_system=task.get("field_system"),
                    field_prompt=task.get("field_prompt"),
                    field_chosen=task.get("field_chosen"),
                    field_rejected=task.get("field_rejected"),
                    reward_functions=task.get("reward_functions"),
                    use_kl=task.get("use_kl"),
                    kl_coef=task.get("kl_coef"),
                )
            )

        seen_types: set[str] = set()
        regression_suite: list[dict[str, str]] = []
        for entry in boss_battery:
            if entry.task_type not in seen_types:
                seen_types.add(entry.task_type)
                regression_suite.append(
                    {
                        "task_id": entry.task_id,
                        "task_type": entry.task_type,
                        "winner_repo": entry.winner_repo or "",
                    }
                )

        grpo_tasks = [e for e in boss_battery if e.task_type == "GrpoTask"]
        smoke_source = min(grpo_tasks, key=lambda e: e.hours_to_complete) if grpo_tasks else boss_battery[0]
        smoke = {
            "task_id": smoke_source.task_id,
            "task_type": smoke_source.task_type,
            "model_id": smoke_source.model_id,
            "hours_to_complete": smoke_source.hours_to_complete,
            "winner_repo": smoke_source.winner_repo,
        }

    manifest = Manifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        source=API_BASE,
        tournament_id=tournament_id,
        winner_hotkey=winner_hotkey,
        base_winner_hotkey=base_winner_hotkey,
        boss_round_wins_required=4,
        boss_round_total_tasks=6,
        boss_battery=boss_battery,
        regression_suite=regression_suite,
        smoke=smoke,
    )
    manifest.save(path)
    return manifest
