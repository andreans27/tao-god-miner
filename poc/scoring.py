"""Scoring logic mirroring validator tournament boss-round rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Mirror validator/tournament/constants.py (thresholds currently disabled on mainnet).
EXPONENTIAL_BASE_THRESHOLD = 0.0
EXPONENTIAL_DECAY_RATE = 0.8
EXPONENTIAL_MIN_THRESHOLD = 0.0

HIGHER_IS_BETTER_TYPES = frozenset({"GrpoTask", "EnvTask"})


def get_progressive_threshold(consecutive_wins: int) -> float:
    """Progressive boss defense threshold (text/image tournaments)."""
    current = EXPONENTIAL_BASE_THRESHOLD * (EXPONENTIAL_DECAY_RATE ** max(consecutive_wins - 1, 0))
    return max(EXPONENTIAL_MIN_THRESHOLD, current)


def is_higher_better(task_type: str) -> bool:
    return task_type in HIGHER_IS_BETTER_TYPES


@dataclass
class TaskComparison:
    task_id: str
    task_type: str
    your_score: float
    opponent_score: float
    opponent_label: str
    threshold: float
    you_won: bool
    margin_pct: float | None
    your_is_finetune: bool | None = None
    opponent_is_finetune: bool | None = None


def compare_scores(
    task_id: str,
    task_type: str,
    your_score: float,
    opponent_score: float,
    *,
    opponent_label: str = "winner",
    threshold: float = 0.0,
) -> TaskComparison:
    """Return whether *your* submission beats *opponent* on this task.

  Text instruct/DPO/chat: lower eval_loss wins.
  GRPO: higher eval_loss (reward) wins.
  Threshold mirrors performance_calculator.py boss-round logic.
    """
    higher = is_higher_better(task_type)

    if higher:
        if opponent_score != 0:
            margin_pct = (your_score - opponent_score) / abs(opponent_score)
        else:
            margin_pct = 0.0
        you_won = your_score > opponent_score * (1 + threshold)
    else:
        if your_score != 0:
            margin_pct = (opponent_score - your_score) / abs(your_score)
        else:
            margin_pct = 0.0
        you_won = your_score < opponent_score * (1 - threshold)

    return TaskComparison(
        task_id=task_id,
        task_type=task_type,
        your_score=your_score,
        opponent_score=opponent_score,
        opponent_label=opponent_label,
        threshold=threshold,
        you_won=you_won,
        margin_pct=margin_pct,
    )


@dataclass
class BatteryReport:
    mode: str
    tournament_id: str
    comparisons: list[TaskComparison]
    wins: int
    losses: int
    wins_required: int
    passed: bool
    confidence: Literal["READY", "NOT_READY", "PARTIAL"]

    def summary_lines(self) -> list[str]:
        lines = [
            f"TEXT POC — {self.mode}",
            f"Tournament: {self.tournament_id}",
            "━" * 48,
        ]
        for c in self.comparisons:
            status = "WIN" if c.you_won else "LOSS"
            margin = f"{c.margin_pct * 100:+.2f}%" if c.margin_pct is not None else "n/a"
            lines.append(
                f"  {c.task_type:18} {c.task_id[:8]}…  "
                f"you={c.your_score:.6f}  {c.opponent_label}={c.opponent_score:.6f}  "
                f"{status} ({margin})"
            )
        lines.append("━" * 48)
        lines.append(f"Record: {self.wins}/{len(self.comparisons)} wins (need {self.wins_required})")
        lines.append(f"CONFIDENCE: {self.confidence}")
        lines.append(f"GATE: {'PASS' if self.passed else 'FAIL'}")
        return lines


def build_battery_report(
    mode: str,
    tournament_id: str,
    comparisons: list[TaskComparison],
    wins_required: int,
) -> BatteryReport:
    wins = sum(1 for c in comparisons if c.you_won)
    losses = len(comparisons) - wins
    passed = wins >= wins_required

    # Boss round needs majority (4/6); regression needs all task types.
    if mode == "boss-battery":
        confidence: Literal["READY", "NOT_READY", "PARTIAL"] = (
            "READY" if passed else "PARTIAL" if wins >= wins_required - 1 else "NOT_READY"
        )
    elif mode == "regression":
        passed = wins == len(comparisons) and len(comparisons) > 0
        confidence = "READY" if passed else "NOT_READY"
    else:
        confidence = "READY" if passed else "NOT_READY"

    return BatteryReport(
        mode=mode,
        tournament_id=tournament_id,
        comparisons=comparisons,
        wins=wins,
        losses=losses,
        wins_required=wins_required,
        passed=passed,
        confidence=confidence,
    )
