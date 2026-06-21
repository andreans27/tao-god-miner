"""Unit tests for POC scoring logic (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "original"))

from poc.scoring import build_battery_report
from poc.scoring import compare_scores
from poc.scoring import get_progressive_threshold
from poc.scoring import is_higher_better


class TestCompareScores:
    def test_instruct_lower_wins(self):
        result = compare_scores(
            "task-1",
            "InstructTextTask",
            your_score=0.82,
            opponent_score=0.83,
            threshold=0.0,
        )
        assert result.you_won is True
        assert result.margin_pct is not None
        assert result.margin_pct > 0

    def test_instruct_higher_loses(self):
        result = compare_scores(
            "task-1",
            "InstructTextTask",
            your_score=0.475,
            opponent_score=0.470,
            threshold=0.0,
        )
        assert result.you_won is False

    def test_grpo_higher_wins(self):
        result = compare_scores(
            "task-2",
            "GrpoTask",
            your_score=0.62,
            opponent_score=0.57,
            threshold=0.0,
        )
        assert result.you_won is True

    def test_grpo_lower_loses(self):
        result = compare_scores(
            "task-2",
            "GrpoTask",
            your_score=0.14,
            opponent_score=0.24,
            threshold=0.0,
        )
        assert result.you_won is False

    def test_dpo_lower_wins(self):
        result = compare_scores(
            "task-3",
            "DpoTask",
            your_score=0.031,
            opponent_score=0.033,
            threshold=0.0,
        )
        assert result.you_won is True

    def test_real_boss_round_instruct_loss(self):
        """Reproduce last tournament instruct task where challenger lost."""
        result = compare_scores(
            "d573960c-4c46-4d22-9469-55c8c20ddf00",
            "InstructTextTask",
            your_score=0.4750213623046875,
            opponent_score=0.47043725848197937,
            opponent_label="boss",
            threshold=0.0,
        )
        assert result.you_won is False

    def test_real_boss_round_grpo_win(self):
        """Reproduce last tournament GRPO task where challenger won."""
        result = compare_scores(
            "d5f8b49d-8bda-4e10-8c47-d452747a0210",
            "GrpoTask",
            your_score=0.618167835551044,
            opponent_score=0.5652494533076309,
            opponent_label="boss",
            threshold=0.0,
        )
        assert result.you_won is True


class TestBatteryReport:
    def test_boss_battery_pass_at_four_wins(self):
        comparisons = [
            compare_scores("a", "GrpoTask", 1.0, 0.5),
            compare_scores("b", "GrpoTask", 1.0, 0.5),
            compare_scores("c", "GrpoTask", 1.0, 0.5),
            compare_scores("d", "GrpoTask", 1.0, 0.5),
            compare_scores("e", "InstructTextTask", 1.0, 0.5),
            compare_scores("f", "InstructTextTask", 0.6, 0.5),
        ]
        report = build_battery_report("boss-battery", "tourn_test", comparisons, wins_required=4)
        assert report.wins == 4
        assert report.passed is True

    def test_regression_requires_all_wins(self):
        comparisons = [
            compare_scores("a", "InstructTextTask", 0.5, 0.6),
            compare_scores("b", "DpoTask", 0.5, 0.6),
            compare_scores("c", "GrpoTask", 0.6, 0.5),
        ]
        report = build_battery_report("regression", "tourn_test", comparisons, wins_required=3)
        assert report.wins == 3
        assert report.passed is True
        assert report.confidence == "READY"


class TestHelpers:
    def test_is_higher_better_grpo_only(self):
        assert is_higher_better("GrpoTask") is True
        assert is_higher_better("InstructTextTask") is False
        assert is_higher_better("DpoTask") is False

    def test_progressive_threshold_currently_zero(self):
        assert get_progressive_threshold(1) == 0.0
        assert get_progressive_threshold(5) == 0.0
