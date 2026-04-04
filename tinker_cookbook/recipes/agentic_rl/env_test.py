"""Tests for _normalize_number and MathAnswerReward in env.py."""

from __future__ import annotations

import asyncio

import pytest

from tinker_cookbook.recipes.agentic_rl.env import MathAnswerReward, _normalize_number


# ---------------------------------------------------------------------------
# _normalize_number
# ---------------------------------------------------------------------------


class TestNormalizeNumber:
    def test_plain_integer(self) -> None:
        assert _normalize_number("42") == "42"

    def test_negative_integer(self) -> None:
        assert _normalize_number("-7") == "-7"

    def test_decimal(self) -> None:
        assert _normalize_number("3.14") == "3.14"

    def test_integer_as_float(self) -> None:
        # 5.0 should normalize to "5"
        assert _normalize_number("5.0") == "5"

    def test_commas(self) -> None:
        assert _normalize_number("1,234") == "1234"
        assert _normalize_number("1,000,000") == "1000000"

    def test_dollar_sign(self) -> None:
        assert _normalize_number("$60") == "60"
        assert _normalize_number("$1,250.50") == "1250.5"

    def test_percent_sign(self) -> None:
        assert _normalize_number("25%") == "25"

    def test_whitespace(self) -> None:
        assert _normalize_number("  42  ") == "42"

    def test_embedded_in_text(self) -> None:
        assert _normalize_number("The answer is 396.") == "396"

    def test_negative_decimal(self) -> None:
        assert _normalize_number("-3.5") == "-3.5"

    def test_no_number(self) -> None:
        assert _normalize_number("hello world") is None

    def test_empty_string(self) -> None:
        assert _normalize_number("") is None


# ---------------------------------------------------------------------------
# MathAnswerReward
# ---------------------------------------------------------------------------


class TestMathAnswerReward:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _make_history_with_submission(self, answer: str) -> list[dict]:
        return [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Let me compute that."},
            {
                "role": "tool",
                "name": "submit_answer",
                "content": f'{{"status": "submitted", "answer": "{answer}"}}',
            },
        ]

    def test_correct_answer(self) -> None:
        reward_fn = MathAnswerReward(gold_answer="4")
        reward, metrics = self._run(
            reward_fn(self._make_history_with_submission("4"))
        )
        assert reward == 1.0
        assert metrics["correct"] == 1.0

    def test_wrong_answer(self) -> None:
        reward_fn = MathAnswerReward(gold_answer="4")
        reward, metrics = self._run(
            reward_fn(self._make_history_with_submission("5"))
        )
        assert reward == 0.0
        assert metrics["incorrect"] == 1.0

    def test_no_submission(self) -> None:
        reward_fn = MathAnswerReward(gold_answer="4")
        history = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        reward, metrics = self._run(reward_fn(history))
        assert reward == -0.5
        assert metrics["no_submission"] == 1.0

    def test_equivalent_formats(self) -> None:
        """Gold '1234' should match submitted '1,234'."""
        reward_fn = MathAnswerReward(gold_answer="1234")
        reward, _ = self._run(
            reward_fn(self._make_history_with_submission("1,234"))
        )
        assert reward == 1.0

    def test_dollar_sign_match(self) -> None:
        reward_fn = MathAnswerReward(gold_answer="60")
        reward, _ = self._run(
            reward_fn(self._make_history_with_submission("$60"))
        )
        assert reward == 1.0

    def test_unparseable_answer(self) -> None:
        reward_fn = MathAnswerReward(gold_answer="42")
        reward, metrics = self._run(
            reward_fn(self._make_history_with_submission("no idea"))
        )
        assert reward == 0.0
        assert metrics["unparseable_answer"] == 1.0
