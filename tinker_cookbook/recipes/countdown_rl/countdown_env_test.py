"""Unit tests for Countdown environment reward logic."""

from tinker_cookbook.recipes.countdown_rl.countdown_env import (
    evaluate_countdown_expression,
    extract_answer,
)


def test_evaluate_valid_expression():
    assert evaluate_countdown_expression("44 + 19 + 35", [44, 19, 35], 98)


def test_evaluate_with_multiplication():
    assert evaluate_countdown_expression("3 * 2 + 7", [3, 7, 2], 13)


def test_evaluate_with_parentheses():
    assert evaluate_countdown_expression("(10 + 5) * 2", [10, 5, 2], 30)


def test_evaluate_wrong_result():
    assert not evaluate_countdown_expression("44 + 19", [44, 19, 35], 98)


def test_evaluate_reuses_number():
    # Uses 44 twice but only one 44 is available
    assert not evaluate_countdown_expression("44 + 44 + 10", [44, 19, 35], 98)


def test_evaluate_uses_unavailable_number():
    assert not evaluate_countdown_expression("50 + 48", [44, 19, 35], 98)


def test_evaluate_division():
    assert evaluate_countdown_expression("10 / 2 + 5", [10, 2, 5], 10)


def test_evaluate_invalid_expression():
    assert not evaluate_countdown_expression("hello world", [44, 19, 35], 98)


def test_extract_answer_boxed():
    response = "Let me think...\n\\boxed{44 + 19 + 35}"
    assert extract_answer(response) == "44 + 19 + 35"


def test_extract_answer_fallback():
    response = "The answer is:\n44 + 19 + 35"
    assert extract_answer(response) == "44 + 19 + 35"


def test_extract_answer_none():
    response = "I don't know how to solve this."
    assert extract_answer(response) is None


def test_extract_answer_boxed_with_parens():
    response = "\\boxed{(10 + 5) * 2}"
    assert extract_answer(response) == "(10 + 5) * 2"
