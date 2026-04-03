"""Countdown task environment for RL training.

The Countdown task asks the model to reach a target number by combining
a set of 3-4 input numbers using basic arithmetic operations (+, -, *, /).
Each number can be used at most once.

Dataset: https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
"""

import math
import re
from collections.abc import Sequence
from functools import partial
from typing import Literal

import chz
from datasets import load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


def evaluate_countdown_expression(expression: str, available_nums: list[int], target: int) -> bool:
    """Check if an arithmetic expression correctly reaches the target using only available numbers.

    Args:
        expression: An arithmetic expression string (e.g. "44 + 19 + 35").
        available_nums: The numbers that are allowed to be used.
        target: The target number to reach.

    Returns:
        True if the expression evaluates to the target and uses only available numbers
        (each at most once).
    """
    try:
        # Extract all numbers used in the expression
        used_nums = [int(n) for n in re.findall(r"\d+", expression)]

        # Check that each used number is available (respecting multiplicity)
        remaining = list(available_nums)
        for n in used_nums:
            if n in remaining:
                remaining.remove(n)
            else:
                return False

        # Only allow basic arithmetic operators, digits, spaces, and parens
        if not re.match(r"^[\d\s\+\-\*/\(\)\.]+$", expression):
            return False

        # Evaluate the expression safely
        result = eval(expression)  # noqa: S307
        return abs(result - target) < 1e-6
    except Exception:
        return False


def extract_answer(response: str) -> str | None:
    """Extract the answer from a model response.

    Looks for content inside \\boxed{} first, then falls back to the last
    line containing arithmetic operators.
    """
    # Try \boxed{} format first
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Fallback: find the last line that looks like an arithmetic expression
    for line in reversed(response.strip().splitlines()):
        line = line.strip()
        if re.search(r"\d+\s*[\+\-\*/]", line):
            # Clean up: remove leading "=" or other prefixes
            line = re.sub(r"^[=:\s]+", "", line)
            return line.strip()

    return None


class CountdownEnv(ProblemEnv):
    """Environment for Countdown number game tasks."""

    def __init__(
        self,
        target: int,
        nums: list[int],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        super().__init__(renderer, convo_prefix)
        self.target = target
        self.nums = nums

    def get_question(self) -> str:
        nums_str = ", ".join(str(n) for n in self.nums)
        return (
            f"Using the numbers [{nums_str}], create an arithmetic expression that equals {self.target}. "
            f"You can use +, -, *, / and each number at most once. "
            f"Show your work, then write your final expression in \\boxed{{}} format."
        )

    def check_answer(self, sample_str: str) -> bool:
        expr = extract_answer(sample_str)
        if expr is None:
            return False
        return evaluate_countdown_expression(expr, self.nums, self.target)

    def check_format(self, sample_str: str) -> bool:
        return extract_answer(sample_str) is not None

    def get_reference_answer(self) -> str:
        return f"target={self.target}, nums={self.nums}"

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": (
                    "Using the numbers [3, 7, 2], create an arithmetic expression that equals 13. "
                    "You can use +, -, *, / and each number at most once. "
                    "Show your work, then write your final expression in \\boxed{} format."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I need to reach 13 using 3, 7, and 2.\n"
                    "Let me try: 7 * 2 = 14, then 14 - 3 = 11. No.\n"
                    "Try: 3 + 7 + 2 = 12. No.\n"
                    "Try: 7 * 2 - 3 = 11. No.\n"
                    "Try: 3 * 2 + 7 = 13. Yes!\n"
                    "\\boxed{3 * 2 + 7}"
                ),
            },
        ]


class CountdownDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        data: list[dict],
        convo_prefix: list[renderers.Message] | None = None,
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.data = data
        self.convo_prefix = convo_prefix

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.data))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    CountdownEnv,
                    row["target"],
                    row["nums"],
                    self.renderer,
                    convo_prefix=self.convo_prefix,
                ),
                num_envs=self.group_size,
                dataset_name="countdown",
            )
            for row in self.data[batch_start:batch_end]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)


@chz.chz
class CountdownDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    n_train: int = 10000
    n_test: int = 500
    seed: int = 0
    include_fewshot: bool = True

    async def __call__(self) -> tuple[CountdownDataset, CountdownDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        convo_prefix = CountdownEnv.standard_fewshot_prefix() if self.include_fewshot else None

        ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
        ds = ds.shuffle(seed=self.seed)

        # Split into train and test
        train_data = [{"target": row["target"], "nums": row["nums"]} for row in ds.select(range(self.n_train))]
        test_data = [{"target": row["target"], "nums": row["nums"]} for row in ds.select(range(self.n_train, self.n_train + self.n_test))]

        train_dataset = CountdownDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            data=train_data,
            convo_prefix=convo_prefix,
        )
        test_dataset = CountdownDataset(
            batch_size=self.batch_size,
            group_size=1,
            renderer=renderer,
            data=test_data,
            convo_prefix=convo_prefix,
        )
        return train_dataset, test_dataset
