"""Agentic RL environment for multi-turn tool-calling.

Defines ``AgenticMathEnv`` -- a MessageEnv where the agent solves math word
problems by calling tools (calculator, python_exec) and submitting a final
answer via the ``submit_answer`` tool.

The environment demonstrates the key patterns for agentic multi-turn RL:

1. **Tool dispatch**: Assistant messages may contain tool calls which are
   executed and whose results become the next observation.
2. **Episode termination**: The episode ends when the agent calls
   ``submit_answer``, hits the max-turn limit, or fails to parse.
3. **Per-turn reward**: Intermediate turns receive 0 reward; the final
   ``submit_answer`` turn receives +1 (correct) or -1 (wrong).
4. **Loss masking**: Handled automatically by ``trajectory_to_data`` in
   ``data_processing.py`` -- observation tokens (tool results, user messages)
   get mask=0, action tokens (model generations) get mask=1.

Usage::

    from tinker_cookbook.recipes.agentic_rl.env import AgenticMathEnvGroupBuilder

    builder = AgenticMathEnvGroupBuilder(
        question="What is 17 * 23 + 5?",
        gold_answer="396",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        group_size=4,
        max_turns=5,
    )
    envs = await builder.make_envs()
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Annotated

import chz

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.recipes.agentic_rl.tools import calculator, python_exec
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import ToolResult, build_agent_tool_env, simple_tool_result, tool
from tinker_cookbook.tool_use.types import Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

AGENTIC_MATH_SYSTEM_PROMPT = """\
You are a helpful assistant that solves math problems step by step.

You have access to tools that can help you compute answers:
- calculator: Evaluate arithmetic expressions (e.g., "17 * 23 + 5")
- python_exec: Run Python code for more complex computations
- submit_answer: Submit your final numerical answer

Strategy:
1. Read the problem carefully.
2. Break it into steps. Use the calculator or python_exec tool for each computation.
3. When you have the final answer, call submit_answer with the result.

Important: Always call submit_answer when you have your answer. Do not just \
state the answer in text.\
"""


# ---------------------------------------------------------------------------
# submit_answer tool -- signals episode completion
# ---------------------------------------------------------------------------


class SubmitAnswerTool:
    """Tool that the agent calls to submit its final answer.

    This tool always returns ``should_stop=True``, which causes the
    AgentToolMessageEnv to end the episode. The actual correctness check
    happens in the reward function, not here.
    """

    @tool
    async def submit_answer(
        self,
        answer: Annotated[str, "The final numerical answer to the problem"],
    ) -> ToolResult:
        """Submit your final answer to the math problem."""
        return simple_tool_result(
            json.dumps({"status": "submitted", "answer": answer}),
            should_stop=True,
            metrics={"answer_submitted": 1.0},
        )


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def _normalize_number(s: str) -> str | None:
    """Try to extract a canonical number string from free-form text.

    Returns None if no number can be extracted.
    """
    # Remove commas, whitespace, dollar signs, percent signs
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    # Try to find a number (possibly negative, possibly decimal)
    match = re.search(r"-?\d+\.?\d*", s)
    if match is None:
        return None
    num_str = match.group(0)
    # Normalize: remove trailing zeros after decimal point
    try:
        val = float(num_str)
        # If it's an integer, return without decimal
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return num_str


class MathAnswerReward:
    """Reward function that grades the agent's submitted answer.

    Scans the message history for the last ``submit_answer`` tool call,
    extracts the answer, and compares it against the gold answer.

    Returns:
        (reward, metrics) tuple where reward is +1.0 for correct, 0.0 for
        wrong, and -0.5 if no answer was submitted.
    """

    def __init__(self, gold_answer: str) -> None:
        self.gold_answer = gold_answer
        self._gold_normalized = _normalize_number(gold_answer)

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        # Find the last submit_answer call in the history
        submitted_answer: str | None = None
        for msg in reversed(history):
            if msg["role"] == "tool" and msg.get("name") == "submit_answer":
                try:
                    content = msg["content"]
                    if isinstance(content, str):
                        data = json.loads(content)
                        submitted_answer = data.get("answer")
                except (json.JSONDecodeError, TypeError):
                    pass
                break

        if submitted_answer is None:
            # Reward shaping: no submission gets -0.5 while a wrong answer gets
            # 0.0.  This encourages the model to learn to use the submit_answer
            # tool (a prerequisite for getting anything right) rather than
            # running out of turns without submitting.
            return -0.5, {"no_submission": 1.0}

        # Compare normalized answers
        submitted_normalized = _normalize_number(submitted_answer)
        if submitted_normalized is None:
            return 0.0, {"unparseable_answer": 1.0}

        correct = submitted_normalized == self._gold_normalized
        reward = 1.0 if correct else 0.0
        metrics = {
            "correct": float(correct),
            "incorrect": float(not correct),
        }
        return reward, metrics


# ---------------------------------------------------------------------------
# EnvGroupBuilder
# ---------------------------------------------------------------------------


def _build_initial_messages(
    question: str,
    renderer: Renderer,
    tools: list[Tool],
) -> list[Message]:
    """Build the initial message list with tool schemas and the question."""
    tool_specs = [t.to_spec() for t in tools]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_specs,
        system_prompt=AGENTIC_MATH_SYSTEM_PROMPT,
    )
    return prefix + [{"role": "user", "content": question}]


class AgenticMathEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of agentic math environments for GRPO.

    Each environment in the group gets the same question but runs an
    independent episode. GRPO centers advantages across the group so
    that trajectories that find the answer are upweighted relative to
    those that do not.

    Args:
        question: The math word problem.
        gold_answer: The correct numerical answer as a string.
        model_name: Model name for tokenizer lookup.
        renderer_name: Renderer name override (auto-detected if None).
        group_size: Number of parallel episodes per problem.
        max_turns: Maximum tool-calling turns before forced termination.
        max_trajectory_tokens: Context window budget.
        max_generation_tokens: Per-generation token limit.
        context_overflow_reward: Reward when context overflows.
    """

    def __init__(
        self,
        question: str,
        gold_answer: str,
        model_name: str,
        renderer_name: str | None = None,
        group_size: int = 4,
        max_turns: int = 5,
        max_trajectory_tokens: int = 8192,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.1,
    ) -> None:
        self.question = question
        self.gold_answer = gold_answer
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)

        # Build tool list: calculator, python_exec, and submit_answer
        submit_tool_holder = SubmitAnswerTool()
        tools: list[Tool] = [calculator, python_exec, submit_tool_holder.submit_answer]

        initial_messages = _build_initial_messages(self.question, renderer, tools)
        reward_fn = MathAnswerReward(gold_answer=self.gold_answer)

        return [
            build_agent_tool_env(
                renderer=renderer,
                tools=tools,
                initial_messages=initial_messages,
                reward_fn=reward_fn,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["agentic_math"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Small built-in dataset of math problems for demonstration.
# In production, replace this with a real dataset (e.g., GSM8K, MATH).
DEMO_PROBLEMS: list[dict[str, str]] = [
    {"question": "What is 17 * 23 + 5?", "answer": "396"},
    {"question": "If a train travels at 60 mph for 2.5 hours, how far does it go in miles?", "answer": "150"},
    {"question": "What is the square root of 144 plus the cube of 3?", "answer": "39"},
    {"question": "A store has a 25% off sale. If an item costs $80, what is the sale price in dollars?", "answer": "60"},
    {"question": "What is 2^10 - 1?", "answer": "1023"},
    {"question": "If you have 3 dozen eggs and use 7, how many are left?", "answer": "29"},
    {"question": "What is the area of a circle with radius 5? Round to 2 decimal places.", "answer": "78.54"},
    {"question": "A car uses 8 gallons of gas to travel 240 miles. How many miles per gallon does it get?", "answer": "30"},
    {"question": "What is 15% of 200 plus 20% of 150?", "answer": "60"},
    {"question": "If you invest $1000 at 5% simple interest for 3 years, what is the total interest earned in dollars?", "answer": "150"},
    {"question": "What is the sum of the first 10 positive integers?", "answer": "55"},
    {"question": "A rectangle has length 12 and width 8. What is its perimeter?", "answer": "40"},
    {"question": "What is 7 factorial (7!)?", "answer": "5040"},
    {"question": "If 3x + 7 = 22, what is x?", "answer": "5"},
    {"question": "What is the average of 15, 22, 31, 44, and 58?", "answer": "34"},
    {"question": "A pizza is cut into 8 slices. If you eat 3 slices, what fraction of the pizza is left? Express as a decimal.", "answer": "0.625"},
    # Multi-step problems that benefit from intermediate tool use:
    {"question": "A farmer has a rectangular field that is 150 meters long and 80 meters wide. He wants to fence it and also add a diagonal fence from one corner to the opposite corner. How many meters of fencing does he need in total? Round to 2 decimal places.", "answer": "630.17"},
    {"question": "What is the sum of all prime numbers less than 50?", "answer": "328"},
    {"question": "A ball is dropped from 100 meters. Each bounce reaches 60% of the previous height. What is the total distance traveled after 5 bounces (including the initial drop)? Round to 2 decimal places.", "answer": "318.08"},
    {"question": "Compute 13^7 mod 97.", "answer": "22"},
    {"question": "A store sells apples at $1.25 each and oranges at $0.85 each. If you buy 17 apples and 23 oranges, and there is 8.5% sales tax, how much do you pay in total? Round to 2 decimal places.", "answer": "44.29"},
    {"question": "What is the greatest common divisor of 462 and 1071?", "answer": "21"},
    {"question": "How many ways can you choose 5 items from a set of 20? (i.e., 20 choose 5)", "answer": "15504"},
    {"question": "A cylinder has radius 3 and height 10. What is its volume? Round to 2 decimal places.", "answer": "282.74"},
    {"question": "If you compound $5000 at 4% annual interest compounded monthly for 3 years, what is the final amount? Round to 2 decimal places.", "answer": "5636.36"},
    {"question": "What is the sum of the squares of the first 15 positive integers?", "answer": "1240"},
    {"question": "A triangle has sides of length 7, 10, and 12. What is its area using Heron's formula? Round to 2 decimal places.", "answer": "34.98"},
    {"question": "Convert the binary number 11010110 to decimal.", "answer": "214"},
    {"question": "What is the least common multiple of 18, 24, and 36?", "answer": "72"},
    {"question": "A recipe calls for 2/3 cup of sugar. If you want to make 2.5 times the recipe, how many cups of sugar do you need? Express as a decimal, rounded to 2 decimal places.", "answer": "1.67"},
    {"question": "What is the 10th term of the Fibonacci sequence, starting with 1, 1?", "answer": "55"},
    {"question": "A cone has base radius 4 and slant height 9. What is its total surface area (base + lateral)? Round to 2 decimal places.", "answer": "163.36"},
    {"question": "Compute the determinant of the 3x3 matrix [[2, 1, 3], [4, -1, 2], [1, 5, -2]].", "answer": "49"},
    {"question": "How many integers between 1 and 1000 (inclusive) are divisible by 3 or 5 but not both?", "answer": "467"},
]


class AgenticMathDataset(RLDataset):
    """Simple dataset that serves AgenticMathEnvGroupBuilder instances."""

    def __init__(
        self,
        builders: list[AgenticMathEnvGroupBuilder],
        batch_size: int,
    ) -> None:
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.builders[start:end]

    def __len__(self) -> int:
        return len(self.builders) // self.batch_size


@chz.chz
class AgenticMathDatasetBuilder(RLDatasetBuilder):
    """Builds train (and optional test) datasets of agentic math problems.

    Uses the built-in DEMO_PROBLEMS by default. Override ``custom_problems``
    to supply your own list of ``{"question": ..., "answer": ...}`` dicts.

    Attributes:
        model_name: Model name for tokenizer/renderer.
        batch_size: Number of EnvGroupBuilders per training batch.
        group_size: Number of parallel episodes per problem (for GRPO).
        renderer_name: Renderer override (auto-detected if None).
        max_turns: Maximum tool-calling turns per episode.
        max_trajectory_tokens: Context window budget.
        max_generation_tokens: Per-generation token limit.
        test_fraction: Fraction of problems held out for evaluation.
        seed: Random seed for train/test split.
    """

    model_name: str
    batch_size: int = 4
    group_size: int = 4
    renderer_name: str | None = None
    max_turns: int = 5
    max_trajectory_tokens: int = 8192
    max_generation_tokens: int | None = None
    test_fraction: float = 0.2
    seed: int = 42

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        import random

        problems = list(DEMO_PROBLEMS)
        rng = random.Random(self.seed)
        rng.shuffle(problems)

        # Train/test split
        n_test = max(1, int(len(problems) * self.test_fraction))
        test_problems = problems[:n_test]
        train_problems = problems[n_test:]

        def _make_builders(problem_list: list[dict[str, str]]) -> list[AgenticMathEnvGroupBuilder]:
            return [
                AgenticMathEnvGroupBuilder(
                    question=p["question"],
                    gold_answer=p["answer"],
                    model_name=self.model_name,
                    renderer_name=self.renderer_name,
                    group_size=self.group_size,
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                )
                for p in problem_list
            ]

        train_dataset = AgenticMathDataset(
            builders=_make_builders(train_problems),
            batch_size=self.batch_size,
        )
        test_dataset = AgenticMathDataset(
            builders=_make_builders(test_problems),
            batch_size=max(1, len(test_problems)),
        )

        return train_dataset, test_dataset
