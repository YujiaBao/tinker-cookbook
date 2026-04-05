# Evaluation Framework

Evaluate models trained with Tinker using 21 built-in benchmarks. Benchmarks reuse the same `Env` protocol as RL training — grading logic is shared, not duplicated.

## Installation

```bash
pip install 'tinker-cookbook[eval]'
```

This installs all benchmark dependencies including Modal (code execution sandbox), math-verify (math grading), and antlr4 (LaTeX parsing).

For gated datasets (GPQA): `huggingface-cli login` or set `HF_TOKEN`.
For sandbox benchmarks (MBPP, LiveCodeBench, Terminal Bench, SWE-bench): `modal token new`.

## Quickstart

```python
import asyncio
import tinker
from tinker_cookbook.eval.benchmarks import run_benchmark, BenchmarkConfig
from tinker_cookbook.renderers import get_renderer

# Set up model
sc = tinker.ServiceClient()
sampling_client = sc.create_sampling_client(base_model="Qwen/Qwen3.5-35B-A3B")
renderer = get_renderer("qwen3_5", sampling_client.get_tokenizer())

# Run a benchmark
result = asyncio.run(run_benchmark("gsm8k", sampling_client, renderer))
print(f"GSM8K: {result.score:.1%}")  # e.g. "GSM8K: 81.7%"
```

## Saving and resuming

Set ``save_dir`` to persist trajectories to disk. If a run is interrupted, restarting with the same ``save_dir`` automatically resumes from where it left off.

```python
config = BenchmarkConfig(
    save_dir="evals/my_model",
    timeout_seconds=1800,   # 30 min — recommended for thinking models
    max_tokens=32768,
)
result = await run_benchmark("gsm8k", sampling_client, renderer, config)

# Inspect what's stored:
# evals/my_model/gsm8k/result.json          — aggregated BenchmarkResult
# evals/my_model/gsm8k/trajectories.jsonl   — one StoredTrajectory per line
```

## Running multiple benchmarks

```python
from tinker_cookbook.eval.benchmarks import run_benchmarks

# All benchmarks run in parallel by default
results = await run_benchmarks(
    ["gsm8k", "mmlu_pro", "ifeval", "gpqa"],
    sampling_client, renderer,
    BenchmarkConfig(save_dir="evals/my_model", timeout_seconds=1800),
)

for name, result in results.items():
    print(f"{name}: {result.score:.1%} ({result.num_truncated} truncated, {result.num_errors} errors)")
```

## Understanding scores

Thinking models often hit `max_tokens` before producing an answer. The framework tracks this separately from genuine wrong answers:

```python
result = await run_benchmark("math500", sampling_client, renderer, config)

print(f"Raw score:       {result.score:.1%}")           # includes truncated as 0
print(f"Completed score: {result.score_completed:.1%}")  # only examples that finished
print(f"Truncated:       {result.num_truncated}")        # hit max_tokens
print(f"Errors:          {result.num_errors}")           # timeout / crash
print(f"Completed:       {result.num_completed}")        # actually graded
```

`score_completed` is typically the right metric to compare against published model card scores, which don't penalize for context overflow.

> **Scores are setup-dependent.** Small changes in `max_tokens`, `timeout_seconds`, `temperature`, or `system_prompt` can shift scores by 10–30%. Always document your exact configuration when reporting results.

## Browsing and re-grading results

```python
from tinker_cookbook.eval.benchmarks import load_result, load_trajectories, print_trajectory

# Load the aggregated score
result = load_result("evals/my_model", "gsm8k")
print(f"{result.score:.1%}")

# Browse incorrect examples
for traj in load_trajectories("evals/my_model", "gsm8k", incorrect_only=True)[:3]:
    print(f"Expected: {traj.logs['expected']}, Got: {traj.logs['extracted']}")
    print_trajectory(traj)

# Re-grade with a different answer extraction — no re-running the model
from tinker_cookbook.eval.benchmarks import regrade_trajectories

def strict_grader(response: str, logs: dict) -> float:
    return 1.0 if logs["expected"].strip() == logs["extracted"].strip() else 0.0

new_result = regrade_trajectories("evals/my_model", "gsm8k", strict_grader)
print(f"Re-graded: {new_result.score:.1%}")
```

## Pass@k evaluation

Evaluate each example multiple times and compute unbiased pass@k estimates (Codex paper formula):

```python
config = BenchmarkConfig(num_samples=10, save_dir="evals/pass_at_k")
result = await run_benchmark("mbpp", sampling_client, renderer, config)

print(result.pass_at_k)  # {1: 0.45, 5: 0.72, 10: 0.85}
```

## Using benchmarks during training

`BenchmarkEvaluator` bridges any benchmark into the training loop's evaluator interface:

```python
from tinker_cookbook.eval.benchmark_evaluator import BenchmarkEvaluator

# Evaluate GSM8K on 100 examples every N training steps
evaluator_builders = [
    lambda: BenchmarkEvaluator("gsm8k", renderer, max_examples=100),
    lambda: BenchmarkEvaluator("ifeval", renderer, max_examples=50),
]

# The training loop calls: metrics = await evaluator(sampling_client)
# Returns: {"eval/gsm8k/score": 0.85, "eval/gsm8k/num_correct": 85, ...}
```

## Comparing checkpoints

Track evaluation across training checkpoints and detect regressions:

```python
from tinker_cookbook.eval.store import EvalStore

store = EvalStore("~/experiments/evals")

# Evaluate a checkpoint
run_id = store.create_run(
    model_name="Qwen/Qwen3.5-35B-A3B",
    checkpoint_name="sft_step500",
    benchmarks=["gsm8k", "ifeval"],
)
await run_benchmarks(
    ["gsm8k", "ifeval"], sampling_client, renderer,
    BenchmarkConfig(save_dir=store.run_dir(run_id)),
)
store.finalize_run(run_id)

# Compare two runs — matches examples by stable example_id
comp = store.compare_runs("sft_step500_20260327", "rl_step30_20260327", "gsm8k")
store.print_comparison(comp)
# === gsm8k: sft_step500 vs rl_step30 ===
#   Score: 0.743 -> 0.781 (delta=+0.038)
#   Regressions: 3 (correct in A, wrong in B)
#   Improvements: 18 (wrong in A, correct in B)
```

## Available benchmarks

### Stable (11)

| Benchmark | Examples | Type | Grading |
|-----------|---------|------|---------|
| `gsm8k` | 1,319 | Math | Numeric extraction |
| `math500` | 500 | Math | Boxed answer (requires `[eval]`) |
| `aime_2025` | 30 | Math competition | Integer 0-999 |
| `aime_2026` | 30 | Math competition | Integer 0-999 |
| `mmlu_pro` | 12,032 | MCQA (4-10 options) | Letter extraction |
| `mmlu_redux` | 2,722 | MCQA | Letter extraction |
| `gpqa` | 198 | MCQA (graduate science) | Letter extraction (gated dataset) |
| `ifeval` | 541 | Instruction following | Constraint verification |
| `mbpp` | 257 | Code execution | Pytest in Modal sandbox |
| `ceval` | 1,346 | MCQA (Chinese, 52 subjects) | Letter extraction |
| `supergpqa` | 26,529 | MCQA (285 disciplines) | Letter extraction |

### Experimental (10)

Experimental benchmarks are `_`-prefixed modules that log a runtime warning. They are functional but may not match published scores.

| Benchmark | Examples | Type | Notes |
|-----------|---------|------|-------|
| `hmmt_feb_2025` | 30 | Math competition | Sympy grading (requires antlr4) |
| `hmmt_nov_2025` | 30 | Math competition | Sympy grading (requires antlr4) |
| `livecodebench` | 175 | Code (LiveCodeBench v6) | Modal sandbox, needs 1800s timeout |
| `terminal_bench` | 112 | Multi-turn agent | Modal sandbox, limited by context window |
| `swe_bench` | 500 | Multi-turn SWE agent | Modal sandbox, needs large context |
| `tau2_bench` | ~1,000 | Multi-turn tool use | Needs separate judge model |
| `arena_hard` | 500 | LLM-as-judge | Needs separate judge model |
| `longbench` | varies | Long context | Limited by model context window |
| `ifbench` | 300 | Instruction following | Verifier has coverage gaps |
| `bfcl` | ~1,000 | Function calling | Ground truth format issues |

## Adding a new benchmark

Create a file in `tinker_cookbook/eval/benchmarks/` (prefix with `_` for experimental):

```python
"""My benchmark -- short description.

Dataset: ``org/dataset`` on HuggingFace.
Metric: Accuracy.
Pattern: Single-turn ``MessageEnv`` + programmatic grading.
"""

from tinker_cookbook.eval.benchmarks._common import (
    build_messages, extract_mcq_answer, format_mcq_choices,
    limit_dataset, load_benchmark_dataset, make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env


class MyMessageEnv(MessageEnv):
    """Grading logic for one example."""

    def __init__(self, prompt: str, expected: str, example_id: str = "",
                 system_prompt: str | None = None):
        self.prompt = prompt
        self.expected = expected
        self.example_id = example_id
        self.system_prompt = system_prompt

    async def initial_observation(self) -> list[Message]:
        return build_messages(self.prompt, self.system_prompt)

    async def step(self, message: Message) -> MessageStepResult:
        response = get_text_content(message)  # thinking tokens already stripped
        extracted = extract_mcq_answer(response)
        correct = extracted == self.expected
        return MessageStepResult(
            reward=1.0 if correct else 0.0,
            episode_done=True,
            next_messages=[],
            metrics={"correct": float(correct)},
            logs={
                "example_id": self.example_id,
                "input": self.prompt[:200],
                "expected": self.expected,
                "extracted": extracted,
                "output": response[:500],
            },
        )


class MyBenchmarkBuilder(BenchmarkBuilder):
    """My benchmark: short description."""
    name = "my_benchmark"

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig):
        ds = load_benchmark_dataset("org/dataset")
        ds = limit_dataset(ds, config.max_examples)
        envs = []
        for row in ds:
            row = dict(row)
            prompt = f"{row['question']}\n\n..."
            msg_env = MyMessageEnv(
                prompt, row["answer"],
                example_id=make_example_id("my_benchmark", row["question"]),
                system_prompt=config.system_prompt,
            )
            envs.append(EnvFromMessageEnv(
                renderer=renderer,
                message_env=msg_env,
                failed_parse_reward=0.0,
                context_overflow_reward=0.0,
            ))
        return envs


from tinker_cookbook.eval.benchmarks import register
register(MyBenchmarkBuilder())
```

Key points:
- **`MessageEnv` + `EnvFromMessageEnv`**: Thinking-token stripping and context overflow handling are automatic. Your `step()` receives a clean message with thinking already removed.
- **`example_id`**: Deterministic ID for cross-run comparison. Use `make_example_id(prefix, text)` for a stable hash.
- **`failed_parse_reward=0.0, context_overflow_reward=0.0`**: Truncated or unparseable responses score 0 and are tracked in `BenchmarkResult.num_truncated`.
- **Sandbox benchmarks**: Use `SandboxMixin` from `_common.py` and set `requires_sandbox = True` on the builder. See `mbpp.py` for an example.
- **Multi-turn benchmarks**: Set `multi_turn = True` on the builder (uses `agent_concurrency` instead of `concurrency`). See `_terminal_bench.py` for an example.

## Configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_examples` | `None` | Limit number of examples (`None` = all) |
| `concurrency` | `64` | Max parallel rollouts (single-turn) |
| `agent_concurrency` | `8` | Max parallel rollouts (multi-turn/sandbox) |
| `timeout_seconds` | `300` | Per-example timeout in seconds |
| `max_tokens` | `32768` | Max generation tokens per request |
| `temperature` | `0.6` | Sampling temperature |
| `context_window` | `None` | Model context size; dynamically caps `max_tokens` per request |
| `num_samples` | `1` | Samples per example for pass@k (>1 enables pass@k) |
| `save_dir` | `None` | Directory for trajectories and results |
| `system_prompt` | `None` | System prompt prepended to all examples |
| `grade_fn` | `None` | Custom `(response, logs) -> reward` grading function |
| `judge_sampling_client` | `None` | Sampling client for LLM-as-judge benchmarks |
| `sandbox_factory` | `None` | Custom sandbox factory (defaults to Modal) |

**For thinking models**, increase `timeout_seconds` to `1800` and `max_tokens` to match the model's context window. See the [setup-dependency notes](#scores-are-setup-dependent) below.

## Storage layout

```
{save_dir}/
  summary.json                     # Combined scores across all benchmarks
  {benchmark_name}/
    result.json                    # Aggregated BenchmarkResult
    trajectories.jsonl             # One StoredTrajectory per line (resumable)
```

Each trajectory contains the full decoded conversation, reward, per-example metrics, and grading logs — enabling post-hoc browsing, re-grading, and cross-run comparison without re-running the model.

## Testing

```bash
pytest tinker_cookbook/eval/benchmarks/benchmark_test.py  # 70 tests
```
