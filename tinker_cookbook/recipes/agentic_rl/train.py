"""CLI for agentic multi-turn RL training.

Trains an LLM to solve math problems by calling tools (calculator, python_exec)
over multiple conversation turns, then submitting a final answer. Uses GRPO
for policy optimization.

Example usage::

    python -m tinker_cookbook.recipes.agentic_rl.train

    # With custom model and hyperparameters:
    python -m tinker_cookbook.recipes.agentic_rl.train \\
        --model_name Qwen/Qwen3-4B-Instruct-2507 \\
        --learning_rate 4e-5 \\
        --batch_size 16 \\
        --group_size 8 \\
        --max_turns 5

How loss masking works
----------------------
In agentic RL, each trajectory contains alternating model turns (actions) and
environment turns (tool results). We want gradients only on the model's tokens,
not the environment's.

This is handled automatically by ``trajectory_to_data`` in
``tinker_cookbook/rl/data_processing.py``:

- **Observation tokens** (system prompt, user question, tool results) get
  ``mask=0.0`` and ``advantage=0.0`` -- no gradient contribution.
- **Action tokens** (model-generated text, tool calls) get ``mask=1.0`` and
  the trajectory's computed advantage -- full gradient contribution.

The multi-turn structure is preserved because ``EnvFromMessageEnv`` re-renders
the full conversation history at each turn. The ``trajectory_to_data`` function
detects that successive observations share a common prefix and merges them
into a single training datum, correctly assigning masks per-token.
"""

import asyncio
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.agentic_rl.env import AgenticMathDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    """Configuration for agentic RL training."""

    # Model parameters
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Training parameters
    learning_rate: float = 4e-5
    batch_size: int = 4
    seed: int = 42
    max_tokens: int = 1024
    eval_every: int = 1

    # Environment parameters
    group_size: int = 4
    max_turns: int = 5
    max_trajectory_tokens: int = 8192

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig) -> None:
    """Build the dataset and training config, then run the RL loop."""
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    builder = AgenticMathDatasetBuilder(
        model_name=cli_config.model_name,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        max_turns=cli_config.max_turns,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        seed=cli_config.seed,
    )

    # Build run name for logging
    model_name_short = cli_config.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"agentic_rl_{model_name_short}_bs{cli_config.batch_size}_"
        f"gs{cli_config.group_size}_turns{cli_config.max_turns}_"
        f"seed{cli_config.seed}_lr{cli_config.learning_rate}_"
        f"rank{cli_config.lora_rank}_{date_and_time}"
    )

    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/agentic_rl/{run_name}"

    wandb_name = cli_config.wandb_name or run_name

    # Validate /tmp exists
    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Build training config
    config = train.Config(
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        max_steps=cli_config.max_steps,
    )

    # Run training
    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
