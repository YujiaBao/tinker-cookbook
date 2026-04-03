"""GRPO training on the Countdown number game.

The model learns to reach a target number by combining 3-4 numbers with
basic arithmetic. Uses verifiable rewards: the expression must evaluate
to the target and use only the provided numbers.

Example usage:
    python -m tinker_cookbook.recipes.countdown_rl.train

    # With custom model and hyperparameters:
    python -m tinker_cookbook.recipes.countdown_rl.train \
        model_name=Qwen/Qwen3.5-4B learning_rate=1e-4 group_size=16
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import chz
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.countdown_rl.countdown_env import CountdownDatasetBuilder
from tinker_cookbook.rl.train import Config, StreamMinibatchConfig, main

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for Countdown RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Training hyperparameters
    group_size: int = 16
    groups_per_batch: int = 16
    learning_rate: float = 1e-4
    max_tokens: int = 512
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0

    # Dataset configuration
    n_train: int = 10000
    n_test: int = 500
    seed: int = 0
    include_fewshot: bool = True

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Evals and checkpointing
    eval_every: int = 10
    save_every: int = 10

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Stream minibatch: train on minibatches as soon as they are ready
    stream_minibatch_config: StreamMinibatchConfig | None = None

    # Loss function
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )
    model_short = cli_config.model_name.replace("/", "-")
    run_name = (
        f"countdown-{model_short}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr"
        f"-{cli_config.group_size}group-{cli_config.groups_per_batch}batch"
        f"-seed{cli_config.seed}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/countdown_rl/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=CountdownDatasetBuilder(
            batch_size=cli_config.groups_per_batch,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            n_train=cli_config.n_train,
            n_test=cli_config.n_test,
            seed=cli_config.seed,
            include_fewshot=cli_config.include_fewshot,
        ),
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=cli_config.groups_per_batch,
            num_minibatches=cli_config.stream_minibatch_config.num_minibatches,
        )
        if cli_config.stream_minibatch_config is not None
        else None,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
