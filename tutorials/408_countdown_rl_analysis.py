import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial: RL Reward Design — A Countdown Case Study

    In tutorial 04, you learned the raw GRPO loop. In this tutorial, you will build
    a complete RL environment for the **Countdown number game** and see how **reward
    function design** directly affects training. You will:

    1. Define a `ProblemEnv` with a verifiable reward function
    2. Compare **binary** vs **partial credit** rewards on the same task
    3. Train with GRPO and watch accuracy, token length, and useful groups evolve
    4. Analyze rollouts to understand what the model learns

    The Countdown task: given 3–4 numbers and a target, combine them with `+`, `-`,
    `*`, `/` to reach the target. Each number can be used at most once.
    """)
    return


@app.cell
def _():
    import math
    import re
    import warnings
    from collections.abc import Sequence
    from functools import partial

    warnings.filterwarnings("ignore", message="IProgress not found")

    import tinker

    from tinker_cookbook.renderers import get_renderer, get_text_content
    from tinker_cookbook.rl.data_processing import (
        assemble_training_data,
        compute_advantages,
    )
    from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
    from tinker_cookbook.rl.rollouts import (
        do_group_rollout_and_filter_constant_reward,
    )
    from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return (
        EnvGroupBuilder,
        ProblemEnv,
        ProblemGroupBuilder,
        RLDataset,
        Sequence,
        assemble_training_data,
        compute_advantages,
        do_group_rollout_and_filter_constant_reward,
        get_renderer,
        get_text_content,
        get_tokenizer,
        math,
        partial,
        re,
        tinker,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 — The reward function

    A good RL reward function is **verifiable** (we can check correctness programmatically)
    and **informative** (it gives the model useful gradient signal).

    We start with two grading functions: one that returns binary pass/fail, and one
    that gives **partial credit** for expressions that use valid numbers but get the
    wrong result. The partial score includes a proximity bonus — closer to the target
    means higher reward.
    """)
    return


@app.cell
def _(re):
    def evaluate_expression(
        expression: str, available_nums: list[int], target: int
    ) -> tuple[bool, float]:
        """Grade a countdown expression.

        Returns:
            (is_correct, partial_score) where partial_score is:
            - 0.0 if invalid expression or uses wrong numbers
            - 0.3 + proximity bonus (up to 0.3) if valid but wrong result
            - 1.0 if exactly correct
        """
        try:
            if not re.match(r"^[\d\s\+\-\*/\(\)\.]+$", expression):
                return False, 0.0

            used_nums = [int(n) for n in re.findall(r"\d+", expression)]
            remaining = list(available_nums)
            for n in used_nums:
                if n in remaining:
                    remaining.remove(n)
                else:
                    return False, 0.0

            result = eval(expression)  # noqa: S307
            if abs(result - target) < 1e-6:
                return True, 1.0

            # Partial credit: proximity to target
            if target != 0:
                relative_error = abs(result - target) / abs(target)
                proximity = max(0.0, 1.0 - relative_error)
            else:
                proximity = 1.0 if abs(result) < 1e-6 else 0.0
            return False, 0.3 + 0.3 * proximity
        except Exception:
            return False, 0.0

    def extract_boxed(response: str) -> str | None:
        """Extract expression from \\boxed{} or last arithmetic line."""
        match = re.search(r"\\boxed\{([^}]+)\}", response)
        if match:
            return match.group(1).strip()
        for line in reversed(response.strip().splitlines()):
            line = line.strip()
            if re.search(r"\d+\s*[\+\-\*/]", line):
                return re.sub(r"^[=:\s]+", "", line).strip()
        return None

    return evaluate_expression, extract_boxed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### See the difference: binary vs partial credit

    Let's grade the same set of responses both ways:
    """)
    return


@app.cell
def _(evaluate_expression):
    _target = 98
    _nums = [44, 19, 35]

    _examples = [
        "44 + 19 + 35",   # correct
        "44 + 35",         # valid numbers, result=79 (close)
        "44 + 19",         # valid numbers, result=63 (farther)
        "50 + 48",         # invalid numbers
    ]

    print(f"Target: {_target}, Numbers: {_nums}\n")
    print(f"{'Expression':<20} {'Binary':>8} {'Partial':>8}  {'Eval':>6}")
    print("-" * 60)
    for _expr in _examples:
        _correct, _partial = evaluate_expression(_expr, _nums, _target)
        _binary = 1.0 if _correct else 0.0
        try:
            _val = eval(_expr)
        except Exception:
            _val = "err"
        print(f"{_expr:<20} {_binary:>8.1f} {_partial:>8.2f}  {_val!s:>6}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Binary gives 0.0 to everything except the perfect answer. Partial credit gives
    **0.54** for "close to target" and **0.49** for "farther off." This variance within
    a GRPO group is what creates learning signal — if all completions score 0.0, every
    advantage is zero and the group contributes nothing to the gradient.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — Define the CountdownEnv

    Subclass `ProblemEnv` and implement four methods. We also override `step()` to
    support partial-credit rewards — the base class only does binary scoring.
    """)
    return


@app.cell
def _(
    ProblemEnv,
    evaluate_expression,
    extract_boxed,
    get_text_content,
    re,
    tinker,
):
    class CountdownEnv(ProblemEnv):
        """Single-turn env: reach the target using arithmetic on the given numbers."""

        def __init__(self, target, nums, renderer, convo_prefix=None, use_partial=True):
            super().__init__(renderer, convo_prefix)
            self.target = target
            self.nums = nums
            self.use_partial = use_partial

        def get_question(self) -> str:
            nums_str = ", ".join(str(n) for n in self.nums)
            return (
                f"Using the numbers [{nums_str}], create an arithmetic expression "
                f"that equals {self.target}. You can use +, -, *, / and each number "
                f"at most once. Put your final expression in \\boxed{{}}."
            )

        def check_answer(self, sample_str: str) -> bool:
            expr = extract_boxed(sample_str)
            if expr is None:
                return False
            correct, _ = evaluate_expression(expr, self.nums, self.target)
            return correct

        def check_format(self, sample_str: str) -> bool:
            return extract_boxed(sample_str) is not None

        def get_reference_answer(self) -> str:
            return f"target={self.target}, nums={self.nums}"

        async def step(self, action, *, extra=None):
            """Score with partial credit when use_partial=True."""
            if not self.use_partial:
                return await super().step(action, extra=extra)

            # Parse the model's response
            message, parse_success = self.renderer.parse_response(action)
            content = get_text_content(message)
            correct_format = float(parse_success) and float(self.check_format(content))
            correct_answer = float(self.check_answer(content))

            # Partial reward: grade proximity to target
            expr = extract_boxed(content)
            if expr is not None and not correct_answer:
                _, partial_score = evaluate_expression(expr, self.nums, self.target)
            else:
                partial_score = 1.0 if correct_answer else 0.0

            reward_value = partial_score if not correct_answer else 1.0
            total_reward = self.format_coef * (correct_format - 1) + reward_value

            from tinker_cookbook.rl.types import StepResult

            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "format": correct_format,
                    "correct": correct_answer,
                    "partial_reward": partial_score,
                },
            )

    # Quick check
    print("CountdownEnv methods: get_question, check_answer, check_format, step")
    print()
    print("Example question for target=98, nums=[44, 19, 35]:")
    print(f"  Using the numbers [44, 19, 35], create an arithmetic expression ...")
    print(f"  ... that equals 98. Put your final expression in \\boxed{{}}.")
    return (CountdownEnv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test the env manually

    Before training, verify the reward function works by feeding it real token sequences.
    """)
    return


@app.cell
async def _(CountdownEnv, get_renderer, get_tokenizer, tinker):
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    _service_client = tinker.ServiceClient()
    _tokenizer = get_tokenizer(MODEL_NAME)
    _renderer = get_renderer("qwen3_instruct", _tokenizer)

    # Test with a correct response
    _env = CountdownEnv(13, [3, 7, 2], _renderer, use_partial=True)
    _ob, _stop = await _env.initial_observation()

    _good = _tokenizer.encode("3 * 2 + 7 = 13. \\boxed{3 * 2 + 7}")
    _result = await _env.step(_good)
    print(f"Correct answer:  reward={_result.reward:.3f}  metrics={_result.metrics}")

    # Test with a wrong but close answer
    _env2 = CountdownEnv(13, [3, 7, 2], _renderer, use_partial=True)
    await _env2.initial_observation()
    _bad = _tokenizer.encode("3 + 7 = 10. \\boxed{3 + 7}")
    _result2 = await _env2.step(_bad)
    print(f"Wrong but close: reward={_result2.reward:.3f}  metrics={_result2.metrics}")

    # Test with no boxed answer
    _env3 = CountdownEnv(13, [3, 7, 2], _renderer, use_partial=True)
    await _env3.initial_observation()
    _none = _tokenizer.encode("I'm not sure how to solve this.")
    _result3 = await _env3.step(_none)
    print(f"No answer:       reward={_result3.reward:.3f}  metrics={_result3.metrics}")
    return MODEL_NAME, _renderer, _service_client, _tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 — Build the dataset

    `RLDataset.get_batch()` returns a list of `ProblemGroupBuilder`s — one per problem.
    Each builder creates `group_size` copies of the env for GRPO rollouts.

    We load problems from [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
    (490K problems with 3–4 numbers each).
    """)
    return


@app.cell
def _(
    CountdownEnv,
    EnvGroupBuilder,
    ProblemGroupBuilder,
    RLDataset,
    Sequence,
    math,
    partial,
    _renderer,
):
    from datasets import load_dataset

    class CountdownDataset(RLDataset):
        def __init__(self, data, batch_size, group_size, renderer, use_partial=True):
            self.data = data
            self.batch_size = batch_size
            self.group_size = group_size
            self.renderer = renderer
            self.use_partial = use_partial
            # Fewshot prefix to teach the model the expected format
            self.convo_prefix = [
                {
                    "role": "user",
                    "content": (
                        "Using the numbers [3, 7, 2], create an arithmetic expression "
                        "that equals 13. You can use +, -, *, / and each number at most "
                        "once. Put your final expression in \\boxed{}."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "3 * 2 = 6, 6 + 7 = 13. Yes!\n\\boxed{3 * 2 + 7}",
                },
            ]

        def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
            start = index * self.batch_size
            end = min(start + self.batch_size, len(self.data))
            return [
                ProblemGroupBuilder(
                    env_thunk=partial(
                        CountdownEnv,
                        row["target"],
                        row["nums"],
                        self.renderer,
                        convo_prefix=self.convo_prefix,
                        use_partial=self.use_partial,
                    ),
                    num_envs=self.group_size,
                    dataset_name="countdown",
                )
                for row in self.data[start:end]
            ]

        def __len__(self) -> int:
            return math.ceil(len(self.data) / self.batch_size)

    # Load and split the dataset
    _ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train").shuffle(seed=42)
    _train_data = [{"target": r["target"], "nums": r["nums"]} for r in _ds.select(range(800))]
    _test_data = [{"target": r["target"], "nums": r["nums"]} for r in _ds.select(range(800, 900))]

    GROUP_SIZE = 8
    BATCH_SIZE = 16

    train_dataset = CountdownDataset(_train_data, BATCH_SIZE, GROUP_SIZE, _renderer, use_partial=True)
    test_dataset = CountdownDataset(_test_data, BATCH_SIZE, 1, _renderer, use_partial=False)

    print(f"Train: {len(_train_data)} problems, {len(train_dataset)} batches")
    print(f"Test:  {len(_test_data)} problems (binary reward for clean eval)")
    print(f"Config: group_size={GROUP_SIZE}, batch_size={BATCH_SIZE}")
    return BATCH_SIZE, GROUP_SIZE, CountdownDataset, test_dataset, train_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 — Train with GRPO

    We run the standard rollout → advantage → train loop, tracking three metrics:

    - **Correct**: fraction of completions that reach the target (the real success metric)
    - **Useful groups**: fraction of groups where at least one completion differs in reward (provides GRPO signal)
    - **Avg tokens**: average response length (does the model learn to be concise?)
    """)
    return


@app.cell
async def _(MODEL_NAME, _service_client, tinker):
    training_client = await _service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=32
    )

    MAX_TOKENS = 1024
    TEMPERATURE = 1.0
    N_STEPS = 10  # ~10 min with 16 problems/step, group_size=8
    _lr = 1e-4
    adam_params = tinker.AdamParams(learning_rate=_lr, beta1=0.9, beta2=0.95)

    print(f"Model: {MODEL_NAME}, LoRA rank 32, lr={_lr}")
    print(f"Max tokens: {MAX_TOKENS}, temperature: {TEMPERATURE}")
    print(f"Training for {N_STEPS} steps...")
    return MAX_TOKENS, N_STEPS, TEMPERATURE, adam_params, training_client


@app.cell
async def _(
    MAX_TOKENS,
    N_STEPS,
    TEMPERATURE,
    adam_params,
    assemble_training_data,
    compute_advantages,
    do_group_rollout_and_filter_constant_reward,
    tinker,
    train_dataset,
    training_client,
):
    def _remove_mask(datum: tinker.Datum) -> tinker.Datum:
        return tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
        )

    metrics_history = []

    for _step in range(N_STEPS):
        _batch_index = _step % len(train_dataset)
        _env_group_builders = train_dataset.get_batch(_batch_index)

        # 1. Save weights and get sampling client for current policy
        _sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        # 2. Rollout each group
        _trajectory_groups = []
        _n_all_bad = 0
        _all_rewards = []
        _all_token_lens = []

        for _builder in _env_group_builders:
            _tg = await do_group_rollout_and_filter_constant_reward(
                sampling_client=_sampling_client,
                env_group_builder=_builder,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_remove_constant_reward_groups=False,  # Keep all groups for metrics
                enable_logging=False,
            )
            if _tg is not None:
                _rewards = _tg.get_total_rewards()
                _all_rewards.extend(_rewards)
                for _traj in _tg.trajectories_G:
                    _all_token_lens.append(
                        sum(len(t.action) for t in _traj.transitions)
                    )
                # Check if this group has variance (useful for GRPO)
                if len(set(round(r, 6) for r in _rewards)) > 1:
                    _trajectory_groups.append(_tg)
                else:
                    _n_all_bad += 1

        _n_total_groups = len(_env_group_builders)
        _n_useful = len(_trajectory_groups)
        _frac_correct = sum(1 for r in _all_rewards if r >= 0.9) / max(len(_all_rewards), 1)
        _avg_tokens = sum(_all_token_lens) / max(len(_all_token_lens), 1)

        # 3. Compute advantages and train (skip if no useful groups)
        _n_datums = 0
        if _trajectory_groups:
            _advantages = compute_advantages(_trajectory_groups)
            _datums, _ = assemble_training_data(_trajectory_groups, _advantages)
            _n_datums = len(_datums)

            _fwd = await training_client.forward_backward_async(
                [_remove_mask(d) for d in _datums], loss_fn="importance_sampling"
            )
            _opt = await training_client.optim_step_async(adam_params)
            await _fwd.result_async()
            await _opt.result_async()

        metrics_history.append({
            "step": _step,
            "correct": _frac_correct,
            "useful_groups": _n_useful / _n_total_groups,
            "avg_tokens": _avg_tokens,
            "n_datums": _n_datums,
        })

        print(
            f"Step {_step:2d} | correct: {_frac_correct:.1%} | "
            f"useful groups: {_n_useful}/{_n_total_groups} | "
            f"avg tokens: {_avg_tokens:.0f} | datums: {_n_datums}"
        )
    return (metrics_history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 — Visualize training dynamics

    Three panels showing how GRPO shapes the model's behavior:
    """)
    return


@app.cell
def _(metrics_history, plt):
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    _steps = [m["step"] for m in metrics_history]

    # Panel 1: accuracy
    _ax1.plot(_steps, [m["correct"] for m in metrics_history], "o-", color="#2563eb", linewidth=2)
    _ax1.set_xlabel("Training step")
    _ax1.set_ylabel("Fraction correct")
    _ax1.set_title("Success rate")
    _ax1.set_ylim(0, 1.05)
    _ax1.grid(True, alpha=0.3)

    # Panel 2: useful groups
    _ax2.plot(_steps, [m["useful_groups"] for m in metrics_history], "s-", color="#10b981", linewidth=2)
    _ax2.set_xlabel("Training step")
    _ax2.set_ylabel("Fraction of groups")
    _ax2.set_title("Useful groups (have reward variance)")
    _ax2.set_ylim(0, 1.05)
    _ax2.grid(True, alpha=0.3)

    # Panel 3: token length
    _ax3.plot(_steps, [m["avg_tokens"] for m in metrics_history], "^-", color="#f59e0b", linewidth=2)
    _ax3.set_xlabel("Training step")
    _ax3.set_ylabel("Avg response tokens")
    _ax3.set_title("Response length (conciseness)")
    _ax3.grid(True, alpha=0.3)

    _fig.suptitle("GRPO Training Dynamics (partial credit reward)", fontweight="bold", y=1.02)
    _fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6 — Evaluate on held-out problems

    Sample from the trained model on problems it never saw during training.
    We use greedy decoding (temperature=0) and binary grading.
    """)
    return


@app.cell
async def _(
    MAX_TOKENS,
    _renderer,
    _tokenizer,
    extract_boxed,
    evaluate_expression,
    get_text_content,
    test_dataset,
    tinker,
    training_client,
):
    _eval_client = await training_client.save_weights_and_get_sampling_client_async()

    _batch = test_dataset.get_batch(0)  # first 16 test problems
    _n_correct = 0
    _n_total = 0

    for _builder in _batch[:16]:
        _envs = await _builder.make_envs()
        _env = _envs[0]
        _ob, _stop = await _env.initial_observation()

        _result = await _eval_client.sample_async(
            prompt=_ob,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=MAX_TOKENS, temperature=0.0, stop=_stop
            ),
        )
        _tokens = _result.sequences[0].tokens
        _msg, _ = _renderer.parse_response(_tokens)
        _content = get_text_content(_msg)
        _expr = extract_boxed(_content)
        _is_correct = False
        if _expr is not None:
            _is_correct, _ = evaluate_expression(_expr, _env.nums, _env.target)

        _n_total += 1
        if _is_correct:
            _n_correct += 1

        _status = "PASS" if _is_correct else "FAIL"
        _preview = _content[:120].replace("\n", " ")
        print(f"[{_status}] target={_env.target}, nums={_env.nums}")
        print(f"       {_preview}...")
        print()

    print(f"Test accuracy: {_n_correct}/{_n_total} ({_n_correct/_n_total:.0%})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we learned

    - **Partial credit** creates reward variance within groups that would otherwise be
      "all-bad" (every completion wrong, zero advantage, zero gradient). This is the
      key mechanism: GRPO needs *within-group* differences to learn.

    - **Token budget matters**: if the model runs out of tokens before writing `\boxed{}`,
      it gets zero reward even if the reasoning was correct. A generous budget lets the
      model explore; GRPO then teaches it to be concise (response length drops naturally).

    - **Look at your rollouts**: metrics tell you *what* is happening, rollouts tell
      you *why*. In our experiments, 100% of remaining failures at 85% accuracy were
      token truncations — not wrong answers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other settings to try

    The recipe at `tinker_cookbook/recipes/countdown_rl/` supports several configurations.
    Here is what we found from a sweep of 8 experiments:

    | Change | Effect |
    |---|---|
    | `reward_mode=binary` | −4% test accuracy (fewer useful groups) |
    | `max_tokens=2048` | +9% test accuracy (fewer truncations) |
    | `max_tokens=512` | −8% test accuracy (many truncations) |
    | `group_size=32` | Eliminates all-bad groups, same peak accuracy |
    | `kl_penalty_coef=0.02` | −6% (too conservative for this task) |
    | `include_fewshot=False` | −4% (model struggles with format cold-start) |
    | `temperature=0.7` | −1.5% (less exploration hurts GRPO) |

    Run the full recipe with:
    ```bash
    python -m tinker_cookbook.recipes.countdown_rl.train \
        max_tokens=2048 n_train=3200 n_test=200 max_steps=40
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
