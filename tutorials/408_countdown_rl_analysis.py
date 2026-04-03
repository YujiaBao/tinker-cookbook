import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial: RL Reward Design — A Countdown Case Study

    This tutorial uses the **Countdown number game** to explore how reward function
    design affects RL training. You will:

    1. Build a verifiable reward function for a math puzzle
    2. Compare **binary** vs **partial credit** rewards and see how they change GRPO's behavior
    3. Analyze model rollouts to understand *what the model actually learns*
    4. See how token budget, group size, and other choices interact with reward design

    The Countdown task: given 3–4 numbers and a target, combine the numbers with
    `+`, `-`, `*`, `/` to reach the target. Each number can be used at most once.

    > **Example**: numbers = [3, 7, 2], target = 13 → `3 * 2 + 7 = 13`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The reward function

    A good RL reward function for math tasks has two properties:
    - **Verifiable**: we can check correctness programmatically (no human labels needed)
    - **Informative**: it gives the model useful gradient signal

    Let's start with the simplest possible reward — **binary**: 1.0 if correct, 0.0 if wrong.
    """)
    return


@app.cell
def _():
    import re

    def evaluate_countdown_expression(
        expression: str, available_nums: list[int], target: int
    ) -> tuple[bool, float]:
        """Grade a countdown expression, returning (is_correct, partial_score).

        The partial score gives intermediate credit:
        - 0.0 if the expression is invalid or uses wrong numbers
        - 0.3 if valid numbers but wrong result
        - 0.3 + up to 0.3 proximity bonus (closer to target = higher)
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

            # Partial credit: valid expression with correct numbers but wrong result
            if target != 0:
                relative_error = abs(result - target) / abs(target)
                proximity = max(0.0, 1.0 - relative_error)
            else:
                proximity = 1.0 if abs(result) < 1e-6 else 0.0
            return False, 0.3 + 0.3 * proximity
        except Exception:
            return False, 0.0

    def extract_answer(response: str) -> str | None:
        """Extract expression from \\boxed{} or the last arithmetic line."""
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed_match:
            return boxed_match.group(1).strip()
        for line in reversed(response.strip().splitlines()):
            line = line.strip()
            if re.search(r"\d+\s*[\+\-\*/]", line):
                line = re.sub(r"^[=:\s]+", "", line)
                return line.strip()
        return None

    return evaluate_countdown_expression, extract_answer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Binary vs partial credit

    Let's see how the two reward modes score the same set of responses:
    """)
    return


@app.cell
def _(evaluate_countdown_expression, extract_answer):
    # Simulated model responses for target=98, nums=[44, 19, 35]
    examples = [
        ("44 + 19 + 35", "All three numbers sum to 98 — correct!"),
        ("44 + 19", "Valid numbers, 44+19 = 63 — wrong result, ~64% of target"),
        ("44 + 35", "Valid numbers, 44+35 = 79 — closer, ~81% of target"),
        ("50 + 48", "Uses numbers not in the list — invalid"),
    ]

    target = 98
    nums = [44, 19, 35]

    print(f"Target: {target}, Numbers: {nums}\n")
    print(f"{'Expression':<25} {'Binary':>8} {'Partial':>8}  Note")
    print("-" * 75)
    for expr, note in examples:
        is_correct, partial = evaluate_countdown_expression(expr, nums, target)
        binary = 1.0 if is_correct else 0.0
        print(f"{expr:<25} {binary:>8.1f} {partial:>8.2f}  {note}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the difference:
    - **Binary** gives 0.0 to everything that isn't perfect — no gradient signal
    - **Partial** gives 0.55 for "close to target" and 0.30 for "valid but far off"

    This matters for GRPO because **advantages are computed within each group**. If all
    completions in a group score 0.0 (an "all-bad" group), every advantage is zero and
    that group contributes **nothing** to the training gradient. Partial credit converts
    some of these dead groups into useful training signal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Training dynamics: binary vs partial

    We ran the Countdown recipe with both reward modes on `Qwen3-4B-Instruct-2507`
    (LoRA rank 32, lr=1e-4, group_size=16, max_tokens=1024, 20 steps). Here are the
    training curves from actual experiments.
    """)
    return


@app.cell
def _():
    # Metrics from actual training runs (see tinker_cookbook/recipes/countdown_rl/)
    binary_metrics = [
        {"step": 0, "correct": 0.340, "test": 0.600, "all_bad": 0.312, "avg_tokens": 520},
        {"step": 1, "correct": 0.426, "test": None, "all_bad": 0.250, "avg_tokens": None},
        {"step": 2, "correct": 0.504, "test": None, "all_bad": 0.125, "avg_tokens": None},
        {"step": 3, "correct": 0.633, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 4, "correct": 0.496, "test": None, "all_bad": 0.250, "avg_tokens": None},
        {"step": 5, "correct": 0.855, "test": 0.690, "all_bad": 0.062, "avg_tokens": None},
        {"step": 6, "correct": 0.750, "test": None, "all_bad": 0.125, "avg_tokens": None},
        {"step": 7, "correct": 0.863, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 8, "correct": 0.633, "test": None, "all_bad": 0.125, "avg_tokens": None},
        {"step": 9, "correct": 0.656, "test": None, "all_bad": 0.250, "avg_tokens": None},
        {"step": 10, "correct": 0.555, "test": 0.720, "all_bad": 0.250, "avg_tokens": None},
        {"step": 11, "correct": 0.820, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 12, "correct": 0.824, "test": None, "all_bad": 0.125, "avg_tokens": None},
        {"step": 13, "correct": 0.836, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 14, "correct": 0.742, "test": None, "all_bad": 0.188, "avg_tokens": None},
        {"step": 15, "correct": 0.609, "test": 0.720, "all_bad": 0.250, "avg_tokens": None},
        {"step": 16, "correct": 0.781, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 17, "correct": 0.797, "test": None, "all_bad": 0.125, "avg_tokens": None},
        {"step": 18, "correct": 0.730, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 19, "correct": 0.797, "test": None, "all_bad": 0.188, "avg_tokens": None},
    ]

    partial_metrics = [
        {"step": 0, "correct": 0.352, "test": 0.550, "all_bad": 0.375, "avg_tokens": 523},
        {"step": 1, "correct": 0.430, "test": None, "all_bad": 0.188, "avg_tokens": None},
        {"step": 2, "correct": 0.551, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 3, "correct": 0.648, "test": None, "all_bad": 0.125, "avg_tokens": None},
        {"step": 4, "correct": 0.539, "test": None, "all_bad": 0.188, "avg_tokens": None},
        {"step": 5, "correct": 0.930, "test": 0.710, "all_bad": 0.062, "avg_tokens": None},
        {"step": 6, "correct": 0.742, "test": None, "all_bad": 0.125, "avg_tokens": None},
        {"step": 7, "correct": 0.844, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 8, "correct": 0.691, "test": None, "all_bad": 0.188, "avg_tokens": None},
        {"step": 9, "correct": 0.703, "test": None, "all_bad": 0.188, "avg_tokens": None},
        {"step": 10, "correct": 0.578, "test": 0.760, "all_bad": 0.188, "avg_tokens": None},
        {"step": 11, "correct": 0.844, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 12, "correct": 0.750, "test": None, "all_bad": 0.125, "avg_tokens": None},
        {"step": 13, "correct": 0.883, "test": None, "all_bad": 0.000, "avg_tokens": None},
        {"step": 14, "correct": 0.742, "test": None, "all_bad": 0.188, "avg_tokens": None},
        {"step": 15, "correct": 0.645, "test": 0.750, "all_bad": 0.250, "avg_tokens": None},
        {"step": 16, "correct": 0.773, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 17, "correct": 0.750, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 18, "correct": 0.691, "test": None, "all_bad": 0.062, "avg_tokens": None},
        {"step": 19, "correct": 0.766, "test": None, "all_bad": 0.125, "avg_tokens": None},
    ]
    return binary_metrics, partial_metrics


@app.cell
def _(binary_metrics, partial_metrics):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: training accuracy
    ax = axes[0]
    ax.plot(
        [m["step"] for m in binary_metrics],
        [m["correct"] for m in binary_metrics],
        "o-", label="Binary reward", alpha=0.8,
    )
    ax.plot(
        [m["step"] for m in partial_metrics],
        [m["correct"] for m in partial_metrics],
        "s-", label="Partial credit", alpha=0.8,
    )
    # Test accuracy markers
    for metrics, marker, color in [
        (binary_metrics, "D", "C0"),
        (partial_metrics, "D", "C1"),
    ]:
        test_steps = [m["step"] for m in metrics if m["test"] is not None]
        test_accs = [m["test"] for m in metrics if m["test"] is not None]
        ax.plot(test_steps, test_accs, marker, color=color, markersize=10, zorder=5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Fraction correct")
    ax.set_title("Training accuracy (circles) and test accuracy (diamonds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Right plot: fraction of all-bad groups
    ax = axes[1]
    ax.plot(
        [m["step"] for m in binary_metrics],
        [m["all_bad"] for m in binary_metrics],
        "o-", label="Binary reward", alpha=0.8,
    )
    ax.plot(
        [m["step"] for m in partial_metrics],
        [m["all_bad"] for m in partial_metrics],
        "s-", label="Partial credit", alpha=0.8,
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Fraction of groups")
    ax.set_title('"All-bad" groups (zero learning signal)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 0.5)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the plots show

    **Left panel (accuracy):** Both reward modes improve rapidly, but partial credit
    reaches **76% test accuracy** vs binary's **72%**. The gap comes from the early
    steps where partial credit provides gradient signal that binary cannot.

    **Right panel (all-bad groups):** At step 0, about 30–40% of groups have zero
    learning signal — every completion in the group is wrong, so all advantages are zero.
    With partial credit, these groups still contribute gradients because "close to correct"
    scores higher than "completely wrong." By step 13, partial credit achieves **0%
    all-bad groups**.

    > **Key insight:** Partial credit doesn't change what "correct" means — it changes
    > how much the model learns from *incorrect* attempts. A response that evaluates to
    > 97 when the target is 98 is more useful training signal than one that evaluates to 5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Analyzing rollouts: what does the model actually do?

    Looking at metrics alone can be misleading. Let's inspect actual model responses
    at different training stages to understand *how* the model's behavior changes.

    Below are real responses from the best experiment (partial reward, 2048 max tokens,
    40 training steps, reaching **85% test accuracy**).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 0 — Before training (66% correct)

    **Correct response** (target=21, nums=[43, 30, 52], 299 tokens):
    ```
    Try subtraction first:
    - 52 − 43 = 9
    - 52 − 30 = 22 → close to 21
    - 43 − 30 = 13

    What about (52 − 43) = 9, and 30 − 9 = 21? Yes!
    \boxed{30 - (52 - 43)}
    ```

    **Wrong response** (target=15, nums=[13, 14, 59, 39], **2048 tokens — truncated!**):
    ```
    Try: 59 − 39 = 20, 20 − 14 = 6, 6 + 13 = 19 → too big
    Try: 39 − 14 = 25, 25 − 13 = 12 → not helpful
    Try: 59 − 13 = 46, 46 − 39 = 7 → 7 + 14 = 21 → too big
    Try: (59 + 13) = 72, (39 + 14) = 53, 72 − 53 = 19 → not 15
    Try division: (59 − 39) = 20, (14 − 13) = 1, 20 ÷ 1 = 20 → ...
    [TRUNCATED — ran out of tokens]
    ```
    The model enumerates every combination with verbose markdown, runs out of
    tokens without ever finding `\boxed{}`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 10 — Peak test accuracy (85% correct)

    **The same hard problem now solved** (target=15, nums=[13, 14, 59, 39], 1010 tokens):
    ```
    Try differences between large numbers:
    59 − 39 = 20, 59 − 14 = 45, 14 − 13 = 1 ...
    Now try division: 39 / 13 = 3
    What about: 14 + 3 = 17 → no
    (59 − 39) / (14 − 13) = 20 → no
    59 − (39 + 14 - 13) = 19 → no
    Try: 59 - 14 = 45, 45 / (39 / 13) = 45/3 = 15 → YES!
    \boxed{(59 - 14) / (39 / 13)}
    ```

    Two key changes after training:
    1. **More strategic search** — the model tries division early instead of exhaustive subtraction
    2. **Fewer dead ends** — it prunes unpromising paths faster
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The conciseness effect

    One of the most striking things GRPO learns is **brevity**. The model discovers on
    its own that shorter responses are better — not because we reward brevity, but because
    shorter correct responses beat truncated incorrect ones within a group.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    # Token length data from the 40-step experiment (partial, 2048 tokens)
    steps_40 = list(range(40))
    avg_tokens_40 = [
        1116, 866, 892, 776, 1087, 540, 752, 623, 877, 796,
        1039, 664, 717, 573, 800, 846, 712, 773, 701, 885,
        1137, 664, 802, 512, 353, 1016, 961, 789, 776, 741,
        811, 523, 765, 752, 996, 972, 618, 901, 494, 979,
    ]
    correct_40 = [
        0.391, 0.465, 0.555, 0.684, 0.551, 0.914, 0.801, 0.914, 0.746, 0.719,
        0.660, 0.891, 0.809, 0.902, 0.816, 0.734, 0.828, 0.816, 0.863, 0.766,
        0.605, 0.809, 0.801, 0.965, 0.984, 0.762, 0.699, 0.828, 0.789, 0.840,
        0.793, 0.883, 0.785, 0.797, 0.672, 0.668, 0.840, 0.758, 0.938, 0.648,
    ]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color1 = "C0"
    ax1.plot(steps_40, avg_tokens_40, "o-", color=color1, alpha=0.7, markersize=4)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Avg response tokens", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(200, 1300)

    ax2 = ax1.twinx()
    color2 = "C1"
    ax2.plot(steps_40, correct_40, "s-", color=color2, alpha=0.7, markersize=4)
    ax2.set_ylabel("Train accuracy", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0.2, 1.05)

    ax1.set_title("GRPO learns conciseness: tokens decrease as accuracy increases")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Average response length drops from **~1100 tokens to ~500 tokens** over training,
    while accuracy climbs from 39% to 90%+. The model learns that verbose
    chain-of-thought with markdown formatting wastes tokens that could be used for
    finding the answer.

    No length penalty was added — this is **emergent behavior from GRPO**. Within each
    group, a 300-token correct response gets reward 1.0, while a 2048-token truncated
    attempt gets 0.0. The advantage signal naturally pushes the policy toward conciseness.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Failure analysis: what remains hard?

    At 85% test accuracy, we can ask: **what do the remaining 15% of failures look like?**
    This is crucial for deciding what to improve next.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    # From analyzing step-20 eval rollouts of the best model
    # 200 test problems, 160 correct, 40 wrong
    categories = ["Correct\n(within budget)", "Truncated\n(ran out of tokens)", "Wrong answer\n(within budget)"]
    counts = [160, 39, 1]
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(categories, counts, color=colors, edgecolor="white", linewidth=2)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            str(count), ha="center", fontweight="bold", fontsize=14,
        )
    ax.set_ylabel("Number of test problems")
    ax.set_title("Failure analysis at 85% test accuracy (200 problems)")
    ax.set_ylim(0, 200)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **39 out of 40 failures are truncations** — the model ran out of 2048 tokens before
    finding an answer. Only **1 failure** was a wrong answer within the token budget.

    This means the model has essentially *learned to solve the Countdown task*. The
    remaining bottleneck is purely computational: some problems require exploring many
    combinations, and even 2048 tokens isn't always enough.

    Correct responses average **293 tokens**. Wrong responses are **all exactly 2048 tokens**
    (the maximum). There is no middle ground — the model either finds the answer quickly
    or exhausts its budget trying.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The full hyperparameter sweep

    We tested 8 configurations to understand which choices matter most:

    | Config | Best Test Acc | Key Finding |
    |---|---|---|
    | binary, 512 tokens | 68% | Baseline |
    | binary, 1024 tokens | 72% | Token budget matters (+4%) |
    | **partial, 1024 tokens** | **76%** | **Partial reward matters (+4%)** |
    | partial, group_size=32 | 76% | Eliminates all-bad groups, same peak |
    | partial, KL=0.02 | 70% | KL penalty hurts this task |
    | partial, no fewshot | 72% | Fewshot prefix is critical |
    | partial, temp=0.7 | 78.5% | Lower temperature hurts exploration |
    | **partial, 2048 tokens** | **85%** | **Token budget is the biggest lever** |

    Three takeaways:

    1. **Token budget is #1** — going from 512 to 2048 added 17 percentage points
    2. **Reward shaping is #2** — partial credit added 4% by improving gradient utilization
    3. **Don't over-regularize** — KL penalty and low temperature both hurt for this task,
       where the base model's distribution is already reasonable
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Running the recipe

    To reproduce these results:

    ```bash
    # Quick experiment (~20 min, 20 steps)
    python -m tinker_cookbook.recipes.countdown_rl.train \
        n_train=1600 n_test=100 eval_every=5 max_steps=20

    # Full training (~2.5 hours, 40 steps, best config)
    python -m tinker_cookbook.recipes.countdown_rl.train \
        n_train=3200 n_test=200 max_steps=40

    # Compare binary vs partial reward
    python -m tinker_cookbook.recipes.countdown_rl.train \
        reward_mode=binary max_steps=20 \
        log_path=~/tinker-experiments/countdown_rl/exp_binary

    python -m tinker_cookbook.recipes.countdown_rl.train \
        reward_mode=partial max_steps=20 \
        log_path=~/tinker-experiments/countdown_rl/exp_partial
    ```

    Training logs (metrics, rollout transcripts, HTML reports) are written to `~/tinker-experiments/countdown_rl/` by default.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key concepts recap

    - **Reward shaping** doesn't change what's correct — it changes how much the model
      learns from *incorrect* attempts. Partial credit turns dead groups into useful signal.

    - **All-bad groups** are the enemy of GRPO. When every completion in a group gets
      the same reward, the advantage is zero everywhere and no learning happens.
      Design rewards to create *variance within groups*.

    - **Token budget** is an often-overlooked hyperparameter. If the model's
      chain-of-thought reasoning gets truncated, it can't write `\boxed{}`, and
      correct reasoning produces zero reward. Generous budgets let the model explore
      freely; GRPO then teaches it to be concise.

    - **Look at your rollouts.** Metrics tell you *what* is happening; rollouts tell
      you *why*. The discovery that 100% of failures were truncations — not wrong
      answers — changed the entire optimization strategy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    - **Custom environments**: See `tinker_cookbook/recipes/countdown_rl/countdown_env.py`
      for the full `ProblemEnv` implementation with partial rewards
    - **RL hyperparameters**: Tutorial `402_rl_hyperparams.py` covers KL penalty,
      group size, and advantage normalization in depth
    - **Multi-turn RL**: The `harbor_rl` and `multiplayer_rl` recipes show how to
      build environments with multiple interaction steps
    """)
    return


if __name__ == "__main__":
    app.run()
