"""Agentic multi-turn RL recipe.

Demonstrates training an LLM agent to use tools (calculator, code executor)
via multi-turn reinforcement learning with GRPO. The recipe shows:

- Multi-turn conversation with tool calls and environment responses
- Per-turn reward assignment for intermediate feedback
- Episode termination conditions (correct answer, max turns, tool errors)
- Token-level loss masking: only model-generated tokens receive gradient,
  environment/tool response tokens are masked out

The loss masking is handled automatically by the existing RL infrastructure:
``trajectory_to_data`` in ``tinker_cookbook/rl/data_processing.py`` assigns
mask=1.0 to action (model-generated) tokens and mask=0.0 to observation
(environment/tool response) tokens. This is the correct behavior for agentic
RL -- we only want gradients on the tokens the model chose, not the tokens
injected by the environment.
"""
