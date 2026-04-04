import pytest

from tests.helpers import run_recipe

MODULE = "tinker_cookbook.recipes.agentic_rl.train"


@pytest.mark.integration
def test_agentic_rl_basic():
    run_recipe(
        MODULE,
        [
            "model_name=Qwen/Qwen3-4B-Instruct-2507",
            "batch_size=4",
            "group_size=4",
            "max_tokens=5",
            "behavior_if_log_dir_exists=delete",
        ],
    )
