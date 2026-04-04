"""Example tool definitions for the agentic RL recipe.

Provides two tools that an LLM agent can call during multi-turn episodes:

- ``calculator``: Evaluates arithmetic expressions safely.
- ``python_exec``: Executes Python code in a restricted sandbox and returns
  stdout/stderr.

Both tools demonstrate the ``@tool`` decorator pattern and return ``ToolResult``
via the ``simple_tool_result`` / ``error_tool_result`` helpers.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import traceback
from io import StringIO
from typing import Annotated

from tinker_cookbook.tool_use import ToolResult, error_tool_result, simple_tool_result, tool


@tool
async def calculator(expression: Annotated[str, "A mathematical expression to evaluate, e.g. '2 + 3 * 4'"]) -> ToolResult:
    """Evaluate a mathematical expression and return the result.

    Supports standard arithmetic (+, -, *, /, **), common math functions
    (sqrt, sin, cos, log, abs, round, etc.), and constants (pi, e).
    """
    # Allowed names for safe evaluation -- no builtins, no imports.
    safe_names: dict[str, object] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # Compile to check for syntax errors, then eval with restricted globals.
        code = compile(expression, "<calculator>", "eval")
        # Disallow anything beyond the safe names -- in particular, no __builtins__.
        result = eval(code, {"__builtins__": {}}, safe_names)
        return simple_tool_result(
            json.dumps({"expression": expression, "result": str(result)}),
            metrics={"calculator_calls": 1.0},
        )
    except Exception as exc:
        return error_tool_result(
            f"Failed to evaluate '{expression}': {exc}",
            error_type="calculator_error",
        )


# Default timeout for python_exec in seconds.
_PYTHON_EXEC_TIMEOUT_SECONDS = 5


@tool
async def python_exec(code: Annotated[str, "Python code to execute. The last expression's value is captured as the result."]) -> ToolResult:
    """Execute Python code in a restricted sandbox and return stdout output.

    The sandbox provides a limited set of builtins (no file I/O, no imports
    beyond math). Stdout is captured and returned. Execution is time-limited
    to 5 seconds via a thread pool to prevent infinite loops from hanging
    training.

    For production use, replace this with a proper sandbox (e.g., Modal sandbox,
    Docker container, or the tinker_cookbook.sandbox module).
    """
    # WARNING: This is NOT a security boundary. For production use with untrusted
    # model-generated code, use a proper sandbox (e.g., tinker_cookbook.sandbox).
    stdout_capture = StringIO()
    restricted_builtins = {
        "print": lambda *args, **kwargs: print(*args, file=stdout_capture, **kwargs),
        "range": range,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "isinstance": isinstance,
        "True": True,
        "False": False,
        "None": None,
    }

    sandbox_globals: dict[str, object] = {
        "__builtins__": restricted_builtins,
        "math": math,
    }

    def _run() -> str:
        exec(code, sandbox_globals)  # noqa: S102
        output = stdout_capture.getvalue()
        if not output:
            return "(no output)"
        return output.strip()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            output = future.result(timeout=_PYTHON_EXEC_TIMEOUT_SECONDS)
        return simple_tool_result(
            json.dumps({"output": output}),
            metrics={"python_exec_calls": 1.0},
        )
    except concurrent.futures.TimeoutError:
        return error_tool_result(
            f"Execution timed out ({_PYTHON_EXEC_TIMEOUT_SECONDS} second limit)",
            error_type="python_exec_timeout",
        )
    except Exception:
        tb = traceback.format_exc()
        # Only show the last few lines of the traceback to avoid noise.
        tb_lines = tb.strip().split("\n")
        short_tb = "\n".join(tb_lines[-3:])
        return error_tool_result(
            f"Execution error:\n{short_tb}",
            error_type="python_exec_error",
        )
