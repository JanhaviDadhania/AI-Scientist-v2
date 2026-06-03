"""Backend that shells out to the local `claude -p` CLI.

Used when the configured model name is `claude-cli` or `claude-code`. Mirrors
the signature of the other treesearch backends so it plugs into `backend/__init__.py`.
"""

import subprocess
import time
from typing import Any

from .utils import FunctionSpec, OutputType


CLAUDE_CLI_MODEL_NAMES = {"claude-cli", "claude-code"}


def _call_claude_cli(system_message: str | None, user_message: str | None) -> str:
    parts: list[str] = []
    if system_message:
        parts.append(f"[SYSTEM]\n{system_message}\n")
    if user_message:
        parts.append(f"[USER]\n{user_message}\n")
    parts.append("[ASSISTANT]\n")
    prompt = "\n".join(parts)

    result = subprocess.run(
        ["claude", "-p"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude -p failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs: Any,
) -> tuple[OutputType, float, int, int, dict]:
    if func_spec is not None:
        raise NotImplementedError(
            "claude-cli backend does not support function calling."
        )

    t0 = time.time()
    output = _call_claude_cli(system_message, user_message)
    req_time = time.time() - t0

    # The Claude CLI doesn't report token counts; estimate them roughly so the
    # rest of the pipeline (cost/latency tracking) doesn't blow up.
    in_tokens = (len(system_message or "") + len(user_message or "")) // 4
    out_tokens = len(output) // 4

    info = {"stop_reason": "end_turn"}
    return output, req_time, in_tokens, out_tokens, info
