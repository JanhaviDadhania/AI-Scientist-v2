"""Backend that shells out to the local `claude -p` CLI.

Used when the configured model name is `claude-cli` or `claude-code`. Mirrors
the signature of the other treesearch backends so it plugs into `backend/__init__.py`.

Function-calling (structured tool output) is supported via the CLI's built-in
`--json-schema` / `--output-format json` flags, which return a JSON envelope
with a `structured_output` field that conforms to the requested schema. This
removes the need to route func-call traffic through Gemini.
"""

import json
import logging
import subprocess
import time
from typing import Any

from .utils import FunctionSpec, OutputType


logger = logging.getLogger("ai-scientist.claude_cli")

CLAUDE_CLI_MODEL_NAMES = {"claude-cli", "claude-code"}


def _flatten_prompt(system_message: str | None, user_message: str | None) -> str:
    parts: list[str] = []
    if system_message:
        parts.append(f"[SYSTEM]\n{system_message}\n")
    if user_message:
        parts.append(f"[USER]\n{user_message}\n")
    parts.append("[ASSISTANT]\n")
    return "\n".join(parts)


def _call_claude_cli(system_message: str | None, user_message: str | None) -> str:
    """Plain text out — used when no function-calling is requested."""
    result = subprocess.run(
        ["claude", "-p"],
        input=_flatten_prompt(system_message, user_message),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude -p failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _call_claude_cli_structured(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec,
) -> dict[str, Any]:
    """Structured output via `claude -p --json-schema ... --output-format json`.

    Returns the dict that conforms to `func_spec.json_schema`.
    """
    schema_str = json.dumps(func_spec.json_schema)
    # Help the model by mentioning the function's purpose in the prompt, since
    # `--json-schema` itself only specifies shape, not intent.
    intent_hint = (
        f"\n\n[TASK]\nProduce structured output for the tool `{func_spec.name}`. "
        f"Purpose: {func_spec.description.strip()}\n"
    )
    prompt = _flatten_prompt(system_message, (user_message or "") + intent_hint)

    result = subprocess.run(
        [
            "claude",
            "-p",
            "--json-schema",
            schema_str,
            "--output-format",
            "json",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude -p (structured) failed (rc={result.returncode}): "
            f"{result.stderr.strip()[:500]}"
        )

    try:
        envelope = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"claude -p returned non-JSON envelope: {e}; first 500 chars: "
            f"{result.stdout[:500]}"
        )

    if envelope.get("is_error"):
        raise RuntimeError(
            f"claude -p reported error: {envelope.get('result', '(no detail)')[:500]}"
        )

    structured = envelope.get("structured_output")
    if structured is None:
        raise RuntimeError(
            "claude -p envelope missing `structured_output`; "
            f"result field: {str(envelope.get('result'))[:500]}"
        )
    return structured


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs: Any,
) -> tuple[OutputType, float, int, int, dict]:
    t0 = time.time()
    if func_spec is None:
        output: OutputType = _call_claude_cli(system_message, user_message)
    else:
        output = _call_claude_cli_structured(system_message, user_message, func_spec)
    req_time = time.time() - t0

    # The CLI doesn't expose token counts in plain-text mode; estimate roughly so
    # the rest of the pipeline (cost/latency tracking) doesn't blow up.
    in_tokens = (len(system_message or "") + len(user_message or "")) // 4
    out_tokens = (
        len(output) // 4
        if isinstance(output, str)
        else len(json.dumps(output)) // 4
    )

    info = {"stop_reason": "end_turn"}
    return output, req_time, in_tokens, out_tokens, info
