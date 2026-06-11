"""Single shared entry point for `claude -p` subprocess calls.

Every claude-cli call site in this fork (treesearch backend, llm.py, vlm.py,
perform_freeform_writeup.py) routes through run_claude() so that timeout,
retry, model pinning, and call logging behave identically everywhere.

Env knobs:
  CLAUDE_CLI_TIMEOUT   per-call timeout in seconds (default 1200)
  CLAUDE_CLI_RETRIES   retries after a failed/timed-out call (default 2)
  CLAUDE_CALL_LOG_DIR  if set, every call's full input + raw output is
                       persisted there — nothing the paid LLM produces
                       should be lost

Model names:
  "claude-cli" / "claude-code"            → the CLI session's default model
  "claude-cli:sonnet" / "claude-cli:haiku" / "claude-cli:opus" (etc.)
                                          → passed as `claude -p --model <alias>`
  This is the only temperature/cost lever the CLI offers — bfts_config's
  temp knobs do not reach the CLI and are reported as ignored by the backend.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
import uuid

logger = logging.getLogger("ai-scientist.claude_cli")

DEFAULT_TIMEOUT = 1200
DEFAULT_RETRIES = 2

_CLI_PREFIXES = ("claude-cli", "claude-code")

_LOG_SEQ = 0


def is_claude_cli(model: object) -> bool:
    """True for 'claude-cli', 'claude-code', and pinned forms like
    'claude-cli:sonnet'."""
    if not isinstance(model, str):
        return False
    base = model.split(":", 1)[0]
    return base in _CLI_PREFIXES


def model_alias(model: str | None) -> str | None:
    """'claude-cli:sonnet' → 'sonnet'; bare 'claude-cli' → None (CLI default)."""
    if model and ":" in model:
        return model.split(":", 1)[1] or None
    return None


def _maybe_log_call(
    kind: str, stdin: str, stdout: str, stderr: str, returncode: int, args: list[str]
) -> None:
    log_dir = os.environ.get("CLAUDE_CALL_LOG_DIR")
    if not log_dir:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        global _LOG_SEQ
        _LOG_SEQ += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = f"{ts}_pid{os.getpid()}_seq{_LOG_SEQ:05d}_{kind}_{uuid.uuid4().hex[:6]}"
        with open(os.path.join(log_dir, f"{tag}_input.txt"), "w", encoding="utf-8") as f:
            f.write(f"# argv: {args}\n# kind: {kind}\n# returncode: {returncode}\n---\n")
            f.write(stdin)
        with open(os.path.join(log_dir, f"{tag}_output.txt"), "w", encoding="utf-8") as f:
            f.write(f"# returncode: {returncode}\n# stderr:\n{stderr}\n---stdout---\n")
            f.write(stdout)
    except Exception as e:  # logging must never crash the run
        logger.warning(f"failed to log claude -p call: {e}")


def run_claude(
    prompt: str,
    kind: str = "plain",
    model: str | None = None,
    timeout: int | None = None,
    retries: int | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """One `claude -p` call with retry/backoff. Returns stripped stdout.

    Retries cover transient failures (nonzero exit, timeout) — a flaky
    5-second CLI hiccup should never cost a 40-minute BFTS node. Raises
    RuntimeError after the final attempt.
    """
    args = ["claude", "-p"]
    alias = model_alias(model)
    if alias:
        args += ["--model", alias]
    args += list(extra_args or [])

    if timeout is None:
        timeout = int(os.environ.get("CLAUDE_CLI_TIMEOUT", str(DEFAULT_TIMEOUT)))
    if retries is None:
        retries = int(os.environ.get("CLAUDE_CLI_RETRIES", str(DEFAULT_RETRIES)))

    last_err = "unknown"
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {timeout}s"
            _maybe_log_call(kind, prompt, "", f"TIMEOUT attempt {attempt + 1}", -1, args)
        else:
            _maybe_log_call(kind, prompt, result.stdout, result.stderr, result.returncode, args)
            if result.returncode == 0:
                return result.stdout.strip()
            last_err = f"rc={result.returncode}: {result.stderr.strip()[:300]}"
        if attempt < retries:
            backoff = 5 * (attempt + 1)
            logger.warning(
                f"claude -p ({kind}) failed ({last_err}); "
                f"retry {attempt + 1}/{retries} in {backoff}s"
            )
            time.sleep(backoff)
    raise RuntimeError(f"claude -p ({kind}) failed after {retries + 1} attempt(s): {last_err}")
