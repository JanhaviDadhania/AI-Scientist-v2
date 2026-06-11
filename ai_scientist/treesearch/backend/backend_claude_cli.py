"""Backend that shells out to the local `claude -p` CLI.

Used when the configured model name is `claude-cli` or `claude-code`. Mirrors
the signature of the other treesearch backends so it plugs into `backend/__init__.py`.

PATCHED 2026-06-09 (Option 2 — strict-prompt structured output):
  - Drop the broken `claude -p --json-schema X --output-format json` path.
    That envelope intermittently omits `structured_output` and the model's
    JSON ends up wrapped in chat-form `result` text — which then either
    cascades into is_buggy=True via the upstream wrapper at
    parallel_agent.py:1654, or KeyError's downstream because the chat-form
    JSON shape doesn't match the wrapper's expectations.
  - Instead: call `claude -p` plain, with a strict prompt header that tells
    the model to output EXACTLY one JSON object matching the schema and
    nothing else. Parse stdout directly with json.loads. Tested 12/12 on
    review / parse-metrics / nested-adversarial schemas (test_strict_json.py).
  - The old regex fallback is retained as `_extract_json_from_chat` for use
    as a deep last-resort if json.loads itself fails (e.g. claude adds a
    fenced block despite instructions).
  - Per-call disk logging via env var CLAUDE_CALL_LOG_DIR is preserved.
    Nothing the paid LLM produces should be lost.
"""

import base64
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from typing import Any

from .utils import FunctionSpec, OutputType


logger = logging.getLogger("ai-scientist.claude_cli")

CLAUDE_CLI_MODEL_NAMES = {"claude-cli", "claude-code"}

_LOG_SEQ = 0


def _maybe_log_call(kind: str, stdin: str, stdout: str, stderr: str, returncode: int, args: list[str]) -> None:
    """If CLAUDE_CALL_LOG_DIR is set, persist this call's IO to disk."""
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
    except Exception as e:
        logger.warning(f"failed to log claude -p call: {e}")


def _extract_json_from_chat(text: str) -> dict:
    """Deep last-resort: extract JSON from chat-style text using fence or brace
    scan. Used only when the strict prompt failed to elicit clean JSON."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None
                    depth = 0
    raise RuntimeError(
        f"no JSON object extractable from chat text; first 500 chars: {text[:500]}"
    )


def _extract_images(message: Any) -> tuple[Any, list[str]]:
    """Split an OpenAI-style multi-modal content-block list into plain text
    plus image file paths.

    compile_prompt_to_md (backend/utils.py) passes lists where every item is a
    dict with a "type" key through UNCHANGED — that is how vision messages
    (e.g. _analyze_plots_with_vlm's text + base64 image_url blocks) reach this
    backend. `claude -p` is multimodal but reads images from DISK via its Read
    tool, so base64 data-URLs are decoded back to image files in a temp dir.
    Any other message shape is returned unchanged with no images.
    """
    if not (
        isinstance(message, list)
        and message
        and all(isinstance(b, dict) and "type" in b for b in message)
    ):
        return message, []
    texts: list[str] = []
    paths: list[str] = []
    img_dir = None
    for block in message:
        btype = block.get("type")
        if btype == "text":
            texts.append(block.get("text", ""))
        elif btype == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            m = re.match(r"data:image/(\w+);base64,(.+)", url, re.DOTALL)
            if m:
                ext = m.group(1) if m.group(1) in ("png", "jpeg", "jpg", "webp", "gif") else "png"
                if img_dir is None:
                    img_dir = tempfile.mkdtemp(prefix="claude_cli_imgs_")
                path = os.path.join(img_dir, f"img_{len(paths):02d}.{ext}")
                try:
                    with open(path, "wb") as f:
                        f.write(base64.b64decode(m.group(2)))
                    paths.append(path)
                except Exception as e:
                    logger.warning(f"failed to decode image block to disk: {e}")
            elif url:
                texts.append(f"(image available at URL: {url})")
        else:
            texts.append(str(block))
    return "\n".join(t for t in texts if t), paths


def _image_instructions(paths: list[str]) -> str:
    listing = "\n".join(f"- {p}" for p in paths)
    return (
        "\n[IMAGES]\n"
        "The following image files are part of this request. Use your Read tool "
        "to view EVERY one of them before answering. Base your analysis only on "
        "what you actually see in the images; never invent plot contents. Refer "
        "to the images in the order listed.\n"
        f"{listing}\n"
    )


def _flatten_prompt(system_message: Any, user_message: Any) -> str:
    parts: list[str] = []
    if isinstance(system_message, dict):
        sub = []
        for k, v in system_message.items():
            sub.append(f"## {k}\n{v}\n")
        sys_text = "\n".join(sub)
        parts.append(f"[SYSTEM]\n{sys_text}\n")
    elif system_message:
        parts.append(f"[SYSTEM]\n{system_message}\n")
    if user_message:
        if isinstance(user_message, dict):
            sub_parts = []
            for k, v in user_message.items():
                sub_parts.append(f"## {k}\n{v}\n")
            user_message = "\n".join(sub_parts)
        parts.append(f"[USER]\n{user_message}\n")
    parts.append("[ASSISTANT]\n")
    return "\n".join(parts)


def _call_claude_cli(
    system_message: Any, user_message: Any, extra_args: list[str] | None = None
) -> str:
    """Plain text out — used when no function-calling is requested."""
    args = ["claude", "-p", *(extra_args or [])]
    stdin = _flatten_prompt(system_message, user_message)
    result = subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=900,
    )
    _maybe_log_call("plain", stdin, result.stdout, result.stderr, result.returncode, args)
    if result.returncode != 0:
        raise RuntimeError(
            f"claude -p failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


# ─── Option 2 strict-output prompt header ──────────────────────────────────
# Drop the --json-schema flag entirely. Instead, prepend this header so the
# model knows it must emit one JSON object and nothing else. Tested 12/12 on
# the review/parse_metrics/adversarial schemas. Note: this string contains
# literal `{` and `}` in the rules text, so it must NOT be passed through
# str.format() — the tool name/purpose line is composed separately below.
STRICT_RULES = """You must respond with EXACTLY one JSON object that matches the schema below, and ABSOLUTELY NOTHING else.

Hard rules:
- No prose before or after the JSON.
- No markdown code fences. No backticks.
- No commentary, no explanation, no preamble like "Here's the JSON:".
- Begin your response with `{` and end with `}`.
- The result must be valid JSON, parseable by Python's `json.loads()`.

"""


def _strip_common_wrappers(text: str) -> str:
    """Minimal cleanup if the model defied the strict instruction. Strips
    surrounding whitespace, a single ``` fence pair, a leading 'json' tag,
    and leading/trailing prose lines that don't start with '{' or end with '}'."""
    t = text.strip()
    # Strip a single ``` fence pair if present
    if t.startswith("```"):
        # Drop the opening line (which may say ```json)
        if "\n" in t:
            t = t.split("\n", 1)[1]
        # Drop the closing fence
        if t.rstrip().endswith("```"):
            t = t.rstrip()[: -len("```")].rstrip()
    # Drop a leading 'json' tag on its own line
    if t.lower().startswith("json\n"):
        t = t[5:].strip()
    return t.strip()


def _call_claude_cli_structured(
    system_message: Any,
    user_message: Any,
    func_spec: FunctionSpec,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Structured output via strict-prompt (Option 2). Calls `claude -p` with
    no `--json-schema` flag and parses stdout directly. Falls back to a
    chat-text extractor only if json.loads can't recover even after cleanup.
    """
    schema_str = json.dumps(func_spec.json_schema, indent=2)
    # Compose the prompt header WITHOUT using str.format() — the rules text
    # contains literal `{` and `}` which would break .format() (this was the
    # KeyError we hit on first run). Tool name / purpose are appended via
    # plain concatenation instead.
    schema_line = (
        "SCHEMA (for tool `" + func_spec.name + "` — "
        + func_spec.description.strip() + "):\n"
    )
    body = _flatten_prompt(system_message, user_message)
    stdin = STRICT_RULES + schema_line + schema_str + "\n\n---\n\n" + body

    args = ["claude", "-p", *(extra_args or [])]
    result = subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=900,
    )
    _maybe_log_call("structured-strict", stdin, result.stdout, result.stderr, result.returncode, args)

    if result.returncode != 0:
        raise RuntimeError(
            f"claude -p (strict) failed (rc={result.returncode}): "
            f"{result.stderr.strip()[:500]}"
        )

    raw = result.stdout.strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Light cleanup (strip a stray fence or 'json' tag)
    cleaned = _strip_common_wrappers(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Deep fallback: extract a JSON object from chat-style text
    try:
        extracted = _extract_json_from_chat(raw)
        logger.info("strict-prompt parse failed; deep fallback succeeded")
        return extracted
    except RuntimeError as e:
        raise RuntimeError(
            f"strict-prompt + cleanup + chat-extract all failed for tool "
            f"`{func_spec.name}`. Raw response (first 500 chars): {raw[:500]}"
        )


def query(
    system_message: Any,
    user_message: Any,
    func_spec: FunctionSpec | None = None,
    **model_kwargs: Any,
) -> tuple[OutputType, float, int, int, dict]:
    t0 = time.time()

    # Vision support: pull image blocks out of multi-modal messages, write
    # them to disk, and instruct claude -p to Read them.
    system_message, sys_imgs = _extract_images(system_message)
    user_message, usr_imgs = _extract_images(user_message)
    image_paths = sys_imgs + usr_imgs
    extra_args: list[str] = []
    if image_paths:
        instr = _image_instructions(image_paths)
        if isinstance(user_message, str) or user_message is None:
            user_message = f"{user_message}\n{instr}" if user_message else instr
        else:  # dict-style prompt: add as its own section
            user_message = dict(user_message)
            user_message["Images"] = instr
        # Non-interactive `claude -p` can only Read files inside its working
        # directories; grant the image dirs explicitly via --add-dir.
        for d in sorted({os.path.dirname(os.path.abspath(p)) for p in image_paths}):
            extra_args += ["--add-dir", d]

    if func_spec is None:
        output: OutputType = _call_claude_cli(system_message, user_message, extra_args)
    else:
        output = _call_claude_cli_structured(
            system_message, user_message, func_spec, extra_args
        )
    req_time = time.time() - t0

    in_tokens = (len(str(system_message) if system_message else "") + len(str(user_message) if user_message else "")) // 4
    out_tokens = (
        len(output) // 4
        if isinstance(output, str)
        else len(json.dumps(output)) // 4
    )

    info = {"stop_reason": "end_turn"}
    return output, req_time, in_tokens, out_tokens, info
