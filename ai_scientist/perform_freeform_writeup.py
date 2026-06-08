"""Free-form markdown writeup for a Sakana BFTS experiment run.

This module replaces Sakana's `perform_icbinb_writeup.py` for Leaflet. It
intentionally does NOT do citation gathering (Leaflet's STORM step covers
that), does NOT call a VLM on plots (the plot's source code is enough to
describe it), and does NOT enforce a page limit. Output is one self-contained
`report.md` with all plots embedded as relative-path image references; render
with any markdown viewer, or convert to PDF later via `pandoc` if you want.

CLI:

  python -m ai_scientist.perform_freeform_writeup \
      --experiment-dir experiments/<timestamp>_<idea>_attempt_0 \
      --storm-survey path/to/storm_gen_article_polished.txt \
      --coscientist-overview path/to/overview.md \
      [--output <experiment-dir>/report.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------- I/O helpers ----------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_npy_summary(path: Path) -> str:
    """One-line description of a .npy file's contents."""
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as e:
        return f"(failed to load: {e})"
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        obj = arr.item()
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                parts.append(_describe_value(k, v))
            return "; ".join(parts)
        return f"object-scalar of type {type(obj).__name__}"
    return _describe_value(path.name, arr)


def _describe_value(key: Any, v: Any) -> str:
    if isinstance(v, np.ndarray):
        if np.issubdtype(v.dtype, np.number) and v.size:
            return (
                f"{key}: ndarray shape={tuple(v.shape)} dtype={v.dtype} "
                f"min={float(np.nanmin(v)):.4g} max={float(np.nanmax(v)):.4g} "
                f"mean={float(np.nanmean(v)):.4g}"
            )
        return f"{key}: ndarray shape={tuple(v.shape)} dtype={v.dtype}"
    if isinstance(v, dict):
        return f"{key}: dict keys={list(v.keys())[:8]}"
    if isinstance(v, (list, tuple)):
        return f"{key}: {type(v).__name__} len={len(v)}"
    return f"{key}: {type(v).__name__}={str(v)[:60]}"


def _extract_savefig_context(code: str, plot_filename: str, window: int = 25) -> str:
    """Find the savefig call for `plot_filename` in `code` and return ~window lines above it.

    Handles multi-line `plt.savefig(\\n    os.path.join(dir, "name.png"))` patterns
    where the filename literal is on a different line from the `savefig` keyword.
    """
    lines = code.splitlines()
    stem = Path(plot_filename).stem
    for target in (plot_filename, stem):
        # Find every line where the filename literal appears
        for i, line in enumerate(lines):
            if target not in line:
                continue
            # Walk back up to 4 lines to see if there's a `savefig(` opening this call
            for back in range(0, 5):
                j = i - back
                if j < 0:
                    break
                if "savefig" in lines[j]:
                    start = max(0, j - window)
                    end = min(len(lines), i + 2)
                    return "\n".join(lines[start:end])
    return ""


# ---------- Plot discovery ----------


@dataclass
class PlotRecord:
    path: Path                # absolute path on disk
    rel_path: str             # path relative to the experiment dir, for embedding
    filename: str             # basename only
    node_id: str              # BFTS node hash
    stage_hint: str           # which stage it likely came from
    code_snippet: str         # ~25 lines of plotting code around the savefig
    data_summary: str         # 1-line summary of the sibling experiment_data.npy
    code_hash: str            # to dedupe across re-runs / seeds


def discover_plots(experiment_dir: Path) -> list[PlotRecord]:
    records: list[PlotRecord] = []
    results_dir = experiment_dir / "logs" / "0-run" / "experiment_results"
    if not results_dir.exists():
        return records

    # First pass: collect all experiment_code.py files so we can look up a
    # plot's source even when the savefig was emitted by a sibling node.
    all_code_files: list[tuple[str, str]] = []  # (node_id, code)
    for proc_dir in sorted(results_dir.iterdir()):
        if not proc_dir.is_dir():
            continue
        m = re.match(r"experiment_([0-9a-f]+)_proc_(\d+)", proc_dir.name)
        node_id = m.group(1) if m else proc_dir.name
        code_path = proc_dir / "experiment_code.py"
        if code_path.exists():
            all_code_files.append((node_id, _read(code_path)))

    def find_snippet_anywhere(plot_filename: str) -> str:
        for _, code in all_code_files:
            snippet = _extract_savefig_context(code, plot_filename)
            if snippet:
                return snippet
        return ""

    for proc_dir in sorted(results_dir.iterdir()):
        if not proc_dir.is_dir():
            continue
        m = re.match(r"experiment_([0-9a-f]+)_proc_(\d+)", proc_dir.name)
        node_id = m.group(1) if m else proc_dir.name
        code_path = proc_dir / "experiment_code.py"
        code = _read(code_path) if code_path.exists() else ""
        code_hash = hashlib.sha1(code.encode()).hexdigest()[:8] if code else "nocode"
        npy_path = proc_dir / "experiment_data.npy"
        data_summary = _safe_npy_summary(npy_path) if npy_path.exists() else ""

        for png in sorted(proc_dir.glob("*.png")):
            snippet = _extract_savefig_context(code, png.name)
            if not snippet:
                snippet = find_snippet_anywhere(png.name)
            records.append(
                PlotRecord(
                    path=png,
                    rel_path=os.path.relpath(png, experiment_dir),
                    filename=png.name,
                    node_id=node_id,
                    stage_hint="(unknown)",  # filled below
                    code_snippet=snippet,
                    data_summary=data_summary,
                    code_hash=code_hash,
                )
            )
    return records


def dedupe_plots(records: list[PlotRecord]) -> tuple[list[PlotRecord], list[PlotRecord]]:
    """Group by (filename, code_hash) and pick one representative per group.

    Returns (headline, appendix). Headline gets plots whose filename hints at
    a *result* (delta, worst, per_class, final, summary); appendix is the rest.
    """
    seen: dict[tuple[str, str], PlotRecord] = {}
    for r in records:
        key = (r.filename, r.code_hash)
        if key not in seen:
            seen[key] = r
    unique = list(seen.values())

    def is_result(r: PlotRecord) -> bool:
        f = r.filename.lower()
        return any(t in f for t in ("delta", "worst", "per_class", "final", "summary", "ablation"))

    headline = [r for r in unique if is_result(r)]
    appendix = [r for r in unique if not is_result(r)]
    return headline, appendix


# ---------- Journal summarization ----------


def summarize_journal(experiment_dir: Path) -> str:
    """Concatenate per-stage journal summaries.

    Picks the latest journal.json under each stage dir and pulls plan/analysis/metric
    for each node.
    """
    logs_dir = experiment_dir / "logs" / "0-run"
    if not logs_dir.exists():
        return ""
    out_chunks: list[str] = []
    for stage_dir in sorted(logs_dir.glob("stage_*")):
        j = stage_dir / "journal.json"
        if not j.exists():
            continue
        try:
            data = json.loads(_read(j))
        except json.JSONDecodeError:
            continue
        out_chunks.append(f"### Stage: {stage_dir.name}\n")
        nodes = data.get("nodes", []) if isinstance(data, dict) else data
        for n in nodes if isinstance(nodes, list) else []:
            nid = n.get("id", "?")
            plan = (n.get("plan") or "").strip()
            analysis = (n.get("analysis") or "").strip()
            metric = n.get("metric")
            is_buggy = n.get("is_buggy")
            head = f"- **node {nid[:8]}** "
            if is_buggy:
                head += "(buggy)"
            elif metric:
                head += f"(metric: {metric})"
            out_chunks.append(head)
            if plan:
                out_chunks.append(f"  - *plan:* {plan[:400]}")
            if analysis:
                out_chunks.append(f"  - *analysis:* {analysis[:400]}")
        out_chunks.append("")
    return "\n".join(out_chunks)


# ---------- Prompt assembly ----------

SYSTEM_PROMPT = """You are writing a free-form research report on an experiment that has already been run. The experimental data, plots, and code are given to you below. Your job is to write a clear, readable report in GitHub-flavored Markdown.

Rules:
- DO NOT invent citations. The only citations in the report live in the "Field state" section, which is taken verbatim from STORM (an external literature-survey agent). You may refer back to that section with phrases like "as the field-state survey notes above", but do not add new citation markers.
- DO NOT enforce a page limit. Write as much as the material warrants, no more.
- DO NOT generate a stereotypical conference paper. This is a free-form report. Use whatever structure best fits the material.
- DO embed plots inline as `![<short description>](relative/path/to/plot.png)`. The relative paths are given to you in the plot list.
- DO use the plotting code and data summaries given to you to describe what each plot shows. You do NOT have vision access; rely on the code that generated the plot.
- DO group similar plots and avoid restating the same result multiple times.
- DO include an Appendix at the end listing every plot (with embeds) so nothing the experiment produced is lost.
- DO be honest about what the experiment actually showed, including null/negative results.

Suggested structure (deviate as needed):
1. Question — one paragraph, what we were trying to find out
2. Field state — embed the STORM survey block verbatim (it will be marked clearly in the inputs)
3. Hypothesis — from the Co-Scientist overview, briefly
4. What we ran — a short walkthrough of the BFTS journey, stage by stage, with the actual experiments tried
5. Results — headline plots inline, with what each shows from the plotting code
6. What this means — your honest read of the results
7. Limitations and what we'd do next
8. Appendix — all plots + paths to code and data
"""


def build_prompt(
    experiment_dir: Path,
    idea_md: str,
    storm_survey: str | None,
    coscientist_overview: str | None,
    journal_summary: str,
    headline: list[PlotRecord],
    appendix: list[PlotRecord],
) -> str:
    parts: list[str] = []

    parts.append("# Inputs for the report\n")

    parts.append("## Sakana idea seed (idea.md, verbatim)\n")
    parts.append(idea_md)
    parts.append("")

    if storm_survey:
        parts.append("## STORM literature survey (verbatim — this is the Field state section)\n")
        parts.append("<<STORM_BEGIN>>\n")
        parts.append(storm_survey.strip())
        parts.append("\n<<STORM_END>>\n")
    else:
        parts.append("## STORM literature survey\n")
        parts.append("(no STORM survey was provided for this run — omit the Field state section)\n")

    if coscientist_overview:
        parts.append("## Co-Scientist hypothesis overview (verbatim)\n")
        parts.append("<<COSCIENTIST_BEGIN>>\n")
        parts.append(coscientist_overview.strip())
        parts.append("\n<<COSCIENTIST_END>>\n")
    else:
        parts.append("## Co-Scientist hypothesis overview\n")
        parts.append("(no Co-Scientist overview was provided)\n")

    parts.append("## BFTS journal summary\n")
    parts.append(journal_summary if journal_summary else "(no journal entries found)")
    parts.append("")

    parts.append("## Plots — headline candidates (use these inline in the Results section)\n")
    for r in headline:
        parts.append(f"### {r.filename}")
        parts.append(f"- relative path (use this in the embed): `{r.rel_path}`")
        parts.append(f"- BFTS node: `{r.node_id[:12]}` (code-hash `{r.code_hash}`)")
        if r.data_summary:
            parts.append(f"- sibling experiment_data.npy: {r.data_summary}")
        if r.code_snippet:
            parts.append("- plotting code that produced this:")
            parts.append("```python")
            parts.append(r.code_snippet)
            parts.append("```")
        parts.append("")

    parts.append("## Plots — appendix candidates (training curves, supporting plots)\n")
    for r in appendix:
        parts.append(f"### {r.filename}")
        parts.append(f"- relative path: `{r.rel_path}`")
        parts.append(f"- BFTS node: `{r.node_id[:12]}` (code-hash `{r.code_hash}`)")
        if r.data_summary:
            parts.append(f"- sibling experiment_data.npy: {r.data_summary}")
        parts.append("")

    parts.append(
        "\n---\n\n"
        "Now write the report. Output ONLY the markdown for report.md, nothing else.\n"
    )
    return "\n".join(parts)


# ---------- Driver ----------


def _call_claude(prompt: str, system: str, timeout: int = 1200) -> str:
    """Call `claude -p` with the assembled prompt. Returns stdout."""
    full = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}\n\n[ASSISTANT]\n"
    result = subprocess.run(
        ["claude", "-p"],
        input=full,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def perform_freeform_writeup(
    experiment_dir: Path,
    storm_survey_path: Path | None,
    coscientist_overview_path: Path | None,
    output_path: Path | None = None,
) -> Path:
    experiment_dir = experiment_dir.resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(experiment_dir)

    idea_md_path = experiment_dir / "idea.md"
    idea_md = _read(idea_md_path) if idea_md_path.exists() else ""
    storm_survey = _read(storm_survey_path) if storm_survey_path else None
    cosci = _read(coscientist_overview_path) if coscientist_overview_path else None

    records = discover_plots(experiment_dir)
    headline, appendix = dedupe_plots(records)
    journal_summary = summarize_journal(experiment_dir)

    print(
        f"Discovered {len(records)} plot files "
        f"({len(headline)} headline, {len(appendix)} appendix after dedupe)."
    )
    print(f"Journal summary: {len(journal_summary)} chars")

    prompt = build_prompt(
        experiment_dir=experiment_dir,
        idea_md=idea_md,
        storm_survey=storm_survey,
        coscientist_overview=cosci,
        journal_summary=journal_summary,
        headline=headline,
        appendix=appendix,
    )

    prompt_path = experiment_dir / "freeform_writeup_prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"Prompt ({len(prompt)} chars) saved to {prompt_path}")

    print("Calling claude -p (this may take several minutes)...")
    report_md = _call_claude(prompt, SYSTEM_PROMPT)

    output_path = output_path or (experiment_dir / "report.md")
    output_path.write_text(report_md, encoding="utf-8")
    print(f"Wrote report to {output_path} ({len(report_md)} chars)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--storm-survey", type=Path, default=None)
    parser.add_argument("--coscientist-overview", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    perform_freeform_writeup(
        experiment_dir=args.experiment_dir,
        storm_survey_path=args.storm_survey,
        coscientist_overview_path=args.coscientist_overview,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
