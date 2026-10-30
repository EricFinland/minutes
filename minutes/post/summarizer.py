"""Meeting summary with a local Ollama model.

Short meetings are summarized in one pass. Long ones are split into chunks that fit
the model's context (map), each chunk becomes notes, and the notes are merged (reduce).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import requests

from minutes.models import SUMMARY_FILE, Line, meeting_title, write_text_atomic
from minutes.post.export import load_lines

SECTIONS = ["Summary", "Key points", "Decisions", "Action items", "Open questions"]

CONTEXT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 1024
SAFETY = 0.85          # the chars/4 estimate is rough, leave headroom
TEMPERATURE = 0.2
MIN_WORDS = 20         # below this there is nothing worth summarizing
REQUEST_TIMEOUT = (5, 900)
START_WAIT_S = 15

Progress = Callable[[str], None] | None


class SummaryError(Exception):
    """Raised with a message that can be shown to the user as is."""


SYSTEM_PROMPT = """You write accurate meeting notes from a transcript.
The transcript has exactly two speakers: "Me" (the person taking the notes) and "Them" (everyone else on the call).

Rules:
- Use only facts that are stated in the transcript. Do not guess or add background knowledge.
- Never invent names, dates, times, numbers, prices, owners or deadlines. Only repeat them if they appear in the transcript.
- Attribute statements, decisions and tasks to "Me" or "Them", using the speaker label of the line where it was said. Use a person's name only if it is spoken in the transcript.
- If the transcript is unclear about something, leave it out rather than guessing.
- If a section has nothing to report, write exactly: None
- No preamble, no closing remarks, no commentary about these rules.
- Write in the same language as the transcript."""

FORMAT_SPEC = """Use exactly this Markdown format:

## Summary
3 to 5 plain sentences describing what the meeting was about and its outcome.

## Key points
- one bullet per important point that was discussed

## Decisions
- one bullet per decision that was clearly agreed

## Action items
- Me: task (deadline only if one was stated)
- Them: task
- task (when the owner is not clear)

## Open questions
- one bullet per question or issue left unresolved"""

NOTES_FORMAT = """## Summary
- short bullets describing what was discussed

## Key points
- bullets

## Decisions
- bullets

## Action items
- Me: task, or Them: task, or just the task if the owner is not clear

## Open questions
- bullets"""

MAP_SPEC = (
    "This is part {part} of {total} of a longer meeting transcript. Write notes for this "
    "part only, using exactly this Markdown format:\n\n" + NOTES_FORMAT
)

REDUCE_SPEC = """Below are notes taken from consecutive parts of one meeting, in order. Merge them into a single set of meeting notes. Remove duplicates, keep every distinct decision and action item, and do not add anything that is not in the notes."""


# Token budgeting ------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def _overhead_tokens() -> int:
    longest = max(len(FORMAT_SPEC), len(MAP_SPEC) + 20, len(REDUCE_SPEC) + len(FORMAT_SPEC))
    return estimate_tokens(SYSTEM_PROMPT) + longest // 4 + 64


def chunk_budget_tokens() -> int:
    """Tokens of transcript (or notes) that fit in one request."""
    return int((CONTEXT_TOKENS - MAX_OUTPUT_TOKENS - _overhead_tokens()) * SAFETY)


def chunk_text_lines(rows: list[str], budget_tokens: int) -> list[str]:
    """Group rows into chunks under budget_tokens, never splitting a row unless it alone
    is too long (then it is cut on word boundaries)."""
    max_chars = max(budget_tokens * 4, 200)
    pieces: list[str] = []
    for row in rows:
        if len(row) <= max_chars:
            pieces.append(row)
            continue
        cur = ""
        for word in row.split(" "):
            if cur and len(cur) + 1 + len(word) > max_chars:
                pieces.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}" if cur else word
        if cur:
            pieces.append(cur)

    chunks: list[str] = []
    cur_rows: list[str] = []
    cur_len = 0
    for p in pieces:
        add = len(p) + 1
        if cur_rows and cur_len + add > max_chars:
            chunks.append("\n".join(cur_rows))
            cur_rows, cur_len = [], 0
        cur_rows.append(p)
        cur_len += add
    if cur_rows:
        chunks.append("\n".join(cur_rows))
    return chunks


def _ts(seconds: float) -> str:
    s = max(0, int(seconds))
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcript_rows(lines: list[Line]) -> list[str]:
    """Compact transcript rows for the model (short timestamps save tokens)."""
    return [f"[{_ts(l.start)}] {l.speaker}: {l.text.strip()}" for l in lines if l.text.strip()]


# Prompt assembly ------------------------------------------------------------------

def build_single_messages(transcript: str) -> list[dict]:
    user = f"{FORMAT_SPEC}\n\nTranscript:\n<<<\n{transcript}\n>>>"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_map_messages(chunk: str, part: int, total: int) -> list[dict]:
    user = f"{MAP_SPEC.format(part=part, total=total)}\n\nTranscript part {part}:\n<<<\n{chunk}\n>>>"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_reduce_messages(notes: list[str], final: bool = True) -> list[dict]:
    joined = "\n\n".join(f"Notes for part {i}:\n{n.strip()}" for i, n in enumerate(notes, 1))
    spec = FORMAT_SPEC if final else MAP_SPEC.format(part="1", total="1").split("\n", 1)[1]
    user = f"{REDUCE_SPEC}\n\n{spec}\n\n<<<\n{joined}\n>>>"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# Output normalization -------------------------------------------------------------

_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(summary|key points|decisions|action items|open questions)"
    r"\s*(?:\*\*|__)?\s*:?\s*(?:\*\*|__)?\s*$",
    re.IGNORECASE,
)
_NONE_RE = re.compile(r"^[\s\-*]*(none|n/a|nothing)\.?\s*$", re.IGNORECASE)


def _clean(text: str) -> str:
    return text.replace(chr(0x2014), ", ").replace(chr(0x2013), "-").strip()


def parse_sections(text: str) -> dict[str, str]:
    """Split model output into the five known sections. Missing ones are absent."""
    canonical = {s.lower(): s for s in SECTIONS}
    found: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        m = _HEADING_RE.match(raw)
        if m:
            current = canonical[m.group(1).lower()]
            found.setdefault(current, [])
            continue
        if current is not None:
            found[current].append(raw.rstrip())
    return {k: "\n".join(v).strip() for k, v in found.items()}


def render_summary(title: str, sections: dict[str, str]) -> str:
    out = [f"# {title}", ""]
    for name in SECTIONS:
        body = _clean(sections.get(name, ""))
        if not body or _NONE_RE.match(body):
            body = "None"
        out += [f"## {name}", "", body, ""]
    return "\n".join(out).rstrip() + "\n"


def normalize_output(title: str, raw: str) -> str:
    sections = parse_sections(raw)
    if not sections and raw.strip():
        sections = {"Summary": raw.strip()}
    return render_summary(title, sections)


# Ollama ---------------------------------------------------------------------------

def _ollama_exe() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA")
    if local:
        guess = Path(local) / "Programs" / "Ollama" / "ollama.exe"
        if guess.exists():
            return str(guess)
    return None


def _start_ollama() -> bool:
    exe = _ollama_exe()
    if not exe:
        return False
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.Popen(
            [exe, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
        return True
    except OSError:
        return False


def _installed_models(url: str) -> list[str]:
    r = requests.get(f"{url}/api/tags", timeout=5)
    r.raise_for_status()
    return [m.get("name", "") for m in r.json().get("models", [])]


def _model_present(model: str, names: list[str]) -> bool:
    for n in names:
        if n == model:
            return True
        if n.endswith(":latest") and n[: -len(":latest")] == model:
            return True
        if model.endswith(":latest") and model[: -len(":latest")] == n:
            return True
    return False


def _missing_model_error(model: str) -> SummaryError:
    return SummaryError(
        f'The summary model "{model}" is not installed in Ollama. '
        f"Open a terminal and run: ollama pull {model}"
    )


def ensure_ollama(cfg: dict, on_progress: Progress = None) -> None:
    """Make sure Ollama answers and has the model. Starts it once if it is not running."""
    url = cfg["ollama_url"].rstrip("/")
    model = cfg["summary_model"]
    try:
        names = _installed_models(url)
    except requests.RequestException:
        if on_progress:
            on_progress("Starting Ollama...")
        if not _start_ollama():
            raise SummaryError(
                "Ollama is not running and could not be started. Install it from ollama.com "
                "or start it yourself (run: ollama serve), then try again."
            ) from None
        names = None
        deadline = time.monotonic() + START_WAIT_S
        while time.monotonic() < deadline:
            time.sleep(0.5)
            try:
                names = _installed_models(url)
                break
            except requests.RequestException:
                continue
        if names is None:
            raise SummaryError(
                "Ollama is not running. Start it (run: ollama serve, or open the Ollama app) "
                "and try again."
            )
    if not _model_present(model, names):
        raise _missing_model_error(model)


def chat(cfg: dict, messages: list[dict]) -> str:
    url = cfg["ollama_url"].rstrip("/")
    payload = {
        "model": cfg["summary_model"],
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": 0.9,
            "num_ctx": CONTEXT_TOKENS,
            "num_predict": MAX_OUTPUT_TOKENS,
        },
    }
    try:
        r = requests.post(f"{url}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        raise SummaryError("Ollama took too long to answer. Try again in a moment.") from None
    except requests.RequestException:
        raise SummaryError(
            "Lost the connection to Ollama. Make sure it is running (ollama serve) and try again."
        ) from None
    if r.status_code == 404:
        raise _missing_model_error(cfg["summary_model"])
    if r.status_code != 200:
        try:
            detail = r.json().get("error", r.text)
        except ValueError:
            detail = r.text
        raise SummaryError(f"Ollama returned an error: {str(detail)[:300]}")
    try:
        return r.json()["message"]["content"]
    except (ValueError, KeyError, TypeError):
        raise SummaryError("Ollama sent back a response that could not be read.") from None


# Main entry -----------------------------------------------------------------------

def _not_enough_content(title: str) -> str:
    return render_summary(
        title,
        {"Summary": "Not enough was said in this meeting to write a summary."},
    )


def summarize_rows(rows: list[str], cfg: dict, on_progress: Progress = None) -> str:
    """Return raw model notes (five sections) for the given transcript rows."""
    budget = chunk_budget_tokens()
    chunks = chunk_text_lines(rows, budget)
    if len(chunks) == 1:
        if on_progress:
            on_progress("Summarizing...")
        return chat(cfg, build_single_messages(chunks[0]))

    notes = []
    for i, chunk in enumerate(chunks, 1):
        if on_progress:
            on_progress(f"Summarizing part {i} of {len(chunks)}...")
        notes.append(chat(cfg, build_map_messages(chunk, i, len(chunks))))

    # Merge notes, in groups if they do not all fit at once.
    while True:
        joined_tokens = sum(estimate_tokens(n) + 8 for n in notes)
        if joined_tokens <= budget or len(notes) == 1:
            if on_progress:
                on_progress("Combining notes...")
            return chat(cfg, build_reduce_messages(notes, final=True))
        groups: list[list[str]] = [[]]
        size = 0
        for n in notes:
            t = estimate_tokens(n) + 8
            if groups[-1] and size + t > budget:
                groups.append([])
                size = 0
            groups[-1].append(n)
            size += t
        if len(groups) == len(notes):
            # Grouping cannot shrink anything further; merge what we have in one go.
            if on_progress:
                on_progress("Combining notes...")
            return chat(cfg, build_reduce_messages(notes, final=True))
        if on_progress:
            on_progress("Combining notes...")
        notes = [
            chat(cfg, build_reduce_messages(g, final=False)) if len(g) > 1 else g[0]
            for g in groups
        ]


def summarize(meeting_dir, cfg: dict, on_progress: Progress = None) -> Path:
    """Write summary.md for a meeting and return its path. Raises SummaryError."""
    meeting_dir = Path(meeting_dir)
    title = meeting_title(meeting_dir)
    out_path = meeting_dir / SUMMARY_FILE

    lines = load_lines(meeting_dir)
    rows = transcript_rows(lines)
    words = sum(len(l.text.split()) for l in lines)
    if words < MIN_WORDS:
        write_text_atomic(out_path, _not_enough_content(title))
        if on_progress:
            on_progress("Saved summary")
        return out_path

    if on_progress:
        on_progress("Checking Ollama...")
    ensure_ollama(cfg, on_progress)
    raw = summarize_rows(rows, cfg, on_progress)
    write_text_atomic(out_path, normalize_output(title, raw))
    if on_progress:
        on_progress("Saved summary")
    return out_path
