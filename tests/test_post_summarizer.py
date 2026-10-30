"""Summarizer tests: chunking, prompts, output shape with a fake Ollama, and one real call."""
from __future__ import annotations

import time

import pytest
import requests

from minutes import config
from minutes.models import LINES_FILE, SUMMARY_FILE, TRANSCRIPT_FILE, Line, append_line
from minutes.post import SummaryError, summarize
from minutes.post import summarizer as sm

SECTION_HEADERS = ["## Summary", "## Key points", "## Decisions", "## Action items",
                   "## Open questions"]

MEETING = [
    ("Them", 2, "Morning. So the main thing today is the launch date for the mobile app."),
    ("Me", 9, "Right. QA found two crash bugs in the checkout flow on Friday."),
    ("Them", 15, "How long do you need to fix those?"),
    ("Me", 19, "About three days for both, then one more day of testing."),
    ("Them", 26, "Then let's move the launch from the 4th to the 11th."),
    ("Me", 31, "Agreed, the 11th works. I'll update the release plan today."),
    ("Them", 38, "I'll tell the marketing team so they can shift the email campaign."),
    ("Me", 45, "One open thing: do we still want the beta testers to get early access?"),
    ("Them", 52, "Not sure yet. I need to check with Priya before we decide that."),
    ("Me", 58, "Okay. And the app store screenshots still need the new logo."),
    ("Them", 64, "I can redo the screenshots by Wednesday."),
    ("Me", 70, "Great, thanks. That's everything from my side."),
]


def _cfg(**over):
    cfg = dict(config.DEFAULTS)
    cfg.update(over)
    return cfg


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MINUTES_DATA_DIR", str(tmp_path))
    return tmp_path


def _meeting(data_dir, name="2026-10-29_100000", rows=MEETING):
    d = data_dir / "meetings" / name
    d.mkdir(parents=True)
    for speaker, t, text in rows:
        append_line(d / LINES_FILE, Line(speaker, float(t), float(t) + 4, text))
    return d


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeOllama:
    def __init__(self, models=("llama3:latest",), reply="", up=True):
        self.models = list(models)
        self.reply = reply
        self.up = up
        self.chats = []

    def get(self, url, timeout=None):
        if not self.up:
            raise requests.ConnectionError("refused")
        assert url.endswith("/api/tags")
        return FakeResponse(payload={"models": [{"name": m} for m in self.models]})

    def post(self, url, json=None, timeout=None):
        assert url.endswith("/api/chat")
        self.chats.append(json)
        return FakeResponse(payload={"message": {"role": "assistant", "content": self.reply}})


@pytest.fixture
def fake(monkeypatch):
    f = FakeOllama()
    monkeypatch.setattr(sm.requests, "get", f.get)
    monkeypatch.setattr(sm.requests, "post", f.post)
    return f


# Chunking and prompts -----------------------------------------------------------------

def test_budget_fits_in_context():
    budget = sm.chunk_budget_tokens()
    assert 3000 < budget < sm.CONTEXT_TOKENS - sm.MAX_OUTPUT_TOKENS
    assert sm.estimate_tokens("a" * 400) == 101


def test_chunking_keeps_rows_whole_and_under_budget():
    rows = [f"[00:{i:02d}] Me: " + "word " * 30 for i in range(60)]
    chunks = sm.chunk_text_lines(rows, budget_tokens=200)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)
    assert "\n".join(chunks).splitlines() == rows


def test_chunking_splits_one_huge_row_on_words():
    row = "Them: " + " ".join(f"w{i}" for i in range(2000))
    chunks = sm.chunk_text_lines([row], budget_tokens=100)
    assert len(chunks) > 1 and all(len(c) <= 400 for c in chunks)
    assert " ".join(chunks).split() == row.split()


def test_single_pass_prompt_has_rules_format_and_transcript():
    msgs = sm.build_single_messages("[00:02] Them: hello\n[00:05] Me: hi")
    assert [m["role"] for m in msgs] == ["system", "user"]
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert '"Me"' in system and '"Them"' in system
    assert "Never invent names, dates" in system and "None" in system
    for h in SECTION_HEADERS:
        assert h in user
    assert "[00:05] Me: hi" in user


def test_map_and_reduce_prompts():
    m = sm.build_map_messages("[00:01] Me: x", 2, 5)[1]["content"]
    assert "part 2 of 5" in m and "[00:01] Me: x" in m
    r = sm.build_reduce_messages(["## Summary\n- a", "## Summary\n- b"])[1]["content"]
    assert "Notes for part 1" in r and "Notes for part 2" in r and "3 to 5" in r


def test_transcript_rows_are_compact():
    rows = sm.transcript_rows([Line("Me", 65.0, 66.0, " hi "), Line("Them", 3700.0, 3701.0, "yo"),
                               Line("Me", 70.0, 71.0, "  ")])
    assert rows == ["[01:05] Me: hi", "[1:01:40] Them: yo"]


def test_normalize_output_fills_missing_sections_and_strips_preamble():
    raw = ("Here are the notes:\n\n**Summary:**\nWe moved the launch.\n\n"
           "### Decisions\n- Launch on the 11th " + chr(0x2014) + " agreed.\n\n"
           "## Action items\n- None\n")
    out = sm.normalize_output("Meeting X", raw)
    assert out.startswith("# Meeting X\n")
    assert "Here are the notes" not in out
    assert [l for l in out.splitlines() if l.startswith("## ")] == SECTION_HEADERS
    assert "We moved the launch." in out and chr(0x2014) not in out
    assert out.count("\nNone\n") == 3   # key points, action items, open questions


# Summary with a fake Ollama ----------------------------------------------------------

def test_summarize_writes_summary_md(data_dir, fake):
    fake.reply = ("## Summary\nThe launch moved.\n\n## Key points\n- Two crash bugs.\n\n"
                  "## Decisions\n- Launch on the 11th.\n\n## Action items\n- Me: update plan\n\n"
                  "## Open questions\nNone")
    d = _meeting(data_dir)
    progress = []
    out = summarize(d, _cfg(), on_progress=progress.append)
    assert out == d / SUMMARY_FILE
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Meeting 2026-10-29 10:00\n")
    assert [l for l in text.splitlines() if l.startswith("## ")] == SECTION_HEADERS
    assert "- Me: update plan" in text
    assert len(fake.chats) == 1
    body = fake.chats[0]
    assert body["model"] == "llama3:latest" and body["stream"] is False
    assert body["options"]["temperature"] <= 0.3 and body["options"]["num_ctx"] == 8192
    assert "[00:31] Me: Agreed, the 11th works." in body["messages"][1]["content"]
    assert progress[-1] == "Saved summary"


def test_summarize_long_meeting_uses_map_reduce(data_dir, fake, monkeypatch):
    # A tiny budget forces several chunks and a two-level merge of the notes.
    monkeypatch.setattr(sm, "chunk_budget_tokens", lambda: 40)
    fake.reply = "## Summary\n- n"
    d = _meeting(data_dir)
    progress = []
    summarize(d, _cfg(), on_progress=progress.append)
    users = [c["messages"][1]["content"] for c in fake.chats]
    maps = [u for u in users if "of a longer meeting transcript" in u]
    reduces = [u for u in users if "Merge them into a single set of meeting notes" in u]
    assert len(maps) >= 5
    assert len(reduces) >= 2
    assert all("3 to 5" not in u for u in reduces[:-1])   # intermediate merges keep notes form
    assert "3 to 5" in users[-1] and users[-1] == reduces[-1]
    assert any(p.startswith("Summarizing part 1 of") for p in progress)
    text = (d / SUMMARY_FILE).read_text(encoding="utf-8")
    assert [l for l in text.splitlines() if l.startswith("## ")] == SECTION_HEADERS


def test_short_transcript_skips_the_model(data_dir, fake):
    d = _meeting(data_dir, rows=[("Me", 1, "Hello, can you hear me?")])
    text = summarize(d, _cfg()).read_text(encoding="utf-8")
    assert "Not enough was said" in text and fake.chats == []
    empty = data_dir / "meetings" / "2026-10-29_110000"
    empty.mkdir(parents=True)
    assert "Not enough was said" in summarize(empty, _cfg()).read_text(encoding="utf-8")


def test_summary_falls_back_to_transcript_txt(data_dir, fake):
    fake.reply = "## Summary\nok"
    d = data_dir / "meetings" / "2026-10-29_120000"
    d.mkdir(parents=True)
    rows = "\n".join(f"[00:00:{i:02d}] Them: this is sentence number {i} of the talk" for i in range(10))
    (d / TRANSCRIPT_FILE).write_text("Meeting\n\n" + rows + "\n", encoding="utf-8")
    summarize(d, _cfg())
    assert "[00:09] Them: this is sentence number 9" in fake.chats[0]["messages"][1]["content"]


def test_missing_model_names_the_pull_command(data_dir, fake):
    fake.models = ["mistral:latest"]
    with pytest.raises(SummaryError, match=r"ollama pull llama3:latest"):
        summarize(_meeting(data_dir), _cfg())
    fake.models = ["llama3"]   # "llama3" and "llama3:latest" are the same model
    fake.reply = "## Summary\nok"
    summarize(data_dir / "meetings" / "2026-10-29_100000", _cfg())


def test_ollama_down_and_cannot_start(data_dir, fake, monkeypatch):
    fake.up = False
    monkeypatch.setattr(sm, "_start_ollama", lambda: False)
    with pytest.raises(SummaryError, match="Ollama is not running"):
        summarize(_meeting(data_dir), _cfg())


def test_ollama_down_then_started(data_dir, fake, monkeypatch):
    fake.up = False
    fake.reply = "## Summary\nok"
    started = []

    def start():
        started.append(True)
        fake.up = True
        return True

    monkeypatch.setattr(sm, "_start_ollama", start)
    progress = []
    summarize(_meeting(data_dir), _cfg(), on_progress=progress.append)
    assert started == [True] and "Starting Ollama..." in progress


def test_ollama_never_comes_up(data_dir, fake, monkeypatch):
    fake.up = False
    monkeypatch.setattr(sm, "_start_ollama", lambda: True)
    monkeypatch.setattr(sm, "START_WAIT_S", 1)
    with pytest.raises(SummaryError, match="ollama serve"):
        summarize(_meeting(data_dir), _cfg())


# Real Ollama -----------------------------------------------------------------------

def _ollama_ready(cfg) -> bool:
    try:
        names = [m["name"] for m in requests.get(f"{cfg['ollama_url']}/api/tags", timeout=2)
                 .json().get("models", [])]
    except (requests.RequestException, ValueError):
        return False
    return sm._model_present(cfg["summary_model"], names)


@pytest.mark.skipif(not _ollama_ready(config.DEFAULTS), reason="Ollama with llama3 not available")
def test_real_ollama_summary(data_dir):
    d = _meeting(data_dir)
    t0 = time.perf_counter()
    text = summarize(d, _cfg()).read_text(encoding="utf-8")
    elapsed = time.perf_counter() - t0
    print(f"\n--- real summary ({elapsed:.1f} s) ---\n{text}")
    assert [l for l in text.splitlines() if l.startswith("## ")] == SECTION_HEADERS
    low = text.lower()
    assert "11th" in low or "11" in low
    assert "launch" in low
