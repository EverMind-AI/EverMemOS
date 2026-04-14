"""
Session-bucketed markdown rendering + LLM-driven flush for OpenClaw.

This is the "faithful" ingest path - closer to what OpenClaw actually does
at runtime than the raw-transcript dump in bench/openclaw_adapter.py.

Two modes controlled by config["openclaw"]["flush_mode"]:

* ``disabled``: dump the raw session transcript as markdown bullets. Matches
  v0.1 / v0.2 behaviour exactly so numbers stay comparable.

* ``shared_llm``: send each session transcript through the **framework-
  side** LLM (same provider used for the answer prompt) with a prompt
  modelled on OpenClaw's ``buildMemoryFlushPlan`` (extensions/memory-core/
  src/flush-plan.ts). This is an APPROXIMATION of OpenClaw's production
  selective-retention behaviour, not a faithful reproduction of it:
    - OpenClaw's real flush runs mid-turn inside the agent runner with
      the agent's own LLM config, triggered by a token-budget heuristic.
    - We do it once per session at ingest time with the benchmark's LLM
      provider, and OpenClaw's own ``compaction.memoryFlush.enabled`` is
      kept OFF so search never triggers a second flush.
  Called ``shared_llm`` to make the divergence visible in config.

Files are written as ``memory/session-<SX>-<YYYY-MM-DD>.md`` so OpenClaw's
FTS scan picks them up, and so ``source_sessions`` projection downstream
can read the session id straight out of the path.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Awaitable, Callable, Optional

from evaluation.src.adapters.openclaw_manifest import project_message_id_to_session_id
from evaluation.src.core.data_models import Conversation, Message


logger = logging.getLogger(__name__)

# Modelled on extensions/memory-core/src/flush-plan.ts. The original prompt
# instructs the agent to save memories to memory/YYYY-MM-DD.md before the
# context is compacted. We mirror the intent - retain decisions / facts /
# contradictions, drop greetings - and ask for a well-formed markdown body
# so OpenClaw's FTS has something to chunk on.
_NATIVE_FLUSH_SYSTEM_PROMPT = (
    "You are OpenClaw's memory compaction agent. Distill the SESSION TRANSCRIPT "
    "into retention-worthy memories before the conversation context is compacted.\n"
    "\n"
    "Keep:\n"
    "- Concrete facts, decisions, opinions, relationships, promises.\n"
    "- Dates, durations, numbers, names, places.\n"
    "- Preferences and constraints that might matter later.\n"
    "- Contradictions (flag them explicitly).\n"
    "\n"
    "Drop:\n"
    "- Greetings, filler, rhetorical questions.\n"
    "- Repeated information already stated verbatim earlier.\n"
    "\n"
    "Output format: a single markdown body, no preface or epilogue. Use short "
    "bullet points under a second-level heading derived from the session "
    "label. Each bullet is a standalone statement that will still be "
    "interpretable months later."
)

_NATIVE_FLUSH_USER_TEMPLATE = (
    "## Session metadata\n"
    "- session_id: {session_id}\n"
    "- date: {session_date}\n"
    "- speakers: {speakers}\n"
    "- message_count: {message_count}\n"
    "\n"
    "## Session transcript\n"
    "{transcript}\n"
    "\n"
    "Produce the distilled memories now."
)


def bucket_conversation_by_session(
    conversation: Conversation,
) -> "OrderedDict[str, list[Message]]":
    """Group messages into sessions preserving first-seen order.

    Messages without ``metadata['dia_id']`` are silently skipped: they cannot
    be projected to a session id and would leak into a bucket without a
    meaningful label.
    """
    buckets: "OrderedDict[str, list[Message]]" = OrderedDict()
    for msg in conversation.messages:
        dia_id = msg.metadata.get("dia_id")
        if not dia_id:
            continue
        try:
            sid = project_message_id_to_session_id(dia_id)
        except ValueError:
            continue
        buckets.setdefault(sid, []).append(msg)
    return buckets


def session_date(messages: list[Message], fallback: str = "1970-01-01") -> str:
    for msg in messages:
        if msg.timestamp is not None:
            return msg.timestamp.strftime("%Y-%m-%d")
    return fallback


def render_session_transcript(messages: list[Message]) -> str:
    """Render a session's messages as a bullet list for the flush prompt."""
    lines = []
    for msg in messages:
        ts = msg.timestamp.strftime("%H:%M") if msg.timestamp is not None else ""
        prefix = f"[{ts}] " if ts else ""
        lines.append(f"- {prefix}**{msg.speaker_name}**: {msg.content}")
    return "\n".join(lines)


def session_markdown_filename(session_id: str, date_str: str) -> str:
    return f"session-{session_id}-{date_str}.md"


def render_raw_session_markdown(session_id: str, messages: list[Message]) -> str:
    """disabled-mode renderer. Matches v0.1 layout but with session header."""
    body = render_session_transcript(messages)
    header = f"# {session_id}\n\n"
    return header + body + "\n"


async def render_flushed_session_markdown(
    session_id: str,
    messages: list[Message],
    llm_generate: Callable[[str, str], Awaitable[str]],
) -> str:
    """shared_llm-mode renderer. Uses the framework LLM to distil bullets.

    llm_generate(system_prompt, user_prompt) is an awaitable that must return
    a markdown body. Callers pass their adapter's LLM provider in.
    """
    speakers = sorted({m.speaker_name for m in messages})
    user_prompt = _NATIVE_FLUSH_USER_TEMPLATE.format(
        session_id=session_id,
        session_date=session_date(messages),
        speakers=", ".join(speakers) or "(unknown)",
        message_count=len(messages),
        transcript=render_session_transcript(messages),
    )
    distilled = await llm_generate(_NATIVE_FLUSH_SYSTEM_PROMPT, user_prompt)
    distilled = distilled.strip() or render_session_transcript(messages)
    return f"# {session_id}\n\n" + distilled + "\n"


async def write_session_files(
    conversation: Conversation,
    memory_dir: Path,
    flush_mode: str,
    llm_generate: Optional[Callable[[str, str], Awaitable[str]]] = None,
) -> list[dict]:
    """Write one markdown file per session and return metadata rows.

    Returned rows (suitable for events.jsonl) contain session_id, path_rel,
    message_count, flush_mode.
    """
    buckets = bucket_conversation_by_session(conversation)
    if not buckets:
        return []

    memory_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for sid, messages in buckets.items():
        date_str = session_date(messages)
        filename = session_markdown_filename(sid, date_str)
        rel = f"memory/{filename}"
        abs_path = memory_dir / filename

        if flush_mode == "shared_llm":
            if llm_generate is None:
                raise ValueError("shared_llm flush requires llm_generate callable")
            body = await render_flushed_session_markdown(sid, messages, llm_generate)
        elif flush_mode == "disabled":
            body = render_raw_session_markdown(sid, messages)
        else:
            raise ValueError(f"unsupported flush_mode: {flush_mode!r}")

        abs_path.write_text(body, encoding="utf-8")
        rows.append(
            {
                "session_id": sid,
                "path_rel": rel,
                "date": date_str,
                "message_count": len(messages),
                "flush_mode": flush_mode,
            }
        )
        logger.debug(
            "ingested session %s -> %s (flush_mode=%s, bytes=%d)",
            sid,
            rel,
            flush_mode,
            len(body.encode("utf-8")),
        )
    return rows


def session_id_from_path(path_rel: str) -> Optional[str]:
    """Recover the session id from a file path returned by OpenClaw search.

    Matches session-<SX>-<date>.md; anything else returns None so callers
    can fall back to other projections.
    """
    if not path_rel:
        return None
    name = Path(path_rel).name
    if not name.startswith("session-"):
        return None
    # session-S3-2023-06-09.md -> parts = ["session", "S3", "2023", "06", "09.md"]
    parts = name.split("-")
    if len(parts) < 3:
        return None
    candidate = parts[1]
    if candidate.startswith("S") and candidate[1:].isdigit():
        return candidate
    return None
