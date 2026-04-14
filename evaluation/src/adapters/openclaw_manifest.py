"""
Session manifest for OpenClaw benchmark runs.

OpenClaw stores memory as markdown files bucketed by session / date, so
retrieval hits cannot be traced back to an individual dataset message.
Instead we collapse LoCoMo-style evidence (e.g. ``D3:11``) down to session
IDs (``S3``), and the manifest built here records the projection so that
later stages can render human-readable provenance.
"""
import re
from collections import OrderedDict


_MESSAGE_ID_PATTERN = re.compile(r"D(\d+):\d+")

SCHEMA_VERSION = "openclaw-session-manifest/v1"


def project_message_id_to_session_id(message_id: str) -> str:
    """Project a LoCoMo-style ``D<s>:<m>`` message id to session id ``S<s>``.

    Raises ValueError for any other shape — callers that care about being
    lenient (e.g. normalize_gold_sessions) do their own pre-filter first.
    """
    match = _MESSAGE_ID_PATTERN.fullmatch(message_id or "")
    if not match:
        raise ValueError(f"Invalid LoCoMo-style message_id: {message_id!r}")
    return f"S{match.group(1)}"


def build_session_manifest(conversation, dataset_name: str) -> dict:
    """Build a session-level manifest for ``conversation``.

    Messages without ``metadata['dia_id']`` are silently skipped — they cannot
    be projected to a session id and would pollute retrieval metrics if
    included. Order of ``sessions`` follows first-appearance order in the
    message stream.
    """
    sessions: "OrderedDict[str, dict]" = OrderedDict()
    messages = []

    for msg in conversation.messages:
        message_id = msg.metadata.get("dia_id")
        if not message_id:
            continue

        session_id = project_message_id_to_session_id(message_id)
        session_entry = sessions.setdefault(
            session_id,
            {
                "session_id": session_id,
                "raw_session_key": msg.metadata.get("session"),
                "source_message_ids": [],
            },
        )
        session_entry["source_message_ids"].append(message_id)

        messages.append(
            {
                "message_id": message_id,
                "session_id": session_id,
                "speaker_id": msg.speaker_id,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "conversation_id": conversation.conversation_id,
        "dataset_name": dataset_name,
        "sessions": list(sessions.values()),
        "messages": messages,
    }
