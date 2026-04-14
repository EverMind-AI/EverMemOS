"""
Task 2: session manifest + message-to-session projector.
"""
import pytest

from evaluation.src.adapters.openclaw_manifest import (
    build_session_manifest,
    project_message_id_to_session_id,
)
from evaluation.src.core.data_models import Conversation, Message


def test_project_message_id_to_session_id_locomo_style():
    assert project_message_id_to_session_id("D0:0") == "S0"
    assert project_message_id_to_session_id("D3:11") == "S3"
    assert project_message_id_to_session_id("D12:0") == "S12"


def test_project_message_id_to_session_id_rejects_unknown_format():
    with pytest.raises(ValueError):
        project_message_id_to_session_id("session_1")
    with pytest.raises(ValueError):
        project_message_id_to_session_id("")
    with pytest.raises(ValueError):
        project_message_id_to_session_id("D1")


def test_build_session_manifest_projects_messages_to_sessions():
    conv = Conversation(
        conversation_id="locomo_0",
        messages=[
            Message("u1", "A", "hello", metadata={"session": "session_0", "dia_id": "D0:0"}),
            Message("u2", "B", "world", metadata={"session": "session_0", "dia_id": "D0:1"}),
            Message("u1", "A", "next", metadata={"session": "session_1", "dia_id": "D1:0"}),
        ],
        metadata={},
    )
    manifest = build_session_manifest(conv, dataset_name="locomo")

    assert manifest["schema_version"] == "openclaw-session-manifest/v1"
    assert manifest["conversation_id"] == "locomo_0"
    assert manifest["dataset_name"] == "locomo"

    assert [s["session_id"] for s in manifest["sessions"]] == ["S0", "S1"]
    assert manifest["sessions"][0]["source_message_ids"] == ["D0:0", "D0:1"]
    assert manifest["sessions"][0]["raw_session_key"] == "session_0"

    assert [m["session_id"] for m in manifest["messages"]] == ["S0", "S0", "S1"]


def test_build_session_manifest_skips_messages_without_dia_id():
    conv = Conversation(
        conversation_id="locomo_0",
        messages=[
            Message("u1", "A", "hello", metadata={"session": "session_0"}),  # no dia_id
            Message("u1", "A", "real", metadata={"session": "session_0", "dia_id": "D0:0"}),
        ],
        metadata={},
    )
    manifest = build_session_manifest(conv, dataset_name="locomo")
    assert len(manifest["messages"]) == 1
    assert manifest["messages"][0]["message_id"] == "D0:0"
