"""
Task 10: session-bucketed markdown writer + OpenClaw-schema resolved config.
"""
import json
from datetime import datetime
from pathlib import Path

import pytest

from evaluation.src.adapters.openclaw_ingestion import (
    bucket_conversation_by_session,
    render_raw_session_markdown,
    session_id_from_path,
    session_markdown_filename,
    write_session_files,
)
from evaluation.src.adapters.openclaw_resolved_config import (
    build_openclaw_resolved_config,
)
from evaluation.src.core.data_models import Conversation, Message


def _conv(*, conv_id: str = "locomo_0"):
    return Conversation(
        conversation_id=conv_id,
        messages=[
            Message("u1", "Alice", "hi there", timestamp=datetime(2023, 6, 9, 10, 0),
                    metadata={"session": "session_0", "dia_id": "D0:0"}),
            Message("u2", "Bob", "hi back", timestamp=datetime(2023, 6, 9, 10, 5),
                    metadata={"session": "session_0", "dia_id": "D0:1"}),
            Message("u1", "Alice", "moved to Seattle",
                    timestamp=datetime(2023, 6, 10, 11, 0),
                    metadata={"session": "session_1", "dia_id": "D1:0"}),
        ],
        metadata={},
    )


def test_bucket_conversation_by_session_preserves_order():
    conv = _conv()
    buckets = bucket_conversation_by_session(conv)
    assert list(buckets.keys()) == ["S0", "S1"]
    assert len(buckets["S0"]) == 2
    assert len(buckets["S1"]) == 1


def test_bucket_skips_messages_without_dia_id():
    conv = Conversation(
        conversation_id="c0",
        messages=[
            Message("u1", "A", "no-id"),
            Message("u1", "A", "has-id", metadata={"dia_id": "D0:0"}),
        ],
        metadata={},
    )
    buckets = bucket_conversation_by_session(conv)
    assert list(buckets.keys()) == ["S0"]
    assert len(buckets["S0"]) == 1


def test_session_markdown_filename_shape():
    assert session_markdown_filename("S3", "2023-06-09") == "session-S3-2023-06-09.md"


def test_session_id_from_path_roundtrips():
    assert session_id_from_path("memory/session-S3-2023-06-09.md") == "S3"
    assert session_id_from_path("memory/session-S12-2023-06-09.md") == "S12"
    assert session_id_from_path("memory/weird.md") is None
    assert session_id_from_path("") is None


def test_render_raw_session_markdown_contains_speaker_and_content():
    conv = _conv()
    buckets = bucket_conversation_by_session(conv)
    body = render_raw_session_markdown("S0", buckets["S0"])
    assert body.startswith("# S0")
    assert "Alice" in body
    assert "hi there" in body


@pytest.mark.asyncio
async def test_write_session_files_disabled_mode(tmp_path):
    conv = _conv()
    memory_dir = tmp_path / "memory"
    rows = await write_session_files(
        conversation=conv,
        memory_dir=memory_dir,
        flush_mode="disabled",
    )
    assert [r["session_id"] for r in rows] == ["S0", "S1"]
    assert [r["path_rel"] for r in rows] == [
        "memory/session-S0-2023-06-09.md",
        "memory/session-S1-2023-06-10.md",
    ]
    body = (memory_dir / "session-S0-2023-06-09.md").read_text()
    assert "Alice" in body and "hi there" in body


@pytest.mark.asyncio
async def test_write_session_files_shared_llm_mode_uses_llm(tmp_path):
    conv = _conv()
    memory_dir = tmp_path / "memory"
    calls = []

    async def fake_llm(system, user):
        calls.append({"system": system, "user": user})
        return "- fact 1\n- fact 2"

    rows = await write_session_files(
        conversation=conv,
        memory_dir=memory_dir,
        flush_mode="shared_llm",
        llm_generate=fake_llm,
    )
    assert len(rows) == 2
    assert len(calls) == 2
    # Both sessions hit the LLM; user prompt includes the raw transcript
    assert "hi there" in calls[0]["user"]
    body = (memory_dir / "session-S0-2023-06-09.md").read_text()
    assert "# S0" in body
    assert "fact 1" in body


@pytest.mark.asyncio
async def test_write_session_files_shared_llm_requires_llm():
    with pytest.raises(ValueError):
        await write_session_files(
            conversation=_conv(),
            memory_dir=Path("/tmp/should-not-exist-xyz"),
            flush_mode="shared_llm",
            llm_generate=None,
        )


def test_build_openclaw_resolved_config_fts_only(tmp_path):
    cfg = build_openclaw_resolved_config(
        workspace_dir=str(tmp_path / "ws"),
        native_store_dir=str(tmp_path / "state"),
        backend_mode="fts_only",
        flush_mode="disabled",
    )
    memory_search = cfg["agents"]["defaults"]["memorySearch"]
    assert memory_search["provider"] == "auto"
    assert memory_search["store"]["vector"]["enabled"] is False
    assert "remote" not in memory_search
    assert cfg["agents"]["defaults"]["compaction"]["memoryFlush"]["enabled"] is False


@pytest.mark.asyncio
async def test_end_to_end_ingest_writes_config_and_calls_bridge(tmp_path, monkeypatch):
    """add() in flush_mode=disabled goes through the real ingest path.

    The bridge is stubbed at module level so no Node invocation happens;
    we verify the adapter writes session files + an OpenClaw-schema config,
    and that the bridge was called once for index + once for status.
    """
    from evaluation.src.adapters import openclaw_adapter as adapter_mod
    from evaluation.src.adapters.openclaw_adapter import OpenClawAdapter

    calls = []

    async def fake_arun_bridge(script, payload, timeout=600.0):
        calls.append(dict(payload))
        cmd = payload.get("command")
        if cmd == "index":
            return {"ok": True, "command": "index", "index_epoch": 42,
                    "flush_epoch": 0, "input_artifacts": [], "output_artifacts": []}
        if cmd == "status":
            return {"ok": True, "command": "status", "settled": True,
                    "flush_epoch": 42, "index_epoch": 42, "active_artifacts": []}
        return {"ok": True, "command": cmd}

    monkeypatch.setattr(adapter_mod, "arun_bridge", fake_arun_bridge)

    adapter = OpenClawAdapter(
        {
            "adapter": "openclaw",
            "dataset_name": "locomo",
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "openclaw": {
                "visibility_mode": "settled",
                "backend_mode": "fts_only",
                "retrieval_route": "search_only",
                "flush_mode": "disabled",
            },
        },
        output_dir=tmp_path,
    )

    handle = await adapter.add([_conv(conv_id="locomo_0")])
    sandbox = handle["conversations"]["locomo_0"]

    # Resolved config file exists and has OpenClaw's native schema.
    cfg_path = Path(sandbox["resolved_config_path"])
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text())
    assert cfg["memory"]["backend"] == "builtin"
    assert cfg["agents"]["defaults"]["workspace"] == sandbox["workspace_dir"]
    assert cfg["agents"]["defaults"]["memorySearch"]["provider"] == "auto"

    # Session files landed under memory/.
    memory_dir = Path(sandbox["memory_dir"])
    md_files = sorted(p.name for p in memory_dir.glob("*.md"))
    assert md_files == [
        "session-S0-2023-06-09.md",
        "session-S1-2023-06-10.md",
    ]

    # Bridge was called exactly once for index and once for status.
    commands = [c.get("command") for c in calls]
    assert commands == ["index", "status"]

    # Handle reflects settled lifecycle state.
    assert sandbox["run_status"] == "ready"
    assert sandbox["visibility_state"] == "settled"
    assert sandbox["last_index_epoch"] == 42


def test_build_openclaw_resolved_config_hybrid_with_embedding(tmp_path):
    cfg = build_openclaw_resolved_config(
        workspace_dir=str(tmp_path / "ws"),
        native_store_dir=str(tmp_path / "state"),
        backend_mode="hybrid",
        flush_mode="shared_llm",
        embedding={
            "provider": "sophnet",
            "model": "text-embeddings",
            "api_key": "SECRET",
            "base_url": "https://example/embeddings",
            "easyllm_id": "ez-123",
            "output_dimensionality": 1024,
        },
    )
    memory_search = cfg["agents"]["defaults"]["memorySearch"]
    assert memory_search["provider"] == "sophnet"
    assert memory_search["store"]["vector"]["enabled"] is True
    assert memory_search["remote"]["apiKey"] == "SECRET"
    assert memory_search["remote"]["easyllmId"] == "ez-123"
    assert memory_search["outputDimensionality"] == 1024
    # Adapter still drives the flush - OpenClaw's own flush stays off.
    assert cfg["agents"]["defaults"]["compaction"]["memoryFlush"]["enabled"] is False
