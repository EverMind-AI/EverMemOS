"""
Task 4: add() + build_lazy_index() + prepare() idempotency.

The real Node bridge is replaced with noop monkey-patches so these tests
exercise sandbox layout / handle persistence / resume semantics without
needing an OpenClaw runtime.
"""
import json
from pathlib import Path

import pytest

from evaluation.src.adapters.openclaw_adapter import OpenClawAdapter
from evaluation.src.core.data_models import Conversation


def _config(tmp_path):
    return {
        "adapter": "openclaw",
        "dataset_name": "locomo",
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "openclaw": {
            "repo_path": "/tmp/openclaw",
            "visibility_mode": "settled",
            "backend_mode": "hybrid",
            "retrieval_route": "search_then_get",
            "flush_mode": "shared_llm",
        },
    }


def _make_adapter(tmp_path, monkeypatch):
    """Adapter with bridge calls stubbed to simulate a successful settle.

    _flush_and_settle_if_needed is the authoritative transition into
    visibility_state='settled'; stub impls must honor that contract or
    _assert_visibility_contract will raise and break the add path.
    """
    adapter = OpenClawAdapter(_config(tmp_path), output_dir=tmp_path)

    async def _ingest_noop(sandbox, conv):
        sandbox["visibility_state"] = "ingested"

    async def _settle_ok(sandbox):
        sandbox["visibility_state"] = "settled"
        sandbox["last_flush_epoch"] = 1

    monkeypatch.setattr(adapter, "_ingest_conversation", _ingest_noop)
    monkeypatch.setattr(adapter, "_flush_and_settle_if_needed", _settle_ok)
    return adapter


@pytest.mark.asyncio
async def test_add_returns_rebuildable_index_handle(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch)

    handle = await adapter.add(
        [Conversation(conversation_id="locomo_0", messages=[], metadata={})]
    )

    assert handle["type"] == "openclaw_sandboxes"
    sandbox = handle["conversations"]["locomo_0"]
    assert sandbox["run_status"] == "ready"
    assert sandbox["visibility_state"] == "settled"

    # Sandbox files exist
    assert Path(sandbox["events_path"]).exists()
    assert (Path(sandbox["metrics_dir"]) / "add_summary.json").exists()

    # Rebuild from disk reproduces the handle
    rebuilt = adapter.build_lazy_index(
        [Conversation(conversation_id="locomo_0", messages=[], metadata={})],
        tmp_path,
    )
    assert rebuilt["type"] == "openclaw_sandboxes"
    assert rebuilt["conversations"]["locomo_0"]["run_status"] == "ready"
    assert rebuilt["conversations"]["locomo_0"]["visibility_state"] == "settled"


@pytest.mark.asyncio
async def test_build_lazy_index_skips_non_ready_handles(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch)
    await adapter.add(
        [
            Conversation(conversation_id="conv_ok", messages=[], metadata={}),
            Conversation(conversation_id="conv_fail", messages=[], metadata={}),
        ]
    )

    # Poison conv_fail's handle
    run_root = adapter._locate_existing_run_root(Path(tmp_path))
    fail_handle_path = run_root / "conversations" / "conv_fail" / "handle.json"
    poisoned = json.loads(fail_handle_path.read_text())
    poisoned["run_status"] = "failed"
    fail_handle_path.write_text(json.dumps(poisoned))

    rebuilt = adapter.build_lazy_index(
        [
            Conversation(conversation_id="conv_ok", messages=[], metadata={}),
            Conversation(conversation_id="conv_fail", messages=[], metadata={}),
        ],
        tmp_path,
    )
    assert "conv_ok" in rebuilt["conversations"]
    assert "conv_fail" not in rebuilt["conversations"]


@pytest.mark.asyncio
async def test_prepare_is_idempotent(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch)
    conv = Conversation(conversation_id="c0", messages=[], metadata={})

    await adapter.prepare([conv])
    first = adapter._prepared
    await adapter.prepare([conv])  # second call must not re-initialize
    assert adapter._prepared is first is True


@pytest.mark.asyncio
async def test_settled_mode_refuses_unsettled_status(tmp_path, monkeypatch):
    """add() must refuse to claim visibility_state=settled if the backend
    reports otherwise. Regression test for codex review P0-1."""
    adapter = OpenClawAdapter(_config(tmp_path), output_dir=tmp_path)

    async def _ingest_noop(sandbox, conv):
        sandbox["visibility_state"] = "ingested"

    async def _settle_fails(sandbox):
        # Simulate the stub-mode bridge where status.settled=true always,
        # but we override here to test the unsettled path explicitly.
        sandbox["visibility_state"] = "indexed"  # stays indexed
        raise RuntimeError("openclaw not settled")

    monkeypatch.setattr(adapter, "_ingest_conversation", _ingest_noop)
    monkeypatch.setattr(adapter, "_flush_and_settle_if_needed", _settle_fails)

    with pytest.raises(RuntimeError, match="not settled"):
        await adapter.add(
            [Conversation(conversation_id="failing_conv", messages=[], metadata={})]
        )

    # Persisted handle must be marked failed so lazy rebuild skips it.
    import json as _json
    run_root = adapter._locate_existing_run_root(Path(tmp_path))
    h = _json.loads(
        (run_root / "conversations" / "failing_conv" / "handle.json").read_text()
    )
    assert h["run_status"] == "failed"
    assert h["visibility_state"] != "settled"


@pytest.mark.asyncio
async def test_eventual_mode_accepts_indexed_state(tmp_path, monkeypatch):
    """visibility_mode='eventual' means build_lazy_index accepts 'indexed'."""
    cfg = _config(tmp_path)
    cfg["openclaw"]["visibility_mode"] = "eventual"
    adapter = OpenClawAdapter(cfg, output_dir=tmp_path)

    async def _ingest_noop(sandbox, conv):
        sandbox["visibility_state"] = "ingested"

    async def _settle_eventual(sandbox):
        sandbox["visibility_state"] = "indexed"  # eventual never promotes

    monkeypatch.setattr(adapter, "_ingest_conversation", _ingest_noop)
    monkeypatch.setattr(adapter, "_flush_and_settle_if_needed", _settle_eventual)

    handle = await adapter.add(
        [Conversation(conversation_id="eventual_conv", messages=[], metadata={})]
    )
    assert handle["conversations"]["eventual_conv"]["visibility_state"] == "indexed"

    rebuilt = adapter.build_lazy_index(
        [Conversation(conversation_id="eventual_conv", messages=[], metadata={})],
        tmp_path,
    )
    assert "eventual_conv" in rebuilt["conversations"]


@pytest.mark.asyncio
async def test_add_writes_session_manifest(tmp_path, monkeypatch):
    from evaluation.src.core.data_models import Message

    adapter = _make_adapter(tmp_path, monkeypatch)
    conv = Conversation(
        conversation_id="locomo_5",
        messages=[
            Message("u1", "A", "hi", metadata={"session": "session_0", "dia_id": "D0:0"}),
        ],
        metadata={},
    )

    handle = await adapter.add([conv])
    manifest_path = Path(handle["conversations"]["locomo_5"]["session_manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "openclaw-session-manifest/v1"
    assert manifest["sessions"][0]["session_id"] == "S0"
