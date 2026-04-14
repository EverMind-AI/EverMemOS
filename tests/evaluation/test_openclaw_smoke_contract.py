"""
Task 8: lock the BridgeCommand / BridgeResponse schema on both stub mode
(OPENCLAW_REPO_PATH unset) and dispatcher mode.

Actual smoke validation against a live OpenClaw runtime is documented in
plan Task 8 Step 4 and must be run manually; these tests only cover the
wire contract so regressions surface in CI.
"""
from pathlib import Path

import pytest

from evaluation.src.adapters.openclaw_runtime import arun_bridge, run_bridge


BRIDGE_PATH = (
    Path(__file__).parents[2]
    / "evaluation"
    / "scripts"
    / "openclaw_eval_bridge.mjs"
)


def _stub_env(monkeypatch):
    monkeypatch.delenv("OPENCLAW_REPO_PATH", raising=False)


def test_stub_index_returns_lifecycle_fields(monkeypatch):
    _stub_env(monkeypatch)
    res = run_bridge(
        BRIDGE_PATH,
        {"command": "index", "workspace_dir": "/tmp/ws", "state_dir": "/tmp/s"},
    )
    assert res["ok"] is True
    assert res["command"] == "index"
    assert "flush_epoch" in res
    assert "index_epoch" in res
    assert isinstance(res.get("input_artifacts", []), list)
    assert isinstance(res.get("output_artifacts", []), list)


def test_stub_flush_returns_epochs_and_artifacts(monkeypatch):
    _stub_env(monkeypatch)
    res = run_bridge(
        BRIDGE_PATH,
        {"command": "flush", "workspace_dir": "/tmp/ws", "state_dir": "/tmp/s"},
    )
    assert res["ok"] is True
    assert res["command"] == "flush"
    assert "flush_epoch" in res
    assert "index_epoch" in res


def test_stub_status_returns_settled(monkeypatch):
    _stub_env(monkeypatch)
    res = run_bridge(
        BRIDGE_PATH,
        {"command": "status", "workspace_dir": "/tmp/ws", "state_dir": "/tmp/s"},
    )
    assert res["ok"] is True
    assert res["command"] == "status"
    assert res["settled"] is True
    assert "flush_epoch" in res
    assert "active_artifacts" in res


def test_stub_search_returns_hits_with_artifact_locator(monkeypatch):
    _stub_env(monkeypatch)
    res = run_bridge(
        BRIDGE_PATH,
        {
            "command": "search",
            "workspace_dir": "/tmp/ws",
            "state_dir": "/tmp/s",
            "query": "Where did Caroline move from?",
            "top_k": 3,
        },
    )
    assert res["ok"] is True
    assert res["command"] == "search"
    assert isinstance(res["hits"], list)
    # Every hit the stub returns (even an empty list is OK) must conform to
    # the BridgeSearchHit shape when present.
    for hit in res["hits"]:
        assert "score" in hit
        assert "snippet" in hit
        if "artifact_locator" in hit and hit["artifact_locator"]:
            assert hit["artifact_locator"]["kind"] in {"index_doc", "memory_file_range"}


def test_stub_get_echoes_artifact_locator(monkeypatch, tmp_path):
    _stub_env(monkeypatch)
    locator = {
        "kind": "memory_file_range",
        "path_rel": "native_store/memory/2023-06-09.md",
        "line_start": 10,
        "line_end": 12,
    }
    res = run_bridge(
        BRIDGE_PATH,
        {
            "command": "get",
            "workspace_dir": str(tmp_path),
            "state_dir": "/tmp/s",
            "artifact_locator": locator,
        },
    )
    assert res["ok"] is True
    assert res["command"] == "get"
    assert res["artifact_locator"]["kind"] == locator["kind"]


@pytest.mark.asyncio
async def test_stub_get_via_async_bridge(monkeypatch, tmp_path):
    _stub_env(monkeypatch)
    locator = {
        "kind": "memory_file_range",
        "path_rel": "x.md",
        "line_start": 1,
        "line_end": 2,
    }
    res = await arun_bridge(
        BRIDGE_PATH,
        {
            "command": "get",
            "workspace_dir": str(tmp_path),
            "state_dir": "/tmp/s",
            "artifact_locator": locator,
        },
    )
    assert res["ok"] is True
    assert res["artifact_locator"] == locator


def test_stub_reports_unknown_command():
    with pytest.raises(Exception):
        run_bridge(
            BRIDGE_PATH,
            {"command": "nuke", "workspace_dir": "/tmp", "state_dir": "/tmp"},
        )


def test_repo_path_in_payload_is_honored_over_env(monkeypatch, tmp_path):
    """P0-2: bridge prefers repo_path from the BridgeCommand payload so the
    system YAML value actually drives which launcher we spawn.

    Using a bogus repo_path should flip the bridge into stub mode even if
    OPENCLAW_REPO_PATH points at a valid repo, because the payload wins.
    """
    monkeypatch.setenv("OPENCLAW_REPO_PATH", "/Data3/shutong.shan/openclaw/repo")
    # bogus path the payload ships - repo doesn't exist
    res = run_bridge(
        BRIDGE_PATH,
        {
            "command": "status",
            "repo_path": str(tmp_path / "nope"),
            "workspace_dir": str(tmp_path),
            "state_dir": str(tmp_path),
        },
    )
    # stub mode returns the deterministic happy-path response, not OpenClaw's
    # real payload. `native: true` would appear if we had hit the launcher.
    assert res["command"] == "status"
    assert res.get("native") is not True
