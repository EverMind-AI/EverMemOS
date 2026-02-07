"""Tests for MemoryController request parameter extraction."""

import json

import pytest

from infra_layer.adapters.input.api.memory.memory_controller import MemoryController


class DummyRequest:
    """Minimal request object for controller parameter parsing tests."""

    def __init__(self, query_params=None, body: bytes = b""):
        self.query_params = query_params or {}
        self._body = body

    async def body(self) -> bytes:
        return self._body


@pytest.mark.asyncio
async def test_collect_request_params_merges_json_body_into_query_params():
    request = DummyRequest(
        query_params={"query": "from_query", "top_k": "10"},
        body=json.dumps({"query": "from_body", "top_k": 25}).encode(),
    )

    params = await MemoryController._collect_request_params(request)

    assert params == {"query": "from_body", "top_k": 25}


@pytest.mark.asyncio
async def test_collect_request_params_ignores_invalid_json_body():
    request = DummyRequest(query_params={"user_id": "u1"}, body=b"{invalid json")

    params = await MemoryController._collect_request_params(request)

    assert params == {"user_id": "u1"}


@pytest.mark.asyncio
async def test_collect_request_params_ignores_non_object_json_body():
    request = DummyRequest(
        query_params={"group_id": "g1"}, body=json.dumps([1, 2]).encode()
    )

    params = await MemoryController._collect_request_params(request)

    assert params == {"group_id": "g1"}
