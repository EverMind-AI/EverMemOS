"""Tests for business lifespan shutdown cleanup."""

from types import SimpleNamespace

import pytest

from core.lifespan import business_lifespan
from core.lifespan.business_lifespan import BusinessLifespanProvider


class DummyService:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_shutdown_closes_vectorize_and_rerank_services(monkeypatch):
    vectorize = DummyService()
    rerank = DummyService()

    monkeypatch.setattr(
        business_lifespan, "get_vectorize_service", lambda: vectorize, raising=False
    )
    monkeypatch.setattr(
        business_lifespan, "get_rerank_service", lambda: rerank, raising=False
    )

    provider = BusinessLifespanProvider()
    app = SimpleNamespace(state=SimpleNamespace(graphs={"k": "v"}))

    await provider.shutdown(app)

    assert vectorize.closed is True
    assert rerank.closed is True
    assert not hasattr(app.state, "graphs")
