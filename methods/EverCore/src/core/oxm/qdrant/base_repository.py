"""
Qdrant Base Repository class — analog ``BaseMilvusRepository``.

Provides common CRUD primitives that all Qdrant-backed repositories inherit.
The repository layer sits between the domain code and ``QdrantCollectionBase``:

- domain code calls ``repo.upsert(point)``, ``repo.find_by_id(id)``, ...
- the repository delegates to the wrapped ``QdrantCollectionBase`` instance,
  adding unified async wrapping, logging, and error handling.

Async wrapping: ``qdrant-client``'s sync API is used (more battle-tested
than ``AsyncQdrantClient`` for Phase 1) and wrapped with ``asyncio.to_thread``
so we keep the same async repository surface as the Milvus counterpart.
"""

import asyncio
import uuid
from abc import ABC
from datetime import datetime, timezone
from typing import Any, Generic, List, Optional, Type, TypeVar

from qdrant_client.http import models as qmodels

from core.observation.logger import get_logger
from core.oxm.qdrant.qdrant_collection_base import QdrantCollectionBase

logger = get_logger(__name__)

T = TypeVar("T", bound=QdrantCollectionBase)


# Stable namespace for Mongo ObjectId -> Qdrant UUID translation.
# Qdrant point ids accept only unsigned integers or RFC-4122 UUIDs;
# Mongo ObjectIds (24 hex chars) are neither. Mapping is via ``uuid5``
# (SHA-1, deterministic) so the same Mongo id always maps to the same
# Qdrant point id — required for idempotent re-embed + lookup by Mongo
# back-reference. NEVER change this namespace without a full data-side
# remigration.
_MONGO_TO_QDRANT_NS = uuid.UUID("ec57c0e3-5e90-4d4a-9c1c-a8b9c7d8e7d6")


def mongo_id_to_qdrant_id(mongo_id: Any) -> str:
    """
    Deterministic UUID5 mapping of any Mongo doc id (ObjectId/str/int) to a
    Qdrant-compatible point id string.

    The mapping is one-way (idempotent), so callers that need the Mongo
    original keep it in the payload (e.g. as ``parent_id``).

    Raises:
        ValueError: when ``mongo_id`` is ``None`` or an empty string. Both
            would map to the same fixed Qdrant id and silently collide with
            real records — usually a sign of an upstream bug.
    """
    if mongo_id is None or (isinstance(mongo_id, str) and not mongo_id):
        raise ValueError(
            "mongo_id_to_qdrant_id requires a non-empty source id; got "
            f"{mongo_id!r}"
        )
    return str(uuid.uuid5(_MONGO_TO_QDRANT_NS, str(mongo_id)))


def compute_effective_threshold(
    radius: Optional[float], score_threshold: float
) -> Optional[float]:
    """
    Two-stage gating: pick the *more permissive* (smaller) positive bound to
    pass to Qdrant server-side. Returns ``None`` when neither bound is
    positive — caller passes that ``None`` to skip server-side filtering and
    relies on the client-side ``point.score < score_threshold`` post-filter.

    Semantics:
        - ``score_threshold = 0.0`` is the parameter default and means
          "no minimum"; treated as unset.
        - ``radius is None`` or ``radius <= -1.0`` means "no radius
          expansion"; treated as unset.
        - With both set positive, return the smaller value so server-side
          recall is the wider net (and the hard caller-facing cut-off is
          still enforced client-side).

    Without this helper, a literal ``min(radius, score_threshold)`` with a
    default ``score_threshold=0.0`` evaluates to ``0.0`` and silently
    disables both server-side and client-side filtering.
    """
    candidates: List[float] = []
    if radius is not None and radius > 0:
        candidates.append(radius)
    if score_threshold > 0:
        candidates.append(score_threshold)
    return min(candidates) if candidates else None


def to_epoch_ms(dt: datetime) -> int:
    """
    Convert a ``datetime`` to epoch milliseconds.

    Naive datetimes (``tzinfo is None``) are interpreted as UTC. Callers that
    operate in a local timezone should attach an explicit tzinfo before
    handing the datetime to repository methods to avoid silent drift.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def to_epoch_s(dt: datetime) -> int:
    """Same as :func:`to_epoch_ms` but in seconds (used by ``agent_case``)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


class BaseQdrantRepository(ABC, Generic[T]):
    """
    Base class for all Qdrant repositories.

    Subclasses set the bound collection model via the generic parameter and
    pass the model class to ``__init__``::

        class EpisodicMemoryRepository(
            BaseQdrantRepository[EpisodicMemoryCollection]
        ):
            def __init__(self):
                super().__init__(EpisodicMemoryCollection)

    Subclasses may add domain-specific finders on top of the CRUD primitives.
    """

    def __init__(self, model: Type[T]):
        self.model = model
        self.model_name = model.__name__

    # ------------------------------------------------------------------ shape

    @property
    def collection(self) -> T:
        """
        Instantiate the bound ``QdrantCollectionBase`` subclass for the
        **current tenant context**.

        Not cached on the repository instance: the model's ``__init__``
        resolves the tenant-prefixed collection name at construction time
        (see ``TenantAwareQdrantCollectionWithSuffix``). With a typical
        DI singleton repository scope, caching the result would lock the
        repository to whichever tenant happened to make the first call,
        which would silently route subsequent tenants' reads and writes to
        the wrong collection.

        The model construction itself is cheap (a tenant-name lookup plus
        the base validation in ``QdrantCollectionBase.__init__``), so the
        per-call cost is negligible compared to the round-trip to Qdrant.
        """
        return self.model()

    def get_model_name(self) -> str:
        return self.model_name

    # =================================================== Basic CRUD (async)

    async def upsert(
        self,
        point: qmodels.PointStruct,
        wait: bool = True,
    ) -> str:
        """
        Insert-or-update a single point.

        Qdrant has no separate ``insert`` semantics — upsert is the
        idempotent primitive. The returned id is taken from the passed
        PointStruct (caller-supplied).

        Note: this returns a ``str`` (the point id) for parity with the
        Milvus repository's ``insert`` method. The underlying Qdrant
        ``UpdateResult`` is intentionally discarded here. Callers that
        need the wire-level ``UpdateResult`` (e.g., to assert
        ``status == completed``) should use ``upsert_batch([point])``.
        """
        try:
            await self.collection.upsert([point], wait)
            logger.debug(
                "Qdrant upsert successful [%s]: %s", self.model_name, point.id
            )
            return str(point.id)
        except Exception as e:
            logger.exception("Qdrant upsert failed [%s]: %s", self.model_name, e)
            raise

    async def upsert_batch(
        self,
        points: List[qmodels.PointStruct],
        wait: bool = True,
    ) -> qmodels.UpdateResult:
        """Batch upsert. ``wait=True`` blocks until the operation is durable."""
        try:
            result = await self.collection.upsert(points, wait)
            logger.debug(
                "Qdrant batch upsert successful [%s]: %d points",
                self.model_name,
                len(points),
            )
            return result
        except Exception as e:
            logger.exception(
                "Qdrant batch upsert failed [%s]: %s", self.model_name, e
            )
            raise

    async def find_by_id(
        self,
        point_id: Any,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Optional[qmodels.Record]:
        """
        Retrieve a single point by id. Returns ``None`` if not found.

        Qdrant accepts both ``int`` and ``str`` (UUID) point ids — pass
        whichever id type was used at upsert time. Operational errors
        (network, auth, malformed id type) are logged and re-raised; only
        the legitimate "not found" case yields ``None``.
        """
        try:
            records = await asyncio.to_thread(
                self.collection.client().retrieve,
                self.collection.name,
                [point_id],
                with_payload,
                with_vectors,
            )
        except Exception as e:
            logger.error(
                "Qdrant find_by_id failed [%s, id=%s]: %s",
                self.model_name,
                point_id,
                e,
            )
            raise
        return records[0] if records else None

    async def find_by_ids(
        self,
        point_ids: List[Any],
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[qmodels.Record]:
        """
        Batch retrieval by ids. Order of result is not guaranteed.

        Returns an empty list when none of the ids exist; raises on any
        operational error so callers can distinguish "all-missing" from a
        retrieval failure.
        """
        try:
            return await asyncio.to_thread(
                self.collection.client().retrieve,
                self.collection.name,
                point_ids,
                with_payload,
                with_vectors,
            )
        except Exception as e:
            logger.error(
                "Qdrant find_by_ids failed [%s, %d ids]: %s",
                self.model_name,
                len(point_ids),
                e,
            )
            raise

    async def delete_by_id(
        self,
        point_id: Any,
        wait: bool = True,
    ) -> bool:
        """
        Delete a single point. Returns ``True`` on a successful round-trip.

        Operational errors are logged and re-raised (consistent with
        ``upsert`` / ``delete_batch``); the ``bool`` return type is kept
        for caller-parity with the Milvus repository.
        """
        try:
            await self.collection.delete([point_id], wait)
        except Exception as e:
            logger.exception(
                "Qdrant delete failed [%s, id=%s]: %s",
                self.model_name,
                point_id,
                e,
            )
            raise
        logger.debug(
            "Qdrant delete successful [%s]: %s", self.model_name, point_id
        )
        return True

    async def delete_batch(
        self,
        point_ids: List[Any],
        wait: bool = True,
    ) -> qmodels.UpdateResult:
        """Batch delete by ids."""
        try:
            result = await self.collection.delete(point_ids, wait)
            logger.debug(
                "Qdrant batch delete successful [%s]: %d ids",
                self.model_name,
                len(point_ids),
            )
            return result
        except Exception as e:
            logger.exception(
                "Qdrant batch delete failed [%s, %d ids]: %s",
                self.model_name,
                len(point_ids),
                e,
            )
            raise

    # ============================================================ Search/Count

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[qmodels.Filter] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[qmodels.ScoredPoint]:
        """ANN search with optional payload-filter."""
        try:
            return await self.collection.search(
                query_vector,
                limit,
                query_filter,
                with_payload,
                with_vectors,
                score_threshold,
                **kwargs,
            )
        except Exception as e:
            logger.exception(
                "Qdrant search failed [%s, limit=%d]: %s",
                self.model_name,
                limit,
                e,
            )
            raise

    async def count(self, exact: bool = True) -> int:
        """Number of points in the underlying collection."""
        try:
            result = await self.collection.count(exact)
        except Exception as e:
            logger.exception("Qdrant count failed [%s]: %s", self.model_name, e)
            raise
        logger.debug(
            "Qdrant count successful [%s]: %d points", self.model_name, result
        )
        return result
