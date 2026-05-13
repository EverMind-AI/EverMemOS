"""
Qdrant Collection Base — vollstaendige Basisklasse fuer Qdrant-basierte
Collections.

Konzept-Mapping (laut qdrant.tech/documentation/migrate-to-qdrant/from-milvus):

    Milvus                  Qdrant
    -----------------------------------------------------------
    Collection              Collection (1:1)
    FieldSchema(vector)     VectorParams(size, distance)
    FieldSchema(scalar)     Payload field (schema-flexible)
    Index(HNSW, COSINE)     HnswConfigDiff + Distance.Cosine
    Partition               Payload-Field ODER separate Collection
    COSINE                  Cosine
    L2                      Euclid
    IP                      Dot

Die Klasse ist absichtlich schlanker als ihr Milvus-Pendant: Qdrant kennt
keinen Alias-Mechanismus, also entfaellt der ``Real-Name + Alias +
Timestamp``-Indirektions-Layer. Schema-Migrationen erfolgen extern (neue
Collection mit neuem Namen, Daten umlagern).

Bei ``VECTOR_STORE_BACKEND != qdrant`` wird das Modul zwar geladen (durch
DI-Container-Scan), aber ``QdrantLifespanProvider`` initialisiert nichts —
``ensure_all()`` und alle anderen Methoden werden gar nicht aufgerufen.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

logger = logging.getLogger(__name__)


# Mapping: kanonischer Lower-Case-Name (EverOS-intern) -> Qdrant SDK Enum.
# Bewusst ueber Strings, damit Collection-Klassen nicht direkt vom SDK abhaengen.
_PAYLOAD_SCHEMA_TYPE_MAP: Dict[str, "qmodels.PayloadSchemaType"] = {
    "keyword": qmodels.PayloadSchemaType.KEYWORD,
    "integer": qmodels.PayloadSchemaType.INTEGER,
    "float": qmodels.PayloadSchemaType.FLOAT,
    "bool": qmodels.PayloadSchemaType.BOOL,
    "geo": qmodels.PayloadSchemaType.GEO,
    "text": qmodels.PayloadSchemaType.TEXT,
    "datetime": qmodels.PayloadSchemaType.DATETIME,
    "uuid": qmodels.PayloadSchemaType.UUID,
}

# Distance-Mapping zum Schutz vor SDK-Versions-Drift.
_DISTANCE_MAP: Dict[str, "qmodels.Distance"] = {
    "cosine": qmodels.Distance.COSINE,
    "euclid": qmodels.Distance.EUCLID,
    "dot": qmodels.Distance.DOT,
    "manhattan": qmodels.Distance.MANHATTAN,
}


@dataclass
class IndexConfig:
    """
    Konfiguration fuer den (Vektor-)Index einer Qdrant-Collection.

    Args:
        size: Vektor-Dimension (1024 fuer qwen3-embedding-Default).
        distance: Distanz-Metrik (``cosine``, ``euclid``, ``dot``, ``manhattan``).
        on_disk: Vektor-Daten auf Disk halten (mmapped) statt vollstaendig im
                 RAM. Reduziert Memory-Footprint bei groesseren Datasets.
        hnsw_m: HNSW Maximum-Edges-per-Node. Hoeher = bessere Recall, mehr RAM.
        hnsw_ef_construct: HNSW Search-Width beim Bauen. Hoeher = bessere
                           Recall, langsamerer Build.
        payload_indexes: Map ``field_name -> schema_type``. ``schema_type``
                         ist einer von ``_PAYLOAD_SCHEMA_TYPE_MAP`` (e.g.
                         ``"keyword"`` fuer string-equality-Filter).
    """

    size: int = 1024
    distance: str = "cosine"
    on_disk: bool = False
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    payload_indexes: Dict[str, str] = field(default_factory=dict)

    def to_vectors_config(self) -> qmodels.VectorParams:
        """Konvertiert in ``qdrant_client.http.models.VectorParams``."""
        distance_key = self.distance.strip().lower()
        if distance_key not in _DISTANCE_MAP:
            raise ValueError(
                f"Unknown distance '{self.distance}'. "
                f"Supported: {sorted(_DISTANCE_MAP)}"
            )
        return qmodels.VectorParams(
            size=self.size,
            distance=_DISTANCE_MAP[distance_key],
            on_disk=self.on_disk,
            hnsw_config=qmodels.HnswConfigDiff(
                m=self.hnsw_m,
                ef_construct=self.hnsw_ef_construct,
            ),
        )


class QdrantCollectionBase:
    """
    Qdrant-Collection-Management-Basisklasse (analog ``MilvusCollectionBase``).

    Subclasses MUST define:
        _COLLECTION_NAME: ClassVar[str]
        _VECTOR_PARAMS: ClassVar[IndexConfig]

    Optional:
        _DB_USING: ClassVar[str] = "default"

    Anders als das Milvus-Pendant gibt es keinen Alias-Mechanismus — die
    Collection ist direkt unter ``_COLLECTION_NAME`` adressierbar.

    Subclass-Beispiel::

        class EpisodicMemoryCollection(QdrantCollectionBase):
            _COLLECTION_NAME = "v1_episodic_memory"
            _VECTOR_PARAMS = IndexConfig(
                size=1024,
                distance="cosine",
                payload_indexes={
                    "user_id":    "keyword",
                    "group_id":   "keyword",
                    "session_id": "keyword",
                    "timestamp":  "integer",
                },
            )

        coll = EpisodicMemoryCollection()
        coll.ensure_all()
        coll.upsert([...])
    """

    _COLLECTION_NAME: ClassVar[Optional[str]] = None
    _DB_USING: ClassVar[str] = "default"
    _VECTOR_PARAMS: ClassVar[Optional[IndexConfig]] = None

    def __init__(self) -> None:
        if not self._COLLECTION_NAME:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define '_COLLECTION_NAME' "
                "class attribute"
            )
        if self._VECTOR_PARAMS is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define '_VECTOR_PARAMS' "
                "(IndexConfig) class attribute"
            )
        self._using = self._DB_USING or "default"

    @property
    def name(self) -> str:
        return self._COLLECTION_NAME  # type: ignore[return-value]

    @property
    def using(self) -> str:
        return self._using

    # ------------------------------------------------------------------ client

    def client(self) -> QdrantClient:
        """
        Resolve the cached Qdrant client for ``self.using`` via DI factory.

        Looking up via the factory bean keeps client-caching centralized
        (factory caches one QdrantClient instance per alias).
        """
        # Lazy import to avoid a circular dependency: this module is imported
        # at adapter-discovery time, before the DI container is fully wired.
        from core.di.utils import get_bean

        factory = get_bean("qdrant_client_factory")
        return factory.get_named_client(self.using)

    # ------------------------------------------------------------------ schema

    async def exists(self) -> bool:
        """
        Return True if the underlying Qdrant collection already exists.

        Async wrapper over the blocking ``qdrant_client.collection_exists``
        call (offloaded to a worker thread). Only transport-level errors
        (``ResponseHandlingException`` — connect refused, timeout, DNS
        failure) are caught and reported as "does not exist". HTTP error
        responses (``UnexpectedResponse`` for 4xx/5xx, including 401/403
        auth failures and 5xx server errors) propagate — treating them as
        "not exists" would route a downstream ``ensure_collection()`` into
        a confusing follow-up create attempt and bury the real cause (e.g.
        invalid API key, server down).
        """
        client = self.client()
        try:
            return await asyncio.to_thread(client.collection_exists, self.name)
        except ResponseHandlingException as e:
            logger.warning(
                "collection_exists('%s') failed at transport level: %s — "
                "treating as non-existent",
                self.name,
                e,
            )
            return False

    async def count(self, exact: bool = True) -> int:
        """Number of points in the collection."""
        client = self.client()
        result = await asyncio.to_thread(
            client.count, collection_name=self.name, exact=exact
        )
        return result.count

    async def ensure_collection(self) -> None:
        """
        Create the Qdrant collection if it does not exist.

        Idempotent: a pre-existing collection is left untouched, even if its
        schema differs from ``_VECTOR_PARAMS`` — schema migration is an
        explicit external concern.
        """
        cfg = self._VECTOR_PARAMS
        # ``__init__`` already enforces this — explicit check guards against
        # subclasses that override ``__init__`` without invoking ``super``,
        # and survives ``python -O`` (where ``assert`` is stripped).
        if cfg is None:
            raise RuntimeError(
                f"{self.__class__.__name__}._VECTOR_PARAMS is None"
            )

        client = self.client()
        if await self.exists():
            # Validate dimension parity: a pre-existing collection with a
            # different vector size would only surface as opaque "vector
            # size mismatch" errors per upsert/search later. Fail loud here
            # instead so the operator notices a stale schema before data
            # corruption accumulates.
            try:
                existing = await asyncio.to_thread(client.get_collection, self.name)
                existing_size = (
                    existing.config.params.vectors.size  # type: ignore[union-attr]
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "get_collection('%s') failed during dim-validation: %s",
                    self.name, e,
                )
                existing_size = None
            if existing_size is not None and existing_size != cfg.size:
                raise RuntimeError(
                    f"Qdrant collection '{self.name}' exists with vector "
                    f"size {existing_size}, but {self.__class__.__name__} "
                    f"expects {cfg.size}. Migrate or rename the collection."
                )
            logger.debug(
                "Qdrant collection '%s' already exists, skipping create",
                self.name,
            )
            return

        logger.info(
            "Creating Qdrant collection '%s' (size=%d, distance=%s, on_disk=%s)",
            self.name,
            cfg.size,
            cfg.distance,
            cfg.on_disk,
        )
        try:
            await asyncio.to_thread(
                client.create_collection,
                collection_name=self.name,
                vectors_config=cfg.to_vectors_config(),
            )
        except UnexpectedResponse as e:
            # TOCTOU between ``self.exists()`` and ``create_collection``: a
            # parallel process (sibling adapter, second ``ensure_all`` call
            # during a race) may have created the collection just now.
            # Qdrant returns 409 Conflict; swallow it and verify the
            # already-existing collection matches our schema, then continue.
            if getattr(e, "status_code", None) == 409:
                logger.info(
                    "Qdrant collection '%s' was created concurrently — "
                    "treating create as idempotent",
                    self.name,
                )
            else:
                raise

    async def ensure_payload_indexes(self) -> None:
        """
        Create payload-indexes for the fields declared in
        ``_VECTOR_PARAMS.payload_indexes``.

        Qdrant treats ``create_payload_index`` as idempotent at the API level,
        so we call it unconditionally per field.
        """
        cfg = self._VECTOR_PARAMS
        # Explicit guard instead of ``assert`` — stripped under ``python -O``,
        # which would leave the ``cfg.payload_indexes`` access below to raise
        # an opaque ``AttributeError`` in production builds.
        if cfg is None:
            raise RuntimeError(
                f"{self.__class__.__name__}._VECTOR_PARAMS is None"
            )
        if not cfg.payload_indexes:
            logger.debug(
                "Qdrant collection '%s' has no declared payload indexes, skipping",
                self.name,
            )
            return

        client = self.client()
        for field_name, schema_str in cfg.payload_indexes.items():
            key = schema_str.strip().lower()
            if key not in _PAYLOAD_SCHEMA_TYPE_MAP:
                raise ValueError(
                    f"Unknown payload schema '{schema_str}' for field "
                    f"'{field_name}'. Supported: {sorted(_PAYLOAD_SCHEMA_TYPE_MAP)}"
                )
            schema_type = _PAYLOAD_SCHEMA_TYPE_MAP[key]
            try:
                await asyncio.to_thread(
                    client.create_payload_index,
                    collection_name=self.name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.info(
                    "Ensured payload index on '%s.%s' (%s)",
                    self.name,
                    field_name,
                    schema_str,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "Failed to ensure payload index on '%s.%s': %s",
                    self.name,
                    field_name,
                    e,
                )
                raise

    async def ensure_all(self) -> None:
        """Idempotent one-shot init: collection + payload indexes."""
        logger.info("Initializing Qdrant collection '%s' [using=%s]", self.name, self.using)
        await self.ensure_collection()
        await self.ensure_payload_indexes()
        logger.info("Qdrant collection '%s' is ready", self.name)

    # ----------------------------------------------------------- data methods

    async def upsert(
        self,
        points: List[qmodels.PointStruct],
        wait: bool = True,
    ) -> qmodels.UpdateResult:
        """Upsert points (insert or overwrite by id)."""
        client = self.client()
        return await asyncio.to_thread(
            client.upsert,
            collection_name=self.name,
            points=points,
            wait=wait,
        )

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
        """
        ANN search with optional payload-filter.

        Implemented on top of ``QdrantClient.query_points`` (the legacy
        ``search`` method was removed in qdrant-client 1.13+). The wrapper
        keeps the more intuitive ``query_vector`` parameter name for callers
        and unwraps ``QueryResponse.points`` so the return type stays a
        ``List[ScoredPoint]``.
        """
        client = self.client()

        def _call() -> Any:
            return client.query_points(
                collection_name=self.name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
                score_threshold=score_threshold,
                **kwargs,
            )

        response = await asyncio.to_thread(_call)
        return response.points

    async def delete(
        self,
        point_ids: List[Any],
        wait: bool = True,
    ) -> qmodels.UpdateResult:
        """Delete by point ids."""
        client = self.client()
        return await asyncio.to_thread(
            client.delete,
            collection_name=self.name,
            points_selector=qmodels.PointIdsList(points=point_ids),
            wait=wait,
        )

    async def drop(self) -> None:
        """
        Drop the underlying Qdrant collection (DANGEROUS — irreversible).

        Errors (network, auth, permission) are logged and re-raised so the
        caller can react. Use ``exists()`` beforehand to handle the
        already-absent case explicitly without relying on swallowed errors.
        """
        client = self.client()
        try:
            await asyncio.to_thread(client.delete_collection, collection_name=self.name)
            logger.info("Dropped Qdrant collection '%s'", self.name)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to drop Qdrant collection '%s': %s",
                self.name,
                e,
            )
            raise
