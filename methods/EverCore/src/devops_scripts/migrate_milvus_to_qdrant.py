#!/usr/bin/env python3
"""
Re-embed MongoDB source-of-truth into Qdrant via OpenRouter qwen3-embedding-8b.

Standalone CLI — does not rely on EverOS' DI container. Reads connection
config from environment / ``.env`` (loaded via python-dotenv if present):

    OPENROUTER_API_KEY        # required
    OPENROUTER_BASE_URL       # default: https://openrouter.ai/api/v1
    VECTORIZE_MODEL           # default: qwen/qwen3-embedding-8b
    VECTORIZE_DIMENSIONS      # default: 1024 (matches memory_layer/constants.py)
    MONGO_URI                 # default: mongodb://localhost:27017
    QDRANT_HOST               # default: localhost
    QDRANT_PORT               # default: 6333

Workhorse migrates a single (mongo-db, mongo-collection) -> qdrant-collection
pair. Use a shell loop over the 6 EverOS collection-types × N tenants to do
the full sweep (see ``re_embed_all.sh`` next to this file).

Idempotent: existing point ids in the target Qdrant collection are skipped
unless ``--force`` is passed.

Security note: at ``--log-level DEBUG`` PyMongo emits connection events
that include the raw Mongo URI. If your ``MONGO_URI`` carries credentials
(``mongodb://user:pass@host``) avoid DEBUG in shared terminals or pipe the
output through a redactor.

Usage::

    python migrate_milvus_to_qdrant.py \\
        --mongo-db <tenant>_episodic_memsys \\
        --mongo-coll v1_episodic_memories \\
        --qdrant-coll <tenant>_v1_episodic_memory \\
        --text-field episode \\
        --extra-text-fields subject,summary \\
        --timestamp-field timestamp --timestamp-unit ms \\
        --payload-fields user_id,group_id,session_id,participants,sender_ids,type,parent_type,parent_id \\
        --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Standalone CLI: add ``src/`` to sys.path so EverOS-internal modules
# (``core.oxm.qdrant.base_repository``) resolve when this script is
# invoked directly with ``python src/devops_scripts/migrate_milvus_to_qdrant.py``
# (no install / no PYTHONPATH).
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from dotenv import load_dotenv

    _dotenv_path = os.environ.get("EVEROS_ENV_FILE")
    if _dotenv_path:
        load_dotenv(_dotenv_path)
    else:
        load_dotenv()  # picks up ./.env if present
except ImportError:
    pass

from openai import OpenAI
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Stable, namespace-shared with the repository layer so script-side and
# service-side ids agree on the same Mongo->Qdrant translation.
from core.oxm.qdrant.base_repository import mongo_id_to_qdrant_id

logger = logging.getLogger("migrate")


# ============================================================ Configuration


@dataclass(frozen=True)
class Config:
    openrouter_api_key: str
    openrouter_base_url: str
    vectorize_model: str
    vectorize_dimensions: int
    mongo_uri: str
    qdrant_host: str
    qdrant_port: int

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is required (env or .env)")

        return cls(
            openrouter_api_key=api_key,
            openrouter_base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            vectorize_model=os.environ.get(
                "VECTORIZE_MODEL", "qwen/qwen3-embedding-8b"
            ),
            # Default mirrors ``memory_layer/constants.py`` (1024) so a migration
            # run with no ``VECTORIZE_DIMENSIONS`` env produces collections that
            # are immediately usable by the runtime service. Sites running a
            # different dimension (e.g. 4096) MUST set the env var in both
            # places (migration + runtime) — a default mismatch would silently
            # produce dim-incompatible collections at cutover.
            vectorize_dimensions=int(os.environ.get("VECTORIZE_DIMENSIONS", "1024")),
            mongo_uri=os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
            qdrant_host=os.environ.get("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.environ.get("QDRANT_PORT", "6333")),
        )


# ============================================================== Embedding


def embed_batch(
    client: OpenAI,
    model: str,
    dimensions: int,
    texts: List[str],
) -> List[List[float]]:
    """Call OpenRouter ``/embeddings`` for a batch of texts."""
    response = client.embeddings.create(
        model=model,
        input=texts,
        dimensions=dimensions,
    )
    # OpenAI client returns objects sorted by ``index`` in the response.
    return [item.embedding for item in response.data]


# ============================================================ Doc handling


def extract_text(
    doc: Dict[str, Any],
    primary_field: str,
    extra_fields: Tuple[str, ...],
) -> str:
    """
    Concatenate primary + extra text fields into a single embedding input.

    Each non-empty value is joined with newlines. The primary field is
    always first.
    """
    parts: List[str] = []
    primary = doc.get(primary_field)
    if primary:
        parts.append(str(primary))
    for field in extra_fields:
        value = doc.get(field)
        if value:
            parts.append(str(value))
    return "\n".join(parts).strip()


# Heuristic threshold: anything above this is interpreted as epoch
# milliseconds when the target unit is seconds (and the inverse for ms).
# ~year 2065 in seconds, ~year 2001 in milliseconds — gives plenty of head
# room before either branch starts misclassifying real timestamps.
_NUMERIC_TS_HEURISTIC_MS = 3_000_000_000


def _normalize_timestamp_to_epoch(
    value: Any,
    target_unit: str,
    doc_id: Any,
    field_name: str,
) -> Optional[int]:
    """
    Coerce a timestamp value to integer epoch in ``target_unit`` (``"ms"`` or
    ``"s"``). ``datetime`` values are exact; numeric values are coerced
    using the heuristic above (a value larger than ``3e9`` is treated as ms,
    smaller as seconds — covers any realistic post-1970 timestamp).

    Returns ``None`` for unsupported types and emits a warning so the bad
    document surfaces.
    """
    if hasattr(value, "timestamp"):  # datetime / pandas Timestamp
        secs = value.timestamp()
        return int(secs * 1000) if target_unit == "ms" else int(secs)
    if isinstance(value, (int, float)):
        n = float(value)
        # Decide source unit from magnitude.
        source_is_ms = n >= _NUMERIC_TS_HEURISTIC_MS
        if target_unit == "ms":
            return int(n) if source_is_ms else int(n * 1000)
        # target seconds
        return int(n // 1000) if source_is_ms else int(n)
    logger.warning(
        "Skipping timestamp field '%s' with unexpected type %s for doc %s",
        field_name, type(value).__name__, doc_id,
    )
    return None


def build_payload(
    doc: Dict[str, Any],
    payload_fields: Tuple[str, ...],
    timestamp_field: Optional[str],
    timestamp_unit: str,
    primary_text: str,
    extra_text_fields: Tuple[str, ...],
    extra_timestamp_fields: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Project mongo doc fields into a Qdrant payload dict."""
    payload: Dict[str, Any] = {}
    for field in payload_fields:
        if field in doc:
            payload[field] = doc[field]

    # Timestamp normalization to epoch (the unit is collection-dependent).
    if timestamp_field and timestamp_field in doc:
        payload[timestamp_field] = _normalize_timestamp_to_epoch(
            doc[timestamp_field], timestamp_unit, doc.get("_id"), timestamp_field,
        )
        if payload[timestamp_field] is None:
            payload.pop(timestamp_field, None)

    # Apply the same normalization to declared extra time fields (e.g.
    # foresight's ``end_time``). The whitelist is required because a
    # magnitude-based heuristic would also rewrite legitimate non-time
    # numeric fields like ``maturity_score``, ``duration_days``, or
    # ``memcell_count`` whenever they happened to exceed the threshold.
    for field in extra_timestamp_fields:
        if field == timestamp_field or field not in payload:
            continue
        normalized = _normalize_timestamp_to_epoch(
            payload[field], timestamp_unit, doc.get("_id"), field,
        )
        if normalized is not None:
            payload[field] = normalized

    # Persist the text used for the embedding for downstream search-result
    # surfaces (matches the Milvus converter's ``search_content`` payload).
    text_pieces = []
    for field in (primary_text, *extra_text_fields):
        value = doc.get(field)
        if value:
            text_pieces.append(value)
    if text_pieces:
        payload["search_content"] = json.dumps(text_pieces, ensure_ascii=False)

    return payload


# ============================================================== Qdrant ops


def ensure_qdrant_collection(
    client: QdrantClient, name: str, vector_size: int
) -> None:
    """
    Create the target Qdrant collection if it does not exist yet.

    Raises:
        RuntimeError: when a pre-existing collection has a different vector
            size. Migrating into a dim-mismatched collection would only
            surface as opaque "vector size mismatch" errors at upsert time
            (per batch, with no hint at the schema drift cause).
    """
    if client.collection_exists(name):
        existing = client.get_collection(name)
        existing_size = existing.config.params.vectors.size  # type: ignore[union-attr]
        if existing_size != vector_size:
            raise RuntimeError(
                f"Qdrant collection '{name}' exists with vector size "
                f"{existing_size}, but this migration expects {vector_size}. "
                "Aborting before the per-batch dim-mismatch errors. Either "
                "set VECTORIZE_DIMENSIONS to match, or rename/delete the "
                "stale collection."
            )
        logger.info(
            "Qdrant collection '%s' already exists (size=%d) — keeping schema",
            name, existing_size,
        )
        return

    logger.info(
        "Creating Qdrant collection '%s' (size=%d, distance=Cosine, HNSW m=16 ef=200)",
        name, vector_size,
    )
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
            hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=200),
        ),
    )


def filter_existing_ids(
    client: QdrantClient, collection_name: str, ids: List[str]
) -> List[str]:
    """Return the subset of ``ids`` not already present in the collection."""
    if not ids:
        return []
    existing = client.retrieve(
        collection_name=collection_name,
        ids=ids,
        with_payload=False,
        with_vectors=False,
    )
    present = {str(p.id) for p in existing}
    return [i for i in ids if i not in present]


# ============================================================ Orchestration


def migrate(
    config: Config,
    mongo_db: str,
    mongo_coll: str,
    qdrant_coll: str,
    text_field: str,
    extra_text_fields: Tuple[str, ...],
    timestamp_field: Optional[str],
    timestamp_unit: str,
    payload_fields: Tuple[str, ...],
    batch_size: int,
    limit: Optional[int],
    force: bool,
    dry_run: bool,
    extra_timestamp_fields: Tuple[str, ...] = (),
) -> None:
    """Run the migration for one (mongo-db, mongo-collection) pair."""
    logger.info(
        "Migrate %s.%s -> Qdrant '%s' (model=%s, dim=%d, batch=%d, force=%s, dry_run=%s)",
        mongo_db, mongo_coll, qdrant_coll, config.vectorize_model,
        config.vectorize_dimensions, batch_size, force, dry_run,
    )

    mongo = MongoClient(config.mongo_uri)
    # Pre-initialize both clients to ``None`` so the ``finally`` block can
    # safely call ``.close()`` even if construction of ``qdrant`` or ``openai``
    # raises mid-setup. Previously a failing ``QdrantClient(...)`` left
    # ``openai`` unbound and the finally-cleanup raised ``NameError``,
    # masking the original connection error.
    qdrant: Optional[QdrantClient] = None
    openai: Optional[OpenAI] = None
    cursor = None
    try:
        qdrant = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        openai = OpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
        )

        coll = mongo[mongo_db][mongo_coll]
        total_docs = coll.estimated_document_count()
        logger.info("Source has ~%d documents", total_docs)

        if not dry_run:
            ensure_qdrant_collection(qdrant, qdrant_coll, config.vectorize_dimensions)

        # ``no_cursor_timeout=True``: a slow embedding batch (OpenRouter
        # rate-limit, retry) can easily exceed the server-side default cursor
        # idle timeout (10 min), which would surface as ``CursorNotFound``
        # mid-sweep with no progress signal. The cursor is closed in finally.
        cursor = coll.find(no_cursor_timeout=True)
        if limit:
            cursor = cursor.limit(limit)

        processed = 0
        skipped_existing = 0
        skipped_no_text = 0
        upserted = 0
        started = time.time()

        batch_docs: List[Dict[str, Any]] = []

        def flush(batch: List[Dict[str, Any]]) -> Tuple[int, int, int]:
            """Embed + upsert one batch. Returns (upserted, skipped_existing, skipped_no_text)."""
            # Mongo ids are mapped to Qdrant point ids via uuid5; idempotent so
            # the existence-check below works across reruns.
            qdrant_ids = [mongo_id_to_qdrant_id(d["_id"]) for d in batch]
            if force:
                new_ids = qdrant_ids
            else:
                new_ids = (
                    filter_existing_ids(qdrant, qdrant_coll, qdrant_ids)
                    if not dry_run
                    else qdrant_ids
                )
            new_set = set(new_ids)
            new_docs = [
                d for d, qid in zip(batch, qdrant_ids) if qid in new_set
            ]
            # Carry the resolved qdrant id alongside the doc so we don't recompute
            # the uuid5 twice; attach as a temporary key on a shallow copy.
            new_pairs: List[Tuple[Dict[str, Any], str]] = [
                (d, qid) for d, qid in zip(batch, qdrant_ids) if qid in new_set
            ]

            texts: List[str] = []
            kept_pairs: List[Tuple[Dict[str, Any], str]] = []
            for d, qid in new_pairs:
                text = extract_text(d, text_field, extra_text_fields)
                if not text:
                    continue
                texts.append(text)
                kept_pairs.append((d, qid))

            if dry_run:
                return (
                    len(kept_pairs),
                    len(batch) - len(new_docs),
                    len(new_docs) - len(kept_pairs),
                )

            if not texts:
                return (
                    0,
                    len(batch) - len(new_docs),
                    len(new_docs) - len(kept_pairs),
                )

            vectors = embed_batch(
                openai, config.vectorize_model, config.vectorize_dimensions, texts
            )

            points: List[qmodels.PointStruct] = []
            for (d, qid), vec in zip(kept_pairs, vectors):
                payload = build_payload(
                    d, payload_fields, timestamp_field, timestamp_unit,
                    text_field, extra_text_fields,
                    extra_timestamp_fields=extra_timestamp_fields,
                )
                # Keep the original Mongo id in the payload so reverse-lookup
                # from Qdrant -> Mongo is trivial.
                payload["mongo_id"] = str(d["_id"])
                points.append(
                    qmodels.PointStruct(id=qid, vector=vec, payload=payload)
                )

            qdrant.upsert(collection_name=qdrant_coll, points=points, wait=True)
            return (
                len(points),
                len(batch) - len(new_docs),
                len(new_docs) - len(kept_pairs),
            )

        for doc in cursor:
            batch_docs.append(doc)
            if len(batch_docs) >= batch_size:
                u, s_e, s_n = flush(batch_docs)
                upserted += u
                skipped_existing += s_e
                skipped_no_text += s_n
                processed += len(batch_docs)
                logger.info(
                    "Progress: processed=%d upserted=%d skipped_existing=%d skipped_no_text=%d elapsed=%.1fs",
                    processed, upserted, skipped_existing, skipped_no_text,
                    time.time() - started,
                )
                batch_docs = []

        if batch_docs:
            u, s_e, s_n = flush(batch_docs)
            upserted += u
            skipped_existing += s_e
            skipped_no_text += s_n
            processed += len(batch_docs)

        logger.info(
            "DONE: processed=%d upserted=%d skipped_existing=%d skipped_no_text=%d elapsed=%.1fs",
            processed, upserted, skipped_existing, skipped_no_text,
            time.time() - started,
        )
    finally:
        # Close in reverse construction order. Best-effort cleanup: a failing
        # close should not mask a real exception from the body. Each handle is
        # tested for ``None`` because construction may have raised mid-setup.
        try:
            if cursor is not None:
                cursor.close()
        except Exception:  # noqa: BLE001
            logger.debug("cursor.close() raised; ignoring during cleanup", exc_info=True)
        try:
            if openai is not None:
                close_fn = getattr(openai, "close", None)
                if callable(close_fn):
                    close_fn()
        except Exception:  # noqa: BLE001
            logger.debug("openai.close() raised; ignoring during cleanup", exc_info=True)
        try:
            if qdrant is not None:
                qdrant.close()
        except Exception:  # noqa: BLE001
            logger.debug("qdrant.close() raised; ignoring during cleanup", exc_info=True)
        try:
            mongo.close()
        except Exception:  # noqa: BLE001
            logger.debug("mongo.close() raised; ignoring during cleanup", exc_info=True)


# =================================================================== CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-embed MongoDB docs into Qdrant via OpenRouter."
    )
    p.add_argument("--mongo-db", required=True, help="Source Mongo database name")
    p.add_argument("--mongo-coll", required=True, help="Source Mongo collection name")
    p.add_argument("--qdrant-coll", required=True, help="Target Qdrant collection name")
    p.add_argument(
        "--text-field",
        required=True,
        help="Primary text field used for embedding (e.g., episode, task_intent)",
    )
    p.add_argument(
        "--extra-text-fields",
        default="",
        help="Comma-separated secondary text fields appended after the primary",
    )
    p.add_argument(
        "--timestamp-field",
        default="",
        help="Mongo field carrying the timestamp (omit to skip)",
    )
    p.add_argument(
        "--timestamp-unit",
        choices=["ms", "s"],
        default="ms",
        help="Target unit for the timestamp payload value",
    )
    p.add_argument(
        "--payload-fields",
        required=True,
        help="Comma-separated list of fields to project from Mongo into the Qdrant payload",
    )
    p.add_argument(
        "--extra-timestamp-fields",
        default="",
        help=(
            "Comma-separated payload field names that should ALSO be normalized "
            "to epoch (in ``--timestamp-unit``). Use for collections that store "
            "additional time fields beyond ``--timestamp-field`` "
            "(e.g. foresight's ``end_time``)."
        ),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of docs to process (for smoke tests)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-embed and overwrite even if the Qdrant point already exists",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Count what would happen without calling OpenRouter or Qdrant.upsert",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    config = Config.from_env()

    extra_text_fields = tuple(
        f.strip() for f in args.extra_text_fields.split(",") if f.strip()
    )
    payload_fields = tuple(
        f.strip() for f in args.payload_fields.split(",") if f.strip()
    )
    extra_timestamp_fields = tuple(
        f.strip() for f in args.extra_timestamp_fields.split(",") if f.strip()
    )
    timestamp_field = args.timestamp_field.strip() or None

    migrate(
        config=config,
        mongo_db=args.mongo_db,
        mongo_coll=args.mongo_coll,
        qdrant_coll=args.qdrant_coll,
        text_field=args.text_field,
        extra_text_fields=extra_text_fields,
        timestamp_field=timestamp_field,
        timestamp_unit=args.timestamp_unit,
        payload_fields=payload_fields,
        extra_timestamp_fields=extra_timestamp_fields,
        batch_size=args.batch_size,
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
