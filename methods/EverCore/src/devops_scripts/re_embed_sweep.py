#!/usr/bin/env python3
"""
Sweep wrapper for re-embedding all active Mongo databases into Qdrant.

Iterates the underscore-named Mongo DBs (hyphen DBs are the abandoned
2026-04-25 generation — skipped) × 5 collection types and calls the
workhorse ``migrate`` for each non-empty pair.

``v1_user_profiles`` is intentionally excluded: it needs per-doc splitting
(one Mongo doc → many Qdrant points), which the workhorse does not do.
That migration runs separately (Phase 3.1).

Usage::

    # Full sweep of every active DB × every supported collection
    python re_embed_sweep.py --batch-size 64

    # Smoke: one tenant, one collection, dry-run
    python re_embed_sweep.py --tenant <prefix> --collection episodic_memory --dry-run

    # Per-pair cap (smoke before full run)
    python re_embed_sweep.py --limit-per-pair 5 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Add src/ to sys.path so devops_scripts.migrate_milvus_to_qdrant resolves
# when invoked directly.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from dotenv import load_dotenv

    _p = os.environ.get("EVEROS_ENV_FILE")
    if _p:
        load_dotenv(_p)
    else:
        load_dotenv()
except ImportError:
    pass

from pymongo import MongoClient

from devops_scripts.migrate_milvus_to_qdrant import Config, migrate

logger = logging.getLogger("sweep")


# =============================================================== Spec map


@dataclass(frozen=True)
class CollectionSpec:
    """Per-collection-type config for the sweep."""

    mongo_collection: str
    qdrant_base: str
    text_field: str
    extra_text_fields: Tuple[str, ...] = ()
    timestamp_field: Optional[str] = "timestamp"
    timestamp_unit: str = "ms"
    payload_fields: Tuple[str, ...] = field(default_factory=tuple)
    # Additional payload field names that should be epoch-normalized
    # alongside ``timestamp_field`` (e.g. foresight's ``end_time``).
    extra_timestamp_fields: Tuple[str, ...] = ()


SPECS = {
    "episodic_memory": CollectionSpec(
        mongo_collection="v1_episodic_memories",
        qdrant_base="v1_episodic_memory",
        text_field="episode",
        extra_text_fields=("subject", "summary"),
        timestamp_field="timestamp",
        timestamp_unit="ms",
        payload_fields=(
            "user_id", "group_id", "session_id",
            "participants", "sender_ids", "type",
            "parent_type", "parent_id",
        ),
    ),
    "atomic_fact": CollectionSpec(
        mongo_collection="v1_atomic_fact_records",
        qdrant_base="v1_atomic_fact_record",
        text_field="atomic_fact",
        timestamp_field="timestamp",
        timestamp_unit="ms",
        payload_fields=(
            "user_id", "group_id", "session_id",
            "participants", "sender_ids", "type",
            "parent_type", "parent_id",
            # ``AtomicFactQdrantRepository.vector_search`` surfaces the raw
            # text from ``payload.atomic_fact`` so callers don't need a
            # Mongo round-trip; the converter writes this field, so the
            # sweep must persist it too — otherwise migrated records would
            # come back with ``atomic_fact=None``.
            "atomic_fact",
        ),
    ),
    "foresight": CollectionSpec(
        mongo_collection="v1_foresight_records",
        qdrant_base="v1_foresight_record",
        text_field="content",
        extra_text_fields=("evidence",),
        # Foresight stores start_time / end_time (epoch ms). For the sweep
        # we use start_time as the primary time-axis filter (most common
        # range query semantics). ``end_time`` is normalized via the
        # ``extra_timestamp_fields`` whitelist below so the foresight
        # repository's overlap filter can use a consistent epoch-ms type
        # on both ends. ``duration_days`` is NOT in the whitelist — it is
        # a non-time numeric field that must stay verbatim.
        timestamp_field="start_time",
        timestamp_unit="ms",
        payload_fields=(
            "user_id", "group_id", "session_id",
            "participants", "sender_ids", "type",
            "start_time", "end_time", "duration_days",
            "parent_type", "parent_id",
        ),
        extra_timestamp_fields=("end_time",),
    ),
    "agent_case": CollectionSpec(
        mongo_collection="v1_agent_cases",
        qdrant_base="v1_agent_case",
        text_field="task_intent",
        timestamp_field="timestamp",
        timestamp_unit="s",  # epoch SECONDS (not ms!) — Milvus parity
        payload_fields=(
            "user_id", "group_id", "session_id",
            "parent_type", "parent_id",
        ),
    ),
    "agent_skill": CollectionSpec(
        mongo_collection="v1_agent_skills",
        qdrant_base="v1_agent_skill",
        text_field="name",
        extra_text_fields=("description",),
        timestamp_field=None,  # no time-axis for skills
        timestamp_unit="ms",
        payload_fields=(
            "user_id", "group_id", "cluster_id",
            "name", "description",
            "maturity_score", "confidence",
        ),
    ),
    # user_profile intentionally not included — needs doc splitting (Phase 3.1)
}


# ============================================================ Mongo helpers


def derive_tenant_prefix(mongo_db: str) -> str:
    """
    Strip the trailing ``_memsys`` (or ``memsys``) suffix from a Mongo DB
    name to get the Qdrant collection prefix.

    Examples::

        <prefix>_memsys           -> <prefix>
        <prefix>_<sub>_memsys     -> <prefix>_<sub>
    """
    stripped = mongo_db
    for suffix in ("_memsys", "memsys"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)].rstrip("_")
            break
    return stripped


def list_active_dbs(client: MongoClient) -> List[str]:
    """All non-system DBs whose name has no hyphen (hyphen = abandoned generation)."""
    result = client.admin.command({"listDatabases": 1})
    return sorted(
        d["name"]
        for d in result["databases"]
        if d["name"] not in ("admin", "config", "local")
        and "-" not in d["name"]
    )


def estimated_count(client: MongoClient, db_name: str, coll_name: str) -> int:
    """Cheap ``estimatedDocumentCount``; returns 0 if collection is absent."""
    return client[db_name][coll_name].estimated_document_count()


# =============================================================== Sweep loop


def sweep(
    config: Config,
    spec_keys: List[str],
    tenant_filter: Optional[str],
    batch_size: int,
    limit_per_pair: Optional[int],
    force: bool,
    dry_run: bool,
) -> int:
    """
    Iterate active DBs × selected specs and run ``migrate`` per non-empty pair.

    Returns:
        Number of pairs that failed. Callers (cron, CI) propagate this as a
        non-zero exit code — silent partial-failure sweeps used to be marked
        green by the previous unconditional ``return 0`` in ``main()``.
    """
    # Single shared Mongo client for the discovery / count phase. The
    # workhorse ``migrate()`` opens its own connection inside its try/finally
    # block — that is intentional (each pair is self-contained and survives
    # cleanup independently). Before this consolidation, ``list_active_dbs``
    # and ``estimated_count`` each opened and closed their own client per
    # call, producing N×M connection churn for the discovery scan alone.
    mongo = MongoClient(config.mongo_uri)
    try:
        active_dbs = list_active_dbs(mongo)
        if tenant_filter:
            active_dbs = [d for d in active_dbs if d.startswith(tenant_filter)]

        target_specs = {k: SPECS[k] for k in spec_keys}

        logger.info(
            "Sweep plan: %d active DBs × %d collection types -> up to %d pairs"
            " (dry_run=%s, batch=%d, limit_per_pair=%s, force=%s)",
            len(active_dbs), len(target_specs),
            len(active_dbs) * len(target_specs),
            dry_run, batch_size, limit_per_pair, force,
        )

        overall_start = time.time()
        pairs_run = 0
        pairs_skipped_empty = 0
        pairs_failed = 0

        for db in active_dbs:
            prefix = derive_tenant_prefix(db)
            for spec_name, spec in target_specs.items():
                count = estimated_count(mongo, db, spec.mongo_collection)
                if count == 0:
                    pairs_skipped_empty += 1
                    continue

                qdrant_coll = f"{prefix}_{spec.qdrant_base}"
                logger.info(
                    "==> [%s] %s.%s -> %s (count=%d)",
                    spec_name, db, spec.mongo_collection, qdrant_coll, count,
                )
                try:
                    migrate(
                        config=config,
                        mongo_db=db,
                        mongo_coll=spec.mongo_collection,
                        qdrant_coll=qdrant_coll,
                        text_field=spec.text_field,
                        extra_text_fields=spec.extra_text_fields,
                        timestamp_field=spec.timestamp_field,
                        timestamp_unit=spec.timestamp_unit,
                        payload_fields=spec.payload_fields,
                        extra_timestamp_fields=spec.extra_timestamp_fields,
                        batch_size=batch_size,
                        limit=limit_per_pair,
                        force=force,
                        dry_run=dry_run,
                    )
                    pairs_run += 1
                except Exception as e:
                    logger.exception(
                        "Pair %s.%s -> %s FAILED: %s",
                        db, spec.mongo_collection, qdrant_coll, e,
                    )
                    pairs_failed += 1

        logger.info(
            "SWEEP DONE: pairs_run=%d pairs_skipped_empty=%d pairs_failed=%d elapsed=%.1fs",
            pairs_run, pairs_skipped_empty, pairs_failed,
            time.time() - overall_start,
        )
        return pairs_failed
    finally:
        try:
            mongo.close()
        except Exception:  # noqa: BLE001
            logger.debug("mongo.close() raised during sweep cleanup", exc_info=True)


# =================================================================== CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep wrapper for re-embed migration")
    p.add_argument(
        "--tenant",
        default=None,
        help="Only DBs whose name starts with this prefix",
    )
    p.add_argument(
        "--collection",
        default=None,
        choices=sorted(SPECS),
        help="Only this collection type (default: all 5)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--limit-per-pair",
        type=int,
        default=None,
        help="Cap docs processed per (db, coll) pair (smoke testing)",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
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
    spec_keys = [args.collection] if args.collection else list(SPECS)
    failed = sweep(
        config=config,
        spec_keys=spec_keys,
        tenant_filter=args.tenant,
        batch_size=args.batch_size,
        limit_per_pair=args.limit_per_pair,
        force=args.force,
        dry_run=args.dry_run,
    )
    # Non-zero exit when any pair failed so the surrounding cron / CI run
    # treats the sweep as failed instead of silently green.
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
