"""
TenantAwareQdrantCollectionWithSuffix — analog zu
``TenantAwareMilvusCollectionWithSuffix``, aber deutlich schlanker:

- Qdrant kennt keinen Alias-Mechanismus, daher entfaellt der
  Real-Name-/Alias-Switch-Layer.
- Multi-Tenancy laeuft via **Collection-per-Tenant-Naming**:
  ``f"{tenant_prefix}_{base_collection_name}"``. ``tenant_prefix`` wird vom
  Tenant-Context aufgeloest (siehe ``config_utils.get_tenant_aware_collection_name``).
- Optional kann ein expliziter Suffix uebergeben werden (z.B. fuer
  Test-/Sandbox-Collections); er wird an den Tenant-Prefixed-Namen
  angehaengt: ``f"{tenant_prefix}_{base}_{suffix}"``.

Subclass-Beispiel::

    class EpisodicMemoryCollection(TenantAwareQdrantCollectionWithSuffix):
        _COLLECTION_NAME = "v1_episodic_memory"
        _VECTOR_PARAMS = IndexConfig(
            size=1024,
            distance="cosine",
            payload_indexes={"user_id": "keyword", "timestamp": "integer"},
        )

    # Im Tenant-Context "acme":
    coll = EpisodicMemoryCollection()
    coll.name           # -> "acme_v1_episodic_memory"
    coll.ensure_all()   # idempotent

    # Mit explizitem Suffix:
    coll_v2 = EpisodicMemoryCollection(suffix="staging")
    coll_v2.name        # -> "acme_v1_episodic_memory_staging"
"""

import os
from typing import ClassVar, Optional

from core.observation.logger import get_logger
from core.oxm.qdrant.qdrant_collection_base import QdrantCollectionBase
from core.tenants.tenantize.oxm.qdrant.config_utils import (
    get_tenant_aware_collection_name,
)

logger = get_logger(__name__)

# Umgebungsvariable, die einen statischen Collection-Suffix erzwingt,
# z.B. fuer dev/staging-Builds. Wird nur verwendet wenn kein Suffix im
# Konstruktor uebergeben wurde.
_ENV_SUFFIX = "SELF_QDRANT_COLLECTION_NS"


def _resolve_suffix(suffix: Optional[str]) -> str:
    """``suffix`` Argument > Env-Var > leerer String."""
    if suffix is not None:
        return suffix
    return os.getenv(_ENV_SUFFIX, "")


class TenantAwareQdrantCollectionWithSuffix(QdrantCollectionBase):
    """
    Tenant-aware Qdrant collection with optional explicit suffix.

    Differences from the base class:

    - ``name`` property returns the **tenant-prefixed** name (and optionally
      adds the explicit suffix). The base ``_COLLECTION_NAME`` stays the
      logical/business identifier.
    - ``__init__`` accepts an optional ``suffix`` parameter (or reads it from
      the ``SELF_QDRANT_COLLECTION_NS`` environment variable).
    - All other behaviour (collection creation, payload indexes, upsert/
      search/delete) is inherited unchanged from ``QdrantCollectionBase``.
    """

    # Subclasses MAY pin a partitioning strategy here. Currently informational
    # only; future versions may use it to opt into Qdrant's native multi-
    # tenancy via payload partitioning + ``group_id``-based sharding.
    _MULTI_TENANT_STRATEGY: ClassVar[str] = "collection_per_tenant"

    def __init__(self, suffix: Optional[str] = None):
        """
        Args:
            suffix: Optional explicit suffix (e.g., ``"staging"``). If not
                    provided, falls back to ``SELF_QDRANT_COLLECTION_NS``
                    env-var, then to empty string.
        """
        # Reuse the base validation (requires _COLLECTION_NAME + _VECTOR_PARAMS).
        super().__init__()

        self._suffix = _resolve_suffix(suffix)
        # Resolve tenant-aware base name lazily — at __init__ the tenant context
        # is whatever is active when the object is instantiated. If callers need
        # to materialize a collection for a different tenant context, they
        # instantiate within that context.
        tenant_prefixed = get_tenant_aware_collection_name(self._COLLECTION_NAME)
        if self._suffix:
            self._resolved_name = f"{tenant_prefixed}_{self._suffix}"
        else:
            self._resolved_name = tenant_prefixed

        logger.debug(
            "TenantAwareQdrantCollectionWithSuffix resolved name: %s "
            "(base=%s, tenant_prefixed=%s, suffix=%s)",
            self._resolved_name,
            self._COLLECTION_NAME,
            tenant_prefixed,
            self._suffix or "<none>",
        )

    @property
    def name(self) -> str:
        """Tenant-prefixed Qdrant collection name (with optional suffix)."""
        return self._resolved_name

    @property
    def base_name(self) -> str:
        """The original logical ``_COLLECTION_NAME`` without tenant prefix."""
        return self._COLLECTION_NAME  # type: ignore[return-value]

    @property
    def suffix(self) -> str:
        """The explicit suffix, or empty string if none was set."""
        return self._suffix
