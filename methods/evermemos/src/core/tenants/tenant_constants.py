"""
Tenant Constants

Shared constants for tenant isolation across all storage backends
(MongoDB, Elasticsearch, Milvus).

Resource prefix convention (all modes):

    Prefix   Mode                 Example            Source
    ──────   ──────────────────   ────────────────   ──────────────────────────
    b0001    Base (no tenant)     b0001_memsys       get_base_resource_prefix()
    dev      Single-tenant        dev_memsys         TENANT_SINGLE_TENANT_ID
    t3a7b2c  Multi-tenant excl.   t3a7b2c_memsys     enterprise tenant_id_generator
    s0001    Multi-tenant shared  s0001_memsys       enterprise tenant_id_generator

All separators use underscore (_). No hyphens in resource names.

All resources always have a prefix — no bare names.
"""

import os
from functools import lru_cache

# ============================================================
# Tenant field constants
# ============================================================

# Field name used for logical tenant isolation across all storage systems.
# All interceptors (MongoDB TenantCommandInterceptor, ES TenantAwareAsyncElasticsearch,
# Milvus TenantFieldCollectionProxy) use this constant for consistency.
TENANT_ID_FIELD = "tenant_id"

# Maximum length for tenant_id string values.
# Used by Milvus FieldSchema (VARCHAR requires max_length).
TENANT_ID_MAX_LENGTH = 128

# ============================================================
# Isolation modes
# ============================================================

# - SHARED: Multiple tenants share the same database/index/collection.
#   Query filter injection is REQUIRED to prevent cross-tenant data leakage.
# - EXCLUSIVE: Each tenant has its own database/index/collection.
#   Query filter injection is SKIPPED (physical isolation is sufficient).
#   Write injection is still performed for data consistency and future migration.
ISOLATION_MODE_SHARED = "shared"
ISOLATION_MODE_EXCLUSIVE = "exclusive"

# ============================================================
# Resource prefix: base ("b") — no tenant context
# ============================================================
# Used during multi-tenant startup before any request arrives.
# The version suffix is configurable via TENANT_BASE_RESOURCE_VERSION env var,
# allowing operators to bump the version (b0001 -> b0002) on upgrades so
# new resources are auto-created while old ones are left intact for rollback.
#
# Companion prefixes defined in enterprise (tenant_id_generator.py):
#   "t" — exclusive tenant (t + 10-hex hash of org+space, e.g., "t3a7b2c1d9e")
#   "s" — shared node     (s + 4-digit node id, e.g., "s0001")

BASE_RESOURCE_PREFIX_LETTER = "b"


@lru_cache(maxsize=1)
def get_base_resource_prefix() -> str:
    """
    Get the base resource prefix for resources created without tenant context.

    Format: "b" + version (e.g., "b0001", "b0002").
    Version is read from env TENANT_BASE_RESOURCE_VERSION, defaults to "0001".

    Returns:
        str: e.g., "b0001"
    """
    version = os.getenv("TENANT_BASE_RESOURCE_VERSION", "0001")
    return f"{BASE_RESOURCE_PREFIX_LETTER}{version}"
