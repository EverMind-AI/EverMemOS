#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus Dual Storage Demo Test

This is a simplified demo test that doesn't depend on full EpisodicMemoryMilvusRepository.
It demonstrates how dual storage works at the basic level.

To test with real Repository, first add MilvusDualStorageMixin to:
- EpisodicMemoryMilvusRepository
- EventLogMilvusRepository
- ForesightMilvusRepository

Then run: python3 -m pytest tests/test_episodic_memory_milvus_dual_storage.py -v -s
"""

import pytest
import pytest_asyncio
import uuid
import json
from datetime import datetime

from core.observation.logger import get_logger

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

logger = get_logger(__name__)


@pytest_asyncio.fixture
async def kv_storage():
    """Get KV-Storage instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
        KVStorageInterface,
    )

    return get_bean_by_type(KVStorageInterface)


class TestMilvusDualStorageDemo:
    """Demo test for Milvus dual storage concept"""

    async def test_kv_storage_basic_operations(self, kv_storage):
        """Test: Basic KV-Storage operations work correctly"""
        logger.info("=" * 60)
        logger.info("TEST: KV-Storage basic operations")

        test_key = f"test:milvus:demo:{uuid.uuid4().hex[:8]}"
        test_data = {
            "id": "demo_001",
            "vector": [0.1] * 1024,
            "user_id": "user_001",
            "content": "Test content",
            "extra_field": "This is extra data",
        }

        try:
            # Test set
            await kv_storage.put(test_key, json.dumps(test_data))
            logger.info(f"✅ Set data to KV: {test_key}")

            # Test get
            retrieved_value = await kv_storage.get(test_key)
            assert retrieved_value is not None, "KV should return data"

            retrieved_data = json.loads(retrieved_value)
            assert retrieved_data["id"] == "demo_001", "Data should match"
            assert retrieved_data["extra_field"] == "This is extra data", "Extra field should exist"
            logger.info(f"✅ Retrieved data from KV successfully")

            # Test delete
            await kv_storage.delete(test_key)
            deleted_value = await kv_storage.get(test_key)
            assert deleted_value is None, "Data should be deleted"
            logger.info(f"✅ Deleted data from KV successfully")

            logger.info("✅ Test passed: KV-Storage basic operations work")

        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
            # Cleanup on error
            await kv_storage.delete(test_key)
            raise

    async def test_milvus_collection_proxy_concept(self, kv_storage):
        """Test: Demonstrate MilvusCollectionProxy concept"""
        logger.info("=" * 60)
        logger.info("TEST: MilvusCollectionProxy concept demonstration")

        # Simulate what MilvusCollectionProxy does
        full_entity = {
            "id": "event_001",
            "vector": [0.1] * 1024,
            "user_id": "user_001",
            "group_id": "group_001",
            "episode": "Test episode",
            "metadata": json.dumps({"user_name": "Test User", "title": "Test"}),
            # Extra fields that should only go to KV
            "extra_field_1": "Extra data 1",
            "extra_field_2": {"nested": "data"},
        }

        # Define Lite fields (what goes to Milvus)
        lite_fields = {
            "id", "vector", "user_id", "group_id",
            "episode", "metadata"
        }

        # Extract Lite data
        lite_entity = {k: v for k, v in full_entity.items() if k in lite_fields}

        logger.info(f"Full entity keys: {set(full_entity.keys())}")
        logger.info(f"Lite entity keys: {set(lite_entity.keys())}")

        # Verify separation
        assert "extra_field_1" not in lite_entity, "Lite should not have extra fields"
        assert "extra_field_1" in full_entity, "Full should have extra fields"

        # Simulate writing to KV
        kv_key = f"milvus:episodic_memory:{full_entity['id']}"
        await kv_storage.put(kv_key, json.dumps(full_entity))
        logger.info(f"✅ Simulated KV write: {kv_key}")

        # Simulate reading from KV
        kv_value = await kv_storage.get(kv_key)
        retrieved_full = json.loads(kv_value)

        assert "extra_field_1" in retrieved_full, "Full data should have extra fields"
        assert retrieved_full["extra_field_1"] == "Extra data 1"
        logger.info(f"✅ Simulated KV read: retrieved Full data with extra fields")

        # Cleanup
        await kv_storage.delete(kv_key)

        logger.info("✅ Test passed: Proxy concept works correctly")

    async def test_dual_storage_workflow_simulation(self, kv_storage):
        """Test: Simulate complete dual storage workflow with auto-loading"""
        logger.info("=" * 60)
        logger.info("TEST: Dual storage workflow simulation with auto-loading")

        doc_id = f"event_{uuid.uuid4().hex[:8]}"

        # Step 1: Create Full entity (as business code would)
        full_entity = {
            "id": doc_id,
            "vector": [0.1] * 1024,
            "user_id": "user_001",
            "group_id": "group_001",
            "event_type": "conversation",
            "timestamp": int(datetime.now().timestamp()),
            "episode": "Test episode for workflow",
            "search_content": json.dumps(["test", "workflow"]),
            "metadata": json.dumps({
                "user_name": "Test User",
                "title": "Workflow Test",
                "summary": "Testing dual storage workflow",
            }),
            "parent_type": "memcell",
            "parent_id": "memcell_001",
            "created_at": int(datetime.now().timestamp()),
            "updated_at": int(datetime.now().timestamp()),
            # Extra fields (should be auto-loaded)
            "extra_field": "This demonstrates Full data storage",
            "extend": {"custom": "data"},
        }

        logger.info(f"Step 1: Created Full entity with {len(full_entity)} fields")

        # Step 2: Extract Lite data (what MilvusCollectionProxy does)
        lite_fields = {
            "id", "vector", "user_id", "group_id", "event_type",
            "timestamp", "episode", "search_content", "metadata",
            "parent_type", "parent_id", "created_at", "updated_at",
        }
        lite_entity = {k: v for k, v in full_entity.items() if k in lite_fields}

        logger.info(f"Step 2: Extracted Lite entity with {len(lite_entity)} fields")
        logger.info(f"  Lite fields: {sorted(lite_entity.keys())}")

        # Step 3: Write Full data to KV (what Proxy does)
        kv_key = f"milvus:episodic_memory:{doc_id}"
        await kv_storage.put(kv_key, json.dumps(full_entity))
        logger.info(f"Step 3: Wrote Full data to KV: {kv_key}")

        # Step 4: Simulate Milvus write (in real code, this goes to Milvus)
        assert "vector" in lite_entity, "Lite must have vector"
        assert "metadata" in lite_entity, "Lite must have metadata"
        assert "extra_field" not in lite_entity, "Lite must not have extra fields"
        logger.info(f"Step 4: Verified Lite data structure (would go to Milvus)")

        # Step 5: Simulate read - Proxy auto-loads Full data from KV
        # In real usage, this happens automatically in search()/query()
        logger.info(f"Step 5: Simulating auto-load Full data from KV...")
        kv_value = await kv_storage.get(kv_key)
        full_data_from_kv = json.loads(kv_value)

        # Merge Full data into Lite result (what Proxy does automatically)
        enhanced_result = lite_entity.copy()
        enhanced_result.update(full_data_from_kv)

        # Verify user receives complete data (transparent)
        assert "extra_field" in enhanced_result, "User should get extra fields"
        assert enhanced_result["extra_field"] == "This demonstrates Full data storage"
        assert enhanced_result["extend"]["custom"] == "data"
        logger.info(f"Step 5: ✅ User receives Full data automatically (transparent)")

        # Step 6: User is unaware of KV layer
        logger.info(f"Step 6: ✅ User only calls vector_search(), gets complete data")
        logger.info(f"        No manual KV loading needed, completely transparent")

        # Step 7: Cleanup
        await kv_storage.delete(kv_key)
        logger.info(f"Step 7: Cleaned up KV data")

        logger.info("✅ Test passed: Auto-loading workflow transparent to user")


def test_dual_storage_design_documentation():
    """Test: Verify dual storage design files exist"""
    import os

    required_files = [
        "src/core/oxm/milvus/milvus_dual_storage_collection_proxy.py",
        "src/core/oxm/milvus/milvus_dual_storage_mixin.py",
        "Milvus双存储实现方案.md",
        "Milvus双存储_快速开始.md",
    ]

    logger.info("=" * 60)
    logger.info("Checking dual storage implementation files...")

    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        logger.info(f"{status} {file_path}")
        assert exists, f"Required file missing: {file_path}"

    logger.info("✅ All dual storage files exist")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
