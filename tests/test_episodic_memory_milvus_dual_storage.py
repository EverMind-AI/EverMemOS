#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Episodic Memory Milvus Repository with Dual Storage

Verify that dual storage works correctly for EpisodicMemoryMilvusRepository:
- Milvus stores Lite data (vector + index fields + metadata)
- KV-Storage stores Full data (complete entity dict)
- Automatic synchronization on write operations
- Optional Full data loading from KV

Test Coverage:
1. insert/create_and_save syncs to KV-Storage
2. vector_search returns Lite data
3. load_full_data_from_kv retrieves complete data
4. delete removes data from both Milvus and KV
5. Verify Milvus stores Lite, KV stores Full
"""

import pytest
import pytest_asyncio
import uuid
import json
from datetime import datetime
from typing import TYPE_CHECKING

from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

if TYPE_CHECKING:
    from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
        EpisodicMemoryMilvusRepository,
    )
    from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
        KVStorageInterface,
    )


@pytest_asyncio.fixture
async def milvus_repo():
    """Get EpisodicMemoryMilvusRepository instance with dual storage enabled"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
        EpisodicMemoryCollection,
    )
    from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
        EpisodicMemoryMilvusRepository,
    )
    from pymilvus.exceptions import SchemaNotReadyException
    from pymilvus import utility

    try:
        # Initialize collection using ensure_all() method (same as app startup)
        collection = EpisodicMemoryCollection()
        print(f"Initializing collection: {collection.name}")

        try:
            # Use ensure_all() method - same as application startup
            collection.ensure_all()
            print(f"✅ Collection '{collection.name}' initialized successfully")
        except SchemaNotReadyException as e:
            # Handle schema mismatch: drop and recreate
            print(f"⚠️  Schema mismatch, recreating collection: {e}")
            from pymilvus import Collection
            try:
                temp_coll = Collection(name=collection.name)
                real_name = temp_coll.name
                if utility.has_collection(real_name):
                    utility.drop_collection(real_name)
                    print(f"✅ Dropped old collection: {real_name}")
            except:
                pass
            # Retry initialization after dropping
            collection.ensure_all()
            print(f"✅ Collection '{collection.name}' recreated and initialized")

        # Get repository from DI container
        repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
        print("✅ Repository obtained from DI container")
        return repo
    except Exception as e:
        # If Milvus is not available in test environment, skip the test
        pytest.skip(f"Milvus not available in test environment: {e}")


@pytest_asyncio.fixture
async def kv_storage():
    """Get KV-Storage instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
        KVStorageInterface,
    )

    return get_bean_by_type(KVStorageInterface)


@pytest.fixture
def test_user_id():
    """Generate unique test user ID"""
    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_group_id():
    """Generate unique test group ID"""
    return f"test_group_{uuid.uuid4().hex[:8]}"


def create_test_entity(
    user_id: str,
    group_id: str,
    with_extra_fields: bool = True,
) -> dict:
    """
    Create test entity for Milvus

    Args:
        user_id: User ID
        group_id: Group ID
        with_extra_fields: Whether to include extra fields (for Full data testing)

    Returns:
        Entity dict with all required fields
    """
    now = get_now_with_timezone()
    entity_id = f"test_event_{uuid.uuid4().hex[:8]}"

    # Lite fields (should be stored in Milvus)
    entity = {
        "id": entity_id,
        "vector": [0.1] * 1024,  # 1024-dim vector
        "user_id": user_id,
        "group_id": group_id,
        "participants": [user_id],
        "event_type": "conversation",
        "timestamp": int(now.timestamp()),
        "episode": f"Test episode for {entity_id}",
        "search_content": json.dumps(["test", "episode"]),
        "metadata": json.dumps({
            "user_name": "Test User",
            "title": "Test Event",
            "summary": "This is a test event",
            "participants": [user_id],
            "keywords": ["test", "episode"],
            "linked_entities": [],
            "subject": "Testing",
        }),
        "parent_type": "memcell",
        "parent_id": f"memcell_{uuid.uuid4().hex[:8]}",
        "created_at": int(now.timestamp()),
        "updated_at": int(now.timestamp()),
    }

    # Extra fields (should only be in KV, not in Milvus)
    if with_extra_fields:
        entity["extra_field_1"] = "This should only be in KV-Storage"
        entity["extra_field_2"] = {"nested": "data", "count": 42}
        entity["extra_large_field"] = "x" * 10000  # 10KB data

    return entity


def get_logger_instance():
    """Helper to get logger"""
    return get_logger(__name__)


class TestEpisodicMemoryMilvusDualStorage:
    """Test EpisodicMemoryMilvusRepository dual storage functionality"""

    async def test_01_create_and_save_syncs_to_kv(
        self, milvus_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: create_and_save_episodic_memory syncs Full data to KV-Storage"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: create_and_save syncs to KV-Storage")

        # Create test entity with extra fields
        entity = create_test_entity(test_user_id, test_group_id, with_extra_fields=True)
        entity_id = entity["id"]

        try:
            # Save to Milvus (should auto-sync to KV)
            result = await milvus_repo.create_and_save_episodic_memory(
                id=entity["id"],
                user_id=entity["user_id"],
                timestamp=datetime.fromtimestamp(entity["timestamp"]),
                episode=entity["episode"],
                search_content=json.loads(entity["search_content"]),
                vector=entity["vector"],
                group_id=entity["group_id"],
                participants=entity["participants"],
                event_type=entity["event_type"],
                metadata=entity["metadata"],
                parent_type=entity["parent_type"],
                parent_id=entity["parent_id"],
            )

            logger.info(f"✅ Created episodic memory: {entity_id}")

            # Verify KV-Storage has Full data
            kv_key = f"milvus:episodic_memory:{entity_id}"
            kv_value = await kv_storage.get(kv_key)

            assert kv_value is not None, "KV should have data after insert"

            full_data = json.loads(kv_value)
            assert full_data["id"] == entity_id, "KV should have correct ID"
            assert full_data["user_id"] == test_user_id, "KV should have user_id"

            logger.info(f"✅ KV-Storage has Full data: {kv_key}")

        finally:
            # Cleanup
            await milvus_repo.delete_by_event_id(entity_id)
            await kv_storage.delete(f"milvus:episodic_memory:{entity_id}")
            logger.info("✅ Test passed: create_and_save syncs to KV-Storage")

    async def test_02_vector_search_returns_lite_data(
        self, milvus_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: vector_search returns Lite data with metadata"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: vector_search returns Lite data")

        entity = create_test_entity(test_user_id, test_group_id, with_extra_fields=False)
        entity_id = entity["id"]

        try:
            # Create test data
            await milvus_repo.create_and_save_episodic_memory(
                id=entity["id"],
                user_id=entity["user_id"],
                timestamp=datetime.fromtimestamp(entity["timestamp"]),
                episode=entity["episode"],
                search_content=json.loads(entity["search_content"]),
                vector=entity["vector"],
                group_id=entity["group_id"],
                participants=entity["participants"],
                event_type=entity["event_type"],
                metadata=entity["metadata"],
                parent_type=entity["parent_type"],
                parent_id=entity["parent_id"],
            )

            logger.info(f"✅ Created test data: {entity_id}")

            # Flush to make data searchable immediately
            await milvus_repo.collection.flush()
            logger.info("✅ Flushed collection")

            # Perform vector search
            query_vector = [0.1] * 1024  # Same as test vector
            results = await milvus_repo.vector_search(
                query_vector=query_vector,
                user_id=test_user_id,
                group_id=test_group_id,
                limit=10,
            )

            # Verify results
            assert len(results) > 0, "Should find at least one result"

            result = results[0]
            assert result["id"] == entity_id, "Should find the test entity"
            assert "metadata" in result, "Result should contain metadata"

            metadata = result["metadata"]
            assert metadata["user_name"] == "Test User", "Metadata should contain user_name"
            assert metadata["title"] == "Test Event", "Metadata should contain title"

            logger.info("✅ vector_search returned Lite data with metadata")

        finally:
            # Cleanup
            await milvus_repo.delete_by_event_id(entity_id)
            await kv_storage.delete(f"milvus:episodic_memory:{entity_id}")
            logger.info("✅ Test passed: vector_search returns Lite data")

    async def test_03_vector_search_auto_loads_full_data(
        self, milvus_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: vector_search automatically loads Full data from KV (transparent to user)"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: vector_search auto-loads Full data")

        # Create entity with extra fields
        entity = create_test_entity(test_user_id, test_group_id, with_extra_fields=True)
        entity_id = entity["id"]

        try:
            # Create test data (Proxy will auto-sync to KV)
            await milvus_repo.create_and_save_episodic_memory(
                id=entity["id"],
                user_id=entity["user_id"],
                timestamp=datetime.fromtimestamp(entity["timestamp"]),
                episode=entity["episode"],
                search_content=json.loads(entity["search_content"]),
                vector=entity["vector"],
                group_id=entity["group_id"],
                participants=entity["participants"],
                event_type=entity["event_type"],
                metadata=entity["metadata"],
                parent_type=entity["parent_type"],
                parent_id=entity["parent_id"],
            )

            # Manually add extra fields to KV (simulating Full data)
            kv_key = f"milvus:episodic_memory:{entity_id}"
            kv_value = await kv_storage.get(kv_key)
            full_data = json.loads(kv_value)
            full_data["extra_field_1"] = "This should be auto-loaded"
            full_data["extra_field_2"] = {"count": 42}
            await kv_storage.put(kv_key, json.dumps(full_data))

            logger.info(f"✅ Created test data with extra fields in KV")

            # Flush to make data searchable immediately
            await milvus_repo.collection.flush()
            logger.info("✅ Flushed collection")

            # Perform vector search - should auto-load Full data
            query_vector = [0.1] * 1024
            results = await milvus_repo.vector_search(
                query_vector=query_vector,
                user_id=test_user_id,
                group_id=test_group_id,
                limit=10,
            )

            # Verify results contain Full data (including extra fields)
            assert len(results) > 0, "Should find results"
            result = results[0]

            # Should have extra fields (auto-loaded from KV)
            assert "extra_field_1" in result, "Should have extra_field_1 (auto-loaded from KV)"
            assert result["extra_field_1"] == "This should be auto-loaded"
            assert "extra_field_2" in result, "Should have extra_field_2 (auto-loaded from KV)"
            assert result["extra_field_2"]["count"] == 42

            logger.info("✅ vector_search auto-loaded Full data from KV (transparent to user)")

        finally:
            # Cleanup
            await milvus_repo.delete_by_event_id(entity_id)
            await kv_storage.delete(kv_key)
            logger.info("✅ Test passed: vector_search auto-loads Full data")

    async def test_04_delete_removes_both_milvus_and_kv(
        self, milvus_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: delete removes data from both Milvus and KV-Storage"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: delete removes both Milvus and KV")

        entity = create_test_entity(test_user_id, test_group_id, with_extra_fields=False)
        entity_id = entity["id"]
        kv_key = f"milvus:episodic_memory:{entity_id}"

        # Create test data
        await milvus_repo.create_and_save_episodic_memory(
            id=entity["id"],
            user_id=entity["user_id"],
            timestamp=datetime.fromtimestamp(entity["timestamp"]),
            episode=entity["episode"],
            search_content=json.loads(entity["search_content"]),
            vector=entity["vector"],
            group_id=entity["group_id"],
            participants=entity["participants"],
            event_type=entity["event_type"],
            metadata=entity["metadata"],
            parent_type=entity["parent_type"],
            parent_id=entity["parent_id"],
        )

        logger.info(f"✅ Created test data: {entity_id}")

        # Flush to make data accessible immediately
        await milvus_repo.collection.flush()
        logger.info("✅ Flushed collection")

        # Verify data exists in both Milvus and KV
        milvus_data = await milvus_repo.get_by_id(entity_id)
        assert milvus_data is not None, "Milvus should have data before delete"

        kv_key = f"milvus:episodic_memory:{entity_id}"
        kv_value = await kv_storage.get(kv_key)
        assert kv_value is not None, "KV should have data before delete"

        logger.info("✅ Data exists in both Milvus and KV")

        # Delete data
        result = await milvus_repo.delete_by_event_id(entity_id)
        assert result is True, "Delete should succeed"

        logger.info(f"✅ Deleted data: {entity_id}")

        # Flush to make delete visible immediately
        await milvus_repo.collection.flush()
        logger.info("✅ Flushed after delete")

        # Verify data removed from Milvus
        milvus_data_after = await milvus_repo.get_by_id(entity_id)
        assert milvus_data_after is None, "Milvus should not have data after delete"

        # Verify data removed from KV
        kv_value_after = await kv_storage.get(kv_key)
        assert kv_value_after is None, "KV should not have data after delete"

        logger.info("✅ Test passed: delete removes both Milvus and KV")

    async def test_05_milvus_lite_vs_kv_full_comparison(
        self, milvus_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: Verify Milvus stores Lite, KV stores Full"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: Milvus Lite vs KV Full comparison")

        # Create entity with extra fields
        entity = create_test_entity(test_user_id, test_group_id, with_extra_fields=True)
        entity_id = entity["id"]
        kv_key = f"milvus:episodic_memory:{entity_id}"

        try:
            # Save entity
            await milvus_repo.create_and_save_episodic_memory(
                id=entity["id"],
                user_id=entity["user_id"],
                timestamp=datetime.fromtimestamp(entity["timestamp"]),
                episode=entity["episode"],
                search_content=json.loads(entity["search_content"]),
                vector=entity["vector"],
                group_id=entity["group_id"],
                participants=entity["participants"],
                event_type=entity["event_type"],
                metadata=entity["metadata"],
                parent_type=entity["parent_type"],
                parent_id=entity["parent_id"],
            )

            logger.info(f"✅ Created entity: {entity_id}")

            # Flush to make data accessible immediately
            await milvus_repo.collection.flush()
            logger.info("✅ Flushed collection")

            # Get data from Milvus
            milvus_data = await milvus_repo.get_by_id(entity_id)
            assert milvus_data is not None, "Milvus should have data"

            # Verify Milvus has Lite fields
            assert "id" in milvus_data, "Milvus should have id"
            assert "user_id" in milvus_data, "Milvus should have user_id"
            assert "episode" in milvus_data, "Milvus should have episode"
            assert "metadata" in milvus_data, "Milvus should have metadata"

            logger.info("✅ Milvus has Lite fields")

            # Get data from KV
            kv_value = await kv_storage.get(kv_key)
            assert kv_value is not None, "KV should have data"

            full_data = json.loads(kv_value)

            # Verify KV has Full data
            assert full_data["id"] == entity_id, "KV should have id"
            assert full_data["user_id"] == test_user_id, "KV should have user_id"

            logger.info("✅ KV has Full data")

            # Verify metadata contains most necessary info
            metadata = json.loads(milvus_data["metadata"])
            assert "user_name" in metadata, "Metadata should have user_name"
            assert "title" in metadata, "Metadata should have title"
            assert "summary" in metadata, "Metadata should have summary"

            logger.info("✅ Metadata contains necessary info")

        finally:
            # Cleanup
            await milvus_repo.delete_by_event_id(entity_id)
            await kv_storage.delete(kv_key)
            logger.info("✅ Test passed: Milvus Lite vs KV Full verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
