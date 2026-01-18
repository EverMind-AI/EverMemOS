#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete CRUD Test for ConversationMetaRawRepository with KV-Storage

This test file comprehensively tests all CRUD methods in ConversationMetaRawRepository
with the dual MongoDB + KV-Storage pattern. Each test follows the pattern:
1. Create test data
2. Read/Query test data
3. Verify data consistency between MongoDB and KV-Storage
4. Verify data integrity (inserted == retrieved)

Methods tested:
- get_by_group_id
- list_by_scene
- create_conversation_meta
- update_by_group_id
- upsert_by_group_id
- delete_by_group_id
"""

import asyncio
import pytest
import pytest_asyncio
import uuid
from typing import Dict, Any, TYPE_CHECKING

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

# Delay imports to avoid loading beanie at module level
if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
        ConversationMeta,
        UserDetailModel,
    )
    from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
        ConversationMetaRawRepository,
    )
    from infra_layer.adapters.out.persistence.kv_storage import KVStorageInterface


# ==================== Test Fixtures ====================


@pytest_asyncio.fixture
async def repository():
    """Get ConversationMetaRawRepository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
        ConversationMetaRawRepository,
    )
    repo = get_bean_by_type(ConversationMetaRawRepository)
    yield repo


@pytest_asyncio.fixture
async def kv_storage():
    """Get KV-Storage instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.kv_storage import KVStorageInterface
    kv = get_bean_by_type(KVStorageInterface)
    yield kv


@pytest.fixture
def test_group_id():
    """Generate unique test group ID"""
    return f"test_group_{uuid.uuid4().hex[:8]}"


# ==================== Test Helpers ====================


def create_test_conversation_meta(
    group_id: str,
    scene: str = "assistant",
    name: str = "Test Conversation",
    description: str = "Test conversation description",
    version: str = "1.0.0",
):
    """Helper function to create a test ConversationMeta with all fields"""
    from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
        ConversationMeta,
        UserDetailModel,
    )
    from common_utils.datetime_utils import get_now_with_timezone

    now = get_now_with_timezone()

    return ConversationMeta(
        version=version,
        scene=scene,
        scene_desc={"bot_ids": ["bot_001", "bot_002"]},
        name=name,
        description=description,
        group_id=group_id,
        conversation_created_at=now.isoformat(),
        default_timezone="UTC",
        user_details={
            "user_001": UserDetailModel(
                full_name="Test User",
                role="User",
                extra={"age": 30, "location": "Test City"},
            ),
            "robot_001": UserDetailModel(
                full_name="AI Assistant",
                role="Assistant",
                extra={"type": "assistant"},
            ),
        },
        tags=["test", "conversation", "metadata"],
    )


def assert_conversation_meta_equal(cm1, cm2, check_id: bool = True):
    """Assert two ConversationMeta objects are equal"""
    if check_id:
        assert str(cm1.id) == str(cm2.id), "IDs don't match"

    assert cm1.version == cm2.version, "version doesn't match"
    assert cm1.scene == cm2.scene, "scene doesn't match"
    assert cm1.name == cm2.name, "name doesn't match"
    assert cm1.description == cm2.description, "description doesn't match"
    assert cm1.group_id == cm2.group_id, "group_id doesn't match"
    assert cm1.conversation_created_at == cm2.conversation_created_at, "conversation_created_at doesn't match"
    assert cm1.default_timezone == cm2.default_timezone, "default_timezone doesn't match"
    assert set(cm1.tags or []) == set(cm2.tags or []), "tags don't match"

    # Check user_details
    assert set(cm1.user_details.keys()) == set(cm2.user_details.keys()), "user_details keys don't match"
    for key in cm1.user_details:
        assert cm1.user_details[key].full_name == cm2.user_details[key].full_name
        assert cm1.user_details[key].role == cm2.user_details[key].role


async def verify_kv_storage(repository, conversation_meta_id: str) -> bool:
    """Verify data exists in KV-Storage"""
    from core.observation.logger import get_logger

    logger = get_logger(__name__)

    kv_storage = repository._dual_storage.get_kv_storage()
    if not kv_storage:
        logger.warning("KV-Storage not available")
        return False

    kv_json = await kv_storage.get(key=conversation_meta_id)
    return kv_json is not None


# ==================== Test Cases ====================


def get_logger():
    """Helper to get logger instance"""
    from core.observation.logger import get_logger as _get_logger

    return _get_logger(__name__)


class TestBasicCRUD:
    """Test basic CRUD operations: Create, Read, Delete"""

    async def test_01_create_and_get_by_group_id(self, repository, test_group_id):
        """
        Test: create_conversation_meta + get_by_group_id
        Flow: Create a ConversationMeta -> Read it back -> Verify data matches
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: create_conversation_meta + get_by_group_id")

        # 1. Create test ConversationMeta
        original = create_test_conversation_meta(
            group_id=test_group_id,
            name="Test conversation for get_by_group_id",
            description="This is a test conversation meta",
        )

        # 2. Create in repository
        created = await repository.create_conversation_meta(original)
        assert created is not None, "Failed to create ConversationMeta"
        assert created.id is not None, "Created ConversationMeta should have ID"

        conversation_meta_id = str(created.id)
        logger.info(f"✅ Created ConversationMeta with ID: {conversation_meta_id}")

        # 3. Verify KV-Storage
        kv_exists = await verify_kv_storage(repository, conversation_meta_id)
        logger.info(f"KV-Storage: {'✅ Exists' if kv_exists else '⚠️  Not found'}")

        # 4. Read back using get_by_group_id
        retrieved = await repository.get_by_group_id(test_group_id)
        assert retrieved is not None, f"Failed to retrieve ConversationMeta by group_id: {test_group_id}"
        logger.info(f"✅ Retrieved ConversationMeta by group_id: {test_group_id}")

        # 5. Verify data integrity
        assert_conversation_meta_equal(created, retrieved)
        logger.info("✅ Data integrity verified: created == retrieved")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)

    async def test_02_upsert_insert(self, repository, test_group_id):
        """
        Test: upsert_by_group_id (insert mode)
        Flow: Upsert new ConversationMeta -> Verify it was created
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: upsert_by_group_id (insert)")

        # 1. Prepare conversation data
        conversation_data = {
            "version": "1.0.0",
            "scene": "companion",
            "scene_desc": {"bot_ids": ["bot_003"]},
            "name": "Upsert Test Conversation",
            "description": "Test upsert insert mode",
            "conversation_created_at": "2025-01-01T00:00:00Z",
            "default_timezone": "UTC",
            "user_details": {},
            "tags": ["upsert", "insert"],
        }

        # 2. Upsert (should insert)
        result = await repository.upsert_by_group_id(test_group_id, conversation_data)
        assert result is not None, "Upsert failed"
        assert result.id is not None, "Upserted ConversationMeta should have ID"

        conversation_meta_id = str(result.id)
        logger.info(f"✅ Upserted (inserted) ConversationMeta with ID: {conversation_meta_id}")

        # 3. Verify KV-Storage
        kv_exists = await verify_kv_storage(repository, conversation_meta_id)
        logger.info(f"KV-Storage: {'✅ Exists' if kv_exists else '⚠️  Not found'}")

        # 4. Verify data
        assert result.group_id == test_group_id
        assert result.name == "Upsert Test Conversation"
        assert result.scene == "companion"
        logger.info("✅ Upsert insert data verified")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)

    async def test_03_upsert_update(self, repository, test_group_id):
        """
        Test: upsert_by_group_id (update mode)
        Flow: Create ConversationMeta -> Upsert to update -> Verify update worked
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: upsert_by_group_id (update)")

        # 1. Create initial ConversationMeta
        original = create_test_conversation_meta(
            group_id=test_group_id,
            name="Original Name",
            description="Original Description",
        )
        created = await repository.create_conversation_meta(original)
        assert created is not None
        logger.info(f"✅ Created initial ConversationMeta: {created.name}")

        # 2. Upsert with updated data (should update)
        update_data = {
            "version": "2.0.0",
            "scene": "assistant",
            "scene_desc": {"bot_ids": ["bot_004", "bot_005"]},
            "name": "Updated Name",
            "description": "Updated Description",
            "conversation_created_at": "2025-02-01T00:00:00Z",
            "default_timezone": "UTC",
            "user_details": {},
            "tags": ["upsert", "update"],
        }

        result = await repository.upsert_by_group_id(test_group_id, update_data)
        assert result is not None, "Upsert update failed"
        logger.info(f"✅ Upserted (updated) ConversationMeta: {result.name}")

        # 3. Verify update
        assert result.name == "Updated Name"
        assert result.description == "Updated Description"
        assert result.version == "2.0.0"
        assert str(result.id) == str(created.id), "ID should remain the same"
        logger.info("✅ Upsert update data verified")

        # 4. Verify KV-Storage
        kv_exists = await verify_kv_storage(repository, str(result.id))
        logger.info(f"KV-Storage: {'✅ Exists' if kv_exists else '⚠️  Not found'}")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)

    async def test_04_update_by_group_id(self, repository, test_group_id):
        """
        Test: update_by_group_id
        Flow: Create -> Update -> Verify update
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: update_by_group_id")

        # 1. Create initial ConversationMeta
        original = create_test_conversation_meta(
            group_id=test_group_id,
            name="Original Name",
        )
        created = await repository.create_conversation_meta(original)
        assert created is not None
        logger.info(f"✅ Created ConversationMeta: {created.name}")

        # 2. Update
        update_data = {
            "name": "Updated via update_by_group_id",
            "description": "Updated description",
            "tags": ["updated", "test"],
        }
        updated = await repository.update_by_group_id(test_group_id, update_data)
        assert updated is not None, "Update failed"
        logger.info(f"✅ Updated ConversationMeta: {updated.name}")

        # 3. Verify update
        assert updated.name == "Updated via update_by_group_id"
        assert updated.description == "Updated description"
        assert "updated" in updated.tags
        assert str(updated.id) == str(created.id), "ID should remain the same"
        logger.info("✅ Update data verified")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)

    async def test_05_delete_by_group_id(self, repository, test_group_id):
        """
        Test: delete_by_group_id
        Flow: Create -> Delete -> Verify deletion
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_group_id")

        # 1. Create ConversationMeta
        original = create_test_conversation_meta(
            group_id=test_group_id,
            name="To be deleted",
        )
        created = await repository.create_conversation_meta(original)
        assert created is not None
        conversation_meta_id = str(created.id)
        logger.info(f"✅ Created ConversationMeta with ID: {conversation_meta_id}")

        # 2. Verify exists before deletion
        before_delete = await repository.get_by_group_id(test_group_id)
        assert before_delete is not None, "Should exist before deletion"

        # 3. Delete
        delete_result = await repository.delete_by_group_id(test_group_id)
        assert delete_result is True, "Delete should return True"
        logger.info(f"✅ Deleted ConversationMeta by group_id: {test_group_id}")

        # 4. Verify deletion
        after_delete = await repository.get_by_group_id(test_group_id)
        assert after_delete is None, "Should not exist after deletion"
        logger.info("✅ Verified deletion from MongoDB")

        # 5. Verify KV-Storage deletion
        kv_exists = await verify_kv_storage(repository, conversation_meta_id)
        assert kv_exists is False, "Should not exist in KV-Storage after deletion"
        logger.info("✅ Verified deletion from KV-Storage")


class TestSceneQuery:
    """Test scene-based queries"""

    async def test_06_list_by_scene(self, repository):
        """
        Test: list_by_scene
        Flow: Create multiple ConversationMeta with different scenes -> Query by scene
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: list_by_scene")

        # 1. Create test data with different scenes
        test_data = [
            (f"group_assistant_{uuid.uuid4().hex[:8]}", "assistant", "Assistant Conversation 1"),
            (f"group_assistant_{uuid.uuid4().hex[:8]}", "assistant", "Assistant Conversation 2"),
            (f"group_companion_{uuid.uuid4().hex[:8]}", "companion", "Companion Conversation 1"),
        ]

        created_ids = []
        for group_id, scene, name in test_data:
            meta = create_test_conversation_meta(
                group_id=group_id,
                scene=scene,
                name=name,
            )
            created = await repository.create_conversation_meta(meta)
            assert created is not None
            created_ids.append((group_id, str(created.id)))
            logger.info(f"✅ Created {scene} conversation: {name}")

        # 2. Query by scene: assistant
        assistant_list = await repository.list_by_scene("assistant")
        logger.info(f"✅ Retrieved {len(assistant_list)} assistant conversations")

        # Should have at least 2 assistant conversations
        assistant_groups = [cm.group_id for cm in assistant_list]
        assert any(gid.startswith("group_assistant_") for gid in assistant_groups), "Should find assistant conversations"

        # 3. Query by scene: companion
        companion_list = await repository.list_by_scene("companion")
        logger.info(f"✅ Retrieved {len(companion_list)} companion conversations")

        # Should have at least 1 companion conversation
        companion_groups = [cm.group_id for cm in companion_list]
        assert any(gid.startswith("group_companion_") for gid in companion_groups), "Should find companion conversations"

        # 4. Verify all retrieved objects are complete (not Lite)
        for cm in assistant_list + companion_list:
            assert cm.name is not None, "Should have full data"
            assert cm.user_details is not None, "Should have user_details"
            logger.debug(f"Verified full data for group_id: {cm.group_id}")

        # Cleanup
        for group_id, _ in created_ids:
            await repository.delete_by_group_id(group_id)


class TestEdgeCases:
    """Test edge cases and error handling"""

    async def test_07_get_nonexistent(self, repository):
        """
        Test: Get non-existent ConversationMeta
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_group_id (non-existent)")

        result = await repository.get_by_group_id("nonexistent_group_id")
        assert result is None, "Should return None for non-existent group_id"
        logger.info("✅ Correctly returned None for non-existent group_id")

    async def test_08_delete_nonexistent(self, repository):
        """
        Test: Delete non-existent ConversationMeta
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_group_id (non-existent)")

        result = await repository.delete_by_group_id("nonexistent_group_id")
        assert result is False, "Should return False for non-existent group_id"
        logger.info("✅ Correctly returned False for non-existent group_id")

    async def test_09_update_nonexistent(self, repository):
        """
        Test: Update non-existent ConversationMeta
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: update_by_group_id (non-existent)")

        update_data = {"name": "Should not work"}
        result = await repository.update_by_group_id("nonexistent_group_id", update_data)
        assert result is None, "Should return None for non-existent group_id"
        logger.info("✅ Correctly returned None for non-existent group_id")

    async def test_10_invalid_scene(self, repository, test_group_id):
        """
        Test: Create ConversationMeta with invalid scene
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: create_conversation_meta (invalid scene)")

        meta = create_test_conversation_meta(
            group_id=test_group_id,
            scene="invalid_scene",  # Invalid scene
        )
        result = await repository.create_conversation_meta(meta)
        assert result is None, "Should return None for invalid scene"
        logger.info("✅ Correctly rejected invalid scene")


class TestDualStorage:
    """Test Dual Storage consistency between MongoDB and KV-Storage"""

    async def test_11_dual_storage_consistency(self, repository, kv_storage, test_group_id):
        """
        Test: Verify MongoDB Lite and KV-Storage consistency
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Dual Storage consistency")

        # 1. Create ConversationMeta
        from infra_layer.adapters.out.persistence.document.memory.conversation_meta_lite import (
            ConversationMetaLite,
        )
        original = create_test_conversation_meta(
            group_id=test_group_id,
            name="Dual Storage Test",
            description="Test dual storage consistency",
        )
        created = await repository.create_conversation_meta(original)
        assert created is not None
        conversation_meta_id = str(created.id)
        logger.info(f"✅ Created ConversationMeta with ID: {conversation_meta_id}")

        # 2. Verify MongoDB Lite
        lite = await ConversationMetaLite.find_one({"group_id": test_group_id})
        assert lite is not None, "Should exist in MongoDB as Lite"
        assert lite.group_id == test_group_id
        assert lite.scene == original.scene
        logger.info("✅ Verified MongoDB Lite record")

        # 3. Verify KV-Storage
        kv_json = await kv_storage.get(conversation_meta_id)
        assert kv_json is not None, "Should exist in KV-Storage"

        from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
            ConversationMeta,
        )
        full_from_kv = ConversationMeta.model_validate_json(kv_json)
        assert full_from_kv.group_id == test_group_id
        assert full_from_kv.name == "Dual Storage Test"
        assert full_from_kv.description == "Test dual storage consistency"
        logger.info("✅ Verified KV-Storage full record")

        # 4. Verify consistency
        assert str(lite.id) == str(full_from_kv.id), "IDs should match"
        assert lite.group_id == full_from_kv.group_id, "group_id should match"
        assert lite.scene == full_from_kv.scene, "scene should match"
        logger.info("✅ Verified Dual Storage consistency")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)
