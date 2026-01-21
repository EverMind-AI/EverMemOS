#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test ConversationMetaRawRepository with DualStorageMixin

Verify that DualStorageMixin works correctly for ConversationMeta.
Repository code remains unchanged, all dual storage logic is handled transparently by Mixin.
"""

import pytest
import pytest_asyncio
import uuid
from typing import TYPE_CHECKING

from core.observation.logger import get_logger
from memory_layer.profile_manager.config import ScenarioType

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
        ConversationMetaRawRepository,
    )


@pytest_asyncio.fixture
async def repository():
    """Get repository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
        ConversationMetaRawRepository,
    )
    return get_bean_by_type(ConversationMetaRawRepository)


@pytest_asyncio.fixture
async def kv_storage():
    """Get KV-Storage instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
        KVStorageInterface,
    )
    return get_bean_by_type(KVStorageInterface)


@pytest.fixture
def test_group_id():
    """Generate unique test group ID"""
    return f"test_group_{uuid.uuid4().hex[:8]}"


def create_test_conversation_meta(
    group_id: str = None,
    scene: str = ScenarioType.GROUP_CHAT.value,
    name: str = "Test Conversation",
):
    """Create test ConversationMeta"""
    from datetime import datetime
    from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
        ConversationMeta,
    )

    return ConversationMeta(
        version="1.0.0",
        group_id=group_id,
        scene=scene,
        name=name,
        conversation_created_at=datetime.now().isoformat(),
    )


@pytest.mark.asyncio
class TestConversationMetaDualStorage:
    """Test ConversationMeta dual storage functionality"""

    async def test_01_create_syncs_to_kv(self, repository, kv_storage, test_group_id):
        """Test: create_conversation_meta() syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationMeta create_conversation_meta() syncs to KV-Storage")

        # Create test data
        test_data = create_test_conversation_meta(group_id=test_group_id)
        created = await repository.create_conversation_meta(test_data)

        assert created is not None, "create failed"
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Verify KV-Storage has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"

        # Verify full data in KV
        from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
            ConversationMeta,
        )

        kv_doc = ConversationMeta.model_validate_json(kv_value)
        assert kv_doc.group_id == test_group_id
        assert kv_doc.scene == ScenarioType.GROUP_CHAT.value
        logger.info("✅ Test passed: create_conversation_meta() syncs to KV-Storage")

    async def test_02_get_by_group_id_reads_from_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: get_by_group_id() reads from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationMeta get_by_group_id() reads from KV-Storage")

        # Create test data
        test_data = create_test_conversation_meta(
            group_id=test_group_id, name="Test Conversation 2"
        )
        created = await repository.create_conversation_meta(test_data)
        assert created is not None
        logger.info(f"✅ Created: {created.id}")

        # Get by group_id
        retrieved = await repository.get_by_group_id(test_group_id)
        assert retrieved is not None, "get_by_group_id failed"
        assert retrieved.group_id == test_group_id
        assert retrieved.name == "Test Conversation 2"
        logger.info("✅ Test passed: get_by_group_id() reads from KV-Storage")

    async def test_03_list_by_scene_works(self, repository, kv_storage, test_group_id):
        """Test: list_by_scene() returns full data from KV"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationMeta list_by_scene() works with dual storage")

        # Create multiple test records with same scene
        created_ids = []
        for i in range(3):
            test_data = create_test_conversation_meta(
                group_id=f"{test_group_id}_{i}",
                scene=ScenarioType.GROUP_CHAT.value,
                name=f"Test Conversation {i}",
            )
            created = await repository.create_conversation_meta(test_data)
            if created:
                created_ids.append(created.id)

        # Query by scene - only verify the records we just created
        results = await repository.list_by_scene(scene=ScenarioType.GROUP_CHAT.value, limit=100)

        # Filter to only our test records (by matching group_id prefix)
        test_results = [r for r in results if r.group_id and r.group_id.startswith(test_group_id)]

        assert len(test_results) >= 3, f"Should return at least 3 test records, got {len(test_results)}"

        # Verify full data for our test records
        found_count = 0
        for result in test_results:
            if result.group_id.startswith(test_group_id):
                found_count += 1
                assert result.scene == ScenarioType.GROUP_CHAT.value
                assert result.name is not None
                logger.info(f"  Found test record: {result.group_id} - {result.name}")

        assert found_count == 3, f"Should find 3 test records, found {found_count}"

        # Clean up test data
        for result in test_results:
            if result.group_id:
                await repository.delete_by_group_id(result.group_id)

        logger.info("✅ Test passed: list_by_scene() returns full data")

    async def test_04_update_by_group_id_syncs_to_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: update_by_group_id() syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationMeta update_by_group_id() syncs to KV-Storage")

        # Create test data
        test_data = create_test_conversation_meta(group_id=test_group_id)
        created = await repository.create_conversation_meta(test_data)
        doc_id = str(created.id)

        # Update
        updated = await repository.update_by_group_id(
            test_group_id, {"name": "Updated Conversation Name"}
        )
        assert updated is not None, "update_by_group_id should succeed"
        assert updated.name == "Updated Conversation Name"

        # Verify KV has updated data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
            ConversationMeta,
        )

        kv_doc = ConversationMeta.model_validate_json(kv_value)
        assert kv_doc.name == "Updated Conversation Name"

        logger.info("✅ Test passed: update_by_group_id() syncs to KV-Storage")

    async def test_05_upsert_by_group_id_syncs_to_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: upsert_by_group_id() syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationMeta upsert_by_group_id() syncs to KV-Storage")

        # First upsert (insert)
        from datetime import datetime
        upserted = await repository.upsert_by_group_id(
            test_group_id,
            {
                "version": "1.0.0",
                "scene": ScenarioType.GROUP_CHAT.value,
                "name": "Upserted Conversation",
                "conversation_created_at": datetime.now().isoformat(),
            },
        )
        assert upserted is not None, "upsert should succeed"
        doc_id = str(upserted.id)

        # Verify KV has data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        # Second upsert (update)
        upserted2 = await repository.upsert_by_group_id(
            test_group_id, {"name": "Upserted Conversation 2"}
        )
        assert upserted2 is not None
        assert upserted2.name == "Upserted Conversation 2"
        assert str(upserted2.id) == doc_id  # Same document

        # Verify KV has updated data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
            ConversationMeta,
        )

        kv_doc = ConversationMeta.model_validate_json(kv_value)
        assert kv_doc.name == "Upserted Conversation 2"

        logger.info("✅ Test passed: upsert_by_group_id() syncs to KV-Storage")

    async def test_06_delete_by_group_id_removes_from_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: delete_by_group_id() removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationMeta delete_by_group_id() removes from KV-Storage")

        # Create test data
        test_data = create_test_conversation_meta(group_id=test_group_id)
        created = await repository.create_conversation_meta(test_data)
        doc_id = str(created.id)

        # Verify KV has data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        # Delete
        success = await repository.delete_by_group_id(test_group_id)
        assert success, "delete_by_group_id should return True"

        # Verify KV removed
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is None, "KV-Storage should not have the data after delete"

        logger.info("✅ Test passed: delete_by_group_id() removes from KV-Storage")

    async def test_07_default_config_works(self, repository, kv_storage):
        """Test: Default configuration (group_id=None) works with dual storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationMeta default config works with dual storage")

        # Clean up any existing default config using direct MongoDB access
        # This avoids the KV miss issue when old data exists in MongoDB but not in KV
        try:
            from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
                ConversationMeta,
            )

            # Use Beanie's find().delete() to bypass Repository proxy
            delete_result = await ConversationMeta.find({"group_id": None}).delete()
            if delete_result and delete_result.deleted_count > 0:
                logger.info(f"Cleaned up {delete_result.deleted_count} existing default config(s)")

            # Wait a moment for deletion to complete
            import asyncio
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Failed to clean up existing default config: {e}")

        # Create default config (group_id=None)
        test_data = create_test_conversation_meta(
            group_id=None, name="Default Conversation Test07"
        )

        try:
            created = await repository.create_conversation_meta(test_data)
            assert created is not None, "Failed to create default config"
            doc_id = str(created.id)

            # Verify KV has data
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is not None, "KV should have default config"

            # Get default config
            retrieved = await repository.get_by_group_id(None)
            assert retrieved is not None, "Should retrieve default config"
            assert retrieved.group_id is None, "group_id should be None"
            assert "Default Conversation Test07" in retrieved.name, f"Name mismatch: {retrieved.name}"

            logger.info("✅ Test passed: Default config works with dual storage")

        finally:
            # Clean up using direct MongoDB access
            try:
                from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
                    ConversationMeta,
                )

                await ConversationMeta.find({"group_id": None}).delete()
                logger.info("Cleaned up test default config")
            except Exception as e:
                logger.warning(f"Failed to clean up test default config: {e}")

    async def test_08_fallback_to_default_works(self, repository, kv_storage):
        """Test: Fallback to default config works with dual storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationMeta fallback to default works")

        # Clean up any existing default config using direct MongoDB access
        try:
            from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
                ConversationMeta,
            )

            # Use Beanie's find().delete() to bypass Repository proxy
            delete_result = await ConversationMeta.find({"group_id": None}).delete()
            if delete_result and delete_result.deleted_count > 0:
                logger.info(f"Cleaned up {delete_result.deleted_count} existing default config(s)")

            import asyncio
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Failed to clean up existing default config: {e}")

        # Create default config
        default_data = create_test_conversation_meta(
            group_id=None, name="Fallback Default Conversation Test08"
        )

        try:
            created_default = await repository.create_conversation_meta(default_data)
            assert created_default is not None, "Failed to create default config"
            doc_id = str(created_default.id)

            # Verify KV has data
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is not None, "KV should have default config"

            # Try to get non-existent group_id, should fallback to default
            non_existent_group_id = f"non_existent_{uuid.uuid4().hex[:8]}"
            retrieved = await repository.get_by_group_id(non_existent_group_id)
            assert retrieved is not None, "Should fallback to default config"
            assert retrieved.group_id is None, "group_id should be None for default config"
            assert "Fallback Default Conversation Test08" in retrieved.name, f"Name mismatch: {retrieved.name}"

            logger.info("✅ Test passed: Fallback to default works")

        finally:
            # Clean up using direct MongoDB access
            try:
                from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
                    ConversationMeta,
                )

                await ConversationMeta.find({"group_id": None}).delete()
                logger.info("Cleaned up test default config")
            except Exception as e:
                logger.warning(f"Failed to clean up test default config: {e}")

    async def test_09_scene_validation_works(self, repository, test_group_id):
        """Test: Scene validation works correctly"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationMeta scene validation")

        from core.constants.exceptions import ValidationException

        # Try to create with invalid scene
        from datetime import datetime
        from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
            ConversationMeta,
        )

        invalid_data = ConversationMeta(
            version="1.0.0",
            group_id=test_group_id,
            scene="INVALID_SCENE",  # Invalid scene
            name="Invalid Scene Test",
            conversation_created_at=datetime.now().isoformat(),
        )

        # Should raise ValidationException
        with pytest.raises(ValidationException):
            await repository.create_conversation_meta(invalid_data)

        logger.info("✅ Test passed: Scene validation works")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
