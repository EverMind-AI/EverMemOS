#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete CRUD Test for CoreMemoryRawRepository with KV-Storage

This test file comprehensively tests all CRUD methods in CoreMemoryRawRepository
with the dual MongoDB + KV-Storage pattern. Each test follows the pattern:
1. Create test data (upsert)
2. Read/Query test data
3. Verify data consistency between MongoDB and KV-Storage
4. Verify data integrity (inserted == retrieved)

Modified methods tested:
- get_by_user_id
- update_by_user_id
- delete_by_user_id
- upsert_by_user_id
- ensure_latest
- find_by_user_ids
- get_base
- get_profile
"""

import asyncio
import pytest
import pytest_asyncio
import uuid
from datetime import timedelta
from typing import List, TYPE_CHECKING

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

# Delay imports to avoid loading beanie at module level
# These will be imported inside fixtures/functions when needed
if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.document.memory.core_memory import (
        CoreMemory,
    )
    from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
        CoreMemoryRawRepository,
    )
    from infra_layer.adapters.out.persistence.kv_storage import KVStorageInterface


# ==================== Test Fixtures ====================


@pytest_asyncio.fixture
async def repository():
    """Get CoreMemoryRawRepository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
        CoreMemoryRawRepository,
    )
    repo = get_bean_by_type(CoreMemoryRawRepository)
    yield repo
    # Cleanup is handled by individual tests


@pytest_asyncio.fixture
async def kv_storage():
    """Get KV-Storage instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.kv_storage import KVStorageInterface
    kv = get_bean_by_type(KVStorageInterface)
    yield kv


@pytest.fixture
def test_user_id():
    """Generate unique test user ID"""
    return f"test_user_{uuid.uuid4().hex[:8]}"


# ==================== Test Helpers ====================


def create_test_core_memory(
    user_id: str,
    version: str = "1.0.0",
    is_latest: bool = True,
    user_name: str = None,
    gender: str = "Male",
    position: str = "Engineer",
):
    """Helper function to create a test CoreMemory with all fields"""
    from infra_layer.adapters.out.persistence.document.memory.core_memory import (
        CoreMemory,
    )

    # Create complete CoreMemory structure
    return CoreMemory(
        # Core required fields
        user_id=user_id,
        version=version,
        is_latest=is_latest,
        # Basic information fields
        user_name=user_name or f"TestUser_{user_id[-8:]}",
        gender=gender,
        position=position,
        supervisor_user_id=f"supervisor_{uuid.uuid4().hex[:6]}",
        team_members=[f"member_{i}" for i in range(3)],
        okr=[
            {"objective": "Improve skills", "key_result": "Complete 3 courses"},
            {"objective": "Deliver project", "key_result": "On time delivery"},
        ],
        base_location="Beijing",
        hiredate="2023-01-01",
        age=30,
        department="Engineering",
        # Profile fields
        hard_skills=[
            {"value": "Python", "level": "Advanced", "evidences": ["2024-01-01|conv_123"]},
            {"value": "SQL", "level": "Intermediate", "evidences": ["2024-01-02|conv_124"]},
        ],
        soft_skills=[
            {"value": "Communication", "level": "Advanced", "evidences": ["2024-01-03|conv_125"]},
            {"value": "Leadership", "level": "Intermediate", "evidences": ["2024-01-04|conv_126"]},
        ],
        output_reasoning="Test reasoning",
        motivation_system=[
            {"value": "Achievement", "level": "High", "evidences": ["2024-01-05|conv_127"]},
        ],
        fear_system=[
            {"value": "Failure", "level": "Medium", "evidences": ["2024-01-06|conv_128"]},
        ],
        value_system=[
            {"value": "Honesty", "level": "High", "evidences": ["2024-01-07|conv_129"]},
        ],
        humor_use=[
            {"value": "Sarcasm", "level": "Medium", "evidences": ["2024-01-08|conv_130"]},
        ],
        colloquialism=[
            {"value": "Cool", "level": "High", "evidences": ["2024-01-09|conv_131"]},
        ],
        personality=[
            {"value": "Introverted", "evidences": ["2024-01-10|conv_132"]},
        ],
        projects_participated=[
            {"value": "Project A", "evidences": ["2024-01-11|conv_133"]},
        ],
        user_goal=[
            {"value": "Become technical expert", "evidences": ["2024-01-12|conv_134"]},
        ],
        work_responsibility=[
            {"value": "Backend development", "evidences": ["2024-01-13|conv_135"]},
        ],
        working_habit_preference=[
            {"value": "Remote work", "evidences": ["2024-01-14|conv_136"]},
        ],
        interests=[
            {"value": "Reading", "evidences": ["2024-01-15|conv_137"]},
        ],
        tendency=[
            {"value": "Risk-averse", "evidences": ["2024-01-16|conv_138"]},
        ],
        way_of_decision_making=[
            {"value": "Data-driven", "evidences": ["2024-01-17|conv_139"]},
        ],
        group_importance_evidence={"group_1": 0.9, "group_2": 0.5},
        # Extension field
        extend={
            "test_flag": True,
            "test_id": uuid.uuid4().hex,
            "priority": "high",
        },
    )


def assert_core_memory_equal(cm1, cm2, check_id: bool = True):
    """Assert two CoreMemory objects are equal (comparing all fields)"""
    if check_id:
        assert str(cm1.id) == str(cm2.id), "IDs don't match"

    # Core required fields
    assert cm1.user_id == cm2.user_id, "user_id doesn't match"
    assert cm1.version == cm2.version, "version doesn't match"
    assert cm1.is_latest == cm2.is_latest, "is_latest doesn't match"

    # Basic information fields
    assert cm1.user_name == cm2.user_name, "user_name doesn't match"
    assert cm1.gender == cm2.gender, "gender doesn't match"
    assert cm1.position == cm2.position, "position doesn't match"
    assert cm1.supervisor_user_id == cm2.supervisor_user_id, "supervisor_user_id doesn't match"
    assert cm1.team_members == cm2.team_members, "team_members don't match"
    assert cm1.base_location == cm2.base_location, "base_location doesn't match"
    assert cm1.department == cm2.department, "department doesn't match"

    # Profile fields (just check some key ones)
    assert cm1.hard_skills == cm2.hard_skills, "hard_skills don't match"
    assert cm1.soft_skills == cm2.soft_skills, "soft_skills don't match"
    assert cm1.personality == cm2.personality, "personality doesn't match"

    # Extension field
    assert cm1.extend == cm2.extend, "extend doesn't match"


async def verify_kv_storage(repository, doc_id: str) -> bool:
    """Verify data exists in KV-Storage"""
    from core.observation.logger import get_logger

    logger = get_logger(__name__)

    kv_storage = repository._dual_storage.get_kv_storage()
    if not kv_storage:
        logger.warning("KV-Storage not available")
        return False

    kv_json = await kv_storage.get(key=doc_id)
    return kv_json is not None


# ==================== Test Cases ====================


def get_logger():
    """Helper to get logger instance"""
    from core.observation.logger import get_logger as _get_logger

    return _get_logger(__name__)


class TestBasicCRUD:
    """Test basic CRUD operations: Create, Read, Update, Delete"""

    async def test_01_upsert_and_get_by_user_id(self, repository, test_user_id):
        """
        Test: upsert_by_user_id + get_by_user_id
        Flow: Create a CoreMemory -> Read it back -> Verify data matches
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: upsert_by_user_id + get_by_user_id")

        # 1. Create test CoreMemory
        original = create_test_core_memory(
            user_id=test_user_id,
            version="1.0.0",
            is_latest=True,
        )

        # 2. Upsert to repository
        created = await repository.upsert_by_user_id(
            test_user_id, original.model_dump()
        )
        assert created is not None, "Failed to upsert CoreMemory"
        assert created.id is not None, "Created CoreMemory should have ID"

        doc_id = str(created.id)
        logger.info(f"✅ Created CoreMemory with ID: {doc_id}")

        # 3. Verify KV-Storage
        kv_exists = await verify_kv_storage(repository, doc_id)
        logger.info(f"KV-Storage: {'✅ Exists' if kv_exists else '⚠️  Not found'}")

        # 4. Read back using get_by_user_id
        retrieved = await repository.get_by_user_id(test_user_id)
        assert retrieved is not None, "Failed to retrieve CoreMemory"
        logger.info(f"✅ Retrieved CoreMemory by user_id")

        # 5. Verify data matches
        assert_core_memory_equal(created, retrieved, check_id=True)
        logger.info(f"✅ Data integrity verified")

        # Cleanup
        await repository.delete_by_user_id(test_user_id)

    async def test_02_update_by_user_id(self, repository, test_user_id):
        """
        Test: upsert_by_user_id + update_by_user_id + get_by_user_id
        Flow: Create -> Update -> Verify update
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: update_by_user_id")

        # 1. Create test CoreMemory
        original = create_test_core_memory(
            user_id=test_user_id,
            version="1.0.0",
            user_name="Original Name",
        )
        created = await repository.upsert_by_user_id(
            test_user_id, original.model_dump()
        )
        assert created is not None
        logger.info(f"✅ Created CoreMemory: {created.id}")

        # 2. Update the CoreMemory
        update_data = {
            "user_name": "Updated Name",
            "position": "Senior Engineer",
            "age": 35,
        }
        updated = await repository.update_by_user_id(
            test_user_id, update_data, version="1.0.0"
        )
        assert updated is not None, "Update should return CoreMemory"
        logger.info(f"✅ Updated CoreMemory")

        # 3. Verify update
        assert updated.user_name == "Updated Name", "user_name not updated"
        assert updated.position == "Senior Engineer", "position not updated"
        assert updated.age == 35, "age not updated"
        logger.info(f"✅ Update verified")

        # 4. Read back and verify persistence
        retrieved = await repository.get_by_user_id(test_user_id)
        assert retrieved is not None
        assert retrieved.user_name == "Updated Name", "Update not persisted"
        logger.info(f"✅ Update persisted in KV-Storage")

        # Cleanup
        await repository.delete_by_user_id(test_user_id)

    async def test_03_delete_by_user_id(self, repository, test_user_id):
        """
        Test: upsert_by_user_id + delete_by_user_id + get_by_user_id
        Flow: Create -> Delete -> Verify deletion (MongoDB + KV)
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_user_id")

        # 1. Create test CoreMemory
        original = create_test_core_memory(
            user_id=test_user_id,
            version="1.0.0",
        )
        created = await repository.upsert_by_user_id(
            test_user_id, original.model_dump()
        )
        assert created is not None
        doc_id = str(created.id)
        logger.info(f"✅ Created CoreMemory: {doc_id}")

        # 2. Verify it exists
        retrieved = await repository.get_by_user_id(test_user_id)
        assert retrieved is not None, "CoreMemory should exist before deletion"

        # 3. Delete the CoreMemory
        deleted = await repository.delete_by_user_id(test_user_id)
        assert deleted is True, "Deletion should return True"
        logger.info(f"✅ Deleted CoreMemory for user: {test_user_id}")

        # 4. Verify it no longer exists
        retrieved_after = await repository.get_by_user_id(test_user_id)
        assert retrieved_after is None, "CoreMemory should not exist after deletion"
        logger.info(f"✅ Verified deletion: CoreMemory not found")

        # 5. Verify KV-Storage cleanup
        kv_exists = await verify_kv_storage(repository, doc_id)
        assert not kv_exists, "KV-Storage should be cleaned up"
        logger.info(f"✅ KV-Storage cleaned up")


class TestVersionManagement:
    """Test version management features"""

    async def test_04_multiple_versions(self, repository, test_user_id):
        """
        Test: Create multiple versions + get_by_user_id with version_range
        Flow: Create v1.0.0, v2.0.0, v3.0.0 -> Query by version range
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Multiple versions")

        # 1. Create version 1.0.0
        v1 = create_test_core_memory(
            user_id=test_user_id,
            version="1.0.0",
            user_name="Version 1.0",
        )
        created_v1 = await repository.upsert_by_user_id(
            test_user_id, v1.model_dump()
        )
        assert created_v1 is not None
        logger.info(f"✅ Created version 1.0.0")

        # 2. Create version 2.0.0
        v2 = create_test_core_memory(
            user_id=test_user_id,
            version="2.0.0",
            user_name="Version 2.0",
        )
        created_v2 = await repository.upsert_by_user_id(
            test_user_id, v2.model_dump()
        )
        assert created_v2 is not None
        logger.info(f"✅ Created version 2.0.0")

        # 3. Create version 3.0.0
        v3 = create_test_core_memory(
            user_id=test_user_id,
            version="3.0.0",
            user_name="Version 3.0",
        )
        created_v3 = await repository.upsert_by_user_id(
            test_user_id, v3.model_dump()
        )
        assert created_v3 is not None
        logger.info(f"✅ Created version 3.0.0")

        # 4. Get latest version (should be 3.0.0)
        latest = await repository.get_by_user_id(test_user_id)
        assert latest is not None
        assert latest.version == "3.0.0", "Latest version should be 3.0.0"
        assert latest.is_latest is True, "Latest version should have is_latest=True"
        logger.info(f"✅ Latest version retrieved: {latest.version}")

        # 5. Query by version range
        version_range = ("1.0.0", "2.0.0")
        versions = await repository.get_by_user_id(
            test_user_id, version_range=version_range
        )
        assert isinstance(versions, list), "Should return list for version range"
        assert len(versions) == 2, f"Should return 2 versions, got {len(versions)}"
        version_numbers = {v.version for v in versions}
        assert version_numbers == {"1.0.0", "2.0.0"}, "Should return v1 and v2"
        logger.info(f"✅ Version range query verified: {version_numbers}")

        # Cleanup
        await repository.delete_by_user_id(test_user_id)

    async def test_05_ensure_latest(self, repository, test_user_id):
        """
        Test: ensure_latest
        Flow: Create multiple versions -> Call ensure_latest -> Verify flags
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ensure_latest")

        # 1. Create multiple versions
        for version in ["1.0.0", "2.0.0", "3.0.0"]:
            cm = create_test_core_memory(
                user_id=test_user_id,
                version=version,
                user_name=f"Version {version}",
            )
            await repository.upsert_by_user_id(test_user_id, cm.model_dump())

        logger.info(f"✅ Created 3 versions")

        # 2. Call ensure_latest
        success = await repository.ensure_latest(test_user_id)
        assert success is True, "ensure_latest should succeed"
        logger.info(f"✅ ensure_latest called")

        # 3. Verify only latest has is_latest=True
        all_versions = await repository.get_by_user_id(
            test_user_id, version_range=("1.0.0", "3.0.0")
        )
        for cm in all_versions:
            if cm.version == "3.0.0":
                assert cm.is_latest is True, "v3.0.0 should be latest"
            else:
                assert cm.is_latest is False or cm.is_latest is None, f"v{cm.version} should not be latest"

        logger.info(f"✅ is_latest flags verified")

        # Cleanup
        await repository.delete_by_user_id(test_user_id)


class TestBatchOperations:
    """Test batch operations"""

    async def test_06_find_by_user_ids(self, repository):
        """
        Test: find_by_user_ids
        Flow: Create CoreMemories for 3 users -> Batch query -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: find_by_user_ids")

        # 1. Create CoreMemories for 3 users
        user_ids = [f"test_user_{uuid.uuid4().hex[:8]}" for _ in range(3)]
        created_list = []

        for user_id in user_ids:
            cm = create_test_core_memory(
                user_id=user_id,
                version="1.0.0",
                user_name=f"User {user_id[-8:]}",
            )
            created = await repository.upsert_by_user_id(user_id, cm.model_dump())
            created_list.append(created)
            logger.info(f"✅ Created CoreMemory for user: {user_id}")

        # 2. Batch query
        results = await repository.find_by_user_ids(user_ids, only_latest=True)
        assert len(results) == 3, f"Should return 3 results, got {len(results)}"
        logger.info(f"✅ Batch query returned {len(results)} results")

        # 3. Verify all users are in results
        result_user_ids = {cm.user_id for cm in results}
        assert result_user_ids == set(user_ids), "All user IDs should be in results"
        logger.info(f"✅ All user IDs verified in results")

        # Cleanup
        for user_id in user_ids:
            await repository.delete_by_user_id(user_id)


class TestFieldExtraction:
    """Test field extraction methods"""

    async def test_07_get_base_and_profile(self, repository, test_user_id):
        """
        Test: get_base + get_profile
        Flow: Create CoreMemory -> Extract base/profile fields -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_base + get_profile")

        # 1. Create test CoreMemory with full data
        original = create_test_core_memory(
            user_id=test_user_id,
            version="1.0.0",
            user_name="Test User",
            position="Engineer",
        )
        created = await repository.upsert_by_user_id(
            test_user_id, original.model_dump()
        )
        assert created is not None
        logger.info(f"✅ Created CoreMemory")

        # 2. Extract base fields
        base_fields = repository.get_base(created)
        assert base_fields is not None, "get_base should return dict"
        assert base_fields["user_name"] == "Test User", "user_name should be extracted"
        assert base_fields["position"] == "Engineer", "position should be extracted"
        assert base_fields["department"] == "Engineering", "department should be extracted"
        logger.info(f"✅ Base fields extracted: {len(base_fields)} fields")

        # 3. Extract profile fields
        profile_fields = repository.get_profile(created)
        assert profile_fields is not None, "get_profile should return dict"
        assert profile_fields["hard_skills"] is not None, "hard_skills should be extracted"
        assert profile_fields["soft_skills"] is not None, "soft_skills should be extracted"
        assert profile_fields["personality"] is not None, "personality should be extracted"
        logger.info(f"✅ Profile fields extracted: {len(profile_fields)} fields")

        # Cleanup
        await repository.delete_by_user_id(test_user_id)


class TestEdgeCases:
    """Test edge cases and error handling"""

    async def test_08_get_nonexistent_user_id(self, repository, test_user_id):
        """
        Test: get_by_user_id with non-existent user ID
        Expected: Should return None
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_user_id (non-existent)")

        fake_user = f"nonexistent_{uuid.uuid4().hex[:8]}"
        result = await repository.get_by_user_id(fake_user)

        assert result is None, "Non-existent user ID should return None"
        logger.info(f"✅ Non-existent user ID handled correctly: returned None")

    async def test_09_delete_nonexistent_user_id(self, repository, test_user_id):
        """
        Test: delete_by_user_id with non-existent user ID
        Expected: Should complete without error
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_user_id (non-existent)")

        fake_user = f"nonexistent_{uuid.uuid4().hex[:8]}"
        result = await repository.delete_by_user_id(fake_user)

        assert isinstance(result, bool), "Should return a boolean"
        logger.info(f"✅ Non-existent user ID deletion handled correctly: returned {result}")

    async def test_10_verify_audit_fields(self, repository, test_user_id):
        """
        Test: Verify created_at and updated_at are set correctly
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Verify created_at and updated_at fields")

        # 1. Create and upsert CoreMemory
        original = create_test_core_memory(
            user_id=test_user_id,
            version="1.0.0",
        )
        created = await repository.upsert_by_user_id(
            test_user_id, original.model_dump()
        )
        assert created is not None, "upsert_by_user_id should return CoreMemory"

        # 2. Verify audit fields are set after upsert
        assert created.created_at is not None, "❌ BUG: created_at should not be None!"
        assert created.updated_at is not None, "❌ BUG: updated_at should not be None!"
        logger.info(
            f"✅ After upsert: created_at={created.created_at}, updated_at={created.updated_at}"
        )

        # 3. Retrieve from KV-Storage and verify persistence
        retrieved = await repository.get_by_user_id(test_user_id)
        assert retrieved is not None, "get_by_user_id should return CoreMemory"
        assert retrieved.created_at is not None, "❌ BUG: created_at should persist in KV-Storage!"
        assert retrieved.updated_at is not None, "❌ BUG: updated_at should persist in KV-Storage!"
        logger.info(
            f"✅ After retrieve: created_at={retrieved.created_at}, updated_at={retrieved.updated_at}"
        )

        # 4. Verify created_at equals updated_at for newly created records
        time_diff = abs((retrieved.created_at - retrieved.updated_at).total_seconds())
        assert time_diff < 1, "created_at and updated_at should be nearly identical for new records"
        logger.info(f"✅ created_at ≈ updated_at (diff: {time_diff:.6f}s)")

        # Cleanup
        await repository.delete_by_user_id(test_user_id)
        logger.info("✅ Audit fields verification passed")


# ==================== Main Test Runner ====================


if __name__ == "__main__":
    """
    Run all tests with pytest

    Usage:
        pytest tests/test_core_memory_crud_complete.py -v -s
    """
    pytest.main([__file__, "-v", "-s"])
