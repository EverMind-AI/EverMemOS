#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test UserProfileRawRepository with DualStorageMixin

Verify that DualStorageMixin works correctly for UserProfile.
Repository code remains unchanged, all dual storage logic is handled transparently by Mixin.
"""

import pytest
import pytest_asyncio
import uuid
from typing import TYPE_CHECKING

from core.observation.logger import get_logger

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
        UserProfileRawRepository,
    )


@pytest_asyncio.fixture
async def repository():
    """Get repository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
        UserProfileRawRepository,
    )
    return get_bean_by_type(UserProfileRawRepository)


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


def create_test_user_profile(
    user_id: str,
    group_id: str = "default",
    profile_data: dict = None,
    scenario: str = "group_chat",
    confidence: float = 0.8,
):
    """Create test UserProfile"""
    from infra_layer.adapters.out.persistence.document.memory.user_profile import (
        UserProfile,
    )

    if profile_data is None:
        profile_data = {
            "name": "Test User",
            "role": "Developer",
            "skills": ["Python", "Testing"],
            "preferences": {"theme": "dark", "language": "en"},
        }

    return UserProfile(
        user_id=user_id,
        group_id=group_id,
        profile_data=profile_data,
        scenario=scenario,
        confidence=confidence,
        version=1,
        cluster_ids=[f"cluster_{uuid.uuid4().hex[:8]}"],
        memcell_count=10,
    )


@pytest.mark.asyncio
class TestUserProfileDualStorage:
    """Test UserProfile dual storage functionality"""

    async def test_01_upsert_syncs_to_kv(self, repository, kv_storage, test_user_id, test_group_id):
        """Test: upsert() syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: UserProfile upsert() syncs to KV-Storage")

        # Create test data
        profile_data = {
            "name": "Alice",
            "role": "Engineer",
            "skills": ["Python", "Go"],
        }
        saved = await repository.upsert(
            user_id=test_user_id,
            group_id=test_group_id,
            profile_data=profile_data,
            metadata={"confidence": 0.9, "memcell_count": 15},
        )

        assert saved is not None, "upsert failed"
        doc_id = str(saved.id)
        logger.info(f"✅ Saved: {doc_id}")

        # Verify KV-Storage has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"

        # Verify full data in KV
        from infra_layer.adapters.out.persistence.document.memory.user_profile import (
            UserProfile,
        )

        kv_doc = UserProfile.model_validate_json(kv_value)
        assert kv_doc.profile_data == profile_data
        assert kv_doc.user_id == test_user_id
        assert kv_doc.group_id == test_group_id
        assert kv_doc.confidence == 0.9
        logger.info("✅ Test passed: upsert() syncs to KV-Storage")

    async def test_02_get_by_user_and_group_reads_from_kv(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: get_by_user_and_group() reads from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: UserProfile get_by_user_and_group() reads from KV-Storage")

        # Create test data
        profile_data = {"name": "Bob", "role": "Designer"}
        saved = await repository.upsert(
            user_id=test_user_id, group_id=test_group_id, profile_data=profile_data
        )
        assert saved is not None
        doc_id = str(saved.id)
        logger.info(f"✅ Created: {doc_id}")

        # Get by user and group
        retrieved = await repository.get_by_user_and_group(test_user_id, test_group_id)
        assert retrieved is not None, "get_by_user_and_group failed"
        assert retrieved.profile_data == profile_data
        assert retrieved.user_id == test_user_id
        assert retrieved.group_id == test_group_id
        logger.info("✅ Test passed: get_by_user_and_group() reads from KV-Storage")

    async def test_03_find_by_filters_works(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: find_by_filters() returns full data from KV"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: UserProfile find_by_filters() works with dual storage")

        # Create multiple test records for same user
        for i in range(3):
            profile_data = {
                "name": f"User {i+1}",
                "iteration": i + 1,
            }
            await repository.upsert(
                user_id=test_user_id,
                group_id=f"{test_group_id}_{i}",
                profile_data=profile_data,
            )

        # Query by user_id
        results = await repository.find_by_filters(user_id=test_user_id, limit=10)
        assert len(results) >= 3, f"Should return at least 3 records, got {len(results)}"

        # Verify full data
        found_groups = set()
        for result in results:
            assert result.user_id == test_user_id
            assert result.profile_data is not None
            if result.group_id.startswith(test_group_id):
                found_groups.add(result.group_id)

        assert len(found_groups) == 3, f"Should find 3 unique groups, got {len(found_groups)}"
        logger.info("✅ Test passed: find_by_filters() returns full data")

    async def test_04_get_all_by_group_works(
        self, repository, kv_storage, test_group_id
    ):
        """Test: get_all_by_group() returns full data from KV"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: UserProfile get_all_by_group() works")

        # Create multiple users in same group
        user_ids = []
        for i in range(3):
            user_id = f"test_user_{uuid.uuid4().hex[:8]}"
            user_ids.append(user_id)
            profile_data = {"name": f"User {i+1}"}
            await repository.upsert(
                user_id=user_id, group_id=test_group_id, profile_data=profile_data
            )

        # Query by group_id
        results = await repository.get_all_by_group(test_group_id)
        assert len(results) >= 3, f"Should return at least 3 records, got {len(results)}"

        # Verify full data
        found_users = set()
        for result in results:
            assert result.group_id == test_group_id
            assert result.profile_data is not None
            if result.user_id in user_ids:
                found_users.add(result.user_id)

        assert len(found_users) == 3, f"Should find 3 users, got {len(found_users)}"
        logger.info("✅ Test passed: get_all_by_group() returns full data")

    async def test_05_get_all_by_user_works(
        self, repository, kv_storage, test_user_id
    ):
        """Test: get_all_by_user() returns full data from KV"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: UserProfile get_all_by_user() works")

        # Create multiple profiles for same user in different groups
        group_ids = []
        for i in range(3):
            group_id = f"group_{uuid.uuid4().hex[:8]}"
            group_ids.append(group_id)
            profile_data = {"name": f"Profile {i+1}"}
            await repository.upsert(
                user_id=test_user_id, group_id=group_id, profile_data=profile_data
            )

        # Query by user_id
        results = await repository.get_all_by_user(test_user_id, limit=10)
        assert len(results) >= 3, f"Should return at least 3 records, got {len(results)}"

        # Verify full data
        found_groups = set()
        for result in results:
            assert result.user_id == test_user_id
            assert result.profile_data is not None
            if result.group_id in group_ids:
                found_groups.add(result.group_id)

        assert len(found_groups) == 3, f"Should find 3 groups, got {len(found_groups)}"
        logger.info("✅ Test passed: get_all_by_user() returns full data")

    async def test_06_delete_by_group_removes_from_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: delete_by_group() removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: UserProfile delete_by_group() removes from KV")

        # Create multiple users in same group
        doc_ids = []
        for i in range(3):
            user_id = f"test_user_{uuid.uuid4().hex[:8]}"
            saved = await repository.upsert(
                user_id=user_id,
                group_id=test_group_id,
                profile_data={"name": f"User {i+1}"},
            )
            doc_ids.append(str(saved.id))

        # Verify KV has all data
        for doc_id in doc_ids:
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is not None

        # Delete by group_id
        deleted_count = await repository.delete_by_group(test_group_id)
        assert deleted_count == 3, f"Should delete 3 records, got {deleted_count}"

        # Verify KV removed all
        for doc_id in doc_ids:
            kv_value = await kv_storage.get(doc_id)
            assert (
                kv_value is None
            ), f"KV-Storage should not have {doc_id} after delete"

        logger.info("✅ Test passed: delete_by_group() removes from KV")

    async def test_07_save_profile_works(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: save_profile() (ProfileStorage interface) works with dual storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: UserProfile save_profile() interface works")

        # Create profile using ProfileStorage interface
        profile_data = {
            "name": "Charlie",
            "role": "Manager",
            "team": "Engineering",
        }

        # Use dict directly (duck typing)
        success = await repository.save_profile(
            user_id=test_user_id,
            profile=profile_data,
            metadata={"group_id": test_group_id, "confidence": 0.95},
        )
        assert success, "save_profile should return True"

        # Verify data was saved
        retrieved = await repository.get_by_user_and_group(test_user_id, test_group_id)
        assert retrieved is not None
        assert retrieved.profile_data == profile_data
        assert retrieved.confidence == 0.95

        # Verify KV has the data
        doc_id = str(retrieved.id)
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        logger.info("✅ Test passed: save_profile() interface works")

    async def test_08_get_profile_works(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: get_profile() (ProfileStorage interface) works with dual storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: UserProfile get_profile() interface works")

        # Create test profile
        profile_data = {"name": "Diana", "department": "Sales"}
        await repository.upsert(
            user_id=test_user_id, group_id=test_group_id, profile_data=profile_data
        )

        # Get profile using ProfileStorage interface
        retrieved_profile = await repository.get_profile(test_user_id, test_group_id)
        assert retrieved_profile is not None
        assert retrieved_profile == profile_data

        logger.info("✅ Test passed: get_profile() interface works")

    async def test_09_get_all_profiles_works(
        self, repository, kv_storage, test_group_id
    ):
        """Test: get_all_profiles() (ProfileStorage interface) works"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: UserProfile get_all_profiles() interface works")

        # Create multiple profiles in same group
        test_profiles = {}
        for i in range(3):
            user_id = f"test_user_{uuid.uuid4().hex[:8]}"
            profile_data = {"name": f"User {i+1}", "index": i}
            test_profiles[user_id] = profile_data
            await repository.upsert(
                user_id=user_id, group_id=test_group_id, profile_data=profile_data
            )

        # Get all profiles using ProfileStorage interface
        all_profiles = await repository.get_all_profiles(test_group_id)
        assert isinstance(all_profiles, dict)

        # Verify we got our test profiles
        for user_id, expected_profile in test_profiles.items():
            if user_id in all_profiles:
                assert all_profiles[user_id] == expected_profile

        logger.info("✅ Test passed: get_all_profiles() interface works")

    async def test_10_clear_works(
        self, repository, kv_storage, test_group_id
    ):
        """Test: clear() (ProfileStorage interface) removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: UserProfile clear() interface works")

        # Create test profiles
        doc_ids = []
        for i in range(3):
            user_id = f"test_user_{uuid.uuid4().hex[:8]}"
            saved = await repository.upsert(
                user_id=user_id,
                group_id=test_group_id,
                profile_data={"name": f"User {i+1}"},
            )
            doc_ids.append(str(saved.id))

        # Verify KV has all data
        for doc_id in doc_ids:
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is not None

        # Clear using ProfileStorage interface
        success = await repository.clear(test_group_id)
        assert success, "clear should return True"

        # Verify KV removed all
        for doc_id in doc_ids:
            kv_value = await kv_storage.get(doc_id)
            assert (
                kv_value is None
            ), f"KV-Storage should not have {doc_id} after clear"

        logger.info("✅ Test passed: clear() interface works")

    async def test_11_upsert_update_syncs_to_kv(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: upsert() on existing record updates KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: UserProfile upsert() update syncs to KV-Storage")

        # Create initial profile
        profile_data_v1 = {"name": "Eve", "version": 1}
        saved_v1 = await repository.upsert(
            user_id=test_user_id,
            group_id=test_group_id,
            profile_data=profile_data_v1,
        )
        doc_id = str(saved_v1.id)
        assert saved_v1.version == 1

        # Update profile
        profile_data_v2 = {"name": "Eve Updated", "version": 2}
        saved_v2 = await repository.upsert(
            user_id=test_user_id,
            group_id=test_group_id,
            profile_data=profile_data_v2,
            metadata={"confidence": 0.99},
        )

        # Should be same document ID
        assert str(saved_v2.id) == doc_id
        assert saved_v2.version == 2

        # Verify KV has updated data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        from infra_layer.adapters.out.persistence.document.memory.user_profile import (
            UserProfile,
        )

        kv_doc = UserProfile.model_validate_json(kv_value)
        assert kv_doc.profile_data == profile_data_v2
        assert kv_doc.version == 2
        assert kv_doc.confidence == 0.99

        logger.info("✅ Test passed: upsert() update syncs to KV-Storage")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
