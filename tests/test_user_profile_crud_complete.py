#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete CRUD Test for UserProfileRawRepository with KV-Storage

This test file comprehensively tests all CRUD methods in UserProfileRawRepository
with the dual MongoDB + KV-Storage pattern. Each test follows the pattern:
1. Create test data (upsert)
2. Read/Query test data
3. Verify data consistency between MongoDB and KV-Storage
4. Verify data integrity (inserted == retrieved)

Modified methods tested:
- save_profile / get_profile (ProfileStorage interface)
- get_all_profiles (ProfileStorage interface)
- get_profile_history (ProfileStorage interface)
- clear (ProfileStorage interface)
- get_by_user_and_group (native)
- get_all_by_group (native)
- get_all_by_user (native)
- find_by_filters (native) - NEW
- upsert (native)
- delete_by_group (native)
- delete_all (native)
"""

import asyncio
import pytest
import pytest_asyncio
import uuid
from typing import Dict, Any, List, TYPE_CHECKING

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

# Delay imports to avoid loading beanie at module level
if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.document.memory.user_profile import (
        UserProfile,
    )
    from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
        UserProfileRawRepository,
    )
    from infra_layer.adapters.out.persistence.kv_storage import KVStorageInterface


# ==================== Test Fixtures ====================


@pytest_asyncio.fixture
async def repository():
    """Get UserProfileRawRepository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
        UserProfileRawRepository,
    )
    repo = get_bean_by_type(UserProfileRawRepository)
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


@pytest.fixture
def test_group_id():
    """Generate unique test group ID"""
    return f"test_group_{uuid.uuid4().hex[:8]}"


# ==================== Test Helpers ====================


def create_test_profile_data() -> Dict[str, Any]:
    """Helper function to create test profile data"""
    return {
        "hard_skills": ["Python", "Machine Learning", "System Design"],
        "soft_skills": ["Communication", "Leadership", "Problem Solving"],
        "personality": {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.6,
            "agreeableness": 0.75,
            "neuroticism": 0.3,
        },
        "projects_participated": [
            {
                "name": "Project Alpha",
                "role": "Tech Lead",
                "description": "Led the development of core features",
            },
            {
                "name": "Project Beta",
                "role": "Developer",
                "description": "Implemented backend services",
            },
        ],
        "user_goal": "Build scalable distributed systems",
        "work_responsibility": "Full-stack development and architecture",
        "working_habit_preference": "Morning coding sessions, afternoon meetings",
        "interests": ["Open Source", "Cloud Architecture", "AI/ML"],
        "tendency": "Detail-oriented, systematic approach",
    }


def assert_user_profile_equal(up1, up2, check_id: bool = True):
    """Assert two UserProfile objects are equal (comparing all fields)"""
    if check_id:
        assert str(up1.id) == str(up2.id), "IDs don't match"

    assert up1.user_id == up2.user_id, "user_id doesn't match"
    assert up1.group_id == up2.group_id, "group_id doesn't match"
    assert up1.profile_data == up2.profile_data, "profile_data doesn't match"
    assert up1.scenario == up2.scenario, "scenario doesn't match"
    assert up1.confidence == up2.confidence, "confidence doesn't match"
    assert up1.version == up2.version, "version doesn't match"
    assert up1.cluster_ids == up2.cluster_ids, "cluster_ids don't match"
    assert up1.memcell_count == up2.memcell_count, "memcell_count doesn't match"
    assert (
        up1.last_updated_cluster == up2.last_updated_cluster
    ), "last_updated_cluster doesn't match"


# ==================== Test Classes ====================


class TestProfileStorageInterface:
    """Test ProfileStorage interface methods"""

    @pytest.mark.asyncio
    async def test_save_and_get_profile(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test save_profile and get_profile (ProfileStorage interface)"""
        from infra_layer.adapters.out.persistence.document.memory.user_profile import (
            UserProfile,
        )

        # 1. Create test profile data
        profile_data = create_test_profile_data()
        metadata = {
            "group_id": test_group_id,
            "scenario": "group_chat",
            "confidence": 0.85,
            "cluster_id": "cluster_a",
            "memcell_count": 150,
        }

        # 2. Save using save_profile (ProfileStorage interface)
        success = await repository.save_profile(test_user_id, profile_data, metadata)
        assert success, "save_profile should return True"

        # 3. Get using get_profile
        retrieved_data = await repository.get_profile(test_user_id, test_group_id)
        assert retrieved_data is not None, "get_profile should return profile data"
        assert retrieved_data == profile_data, "Retrieved data should match saved data"

        # 4. Verify MongoDB Lite exists
        from infra_layer.adapters.out.persistence.document.memory.user_profile_lite import (
            UserProfileLite,
        )
        lite = await UserProfileLite.find_one(
            {"user_id": test_user_id, "group_id": test_group_id}
        )
        assert lite is not None, "UserProfileLite should exist in MongoDB"
        assert lite.user_id == test_user_id
        assert lite.group_id == test_group_id

        # 5. Verify KV-Storage has complete data
        user_profile_id = str(lite.id)
        kv_json = await kv_storage.get(user_profile_id)
        assert kv_json is not None, "KV-Storage should contain full UserProfile"

        full_profile = UserProfile.model_validate_json(kv_json)
        assert full_profile.user_id == test_user_id
        assert full_profile.group_id == test_group_id
        assert full_profile.profile_data == profile_data

        # Cleanup
        await repository.delete_by_group(test_group_id)

    @pytest.mark.asyncio
    async def test_get_all_profiles(
        self, repository, test_user_id, test_group_id
    ):
        """Test get_all_profiles method"""
        # Create multiple profiles in same group
        user_ids = [f"{test_user_id}_{i}" for i in range(3)]
        for user_id in user_ids:
            profile_data = create_test_profile_data()
            profile_data["user_goal"] = f"Goal for {user_id}"
            await repository.save_profile(
                user_id, profile_data, {"group_id": test_group_id}
            )

        # Get all profiles in group
        all_profiles = await repository.get_all_profiles(test_group_id)

        assert len(all_profiles) == 3, "Should have 3 profiles"
        assert set(all_profiles.keys()) == set(user_ids), "Should have all user IDs"

        # Cleanup
        await repository.delete_by_group(test_group_id)

    @pytest.mark.asyncio
    async def test_get_profile_history(
        self, repository, test_user_id, test_group_id
    ):
        """Test get_profile_history method"""
        # Create profile
        profile_data = create_test_profile_data()
        metadata = {
            "group_id": test_group_id,
            "cluster_id": "cluster_a",
            "memcell_count": 100,
        }
        await repository.save_profile(test_user_id, profile_data, metadata)

        # Get history
        history = await repository.get_profile_history(test_user_id, test_group_id)

        assert len(history) == 1, "Should have 1 history entry"
        assert history[0]["version"] == 1
        assert history[0]["profile"] == profile_data
        assert history[0]["cluster_id"] == "cluster_a"
        assert history[0]["memcell_count"] == 100

        # Cleanup
        await repository.delete_by_group(test_group_id)

    @pytest.mark.asyncio
    async def test_clear_by_group(self, repository, test_user_id, test_group_id):
        """Test clear method with group_id"""
        # Create profiles
        for i in range(2):
            profile_data = create_test_profile_data()
            await repository.save_profile(
                f"{test_user_id}_{i}", profile_data, {"group_id": test_group_id}
            )

        # Clear by group
        success = await repository.clear(test_group_id)
        assert success, "clear should return True"

        # Verify deleted
        all_profiles = await repository.get_all_profiles(test_group_id)
        assert len(all_profiles) == 0, "All profiles should be deleted"


class TestNativeCRUD:
    """Test native CRUD methods"""

    @pytest.mark.asyncio
    async def test_upsert_new(self, repository, kv_storage, test_user_id, test_group_id):
        """Test upsert for new profile"""
        from infra_layer.adapters.out.persistence.document.memory.user_profile import (
            UserProfile,
        )

        # 1. Create test data
        profile_data = create_test_profile_data()
        metadata = {
            "scenario": "group_chat",
            "confidence": 0.85,
            "cluster_id": "cluster_a",
            "memcell_count": 150,
        }

        # 2. Upsert (insert)
        result = await repository.upsert(
            test_user_id, test_group_id, profile_data, metadata
        )
        assert result is not None, "upsert should return UserProfile"
        assert result.user_id == test_user_id
        assert result.group_id == test_group_id
        assert result.version == 1
        assert result.confidence == 0.85
        assert "cluster_a" in result.cluster_ids

        # 3. Verify MongoDB Lite
        from infra_layer.adapters.out.persistence.document.memory.user_profile_lite import (
            UserProfileLite,
        )
        lite = await UserProfileLite.find_one(
            {"user_id": test_user_id, "group_id": test_group_id}
        )
        assert lite is not None
        assert lite.user_id == test_user_id
        assert lite.group_id == test_group_id

        # 4. Verify KV-Storage
        kv_json = await kv_storage.get(str(result.id))
        assert kv_json is not None, "KV-Storage should contain full UserProfile"

        full_profile = UserProfile.model_validate_json(kv_json)
        assert full_profile.profile_data == profile_data

        # Cleanup
        await repository.delete_by_group(test_group_id)

    @pytest.mark.asyncio
    async def test_upsert_update(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test upsert for updating existing profile"""
        # 1. Create initial profile
        profile_data_v1 = create_test_profile_data()
        metadata_v1 = {
            "scenario": "group_chat",
            "confidence": 0.85,
            "cluster_id": "cluster_a",
            "memcell_count": 150,
        }
        result_v1 = await repository.upsert(
            test_user_id, test_group_id, profile_data_v1, metadata_v1
        )
        original_id = result_v1.id

        # 2. Update with new data
        profile_data_v2 = create_test_profile_data()
        profile_data_v2["hard_skills"].append("Docker")
        metadata_v2 = {
            "confidence": 0.90,
            "cluster_id": "cluster_b",
            "memcell_count": 200,
        }
        result_v2 = await repository.upsert(
            test_user_id, test_group_id, profile_data_v2, metadata_v2
        )

        # 3. Verify ID remains the same (update, not insert)
        assert result_v2.id == original_id, "ID should remain the same on update"

        # 4. Verify version incremented
        assert result_v2.version == 2, "Version should increment to 2"

        # 5. Verify new data is stored
        assert result_v2.profile_data == profile_data_v2
        assert result_v2.confidence == 0.90
        assert "cluster_a" in result_v2.cluster_ids
        assert "cluster_b" in result_v2.cluster_ids
        assert result_v2.last_updated_cluster == "cluster_b"
        assert result_v2.memcell_count == 200

        # 6. Verify KV-Storage has updated data
        from infra_layer.adapters.out.persistence.document.memory.user_profile import (
            UserProfile,
        )
        kv_json = await kv_storage.get(str(result_v2.id))
        full_profile = UserProfile.model_validate_json(kv_json)
        assert full_profile.profile_data == profile_data_v2
        assert full_profile.version == 2

        # Cleanup
        await repository.delete_by_group(test_group_id)

    @pytest.mark.asyncio
    async def test_get_by_user_and_group(
        self, repository, test_user_id, test_group_id
    ):
        """Test get_by_user_and_group method"""
        # 1. Create profile
        profile_data = create_test_profile_data()
        saved = await repository.upsert(test_user_id, test_group_id, profile_data)
        assert saved is not None

        # 2. Get by user and group
        retrieved = await repository.get_by_user_and_group(test_user_id, test_group_id)
        assert retrieved is not None, "get_by_user_and_group should return UserProfile"

        # 3. Verify all fields match
        assert_user_profile_equal(saved, retrieved)

        # 4. Test non-existent profile
        non_existent = await repository.get_by_user_and_group(
            "non_existent_user", "non_existent_group"
        )
        assert non_existent is None, "Should return None for non-existent profile"

        # Cleanup
        await repository.delete_by_group(test_group_id)

    @pytest.mark.asyncio
    async def test_get_all_by_group(self, repository, test_user_id, test_group_id):
        """Test get_all_by_group method"""
        # 1. Create multiple profiles in same group
        user_ids = [f"{test_user_id}_{i}" for i in range(3)]
        for user_id in user_ids:
            profile_data = create_test_profile_data()
            await repository.upsert(user_id, test_group_id, profile_data)

        # 2. Get all by group
        profiles = await repository.get_all_by_group(test_group_id)

        # 3. Verify count
        assert len(profiles) == 3, "Should have 3 profiles"

        # 4. Verify all user IDs present
        retrieved_user_ids = {p.user_id for p in profiles}
        assert retrieved_user_ids == set(user_ids), "Should have all user IDs"

        # Cleanup
        await repository.delete_by_group(test_group_id)

    @pytest.mark.asyncio
    async def test_get_all_by_user(self, repository, test_user_id):
        """Test get_all_by_user method"""
        # 1. Create profiles for same user in different groups
        group_ids = [f"group_{i}_{uuid.uuid4().hex[:8]}" for i in range(3)]
        for group_id in group_ids:
            profile_data = create_test_profile_data()
            await repository.upsert(test_user_id, group_id, profile_data)

        # 2. Get all by user
        profiles = await repository.get_all_by_user(test_user_id, limit=10)

        # 3. Verify count
        assert len(profiles) == 3, "Should have 3 profiles"

        # 4. Verify all group IDs present
        retrieved_group_ids = {p.group_id for p in profiles}
        assert retrieved_group_ids == set(group_ids), "Should have all group IDs"

        # Cleanup
        for group_id in group_ids:
            await repository.delete_by_group(group_id)

    @pytest.mark.asyncio
    async def test_find_by_filters(self, repository):
        """Test find_by_filters method with various filter combinations"""
        from core.oxm.constants import MAGIC_ALL

        # 1. Generate fixed user_ids and group_ids first
        user_ids = [f"user_{i}_{uuid.uuid4().hex[:8]}" for i in range(2)]
        group_ids = [f"group_{j}_{uuid.uuid4().hex[:8]}" for j in range(2)]

        # 2. Create test data: each user in each group (2x2 = 4 profiles)
        test_data = []
        for user_id in user_ids:
            for group_id in group_ids:
                profile_data = create_test_profile_data()
                profile_data["user_goal"] = f"Goal for {user_id} in {group_id}"
                saved = await repository.upsert(user_id, group_id, profile_data)
                test_data.append((user_id, group_id, saved))

        # 3. Test filter by user_id only (should find 2 profiles - same user in 2 groups)
        user_0_id = user_ids[0]
        results = await repository.find_by_filters(user_id=user_0_id, group_id=MAGIC_ALL)
        assert len(results) == 2, f"Should find 2 profiles for {user_0_id}, got {len(results)}"
        assert all(p.user_id == user_0_id for p in results), "All should have same user_id"

        # Verify returned objects are full UserProfile (not Lite)
        assert all(hasattr(p, 'profile_data') for p in results), "Should have profile_data field"
        assert all(p.profile_data is not None for p in results), "profile_data should not be None"

        # 4. Test filter by group_id only (should find 2 profiles - 2 users in same group)
        group_0_id = group_ids[0]
        results = await repository.find_by_filters(user_id=MAGIC_ALL, group_id=group_0_id)
        assert len(results) == 2, f"Should find 2 profiles for {group_0_id}, got {len(results)}"
        assert all(p.group_id == group_0_id for p in results), "All should have same group_id"

        # 5. Test filter by both user_id and group_id (should find exactly 1)
        specific_user_id = user_ids[0]
        specific_group_id = group_ids[0]
        results = await repository.find_by_filters(
            user_id=specific_user_id, group_id=specific_group_id
        )
        assert len(results) == 1, "Should find exactly 1 profile"
        assert results[0].user_id == specific_user_id
        assert results[0].group_id == specific_group_id

        # 6. Test no filters (find all) - MAGIC_ALL is default
        results = await repository.find_by_filters()
        assert len(results) >= 4, "Should find at least 4 profiles (all test data)"

        # 7. Test with limit
        results = await repository.find_by_filters(limit=2)
        assert len(results) == 2, "Should respect limit parameter"

        # 8. Test non-existent filters
        results = await repository.find_by_filters(
            user_id="nonexistent_user", group_id=MAGIC_ALL
        )
        assert len(results) == 0, "Should return empty list for non-existent user"

        # Cleanup
        for group_id in group_ids:
            await repository.delete_by_group(group_id)


class TestDeletion:
    """Test deletion operations"""

    @pytest.mark.asyncio
    async def test_delete_by_group(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test delete_by_group method"""
        # 1. Create profiles
        user_ids = [f"{test_user_id}_{i}" for i in range(2)]
        profile_ids = []
        for user_id in user_ids:
            profile_data = create_test_profile_data()
            saved = await repository.upsert(user_id, test_group_id, profile_data)
            profile_ids.append(str(saved.id))

        # 2. Verify exists in both storages
        from infra_layer.adapters.out.persistence.document.memory.user_profile_lite import (
            UserProfileLite,
        )
        lites_before = await UserProfileLite.find(
            {"group_id": test_group_id}
        ).to_list()
        assert len(lites_before) == 2, "Should have 2 profiles in MongoDB"

        for profile_id in profile_ids:
            kv_before = await kv_storage.get(profile_id)
            assert kv_before is not None, "Should exist in KV-Storage"

        # 3. Delete by group
        count = await repository.delete_by_group(test_group_id)
        assert count == 2, "Should delete 2 profiles"

        # 4. Verify deleted from MongoDB
        lites_after = await UserProfileLite.find({"group_id": test_group_id}).to_list()
        assert len(lites_after) == 0, "Should delete all from MongoDB"

        # 5. Verify deleted from KV-Storage
        for profile_id in profile_ids:
            kv_after = await kv_storage.get(profile_id)
            assert kv_after is None, "Should delete from KV-Storage"

    @pytest.mark.asyncio
    async def test_delete_all(self, repository, kv_storage):
        """Test delete_all method"""
        # 1. Create multiple profiles
        test_data = []
        for i in range(3):
            user_id = f"test_user_{uuid.uuid4().hex[:8]}"
            group_id = f"test_group_{uuid.uuid4().hex[:8]}"
            profile_data = create_test_profile_data()
            saved = await repository.upsert(user_id, group_id, profile_data)
            test_data.append((user_id, group_id, str(saved.id)))

        # 2. Verify all exist
        from infra_layer.adapters.out.persistence.document.memory.user_profile_lite import (
            UserProfileLite,
        )
        lites_before = await UserProfileLite.find_all().to_list()
        assert len(lites_before) >= 3, "Should have at least 3 profiles"

        # 3. Delete all
        count = await repository.delete_all()
        assert count >= 3, f"Should delete at least 3 records, got {count}"

        # 4. Verify all deleted from MongoDB
        lites_after = await UserProfileLite.find_all().to_list()
        assert len(lites_after) == 0, "Should delete all from MongoDB"

        # 5. Verify all deleted from KV-Storage
        for _, _, profile_id in test_data:
            kv_after = await kv_storage.get(profile_id)
            assert kv_after is None, f"Should delete {profile_id} from KV-Storage"


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_get_nonexistent_profile(self, repository):
        """Test getting non-existent profile"""
        result = await repository.get_by_user_and_group(
            "nonexistent_user", "nonexistent_group"
        )
        assert result is None, "Should return None for non-existent profile"

    @pytest.mark.asyncio
    async def test_empty_profile_data(
        self, repository, test_user_id, test_group_id
    ):
        """Test profile with minimal data"""
        # Create minimal profile
        minimal_data = {}
        result = await repository.upsert(test_user_id, test_group_id, minimal_data)

        assert result is not None, "Should save empty profile data"
        assert result.profile_data == {}

        # Retrieve and verify
        retrieved = await repository.get_by_user_and_group(test_user_id, test_group_id)
        assert retrieved is not None
        assert retrieved.profile_data == {}

        # Cleanup
        await repository.delete_by_group(test_group_id)

    @pytest.mark.asyncio
    async def test_large_profile_data(
        self, repository, test_user_id, test_group_id
    ):
        """Test profile with large data"""
        # Create large profile data
        large_data = create_test_profile_data()
        large_data["projects_participated"] = [
            {
                "name": f"Project_{i}",
                "role": "Developer",
                "description": f"Description for project {i}",
            }
            for i in range(50)
        ]

        result = await repository.upsert(test_user_id, test_group_id, large_data)
        assert result is not None
        assert len(result.profile_data["projects_participated"]) == 50

        # Retrieve and verify
        retrieved = await repository.get_by_user_and_group(test_user_id, test_group_id)
        assert retrieved is not None
        assert len(retrieved.profile_data["projects_participated"]) == 50

        # Cleanup
        await repository.delete_by_group(test_group_id)

    @pytest.mark.asyncio
    async def test_multiple_updates(
        self, repository, test_user_id, test_group_id
    ):
        """Test multiple consecutive updates"""
        profile_data = create_test_profile_data()

        # First insert
        result1 = await repository.upsert(test_user_id, test_group_id, profile_data)
        assert result1.version == 1

        # Multiple updates
        for i in range(2, 6):
            profile_data["hard_skills"].append(f"Skill_{i}")
            result = await repository.upsert(
                test_user_id,
                test_group_id,
                profile_data,
                {"cluster_id": f"cluster_{chr(96 + i)}"},
            )
            assert result.version == i, f"Version should be {i}"
            # First insert has no cluster, so cluster_ids starts from 0
            # i=2 -> 1 cluster, i=3 -> 2 clusters, etc.
            assert len(result.cluster_ids) == i - 1, f"Should have {i - 1} clusters"

        # Final verify
        final = await repository.get_by_user_and_group(test_user_id, test_group_id)
        assert final.version == 5
        assert len(final.profile_data["hard_skills"]) >= 7  # Original 3 + 4 added

        # Cleanup
        await repository.delete_by_group(test_group_id)
