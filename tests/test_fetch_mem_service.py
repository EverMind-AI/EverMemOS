"""
单元测试：测试 fetch_mem_service.py 的功能
"""

import pytest
from datetime import datetime

from agentic_layer.fetch_mem_service import (
    FakeFetchMemoryService,
    get_fetch_memory_service,
    set_fetch_memory_service,
    find_memories_by_user_id,
)
from agentic_layer.memory_models import (
    MemoryType,
    BaseMemoryModel,
    ProfileModel,
    PreferenceModel,
    EpisodicMemoryModel,
    SemanticMemoryModel,
    EntityModel,
    RelationModel,
    BehaviorHistoryModel,
)


class TestFakeFetchMemoryService:
    """测试 FakeFetchMemoryService"""

    @pytest.fixture
    def service(self):
        """创建服务实例"""
        return FakeFetchMemoryService()

    @pytest.mark.asyncio
    async def test_find_by_user_id_base_memory(self, service):
        """测试查找基础记忆"""
        response = await service.find_by_user_id(
            "user_001", MemoryType.BASE_MEMORY, limit=5
        )

        assert response.total_count > 0
        assert len(response.memories) <= 5
        assert all(isinstance(memory, BaseMemoryModel) for memory in response.memories)
        assert all(memory.user_id == "user_001" for memory in response.memories)
        assert response.metadata["user_id"] == "user_001"
        assert response.metadata["memory_type"] == "base_memory"

    @pytest.mark.asyncio
    async def test_find_by_user_id_profile(self, service):
        """测试查找用户画像"""
        response = await service.find_by_user_id(
            "user_001", MemoryType.PROFILE, limit=5
        )

        assert response.total_count > 0
        assert len(response.memories) <= 5
        assert all(isinstance(memory, ProfileModel) for memory in response.memories)
        assert all(memory.user_id == "user_001" for memory in response.memories)

    @pytest.mark.asyncio
    async def test_find_by_user_id_preference(self, service):
        """测试查找用户偏好"""
        response = await service.find_by_user_id(
            "user_001", MemoryType.PREFERENCE, limit=5
        )

        assert response.total_count > 0
        assert all(isinstance(memory, PreferenceModel) for memory in response.memories)
        assert all(memory.user_id == "user_001" for memory in response.memories)

    @pytest.mark.asyncio
    async def test_find_by_user_id_episodic_memory(self, service):
        """测试查找情景记忆"""
        response = await service.find_by_user_id(
            "user_001", MemoryType.EPISODIC_MEMORY, limit=5
        )

        assert response.total_count > 0
        assert all(
            isinstance(memory, EpisodicMemoryModel) for memory in response.memories
        )
        assert all(memory.user_id == "user_001" for memory in response.memories)

    @pytest.mark.asyncio
    async def test_find_by_user_id_semantic_memory(self, service):
        """测试查找语义记忆"""
        response = await service.find_by_user_id(
            "user_001", MemoryType.SEMANTIC_MEMORY, limit=5
        )

        assert response.total_count > 0
        assert all(
            isinstance(memory, SemanticMemoryModel) for memory in response.memories
        )
        assert all(memory.user_id == "user_001" for memory in response.memories)

    @pytest.mark.asyncio
    async def test_find_by_user_id_entity(self, service):
        """测试查找实体"""
        response = await service.find_by_user_id("user_001", MemoryType.ENTITY, limit=5)

        assert response.total_count > 0
        assert all(isinstance(memory, EntityModel) for memory in response.memories)
        assert all(memory.user_id == "user_001" for memory in response.memories)

    @pytest.mark.asyncio
    async def test_find_by_user_id_relation(self, service):
        """测试查找关系"""
        response = await service.find_by_user_id(
            "user_001", MemoryType.RELATION, limit=5
        )

        assert response.total_count > 0
        assert all(isinstance(memory, RelationModel) for memory in response.memories)
        assert all(memory.user_id == "user_001" for memory in response.memories)

    @pytest.mark.asyncio
    async def test_find_by_user_id_behavior_history(self, service):
        """测试查找行为历史"""
        response = await service.find_by_user_id(
            "user_001", MemoryType.BEHAVIOR_HISTORY, limit=5
        )

        assert response.total_count > 0
        assert all(
            isinstance(memory, BehaviorHistoryModel) for memory in response.memories
        )
        assert all(memory.user_id == "user_001" for memory in response.memories)

    @pytest.mark.asyncio
    async def test_find_by_user_id_nonexistent_user(self, service):
        """测试查找不存在的用户"""
        response = await service.find_by_user_id(
            "nonexistent_user", MemoryType.BASE_MEMORY, limit=5
        )

        assert response.total_count == 0
        assert len(response.memories) == 0
        assert not response.has_more

    @pytest.mark.asyncio
    async def test_find_by_user_id_limit(self, service):
        """测试限制返回数量"""
        response = await service.find_by_user_id(
            "user_001", MemoryType.BASE_MEMORY, limit=1
        )

        assert len(response.memories) <= 1

    @pytest.mark.asyncio
    async def test_different_users_have_different_data(self, service):
        """测试不同用户有不同的数据"""
        response1 = await service.find_by_user_id(
            "user_001", MemoryType.PROFILE, limit=5
        )
        response2 = await service.find_by_user_id(
            "user_002", MemoryType.PROFILE, limit=5
        )

        assert len(response1.memories) > 0
        assert len(response2.memories) > 0

        # 检查用户ID不同
        profile1 = response1.memories[0]
        profile2 = response2.memories[0]
        assert profile1.user_id != profile2.user_id


class TestServiceSingleton:
    """测试服务单例模式"""

    def test_get_fetch_memory_service_singleton(self):
        """测试获取服务实例是单例"""
        service1 = get_fetch_memory_service()
        service2 = get_fetch_memory_service()

        assert service1 is service2
        assert isinstance(service1, FakeFetchMemoryService)

    def test_set_fetch_memory_service(self):
        """测试设置服务实例"""
        original_service = get_fetch_memory_service()

        # 创建新的服务实例
        new_service = FakeFetchMemoryService()
        set_fetch_memory_service(new_service)

        # 验证服务已更换
        current_service = get_fetch_memory_service()
        assert current_service is new_service
        assert current_service is not original_service

        # 恢复原服务
        set_fetch_memory_service(original_service)


class TestConvenienceFunction:
    """测试便捷函数"""

    @pytest.mark.asyncio
    async def test_find_memories_by_user_id(self):
        """测试便捷函数"""
        response = await find_memories_by_user_id(
            "user_001", MemoryType.BASE_MEMORY, limit=3
        )

        assert response.total_count > 0
        assert len(response.memories) <= 3
        assert all(isinstance(memory, BaseMemoryModel) for memory in response.memories)
        assert all(memory.user_id == "user_001" for memory in response.memories)


class TestMemoryTypes:
    """测试所有记忆类型"""

    @pytest.fixture
    def service(self):
        return FakeFetchMemoryService()

    @pytest.mark.asyncio
    async def test_all_memory_types_available(self, service):
        """测试所有记忆类型都有数据"""
        user_id = "user_001"

        for memory_type in MemoryType:
            response = await service.find_by_user_id(user_id, memory_type, limit=5)
            assert (
                response.total_count >= 0
            ), f"Memory type {memory_type} should have data or empty result"

            # 验证返回的记忆类型正确
            if response.memories:
                memory = response.memories[0]
                expected_types = {
                    MemoryType.BASE_MEMORY: BaseMemoryModel,
                    MemoryType.PROFILE: ProfileModel,
                    MemoryType.PREFERENCE: PreferenceModel,
                    MemoryType.EPISODIC_MEMORY: EpisodicMemoryModel,
                    MemoryType.SEMANTIC_MEMORY: SemanticMemoryModel,
                    MemoryType.ENTITY: EntityModel,
                    MemoryType.RELATION: RelationModel,
                    MemoryType.BEHAVIOR_HISTORY: BehaviorHistoryModel,
                }
                expected_type = expected_types[memory_type]
                assert isinstance(
                    memory, expected_type
                ), f"Expected {expected_type}, got {type(memory)}"


if __name__ == "__main__":
    pytest.main([__file__])
