"""
单元测试：测试 fetch_mem_service.py 的功能
"""

import pytest

from agentic_layer.fetch_mem_service import get_fetch_memory_service
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
    EventLogModel,
    SemanticMemoryRecordModel,
)


class TestFakeFetchMemoryService:
    """测试 FakeFetchMemoryService"""

    @pytest.fixture
    def service(self):
        """创建服务实例"""
        return get_fetch_memory_service()

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
    async def test_find_by_user_id_personal_event_log(self, service):
        """测试查找个人事件日志"""
        response = await service.find_by_user_id(
            "user_001", MemoryType.PERSONAL_EVENT_LOG, limit=5
        )

        assert response.total_count > 0
        assert all(isinstance(memory, EventLogModel) for memory in response.memories)
        assert all(memory.user_id == "user_001" for memory in response.memories)

    @pytest.mark.asyncio
    async def test_find_by_user_id_personal_semantic_memory(self, service):
        """测试查找个人语义记忆"""
        response = await service.find_by_user_id(
            "user_001", MemoryType.PERSONAL_SEMANTIC_MEMORY, limit=5
        )

        assert response.total_count > 0
        assert all(
            isinstance(memory, SemanticMemoryRecordModel)
            for memory in response.memories
        )
        # 个人语义记忆的user_id可能为None（群组场景），所以不强制检查

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


class TestGetFetchMemoryService:
    """测试获取服务实例"""

    @pytest.mark.asyncio
    async def test_get_fetch_memory_service(self):
        """测试通过工厂函数获取服务实例"""
        service = get_fetch_memory_service()

        # 验证服务实例可用
        response = await service.find_by_user_id(
            "user_001", MemoryType.BASE_MEMORY, limit=3
        )

        assert response.total_count > 0
        assert len(response.memories) <= 3
        assert all(isinstance(memory, BaseMemoryModel) for memory in response.memories)
        assert all(memory.user_id == "user_001" for memory in response.memories)


if __name__ == "__main__":
    pytest.main([__file__])
