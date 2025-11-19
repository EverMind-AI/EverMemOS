"""
单元测试：测试 memory_models.py 的数据模型
"""

import pytest
from datetime import datetime
from typing import Any, Dict

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
    MemoryQueryRequest,
    MemoryQueryResponse,
    Metadata,
)


class TestMemoryType:
    """测试记忆类型枚举"""

    def test_memory_type_values(self):
        """测试记忆类型枚举值"""
        assert MemoryType.BASE_MEMORY == "base_memory"
        assert MemoryType.PROFILE == "profile"
        assert MemoryType.PREFERENCE == "preference"
        assert MemoryType.EPISODIC_MEMORY == "episodic_memory"
        assert MemoryType.SEMANTIC_MEMORY == "semantic_memory"
        assert MemoryType.ENTITY == "entity"
        assert MemoryType.RELATION == "relation"
        assert MemoryType.BEHAVIOR_HISTORY == "behavior_history"

    def test_memory_type_count(self):
        """测试记忆类型数量"""
        assert len(MemoryType) == 8


class TestBaseMemoryModel:
    """测试基础记忆模型"""

    def test_create_base_memory_model(self):
        """测试创建基础记忆模型"""
        now = datetime.now()
        model = BaseMemoryModel(
            id="base_001",
            user_id="user_001",
            content="测试内容",
            created_at=now,
            updated_at=now,
            metadata={"type": "test"},
        )

        assert model.id == "base_001"
        assert model.user_id == "user_001"
        assert model.content == "测试内容"
        assert model.created_at == now
        assert model.updated_at == now
        assert model.metadata == {"type": "test"}

    def test_base_memory_model_default_metadata(self):
        """测试基础记忆模型默认元数据"""
        now = datetime.now()
        model = BaseMemoryModel(
            id="base_001",
            user_id="user_001",
            content="测试内容",
            created_at=now,
            updated_at=now,
        )

        assert model.metadata == {}


class TestProfileModel:
    """测试用户画像模型"""

    def test_create_profile_model(self):
        """测试创建用户画像模型"""
        model = ProfileModel(
            id="profile_001",
            user_id="user_001",
            user_name="张三",
            group_id="group_001",
            group_name="团队A",
            version="202501",
            is_latest=True,
            hard_skills=[{"value": "Python", "level": "高级", "evidences": ["ev1"]}],
            soft_skills=[{"value": "沟通", "level": "中级", "evidences": ["ev2"]}],
            output_reasoning="基于最近对话补充",
            personality=[
                {"value": "外向", "evidences": ["ev3"]},
                {"value": "乐观", "evidences": ["ev4"]},
            ],
            way_of_decision_making=[{"value": "数据驱动", "evidences": ["ev5"]}],
            projects_participated=[
                {
                    "project_id": "proj_001",
                    "project_name": "系统重构",
                    "subtasks": [{"value": "搭建基础框架", "evidences": ["ev6"]}],
                    "user_objective": [{"value": "完成上线", "evidences": ["ev7"]}],
                    "contributions": [{"value": "主导技术方案", "evidences": ["ev8"]}],
                    "user_concerns": [{"value": "进度风险", "evidences": ["ev9"]}],
                    "evidences": ["ev10"],
                }
            ],
            user_goal=[{"value": "提升团队效率", "evidences": ["ev11"]}],
            work_responsibility=[{"value": "负责后端架构设计", "evidences": ["ev12"]}],
            working_habit_preference=[{"value": "上午专注开发", "evidences": ["ev13"]}],
            interests=[
                {"value": "编程", "evidences": ["ev14"]},
                {"value": "阅读", "evidences": ["ev15"]},
            ],
            tendency=[{"value": "喜欢挑战", "evidences": ["ev16"]}],
            motivation_system=[{"value": "成就感", "evidences": ["ev17"]}],
            fear_system=[{"value": "停滞", "evidences": ["ev18"]}],
            value_system=[{"value": "协作", "evidences": ["ev19"]}],
            humor_use=[{"value": "冷幽默", "evidences": ["ev20"]}],
            colloquialism=[{"value": "没问题", "evidences": ["ev21"]}],
            group_importance_evidence={"score": 0.8, "evidences": ["ev22"]},
            age=25,
            gender="male",
            occupation="软件工程师",
            metadata=Metadata(
                source="test_source", user_id="user_001", memory_type="profile"
            ),
        )

        assert model.id == "profile_001"
        assert model.user_id == "user_001"
        assert model.user_name == "张三"
        assert model.group_id == "group_001"
        assert model.group_name == "团队A"
        assert model.version == "202501"
        assert model.is_latest is True
        assert model.hard_skills == [
            {"value": "Python", "level": "高级", "evidences": ["ev1"]}
        ]
        assert model.soft_skills == [
            {"value": "沟通", "level": "中级", "evidences": ["ev2"]}
        ]
        assert model.output_reasoning == "基于最近对话补充"
        assert model.personality == [
            {"value": "外向", "evidences": ["ev3"]},
            {"value": "乐观", "evidences": ["ev4"]},
        ]
        assert model.way_of_decision_making == [
            {"value": "数据驱动", "evidences": ["ev5"]}
        ]
        assert model.projects_participated == [
            {
                "project_id": "proj_001",
                "project_name": "系统重构",
                "subtasks": [{"value": "搭建基础框架", "evidences": ["ev6"]}],
                "user_objective": [{"value": "完成上线", "evidences": ["ev7"]}],
                "contributions": [{"value": "主导技术方案", "evidences": ["ev8"]}],
                "user_concerns": [{"value": "进度风险", "evidences": ["ev9"]}],
                "evidences": ["ev10"],
            }
        ]
        assert model.user_goal == [{"value": "提升团队效率", "evidences": ["ev11"]}]
        assert model.work_responsibility == [
            {"value": "负责后端架构设计", "evidences": ["ev12"]}
        ]
        assert model.working_habit_preference == [
            {"value": "上午专注开发", "evidences": ["ev13"]}
        ]
        assert model.interests == [
            {"value": "编程", "evidences": ["ev14"]},
            {"value": "阅读", "evidences": ["ev15"]},
        ]
        assert model.tendency == [{"value": "喜欢挑战", "evidences": ["ev16"]}]
        assert model.motivation_system == [{"value": "成就感", "evidences": ["ev17"]}]
        assert model.fear_system == [{"value": "停滞", "evidences": ["ev18"]}]
        assert model.value_system == [{"value": "协作", "evidences": ["ev19"]}]
        assert model.humor_use == [{"value": "冷幽默", "evidences": ["ev20"]}]
        assert model.colloquialism == [{"value": "没问题", "evidences": ["ev21"]}]
        assert model.group_importance_evidence == {"score": 0.8, "evidences": ["ev22"]}
        assert model.age == 25
        assert model.gender == "male"
        assert model.occupation == "软件工程师"
        assert model.metadata.source == "test_source"
        assert model.metadata.user_id == "user_001"
        assert model.metadata.memory_type == "profile"

    def test_profile_model_optional_fields(self):
        """测试用户画像模型可选字段"""
        model = ProfileModel(
            id="profile_001",
            user_id="user_001",
            metadata=Metadata(
                source="test_source", user_id="user_001", memory_type="profile"
            ),
        )

        assert model.user_name is None
        assert model.group_id is None
        assert model.group_name is None
        assert model.version is None
        assert model.is_latest is None
        assert model.hard_skills is None
        assert model.soft_skills is None
        assert model.output_reasoning is None
        assert model.personality is None
        assert model.way_of_decision_making is None
        assert model.projects_participated is None
        assert model.user_goal is None
        assert model.work_responsibility is None
        assert model.working_habit_preference is None
        assert model.interests is None
        assert model.tendency is None
        assert model.motivation_system is None
        assert model.fear_system is None
        assert model.value_system is None
        assert model.humor_use is None
        assert model.colloquialism is None
        assert model.group_importance_evidence is None
        assert model.age is None
        assert model.gender is None
        assert model.occupation is None
        assert model.metadata.source == "test_source"
        assert model.metadata.user_id == "user_001"
        assert model.metadata.memory_type == "profile"


class TestPreferenceModel:
    """测试用户偏好模型"""

    def test_create_preference_model(self):
        """测试创建用户偏好模型"""
        model = PreferenceModel(
            id="pref_001",
            user_id="user_001",
            category="食物偏好",
            preference_key="favorite_cuisine",
            preference_value="中式料理",
            confidence_score=0.9,
        )

        assert model.id == "pref_001"
        assert model.user_id == "user_001"
        assert model.category == "食物偏好"
        assert model.preference_key == "favorite_cuisine"
        assert model.preference_value == "中式料理"
        assert model.confidence_score == 0.9

    def test_preference_model_default_confidence(self):
        """测试用户偏好模型默认置信度"""
        model = PreferenceModel(
            id="pref_001",
            user_id="user_001",
            category="食物偏好",
            preference_key="favorite_cuisine",
            preference_value="中式料理",
        )

        assert model.confidence_score == 1.0


class TestEpisodicMemoryModel:
    """测试情景记忆模型"""

    def test_create_episodic_memory_model(self):
        """测试创建情景记忆模型"""
        now = datetime.now()
        model = EpisodicMemoryModel(
            id="episode_001",
            user_id="user_001",
            episode_id="ep_001",
            title="项目会议",
            summary="讨论项目进度",
            participants=["张三", "李四"],
            location="会议室A",
            start_time=now,
            key_events=["讨论需求", "分配任务"],
            emotions={"积极": 0.8},
        )

        assert model.id == "episode_001"
        assert model.user_id == "user_001"
        assert model.episode_id == "ep_001"
        assert model.title == "项目会议"
        assert model.summary == "讨论项目进度"
        assert model.participants == ["张三", "李四"]
        assert model.location == "会议室A"
        assert model.start_time == now
        assert model.key_events == ["讨论需求", "分配任务"]
        assert model.emotions == {"积极": 0.8}


class TestSemanticMemoryModel:
    """测试语义记忆模型"""

    def test_create_semantic_memory_model(self):
        """测试创建语义记忆模型"""
        model = SemanticMemoryModel(
            id="semantic_001",
            user_id="user_001",
            concept="机器学习",
            definition="让计算机从数据中学习的技术",
            category="技术概念",
            related_concepts=["人工智能", "深度学习"],
            source="学习笔记",
        )

        assert model.id == "semantic_001"
        assert model.user_id == "user_001"
        assert model.concept == "机器学习"
        assert model.definition == "让计算机从数据中学习的技术"
        assert model.category == "技术概念"
        assert model.related_concepts == ["人工智能", "深度学习"]
        assert model.source == "学习笔记"


class TestEntityModel:
    """测试实体模型"""

    def test_create_entity_model(self):
        """测试创建实体模型"""
        model = EntityModel(
            id="entity_001",
            user_id="user_001",
            entity_name="张三",
            entity_type="人物",
            description="同事",
            attributes={"部门": "技术部", "职位": "工程师"},
            aliases=["小张", "张工"],
        )

        assert model.id == "entity_001"
        assert model.user_id == "user_001"
        assert model.entity_name == "张三"
        assert model.entity_type == "人物"
        assert model.description == "同事"
        assert model.attributes == {"部门": "技术部", "职位": "工程师"}
        assert model.aliases == ["小张", "张工"]


class TestRelationModel:
    """测试关系模型"""

    def test_create_relation_model(self):
        """测试创建关系模型"""
        model = RelationModel(
            id="relation_001",
            user_id="user_001",
            source_entity_id="entity_001",
            target_entity_id="entity_002",
            relation_type="同事关系",
            relation_description="工作伙伴",
            strength=0.8,
        )

        assert model.id == "relation_001"
        assert model.user_id == "user_001"
        assert model.source_entity_id == "entity_001"
        assert model.target_entity_id == "entity_002"
        assert model.relation_type == "同事关系"
        assert model.relation_description == "工作伙伴"
        assert model.strength == 0.8


class TestBehaviorHistoryModel:
    """测试行为历史模型"""

    def test_create_behavior_history_model(self):
        """测试创建行为历史模型"""
        now = datetime.now()
        model = BehaviorHistoryModel(
            id="behavior_001",
            user_id="user_001",
            action_type="搜索",
            action_description="搜索技术文档",
            context={"query": "Python教程", "results": 10},
            result="成功",
            timestamp=now,
            session_id="session_001",
        )

        assert model.id == "behavior_001"
        assert model.user_id == "user_001"
        assert model.action_type == "搜索"
        assert model.action_description == "搜索技术文档"
        assert model.context == {"query": "Python教程", "results": 10}
        assert model.result == "成功"
        assert model.timestamp == now
        assert model.session_id == "session_001"


class TestMemoryQueryRequest:
    """测试记忆查询请求"""

    def test_create_memory_query_request(self):
        """测试创建记忆查询请求"""
        request = MemoryQueryRequest(
            user_id="user_001",
            memory_type=MemoryType.PROFILE,
            limit=20,
            offset=10,
            filters={"age": 25},
            sort_by="created_at",
            sort_order="asc",
        )

        assert request.user_id == "user_001"
        assert request.memory_type == MemoryType.PROFILE
        assert request.limit == 20
        assert request.offset == 10
        assert request.filters == {"age": 25}
        assert request.sort_by == "created_at"
        assert request.sort_order == "asc"

    def test_memory_query_request_defaults(self):
        """测试记忆查询请求默认值"""
        request = MemoryQueryRequest(user_id="user_001", memory_type=MemoryType.PROFILE)

        assert request.limit == 10
        assert request.offset == 0
        assert request.filters == {}
        assert request.sort_by is None
        assert request.sort_order == "desc"


class TestMemoryQueryResponse:
    """测试记忆查询响应"""

    def test_create_memory_query_response(self):
        """测试创建记忆查询响应"""
        memories = [
            BaseMemoryModel(
                id="base_001",
                user_id="user_001",
                content="测试内容",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        ]

        response = MemoryQueryResponse(
            memories=memories,
            total_count=100,
            has_more=True,
            metadata={"source": "test"},
        )

        assert response.memories == memories
        assert response.total_count == 100
        assert response.has_more is True
        assert response.metadata == {"source": "test"}

    def test_memory_query_response_defaults(self):
        """测试记忆查询响应默认值"""
        memories = []
        response = MemoryQueryResponse(memories=memories, total_count=0)

        assert response.has_more is False
        assert response.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__])
