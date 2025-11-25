"""
记忆获取服务

此模块提供记忆数据访问的服务层接口，对接访问 DB 的 repository 类文件。
提供基于 ID 的查询功能，支持各种记忆类型的检索。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union


from core.di import get_bean_by_type, get_bean, service
from infra_layer.adapters.out.persistence.document.memory.semantic_memory_record import (
    SemanticMemoryRecord,
    SemanticMemoryRecordProjection,
)
from infra_layer.adapters.out.persistence.repository.semantic_memory_raw_repository import (
    SemanticMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
    CoreMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.entity_raw_repository import (
    EntityRawRepository,
)
from infra_layer.adapters.out.persistence.repository.relationship_raw_repository import (
    RelationshipRawRepository,
)
from infra_layer.adapters.out.persistence.repository.behavior_history_raw_repository import (
    BehaviorHistoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
    ConversationMetaRawRepository,
)
from infra_layer.adapters.out.persistence.repository.event_log_record_raw_repository import (
    EventLogRecordRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
    EventLogRecord,
    EventLogRecordProjection,
)
from infra_layer.adapters.out.persistence.repository.semantic_memory_record_raw_repository import (
    SemanticMemoryRecordRawRepository,
)

from .dtos.memory_query import FetchMemResponse

from .memory_models import (
    MemoryType,
    MemoryModel,
    BaseMemoryModel,
    ProfileModel,
    PreferenceModel,
    EpisodicMemoryModel,
    SemanticMemoryModel,
    EntityModel,
    RelationModel,
    BehaviorHistoryModel,
    CoreMemoryModel,
    EventLogModel,
    SemanticMemoryRecordModel,
    Metadata,
)

logger = logging.getLogger(__name__)


class FetchMemoryServiceInterface(ABC):
    """记忆获取服务接口"""

    @abstractmethod
    async def find_by_user_id(
        self,
        user_id: str,
        memory_type: MemoryType,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        limit: int = 10,
    ) -> FetchMemResponse:
        """
        根据用户ID查找记忆

        Args:
            user_id: 用户ID
            memory_type: 记忆类型
            limit: 返回数量限制

        Returns:
            记忆查询响应
        """
        pass

    @abstractmethod
    async def find_by_id(
        self, memory_id: str, memory_type: MemoryType
    ) -> Optional[MemoryModel]:
        """
        根据记忆ID查找单个记忆

        Args:
            memory_id: 记忆ID
            memory_type: 记忆类型

        Returns:
            记忆模型，如果未找到则返回None
        """
        pass

    @abstractmethod
    async def find_episodic_by_event_id(
        self, event_id: str, user_id: str
    ) -> Optional[EpisodicMemoryModel]:
        """
        根据事件ID查找情景记忆

        Args:
            event_id: 事件ID
            user_id: 用户ID

        Returns:
            情景记忆模型，如果未找到则返回None
        """
        pass

    @abstractmethod
    async def find_entity_by_entity_id(self, entity_id: str) -> Optional[EntityModel]:
        """
        根据实体ID查找实体

        Args:
            entity_id: 实体ID

        Returns:
            实体模型，如果未找到则返回None
        """
        pass

    @abstractmethod
    async def find_relationship_by_entity_ids(
        self, source_entity_id: str, target_entity_id: str
    ) -> Optional[RelationModel]:
        """
        根据实体ID查找关系

        Args:
            source_entity_id: 源实体ID
            target_entity_id: 目标实体ID

        Returns:
            关系模型，如果未找到则返回None
        """
        pass


@service(name="fetch_memory_service", primary=True)
class FetchMemoryServiceImpl(FetchMemoryServiceInterface):
    """记忆获取服务的真实实现

    使用 DI 框架注入的 Repository 实例进行数据库访问。
    """

    def __init__(self):
        """初始化服务"""
        self._semantic_repo = None
        self._episodic_repo = None
        self._core_repo = None
        self._entity_repo = None
        self._relationship_repo = None
        self._behavior_repo = None
        self._conversation_meta_repo = None
        self._event_log_repo = None
        self._semantic_memory_record_repo = None
        logger.info("FetchMemoryServiceImpl initialized")

    def _get_repositories(self):
        """获取 Repository 实例"""
        if self._semantic_repo is None:
            self._semantic_repo = get_bean_by_type(SemanticMemoryRawRepository)
        if self._episodic_repo is None:
            self._episodic_repo = get_bean_by_type(EpisodicMemoryRawRepository)
        if self._core_repo is None:
            self._core_repo = get_bean_by_type(CoreMemoryRawRepository)
        if self._entity_repo is None:
            self._entity_repo = get_bean_by_type(EntityRawRepository)
        if self._relationship_repo is None:
            self._relationship_repo = get_bean_by_type(RelationshipRawRepository)
        if self._behavior_repo is None:
            self._behavior_repo = get_bean_by_type(BehaviorHistoryRawRepository)
        if self._conversation_meta_repo is None:
            self._conversation_meta_repo = get_bean_by_type(
                ConversationMetaRawRepository
            )
        if self._event_log_repo is None:
            self._event_log_repo = get_bean_by_type(EventLogRecordRawRepository)
        if self._semantic_memory_record_repo is None:
            self._semantic_memory_record_repo = get_bean_by_type(
                SemanticMemoryRecordRawRepository
            )

    async def _get_employee_metadata(
        self,
        user_id: str,
        source: str,
        memory_type: str,
        limit: int = None,
        group_id: str = None,
    ) -> Metadata:
        """
        根据用户ID获取用户信息并创建Metadata

        从 conversation-meta 的 user_details 中获取用户详情信息。
        employee_org 相关表已被移除，现在用户信息来自 conversation-meta。

        Args:
            user_id: 用户ID
            source: 数据来源
            memory_type: 记忆类型
            limit: 限制数量（可选）
            group_id: 群组ID（可选），如果提供则从 conversation-meta 获取用户详情

        Returns:
            Metadata 对象
        """
        try:
            # 如果提供了 group_id，尝试从 conversation_meta 获取用户详情
            if group_id:
                # 确保仓库已初始化
                if self._conversation_meta_repo is None:
                    self._get_repositories()

                # 查询对话元数据
                conversation_meta = await self._conversation_meta_repo.get_by_group_id(
                    group_id
                )

                # 如果找到对话元数据，并且包含该用户的详情
                if conversation_meta and conversation_meta.user_details:
                    user_detail = conversation_meta.user_details.get(user_id)
                    if user_detail:
                        # 从 user_details 中提取用户信息
                        # user_detail 是 UserDetailModel 对象，包含 full_name, role, extra
                        return Metadata(
                            source=source,
                            user_id=user_id,
                            memory_type=memory_type,
                            limit=limit,
                            full_name=user_detail.full_name,
                            # 如果 extra 中有 email 和 phone，也提取出来
                            email=(
                                user_detail.extra.get("email")
                                if user_detail.extra
                                else None
                            ),
                            phone=(
                                user_detail.extra.get("phone")
                                if user_detail.extra
                                else None
                            ),
                        )

            # 如果没有 group_id 或没有找到用户信息，返回基本信息
            return Metadata(
                source=source, user_id=user_id, memory_type=memory_type, limit=limit
            )

        except Exception as e:
            logger.warning(f"获取用户信息失败: {e}，使用基本信息创建Metadata")
            return Metadata(
                source=source, user_id=user_id, memory_type=memory_type, limit=limit
            )

    async def _convert_core_memory(self, core_memory) -> CoreMemoryModel:
        """转换核心记忆文档为模型"""
        metadata = await self._get_employee_metadata(
            user_id=core_memory.user_id,
            source="core_memory",
            memory_type=MemoryType.MULTIPLE.value,
        )

        return CoreMemoryModel(
            id=str(core_memory.id),
            user_id=core_memory.user_id,
            version=core_memory.version,
            is_latest=core_memory.is_latest,
            # BaseMemory 字段
            user_name=core_memory.user_name,
            gender=core_memory.gender,
            position=core_memory.position,
            supervisor_user_id=core_memory.supervisor_user_id,
            team_members=core_memory.team_members,
            okr=core_memory.okr,
            base_location=core_memory.base_location,
            hiredate=core_memory.hiredate,
            age=core_memory.age,
            department=core_memory.department,
            # Profile 字段
            hard_skills=core_memory.hard_skills,
            soft_skills=core_memory.soft_skills,
            output_reasoning=core_memory.output_reasoning,
            motivation_system=core_memory.motivation_system,
            fear_system=core_memory.fear_system,
            value_system=core_memory.value_system,
            humor_use=core_memory.humor_use,
            colloquialism=core_memory.colloquialism,
            personality=core_memory.personality,
            way_of_decision_making=core_memory.way_of_decision_making,
            projects_participated=core_memory.projects_participated,
            user_goal=core_memory.user_goal,
            work_responsibility=core_memory.work_responsibility,
            working_habit_preference=core_memory.working_habit_preference,
            interests=core_memory.interests,
            tendency=core_memory.tendency,
            # 通用字段
            extend=core_memory.extend,
            created_at=core_memory.created_at,
            updated_at=core_memory.updated_at,
            metadata=metadata,
        )

    async def _convert_semantic_memory(self, semantic_memory) -> SemanticMemoryModel:
        """转换语义记忆文档为模型"""
        metadata = await self._get_employee_metadata(
            user_id=semantic_memory.user_id,
            source="user_input",
            memory_type=MemoryType.SEMANTIC_MEMORY.value,
        )

        return SemanticMemoryModel(
            id=str(semantic_memory.id),
            user_id=semantic_memory.user_id,
            concept=semantic_memory.subject,
            definition=semantic_memory.description,
            category="语义记忆",
            related_concepts=[],
            confidence_score=1.0,
            source="用户输入",
            created_at=semantic_memory.created_at,
            updated_at=semantic_memory.updated_at,
            metadata=metadata,
        )

    def _convert_episodic_memory(self, episodic_memory) -> EpisodicMemoryModel:
        """转换情景记忆文档为模型"""
        return EpisodicMemoryModel(
            id=str(episodic_memory.id),
            user_id=episodic_memory.user_id,
            episode_id=str(episodic_memory.event_id),
            title=episodic_memory.subject,
            summary=episodic_memory.summary,
            participants=episodic_memory.participants or [],
            location=(
                episodic_memory.extend.get("location", "")
                if episodic_memory.extend
                else ""
            ),
            key_events=episodic_memory.keywords or [],
            created_at=episodic_memory.created_at,
            updated_at=episodic_memory.updated_at,
            metadata=Metadata(
                source="episodic_memory",
                user_id=episodic_memory.user_id,
                memory_type=MemoryType.EPISODIC_MEMORY.value,
            ),
        )

    def _convert_entity(self, entity) -> EntityModel:
        """转换实体文档为模型"""
        return EntityModel(
            id=str(entity.id),
            user_id="",  # 实体可能不属于特定用户
            entity_name=entity.name,
            entity_type=entity.type,
            description=f"实体名称: {entity.name} | 类型: {entity.type}",
            attributes={},
            aliases=entity.aliases or [],
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            metadata=Metadata(
                source="entity",
                user_id=entity.user_id,
                memory_type=MemoryType.ENTITY.value,
            ),
        )

    def _convert_relationship(self, relationship) -> RelationModel:
        """转换关系文档为模型"""
        return RelationModel(
            id=str(relationship.id),
            user_id="",  # 关系可能不属于特定用户
            source_entity_id=relationship.source_entity_id,
            target_entity_id=relationship.target_entity_id,
            relation_type=(
                relationship.relationship[0]["type"]
                if relationship.relationship
                else "未知关系"
            ),
            relation_description=(
                relationship.relationship[0]["content"]
                if relationship.relationship
                else ""
            ),
            strength=0.8,  # 默认强度
            created_at=relationship.created_at,
            updated_at=relationship.updated_at,
            metadata=Metadata(
                source="relationship",
                user_id=relationship.user_id,
                memory_type=MemoryType.RELATION.value,
            ),
        )

    def _convert_behavior_history(self, behavior) -> BehaviorHistoryModel:
        """转换行为历史文档为模型"""
        return BehaviorHistoryModel(
            id=str(behavior.id),
            user_id=behavior.user_id,
            action_type=(
                behavior.behavior_type[0] if behavior.behavior_type else "未知行为"
            ),
            action_description=f"行为类型: {behavior.behavior_type}",
            context=behavior.meta or {},
            result="成功",
            session_id=behavior.event_id,
            created_at=behavior.created_at,
            updated_at=behavior.updated_at,
            metadata=Metadata(
                source="behavior_history",
                user_id=behavior.user_id,
                memory_type=MemoryType.BEHAVIOR_HISTORY.value,
            ),
        )

    def _convert_event_log(
        self, event_log: Union[EventLogRecord, EventLogRecordProjection]
    ) -> EventLogModel:
        """转换事件日志文档为模型

        支持 EventLogRecord 和 EventLogRecordShort 类型。
        EventLogRecordShort 不包含 vector 字段。
        """
        return EventLogModel(
            id=str(event_log.id),
            user_id=event_log.user_id,
            atomic_fact=event_log.atomic_fact,
            parent_episode_id=event_log.parent_episode_id,
            timestamp=event_log.timestamp,
            user_name=event_log.user_name,
            group_id=event_log.group_id,
            group_name=event_log.group_name,
            participants=event_log.participants,
            vector=getattr(
                event_log, 'vector', None
            ),  # EventLogRecordShort 没有 vector 字段
            vector_model=event_log.vector_model,
            event_type=event_log.event_type,
            extend=event_log.extend,
            created_at=event_log.created_at,
            updated_at=event_log.updated_at,
            metadata=Metadata(
                source="event_log",
                user_id=event_log.user_id,
                memory_type=MemoryType.PERSONAL_EVENT_LOG.value,
            ),
        )

    def _convert_semantic_memory_record(
        self,
        semantic_record: Union[SemanticMemoryRecord, SemanticMemoryRecordProjection],
    ) -> SemanticMemoryRecordModel:
        """转换语义记忆记录文档为模型

        支持 SemanticMemoryRecord 和 SemanticMemoryRecordShort 类型。
        SemanticMemoryRecordShort 不包含 vector 字段。
        """
        return SemanticMemoryRecordModel(
            id=str(semantic_record.id),
            content=semantic_record.content,
            parent_episode_id=semantic_record.parent_episode_id,
            user_id=semantic_record.user_id,
            user_name=semantic_record.user_name,
            group_id=semantic_record.group_id,
            group_name=semantic_record.group_name,
            start_time=semantic_record.start_time,
            end_time=semantic_record.end_time,
            duration_days=semantic_record.duration_days,
            participants=semantic_record.participants,
            vector=getattr(
                semantic_record, 'vector', None
            ),  # SemanticMemoryRecordShort 没有 vector 字段
            vector_model=semantic_record.vector_model,
            evidence=semantic_record.evidence,
            extend=semantic_record.extend,
            created_at=semantic_record.created_at,
            updated_at=semantic_record.updated_at,
            metadata=Metadata(
                source="semantic_memory_record",
                user_id=semantic_record.user_id or "",
                memory_type=MemoryType.PERSONAL_SEMANTIC_MEMORY.value,
            ),
        )

    async def find_by_user_id(
        self,
        user_id: str,
        memory_type: MemoryType,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        limit: int = 10,
    ) -> FetchMemResponse:
        """
        根据用户ID查找记忆

        Args:
            user_id: 用户ID
            memory_type: 记忆类型
            version_range: 版本范围 (start, end)，左闭右闭区间 [start, end]。
                          如果不传或为None，则获取最新版本（按version倒序）
            limit: 返回数量限制

        Returns:
            记忆查询响应
        """
        logger.debug(
            f"Fetching {memory_type} memories for user {user_id}, limit: {limit}"
        )

        try:
            self._get_repositories()
            memories = []

            match memory_type:
                case MemoryType.MULTIPLE:
                    # 多类型查询：获取core_memory并转换为CoreMemoryModel
                    core_memory_result = await self._core_repo.get_by_user_id(
                        user_id, version_range=version_range
                    )
                    if core_memory_result:
                        # 如果version_range为None，core_memory_result是单个CoreMemory
                        # 如果version_range不为None，core_memory_result是List[CoreMemory]
                        if isinstance(core_memory_result, list):
                            memories = [
                                await self._convert_core_memory(core_memory)
                                for core_memory in core_memory_result
                            ]
                        else:
                            memories = [
                                await self._convert_core_memory(core_memory_result)
                            ]
                    else:
                        memories = []
                case MemoryType.SEMANTIC_MEMORY:
                    # 语义记忆：每个用户只有一个语义记忆文档
                    semantic_memory = await self._semantic_repo.get_by_user_id(user_id)
                    if semantic_memory:
                        memories = [
                            await self._convert_semantic_memory(semantic_memory)
                        ]
                    else:
                        memories = []

                case MemoryType.EPISODIC_MEMORY:
                    # 情景记忆：按时间排序的事件列表
                    episodic_memories = await self._episodic_repo.get_by_user_id(
                        user_id, limit=limit
                    )
                    memories = [
                        self._convert_episodic_memory(mem) for mem in episodic_memories
                    ]

                case MemoryType.BASE_MEMORY:
                    # 基础记忆：从核心记忆中提取基础信息
                    core_memory = await self._core_repo.get_by_user_id(user_id)
                    if core_memory:
                        base_info = self._core_repo.get_base(core_memory)
                        memories = [
                            BaseMemoryModel(
                                id=str(core_memory.id),
                                user_id=core_memory.user_id,
                                content=f"用户: {base_info.get('user_name', '未知')} | 职位: {base_info.get('position', '未知')} | 部门: {base_info.get('department', '未知')}",
                                created_at=core_memory.created_at,
                                updated_at=core_memory.updated_at,
                                metadata={
                                    "user_name": base_info.get('user_name', ''),
                                    "position": base_info.get('position', ''),
                                    "department": base_info.get('department', ''),
                                    "company": base_info.get('company', ''),
                                    "location": base_info.get('location', ''),
                                    "contact": base_info.get('contact', {}),
                                },
                            )
                        ]
                    else:
                        memories = []

                case MemoryType.PROFILE:
                    # 个人档案：从核心记忆中提取个人特征
                    core_memory = await self._core_repo.get_by_user_id(user_id)
                    if core_memory:
                        profile_info = self._core_repo.get_profile(core_memory)
                        memories = [
                            ProfileModel(
                                id=str(core_memory.id),
                                user_id=core_memory.user_id,
                                name=profile_info.get('personality', '未知'),
                                age=profile_info.get('age', 0),
                                gender=profile_info.get('gender', ''),
                                occupation=profile_info.get('occupation', ''),
                                interests=profile_info.get('interests', []),
                                personality_traits=profile_info.get(
                                    'personality_traits', {}
                                ),
                                created_at=core_memory.created_at,
                                updated_at=core_memory.updated_at,
                                metadata=profile_info,
                            )
                        ]
                    else:
                        memories = []

                case MemoryType.PREFERENCE:
                    # 个人偏好：从核心记忆中提取偏好设置
                    core_memory = await self._core_repo.get_by_user_id(user_id)
                    if core_memory:
                        preference_info = self._core_repo.get_preference(core_memory)
                        # 将偏好信息转换为多个 PreferenceModel
                        memories = []
                        for key, value in preference_info.items():
                            memories.append(
                                PreferenceModel(
                                    id=f"{core_memory.id}_{key}",
                                    user_id=core_memory.user_id,
                                    category="个人偏好",
                                    preference_key=key,
                                    preference_value=str(value),
                                    confidence_score=1.0,
                                    created_at=core_memory.created_at,
                                    updated_at=core_memory.updated_at,
                                    metadata={
                                        "source": "core_memory",
                                        "original_key": key,
                                    },
                                )
                            )
                    else:
                        memories = []

                case MemoryType.ENTITY:
                    # 实体：查询与用户相关的实体
                    entities = await self._entity_repo.get_by_type(
                        "Person", limit=limit
                    )
                    memories = [self._convert_entity(entity) for entity in entities]

                case MemoryType.RELATION:
                    # 关系：查询人际关系
                    relationships = (
                        await self._relationship_repo.get_by_relationship_type(
                            "人际关系", limit=limit
                        )
                    )
                    memories = [
                        self._convert_relationship(rel) for rel in relationships
                    ]

                case MemoryType.BEHAVIOR_HISTORY:
                    # 行为历史：按时间排序的用户行为
                    behaviors = await self._behavior_repo.get_by_user_id(
                        user_id, limit=limit
                    )
                    memories = [
                        self._convert_behavior_history(behavior)
                        for behavior in behaviors
                    ]

                case MemoryType.PERSONAL_EVENT_LOG:
                    # 事件日志：原子事实列表
                    event_logs = await self._event_log_repo.get_by_user_id(
                        user_id, limit=limit, model=EventLogRecordProjection
                    )
                    memories = [
                        self._convert_event_log(event_log) for event_log in event_logs
                    ]

                case MemoryType.PERSONAL_SEMANTIC_MEMORY:
                    # 个人语义记忆：从情景记忆中提取的语义信息
                    semantic_records = (
                        await self._semantic_memory_record_repo.get_by_user_id(
                            user_id, model=SemanticMemoryRecordProjection, limit=limit
                        )
                    )
                    memories = [
                        self._convert_semantic_memory_record(record)
                        for record in semantic_records
                    ]

            # 创建包含员工信息的metadata
            response_metadata = await self._get_employee_metadata(
                user_id=user_id,
                source="real_service",
                memory_type=memory_type.value,
                limit=limit,
            )
            print(f"---- response_metadata: {response_metadata}")

            return FetchMemResponse(
                memories=memories,
                total_count=len(memories),
                has_more=len(memories) == limit,
                metadata=response_metadata,
            )

        except Exception as e:
            logger.error(f"Error fetching memories for user {user_id}: {e}")
            # 即使出错也尝试获取员工信息
            try:
                error_metadata = await self._get_employee_metadata(
                    user_id=user_id,
                    source="real_service",
                    memory_type=memory_type.value,
                    limit=limit,
                )
            except:
                error_metadata = Metadata(
                    source="real_service",
                    user_id=user_id,
                    memory_type=memory_type.value,
                    limit=limit,
                )

            return FetchMemResponse(
                memories=[], total_count=0, has_more=False, metadata=error_metadata
            )

    async def find_by_id(
        self, memory_id: str, memory_type: MemoryType
    ) -> Optional[MemoryModel]:
        """
        根据记忆ID查找单个记忆

        Args:
            memory_id: 记忆ID
            memory_type: 记忆类型

        Returns:
            记忆模型，如果未找到则返回None
        """
        logger.debug(f"Fetching {memory_type} memory by ID: {memory_id}")

        try:
            self._get_repositories()

            match memory_type:
                case MemoryType.SEMANTIC_MEMORY:
                    # 语义记忆通过用户ID查询
                    semantic_memory = await self._semantic_repo.get_by_user_id(
                        memory_id
                    )
                    if semantic_memory:
                        return self._convert_semantic_memory(semantic_memory)

                case MemoryType.EPISODIC_MEMORY:
                    # 情景记忆通过事件ID查询，需要用户ID
                    # 这里假设memory_id是event_id，需要额外的用户ID参数
                    logger.warning(
                        "Episodic memory query by ID requires user_id, use find_episodic_by_event_id instead"
                    )
                    return None

                case MemoryType.ENTITY:
                    entity = await self._entity_repo.get_by_entity_id(memory_id)
                    if entity:
                        return self._convert_entity(entity)

                case MemoryType.RELATION:
                    # 关系查询需要源实体ID和目标实体ID
                    logger.warning(
                        "Relation query by ID requires source and target entity IDs, use find_relationship_by_entity_ids instead"
                    )
                    return None

                case MemoryType.BEHAVIOR_HISTORY:
                    # 行为历史通过用户ID查询
                    behaviors = await self._behavior_repo.get_by_user_id(
                        memory_id, limit=1
                    )
                    if behaviors:
                        return self._convert_behavior_history(behaviors[0])

                case MemoryType.PERSONAL_EVENT_LOG:
                    # 事件日志通过ID查询（使用简化版本减少数据传输）
                    event_log = await self._event_log_repo.get_by_id(
                        memory_id, model=EventLogRecordProjection
                    )
                    if event_log:
                        return self._convert_event_log(event_log)

                case MemoryType.PERSONAL_SEMANTIC_MEMORY:
                    # 个人语义记忆通过ID查询
                    semantic_record = await self._semantic_memory_record_repo.get_by_id(
                        memory_id
                    )
                    if semantic_record:
                        return self._convert_semantic_memory_record(semantic_record)

            return None

        except Exception as e:
            logger.error(f"Error fetching memory by ID {memory_id}: {e}")
            return None

    async def find_episodic_by_event_id(
        self, event_id: str, user_id: str
    ) -> Optional[EpisodicMemoryModel]:
        """
        根据事件ID查找情景记忆

        Args:
            event_id: 事件ID
            user_id: 用户ID

        Returns:
            情景记忆模型，如果未找到则返回None
        """
        logger.debug(
            f"Fetching episodic memory by event_id: {event_id}, user_id: {user_id}"
        )

        try:
            self._get_repositories()
            episodic_memory = await self._episodic_repo.get_by_event_id(
                event_id, user_id
            )
            if episodic_memory:
                return self._convert_episodic_memory(episodic_memory)
            return None

        except Exception as e:
            logger.error(f"Error fetching episodic memory by event_id {event_id}: {e}")
            return None

    async def find_entity_by_entity_id(self, entity_id: str) -> Optional[EntityModel]:
        """
        根据实体ID查找实体

        Args:
            entity_id: 实体ID

        Returns:
            实体模型，如果未找到则返回None
        """
        logger.debug(f"Fetching entity by entity_id: {entity_id}")

        try:
            self._get_repositories()
            entity = await self._entity_repo.get_by_entity_id(entity_id)
            if entity:
                return self._convert_entity(entity)
            return None

        except Exception as e:
            logger.error(f"Error fetching entity by entity_id {entity_id}: {e}")
            return None

    async def find_relationship_by_entity_ids(
        self, source_entity_id: str, target_entity_id: str
    ) -> Optional[RelationModel]:
        """
        根据实体ID查找关系

        Args:
            source_entity_id: 源实体ID
            target_entity_id: 目标实体ID

        Returns:
            关系模型，如果未找到则返回None
        """
        logger.debug(
            f"Fetching relationship by entity_ids: {source_entity_id} -> {target_entity_id}"
        )

        try:
            self._get_repositories()
            relationship = await self._relationship_repo.get_by_entity_ids(
                source_entity_id, target_entity_id
            )
            if relationship:
                return self._convert_relationship(relationship)
            return None

        except Exception as e:
            logger.error(
                f"Error fetching relationship by entity_ids {source_entity_id} -> {target_entity_id}: {e}"
            )
            return None


def get_fetch_memory_service() -> FetchMemoryServiceInterface:
    """获取记忆获取服务实例

    通过依赖注入框架获取服务实例，支持单例模式。
    """
    return get_bean("fetch_memory_service")
