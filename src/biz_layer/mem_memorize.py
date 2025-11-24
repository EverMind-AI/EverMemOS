

from dataclasses import dataclass
import random
import time
import json
import traceback
from memory_layer.memory_manager import MemorizeRequest, MemorizeOfflineRequest
from memory_layer.memory_manager import MemoryManager
from memory_layer.types import (
    MemoryType,
    MemCell,
    Memory,
    RawDataType,
    SemanticMemoryItem,
)
from memory_layer.memory_extractor.event_log_extractor import EventLog
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from infra_layer.adapters.out.persistence.document.memory.memcell import DataTypeEnum
from memory_layer.memory_extractor.profile_memory_extractor import (
    ProfileMemory,
    ProfileMemoryExtractor,
    ProfileMemoryExtractRequest,
    ProfileMemoryMerger,
    ProjectInfo,
)
from memory_layer.memory_extractor.group_profile_memory_extractor import (
    GroupProfileMemoryExtractor,
    GroupProfileMemoryExtractRequest,
    GroupProfileMemory,
)
from core.di import get_bean_by_type
from component.redis_provider import RedisProvider
from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.semantic_memory_record_repository import (
    SemanticMemoryRecordRawRepository,
)
from infra_layer.adapters.out.persistence.repository.event_log_record_repository import (
    EventLogRecordRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
    ConversationStatusRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
    ConversationMetaRawRepository,
)
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
    CoreMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.group_user_profile_memory_raw_repository import (
    GroupUserProfileMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.group_profile_raw_repository import (
    GroupProfileRawRepository,
)
from biz_layer.conversation_data_repo import ConversationDataRepository
from memory_layer.types import RawDataType
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import uuid
from datetime import datetime, timedelta
import os
import asyncio
from collections import defaultdict
from common_utils.datetime_utils import (
    get_now_with_timezone,
    to_iso_format,
    from_iso_format,
)
from memory_layer.memcell_extractor.base_memcell_extractor import StatusResult
import traceback

from core.lock.redis_distributed_lock import distributed_lock
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.elasticsearch.converter.episodic_memory_converter import (
    EpisodicMemoryConverter,
)
from infra_layer.adapters.out.search.milvus.converter.episodic_memory_milvus_converter import (
    EpisodicMemoryMilvusConverter,
)
from infra_layer.adapters.out.search.elasticsearch.converter.semantic_memory_converter import (
    SemanticMemoryConverter,
)
from infra_layer.adapters.out.search.milvus.converter.semantic_memory_milvus_converter import (
    SemanticMemoryMilvusConverter,
)
from infra_layer.adapters.out.search.elasticsearch.converter.event_log_converter import (
    EventLogConverter,
)
from infra_layer.adapters.out.search.milvus.converter.event_log_milvus_converter import (
    EventLogMilvusConverter,
)
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from infra_layer.adapters.out.search.repository.semantic_memory_milvus_repository import (
    SemanticMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.event_log_milvus_repository import (
    EventLogMilvusRepository,
)
from biz_layer.mem_sync import MemorySyncService

logger = get_logger(__name__)

@dataclass
class MemoryDocPayload:
    memory_type: MemoryType
    doc: Any


def _clone_semantic_memory_item(raw_item: Any) -> Optional[SemanticMemoryItem]:
    """将任意结构的语义记忆条目转换为 SemanticMemoryItem 实例"""
    if raw_item is None:
        return None

    if isinstance(raw_item, SemanticMemoryItem):
        return SemanticMemoryItem(
            content=raw_item.content,
            evidence=getattr(raw_item, "evidence", None),
            start_time=getattr(raw_item, "start_time", None),
            end_time=getattr(raw_item, "end_time", None),
            duration_days=getattr(raw_item, "duration_days", None),
            source_episode_id=getattr(raw_item, "source_episode_id", None),
            embedding=getattr(raw_item, "embedding", None),
        )

    if isinstance(raw_item, dict):
        return SemanticMemoryItem(
            content=raw_item.get("content", ""),
            evidence=raw_item.get("evidence"),
            start_time=raw_item.get("start_time"),
            end_time=raw_item.get("end_time"),
            duration_days=raw_item.get("duration_days"),
            source_episode_id=raw_item.get("source_episode_id"),
            embedding=raw_item.get("embedding"),
        )

    return None


def _clone_event_log(raw_event_log: Any) -> Optional[EventLog]:
    """将任意结构的事件日志转换为 EventLog 实例"""
    if raw_event_log is None:
        return None

    if isinstance(raw_event_log, EventLog):
        return EventLog(
            time=getattr(raw_event_log, "time", ""),
            atomic_fact=list(getattr(raw_event_log, "atomic_fact", []) or []),
            fact_embeddings=getattr(raw_event_log, "fact_embeddings", None),
        )

    if isinstance(raw_event_log, dict):
        return EventLog.from_dict(raw_event_log)

    return None

async def _trigger_clustering(
    group_id: str, memcell: MemCell, scene: Optional[str] = None
) -> None:
    """异步触发 MemCell 聚类（后台任务，不阻塞主流程）

    Args:
        group_id: 群组ID
        memcell: 刚保存的 MemCell
        scene: 对话场景（用于决定 Profile 提取策略）
            - None/"work"/"company" 等：使用 group_chat 场景
            - "assistant"/"companion" 等：使用 assistant 场景
    """
    logger.info(
        f"[聚类] 开始触发聚类: group_id={group_id}, event_id={memcell.event_id}, scene={scene}"
    )

    try:
        from memory_layer.cluster_manager import (
            ClusterManager,
            ClusterManagerConfig,
            MongoClusterStorage,
        )
        from memory_layer.profile_manager import (
            ProfileManager,
            ProfileManagerConfig,
            MongoProfileStorage,
        )
        from memory_layer.llm.llm_provider import LLMProvider
        from core.di import get_bean_by_type
        import os

        logger.info(f"[聚类] 正在获取 MongoClusterStorage...")
        # 获取 MongoDB 存储
        mongo_storage = get_bean_by_type(MongoClusterStorage)
        logger.info(f"[聚类] MongoClusterStorage 获取成功: {type(mongo_storage)}")

        # 创建 ClusterManager（使用 MongoDB 存储）
        config = ClusterManagerConfig(
            similarity_threshold=0.65,
            max_time_gap_days=7,
            enable_persistence=False,  # MongoDB 不需要文件持久化
        )
        cluster_manager = ClusterManager(config=config, storage=mongo_storage)
        logger.info(f"[聚类] ClusterManager 创建成功")

        # 创建 ProfileManager 并连接到 ClusterManager
        # 获取 MongoDB Profile 存储
        profile_storage = get_bean_by_type(MongoProfileStorage)
        logger.info(f"[聚类] MongoProfileStorage 获取成功: {type(profile_storage)}")

        llm_provider = LLMProvider(
            provider_type=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-4"),
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
        )

        # 根据 scene 决定 Profile 提取场景
        # assistant/companion -> assistant 场景（提取兴趣、偏好、生活习惯）
        # 其他 -> group_chat 场景（提取工作角色、技能、项目经验）
        profile_scenario = (
            "assistant"
            if scene and scene.lower() in ["assistant", "companion"]
            else "group_chat"
        )

        profile_config = ProfileManagerConfig(
            scenario=profile_scenario,
            min_confidence=0.6,
            enable_versioning=True,
            auto_extract=True,
        )

        profile_manager = ProfileManager(
            llm_provider=llm_provider,
            config=profile_config,
            storage=profile_storage,  # 使用 MongoDB 存储
            group_id=group_id,
            group_name=None,  # 可以从 memcell 中获取
        )

        # 连接 ProfileManager 到 ClusterManager
        profile_manager.attach_to_cluster_manager(cluster_manager)
        logger.info(
            f"[聚类] ProfileManager 已连接到 ClusterManager (场景: {profile_scenario}, 使用 MongoDB 存储)"
        )
        print(
            f"[聚类] ProfileManager 已连接，阈值: {profile_manager._min_memcells_threshold}"
        )

        # 将 MemCell 转换为聚类所需的字典格式
        memcell_dict = {
            "event_id": str(memcell.event_id),
            "episode": memcell.episode,
            "timestamp": memcell.timestamp.timestamp() if memcell.timestamp else None,
            "participants": memcell.participants or [],
            "group_id": group_id,
        }

        logger.info(f"[聚类] 开始执行聚类: {memcell_dict['event_id']}")
        print(f"[聚类] 开始执行聚类: event_id={memcell_dict['event_id']}")

        # 执行聚类（会自动触发 ProfileManager 的回调）
        cluster_id = await cluster_manager.cluster_memcell(
            group_id=group_id, memcell=memcell_dict
        )

        print(f"[聚类] 聚类完成: cluster_id={cluster_id}")

        if cluster_id:
            logger.info(
                f"[聚类] ✅ MemCell {memcell.event_id} -> Cluster {cluster_id} (group: {group_id})"
            )
            print(f"[聚类] ✅ MemCell {memcell.event_id} -> Cluster {cluster_id}")
        else:
            logger.warning(
                f"[聚类] ⚠️ MemCell {memcell.event_id} 聚类返回 None (group: {group_id})"
            )
            print(f"[聚类] ⚠️ 聚类返回 None")

    except Exception as e:
        # 聚类失败，打印详细错误信息并重新抛出
        import traceback

        error_msg = f"[聚类] ❌ 触发聚类失败: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg)  # 确保在控制台能看到
        print(traceback.format_exc())
        raise  # 重新抛出异常，让调用者知道失败了


def _convert_data_type_to_raw_data_type(data_type) -> RawDataType:
    """
    将不同的数据类型枚举转换为统一的RawDataType

    Args:
        data_type: 可能是DataTypeEnum、RawDataType或字符串

    Returns:
        RawDataType: 转换后的统一数据类型
    """
    if isinstance(data_type, RawDataType):
        return data_type

    # 获取字符串值
    if hasattr(data_type, 'value'):
        type_str = data_type.value
    else:
        type_str = str(data_type)

    # 映射转换
    type_mapping = {
        "Conversation": RawDataType.CONVERSATION,
        "CONVERSATION": RawDataType.CONVERSATION,
        # 其他类型映射到CONVERSATION作为默认值
    }

    return type_mapping.get(type_str, RawDataType.CONVERSATION)


from biz_layer.mem_db_operations import (
    _convert_timestamp_to_time,
    _convert_episode_memory_to_doc,
    _convert_semantic_memory_to_doc,
    _convert_event_log_to_docs,
    _save_memcell_to_database,
    _save_profile_memory_to_core,
    ConversationStatus,
    _update_status_for_new_conversation,
    _update_status_for_continuing_conversation,
    _update_status_after_memcell_extraction,
    _convert_original_data_for_profile_extractor,
    _save_group_profile_memory,
    _save_profile_memory_to_group_user_profile_memory,
    _convert_document_to_group_importance_evidence,
    _normalize_datetime_for_storage,
    _convert_projects_participated_list,
    _convert_group_profile_raw_to_memory_format,
)


def if_memorize(memcell: MemCell) -> bool:
    return True


def extract_message_time(raw_data):
    """
    从RawData对象中提取消息时间

    Args:
        raw_data: RawData对象

    Returns:
        datetime: 消息时间，如果无法提取则返回None
    """
    # 优先从timestamp字段获取
    if hasattr(raw_data, 'timestamp') and raw_data.timestamp:
        try:
            return _normalize_datetime_for_storage(raw_data.timestamp)
        except Exception as e:
            logger.debug(f"Failed to parse timestamp from raw_data.timestamp: {e}")
            pass

    # 从extend字段获取
    if (
        hasattr(raw_data, 'extend')
        and raw_data.extend
        and isinstance(raw_data.extend, dict)
    ):
        timestamp_val = raw_data.extend.get('timestamp')
        if timestamp_val:
            try:
                return _normalize_datetime_for_storage(timestamp_val)
            except Exception as e:
                logger.debug(f"Failed to parse timestamp from extend field: {e}")
                pass

    return None


from core.observation.tracing.decorators import trace_logger


@trace_logger(operation_name="mem_memorize preprocess_conv_request", log_level="info")
async def preprocess_conv_request(
    request: MemorizeRequest, current_time: datetime
) -> MemorizeRequest:
    """
    简化版的请求预处理：
    1. 从 Redis 读取所有历史消息
    2. 将历史消息作为 history_raw_data_list
    3. 将当前新消息作为 new_raw_data_list
    4. 边界检测由后续逻辑处理（检测后会清空或保留 Redis）
    """

    logger.info(f"[preprocess] 开始处理: group_id={request.group_id}")

    # 检查是否有新数据
    if not request.new_raw_data_list:
        logger.info("[preprocess] 没有新数据，跳过处理")
        return None

    # 使用 conversation_data_repo 进行先取后存操作
    conversation_data_repo = get_bean_by_type(ConversationDataRepository)

    try:
        # 第一步：先从 conversation_data_repo 获取历史消息
        # 这里不限制时间范围，获取最近1000条历史消息（由缓存管理器的max_length控制）
        history_raw_data_list = await conversation_data_repo.get_conversation_data(
            group_id=request.group_id, start_time=None, end_time=None, limit=1000
        )

        logger.info(
            f"[preprocess] 从 conversation_data_repo 读取 {len(history_raw_data_list)} 条历史消息"
        )

        # 第二步：保存新消息到 conversation_data_repo
        save_success = await conversation_data_repo.save_conversation_data(
            request.new_raw_data_list, request.group_id
        )

        if save_success:
            logger.info(
                f"[preprocess] 成功保存 {len(request.new_raw_data_list)} 条新消息"
            )
        else:
            logger.warning(f"[preprocess] 保存新消息失败")

        # 更新 request
        request.history_raw_data_list = history_raw_data_list
        # new_raw_data_list 保持不变（就是新传入的消息）

        logger.info(
            f"[preprocess] 完成: 历史 {len(history_raw_data_list)} 条, 新消息 {len(request.new_raw_data_list)} 条"
        )

        return request

    except Exception as e:
        logger.error(f"[preprocess] Redis 读取失败: {e}")
        traceback.print_exc()
        # Redis 失败时，使用原始 request
        return request


async def update_status_when_no_memcell(
    request: MemorizeRequest,
    status_result: StatusResult,
    current_time: datetime,
    data_type: RawDataType,
):
    if data_type == RawDataType.CONVERSATION:
        # 尝试更新状态表
        try:
            status_repo = get_bean_by_type(ConversationStatusRawRepository)

            if status_result.should_wait:
                logger.info(f"[mem_memorize] 判断为无法判断边界继续等待，不更新状态表")
                return
            else:
                logger.info(f"[mem_memorize] 判断为非边界，继续累积msg，更新状态表")
                # 获取最新消息时间戳
                latest_time = _convert_timestamp_to_time(current_time, current_time)
                if request.new_raw_data_list:
                    last_msg = request.new_raw_data_list[-1]
                    if hasattr(last_msg, 'content') and isinstance(
                        last_msg.content, dict
                    ):
                        latest_time = last_msg.content.get('timestamp', latest_time)
                    elif hasattr(last_msg, 'timestamp'):
                        latest_time = last_msg.timestamp

                if not latest_time:
                    latest_time = min(latest_time, current_time)

                # 使用封装函数更新对话延续状态
                await _update_status_for_continuing_conversation(
                    status_repo, request, latest_time, current_time
                )

        except Exception as e:
            logger.error(f"更新状态表失败: {e}")
    else:
        pass


async def update_status_after_memcell(
    request: MemorizeRequest,
    memcell: MemCell,
    current_time: datetime,
    data_type: RawDataType,
):
    if data_type == RawDataType.CONVERSATION:
        # 更新状态表中的last_memcell_time至memcell的时间戳
        try:
            status_repo = get_bean_by_type(ConversationStatusRawRepository)

            # 获取MemCell的时间戳
            memcell_time = None
            if memcell and hasattr(memcell, 'timestamp'):
                memcell_time = memcell.timestamp
            else:
                memcell_time = current_time

            # 使用封装函数更新MemCell提取后的状态
            await _update_status_after_memcell_extraction(
                status_repo, request, memcell_time, current_time
            )

            logger.info(f"[mem_memorize] 记忆提取完成，状态表已更新")

        except Exception as e:
            logger.error(f"最终状态表更新失败: {e}")
    else:
        pass


async def save_personal_profile_memory(
    profile_memories: List[ProfileMemory], version: Optional[str] = None
):
    logger.info(f"[mem_memorize] 保存 {len(profile_memories)} 个个人档案记忆到数据库")
    # 初始化Repository实例
    core_memory_repo = get_bean_by_type(CoreMemoryRawRepository)

    # 保存个人档案记忆到GroupUserProfileMemoryRawRepository
    for profile_mem in profile_memories:
        await _save_profile_memory_to_core(profile_mem, core_memory_repo, version)
        # 移除单个操作成功日志


async def save_memory_docs(
    doc_payloads: List[MemoryDocPayload], version: Optional[str] = None
) -> Dict[MemoryType, List[Any]]:
    """
    通用 Doc 保存函数，按 MemoryType 枚举自动保存并同步
    """

    grouped_docs: Dict[MemoryType, List[Any]] = defaultdict(list)
    for payload in doc_payloads:
        if payload and payload.doc:
            grouped_docs[payload.memory_type].append(payload.doc)

    saved_result: Dict[MemoryType, List[Any]] = {}

    # Episodic
    episodic_docs = grouped_docs.get(MemoryType.EPISODIC_MEMORY, [])
    if episodic_docs:
        episodic_repo = get_bean_by_type(EpisodicMemoryRawRepository)
        episodic_milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
        saved_episodic: List[Any] = []

        for doc in episodic_docs:
            saved_doc = await episodic_repo.append_episodic_memory(doc)
            saved_episodic.append(saved_doc)

            es_doc = EpisodicMemoryConverter.from_mongo(saved_doc)
            await es_doc.save()

            milvus_entity = EpisodicMemoryMilvusConverter.from_mongo(saved_doc)
            vector = (
                milvus_entity.get("vector") if isinstance(milvus_entity, dict) else None
            )
            if vector and len(vector) > 0:
                await episodic_milvus_repo.insert(milvus_entity, flush=False)
            else:
                logger.warning(
                    "[mem_memorize] 跳过写入Milvus：向量为空或缺失，event_id=%s",
                    getattr(saved_doc, "event_id", None),
                )

    
        saved_result[MemoryType.EPISODIC_MEMORY] = saved_episodic

    # Semantic
    semantic_docs = grouped_docs.get(MemoryType.SEMANTIC_MEMORY, [])
    if semantic_docs:
        semantic_repo = get_bean_by_type(SemanticMemoryRecordRawRepository)
        saved_semantic = await semantic_repo.create_batch(semantic_docs)
        saved_result[MemoryType.SEMANTIC_MEMORY] = saved_semantic

        sync_service = get_bean_by_type(MemorySyncService)
        await sync_service.sync_batch_semantic_memories(
            saved_semantic, sync_to_es=True, sync_to_milvus=True
        )

    # Event Log
    event_log_docs = grouped_docs.get(MemoryType.PERSONAL_EVENT_LOG, [])
    if event_log_docs:
        event_log_repo = get_bean_by_type(EventLogRecordRawRepository)
        saved_event_logs = await event_log_repo.create_batch(event_log_docs)
        saved_result[MemoryType.PERSONAL_EVENT_LOG] = saved_event_logs

        sync_service = get_bean_by_type(MemorySyncService)
        await sync_service.sync_batch_event_logs(
            saved_event_logs, sync_to_es=True, sync_to_milvus=True
        )

    # Profile
    profile_docs = grouped_docs.get(MemoryType.PROFILE, [])
    if profile_docs:
        group_user_profile_repo = get_bean_by_type(
            GroupUserProfileMemoryRawRepository
        )
        saved_profiles = []
        for profile_mem in profile_docs:
            try:
                await _save_profile_memory_to_group_user_profile_memory(
                    profile_mem, group_user_profile_repo, version
                )
                saved_profiles.append(profile_mem)
            except Exception as exc:
                logger.error(f"保存Profile记忆失败: {exc}")
        if saved_profiles:
            saved_result[MemoryType.PROFILE] = saved_profiles

    group_profile_docs = grouped_docs.get(MemoryType.GROUP_PROFILE, [])
    if group_profile_docs:
        group_profile_repo = get_bean_by_type(GroupProfileRawRepository)
        saved_group_profiles = []
        for mem in group_profile_docs:
            try:
                await _save_group_profile_memory(mem, group_profile_repo, version)
                saved_group_profiles.append(mem)
            except Exception as exc:
                logger.error(f"保存Group Profile记忆失败: {exc}")
        if saved_group_profiles:
            saved_result[MemoryType.GROUP_PROFILE] = saved_group_profiles

    return saved_result


async def load_core_memories(
    request: MemorizeRequest, participants: List[str], current_time: datetime
):
    logger.info(f"[mem_memorize] 读取用户数据: {participants}")
    # 初始化Repository实例
    core_memory_repo = get_bean_by_type(CoreMemoryRawRepository)

    # 读取用户CoreMemory数据
    user_core_memories = {}
    for user_id in participants:
        try:
            core_memory = await core_memory_repo.get_by_user_id(user_id)
            if core_memory:
                user_core_memories[user_id] = core_memory
            # 移除单个用户的成功/失败日志
        except Exception as e:
            logger.error(f"获取用户 {user_id} CoreMemory失败: {e}")

    logger.info(f"[mem_memorize] 获取到 {len(user_core_memories)} 个用户CoreMemory")

    # 直接从CoreMemory转换为ProfileMemory对象列表
    old_memory_list = []
    if user_core_memories:
        for user_id, core_memory in user_core_memories.items():
            if core_memory:
                # 直接创建ProfileMemory对象
                profile_memory = ProfileMemory(
                    # Memory 基类必需字段
                    memory_type=MemoryType.CORE,
                    user_id=user_id,
                    timestamp=to_iso_format(current_time),
                    ori_event_id_list=[],
                    # Memory 基类可选字段
                    subject=f"{getattr(core_memory, 'user_name', user_id)}的个人档案",
                    summary=f"用户{user_id}的基本信息：{getattr(core_memory, 'position', '未知角色')}",
                    group_id=request.group_id,
                    participants=[user_id],
                    type=RawDataType.CONVERSATION,
                    # ProfileMemory 特有字段 - 直接使用原始字典格式
                    hard_skills=getattr(core_memory, 'hard_skills', None),
                    soft_skills=getattr(core_memory, 'soft_skills', None),
                    output_reasoning=getattr(core_memory, 'output_reasoning', None),
                    motivation_system=getattr(core_memory, 'motivation_system', None),
                    fear_system=getattr(core_memory, 'fear_system', None),
                    value_system=getattr(core_memory, 'value_system', None),
                    humor_use=getattr(core_memory, 'humor_use', None),
                    colloquialism=getattr(core_memory, 'colloquialism', None),
                    projects_participated=_convert_projects_participated_list(
                        getattr(core_memory, 'projects_participated', None)
                    ),
                )
                old_memory_list.append(profile_memory)

        logger.info(
            f"[mem_memorize] 直接转换了 {len(old_memory_list)} 个CoreMemory为ProfileMemory"
        )
    else:
        logger.info(f"[mem_memorize] 没有用户CoreMemory数据，old_memory_list为空")


async def memorize(request: MemorizeRequest) -> List[Memory]:
    """
    记忆提取主流程 (全局队列版)
    
    流程:
    1. 提取 MemCell
    2. 保存 MemCell 到数据库
    3. 提交到全局队列由 Worker 异步处理
    4. 立即返回，不等待后续处理完成
    """
    logger.info(f"[mem_memorize] request.current_time: {request.current_time}")
    
    # 获取当前时间
    if request.current_time:
        current_time = request.current_time
    else:
        current_time = get_now_with_timezone() + timedelta(seconds=1)
    logger.info(f"[mem_memorize] 当前时间: {current_time}")

    memory_manager = MemoryManager()
    
    # ===== MemCell 提取阶段 =====
    if request.raw_data_type == RawDataType.CONVERSATION:
        request = await preprocess_conv_request(request, current_time)
        if request == None:
            logger.warning(f"[mem_memorize] preprocess_conv_request 返回 None")
            return None

    # 边界检测
    now = time.time()
    logger.info("=" * 80)
    logger.info(f"[边界检测] 开始检测: group_id={request.group_id}")
    logger.info(f"[边界检测] 历史消息: {len(request.history_raw_data_list)} 条")
    logger.info(f"[边界检测] 新消息: {len(request.new_raw_data_list)} 条")
    logger.info("=" * 80)

    memcell_result = await memory_manager.extract_memcell(
        request.history_raw_data_list,
        request.new_raw_data_list,
        request.raw_data_type,
        request.group_id,
        request.group_name,
        request.user_id_list,
    )
    logger.debug(f"[mem_memorize] 提取 MemCell 耗时: {time.time() - now}秒")

    if memcell_result == None:
        logger.warning(f"[mem_memorize] 跳过提取MemCell")
        return None

    memcell, status_result = memcell_result

    # 检查边界检测结果
    logger.info("=" * 80)
    logger.info(f"[边界检测结果] memcell is None: {memcell is None}")
    if memcell is None:
        logger.info(
            f"[边界检测结果] 判断: {'需要等待更多消息' if status_result.should_wait else '非边界，继续累积'}"
        )
    else:
        logger.info(f"[边界检测结果] 判断: 是边界！event_id={memcell.event_id}")
    logger.info("=" * 80)

    if memcell == None:
        await update_status_when_no_memcell(
            request, status_result, current_time, request.raw_data_type
        )
        logger.warning(f"[mem_memorize] 未检测到边界，返回")
        return None

    # 清空对话历史（判断为边界）
    logger.info(f"[mem_memorize] 成功提取 MemCell，清空对话历史")
    try:
        conversation_data_repo = get_bean_by_type(ConversationDataRepository)
        delete_success = await conversation_data_repo.delete_conversation_data(request.group_id)
        if delete_success:
            logger.info(f"[mem_memorize] 已清空对话历史: group_id={request.group_id}")
        else:
            logger.warning(f"[mem_memorize] 清空对话历史失败: group_id={request.group_id}")
    except Exception as e:
        logger.error(f"[mem_memorize] 清空对话历史异常: {e}")

    # 保存 MemCell 到数据库
    memcell = await _save_memcell_to_database(memcell, current_time)
    logger.info(f"[mem_memorize] 成功保存 MemCell: {memcell.event_id}")

    # 🔥 提交到全局 Worker 队列，异步处理
    from biz_layer.memorize_worker_service import MemorizeWorkerService
    
    try:
        worker_service = await MemorizeWorkerService.get_instance()
        await worker_service.submit_memcell(
            memcell=memcell,
            request=request,
            current_time=current_time,
        )
        logger.info(f"[mem_memorize] ✅ MemCell 已提交到 Worker 队列，立即返回")
    except Exception as e:
        logger.error(f"[mem_memorize] ❌ 提交到 Worker 队列失败: {e}")
        traceback.print_exc()
    
    # 立即返回空列表（记忆将异步保存到数据库）
    return []


def get_version_from_request(request: MemorizeOfflineRequest) -> str:
    # 1. 获取 memorize_to 日期
    target_date = request.memorize_to

    # 2. 倒退一天
    previous_day = target_date - timedelta(days=1)

    # 3. 格式化为 "YYYY-MM" 字符串
    return previous_day.strftime("%Y-%m")
