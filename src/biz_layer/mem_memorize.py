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
from infra_layer.adapters.out.persistence.repository.personal_semantic_memory_raw_repository import (
    PersonalSemanticMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.personal_event_log_raw_repository import (
    PersonalEventLogRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
    ConversationStatusRawRepository,
)
from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
    CoreMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.repository.group_user_profile_memory_raw_repository import (
    GroupUserProfileMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.group_profile_raw_repository import (
    GroupProfileRawRepository,
)
from biz_layer.conversation_data_repo import ConversationDataRepository
from memory_layer.types import RawDataType
from typing import List, Dict, Optional
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
from infra_layer.adapters.out.search.elasticsearch.converter.personal_semantic_memory_converter import (
    PersonalSemanticMemoryConverter,
)
from infra_layer.adapters.out.search.milvus.converter.personal_semantic_memory_milvus_converter import (
    PersonalSemanticMemoryMilvusConverter,
)
from infra_layer.adapters.out.search.elasticsearch.converter.personal_event_log_converter import (
    PersonalEventLogConverter,
)
from infra_layer.adapters.out.search.milvus.converter.personal_event_log_milvus_converter import (
    PersonalEventLogMilvusConverter,
)
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from biz_layer.memcell_milvus_sync import MemCellMilvusSyncService

logger = get_logger(__name__)


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
    _get_user_organization,
    _save_profile_memory_to_group_user_profile_memory,
    _convert_document_to_group_importance_evidence,
    _get_raw_data_by_time_range,
    _normalize_datetime_for_storage,
    _convert_projects_participated_list,
    _convert_group_profile_raw_to_memory_format,
)


def if_memorize(memcells: List[MemCell]) -> bool:
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

    # 从 Redis 读取所有历史消息
    redis_key = f"chat_history:{request.group_id}"

    try:
        redis_provider = get_bean_by_type(RedisProvider)
        messages_json = await redis_provider.lrange(redis_key, 0, -1)

        logger.info(f"[preprocess] 从 Redis 读取 {len(messages_json)} 条历史消息")

        # 将 JSON 字符串转换为 RawData 对象
        history_raw_data_list = []
        for msg_json in messages_json:
            msg_dict = json.loads(msg_json)

            # 提取消息时间
            create_time_str = msg_dict.get('create_time')
            if create_time_str:
                message_time = from_iso_format(create_time_str)
            else:
                message_time = current_time

            # 构造 RawData
            raw_data = RawData(
                content={
                    "speaker_id": msg_dict.get("sender"),
                    "speaker_name": msg_dict.get("sender_name", msg_dict.get("sender")),
                    "content": msg_dict.get("content", ""),
                    "timestamp": message_time,
                },
                data_id=str(msg_dict.get("message_id")),
                data_type=RawDataType.CONVERSATION,
            )
            history_raw_data_list.append(raw_data)

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
    memcells: List[MemCell],
    current_time: datetime,
    data_type: RawDataType,
):
    if data_type == RawDataType.CONVERSATION:
        # 更新状态表中的last_memcell_time至memcells最后一个时间戳
        try:
            status_repo = get_bean_by_type(ConversationStatusRawRepository)

            # 获取MemCell的时间戳
            memcell_time = None
            if memcells and hasattr(memcells[-1], 'timestamp'):
                memcell_time = memcells[-1].timestamp
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


async def save_memories(
    memory_list: List[Memory], current_time: datetime, version: Optional[str] = None
):
    logger.info(f"[mem_memorize] 保存 {len(memory_list)} 个记忆到数据库")
    # 初始化Repository实例
    episodic_memory_repo = get_bean_by_type(EpisodicMemoryRawRepository)
    group_user_profile_memory_repo = get_bean_by_type(
        GroupUserProfileMemoryRawRepository
    )
    group_profile_raw_repo = get_bean_by_type(GroupProfileRawRepository)
    episodic_memory_milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    es_repo = get_bean_by_type(EpisodicMemoryEsRepository)

    # 按对象类型分类保存
    episode_memories = [
        m
        for m in memory_list
        if isinstance(m, Memory)
        and hasattr(m, 'memory_type')
        and m.memory_type == MemoryType.EPISODE_SUMMARY
    ]
    semantic_memories = [m for m in memory_list if isinstance(m, SemanticMemoryItem)]
    event_logs = [m for m in memory_list if isinstance(m, EventLog)]
    profile_memories = [
        m
        for m in memory_list
        if isinstance(m, Memory)
        and hasattr(m, 'memory_type')
        and m.memory_type == MemoryType.PROFILE
    ]
    group_profile_memories = [
        m
        for m in memory_list
        if isinstance(m, Memory)
        and hasattr(m, 'memory_type')
        and m.memory_type == MemoryType.GROUP_PROFILE
    ]

    # 保存个人 episode 到 MongoDB/ES/Milvus
    for episode_mem in episode_memories:
        # 转换为EpisodicMemory文档格式
        doc = _convert_episode_memory_to_doc(episode_mem, current_time)
        doc = await episodic_memory_repo.append_episodic_memory(doc)
        episode_mem.event_id = str(doc.event_id)

        # 保存到 ES
        es_doc = EpisodicMemoryConverter.from_mongo(doc)
        es_doc.type = "personal_episode"  # 标记为个人episode
        await es_doc.save()

        # 保存到 Milvus（添加缺失的字段）
        milvus_entity = EpisodicMemoryMilvusConverter.from_mongo(doc)
        vector = (
            milvus_entity.get("vector") if isinstance(milvus_entity, dict) else None
        )

        if not vector or (isinstance(vector, list) and len(vector) == 0):
            logger.warning(
                "[mem_memorize] 跳过写入Milvus：向量为空或缺失，event_id=%s",
                getattr(doc, 'event_id', None),
            )
        else:
            # 所有通过 save_memories 保存的 episode 都是从 EpisodeMemoryExtractor 提取的个人视角 episode
            # 群组 episode 是存储在 MemCell.episode 中，由 memcell_milvus_sync.py 同步到 Milvus
            milvus_entity["memory_sub_type"] = "personal_episode"
            milvus_entity["start_time"] = 0
            milvus_entity["end_time"] = 0
            # 字段名修正：旧 schema 是 detail → 新 schema 是 metadata
            if "detail" in milvus_entity:
                milvus_entity["metadata"] = milvus_entity.pop("detail")
            else:
                milvus_entity["metadata"] = "{}"
            # 确保 search_content 字段存在
            if "search_content" not in milvus_entity:
                milvus_entity["search_content"] = milvus_entity.get("episode", "")[:500]

            await episodic_memory_milvus_repo.insert(milvus_entity)
            logger.debug(
                f"✅ 保存 personal_episode: user_id={doc.user_id}, event_id={episode_mem.event_id}"
            )

        logger.debug(f"✅ 保存 episode_memory: {episode_mem.event_id}")

    # 保存Profile记忆到CoreMemoryRawRepository
    for profile_mem in profile_memories:
        try:
            await _save_profile_memory_to_group_user_profile_memory(
                profile_mem, group_user_profile_memory_repo, version
            )
        except Exception as e:
            logger.error(f"保存Profile记忆失败: {e}")

    for group_profile_mem in group_profile_memories:
        try:
            await _save_group_profile_memory(
                group_profile_mem, group_profile_raw_repo, version
            )
        except Exception as e:
            logger.error(f"保存Group Profile记忆失败: {e}")

    # 保存个人语义记忆到 MongoDB/Milvus/ES
    semantic_memory_repo = get_bean_by_type(PersonalSemanticMemoryRawRepository)

    for sem_mem in semantic_memories:
        if not sem_mem.content or not sem_mem.embedding:
            continue

        parent_event_id = sem_mem.parent_event_id
        parent_doc = await episodic_memory_repo.get_by_event_id(
            str(parent_event_id), sem_mem.user_id
        )
        if not parent_doc:
            continue

        # 转换为 PersonalSemanticMemory 文档格式并保存到 MongoDB
        doc = _convert_semantic_memory_to_doc(sem_mem, parent_doc, current_time)
        doc = await semantic_memory_repo.save(doc)

        # 保存到 ES
        es_doc = PersonalSemanticMemoryConverter.from_mongo(doc)
        await es_doc.save()

        # 保存到 Milvus
        milvus_entity = PersonalSemanticMemoryMilvusConverter.from_mongo(doc)
        vector = (
            milvus_entity.get("vector") if isinstance(milvus_entity, dict) else None
        )

        if not vector or (isinstance(vector, list) and len(vector) == 0):
            logger.warning(
                "[mem_memorize] 跳过写入Milvus：个人语义记忆向量为空，id=%s",
                getattr(doc, 'id', None),
            )
        else:
            await episodic_memory_milvus_repo.insert(milvus_entity)
            logger.debug(
                f"✅ 保存 personal_semantic_memory 到 MongoDB/ES/Milvus: {doc.id}"
            )

    # 保存个人事件日志到 MongoDB/Milvus/ES
    event_log_repo = get_bean_by_type(PersonalEventLogRawRepository)

    for event_log in event_logs:
        if not event_log.atomic_fact or not event_log.fact_embeddings:
            continue

        parent_event_id = event_log.parent_event_id
        parent_doc = await episodic_memory_repo.get_by_event_id(
            str(parent_event_id), event_log.user_id
        )
        if not parent_doc:
            continue

        # 转换为 PersonalEventLog 文档格式列表并保存到 MongoDB
        docs = _convert_event_log_to_docs(event_log, parent_doc, current_time)

        for doc in docs:
            # 保存到 MongoDB
            doc = await event_log_repo.save(doc)

            # 保存到 ES
            es_doc = PersonalEventLogConverter.from_mongo(doc)
            await es_doc.save()

            # 保存到 Milvus
            milvus_entity = PersonalEventLogMilvusConverter.from_mongo(doc)
            vector = (
                milvus_entity.get("vector") if isinstance(milvus_entity, dict) else None
            )

            if not vector or (isinstance(vector, list) and len(vector) == 0):
                logger.warning(
                    "[mem_memorize] 跳过写入Milvus：个人事件日志向量为空，id=%s",
                    getattr(doc, 'id', None),
                )
            else:
                await episodic_memory_milvus_repo.insert(milvus_entity)

        logger.debug(f"✅ 保存 personal_event_log 到 MongoDB/ES/Milvus: {len(docs)} 条")

    # 刷新 Milvus，确保数据立即可搜索
    if semantic_memories or event_logs or episode_memories:
        await episodic_memory_milvus_repo.flush()
        logger.info("[mem_memorize] Milvus 已刷新，数据立即可搜索")

    logger.info(f"[mem_memorize] 保存完成:")
    logger.info(f"  - EPISODE_SUMMARY: {len(episode_memories)} 个")
    logger.info(f"  - SEMANTIC_SUMMARY: {len(semantic_memories)} 个")
    logger.info(f"  - EVENT_LOG: {len(event_logs)} 个")
    logger.info(f"  - PROFILE: {len(profile_memories)} 个")
    logger.info(f"  - GROUP_PROFILE: {len(group_profile_memories)} 个")


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

    # logger.info(f"[mem_memorize] request: {request}")

    # logger.info(f"[mem_memorize] memorize request: {request}")
    logger.info(f"[mem_memorize] request.current_time: {request.current_time}")
    # 获取当前时间，用于所有时间相关操作
    if request.current_time:
        current_time = request.current_time
    else:
        current_time = get_now_with_timezone() + timedelta(seconds=1)
    logger.info(f"[mem_memorize] 当前时间: {current_time}")

    memory_manager = MemoryManager()

    # 定义需要提取的记忆类型：先提取个人 episode，再基于 episode 提取语义记忆和事件日志
    memory_types = [
        MemoryType.EPISODE_SUMMARY,
        MemoryType.SEMANTIC_SUMMARY,
        MemoryType.EVENT_LOG,
    ]
    if request.raw_data_type == RawDataType.CONVERSATION:
        request = await preprocess_conv_request(request, current_time)
        if request == None:
            return None

    if request.raw_data_type == RawDataType.CONVERSATION:
        # async with distributed_lock(f"memcell_extract_{request.group_id}") as acquired:
        #     # 120s等待，获取不到
        #     if not acquired:
        #         logger.warning(f"[mem_memorize] 获取分布式锁失败: {request.group_id}")
        now = time.time()

        # 添加详细调试日志
        logger.info(f"=" * 80)
        logger.info(f"[边界检测] 开始检测: group_id={request.group_id}")
        logger.info(f"[边界检测] 历史消息: {len(request.history_raw_data_list)} 条")
        logger.info(f"[边界检测] 新消息: {len(request.new_raw_data_list)} 条")
        if request.history_raw_data_list:
            logger.info(
                f"[边界检测] 历史消息范围: {request.history_raw_data_list[0].content.get('timestamp')} ~ {request.history_raw_data_list[-1].content.get('timestamp')}"
            )
        if request.new_raw_data_list:
            for idx, raw in enumerate(request.new_raw_data_list):
                logger.info(
                    f"[边界检测] 新消息[{idx}]: {raw.content.get('speaker_id')} - {raw.content.get('content')[:50]}... @ {raw.content.get('timestamp')}"
                )
        logger.info(f"=" * 80)

        logger.debug(
            f"[memorize memorize] 提取MemCell开始: group_id={request.group_id}, group_name={request.group_name}, "
            f"semantic_extraction={request.enable_semantic_extraction}"
        )
        memcell_result = await memory_manager.extract_memcell(
            request.history_raw_data_list,
            request.new_raw_data_list,
            request.raw_data_type,
            request.group_id,
            request.group_name,
            request.user_id_list,
            enable_semantic_extraction=request.enable_semantic_extraction,
            enable_event_log_extraction=request.enable_event_log_extraction,
        )
        logger.debug(f"[memorize memorize] 提取MemCell耗时: {time.time() - now}秒")
    else:
        now = time.time()
        logger.debug(
            f"[memorize memorize] 提取MemCell开始: group_id={request.group_id}, group_name={request.group_name}, "
            f"semantic_extraction={request.enable_semantic_extraction}, "
            f"event_log_extraction={request.enable_event_log_extraction}"
        )
        memcell_result = await memory_manager.extract_memcell(
            request.history_raw_data_list,
            request.new_raw_data_list,
            request.raw_data_type,
            request.group_id,
            request.group_name,
            request.user_id_list,
            enable_semantic_extraction=request.enable_semantic_extraction,
            enable_event_log_extraction=request.enable_event_log_extraction,
        )
        logger.debug(f"[memorize memorize] 提取MemCell耗时: {time.time() - now}秒")

    if memcell_result == None:
        logger.warning(f"[mem_memorize] 跳过提取MemCell")
        return None

    logger.debug(f"[mem_memorize] memcell_result: {memcell_result}")
    memcell, status_result = memcell_result

    # 添加边界检测结果日志
    logger.info(f"=" * 80)
    logger.info(f"[边界检测结果] memcell is None: {memcell is None}")
    logger.info(
        f"[边界检测结果] should_wait: {status_result.should_wait if status_result else 'N/A'}"
    )
    if memcell is None:
        logger.info(
            f"[边界检测结果] 判断: {'需要等待更多消息' if status_result.should_wait else '非边界，继续累积'}"
        )
    else:
        logger.info(f"[边界检测结果] 判断: 是边界！成功提取MemCell")
        logger.info(f"[边界检测结果] MemCell event_id: {memcell.event_id}")
        logger.info(
            f"[边界检测结果] Episode: {memcell.episode[:100] if memcell.episode else 'None'}..."
        )
    logger.info(f"=" * 80)

    if memcell == None:
        await update_status_when_no_memcell(
            request, status_result, current_time, request.raw_data_type
        )
        logger.warning(f"[mem_memorize] 跳过提取MemCell")
        return None
    else:
        logger.info(f"[mem_memorize] 成功提取MemCell")

        # 判断为边界，清空 Redis 历史消息（重新开始累积）
        redis_key = f"chat_history:{request.group_id}"
        try:
            redis_provider = get_bean_by_type(RedisProvider)
            await redis_provider.delete(redis_key)
            logger.info(f"[mem_memorize] 判断为边界，已清空 Redis 历史: {redis_key}")
        except Exception as e:
            logger.error(f"[mem_memorize] 清空 Redis 失败: {e}")
            traceback.print_exc()

    # TODO: 读状态表，读取累积的MemCell数据表，判断是否要做memorize计算

    # MemCell存表
    memcell = await _save_memcell_to_database(memcell, current_time)

    # 同步 MemCell 到 Milvus 和 ES（包括 episode/semantic_memories/event_log）
    memcell_repo = get_bean_by_type(MemCellRawRepository)
    doc_memcell = await memcell_repo.get_by_event_id(str(memcell.event_id))

    if doc_memcell:
        sync_service = get_bean_by_type(MemCellMilvusSyncService)
        sync_stats = await sync_service.sync_memcell(
            doc_memcell, sync_to_es=True, sync_to_milvus=True
        )
        logger.info(
            f"[mem_memorize] MemCell 同步到 Milvus/ES 完成: {memcell.event_id}, "
            f"stats={sync_stats}"
        )
    else:
        logger.warning(f"[mem_memorize] 无法加载 MemCell 进行同步: {memcell.event_id}")

    # print_memory = random.random() < 0.1

    logger.info(f"[mem_memorize] 成功保存MemCell: {memcell.event_id}")

    # if print_memory:
    #     logger.info(f"[mem_memorize] 打印MemCell: {memcell}")

    memcells = [memcell]

    # 同步触发聚类（等待完成，确保 Profile 提取成功）
    if request.group_id:
        await _trigger_clustering(request.group_id, memcell, request.scene)

    # 读取记忆的流程
    participants = []
    for memcell in memcells:
        if memcell.participants:
            participants.extend(memcell.participants)

    if if_memorize(memcells):
        # 加锁
        # 使用真实Repository读取用户数据
        old_memory_list = await load_core_memories(request, participants, current_time)

        # 提取记忆
        memory_list = []
        episode_memories = []

        # 第一阶段：提取个人 episode
        for memory_type in memory_types:
            if memory_type == MemoryType.EPISODE_SUMMARY:
                extracted_memories = await memory_manager.extract_memory(
                    memcell_list=memcells,
                    memory_type=memory_type,
                    user_ids=participants,
                    group_id=request.group_id,
                    group_name=request.group_name,
                    old_memory_list=old_memory_list,
                )
                if extracted_memories:
                    episode_memories = extracted_memories
                    memory_list += extracted_memories

        # 保存 episode 记忆到数据库
        if episode_memories:
            await save_memories(episode_memories, current_time)

        # 第二阶段：基于已保存的 episode 提取语义记忆和事件日志
        for memory_type in memory_types:
            if memory_type in [MemoryType.SEMANTIC_SUMMARY, MemoryType.EVENT_LOG]:
                for episode_mem in episode_memories:
                    extracted_memories = await memory_manager.extract_memory(
                        memcell_list=[],
                        memory_type=memory_type,
                        user_ids=[episode_mem.user_id],
                        episode_memory=episode_mem,
                    )
                    if extracted_memories:
                        # 为提取的记忆添加元信息
                        if isinstance(extracted_memories, list):
                            for mem in extracted_memories:
                                mem.parent_event_id = episode_mem.event_id
                                mem.user_id = episode_mem.user_id
                                mem.group_id = episode_mem.group_id
                            memory_list += extracted_memories
                        else:
                            # EventLog 类型
                            extracted_memories.parent_event_id = episode_mem.event_id
                            extracted_memories.user_id = episode_mem.user_id
                            extracted_memories.group_id = episode_mem.group_id
                            memory_list.append(extracted_memories)

        # 保存语义记忆和事件日志
        semantic_and_eventlog = [m for m in memory_list if m not in episode_memories]
        if semantic_and_eventlog:
            await save_memories(semantic_and_eventlog, current_time)

        await update_status_after_memcell(
            request, memcells, current_time, request.raw_data_type
        )

        # TODO: 实际项目中应该加锁避免并发问题
        # 释放锁
        return memory_list
    else:
        return None


def get_version_from_request(request: MemorizeOfflineRequest) -> str:
    # 1. 获取 memorize_to 日期
    target_date = request.memorize_to

    # 2. 倒退一天
    previous_day = target_date - timedelta(days=1)

    # 3. 格式化为 "YYYY-MM" 字符串
    return previous_day.strftime("%Y-%m")
