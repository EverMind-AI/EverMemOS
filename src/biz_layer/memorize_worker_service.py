"""
记忆提取 Worker 服务

提供全局队列和常驻 Worker，用于异步处理 MemCell 的记忆提取任务。
"""

import asyncio
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime

from memory_layer.memory_manager import MemorizeRequest, MemoryManager
from memory_layer.types import (
    MemoryType,
    MemCell,
    Memory,
    SemanticMemoryItem,
)
from memory_layer.memory_extractor.event_log_extractor import EventLog
from core.observation.logger import get_logger
from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
    ConversationMetaRawRepository,
)
from biz_layer.mem_db_operations import (
    _convert_episode_memory_to_doc,
    _convert_semantic_memory_to_doc,
    _convert_event_log_to_docs,
)
from biz_layer.mem_memorize import (
    MemoryDocPayload,
    save_memory_docs,
    load_core_memories,
    update_status_after_memcell,
    if_memorize,
    _trigger_clustering,
)

logger = get_logger(__name__)


class MemorizeWorkerService:
    """
    记忆提取 Worker 服务
    
    管理全局队列和常驻 Worker，提供异步的 MemCell 处理能力。
    """
    
    _instance: Optional['MemorizeWorkerService'] = None
    _lock = asyncio.Lock()
    
    def __init__(self, num_workers: int = 3):
        """
        初始化 Worker 服务
        
        Args:
            num_workers: Worker 数量，默认 3 个
        """
        self.num_workers = num_workers
        self.memcell_queue: asyncio.Queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"[MemorizeWorkerService] 初始化，Worker 数量: {num_workers}")
    
    @classmethod
    async def get_instance(cls, num_workers: int = 3) -> 'MemorizeWorkerService':
        """
        获取单例实例（线程安全）
        
        Args:
            num_workers: Worker 数量
            
        Returns:
            MemorizeWorkerService 实例
        """
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(num_workers)
        return cls._instance
    
    async def start(self):
        """启动所有 Worker"""
        if self._running:
            logger.warning("[MemorizeWorkerService] Worker 已经在运行中")
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        logger.info(f"[MemorizeWorkerService] 启动 {self.num_workers} 个 Worker")
        
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(worker_id=i))
            self.workers.append(worker)
            logger.info(f"[MemorizeWorkerService] Worker-{i} 已启动")
    
    async def stop(self, timeout: float = 30.0):
        """
        停止所有 Worker
        
        Args:
            timeout: 等待超时时间（秒）
        """
        if not self._running:
            logger.warning("[MemorizeWorkerService] Worker 未运行")
            return
        
        logger.info(f"[MemorizeWorkerService] 停止 Worker...")
        self._running = False
        
        # 发送结束信号
        for _ in range(self.num_workers):
            await self.memcell_queue.put(None)
        
        # 设置 shutdown 事件
        self._shutdown_event.set()
        
        # 等待所有 Worker 完成
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.workers, return_exceptions=True),
                timeout=timeout
            )
            logger.info("[MemorizeWorkerService] 所有 Worker 已停止")
        except asyncio.TimeoutError:
            logger.warning(f"[MemorizeWorkerService] Worker 停止超时（{timeout}秒），强制取消")
            for worker in self.workers:
                if not worker.done():
                    worker.cancel()
        
        self.workers.clear()
    
    async def submit_memcell(
        self,
        memcell: MemCell,
        request: MemorizeRequest,
        current_time: datetime,
    ):
        """
        提交 MemCell 到队列进行异步处理
        
        Args:
            memcell: 要处理的 MemCell
            request: 原始请求
            current_time: 当前时间
        """
        if not self._running:
            logger.error("[MemorizeWorkerService] Worker 未运行，无法提交任务")
            return
        
        # 将任务放入队列
        task_data = {
            'memcell': memcell,
            'request': request,
            'current_time': current_time,
        }
        await self.memcell_queue.put(task_data)
        logger.info(f"[MemorizeWorkerService] MemCell 已提交到队列: {memcell.event_id}")
    
    async def _worker(self, worker_id: int):
        """
        Worker 主循环
        
        Args:
            worker_id: Worker ID
        """
        logger.info(f"[Worker-{worker_id}] 启动")
        memory_manager = MemoryManager()
        
        while self._running or not self.memcell_queue.empty():
            try:
                # 从队列获取任务（带超时，避免永久阻塞）
                try:
                    task_data = await asyncio.wait_for(
                        self.memcell_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 收到结束信号
                if task_data is None:
                    logger.info(f"[Worker-{worker_id}] 收到结束信号")
                    self.memcell_queue.task_done()
                    break
                
                memcell = task_data['memcell']
                request = task_data['request']
                current_time = task_data['current_time']
                
                logger.info(f"[Worker-{worker_id}] 开始处理 MemCell: {memcell.event_id}")
                
                # 处理 MemCell
                await self._process_memcell(
                    memcell=memcell,
                    request=request,
                    memory_manager=memory_manager,
                    current_time=current_time,
                    worker_id=worker_id,
                )
                
                logger.info(f"[Worker-{worker_id}] 完成处理 MemCell: {memcell.event_id}")
                self.memcell_queue.task_done()
                
            except Exception as e:
                logger.error(f"[Worker-{worker_id}] 处理任务异常: {e}")
                traceback.print_exc()
                try:
                    self.memcell_queue.task_done()
                except ValueError:
                    pass
        
        logger.info(f"[Worker-{worker_id}] 退出")
    
    async def _process_memcell(
        self,
        memcell: MemCell,
        request: MemorizeRequest,
        memory_manager: MemoryManager,
        current_time: datetime,
        worker_id: int,
    ):
        """
        处理单个 MemCell 的所有记忆提取任务
        
        包括:
        1. 提取群组和个人 Episode
        2. 更新 MemCell 的 episode 字段到数据库
        3. 触发聚类 (异步，不等待)
        4. 提取 Semantic 和 EventLog
        5. 保存所有记忆到数据库
        """
        # ===== 第一步：获取场景信息并判断是否只提取群组 Episode =====
        conversation_meta_repo = get_bean_by_type(ConversationMetaRawRepository)
        conversation_meta = await conversation_meta_repo.get_by_group_id(request.group_id)
        scene = conversation_meta.scene if conversation_meta and conversation_meta.scene else "assistant"
        is_assistant_scene = scene.lower() in ["assistant", "companion"]
        
        # ===== 第二步：提取所有参与者 =====
        participants = []
        if memcell.participants:
            participants = list(set(memcell.participants))
        
        # ===== 第三步：根据场景决定是否提取个人 Episode =====
        if is_assistant_scene:
            logger.info(f"[Worker-{worker_id}] 检测到 assistant 场景，仅提取群组 Episode，跳过个人 Episode")
            episode_tasks = [
                # 只提取群组 Episode (user_id=None)
                memory_manager.extract_memory(
                    memcell=memcell,
                    memory_type=MemoryType.EPISODIC_MEMORY,
                    user_id=None,
                    group_id=request.group_id,
                    group_name=request.group_name,
                ),
            ]
        else:
            logger.info(f"[Worker-{worker_id}] 非 assistant 场景，并行提取群组 Episode 和 {len(participants)} 个个人 Episode")
            episode_tasks = [
                # 群组 Episode (user_id=None)
                memory_manager.extract_memory(
                    memcell=memcell,
                    memory_type=MemoryType.EPISODIC_MEMORY,
                    user_id=None,
                    group_id=request.group_id,
                    group_name=request.group_name,
                ),
                # 个人 Episodes
                *[
                    memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.EPISODIC_MEMORY,
                        user_id=user_id,
                        group_id=request.group_id,
                        group_name=request.group_name,
                    )
                    for user_id in participants
                ]
            ]
        
        episode_results = await asyncio.gather(*episode_tasks, return_exceptions=True)
        
        # ===== 第四步：处理 Episode 提取结果 =====
        group_episode_memories: List[Memory] = []
        group_episode = episode_results[0] if episode_results else None
        
        if isinstance(group_episode, Exception):
            logger.error(f"[Worker-{worker_id}] ❌ 群组 Episode 提取异常: {group_episode}")
            group_episode = None
        elif group_episode:
            group_episode.ori_event_id_list = [memcell.event_id]
            group_episode.memcell_event_id_list = [memcell.event_id]
            group_episode_memories.append(group_episode)
            logger.info(f"[Worker-{worker_id}] ✅ 群组 Episode 提取成功")
            
            memcell.episode = group_episode.episode
            memcell.subject = group_episode.subject
        else:
            logger.warning(f"[Worker-{worker_id}] ⚠️  群组 Episode 提取失败")
        
        # 处理个人 Episodes（仅在非 assistant 场景）
        episode_memories: List[Memory] = []
        if not is_assistant_scene:
            for user_id, result in zip(participants, episode_results[1:]):
                if isinstance(result, Exception):
                    logger.error(f"[Worker-{worker_id}] ❌ 个人 Episode 提取异常: user_id={user_id}, error={result}")
                    continue
                
                if result:
                    result.ori_event_id_list = [memcell.event_id]
                    result.memcell_event_id_list = [memcell.event_id]
                    episode_memories.append(result)
                    logger.info(f"[Worker-{worker_id}] ✅ 个人 Episode 提取成功: user_id={user_id}")
                else:
                    logger.warning(f"[Worker-{worker_id}] ⚠️  个人 Episode 提取失败: user_id={user_id}")
        else:
            logger.info(f"[Worker-{worker_id}] assistant 场景，跳过个人 Episode 处理")
        
        # ===== 第五步：更新 MemCell 的 episode 字段到数据库 =====
        if request.group_id and group_episode:
            try:
                memcell_repo = get_bean_by_type(MemCellRawRepository)
                updated_memcell = await memcell_repo.update_by_event_id(
                    event_id=memcell.event_id,
                    update_data={
                        "episode": group_episode.episode,
                        "subject": group_episode.subject,
                    }
                )
                if updated_memcell:
                    logger.info(f"[Worker-{worker_id}] ✅ 更新 MemCell 的 episode 字段: {memcell.event_id}")
                else:
                    logger.warning(f"[Worker-{worker_id}] ⚠️  未找到 MemCell: {memcell.event_id}")
            except Exception as e:
                logger.error(f"[Worker-{worker_id}] ❌ 更新 MemCell episode 失败: {e}")
            
            # ===== 第六步：异步触发聚类 (不等待) =====
            try:
                memcell_for_clustering = MemCell(
                    event_id=memcell.event_id,
                    user_id_list=memcell.user_id_list,
                    original_data=memcell.original_data,
                    timestamp=memcell.timestamp,
                    summary=memcell.summary,
                    group_id=memcell.group_id,
                    group_name=memcell.group_name,
                    participants=memcell.participants,
                    type=memcell.type,
                    episode=group_episode.episode,
                )
                
                asyncio.create_task(_trigger_clustering(request.group_id, memcell_for_clustering, scene))
                logger.info(f"[Worker-{worker_id}] 异步触发聚类任务 (scene={scene})")
            except Exception as e:
                logger.error(f"[Worker-{worker_id}] ❌ 触发聚类失败: {e}")
        
        # ===== 第六步：保存 Episode 到数据库 =====
        if if_memorize(memcell):
            old_memory_list = await load_core_memories(request, participants, current_time)
            
            semantic_memories: List[SemanticMemoryItem] = []
            event_logs: List[EventLog] = []
            parent_docs_map: Dict[str, Any] = {}
            episodic_source_memories: List[Memory] = group_episode_memories + episode_memories
            
            if episodic_source_memories:
                for episode_mem in episodic_source_memories:
                    if getattr(episode_mem, "group_name", None) is None:
                        episode_mem.group_name = request.group_name
                    if getattr(episode_mem, "user_name", None) is None:
                        episode_mem.user_name = episode_mem.user_id
                
                episodic_docs = [
                    _convert_episode_memory_to_doc(episode_mem, current_time)
                    for episode_mem in episodic_source_memories
                ]
                episodic_payloads = [
                    MemoryDocPayload(MemoryType.EPISODIC_MEMORY, doc)
                    for doc in episodic_docs
                ]
                saved_docs_map = await save_memory_docs(episodic_payloads)
                saved_episode_docs = saved_docs_map.get(MemoryType.EPISODIC_MEMORY, [])
                
                for idx, (episode_mem, saved_doc) in enumerate(zip(episodic_source_memories, saved_episode_docs)):
                    episode_mem.event_id = str(saved_doc.event_id)
                    parent_docs_map[str(saved_doc.event_id)] = saved_doc
            
            # ===== 第七步：并行提取 Semantic 和 EventLog =====
            logger.info(f"[Worker-{worker_id}] 开始并行提取 Semantic 和 EventLog，共 {len(episodic_source_memories)} 个 Episode")
            
            extraction_tasks = []
            task_metadata = []
            
            for episode_mem in episodic_source_memories:
                if not episode_mem.event_id:
                    logger.warning(f"[Worker-{worker_id}] ⚠️  Episode 缺少 event_id，跳过")
                    continue
                
                extraction_tasks.append(
                    memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.SEMANTIC_MEMORY,
                        user_id=episode_mem.user_id,
                        episode_memory=episode_mem,
                    )
                )
                task_metadata.append({
                    'type': MemoryType.SEMANTIC_MEMORY,
                    'episode_mem': episode_mem,
                })
                
                extraction_tasks.append(
                    memory_manager.extract_memory(
                        memcell=memcell,
                        memory_type=MemoryType.PERSONAL_EVENT_LOG,
                        user_id=episode_mem.user_id,
                        episode_memory=episode_mem,
                    )
                )
                task_metadata.append({
                    'type': MemoryType.PERSONAL_EVENT_LOG,
                    'episode_mem': episode_mem,
                })
            
            if extraction_tasks:
                extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
                
                for metadata, result in zip(task_metadata, extraction_results):
                    memory_type = metadata['type']
                    episode_mem = metadata['episode_mem']
                    
                    if isinstance(result, Exception):
                        logger.error(
                            f"[Worker-{worker_id}] ❌ 提取异常: user_id={episode_mem.user_id}, "
                            f"memory_type={memory_type}, error={result}"
                        )
                        continue
                    
                    if not result:
                        logger.warning(
                            f"[Worker-{worker_id}] ⚠️  提取失败或为空: user_id={episode_mem.user_id}, "
                            f"memory_type={memory_type}"
                        )
                        continue
                    
                    logger.info(
                        f"[Worker-{worker_id}] ✅ 成功提取: user_id={episode_mem.user_id}, "
                        f"memory_type={memory_type}, "
                        f"数量={len(result) if isinstance(result, list) else 1}"
                    )
                    
                    if memory_type == MemoryType.SEMANTIC_MEMORY:
                        for mem in result:
                            mem.parent_event_id = episode_mem.event_id
                            mem.user_id = episode_mem.user_id
                            mem.group_id = episode_mem.group_id
                            mem.group_name = episode_mem.group_name
                            if getattr(mem, "user_name", None) is None:
                                mem.user_name = episode_mem.user_name
                            semantic_memories.append(mem)
                    elif memory_type == MemoryType.PERSONAL_EVENT_LOG:
                        result.parent_event_id = episode_mem.event_id
                        result.user_id = episode_mem.user_id
                        result.group_id = episode_mem.group_id
                        result.group_name = episode_mem.group_name
                        if getattr(result, "user_name", None) is None:
                            result.user_name = episode_mem.user_name
                        event_logs.append(result)
            
            # ===== 第八步：保存 Semantic 和 EventLog =====
            semantic_docs = []
            for sem_mem in semantic_memories:
                parent_doc = parent_docs_map.get(str(sem_mem.parent_event_id))
                if not parent_doc:
                    logger.warning(
                        f"[Worker-{worker_id}] ⚠️  未找到 parent_event_id={sem_mem.parent_event_id} 对应的 episodic_memory"
                    )
                    continue
                doc = _convert_semantic_memory_to_doc(sem_mem, parent_doc, current_time)
                semantic_docs.append(doc)
            
            event_log_docs = []
            for event_log in event_logs:
                parent_doc = parent_docs_map.get(str(event_log.parent_event_id))
                if not parent_doc:
                    logger.warning(
                        f"[Worker-{worker_id}] ⚠️  未找到 parent_event_id={event_log.parent_event_id} 对应的 episodic_memory"
                    )
                    continue
                docs = _convert_event_log_to_docs(event_log, parent_doc, current_time)
                event_log_docs.extend(docs)
            
            payloads: List[MemoryDocPayload] = []
            if semantic_docs:
                payloads.extend(
                    MemoryDocPayload(MemoryType.SEMANTIC_MEMORY, doc)
                    for doc in semantic_docs
                )
            if event_log_docs:
                payloads.extend(
                    MemoryDocPayload(MemoryType.PERSONAL_EVENT_LOG, doc)
                    for doc in event_log_docs
                )
            if payloads:
                await save_memory_docs(payloads)
            
            # ===== 第九步：更新状态表 =====
            await update_status_after_memcell(
                request, memcell, current_time, request.raw_data_type
            )

