"""
语义记忆提取器 - 基于联想预测方法
从MemCell中生成对用户未来生活、决策可能产生的影响预测
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# 使用动态语言提示词导入（根据 MEMORY_LANGUAGE 环境变量自动选择）
from ..prompts import (
    get_group_semantic_generation_prompt,
    get_semantic_generation_prompt,
)
from ..llm.llm_provider import LLMProvider
from .base_memory_extractor import MemoryExtractor, MemoryExtractRequest
from ..types import MemoryType, MemCell, Memory, SemanticMemoryItem
from agentic_layer.vectorize_service import get_vectorize_service
from core.observation.logger import get_logger
import uuid

logger = get_logger(__name__)


class SemanticMemoryExtractor(MemoryExtractor):
    """
    语义记忆提取器 - 基于联想预测方法

    支持两种模式：
    1. use_group_prompt=False: 基于EpisodeMemory对象生成联想，返回的EpisodeMemory对象中添加semantic_memories字段
    2. use_group_prompt=True: 基于MemCell对象生成联想，返回的MemCell对象中添加semantic_memories字段

    新的策略实现：
    1. 基于内容，大模型联想出10条对用户后续的生活、决策可能产生的影响
    2. 每条联想考虑可能产生的持续时间
    3. 重点关注用户个人层面的影响

    主要方法：
    - generate_semantic_memories_for_memcell(): 为MemCell生成语义记忆联想
    - generate_semantic_memories_for_episode(): 为EpisodeMemory生成语义记忆联想
    """

    def __init__(self, llm_provider: LLMProvider):
        """
        初始化语义记忆提取器

        Args:
            llm_provider: LLM提供者
        """
        super().__init__(MemoryType.SEMANTIC_MEMORY)
        self.llm_provider = llm_provider

        logger.info("语义记忆提取器已初始化（联想预测模式）")

    async def extract_memory(self, request: MemoryExtractRequest) -> Optional[Memory]:
        """
        实现抽象基类要求的 extract_memory 方法

        注意：SemanticMemoryExtractor 不应该直接使用 extract_memory 方法
        应该使用 generate_semantic_memories_for_memcell 或 generate_semantic_memories_for_episode

        Args:
            request: 记忆提取请求

        Returns:
            None - 此方法不应该被调用
        """
        raise NotImplementedError(
            "SemanticMemoryExtractor 不应该直接使用 extract_memory 方法。"
            "请使用 generate_semantic_memories_for_memcell 或 generate_semantic_memories_for_episode 方法。"
        )

    async def generate_semantic_memories_for_memcell(
        self, memcell: MemCell
    ) -> List[SemanticMemoryItem]:
        """
        为MemCell生成语义记忆联想预测
        
        这是新的策略：基于MemCell内容，大模型联想出10条对用户后续的生活、决策可能产生的影响
        
        Args:
            memcell: MemCell对象
        
        Returns:
            语义记忆联想项目列表（10条），包含时间信息
        """
        # 最多重试5次
        for retry in range(5):
            try:
                logger.info(f"🎯 为MemCell生成语义记忆联想: {memcell.subject}，重试次数: {retry+1}/5")

                # 构建提示词
                prompt = get_group_semantic_generation_prompt(
                    memcell_summary=memcell.summary,
                    memcell_episode=memcell.episode or "",
                    user_ids=memcell.user_id_list,
                )

                # 调用LLM生成联想
                logger.debug(f"📝 开始调用LLM生成语义记忆联想，提示词长度: {len(prompt)}")
                response = await self.llm_provider.generate(prompt=prompt, temperature=0.3)
                logger.debug(
                    f"✅ LLM调用完成，响应长度: {len(response) if response else 0}"
                )

                # 解析JSON响应
                start_time = self._extract_start_time_from_timestamp(memcell.timestamp)
                semantic_memories = await self._parse_semantic_memories_response(
                    response, memcell.event_id, start_time
                )

                # 验证必须至少有1条
                if len(semantic_memories) == 0:
                    raise ValueError("LLM返回的语义记忆列表为空")

                # 确保返回恰好10条（不足则警告，但不重试）
                if len(semantic_memories) > 10:
                    semantic_memories = semantic_memories[:10]
                elif len(semantic_memories) < 10:
                    logger.warning(
                        f"生成的语义记忆联想数量不足10条，实际生成: {len(semantic_memories)}"
                    )

                logger.info(f"✅ 成功生成 {len(semantic_memories)} 条语义记忆联想")
                for i, memory in enumerate(semantic_memories[:3], 1):
                    logger.info(f"  联想{i}: {memory.content}")

                return semantic_memories

            except Exception as e:
                logger.warning(f"生成语义记忆联想重试 {retry+1}/5: {e}")
                if retry == 4:
                    logger.error(f"生成语义记忆联想失败，已重试5次")
                    return []
                continue
        
        return []

    async def generate_semantic_memories_for_episode(
        self, episode: Memory
    ) -> List[SemanticMemoryItem]:
        """
        为EpisodeMemory生成语义记忆联想预测
        
        这是个人模式：基于EpisodeMemory内容，大模型联想出10条对用户后续的生活、决策可能产生的影响
        
        Args:
            episode: EpisodeMemory对象
        
        Returns:
            语义记忆联想项目列表（10条），包含时间信息
        """
        # 最多重试5次
        for retry in range(5):
            try:
                logger.info(f"🎯 为EpisodeMemory生成语义记忆联想: {episode.subject}，重试次数: {retry+1}/5")

                # 构建提示词
                # 直接使用episode的user_id
                prompt = get_semantic_generation_prompt(
                    episode_memory=episode.summary or "",
                    episode_content=episode.episode or "",
                    user_id=episode.user_id,
                )

                # 调用LLM生成联想
                response = await self.llm_provider.generate(prompt=prompt, temperature=0.3)

                # 解析JSON响应
                source_episode_id = (
                    episode.ori_event_id_list[0] if episode.ori_event_id_list else None
                )
                start_time = self._extract_start_time_from_timestamp(episode.timestamp)
                semantic_memories = await self._parse_semantic_memories_response(
                    response, source_episode_id, start_time
                )

                # 验证必须至少有1条
                if len(semantic_memories) == 0:
                    raise ValueError("LLM返回的语义记忆列表为空")

                # 确保返回恰好10条（不足则警告，但不重试）
                if len(semantic_memories) > 10:
                    semantic_memories = semantic_memories[:10]
                elif len(semantic_memories) < 10:
                    logger.warning(
                        f"生成的语义记忆联想数量不足10条，实际生成: {len(semantic_memories)}"
                    )

                logger.info(f"✅ 成功生成 {len(semantic_memories)} 条语义记忆联想")
                for i, memory in enumerate(semantic_memories[:3], 1):
                    logger.info(f"  联想{i}: {memory.content}")

                return semantic_memories

            except Exception as e:
                logger.warning(f"生成语义记忆联想重试 {retry+1}/5: {e}")
                if retry == 4:
                    logger.error(f"生成语义记忆联想失败，已重试5次")
                    return []
                continue
        
        return []

    @staticmethod
    def _clean_date_string(date_str: Optional[str]) -> Optional[str]:
        """清理日期字符串，移除非法字符

        Args:
            date_str: 原始日期字符串

        Returns:
            清理后的日期字符串，如果无效则返回 None
        """
        if not date_str or not isinstance(date_str, str):
            return None

        import re

        # 只保留数字和连字符，移除其他字符（如中文、空格等）
        cleaned = re.sub(r'[^\d\-]', '', date_str)

        # 验证格式是否为 YYYY-MM-DD
        if re.match(r'^\d{4}-\d{2}-\d{2}$', cleaned):
            return cleaned
        else:
            logger.warning(
                f"时间格式无效，已清理但仍不符合 YYYY-MM-DD: 原始='{date_str}', 清理后='{cleaned}'"
            )
            return None

    async def _parse_semantic_memories_response(
        self,
        response: str,
        source_episode_id: Optional[str] = None,
        start_time: Optional[str] = None,
    ) -> List[SemanticMemoryItem]:
        """
        解析LLM的JSON响应，提取语义记忆联想列表

        Args:
            response: LLM响应文本
            source_episode_id: 来源事件ID
            start_time: 开始时间，格式为YYYY-MM-DD

        Returns:
            语义记忆联想项目列表
        """
        try:
            # 首先尝试提取代码块中的JSON
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end > start:
                    json_str = response[start:end].strip()
                    data = json.loads(json_str)
                else:
                    data = json.loads(response)
            else:
                # 尝试解析整个响应为JSON数组
                data = json.loads(response)

            # 确保data是列表
            if isinstance(data, list):
                semantic_memories = []
                
                # 先收集所有需要处理的数据
                items_to_process = []
                for item in data:
                    content = item.get('content', '')
                    evidence = item.get('evidence', '')

                    # 使用传入的start_time或LLM提供的时间
                    item_start_time = item.get('start_time', start_time)
                    item_end_time = item.get('end_time')
                    item_duration_days = item.get('duration_days')

                    # 清理时间格式（防止 LLM 输出错误的格式）
                    item_start_time = self._clean_date_string(item_start_time)
                    item_end_time = self._clean_date_string(item_end_time)

                    # 智能时间计算：优先使用LLM提供的时间信息
                    if item_start_time:
                        # 如果LLM提供了duration_days但没有end_time，计算end_time
                        if item_duration_days and not item_end_time:
                            item_end_time = self._calculate_end_time_from_duration(
                                item_start_time, item_duration_days
                            )
                        # 如果LLM提供了end_time但没有duration_days，计算duration_days
                        elif item_end_time and not item_duration_days:
                            item_duration_days = self._calculate_duration_days(
                                item_start_time, item_end_time
                            )
                        # 如果LLM都没有提供，则保持为None（不进行额外提取）

                    items_to_process.append({
                        'content': content,
                        'evidence': evidence,
                        'start_time': item_start_time,
                        'end_time': item_end_time,
                        'duration_days': item_duration_days,
                        'source_episode_id': item.get('source_episode_id', source_episode_id),
                    })

                # 批量计算所有 content 的 embeddings（性能优化）
                vs = get_vectorize_service()
                contents = [item['content'] for item in items_to_process]
                embeddings_batch = await vs.get_embeddings(contents)  # 使用 get_embeddings (List[str])
                
                # 创建SemanticMemoryItem对象
                for i, item_data in enumerate(items_to_process):
                    # 处理 embedding：可能是 numpy 数组或已经是列表
                    embedding = embeddings_batch[i]
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    elif not isinstance(embedding, list):
                        embedding = list(embedding)
                    
                    memory_item = SemanticMemoryItem(
                        content=item_data['content'],
                        evidence=item_data['evidence'],
                        start_time=item_data['start_time'],
                        end_time=item_data['end_time'],
                        duration_days=item_data['duration_days'],
                        source_episode_id=item_data['source_episode_id'],
                        embedding=embedding,
                    )
                    semantic_memories.append(memory_item)

                return semantic_memories
            else:
                logger.error(f"响应不是JSON数组格式: {data}")
                return []

        except json.JSONDecodeError as e:
            logger.error(f"解析JSON响应时出错: {e}")
            logger.debug(f"响应内容: {response[:200]}...")
            return []
        except Exception as e:
            logger.error(f"解析语义记忆响应时出错: {e}")
            return []

    def _extract_start_time_from_timestamp(self, timestamp: datetime) -> str:
        """
        从MemCell的timestamp字段提取开始时间

        Args:
            timestamp: MemCell的时间戳

        Returns:
            开始时间字符串，格式为YYYY-MM-DD
        """
        return timestamp.strftime('%Y-%m-%d')

    def _calculate_end_time_from_duration(
        self, start_time: str, duration_days: int
    ) -> Optional[str]:
        """
        根据开始时间和持续时间计算结束时间

        Args:
            start_time: 开始时间，格式为YYYY-MM-DD
            duration_days: 持续天数

        Returns:
            结束时间字符串，格式为YYYY-MM-DD，如果无法计算则返回None
        """
        try:
            if not start_time or duration_days is None:
                return None

            start_date = datetime.strptime(start_time, '%Y-%m-%d')
            end_date = start_date + timedelta(days=duration_days)

            return end_date.strftime('%Y-%m-%d')

        except Exception as e:
            logger.error(f"根据持续时间计算结束时间时出错: {e}")
            return None

    def _calculate_duration_days(self, start_time: str, end_time: str) -> Optional[int]:
        """
        计算持续时间（天数）

        Args:
            start_time: 开始时间，格式为YYYY-MM-DD
            end_time: 结束时间，格式为YYYY-MM-DD

        Returns:
            持续天数，如果无法计算则返回None
        """
        try:
            if not start_time or not end_time:
                return None

            start_date = datetime.strptime(start_time, '%Y-%m-%d')
            end_date = datetime.strptime(end_time, '%Y-%m-%d')

            duration = end_date - start_date
            return duration.days

        except Exception as e:
            logger.error(f"计算持续时间时出错: {e}")
            return None
