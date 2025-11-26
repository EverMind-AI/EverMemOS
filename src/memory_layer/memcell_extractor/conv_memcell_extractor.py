"""
Simple Boundary Detection Base Class for EverMemOS

This module provides a simple and extensible base class for detecting
boundaries in various types of content (conversations, emails, notes, etc.).
"""

import time
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import uuid
import json, re
import asyncio
from common_utils.datetime_utils import (
    from_iso_format as dt_from_iso_format,
    from_timestamp as dt_from_timestamp,
    get_now_with_timezone,
)
from ..llm.llm_provider import LLMProvider
from api_specs.memory_types import RawDataType
from ..prompts.zh.conv_prompts import CONV_BOUNDARY_DETECTION_PROMPT

from ..prompts.eval.conv_prompts import (
    CONV_BOUNDARY_DETECTION_PROMPT as EVAL_CONV_BOUNDARY_DETECTION_PROMPT,
)
from .base_memcell_extractor import (
    MemCellExtractor,
    RawData,
    MemCell,
    StatusResult,
    MemCellExtractRequest,
)
from ..memory_extractor.episode_memory_extractor import (
    EpisodeMemoryExtractor,
    EpisodeMemoryExtractRequest,
)
from core.observation.logger import get_logger
from agentic_layer.vectorize_service import get_vectorize_service

logger = get_logger(__name__)


@dataclass
class BoundaryDetectionResult:
    """Boundary detection result."""

    should_end: bool
    should_wait: bool
    reasoning: str
    confidence: float
    topic_summary: Optional[str] = None


@dataclass
class ConversationMemCellExtractRequest(MemCellExtractRequest):
    pass


class ConvMemCellExtractor(MemCellExtractor):
    def __init__(self, llm_provider=LLMProvider, use_eval_prompts: bool = False):
        # Ensure base class receives the correct raw_data_type and provider
        super().__init__(RawDataType.CONVERSATION, llm_provider)
        self.llm_provider = llm_provider
        self.use_eval_prompts = use_eval_prompts
        self.episode_extractor = EpisodeMemoryExtractor(llm_provider, use_eval_prompts)

        if use_eval_prompts:
            self.conv_boundary_detection_prompt = EVAL_CONV_BOUNDARY_DETECTION_PROMPT
        else:
            self.conv_boundary_detection_prompt = CONV_BOUNDARY_DETECTION_PROMPT

    def shutdown(self) -> None:
        """Cleanup resources."""
        pass

    def _extract_participant_ids(
        self, chat_raw_data_list: List[Dict[str, Any]]
    ) -> List[str]:
        """
        从chat_raw_data_list中提取所有参与者ID

        从每个元素的content字典中获取：
        1. speaker_id（发言者ID）
        2. referList中所有的_id（@提及的用户ID）

        Args:
            chat_raw_data_list: 聊天原始数据列表

        Returns:
            List[str]: 去重后的所有参与者ID列表
        """
        participant_ids = set()

        for raw_data in chat_raw_data_list:

            # 提取speaker_id
            if 'speaker_id' in raw_data and raw_data['speaker_id']:
                participant_ids.add(raw_data['speaker_id'])

            # 提取referList中的所有ID
            if 'referList' in raw_data and raw_data['referList']:
                for refer_item in raw_data['referList']:
                    # refer_item可能是字典格式，包含_id字段
                    if isinstance(refer_item, dict):
                        # 处理MongoDB ObjectId格式的_id
                        if '_id' in refer_item:
                            refer_id = refer_item['_id']
                            # 如果是ObjectId对象，转换为字符串
                            if hasattr(refer_id, '__str__'):
                                participant_ids.add(str(refer_id))
                            else:
                                participant_ids.add(refer_id)
                        # 也检查普通的id字段
                        elif 'id' in refer_item:
                            participant_ids.add(refer_item['id'])
                    # 如果refer_item直接是ID字符串
                    elif isinstance(refer_item, str):
                        participant_ids.add(refer_item)

        return list(participant_ids)

    def _format_conversation_dicts(
        self, messages: list[dict[str, str]], include_timestamps: bool = False
    ) -> str:
        """Format conversation from message dictionaries into plain text."""
        lines = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            speaker_name = msg.get("speaker_name", "")
            timestamp = msg.get("timestamp", "")

            if content:
                if include_timestamps and timestamp:
                    try:
                        # 处理不同类型的timestamp
                        if isinstance(timestamp, datetime):
                            # 如果是datetime对象，直接格式化
                            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            lines.append(f"[{time_str}] {speaker_name}: {content}")
                        elif isinstance(timestamp, str):
                            # 如果是字符串，先解析再格式化
                            dt = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            )
                            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                            lines.append(f"[{time_str}] {speaker_name}: {content}")
                        else:
                            # 其他类型，不包含时间戳
                            lines.append(f"{speaker_name}: {content}")
                    except (ValueError, AttributeError, TypeError):
                        # Fallback if timestamp parsing fails
                        lines.append(f"{speaker_name}: {content}")
                else:
                    lines.append(f"{speaker_name}: {content}")
            else:
                print(msg)
                print(
                    f"[ConversationEpisodeBuilder] Warning: message {i} has no content"
                )
        return "\n".join(lines)

    def _calculate_time_gap(
        self,
        conversation_history: list[dict[str, str]],
        new_messages: list[dict[str, str]],
    ):
        if not conversation_history or not new_messages:
            return "No time gap information available"

        try:
            # Get the last message from history and first new message
            last_history_msg = conversation_history[-1]
            first_new_msg = new_messages[0]

            last_timestamp_str = last_history_msg.get("timestamp", "")
            first_timestamp_str = first_new_msg.get("timestamp", "")

            if not last_timestamp_str or not first_timestamp_str:
                return "No timestamp information available"

            # Parse timestamps - 处理不同类型的timestamp、
            try:
                if isinstance(last_timestamp_str, datetime):
                    last_time = last_timestamp_str
                elif isinstance(last_timestamp_str, str):
                    last_time = datetime.fromisoformat(
                        last_timestamp_str.replace("Z", "+00:00")
                    )
                else:
                    return "Invalid timestamp format for last message"

                if isinstance(first_timestamp_str, datetime):
                    first_time = first_timestamp_str
                elif isinstance(first_timestamp_str, str):
                    first_time = datetime.fromisoformat(
                        first_timestamp_str.replace("Z", "+00:00")
                    )
                else:
                    return "Invalid timestamp format for first message"
            except (ValueError, TypeError):
                return "Failed to parse timestamps"

            # Calculate time difference
            time_diff = first_time - last_time
            total_seconds = time_diff.total_seconds()

            if total_seconds < 0:
                return "Time gap: Messages appear to be out of order"
            elif total_seconds < 60:  # Less than 1 minute
                return f"Time gap: {int(total_seconds)} seconds (immediate response)"
            elif total_seconds < 3600:  # Less than 1 hour
                minutes = int(total_seconds // 60)
                return f"Time gap: {minutes} minutes (recent conversation)"
            elif total_seconds < 86400:  # Less than 1 day
                hours = int(total_seconds // 3600)
                return f"Time gap: {hours} hours (same day, but significant pause)"
            else:  # More than 1 day
                days = int(total_seconds // 86400)
                return f"Time gap: {days} days (long gap, likely new conversation)"

        except (ValueError, KeyError, AttributeError) as e:
            return f"Time gap calculation error: {str(e)}"

    async def _detect_boundary(
        self,
        conversation_history: list[dict[str, str]],
        new_messages: list[dict[str, str]],
    ) -> BoundaryDetectionResult:
        if not conversation_history:
            return BoundaryDetectionResult(
                should_end=False,
                should_wait=False,
                reasoning="First messages in conversation",
                confidence=1.0,
                topic_summary="",
            )
        history_text = self._format_conversation_dicts(
            conversation_history, include_timestamps=True
        )
        new_text = self._format_conversation_dicts(
            new_messages, include_timestamps=True
        )
        time_gap_info = self._calculate_time_gap(conversation_history, new_messages)

        print(
            f"[ConversationEpisodeBuilder] Detect boundary – history tokens: {len(history_text)} new tokens: {len(new_text)} time gap: {time_gap_info}"
        )

        prompt = self.conv_boundary_detection_prompt.format(
            conversation_history=history_text,
            new_messages=new_text,
            time_gap_info=time_gap_info,
        )
        for i in range(5):
            try:
                resp = await self.llm_provider.generate(prompt)
                print(
                    f"[ConversationEpisodeBuilder] Boundary response length: {len(resp)} chars"
                )

                # Parse JSON response from LLM boundary detection
                json_match = re.search(r"\{[^{}]*\}", resp, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return BoundaryDetectionResult(
                        should_end=data.get("should_end", False),
                        should_wait=data.get("should_wait", True),
                        reasoning=data.get("reasoning", "No reason provided"),
                        confidence=data.get("confidence", 1.0),
                        topic_summary=data.get("topic_summary", ""),
                    )
                else:
                    return BoundaryDetectionResult(
                        should_end=False,
                        should_wait=True,
                        reasoning="Failed to parse LLM response",
                        confidence=1.0,
                        topic_summary="",
                    )
                break
            except Exception as e:
                print('retry: ', i)
                if i == 4:
                    raise Exception("Boundary detection failed")
                continue

    async def extract_memcell(
        self,
        request: ConversationMemCellExtractRequest,
        use_semantic_extraction: bool = False,
    ) -> tuple[Optional[MemCell], Optional[StatusResult]]:
        history_message_dict_list = []
        for raw_data in request.history_raw_data_list:
            processed_data = self._data_process(raw_data)
            if processed_data is not None:  # 过滤掉不支持的消息类型
                history_message_dict_list.append(processed_data)

        # 检查最后一条new_raw_data是否为None
        if (
            request.new_raw_data_list
            and self._data_process(request.new_raw_data_list[-1]) is None
        ):
            logger.warning(
                f"[ConvMemCellExtractor] 最后一条new_raw_data为None，跳过处理"
            )
            status_control_result = StatusResult(should_wait=True)
            return (None, status_control_result)

        new_message_dict_list = []
        for new_raw_data in request.new_raw_data_list:
            processed_data = self._data_process(new_raw_data)
            if processed_data is not None:  # 过滤掉不支持的消息类型
                new_message_dict_list.append(processed_data)

        # 检查是否有有效的消息可处理
        if not new_message_dict_list:
            logger.warning(
                f"[ConvMemCellExtractor] 没有有效的新消息可处理（可能都被过滤了）"
            )
            status_control_result = StatusResult(should_wait=True)
            return (None, status_control_result)

        if request.smart_mask_flag:
            boundary_detection_result = await self._detect_boundary(
                conversation_history=history_message_dict_list[:-1],
                new_messages=new_message_dict_list,
            )
        else:
            boundary_detection_result = await self._detect_boundary(
                conversation_history=history_message_dict_list,
                new_messages=new_message_dict_list,
            )
        should_end = boundary_detection_result.should_end
        should_wait = boundary_detection_result.should_wait
        reason = boundary_detection_result.reasoning

        status_control_result = StatusResult(should_wait=should_wait)

        if should_end:
            # TODO 重构专项：转为int逻辑不对 应该保持为datetime
            ts_value = history_message_dict_list[-1].get("timestamp")
            timestamp = dt_from_iso_format(ts_value)
            participants = self._extract_participant_ids(history_message_dict_list)
            # 创建 MemCell
            # 优先使用边界检测的主题摘要；若为空，回退到最后一条新消息的文本；再不行用占位摘要
            fallback_text = ""
            if new_message_dict_list:
                last_msg = new_message_dict_list[-1]
                if isinstance(last_msg, dict):
                    fallback_text = last_msg.get("content") or ""
                elif isinstance(last_msg, str):
                    fallback_text = last_msg
            summary_text = boundary_detection_result.topic_summary or (
                fallback_text.strip()[:200] if fallback_text else "会话片段"
            )

            memcell = MemCell(
                event_id=str(uuid.uuid4()),
                user_id_list=request.user_id_list,
                original_data=history_message_dict_list,
                timestamp=timestamp,
                summary=summary_text,
                group_id=request.group_id,
                participants=participants,  # 使用合并后的participants
                type=self.raw_data_type,
            )
            # 自动触发情景记忆提取
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    episode_request = EpisodeMemoryExtractRequest(
                        memcell_list=[memcell],
                        user_id_list=request.user_id_list,
                        participants=participants,
                        group_id=request.group_id,
                    )
                    logger.debug(
                        f"📚 自动触发情景记忆提取开始: memcell_list={memcell}, user_id_list={request.user_id_list}, participants={participants}, group_id={request.group_id}"
                    )
                    now = time.time()
                    episode_result = await self.episode_extractor.extract_memory(
                        episode_request,
                        use_group_prompt=True,
                        use_semantic_extraction=use_semantic_extraction,
                    )
                    logger.debug(
                        f"📚 自动触发情景记忆提取, 耗时: {time.time() - now}秒"
                    )
                    if episode_result and isinstance(episode_result, MemCell):
                        # GROUP_EPISODE_GENERATION_PROMPT 模式：返回包含情景记忆的 MemCell
                        logger.info(f"✅ 成功生成情景记忆并存储到 MemCell 中")
                        # Attach embedding info to MemCell (episode preferred)

                        text_for_embed = (
                            episode_result.episode or episode_result.summary or ""
                        )
                        if text_for_embed:
                            vs = get_vectorize_service()
                            vec = await vs.get_embedding(text_for_embed)
                            episode_result.extend = episode_result.extend or {}
                            episode_result.extend["embedding"] = (
                                vec.tolist() if hasattr(vec, "tolist") else list(vec)
                            )
                            episode_result.extend["vector_model"] = vs.get_model_name()

                        # 提交到聚类器（如果存在）
                        if hasattr(self, '_cluster_worker') and self._cluster_worker:
                            try:
                                self._cluster_worker.submit(
                                    request.group_id, episode_result.to_dict()
                                )
                            except Exception as e:
                                logger.debug(f"Failed to submit to cluster worker: {e}")

                        return (episode_result, status_control_result)
                    else:
                        logger.warning(
                            f"⚠️ 情景记忆提取失败 (尝试 {attempt + 1}/{max_retries})"
                        )

                except Exception as e:
                    logger.error(
                        f"❌ 情景记忆提取出错: {e} (尝试 {attempt + 1}/{max_retries})"
                    )

                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                else:
                    logger.error(f"❌ 所有重试次数均失败，未能提取情景记忆")

            # Attach embedding info to MemCell if available
            try:
                text_for_embed = memcell.episode
                if text_for_embed:
                    vs = get_vectorize_service()
                    vec = await vs.get_embedding(text_for_embed)
                    memcell.extend = memcell.extend or {}
                    memcell.extend["embedding"] = (
                        vec.tolist() if hasattr(vec, "tolist") else list(vec)
                    )
                    memcell.extend["vector_model"] = vs.get_model_name()
            except Exception:
                logger.debug("Embedding attach failed; continue without it")

            # 提交到聚类器（如果存在）
            if hasattr(self, '_cluster_worker') and self._cluster_worker:
                try:
                    self._cluster_worker.submit(request.group_id, memcell.to_dict())
                except Exception as e:
                    logger.debug(f"Failed to submit to cluster worker: {e}")

            return (memcell, status_control_result)
        elif should_wait:
            logger.debug(f"⏳ Waiting for more messages: {reason}")
        return (None, status_control_result)

    def _data_process(self, raw_data: RawData) -> Dict[str, Any]:
        """处理原始数据，包括消息类型过滤和预处理"""
        content = (
            raw_data.content.copy()
            if isinstance(raw_data.content, dict)
            else raw_data.content
        )

        # 获取消息类型
        msg_type = content.get('msgType') if isinstance(content, dict) else None

        # 定义支持的消息类型和对应的占位符
        SUPPORTED_MSG_TYPES = {
            1: None,  # TEXT - 保持原文本
            2: "[图片]",  # PICTURE
            3: "[视频]",  # VIDEO
            4: "[音频]",  # AUDIO
            5: "[文件]",  # FILE - 保持原文本（文本和文件在同一个消息里）
            6: "[文件]",  # FILES
        }

        if isinstance(content, dict) and msg_type is not None:
            # 检查是否为支持的消息类型
            if msg_type not in SUPPORTED_MSG_TYPES:
                # 不支持的消息类型，直接跳过（返回None会在上层处理）
                logger.warning(
                    f"[ConvMemCellExtractor] 跳过不支持的消息类型: {msg_type}"
                )
                return None

            # 对非文本消息进行预处理
            placeholder = SUPPORTED_MSG_TYPES[msg_type]
            if placeholder is not None:
                # 替换消息内容为占位符
                content = content.copy()
                content['content'] = placeholder
                logger.debug(
                    f"[ConvMemCellExtractor] 消息类型 {msg_type} 转换为占位符: {placeholder}"
                )

        return content
