"""
Event Log Extractor for EverMemOS

This module extracts atomic event logs from episode memories for optimized retrieval.
Each event log contains a time and a list of atomic facts extracted from the episode.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import re

# 使用动态语言提示词导入（根据 MEMORY_LANGUAGE 环境变量自动选择）
from ..prompts import EVENT_LOG_PROMPT

# 评估专用提示词
from ..prompts.eval.event_log_prompts import EVENT_LOG_PROMPT as EVAL_EVENT_LOG_PROMPT

from ..llm.llm_provider import LLMProvider
from common_utils.datetime_utils import get_now_with_timezone

from core.observation.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EventLog:
    """
    Event log data structure containing time and atomic facts.
    """

    time: str  # 事件发生时间，格式如 "March 10, 2024(Sunday) at 2:00 PM"
    atomic_fact: List[str]  # 原子事实列表，每个事实是一个完整的句子
    fact_embeddings: List[List[float]] = None  # 每个 atomic_fact 对应的 embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert EventLog to dictionary format."""
        result = {"time": self.time, "atomic_fact": self.atomic_fact}
        if self.fact_embeddings:
            result["fact_embeddings"] = self.fact_embeddings
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventLog":
        """Create EventLog from dictionary."""
        return cls(
            time=data.get("time", ""), 
            atomic_fact=data.get("atomic_fact", []),
            fact_embeddings=data.get("fact_embeddings")
        )


class EventLogExtractor:
    """
    Extractor for converting episode memories into structured event logs.

    The event log format is optimized for retrieval:
    - Time field provides temporal context
    - Atomic facts are independent, searchable units
    """

    def __init__(self, llm_provider: LLMProvider, use_eval_prompts: bool = False):
        """
        Initialize the event log extractor.

        Args:
            llm_provider: LLM provider for generating event logs
            use_eval_prompts: Whether to use evaluation-specific prompts
        """
        self.llm_provider = llm_provider
        self.use_eval_prompts = use_eval_prompts
        
        # 根据 use_eval_prompts 选择对应的提示词
        if self.use_eval_prompts:
            self.event_log_prompt = EVAL_EVENT_LOG_PROMPT
        else:
            self.event_log_prompt = EVENT_LOG_PROMPT

    def _parse_timestamp(self, timestamp) -> datetime:
        """
        解析时间戳为 datetime 对象
        支持多种格式：数字时间戳、ISO格式字符串、datetime对象等

        Args:
            timestamp: 时间戳，可以是多种格式

        Returns:
            datetime: 解析后的datetime对象
        """
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            try:
                if timestamp.isdigit():
                    return datetime.fromtimestamp(int(timestamp))
                else:
                    # 尝试解析ISO格式
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                logger.error(f"解析时间戳失败: {timestamp}")
                return get_now_with_timezone()
        else:
            logger.error(f"未知时间戳格式: {timestamp}")
            return get_now_with_timezone()

    def _format_timestamp(self, dt: datetime) -> str:
        """
        格式化 datetime 为事件日志所需的字符串格式
        格式: "March 10, 2024(Sunday) at 2:00 PM"

        Args:
            dt: datetime对象

        Returns:
            str: 格式化后的时间字符串
        """
        weekday = dt.strftime("%A")  # Monday, Tuesday, etc.
        month_day_year = dt.strftime("%B %d, %Y")  # March 10, 2024
        time_of_day = dt.strftime("%I:%M %p")  # 2:00 PM
        return f"{month_day_year}({weekday}) at {time_of_day}"

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM返回的JSON响应
        支持多种格式：纯JSON、JSON代码块等

        Args:
            response: LLM的原始响应

        Returns:
            Dict: 解析后的JSON对象

        Raises:
            ValueError: 如果无法解析响应
        """
        # 1. 尝试提取代码块中的JSON
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end > start:
                json_str = response[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # 2. 尝试提取任何代码块
        if '```' in response:
            start = response.find('```') + 3
            # 跳过语言标识符（如果有）
            if response[start : start + 10].strip().split()[0].isalpha():
                start = response.find('\n', start) + 1
            end = response.find('```', start)
            if end > start:
                json_str = response[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # 3. 尝试提取包含event_log的JSON对象
        json_match = re.search(
            r'\{[^{}]*"event_log"[^{}]*\{[^{}]*"time"[^{}]*"atomic_fact"[^{}]*\}[^{}]*\}',
            response,
            re.DOTALL,
        )
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 4. 尝试直接解析整个响应
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # 5. 如果都失败了，抛出异常
        logger.error(f"无法解析LLM响应: {response[:200]}...")
        raise ValueError(f"无法解析LLM响应为有效的JSON格式")

    async def extract_event_log(
        self, episode_text: str, timestamp: Any
    ) -> Optional[EventLog]:
        """
        从episode memory提取event log

        Args:
            episode_text: episode memory的文本内容
            timestamp: episode的时间戳（可以是多种格式）

        Returns:
            EventLog: 提取的事件日志，如果提取失败则返回None
        """
        try:
            # 1. 解析并格式化时间戳
            dt = self._parse_timestamp(timestamp)
            time_str = self._format_timestamp(dt)

            # 2. 构建prompt（使用实例变量 self.event_log_prompt）
            prompt = self.event_log_prompt.replace("{{EPISODE_TEXT}}", episode_text)
            prompt = prompt.replace("{{TIME}}", time_str)

            # 3. 调用LLM生成event log
            logger.debug(f"开始提取event log，时间: {time_str}")
            response = await self.llm_provider.generate(prompt)

            # 4. 解析LLM响应
            data = self._parse_llm_response(response)

            # 5. 验证响应格式
            if "event_log" not in data:
                logger.error(f"LLM响应中缺少'event_log'字段: {data}")
                return None

            event_log_data = data["event_log"]

            # 验证必需字段
            if "time" not in event_log_data or "atomic_fact" not in event_log_data:
                logger.error(f"event_log中缺少必需字段: {event_log_data}")
                return None

            # 验证atomic_fact是列表
            if not isinstance(event_log_data["atomic_fact"], list):
                logger.error(f"atomic_fact不是列表: {event_log_data['atomic_fact']}")
                return None

            # 6. 创建EventLog对象
            event_log = EventLog(
                time=event_log_data["time"], atomic_fact=event_log_data["atomic_fact"]
            )
            
            # 7. 为每个 atomic_fact 生成 embedding
            from agentic_layer.vectorize_service import get_vectorize_service
            vectorize_service = get_vectorize_service()
            
            fact_embeddings = []
            for fact in event_log.atomic_fact:
                fact_emb = await vectorize_service.get_embedding(fact)
                fact_embeddings.append(fact_emb.tolist() if hasattr(fact_emb, 'tolist') else fact_emb)
            
            event_log.fact_embeddings = fact_embeddings

            logger.debug(
                f"成功提取event log，包含 {len(event_log.atomic_fact)} 个原子事实（已生成 embedding）"
            )
            return event_log

        except Exception as e:
            logger.error(f"提取event log时发生错误: {e}", exc_info=True)
            return None

    async def extract_event_logs_batch(
        self, episodes: List[Dict[str, Any]]
    ) -> List[Optional[EventLog]]:
        """
        批量提取event logs

        Args:
            episodes: episode列表，每个episode包含'episode'和'timestamp'字段

        Returns:
            List[Optional[EventLog]]: 提取的event log列表
        """
        import asyncio

        # 并发提取所有event logs
        tasks = [
            self.extract_event_log(
                episode_text=ep.get("episode", ""), timestamp=ep.get("timestamp")
            )
            for ep in episodes
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        event_logs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批量提取第{i}个event log时失败: {result}")
                event_logs.append(None)
            else:
                event_logs.append(result)

        return event_logs


def format_event_log_for_bm25(event_log: EventLog) -> str:
    """
    格式化event log用于BM25检索
    只使用atomic_fact字段，将所有原子事实拼接成一个字符串

    Args:
        event_log: EventLog对象

    Returns:
        str: 用于BM25检索的文本
    """
    if not event_log or not event_log.atomic_fact:
        return ""

    # 直接拼接所有原子事实，用空格分隔
    return " ".join(event_log.atomic_fact)


def format_event_log_for_rerank(event_log: EventLog) -> str:
    """
    格式化event log用于rerank
    使用 "time" + "：" + "atomic_fact"拼接

    Args:
        event_log: EventLog对象

    Returns:
        str: 用于rerank的文本
    """
    if not event_log:
        return ""

    # 拼接时间和原子事实
    time_part = event_log.time or ""
    facts_part = " ".join(event_log.atomic_fact) if event_log.atomic_fact else ""

    if time_part and facts_part:
        return f"{time_part}：{facts_part}"
    elif time_part:
        return time_part
    elif facts_part:
        return facts_part
    else:
        return ""
