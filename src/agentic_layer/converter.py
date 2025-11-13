"""
请求转换器模块

此模块包含各种外部请求格式到内部 Request 对象的转换函数。
"""

from __future__ import annotations

from typing import Any, Dict, List, Union, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from infra_layer.adapters.input.format_transfer import (
    convert_conversation_to_raw_data_list,
)

from .schemas import DataFields
from .memory_models import MemoryType
from .dtos.memory_query import RetrieveMemRequest, FetchMemRequest
from memory_layer.memory_manager import MemorizeRequest
from memory_layer.types import RawDataType
from memory_layer.memcell_extractor.base_memcell_extractor import RawData

import logging

logger = logging.getLogger(__name__)


def convert_dict_to_fetch_mem_request(data: Dict[str, Any]) -> FetchMemRequest:
    """
    将字典转换为 FetchMemRequest 对象

    Args:
        data: 包含 FetchMemRequest 字段的字典

    Returns:
        FetchMemRequest 对象

    Raises:
        ValueError: 当必需字段缺失或类型不正确时
    """
    try:
        # 验证必需字段
        if "user_id" not in data:
            raise ValueError("user_id 是必需字段")

        # 转换 memory_type，如果未提供则使用默认值
        memory_type = MemoryType(data.get("memory_type", "multiple"))
        logger.debug(f"version_range: {data.get('version_range', None)}")
        # 构建 FetchMemRequest 对象
        return FetchMemRequest(
            user_id=data["user_id"],
            memory_type=memory_type,
            limit=data.get("limit", 10),
            offset=data.get("offset", 0),
            filters=data.get("filters", {}),
            sort_by=data.get("sort_by"),
            sort_order=data.get("sort_order", "desc"),
            version_range=data.get("version_range", None),
        )
    except Exception as e:
        raise ValueError(f"FetchMemRequest 转换失败: {e}")


def convert_dict_to_retrieve_mem_request(
    data: Dict[str, Any], query: Optional[str] = None
) -> RetrieveMemRequest:
    """
    将字典转换为 RetrieveMemRequest 对象

    Args:
        data: 包含 RetrieveMemRequest 字段的字典
        query: 查询文本（可选）

    Returns:
        RetrieveMemRequest 对象

    Raises:
        ValueError: 当必需字段缺失或类型不正确时
    """
    try:
        # 验证必需字段
        if "user_id" not in data:
            raise ValueError("user_id 是必需字段")

        # 处理 retrieve_method，如果未提供则使用默认值 keyword
        from .memory_models import RetrieveMethod

        retrieve_method_str = data.get("retrieve_method", "keyword")

        # 将字符串转换为 RetrieveMethod 枚举
        try:
            retrieve_method = RetrieveMethod(retrieve_method_str)
        except ValueError:
            logger.warning(
                f"无效的 retrieve_method: {retrieve_method_str}, 使用默认值 keyword"
            )
            retrieve_method = RetrieveMethod.KEYWORD

        return RetrieveMemRequest(
            retrieve_method=retrieve_method,
            user_id=data["user_id"],
            query=query or data.get("query", None),
            memory_types=data.get("memory_types", []),
            top_k=data.get("top_k", 10),
            filters=data.get("filters", {}),
            include_metadata=data.get("include_metadata", True),
            start_time=data.get("start_time", None),
            end_time=data.get("end_time", None),
            radius=data.get("radius", None),  # COSINE 相似度阈值
        )
    except Exception as e:
        raise ValueError(f"RetrieveMemRequest 转换失败: {e}")


def _extract_current_time(data: Dict[str, Any]) -> Optional[datetime]:
    """
    从数据中提取 current_time 字段

    Args:
        data: 数据字典

    Returns:
        current_time 或 None
    """
    if "current_time" not in data:
        return None

    current_time_str = data["current_time"]
    if isinstance(current_time_str, str):
        try:
            return datetime.fromisoformat(current_time_str.replace('Z', '+00:00'))
        except ValueError:
            logger.warning(f"无法解析 current_time: {current_time_str}")
            return None
    elif isinstance(current_time_str, datetime):
        # from anhua
        if current_time_str.tzinfo is None:
            return current_time_str.replace(tzinfo=ZoneInfo("UTC"))
        return current_time_str

    return None


def _create_memorize_request(
    history_data: List[RawData],
    new_data: List[RawData],
    data_type: RawDataType,
    participants: List[str],
    group_id: str = None,
    group_name: str = None,
    current_time: datetime = None,
    scene: str = None,
    scene_desc: Dict[str, Any] = None,
) -> MemorizeRequest:
    """
    创建 MemorizeRequest 对象的公共函数

    Args:
        history_data: 历史数据列表
        new_data: 新数据列表
        data_type: 数据类型
        participants: 参与者列表
        group_id: 群组ID
        group_name: 群组名称
        current_time: 当前时间
        scene: 场景标识符（新增，兼容性处理）
        scene_desc: 场景描述信息（新增，兼容性处理）

    Returns:
        MemorizeRequest 对象
    """
    # 确保 participants 不为 None
    if participants is None:
        participants = []

    # 如果 current_time 为 None，尝试从 new_data[0] 的 timestamp 或 updateTime 来获取
    if current_time is None and new_data and new_data[0] is not None:
        first_data = new_data[0]
        if hasattr(first_data, 'content') and first_data.content:
            # 优先使用 updateTime
            if 'updateTime' in first_data.content and first_data.content['updateTime']:
                current_time = first_data.content['updateTime']
            elif 'timestamp' in first_data.content and first_data.content['timestamp']:
                current_time = first_data.content['timestamp']

    return MemorizeRequest(
        history_raw_data_list=history_data,
        new_raw_data_list=new_data,
        raw_data_type=data_type,
        user_id_list=participants,
        group_id=group_id,
        group_name=group_name,
        current_time=current_time,
        scene=scene,
        scene_desc=scene_desc,
    )


async def _handle_conversation_format(data: Dict[str, Any]) -> MemorizeRequest:
    """
    处理聊天消息格式数据

    Args:
        data: 包含 messages 字段的数据

    Returns:
        MemorizeRequest 对象
    """
    logger.debug("处理聊天消息格式数据")
    messages = data.get(DataFields.MESSAGES, [])
    if not messages:
        raise ValueError("messages 字段不能为空")

    # 提取群组级别信息
    group_name = data.get("group_name")
    # 提取场景信息（新增字段，兼容性处理）
    scene = data.get("scene")
    scene_desc = data.get("scene_desc")

    # 转换为 RawData 格式，传递群组名称
    raw_data_list = await convert_conversation_to_raw_data_list(
        messages, group_name=group_name
    )

    # 提取 current_time
    current_time = _extract_current_time(data)

    # 提取参与者
    participants = []

    # 计算分割点（80%作为历史消息）
    split_ratio = data.get("split_ratio", 0.8)
    split_index = int(len(raw_data_list) * split_ratio)

    # 分割历史消息和新消息
    history_raw_data_list = raw_data_list[:split_index]
    new_raw_data_list = raw_data_list[split_index:]

    # 如果没有新消息，将最后一条消息作为新消息
    if not new_raw_data_list and history_raw_data_list:
        new_raw_data_list = [history_raw_data_list.pop()]

    return _create_memorize_request(
        history_data=history_raw_data_list,
        new_data=new_raw_data_list,
        data_type=RawDataType(data.get(DataFields.RAW_DATA_TYPE, "Conversation")),
        participants=participants,
        group_id=data.get(DataFields.GROUP_ID),
        group_name=data.get("group_name"),
        current_time=current_time,
        scene=scene,
        scene_desc=scene_desc,
    )
