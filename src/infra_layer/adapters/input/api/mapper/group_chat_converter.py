"""
群聊格式映射模块

将开源群聊格式（GroupChatFormat）转换为 memorize 接口所需的格式

⚠️ 重要说明：
这是 GroupChatFormat 到内部格式的唯一适配层，所有相关的转换逻辑都必须集中在此文件中。
禁止在其他模块（如 controller、service 等）中添加格式转换逻辑，以保持适配层的单一职责。

主要功能：
1. 格式验证：validate_group_chat_format_input() - 验证输入数据是否符合 GroupChatFormat 规范
2. 格式转换：convert_group_chat_format_to_memorize_input() - 将 GroupChatFormat 转换为内部格式
3. 消息转换：_convert_message_to_internal_format() - 转换单条消息
4. 时间处理：_parse_datetime_with_timezone() - 处理带时区的时间字符串

使用示例：
    from infra_layer.adapters.input.api.mapper.group_chat_converter import (
        validate_group_chat_format_input,
        convert_group_chat_format_to_memorize_input
    )

    # 验证格式
    if validate_group_chat_format_input(data):
        # 转换为内部格式
        memorize_input = convert_group_chat_format_to_memorize_input(data)
        # 使用 memorize_input 调用记忆存储服务
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from common_utils.datetime_utils import from_iso_format


def convert_group_chat_format_to_memorize_input(
    group_chat_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    将 GroupChatFormat 格式转换为 memorize 接口输入格式

    Args:
        group_chat_data: GroupChatFormat 格式的数据字典，包含：
            - version: 格式版本号
            - conversation_meta: 会话元信息（name, description, group_id, user_details等）
            - conversation_list: 消息列表

    Returns:
        Dict[str, Any]: 适合 memorize 接口的输入格式，包含：
            - messages: 转换后的消息列表
            - group_id: 群组ID（可选）
            - raw_data_type: 数据类型（默认为 "Conversation"）
            - current_time: 当前时间（使用最后一条消息的时间）

    Raises:
        ValueError: 当必需字段缺失时
    """
    # 验证必需字段
    if "conversation_meta" not in group_chat_data:
        raise ValueError("缺少必需字段: conversation_meta")
    if "conversation_list" not in group_chat_data:
        raise ValueError("缺少必需字段: conversation_list")

    conversation_meta = group_chat_data["conversation_meta"]
    conversation_list = group_chat_data["conversation_list"]

    if not conversation_list:
        raise ValueError("conversation_list 不能为空")

    # 提取群组信息
    group_id = conversation_meta.get("group_id")
    group_name = conversation_meta.get("name")  # 从 conversation_meta 中提取群组名称
    user_details = conversation_meta.get("user_details", {})
    default_timezone = conversation_meta.get("default_timezone", "UTC")
    # 提取场景信息（新增字段，兼容性处理）
    scene = conversation_meta.get("scene")
    scene_desc = conversation_meta.get("scene_desc", {})

    # 提取所有用户ID列表
    user_id_list = list(user_details.keys())

    # 转换消息列表
    messages = []
    for msg in conversation_list:
        converted_msg = _convert_message_to_internal_format(
            msg,
            group_id=group_id,
            user_id_list=user_id_list,
            user_details=user_details,
            default_timezone=default_timezone,
        )
        messages.append(converted_msg)

    # 获取最后一条消息的时间作为 current_time
    current_time = None
    if messages:
        last_msg = conversation_list[-1]
        create_time_str = last_msg.get("create_time")
        if create_time_str:
            current_time = _parse_datetime_with_timezone(
                create_time_str, default_timezone
            )

    # 构建 memorize 接口输入格式
    result = {"messages": messages, "raw_data_type": "Conversation"}

    # 添加可选字段
    if group_id:
        result["group_id"] = group_id
    if group_name:
        result["group_name"] = group_name
    if current_time:
        result["current_time"] = current_time.isoformat()
    # 添加场景信息（新增字段，兼容性处理）
    if scene:
        result["scene"] = scene
    if scene_desc:
        result["scene_desc"] = scene_desc

    return result


def _convert_message_to_internal_format(
    message: Dict[str, Any],
    group_id: Optional[str] = None,
    user_id_list: Optional[List[str]] = None,
    user_details: Optional[Dict[str, Any]] = None,
    default_timezone: str = "UTC",
) -> Dict[str, Any]:
    """
    将单条消息从 GroupChatFormat 格式转换为内部格式

    Args:
        message: GroupChatFormat 中的单条消息，包含：
            - message_id: 消息ID
            - create_time: 创建时间（ISO 8601格式）
            - sender: 发送者用户ID
            - sender_name: 发送者名称（可选）
            - type: 消息类型
            - content: 消息内容
            - refer_list: 引用消息列表（可选）
        group_id: 群组ID
        user_id_list: 用户ID列表
        user_details: 用户详细信息字典
        default_timezone: 默认时区

    Returns:
        Dict[str, Any]: 转换后的消息格式，包含 _id, fullName, receiverId, roomId,
                       userIdList, referList, content, createTime, createBy, updateTime, orgId 等
    """
    # 提取基本字段
    message_id = message.get("message_id")
    sender_id = message.get("sender")
    sender_name = message.get("sender_name")
    create_time = message.get("create_time")
    content = message.get("content", "")
    refer_list = message.get("refer_list", [])

    # 消息类型：目前只支持文本消息，固定为 1 (TEXT)
    msg_type = 1

    # 如果 sender_name 为空，尝试从 user_details 中获取
    if not sender_name and sender_id and user_details:
        user_detail = user_details.get(sender_id, {})
        sender_name = user_detail.get("full_name", sender_id)

    # 转换 refer_list 格式
    # GroupChatFormat 支持两种格式：字符串列表或 MessageReference 对象列表
    # 需要转换为字符串列表（message_id）
    converted_refer_list = []
    if refer_list:
        for refer in refer_list:
            if isinstance(refer, str):
                # 已经是 message_id 字符串
                converted_refer_list.append(refer)
            elif isinstance(refer, dict):
                # MessageReference 对象，提取 message_id
                ref_msg_id = refer.get("message_id")
                if ref_msg_id:
                    converted_refer_list.append(ref_msg_id)

    # 解析时间（如果没有时区信息，使用默认时区）
    parsed_create_time = _parse_datetime_with_timezone(create_time, default_timezone)
    create_time_iso = (
        parsed_create_time.isoformat() if parsed_create_time else create_time
    )

    # 构建内部格式
    # 注意：这里的格式需要匹配 convert_single_message_to_raw_data 的期望输入
    internal_format = {
        "_id": message_id,
        "fullName": sender_name or sender_id,
        "receiverId": None,  # 群聊消息没有单独的接收者
        "roomId": group_id,
        "userIdList": user_id_list or [],
        "referList": converted_refer_list,
        "content": content,
        "createTime": create_time_iso,
        "createBy": sender_id,
        "updateTime": create_time_iso,  # 使用 createTime 作为 updateTime
        "orgId": None,  # GroupChatFormat 中没有 orgId，设为 None
        "msgType": msg_type,
    }

    # 添加额外信息到 extra 字段（如果有）
    extra = message.get("extra")
    if extra:
        internal_format["extra"] = extra

    return internal_format


def _parse_datetime_with_timezone(
    datetime_str: Optional[str], default_timezone: str = "UTC"
) -> Optional[datetime]:
    """
    解析带时区的日期时间字符串

    Args:
        datetime_str: ISO 8601 格式的日期时间字符串
        default_timezone: 如果字符串中没有时区信息，使用的默认时区

    Returns:
        datetime 对象，如果解析失败则返回 None
    """
    if not datetime_str:
        return None

    try:
        # 尝试使用 from_iso_format 解析
        # 如果没有时区信息，会使用提供的默认时区
        tz = ZoneInfo(default_timezone)
        return from_iso_format(datetime_str, tz)
    except (ValueError, TypeError, KeyError) as e:
        # 解析失败，返回 None
        print(f"解析日期时间失败: {datetime_str}, 错误: {e}")
        return None


def convert_simple_message_to_memorize_input(
    message_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    将简单直接的单条消息格式转换为 memorize 接口输入格式

    这是 V3 memorize 接口使用的简单格式，无需复杂的 GroupChatFormat 结构。

    Args:
        message_data: 简单的单条消息数据，包含：
            - group_id (可选): 群组ID
            - group_name (可选): 群组名称
            - message_id (必需): 消息ID
            - create_time (必需): 创建时间
            - sender (必需): 发送者用户ID
            - sender_name (可选): 发送者名称
            - content (必需): 消息内容
            - refer_list (可选): 引用消息ID列表

    Returns:
        Dict[str, Any]: 适合 memorize 接口的输入格式

    Raises:
        ValueError: 当必需字段缺失时
    """
    # 提取字段
    group_id = message_data.get("group_id")
    group_name = message_data.get("group_name")
    message_id = message_data.get("message_id")
    create_time = message_data.get("create_time")
    sender = message_data.get("sender")
    sender_name = message_data.get("sender_name", sender)
    content = message_data.get("content", "")
    refer_list = message_data.get("refer_list", [])
    # 提取场景信息（新增字段，兼容性处理）
    scene = message_data.get("scene")
    scene_desc = message_data.get("scene_desc")

    # 验证必需字段
    if not message_id:
        raise ValueError("缺少必需字段: message_id")
    if not create_time:
        raise ValueError("缺少必需字段: create_time")
    if not sender:
        raise ValueError("缺少必需字段: sender")
    if not content:
        raise ValueError("缺少必需字段: content")

    # 构建内部格式
    internal_message = {
        "_id": message_id,
        "fullName": sender_name,
        "receiverId": None,
        "roomId": group_id,
        "userIdList": [],
        "referList": refer_list if isinstance(refer_list, list) else [],
        "content": content,
        "createTime": create_time,
        "createBy": sender,
        "updateTime": create_time,
        "orgId": None,
        "msgType": 1,  # TEXT
    }

    # 构建 memorize 接口输入格式
    result = {
        "messages": [internal_message],
        "raw_data_type": "Conversation",
        "split_ratio": 0,  # 全部作为新消息
    }

    # 添加可选字段
    if group_id:
        result["group_id"] = group_id
    if group_name:
        result["group_name"] = group_name
    if create_time:
        result["current_time"] = create_time
    # 添加场景信息（新增字段，兼容性处理）
    if scene:
        result["scene"] = scene
    if scene_desc:
        result["scene_desc"] = scene_desc

    return result


def validate_group_chat_format_input(data: Dict[str, Any]) -> bool:
    """
    验证输入数据是否符合 GroupChatFormat 规范

    Args:
        data: 输入数据字典

    Returns:
        bool: 是否符合规范
    """
    # 检查必需的顶层字段
    if "conversation_meta" not in data or "conversation_list" not in data:
        return False

    meta = data["conversation_meta"]
    if "name" not in meta or "user_details" not in meta:
        return False

    # 检查消息列表
    conversation_list = data["conversation_list"]
    if not isinstance(conversation_list, list):
        return False

    user_ids = set(meta["user_details"].keys())

    # 验证每条消息
    for msg in conversation_list:
        # 检查必需字段
        required_fields = ["message_id", "create_time", "sender", "type", "content"]
        for field in required_fields:
            if field not in msg:
                return False

        # 检查 sender 是否在 user_details 中
        if msg.get("sender") not in user_ids:
            return False

        # 检查 refer_list 格式（如果有）
        refer_list = msg.get("refer_list", [])
        if refer_list:
            for refer in refer_list:
                if isinstance(refer, dict):
                    # MessageReference 对象必须有 message_id
                    if "message_id" not in refer:
                        return False
                elif not isinstance(refer, str):
                    # 必须是字符串或字典
                    return False

    return True
