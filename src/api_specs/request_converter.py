"""
Request converter module

This module contains various functions to convert external request formats to internal Request objects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from api_specs.memory_models import MemoryType
from api_specs.dtos.memory_query import RetrieveMemRequest, FetchMemRequest
from api_specs.dtos.memory_command import MemorizeRequest
from api_specs.memory_types import RawDataType
from api_specs.dtos.memory_command import RawData
from core.oxm.constants import MAGIC_ALL

from typing import Dict, Any, Optional
from common_utils.datetime_utils import from_iso_format
from zoneinfo import ZoneInfo
from core.observation.logger import get_logger
from api_specs.memory_models import RetrieveMethod, MemoryType

logger = get_logger(__name__)


class DataFields:
    """Data field constants"""

    MESSAGES = "messages"
    RAW_DATA_TYPE = "raw_data_type"
    GROUP_ID = "group_id"


def convert_dict_to_fetch_mem_request(data: Dict[str, Any]) -> FetchMemRequest:
    """
    Convert dictionary to FetchMemRequest object

    Args:
        data: Dictionary containing FetchMemRequest fields

    Returns:
        FetchMemRequest object

    Raises:
        ValueError: When required fields are missing or have incorrect types
    """
    try:
        # Convert memory_type, use default if not provided
        memory_type = MemoryType(
            data.get("memory_type", MemoryType.EPISODIC_MEMORY.value)
        )
        logger.debug(f"version_range: {data.get('version_range', None)}")

        # Convert limit and offset to integer type (all obtained from query_params are strings)
        limit = data.get("limit", 10)
        offset = data.get("offset", 0)
        if isinstance(limit, str):
            limit = int(limit)
        if isinstance(offset, str):
            offset = int(offset)

        # Build FetchMemRequest object
        return FetchMemRequest(
            user_id=data.get(
                "user_id", MAGIC_ALL
            ),  # User ID, use MAGIC_ALL to skip user filtering
            group_id=data.get(
                "group_id", MAGIC_ALL
            ),  # Group ID, use MAGIC_ALL to skip group filtering
            memory_type=memory_type,
            limit=limit,
            offset=offset,
            sort_by=data.get("sort_by"),
            sort_order=data.get("sort_order", "desc"),
            version_range=data.get("version_range", None),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
        )
    except Exception as e:
        raise ValueError(f"FetchMemRequest conversion failed: {e}")


def convert_dict_to_retrieve_mem_request(
    data: Dict[str, Any], query: Optional[str] = None
) -> RetrieveMemRequest:
    """
    Convert dictionary to RetrieveMemRequest object

    Args:
        data: Dictionary containing RetrieveMemRequest fields
        query: Query text (optional)

    Returns:
        RetrieveMemRequest object

    Raises:
        ValueError: When required fields are missing or have incorrect types
    """
    try:
        # Validate required fields: user_id or group_id at least one is required
        # if not data.get("user_id") and not data.get("group_id"):
        #     raise ValueError("user_id or group_id at least one is required")

        # Handle retrieve_method, use default keyword if not provided

        retrieve_method_str = data.get("retrieve_method", RetrieveMethod.KEYWORD.value)
        logger.debug(f"[DEBUG] retrieve_method_str from data: {retrieve_method_str!r}")

        # Convert string to RetrieveMethod enum
        try:
            retrieve_method = RetrieveMethod(retrieve_method_str)
            logger.debug(f"[DEBUG] converted to: {retrieve_method}")
        except ValueError:
            raise ValueError(
                f"Invalid retrieve_method: {retrieve_method_str}. "
                f"Supported methods: {[m.value for m in RetrieveMethod]}"
            )

        # Convert top_k to integer type (all obtained from query_params are strings)
        top_k = data.get("top_k", 10)
        if isinstance(top_k, str):
            top_k = int(top_k)

        # Convert include_metadata to boolean type
        include_metadata = data.get("include_metadata", True)
        if isinstance(include_metadata, str):
            include_metadata = include_metadata.lower() in ("true", "1", "yes")

        # Convert radius to float type (if exists)
        radius = data.get("radius", None)
        if radius is not None and isinstance(radius, str):
            radius = float(radius)

        # Convert memory_types string list to MemoryType enum list
        raw_memory_types = data.get("memory_types", [])
        # Handle comma-separated string (from query_params)
        if isinstance(raw_memory_types, str):
            raw_memory_types = [
                mt.strip() for mt in raw_memory_types.split(",") if mt.strip()
            ]
        memory_types = []
        for mt in raw_memory_types:
            if isinstance(mt, str):
                try:
                    memory_types.append(MemoryType(mt))
                except ValueError:
                    logger.error(f"Invalid memory_type: {mt}, skipping")
            elif isinstance(mt, MemoryType):
                memory_types.append(mt)
        # Default to EPISODIC_MEMORY if empty
        if not memory_types:
            memory_types = [MemoryType.EPISODIC_MEMORY]

        return RetrieveMemRequest(
            retrieve_method=retrieve_method,
            user_id=data.get(
                "user_id", MAGIC_ALL
            ),  # User ID, use MAGIC_ALL to skip user filtering
            group_id=data.get(
                "group_id", MAGIC_ALL
            ),  # Group ID, use MAGIC_ALL to skip group filtering
            query=query or data.get("query", None),
            memory_types=memory_types,
            top_k=top_k,
            include_metadata=include_metadata,
            start_time=data.get("start_time", None),
            end_time=data.get("end_time", None),
            radius=radius,  # COSINE similarity threshold
        )
    except Exception as e:
        raise ValueError(f"RetrieveMemRequest conversion failed: {e}")


# =========================================


def normalize_refer_list(refer_list: List[Any]) -> List[str]:
    """
    Normalize refer_list format to a list of message IDs

    Supports two formats:
    1. String list: ["msg_id_1", "msg_id_2"]
    2. MessageReference object list: [{"message_id": "msg_id_1", ...}, ...]

    Args:
        refer_list: Original reference list

    Returns:
        List[str]: Normalized list of message IDs
    """
    if not refer_list:
        return []

    normalized: List[str] = []
    for refer in refer_list:
        if isinstance(refer, str):
            normalized.append(refer)
        elif isinstance(refer, dict):
            ref_msg_id = refer.get("message_id")
            if ref_msg_id:
                normalized.append(str(ref_msg_id))
    return normalized


def build_raw_data_from_simple_message(
    message_id: str,
    sender: str,
    content: str,
    timestamp: datetime,
    sender_name: Optional[str] = None,
    role: Optional[str] = None,
    group_id: Optional[str] = None,
    group_name: Optional[str] = None,
    refer_list: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> RawData:
    """
    Build RawData object from simple message fields.

    This is the canonical function for creating RawData from simple message format.
    All code that needs to create RawData from simple messages should use this function
    to ensure consistency.

    Args:
        message_id: Message ID (required)
        sender: Sender user ID (required)
        content: Message content (required)
        timestamp: Message timestamp as datetime object (required)
        sender_name: Sender display name (defaults to sender if not provided)
        role: Message sender role, "user" for human or "assistant" for AI (optional)
        group_id: Group ID (optional)
        group_name: Group name (optional)
        refer_list: Normalized list of referenced message IDs (optional)
        extra_metadata: Additional metadata to merge (optional)

    Returns:
        RawData: Fully constructed RawData object
    """
    # Use sender as sender_name if not provided
    if sender_name is None:
        sender_name = sender

    # Ensure refer_list is a list
    if refer_list is None:
        refer_list = []

    # Build content dictionary with all required fields
    raw_content = {
        "speaker_name": sender_name,
        "role": role,  # Message sender role: "user" or "assistant"
        "receiverId": None,
        "roomId": group_id,
        "groupName": group_name,
        "userIdList": [],
        "referList": refer_list,
        "content": content,
        "timestamp": timestamp,
        "createBy": sender,
        "updateTime": timestamp,
        "orgId": None,
        "speaker_id": sender,
        "msgType": 1,  # TEXT
        "data_id": message_id,
    }

    # Build metadata
    metadata = {
        "original_id": message_id,
        "createTime": timestamp,
        "updateTime": timestamp,
        "createBy": sender,
        "orgId": None,
    }

    # Merge extra metadata if provided
    if extra_metadata:
        metadata.update(extra_metadata)

    return RawData(content=raw_content, data_id=message_id, metadata=metadata)


async def convert_simple_message_to_memorize_request(
    message_data: Dict[str, Any]
) -> MemorizeRequest:
    """
    Convert simple direct single message format directly to MemorizeRequest

    This is a unified conversion function that combines the previous two-step conversion
    (convert_simple_message_to_memorize_input + handle_conversation_format) into one.

    Args:
        message_data: Simple single message data, containing:
            - group_id (optional): Group ID
            - group_name (optional): Group name
            - message_id (required): Message ID
            - create_time (required): Creation time (ISO 8601 format)
            - sender (required): Sender user ID
            - sender_name (optional): Sender name
            - role (optional): Message sender role ("user" for human, "assistant" for AI)
            - content (required): Message content
            - refer_list (optional): List of referenced message IDs

    Returns:
        MemorizeRequest: Ready-to-use memorize request object

    Raises:
        ValueError: When required fields are missing
    """
    # Extract fields
    group_id = message_data.get("group_id")
    group_name = message_data.get("group_name")
    message_id = message_data.get("message_id")
    create_time_str = message_data.get("create_time")
    sender = message_data.get("sender")
    sender_name = message_data.get("sender_name", sender)
    role = message_data.get("role")  # "user" or "assistant"
    content = message_data.get("content", "")
    refer_list = message_data.get("refer_list", [])

    # Validate required fields
    if not message_id:
        raise ValueError("Missing required field: message_id")
    if not create_time_str:
        raise ValueError("Missing required field: create_time")
    if not sender:
        raise ValueError("Missing required field: sender")
    if not content:
        raise ValueError("Missing required field: content")

    # Normalize refer_list
    normalized_refer_list = normalize_refer_list(refer_list)

    # Parse timestamp
    timestamp = from_iso_format(create_time_str, ZoneInfo("UTC"))

    # Build RawData using the canonical function
    raw_data = build_raw_data_from_simple_message(
        message_id=message_id,
        sender=sender,
        content=content,
        timestamp=timestamp,
        sender_name=sender_name,
        role=role,
        group_id=group_id,
        group_name=group_name,
        refer_list=normalized_refer_list,
    )

    # Create and return MemorizeRequest
    return MemorizeRequest(
        history_raw_data_list=[],
        new_raw_data_list=[raw_data],
        raw_data_type=RawDataType.CONVERSATION,
        user_id_list=[],
        group_id=group_id,
        group_name=group_name,
        current_time=timestamp,
    )
