# -*- coding: utf-8 -*-
"""
MemoryRequestLog <-> RawData Converter

Handles bidirectional conversion between MemoryRequestLog and RawData.
"""

import json
from typing import Optional, List, Dict, Any

from core.observation.logger import get_logger
from common_utils.datetime_utils import from_iso_format
from zoneinfo import ZoneInfo
from api_specs.dtos.memory_command import RawData
from api_specs.request_converter import (
    build_raw_data_from_simple_message,
    normalize_refer_list,
)
from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
    MemoryRequestLog,
)

logger = get_logger(__name__)


class MemoryRequestLogMapper:
    """
    MemoryRequestLog <-> RawData Converter

    Provides bidirectional conversion between MemoryRequestLog and RawData.
    """

    @staticmethod
    def to_raw_data(log: MemoryRequestLog) -> Optional[RawData]:
        """
        Convert MemoryRequestLog to RawData

        Conversion strategy (by priority):
        1. First, parse simple message format from raw_input_str
        2. Then, parse simple message format from raw_input dictionary
        3. Finally, build from individual fields

        Args:
            log: MemoryRequestLog object

        Returns:
            RawData object or None (if conversion fails)
        """
        if log is None:
            return None

        # Strategy 1: First, parse simple message format from raw_input_str
        if log.raw_input_str:
            try:
                data = json.loads(log.raw_input_str)
                raw_data = MemoryRequestLogMapper._convert_simple_message_to_raw_data(
                    data, log.request_id
                )
                if raw_data:
                    return raw_data
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug(
                    "Failed to parse from raw_input_str, trying other methods: %s", e
                )

        # Strategy 2: Use raw_input dictionary to parse simple message format
        if log.raw_input:
            raw_data = MemoryRequestLogMapper._convert_simple_message_to_raw_data(
                log.raw_input, log.request_id
            )
            if raw_data:
                return raw_data

        # Strategy 3: Build from individual fields
        return MemoryRequestLogMapper._build_from_fields(log)

    @staticmethod
    def _convert_simple_message_to_raw_data(
        message_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> Optional[RawData]:
        """
        Convert simple message format to RawData

        Simple message format: {"message_id": "...", "sender": "...", "content": "...", ...}

        Args:
            message_data: Dictionary containing simple message data
            request_id: Request ID (optional, used in metadata)

        Returns:
            RawData object or None
        """
        if not isinstance(message_data, dict):
            return None

        message_id = message_data.get("message_id")
        sender = message_data.get("sender")
        content = message_data.get("content", "")
        create_time_str = message_data.get("create_time")

        if not message_id or not sender:
            return None

        # Parse timestamp
        timestamp = None
        if create_time_str:
            try:
                if isinstance(create_time_str, str):
                    timestamp = from_iso_format(create_time_str, ZoneInfo("UTC"))
                else:
                    timestamp = create_time_str
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to parse create_time: %s, error: %s", create_time_str, e
                )

        # Normalize refer_list
        refer_list = normalize_refer_list(message_data.get("refer_list", []))

        # Build extra_metadata
        extra_metadata = {"request_id": request_id} if request_id else None

        return build_raw_data_from_simple_message(
            message_id=message_id,
            sender=sender,
            content=content,
            timestamp=timestamp,
            sender_name=message_data.get("sender_name"),
            role=message_data.get("role"),
            group_id=message_data.get("group_id"),
            group_name=message_data.get("group_name"),
            refer_list=refer_list,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def _build_from_fields(log: MemoryRequestLog) -> RawData:
        """
        Build RawData from individual fields of MemoryRequestLog

        Use the unified build_raw_data_from_simple_message function to ensure field consistency.

        Args:
            log: MemoryRequestLog object

        Returns:
            RawData object
        """
        # Handle timestamp
        timestamp = None
        if log.message_create_time:
            try:
                # If it's a string, parse it into datetime
                if isinstance(log.message_create_time, str):
                    timestamp = from_iso_format(
                        log.message_create_time, ZoneInfo("UTC")
                    )
                else:
                    timestamp = log.message_create_time
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to parse message_create_time: %s, error: %s",
                    log.message_create_time,
                    e,
                )
                timestamp = None

        # Use unified build function
        return build_raw_data_from_simple_message(
            message_id=log.message_id or str(log.id),
            sender=log.sender or "",
            content=log.content or "",
            timestamp=timestamp,
            sender_name=log.sender_name,
            role=log.role,
            group_id=log.group_id,
            group_name=log.group_name,
            refer_list=log.refer_list or [],
            extra_metadata={"request_id": log.request_id},
        )

    @staticmethod
    def to_raw_data_list(logs: List[MemoryRequestLog]) -> List[RawData]:
        """
        Batch convert a list of MemoryRequestLog objects to a list of RawData objects

        Args:
            logs: List of MemoryRequestLog objects

        Returns:
            List of RawData objects (skip records that fail conversion)
        """
        raw_data_list: List[RawData] = []

        for log in logs:
            try:
                raw_data = MemoryRequestLogMapper.to_raw_data(log)
                if raw_data:
                    raw_data_list.append(raw_data)
            except (ValueError, TypeError) as e:
                logger.error(
                    "❌ Failed to convert MemoryRequestLog to RawData: log_id=%s, error=%s",
                    log.id,
                    e,
                )
                continue

        return raw_data_list
