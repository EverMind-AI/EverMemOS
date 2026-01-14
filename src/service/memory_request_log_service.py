# -*- coding: utf-8 -*-
"""
Memory Request Logging Service

Directly extract data from MemorizeRequest and save to MemoryRequestLog,
replacing the original event listener approach to make timing more controllable.
"""

import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from common_utils.datetime_utils import to_iso_format
from core.di import service
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger
from core.context.context import get_current_app_info
from core.oxm.constants import MAGIC_ALL
from api_specs.dtos.memory_command import MemorizeRequest, RawData
from api_specs.dtos.memory_query import PendingMessage
from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
    MemoryRequestLog,
)
from infra_layer.adapters.out.persistence.repository.memory_request_log_repository import (
    MemoryRequestLogRepository,
)

logger = get_logger(__name__)


@service("memory_request_log_service")
class MemoryRequestLogService:
    """
    Memory Request Logging Service

    Extract each message from new_raw_data_list in MemorizeRequest and save to MemoryRequestLog.
    Return the list of saved message_ids for use in subsequent processes.
    """

    def __init__(self):
        self._repository: Optional[MemoryRequestLogRepository] = None

    def _get_repository(self) -> MemoryRequestLogRepository:
        """Get Repository (lazy loading)"""
        if self._repository is None:
            self._repository = get_bean_by_type(MemoryRequestLogRepository)
        return self._repository

    async def save_request_logs(
        self,
        request: MemorizeRequest,
        version: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        method: Optional[str] = None,
        url: Optional[str] = None,
        raw_input_dict: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Extract data from MemorizeRequest and save to MemoryRequestLog

        Iterate through each RawData in new_raw_data_list, extract core fields and save.
        Saved records have sync_status=-1 (pending confirmation).

        Args:
            request: MemorizeRequest object
            version: API version (optional)
            endpoint_name: Endpoint name (optional)
            method: HTTP method (optional)
            url: Request URL (optional)
            raw_input_dict: Raw input dictionary (optional, used to generate raw_input_str)

        Returns:
            List[str]: List of saved message_ids
        """
        if not request.new_raw_data_list:
            logger.debug("new_raw_data_list is empty, skipping save")
            return []

        # Get current request context information
        app_info = get_current_app_info()
        request_id = app_info.get("request_id", "unknown")

        saved_message_ids = []
        repo = self._get_repository()

        for raw_data in request.new_raw_data_list:
            try:
                message_id = await self._save_single_raw_data(
                    raw_data=raw_data,
                    group_id=request.group_id,
                    group_name=request.group_name,
                    request_id=request_id,
                    repo=repo,
                    version=version,
                    endpoint_name=endpoint_name,
                    method=method,
                    url=url,
                    event_id=request_id,  # Use request_id as event_id
                    raw_input_dict=raw_input_dict,
                )
                if message_id:
                    saved_message_ids.append(message_id)
            except Exception as e:
                logger.error(
                    "Failed to save RawData to MemoryRequestLog: data_id=%s, error=%s",
                    raw_data.data_id,
                    e,
                )

        logger.info(
            "Saved %d request logs: group_id=%s, message_ids=%s",
            len(saved_message_ids),
            request.group_id,
            saved_message_ids,
        )

        return saved_message_ids

    async def _save_single_raw_data(
        self,
        raw_data: RawData,
        group_id: Optional[str],
        group_name: Optional[str],
        request_id: str,
        repo: MemoryRequestLogRepository,
        version: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        method: Optional[str] = None,
        url: Optional[str] = None,
        event_id: Optional[str] = None,
        raw_input_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Save a single RawData to MemoryRequestLog

        Args:
            raw_data: RawData object
            group_id: Group ID
            group_name: Group name
            request_id: Request ID
            repo: Repository instance
            version: API version
            endpoint_name: Endpoint name
            method: HTTP method
            url: Request URL
            event_id: Event ID
            raw_input_dict: Raw input dictionary (used to generate raw_input_str)

        Returns:
            Optional[str]: Returns message_id if saved successfully, None otherwise
        """
        if not group_id:
            logger.debug("group_id is empty, skipping save")
            return None

        # Extract fields from RawData
        content_dict = raw_data.content or {}
        message_id = raw_data.data_id

        # Extract core message fields
        # Note: Field names used by build_raw_data_from_simple_message differ from simple message format
        # speaker_id / createBy -> sender
        # speaker_name -> sender_name
        # timestamp / createTime -> message_create_time
        # referList -> refer_list
        sender = (
            content_dict.get("speaker_id")
            or content_dict.get("createBy")
            or content_dict.get("sender")
        )
        sender_name = (
            content_dict.get("speaker_name")
            or content_dict.get("sender_name")
            or sender
        )
        content = content_dict.get("content")
        # Message sender role ("user" for human, "assistant" for AI)
        role = content_dict.get("role")
        # Support multiple timestamp field names
        message_create_time = self._parse_create_time(
            content_dict.get("timestamp")
            or content_dict.get("createTime")
            or content_dict.get("create_time")
        )
        # Support multiple refer list field names
        refer_list = content_dict.get("referList") or content_dict.get("refer_list")

        # Generate raw_input_str
        raw_input_str = None
        if raw_input_dict:
            try:
                raw_input_str = json.dumps(raw_input_dict, ensure_ascii=False)
            except (TypeError, ValueError):
                pass

        # Create MemoryRequestLog document
        memory_request_log = MemoryRequestLog(
            # Core identifier fields
            group_id=group_id,
            request_id=request_id,
            user_id=sender,
            # Message core fields
            message_id=message_id,
            message_create_time=message_create_time,
            sender=sender,
            sender_name=sender_name,
            role=role,
            content=content,
            group_name=group_name,
            refer_list=self._normalize_refer_list(refer_list),
            # Raw input
            raw_input=raw_input_dict or content_dict,
            raw_input_str=raw_input_str,
            # Request metadata
            version=version,
            endpoint_name=endpoint_name,
            method=method,
            url=url,
            # Event association
            event_id=event_id,
            # sync_status=-1 indicates a newly saved log record
        )

        # Save to MongoDB
        await repo.save(memory_request_log)

        logger.debug(
            "Saved request log: group_id=%s, message_id=%s, content_preview=%s",
            group_id,
            message_id,
            (content or "")[:50],
        )

        return message_id

    def _parse_create_time(self, create_time: Any) -> Optional[str]:
        """Parse creation time and return ISO format string"""
        if create_time is None:
            return None
        if isinstance(create_time, datetime):
            return create_time.isoformat()
        if isinstance(create_time, str):
            # Validate if it's a valid time format, return directly if so
            try:
                from common_utils.datetime_utils import from_iso_format

                parsed = from_iso_format(create_time)
                return parsed.isoformat() if parsed else create_time
            except Exception:
                # Parsing failed, return original string
                return create_time
        return None

    def _normalize_refer_list(self, refer_list: Any) -> Optional[List[str]]:
        """
        Normalize refer_list to a list of strings

        Args:
            refer_list: Original refer_list, could be a list of strings or dictionaries

        Returns:
            Normalized list of strings
        """
        if not refer_list:
            return None

        if not isinstance(refer_list, list):
            return None

        result = []
        for item in refer_list:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # If it's a dictionary, try to extract message_id
                msg_id = item.get("message_id") or item.get("id")
                if msg_id:
                    result.append(str(msg_id))

        return result if result else None

    # ==================== Query Methods ====================

    async def get_pending_request_logs(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        sync_status_list: Optional[List[int]] = None,
        limit: int = 1000,
        skip: int = 0,
        ascending: bool = True,
    ) -> List[MemoryRequestLog]:
        """
        Get pending (unconsumed) Memory request logs

        Query request logs that have not been consumed yet (sync_status=-1 or 0).
        Supports flexible filtering with MAGIC_ALL logic:
        - MAGIC_ALL ("__all__"): Don't filter by this field
        - None or "": Filter for null/empty values
        - Other values: Exact match

        Args:
            user_id: User ID filter
                - MAGIC_ALL: Don't filter by user_id (default)
                - None or "": Filter for null/empty values
                - Other values: Exact match
            group_id: Group ID filter
                - MAGIC_ALL: Don't filter by group_id (default)
                - None or "": Filter for null/empty values
                - Other values: Exact match
            sync_status_list: List of sync_status values to filter by
                - Default: [-1, 0] (pending and accumulating, i.e., unconsumed)
                - [-1]: Just log records
                - [0]: In window accumulation
                - [1]: Already fully used
            limit: Maximum number of records to return (default 100)
            skip: Number of records to skip (default 0)
            ascending: If True (default), sort by created_at ascending (oldest first);
                       if False, sort descending (newest first)

        Returns:
            List[MemoryRequestLog]: List of pending request logs
        """
        # Default to unconsumed statuses
        if sync_status_list is None:
            sync_status_list = [-1, 0]

        repo = self._get_repository()

        try:
            results = await repo.find_pending_by_filters(
                user_id=user_id,
                group_id=group_id,
                sync_status_list=sync_status_list,
                limit=limit,
                skip=skip,
                ascending=ascending,
            )

            logger.debug(
                "Retrieved pending request logs: user_id=%s, group_id=%s, "
                "sync_status_list=%s, count=%d",
                user_id,
                group_id,
                sync_status_list,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "Failed to get pending request logs: user_id=%s, group_id=%s, error=%s",
                user_id,
                group_id,
                e,
            )
            return []

    async def get_pending_messages(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        limit: int = 1000,
    ) -> List[PendingMessage]:
        """
        Get pending (unconsumed) messages as list of PendingMessage objects.

        This is a convenience method that wraps get_pending_request_logs
        and converts the results to PendingMessage dataclass instances.

        Args:
            user_id: User ID filter (MAGIC_ALL to skip filtering)
            group_id: Group ID filter (MAGIC_ALL to skip filtering)
            limit: Maximum number of records to return (default 1000)

        Returns:
            List[PendingMessage]: List of pending messages
        """
        logs = await self.get_pending_request_logs(
            user_id=user_id, group_id=group_id, limit=limit
        )

        # Convert to list of PendingMessage
        result = []
        for log in logs:
            pending_msg = PendingMessage(
                id=str(log.id),
                request_id=log.request_id,
                message_id=log.message_id,
                group_id=log.group_id,
                user_id=log.user_id,
                sender=log.sender,
                sender_name=log.sender_name,
                group_name=log.group_name,
                content=log.content,
                refer_list=log.refer_list,
                message_create_time=log.message_create_time,
                created_at=to_iso_format(log.created_at),
                updated_at=to_iso_format(log.updated_at),
            )
            result.append(pending_msg)

        logger.debug(
            "Converted %d pending request logs to PendingMessage: user_id=%s, group_id=%s",
            len(result),
            user_id,
            group_id,
        )
        return result
