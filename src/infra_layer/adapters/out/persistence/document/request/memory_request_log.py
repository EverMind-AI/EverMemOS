# -*- coding: utf-8 -*-
"""
MemoryRequestLog MongoDB Document Model

Stores key information from memories requests, used to replace the functionality of conversation_data.
Primarily saves message content from memorize requests, which can later be used to replace RawData storage in Redis.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from core.oxm.mongo.document_base import DocumentBase
from core.oxm.mongo.audit_base import AuditBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING


class MemoryRequestLog(DocumentBase, AuditBase):
    """
    Memory Request Log Document Model

    Stores key information from memories interface requests:
    - group_id: conversation group ID
    - request_id: request ID
    - user_id: user ID
    - raw_input: raw input data
    - message core fields: message_id, create_time, sender, sender_name, content, etc.
    """

    # Core fields
    group_id: str = Field(..., description="Conversation group ID")
    request_id: str = Field(..., description="Request ID")
    user_id: Optional[str] = Field(default=None, description="User ID")

    # ========== Message core fields (used to replace RawData) ==========
    # Refer to field definitions in group_chat_converter.py
    message_id: Optional[str] = Field(default=None, description="Message ID")
    message_create_time: Optional[str] = Field(
        default=None, description="Message creation time (ISO 8601 format)"
    )
    sender: Optional[str] = Field(default=None, description="Sender ID")
    sender_name: Optional[str] = Field(default=None, description="Sender name")
    role: Optional[str] = Field(
        default=None,
        description="Message sender role: 'user' for human, 'assistant' for AI",
    )
    content: Optional[str] = Field(default=None, description="Message content")
    group_name: Optional[str] = Field(default=None, description="Group name")
    refer_list: Optional[List[str]] = Field(
        default=None, description="List of referenced message IDs"
    )

    # Raw input (retained for debugging and integrity)
    raw_input: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw input data (parsed JSON body)"
    )
    raw_input_str: Optional[str] = Field(default=None, description="Raw input string")

    # Request metadata
    version: Optional[str] = Field(default=None, description="Code version")
    endpoint_name: Optional[str] = Field(default=None, description="Endpoint name")
    method: Optional[str] = Field(default=None, description="HTTP method")
    url: Optional[str] = Field(default=None, description="Request URL")

    # Original event ID (used to associate with RequestHistory)
    event_id: Optional[str] = Field(default=None, description="Original event ID")

    # Sync status field (numeric)
    # -1: log record only (raw request just saved via listener)
    #  0: accumulating in window (confirmed entering accumulation window via save_conversation_data)
    #  1: already fully used (marked via delete_conversation_data, after boundary detection)
    sync_status: int = Field(
        default=-1,
        description="Sync status: -1=log record, 0=window accumulating, 1=already used",
    )

    model_config = ConfigDict(
        collection="memory_request_logs",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "group_id": "group_123",
                "request_id": "req_456",
                "user_id": "user_789",
                "message_id": "msg_001",
                "message_create_time": "2024-01-01T12:00:00+08:00",
                "sender": "user_789",
                "sender_name": "Zhang San",
                "content": "This is a test message",
                "group_name": "Test Group",
                "refer_list": [],
                "raw_input": {
                    "message_id": "msg_001",
                    "content": "This is a test message",
                },
                "version": "1.0.0",
                "endpoint_name": "memorize",
            }
        },
    )

    class Settings:
        """Beanie settings"""

        name = "memory_request_logs"
        indexes = [
            IndexModel([("group_id", ASCENDING), ("created_at", DESCENDING)]),
            IndexModel([("request_id", ASCENDING)]),
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("event_id", ASCENDING)]),
            IndexModel([("message_id", ASCENDING)]),
            IndexModel([("group_id", ASCENDING), ("message_create_time", DESCENDING)]),
            # Composite index: used for batch updates and querying by status
            # Supports operations like update_many({"group_id": "xxx", "sync_status": -1}, ...)
            IndexModel([("group_id", ASCENDING), ("sync_status", ASCENDING)]),
            IndexModel(
                [
                    ("group_id", ASCENDING),
                    ("user_id", ASCENDING),
                    ("sync_status", ASCENDING),
                ]
            ),
        ]
