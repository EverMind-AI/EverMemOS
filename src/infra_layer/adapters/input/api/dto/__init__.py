"""
API DTO module

Defines data transfer objects for API requests and responses
"""

from .status_dto import RequestStatusResponse, RequestStatusData
from .memory_dto import (
    MemorizeMessageRequest,
    FetchMemoriesParams,
    SearchMemoriesRequest,
    ConversationMetaCreateRequest,
    ConversationMetaPatchRequest,
    UserDetailRequest,
)

__all__ = [
    # Status DTOs
    "RequestStatusResponse",
    "RequestStatusData",
    # Memory DTOs
    "MemorizeMessageRequest",
    "FetchMemoriesParams",
    "SearchMemoriesRequest",
    "ConversationMetaCreateRequest",
    "ConversationMetaPatchRequest",
    "UserDetailRequest",
]
