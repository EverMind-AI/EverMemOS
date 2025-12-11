"""
Memory Controller - Unified memory management controller

Provides complete memory management RESTful API routes, including:
- Memory storage (memorize): Receive simple direct single message format and store
- Conversation metadata (conversation-meta): Save conversation metadata information
- Memory retrieval (fetch): Retrieve user core memories via GET method
- Memory search (search): Support three retrieval methods: keyword, vector, and hybrid
"""

import json
import logging
from contextlib import suppress
from typing import Any, Dict
from fastapi import HTTPException, Request as FastAPIRequest

from core.di.decorators import controller
from core.di import get_bean_by_type
from core.interface.controller.base_controller import BaseController, get, post, patch
from core.constants.errors import ErrorCode, ErrorStatus
from agentic_layer.memory_manager import MemoryManager
from api_specs.request_converter import (
    handle_conversation_format,
    convert_dict_to_fetch_mem_request,
    convert_dict_to_retrieve_mem_request,
)
from api_specs.dtos.memory_query import ConversationMetaRequest, UserDetail
from infra_layer.adapters.input.api.mapper.group_chat_converter import (
    convert_simple_message_to_memorize_input,
)
from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
    ConversationMeta,
    UserDetailModel,
)
from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
    ConversationMetaRawRepository,
)
from component.redis_provider import RedisProvider
from component.timeout_background import timeout_to_background

logger = logging.getLogger(__name__)


@controller("memory_controller", primary=True)
class MemoryController(BaseController):
    """
    Memory Controller
    """

    def __init__(self, conversation_meta_repository: ConversationMetaRawRepository):
        """Initialize controller"""
        super().__init__(
            prefix="/api/v1/memories",
            tags=["Memory Controller"],
            default_auth="none",  # Adjust authentication strategy based on actual needs
        )
        self.memory_manager = MemoryManager()
        self.conversation_meta_repository = conversation_meta_repository
        # Get RedisProvider
        self.redis_provider = get_bean_by_type(RedisProvider)
        logger.info(
            "MemoryController initialized with MemoryManager and ConversationMetaRepository"
        )

    @post(
        "",
        response_model=Dict[str, Any],
        summary="Store single group chat message memory",
        description="""
        Receive simple direct single message format and store as memory
        
        ## Functionality:
        - Receive simple direct single message data (no pre-conversion required)
        - Extract single message into memory units (memcells)
        - Suitable for real-time message processing scenarios
        - Return list of saved memories
        
        ## Input format (simple direct):
        ```json
        {
          "group_id": "group_123",
          "group_name": "Project Discussion Group",
          "message_id": "msg_001",
          "create_time": "2025-01-15T10:00:00+00:00",
          "sender": "user_001",
          "sender_name": "Zhang San",
          "content": "Today let's discuss the technical solution for the new feature",
          "refer_list": ["msg_000"]
        }
        ```
        
        ## Field descriptions:
        - **group_id** (optional): Group ID
        - **group_name** (optional): Group name
        - **message_id** (required): Message ID
        - **create_time** (required): Message creation time (ISO 8601 format)
        - **sender** (required): Sender user ID
        - **sender_name** (optional): Sender name
        - **content** (required): Message content
        - **refer_list** (optional): List of referenced message IDs
        
        ## Interface description:
        - Receive simple direct single message format, no conversion required
        
        ## Use cases:
        - Real-time message stream processing
        - Chatbot integration
        - Message queue consumption
        - Single message import
        """,
        responses={
            200: {
                "description": "Successfully stored memory data",
                "content": {
                    "application/json": {
                        "examples": {
                            "extracted": {
                                "summary": "Extracted memories (boundary triggered)",
                                "value": {
                                    "status": "ok",
                                    "message": "Extracted 1 memories",
                                    "result": {
                                        "saved_memories": [
                                            {
                                                "memory_type": "episode_memory",
                                                "user_id": "user_001",
                                                "group_id": "group_123",
                                                "timestamp": "2025-01-15T10:00:00",
                                                "content": "User discussed the technical solution for the new feature",
                                            }
                                        ],
                                        "count": 1,
                                        "status_info": "extracted",
                                    },
                                },
                            },
                            "accumulated": {
                                "summary": "Message queued (boundary not triggered)",
                                "value": {
                                    "status": "ok",
                                    "message": "Message queued, awaiting boundary detection",
                                    "result": {
                                        "saved_memories": [],
                                        "count": 0,
                                        "status_info": "accumulated",
                                    },
                                },
                            },
                        }
                    }
                },
            },
            400: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Data format error: Required field message_id is missing",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories",
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "Failed to store memory, please try again later",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories",
                        }
                    }
                },
            },
        },
    )
    @timeout_to_background()
    async def memorize_single_message(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        Store single message memory data

        Receive simple direct single message format, convert and store via group_chat_converter

        Args:
            fastapi_request: FastAPI request object

        Returns:
            Dict[str, Any]: Memory storage response, containing list of saved memories

        Raises:
            HTTPException: When request processing fails
        """
        try:
            # 1. Get JSON body from request (simple direct format)
            message_data = await fastapi_request.json()
            logger.info("Received memorize request (single message)")

            # 3. Use group_chat_converter to convert to internal format
            logger.info(
                "Starting conversion from simple message format to internal format"
            )
            memorize_input = convert_simple_message_to_memorize_input(message_data)

            # Extract metadata for logging
            group_name = memorize_input.get("group_name")
            group_id = memorize_input.get("group_id")

            logger.info(
                "Conversion completed: group_id=%s, group_name=%s", group_id, group_name
            )

            # 4. Convert to MemorizeRequest object and call memory_manager
            logger.info("Starting to process memory request")
            memorize_request = await handle_conversation_format(memorize_input)
            # memorize returns count of extracted memories (int)
            memory_count = await self.memory_manager.memorize(memorize_request)

            # 5. Return unified response format
            logger.info(
                "Memory request processing completed, extracted %s memories",
                memory_count,
            )

            # Optimize return message to help users understand runtime status
            if memory_count > 0:
                message = f"Extracted {memory_count} memories"
            else:
                message = "Message queued, awaiting boundary detection"

            return {
                "status": ErrorStatus.OK.value,
                "message": message,
                "result": {
                    "saved_memories": [],  # Memories saved to DB, fetch via API
                    "count": memory_count,
                    "status_info": "accumulated" if memory_count == 0 else "extracted",
                },
            }

        except ValueError as e:
            logger.error("memorize request parameter error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # Re-raise HTTPException
            raise
        except Exception as e:
            logger.error("memorize request processing failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to store memory, please try again later"
            ) from e

    @get(
        "",
        response_model=Dict[str, Any],
        summary="Fetch user memories",
        description="""
        Retrieve user's core memory data via KV method
        
        ## Functionality:
        - Directly retrieve stored core memories based on user ID
        - Support multiple memory types: base memory, user profile, preference settings, etc.
        - Support pagination and sorting
        - Suitable for scenarios requiring quick retrieval of fixed user memory sets
        
        ## Memory type descriptions:
        - **base_memory**: Base memory, user's basic information and commonly used data
        - **profile**: User profile, containing user characteristics and attributes
        - **preference**: User preferences, containing user likes and settings
        - **episode_memory**: Episodic memory summaries
        - **multiple**: Multiple types (default), includes base_memory, profile, preference
        
        ## Use cases:
        - User profile display
        - Personalized recommendation systems
        - User preference settings loading
        """,
        responses={
            200: {
                "description": "Successfully retrieved memory data",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "message": "Memory retrieval successful",
                            "result": {
                                "memories": [
                                    {
                                        "memory_type": "base_memory",
                                        "user_id": "user_123",
                                        "timestamp": "2024-01-15T10:30:00",
                                        "content": "User likes drinking coffee",
                                        "summary": "Coffee preference",
                                    }
                                ],
                                "total_count": 100,
                                "has_more": False,
                                "metadata": {
                                    "source": "fetch_mem_service",
                                    "user_id": "user_123",
                                    "memory_type": "fetch",
                                },
                            },
                        }
                    }
                },
            },
            400: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "user_id cannot be empty",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/fetch",
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "Failed to retrieve memory, please try again later",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/fetch",
                        }
                    }
                },
            },
        },
    )
    async def fetch_memories(self, fastapi_request: FastAPIRequest) -> Dict[str, Any]:
        """
        Retrieve user memory data

        Directly retrieve stored core memories by user ID via KV method

        Args:
            fastapi_request: FastAPI request object

        Returns:
            Dict[str, Any]: Memory retrieval response

        Raises:
            HTTPException: When request processing fails
        """
        try:
            # Prioritize getting parameters from body, fallback to query params
            try:
                params = await fastapi_request.json()
            except Exception:
                # If no body, get from query params (backward compatibility)
                params = dict(fastapi_request.query_params)
            logger.info(
                "Received fetch request: user_id=%s, memory_type=%s",
                params.get("user_id"),
                params.get("memory_type"),
            )

            # Directly use converter to transform
            fetch_request = convert_dict_to_fetch_mem_request(params)

            # Call memory_manager's fetch_mem method
            response = await self.memory_manager.fetch_mem(fetch_request)

            # Return unified response format
            memory_count = len(response.memories) if response.memories else 0
            logger.info(
                "Fetch request processing completed: user_id=%s, returned %s memories",
                params.get("user_id"),
                memory_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"Memory retrieval successful, retrieved {memory_count} memories",
                "result": response,
            }

        except ValueError as e:
            logger.error("Fetch request parameter error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # Re-raise HTTPException
            raise
        except Exception as e:
            logger.error("Fetch request processing failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve memory, please try again later",
            ) from e

    @get(
        "/search",
        response_model=Dict[str, Any],
        summary="Search relevant memories (supports keyword/vector/hybrid search)",
        description="""
        Retrieve relevant memory data based on query text using keyword, vector, or hybrid methods
        
        ## Functionality:
        - Find most relevant memories based on query text
        - Support three methods: keyword (BM25), vector similarity, and hybrid search
        - Support time range filtering
        - Return results organized by group with relevance scores
        - Suitable for scenarios requiring exact matching or semantic retrieval
        
        ## Search method descriptions:
        - **keyword**: Keyword-based BM25 search, suitable for exact matching, fast (default method)
        - **vector**: Semantic vector-based similarity search, suitable for fuzzy queries and semantic similarity
        - **hybrid**: Hybrid search strategy combining advantages of keyword and vector search (recommended)
        
        ## Result description:
        - Memories returned organized by group
        - Each group contains multiple relevant memories sorted by time
        - Groups sorted by importance score, most important group first
        - Each memory has a relevance score indicating match degree with query
        
        ## Use cases:
        - Conversation context understanding
        - Intelligent Q&A systems
        - Relevant content recommendations
        - Memory clue tracing
        """,
        responses={
            200: {
                "description": "Successfully retrieved memory data",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "message": "Memory retrieval successful",
                            "result": {
                                "groups": [
                                    {
                                        "group_id": "group_456",
                                        "memories": [
                                            {
                                                "memory_type": "episode_memory",
                                                "user_id": "user_123",
                                                "timestamp": "2024-01-15T10:30:00",
                                                "summary": "Discussed coffee preference",
                                                "group_id": "group_456",
                                            }
                                        ],
                                        "scores": [0.95],
                                        "original_data": [],
                                    }
                                ],
                                "importance_scores": [0.85],
                                "total_count": 45,
                                "has_more": False,
                                "query_metadata": {
                                    "source": "episodic_memory_es_repository",
                                    "user_id": "user_123",
                                    "memory_type": "retrieve",
                                },
                                "metadata": {
                                    "source": "episodic_memory_es_repository",
                                    "user_id": "user_123",
                                    "memory_type": "retrieve",
                                },
                            },
                        }
                    }
                },
            },
            400: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "query cannot be empty",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/search",
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "Failed to retrieve memory, please try again later",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/search",
                        }
                    }
                },
            },
        },
    )
    async def search_memories(self, fastapi_request: FastAPIRequest) -> Dict[str, Any]:
        """
        Search relevant memory data

        Retrieve relevant memory data based on query text using keyword, vector, or hybrid methods

        Args:
            fastapi_request: FastAPI request object

        Returns:
            Dict[str, Any]: Memory search response

        Raises:
            HTTPException: When request processing fails
        """
        try:
            # Get params from query params first
            query_params = dict(fastapi_request.query_params)

            # Also try to get params from body (for GET + body requests like Elasticsearch)
            if body := await fastapi_request.body():
                with suppress(json.JSONDecodeError, TypeError):
                    if isinstance(body_data := json.loads(body), dict):
                        query_params.update(body_data)

            query = query_params.get("query")
            logger.info(
                "Received search request: user_id=%s, query=%s, retrieve_method=%s",
                query_params.get("user_id"),
                query,
                query_params.get("retrieve_method"),
            )

            # Directly use converter to transform
            retrieve_request = convert_dict_to_retrieve_mem_request(
                query_params, query=query
            )
            logger.info(
                f"After conversion: retrieve_method={retrieve_request.retrieve_method}"
            )

            # Use retrieve_mem method (supports keyword, vector, hybrid)
            response = await self.memory_manager.retrieve_mem(retrieve_request)

            # Return unified response format
            group_count = len(response.memories) if response.memories else 0
            logger.info(
                "Search request complete: user_id=%s, returned %s groups",
                query_params.get("user_id"),
                group_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"Memory search successful, retrieved {group_count} groups",
                "result": response,
            }

        except ValueError as e:
            logger.error("Search request parameter error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # Re-raise HTTPException
            raise
        except Exception as e:
            logger.error("Search request processing failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve memory, please try again later",
            ) from e

    @post(
        "/conversation-meta",
        response_model=Dict[str, Any],
        summary="Save conversation metadata",
        description="""
        Save conversation metadata information, including scene, participants, tags, etc.
        
        ## Functionality:
        - If group_id exists, update the entire record (upsert)
        - If group_id does not exist, create a new record
        - All fields must be provided with complete data
        
        ## Notes:
        - This is a full update interface that will replace the entire record
        - If you only need to update partial fields, use the PATCH /conversation-meta interface
        """,
    )
    async def save_conversation_meta(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        Save conversation metadata

        Receive ConversationMetaRequest format data, convert to ConversationMeta ODM model and save to MongoDB

        Args:
            fastapi_request: FastAPI request object

        Returns:
            Dict[str, Any]: Save response, containing saved metadata information

        Raises:
            HTTPException: When request processing fails
        """
        try:
            # 1. Get JSON body from request
            request_data = await fastapi_request.json()
            logger.info(
                "Received conversation-meta save request: group_id=%s",
                request_data.get("group_id"),
            )

            # 2. Parse into ConversationMetaRequest
            # Handle conversion of user_details
            user_details_data = request_data.get("user_details", {})
            user_details = {}
            for user_id, detail_data in user_details_data.items():
                user_details[user_id] = UserDetail(
                    full_name=detail_data["full_name"],
                    role=detail_data["role"],
                    extra=detail_data.get("extra", {}),
                )

            conversation_meta_request = ConversationMetaRequest(
                version=request_data["version"],
                scene=request_data["scene"],
                scene_desc=request_data["scene_desc"],
                name=request_data["name"],
                description=request_data["description"],
                group_id=request_data["group_id"],
                created_at=request_data["created_at"],
                default_timezone=request_data["default_timezone"],
                user_details=user_details,
                tags=request_data.get("tags", []),
            )

            logger.info(
                "Parsed ConversationMetaRequest successfully: group_id=%s",
                conversation_meta_request.group_id,
            )

            # 3. Convert to ConversationMeta ODM model
            user_details_model = {}
            for user_id, detail in conversation_meta_request.user_details.items():
                user_details_model[user_id] = UserDetailModel(
                    full_name=detail.full_name, role=detail.role, extra=detail.extra
                )

            conversation_meta = ConversationMeta(
                version=conversation_meta_request.version,
                scene=conversation_meta_request.scene,
                scene_desc=conversation_meta_request.scene_desc,
                name=conversation_meta_request.name,
                description=conversation_meta_request.description,
                group_id=conversation_meta_request.group_id,
                conversation_created_at=conversation_meta_request.created_at,
                default_timezone=conversation_meta_request.default_timezone,
                user_details=user_details_model,
                tags=conversation_meta_request.tags,
            )

            # 4. Save using upsert method (update if group_id exists)
            logger.info("Starting to save conversation metadata to MongoDB")
            saved_meta = await self.conversation_meta_repository.upsert_by_group_id(
                group_id=conversation_meta.group_id,
                conversation_data={
                    "version": conversation_meta.version,
                    "scene": conversation_meta.scene,
                    "scene_desc": conversation_meta.scene_desc,
                    "name": conversation_meta.name,
                    "description": conversation_meta.description,
                    "conversation_created_at": conversation_meta.conversation_created_at,
                    "default_timezone": conversation_meta.default_timezone,
                    "user_details": conversation_meta.user_details,
                    "tags": conversation_meta.tags,
                },
            )

            if not saved_meta:
                raise HTTPException(
                    status_code=500, detail="Failed to save conversation metadata"
                )

            logger.info(
                "Conversation metadata saved successfully: id=%s, group_id=%s",
                saved_meta.id,
                saved_meta.group_id,
            )

            # 5. Return success response
            return {
                "status": ErrorStatus.OK.value,
                "message": "Conversation metadata saved successfully",
                "result": {
                    "id": str(saved_meta.id),
                    "group_id": saved_meta.group_id,
                    "scene": saved_meta.scene,
                    "name": saved_meta.name,
                    "version": saved_meta.version,
                    "created_at": (
                        saved_meta.created_at.isoformat()
                        if saved_meta.created_at
                        else None
                    ),
                    "updated_at": (
                        saved_meta.updated_at.isoformat()
                        if saved_meta.updated_at
                        else None
                    ),
                },
            }

        except KeyError as e:
            logger.error("conversation-meta request missing required field: %s", e)
            raise HTTPException(
                status_code=400, detail=f"Missing required field: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error("conversation-meta request parameter error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # Re-raise HTTPException
            raise
        except Exception as e:
            logger.error(
                "conversation-meta request processing failed: %s", e, exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to save conversation metadata, please try again later",
            ) from e

    @patch(
        "/conversation-meta",
        response_model=Dict[str, Any],
        summary="Partially update conversation metadata",
        description="""
        Partially update conversation metadata, only updating provided fields
        
        ## Functionality:
        - Locate the conversation metadata to update by group_id
        - Only update fields provided in the request, keep unchanged fields as-is
        - Suitable for scenarios requiring modification of partial information
        
        ## Request example:
        ```json
        {
          "group_id": "group_123",
          "name": "New conversation name",
          "tags": ["tag1", "tag2"]
        }
        ```
        
        ## Field descriptions:
        - **group_id** (required): Group ID of the conversation to update
        - All other fields are optional, only provide fields that need updating
        
        ## Fields that can be updated:
        - **name**: Conversation name
        - **description**: Conversation description
        - **scene_desc**: Scene description
        - **tags**: Tag list
        - **user_details**: User details (will completely replace existing user_details)
        - **default_timezone**: Default timezone
        
        ## Notes:
        - group_id must exist, otherwise return 404 error
        - If user_details field is provided, it will completely replace existing user details
        - Not allowed to modify core fields such as version, scene, group_id, conversation_created_at
        """,
        responses={
            200: {
                "description": "Successfully updated conversation metadata",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "message": "Conversation metadata updated successfully",
                            "result": {
                                "id": "507f1f77bcf86cd799439011",
                                "group_id": "group_123",
                                "name": "New conversation name",
                                "updated_fields": ["name", "tags"],
                            },
                        }
                    }
                },
            },
            400: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Missing required field group_id",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/conversation-meta",
                        }
                    }
                },
            },
            404: {
                "description": "Conversation metadata not found",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.RESOURCE_NOT_FOUND.value,
                            "message": "Specified conversation metadata not found: group_123",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/conversation-meta",
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "Failed to update conversation metadata, please try again later",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/conversation-meta",
                        }
                    }
                },
            },
        },
    )
    async def patch_conversation_meta(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        Partially update conversation metadata

        Locate record by group_id, only update fields provided in the request

        Args:
            fastapi_request: FastAPI request object

        Returns:
            Dict[str, Any]: Update response, containing updated metadata information

        Raises:
            HTTPException: When request processing fails
        """
        try:
            # 1. Get JSON body from request
            request_data = await fastapi_request.json()
            group_id = request_data.get("group_id")

            # 2. Validate group_id is provided
            if not group_id:
                raise HTTPException(
                    status_code=400, detail="Missing required field group_id"
                )

            logger.info(
                "Received conversation-meta partial update request: group_id=%s",
                group_id,
            )

            # 3. Check if conversation metadata exists
            existing_meta = await self.conversation_meta_repository.get_by_group_id(
                group_id
            )
            if not existing_meta:
                raise HTTPException(
                    status_code=404,
                    detail=f"Specified conversation metadata not found: {group_id}",
                )

            # 4. Prepare update data (exclude immutable fields and null values)
            # Core fields not allowed to be modified via PATCH
            immutable_fields = {
                "version",
                "scene",
                "group_id",
                "conversation_created_at",
            }

            update_data = {}
            updated_fields = []

            # Handle regular fields
            for key, value in request_data.items():
                if key in immutable_fields or key == "group_id":
                    continue

                # Handle user_details field
                if key == "user_details" and value is not None:
                    user_details_model = {}
                    for user_id, detail_data in value.items():
                        user_details_model[user_id] = UserDetailModel(
                            full_name=detail_data["full_name"],
                            role=detail_data["role"],
                            extra=detail_data.get("extra", {}),
                        )
                    update_data["user_details"] = user_details_model
                    updated_fields.append(key)
                elif value is not None:
                    update_data[key] = value
                    updated_fields.append(key)

            # 5. If no fields need updating
            if not update_data:
                logger.warning("No fields provided for update: group_id=%s", group_id)
                return {
                    "status": ErrorStatus.OK.value,
                    "message": "No fields need updating",
                    "result": {
                        "id": str(existing_meta.id),
                        "group_id": existing_meta.group_id,
                        "updated_fields": [],
                    },
                }

            # 6. Perform update
            logger.info(
                "Starting to update conversation metadata, updating fields: %s",
                updated_fields,
            )
            updated_meta = await self.conversation_meta_repository.update_by_group_id(
                group_id=group_id, update_data=update_data
            )

            if not updated_meta:
                raise HTTPException(
                    status_code=500, detail="Failed to update conversation metadata"
                )

            logger.info(
                "Conversation metadata updated successfully: id=%s, group_id=%s, updated_fields=%s",
                updated_meta.id,
                updated_meta.group_id,
                updated_fields,
            )

            # 7. Return success response
            return {
                "status": ErrorStatus.OK.value,
                "message": f"Conversation metadata updated successfully, updated {len(updated_fields)} fields",
                "result": {
                    "id": str(updated_meta.id),
                    "group_id": updated_meta.group_id,
                    "scene": updated_meta.scene,
                    "name": updated_meta.name,
                    "updated_fields": updated_fields,
                    "updated_at": (
                        updated_meta.updated_at.isoformat()
                        if updated_meta.updated_at
                        else None
                    ),
                },
            }

        except HTTPException:
            # Re-raise HTTPException
            raise
        except KeyError as e:
            logger.error(
                "conversation-meta partial update request missing required field: %s", e
            )
            raise HTTPException(
                status_code=400, detail=f"Missing required field: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error(
                "conversation-meta partial update request parameter error: %s", e
            )
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(
                "conversation-meta partial update request processing failed: %s",
                e,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to update conversation metadata, please try again later",
            ) from e
