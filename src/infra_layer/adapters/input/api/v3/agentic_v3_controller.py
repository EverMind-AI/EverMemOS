"""
Agentic Layer V3 Controller

Provides RESTful API routes dedicated to handling group chat memory.
Directly receives simple, straightforward message formats, processes them one by one, and stores them.
"""

import logging
from typing import Any, Dict
from fastapi import HTTPException, Request as FastAPIRequest

from core.di.decorators import controller
from core.di import get_bean_by_type
from core.interface.controller.base_controller import BaseController, post
from core.constants.errors import ErrorCode, ErrorStatus
from agentic_layer.memory_manager import MemoryManager
from api_specs.request_converter import handle_conversation_format
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

logger = logging.getLogger(__name__)


@controller("agentic_v3_controller", primary=True)
class AgenticV3Controller(BaseController):
    """
    Agentic Layer V3 API Controller

    Provides dedicated interfaces for group chat memory:
    - memorize: Store a single message as memory
    - retrieve_lightweight: Lightweight retrieval (Embedding + BM25 + RRF)
    """

    def __init__(self, conversation_meta_repository: ConversationMetaRawRepository):
        """Initialize the controller"""
        super().__init__(
            prefix="/api/v3/agentic",
            tags=["Agentic Layer V3"],
            default_auth="none",  # Adjust authentication strategy based on actual needs
        )
        self.memory_manager = MemoryManager()
        self.conversation_meta_repository = conversation_meta_repository
        # Get RedisProvider
        self.redis_provider = get_bean_by_type(RedisProvider)
        logger.info(
            "AgenticV3Controller initialized with MemoryManager and ConversationMetaRepository"
        )

    @post(
        "/memorize",
        response_model=Dict[str, Any],
        summary="Store a single group chat message as memory",
        description="""
        Receive a simple, direct single message format and store it as memory.

        ## Functionality:
        - Accepts simple, direct single message data (no pre-conversion required)
        - Extracts a single message into memory units (memcells)
        - Suitable for real-time message processing scenarios
        - Returns a list of saved memories

        ## Input Format (simple and direct):
        ```json
        {
          "group_id": "group_123",
          "group_name": "Project Discussion Group",
          "message_id": "msg_001",
          "create_time": "2025-01-15T10:00:00+00:00",
          "sender": "user_001",
          "sender_name": "Zhang San",
          "content": "Discuss the technical solution for the new feature today",
          "refer_list": ["msg_000"]
        }
        ```

        ## Field Descriptions:
        - **group_id** (Optional): Group ID
        - **group_name** (Optional): Group name
        - **message_id** (Required): Message ID
        - **create_time** (Required): Message creation time (ISO 8601 format)
        - **sender** (Required): Sender user ID
        - **sender_name** (Optional): Sender name
        - **content** (Required): Message content
        - **refer_list** (Optional): List of referenced message IDs

        ## Differences from Other Interfaces:
        - **V3 /memorize**: Simple, direct single message format (this interface, recommended)
        - **V2 /memorize**: Accepts internal format, requires external conversion

        ## Use Cases:
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
                        "example": {
                            "status": "ok",
                            "message": "Memory stored successfully, 1 memory saved",
                            "result": {
                                "saved_memories": [
                                    {
                                        "memory_type": "episode_summary",
                                        "user_id": "user_001",
                                        "group_id": "group_123",
                                        "timestamp": "2025-01-15T10:00:00",
                                        "content": "User discussed the technical solution for the new feature",
                                    }
                                ],
                                "count": 1,
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
                            "path": "/api/v3/agentic/memorize",
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
                            "path": "/api/v3/agentic/memorize",
                        }
                    }
                },
            },
        },
    )
    async def memorize_single_message(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        Store a single message as memory data.

        Receives a simple, direct single message format, converts it via group_chat_converter, and stores it.

        Args:
            fastapi_request: FastAPI request object

        Returns:
            Dict[str, Any]: Memory storage response containing a list of saved memories

        Raises:
            HTTPException: When request processing fails
        """
        try:
            # 1. Get JSON body from request (simple, direct format)
            message_data = await fastapi_request.json()
            logger.info("Received V3 memorize request (single message)")

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
            request_id = await self.memory_manager.memorize(memorize_request)

            # 5. Return unified response format
            if request_id:
                # Boundary detected, submitted to Worker queue for asynchronous processing
                logger.info("Memory request submitted: request_id=%s", request_id)
                return {
                    "status": ErrorStatus.OK.value,
                    "message": "Memory extraction submitted",
                    "result": {"request_id": request_id, "status_info": "processing"},
                }
            else:
                # No boundary detected, message accumulated
                logger.info("Message accumulated, awaiting boundary detection")
            return {
                "status": ErrorStatus.OK.value,
                "message": "Message queued, awaiting boundary detection",
                "result": {"request_id": None, "status_info": "accumulated"},
            }

        except ValueError as e:
            logger.error("V3 memorize request parameter error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # Re-raise HTTPException
            raise
        except Exception as e:
            logger.error("V3 memorize request processing failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to store memory, please try again later"
            ) from e

    @post(
        "/retrieve_lightweight",
        response_model=Dict[str, Any],
        summary="Lightweight memory retrieval (Embedding + BM25 + RRF)",
        description="""
        Lightweight memory retrieval interface using Embedding + BM25 + RRF fusion strategy.

        ## Functionality:
        - Execute vector retrieval and keyword retrieval in parallel
        - Use RRF (Reciprocal Rank Fusion) to merge results
        - Fast, suitable for real-time scenarios

        ## Input Format:
        ```json
        {
          "query": "Beijing travel food",
          "user_id": "default",
          "group_id": "assistant",
          "time_range_days": 365,
          "top_k": 20,
          "retrieval_mode": "rrf",
          "data_source": "episode",
        }
        ```

        ## Field Descriptions:
        - **query** (Required): User query
        - **user_id** (Optional): User ID (for filtering)
        - **group_id** (Optional): Group ID (for filtering)
        - **time_range_days** (Optional): Time range in days (default 365 days)
        - **top_k** (Optional): Number of results to return (default 20)
        - **retrieval_mode** (Optional): Retrieval mode
          * "rrf": RRF fusion (default)
          * "embedding": Pure vector retrieval
          * "bm25": Pure keyword retrieval
        - **data_source** (Optional): Data source
          * "episode": Retrieve from MemCell.episode (default)
          * "event_log": Retrieve from event_log.atomic_fact
          * "foresight": Retrieve from foresight
          * "profile": Profile retrieval requiring only user_id + group_id (query can be empty)
        - **current_time** (Optional): Current time, YYYY-MM-DD format, used to filter valid foresight within period (only effective when data_source=foresight)
        - **radius** (Optional): COSINE similarity threshold, range [-1, 1], default 0.6
          * Only return results with similarity >= radius
          * Affects result quality of vector retrieval part (embedding/rrf mode)
          * Effective for foresight and episodic memory (foresight/episode), event log uses L2 distance and does not support currently

        ## Return Format:
        ```json
        {
          "status": "ok",
          "message": "Retrieval successful, found 10 memories",
          "result": {
            "memories": [...],
            "count": 10,
            "metadata": {
              "retrieval_mode": "lightweight",
              "emb_count": 15,
              "bm25_count": 12,
              "final_count": 10,
              "total_latency_ms": 123.45
            }
          }
        }
        ```
        """,
    )
    async def retrieve_lightweight(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        Lightweight memory retrieval (Embedding + BM25 + RRF fusion)

        Args:
            fastapi_request: FastAPI request object

        Returns:
            Dict[str, Any]: Retrieval result response
        """
        try:
            # 1. Parse request parameters
            request_data = await fastapi_request.json()
            query = request_data.get("query")
            user_id = request_data.get("user_id")
            group_id = request_data.get("group_id")
            time_range_days = request_data.get("time_range_days", 365)
            top_k = request_data.get("top_k", 20)
            retrieval_mode = request_data.get("retrieval_mode", "rrf")
            data_source = request_data.get("data_source", "episode")
            current_time_str = request_data.get("current_time")  # YYYY-MM-DD format
            radius = request_data.get(
                "radius"
            )  # COSINE similarity threshold (optional)

            if not query and data_source != "profile":
                raise ValueError("Missing required parameter: query")

            # Validate data_source parameter (compatible with old parameter name memcell)
            if data_source == "memcell":
                data_source = "episode"

            # Validate data_source is a valid value
            VALID_DATA_SOURCES = {"episode", "event_log", "foresight", "profile"}
            if data_source not in VALID_DATA_SOURCES:
                raise ValueError(
                    f"Invalid data_source: '{data_source}', "
                    f"valid values: {', '.join(sorted(VALID_DATA_SOURCES))}"
                )

            if data_source == "profile":
                if not user_id or not group_id:
                    raise ValueError(
                        "user_id and group_id must be provided when data_source=profile"
                    )

            # Parse current_time
            from datetime import datetime

            current_time = None
            if current_time_str:
                try:
                    current_time = datetime.strptime(current_time_str, "%Y-%m-%d")
                except ValueError as e:
                    raise ValueError(
                        f"current_time format error, should be YYYY-MM-DD: {e}"
                    ) from e

            logger.info(
                f"Received lightweight retrieval request: query={query}, group_id={group_id}, "
                f"mode={retrieval_mode}, source={data_source}"
                f"current_time={current_time_str}, top_k={top_k}"
            )

            # 2. Call memory_manager's lightweight retrieval
            result = await self.memory_manager.retrieve_lightweight(
                query=query,
                user_id=user_id,
                group_id=group_id,
                time_range_days=time_range_days,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                data_source=data_source,
                current_time=current_time,
                radius=radius,
            )

            # 3. Return unified format
            return {
                "status": ErrorStatus.OK.value,
                "message": f"Retrieval successful, found {result['count']} memories",
                "result": result,
            }

        except ValueError as e:
            logger.error("V3 retrieve_lightweight request parameter error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "V3 retrieve_lightweight request processing failed: %s",
                e,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail="Retrieval failed, please try again later"
            ) from e

    @post(
        "/retrieve_agentic",
        response_model=Dict[str, Any],
        summary="Agentic Memory Retrieval (LLM-guided multi-round retrieval)",
        description="""
        Agentic memory retrieval interface using LLM-guided multi-round intelligent retrieval.

        ## Functionality:
        - Use LLM to determine retrieval sufficiency
        - Automatically perform multi-round retrieval and query refinement
        - Use Rerank to improve result quality
        - Suitable for complex queries requiring deep understanding

        ## Retrieval Process:
        1. Round 1: RRF hybrid retrieval (Embedding + BM25)
        2. Rerank to optimize results
        3. LLM determines if sufficient
        4. If insufficient: generate multiple refined queries
        5. Round 2: Parallel retrieval with multiple queries
        6. Merge and Rerank to return final results

        ## Input Format:
        ```json
        {
          "query": "What does the user like to eat?",
          "user_id": "default",
          "group_id": "assistant",
          "time_range_days": 365,
          "top_k": 20,
          "llm_config": {
            "api_key": "your_api_key",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini"
          }
        }
        ```

        ## Field Descriptions:
        - **query** (Required): User query
        - **user_id** (Optional): User ID (for filtering)
        - **group_id** (Optional): Group ID (for filtering)
        - **time_range_days** (Optional): Time range in days (default 365 days)
        - **top_k** (Optional): Number of results to return (default 20)
        - **llm_config** (Optional): LLM configuration
          * api_key: LLM API Key (optional, default from environment variable)
          * base_url: LLM API address (optional, default OpenRouter)
          * model: LLM model (optional, default gpt-4o-mini)

        ## Return Format:
        ```json
        {
          "status": "ok",
          "message": "Agentic retrieval successful, found 15 memories",
          "result": {
            "memories": [...],
            "count": 15,
            "metadata": {
              "retrieval_mode": "agentic",
              "is_multi_round": true,
              "round1_count": 20,
              "is_sufficient": false,
              "reasoning": "Need more specific information about dietary preferences",
              "refined_queries": ["What cuisines does the user like most?", "What does the user dislike eating?"],
              "round2_count": 40,
              "final_count": 15,
              "total_latency_ms": 2345.67
            }
          }
        }
        ```

        ## Use Cases:
        - Answering complex questions
        - Deep information mining
        - Multi-dimensional memory retrieval
        - Intelligent dialogue systems

        ## Notes:
        - Requires LLM API Key configuration
        - Retrieval takes longer (typically 2-5 seconds)
        - Will incur LLM API call costs
        """,
    )
    async def retrieve_agentic(self, fastapi_request: FastAPIRequest) -> Dict[str, Any]:
        """
        Agentic memory retrieval (LLM-guided multi-round intelligent retrieval)

        Args:
            fastapi_request: FastAPI request object

        Returns:
            Dict[str, Any]: Retrieval result response
        """
        try:
            # 1. Parse request parameters
            request_data = await fastapi_request.json()
            query = request_data.get("query")
            user_id = request_data.get("user_id")
            group_id = request_data.get("group_id")
            time_range_days = request_data.get("time_range_days", 365)
            top_k = request_data.get("top_k", 20)
            llm_config = request_data.get("llm_config", {})

            if not query:
                raise ValueError("Missing required parameter: query")

            logger.info(
                f"Received agentic retrieval request: query={query}, group_id={group_id}, top_k={top_k}"
            )

            # 2. Create LLM Provider
            from memory_layer.llm.llm_provider import LLMProvider
            import os

            # Get configuration from request or environment variables
            api_key = llm_config.get("api_key") or os.getenv("LLM_API_KEY")
            base_url = llm_config.get("base_url") or os.getenv("LLM_BASE_URL")
            model = llm_config.get("model") or os.getenv("LLM_MODEL")

            if not api_key:
                raise ValueError(
                    "LLM API Key is missing. Please provide it in llm_config.api_key or set environment variable OPENROUTER_API_KEY/OPENAI_API_KEY"
                )

            # Create LLM Provider (using OpenAI compatible interface)
            llm_provider = LLMProvider(
                provider_type="openai",
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0.3,
                max_tokens=2048,
            )

            logger.info(f"Using LLM: {model} @ {base_url}")

            # 3. Call memory_manager's agentic retrieval
            result = await self.memory_manager.retrieve_agentic(
                query=query,
                user_id=user_id,
                group_id=group_id,
                time_range_days=time_range_days,
                top_k=top_k,
                llm_provider=llm_provider,
                agentic_config=None,  # Use default configuration
            )

            # 4. Return unified format
            return {
                "status": ErrorStatus.OK.value,
                "message": f"Agentic retrieval successful, found {result['count']} memories",
                "result": result,
            }

        except ValueError as e:
            logger.error("V3 retrieve_agentic request parameter error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "V3 retrieve_agentic request processing failed: %s", e, exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail="Agentic retrieval failed, please try again later",
            ) from e

    @post(
        "/conversation-meta",
        response_model=Dict[str, Any],
        summary="Save conversation metadata",
        description="""
        Save metadata information of conversations, including scene, participants, tags, etc.
        """,
    )
    async def save_conversation_meta(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        Save conversation metadata

        Receives ConversationMetaRequest formatted data, converts it to ConversationMeta ODM model, and saves to MongoDB

        Args:
            fastapi_request: FastAPI request object

        Returns:
            Dict[str, Any]: Save response containing saved metadata information

        Raises:
            HTTPException: When request processing fails
        """
        try:
            # 1. Get JSON body from request
            request_data = await fastapi_request.json()
            logger.info(
                "Received V3 conversation-meta save request: group_id=%s",
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

            # 4. Save using upsert (update if group_id already exists)
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
                "Saved conversation metadata successfully: id=%s, group_id=%s",
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
            logger.error("V3 conversation-meta request missing required field: %s", e)
            raise HTTPException(
                status_code=400, detail=f"Missing required field: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error("V3 conversation-meta request parameter error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # Re-raise HTTPException
            raise
        except Exception as e:
            logger.error(
                "V3 conversation-meta request processing failed: %s", e, exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to save conversation metadata, please try again later",
            ) from e
