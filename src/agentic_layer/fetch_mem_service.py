"""
Memory retrieval service

This module provides a service layer interface for accessing memory data, interfacing with repository classes that access the database.
Provides ID-based query functionality, supporting retrieval of various memory types.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union


from core.di import get_bean_by_type, get_bean, service
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord,
    ForesightRecordProjection,
)
from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
    CoreMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.entity_raw_repository import (
    EntityRawRepository,
)
from infra_layer.adapters.out.persistence.repository.relationship_raw_repository import (
    RelationshipRawRepository,
)
from infra_layer.adapters.out.persistence.repository.behavior_history_raw_repository import (
    BehaviorHistoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
    ConversationMetaRawRepository,
)
from infra_layer.adapters.out.persistence.repository.event_log_record_raw_repository import (
    EventLogRecordRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
    EventLogRecord,
    EventLogRecordProjection,
)
from infra_layer.adapters.out.persistence.repository.foresight_record_repository import (
    ForesightRecordRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecordProjection,
)
from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
    UserProfileRawRepository,
)
from api_specs.dtos.memory_query import FetchMemResponse
from memory_layer.memory_extractor.profile_memory_life.types import ProfileMemoryLife
from api_specs.memory_models import (
    MemoryType,
    MemoryModel,
    BaseMemoryModel,
    ProfileModel,
    PreferenceModel,
    EpisodicMemoryModel,
    ForesightModel,
    EntityModel,
    RelationModel,
    BehaviorHistoryModel,
    CoreMemoryModel,
    EventLogModel,
    ForesightRecordModel,
    Metadata,
)

logger = logging.getLogger(__name__)


class FetchMemoryServiceInterface(ABC):
    """Memory retrieval service interface"""

    @abstractmethod
    async def find_by_user_id(
        self,
        user_id: str,
        memory_type: MemoryType,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        limit: int = 10,
    ) -> FetchMemResponse:
        """
        Find memories by user ID

        Args:
            user_id: User ID
            memory_type: Memory type
            limit: Limit on number of returned items

        Returns:
            Memory query response
        """
        pass

    @abstractmethod
    async def find_by_id(
        self, memory_id: str, memory_type: MemoryType
    ) -> Optional[MemoryModel]:
        """
        Find a single memory by memory ID

        Args:
            memory_id: Memory ID
            memory_type: Memory type

        Returns:
            Memory model, or None if not found
        """
        pass

    @abstractmethod
    async def find_episodic_by_event_id(
        self, event_id: str, user_id: str
    ) -> Optional[EpisodicMemoryModel]:
        """
        Find episodic memory by event ID

        Args:
            event_id: Event ID
            user_id: User ID

        Returns:
            Episodic memory model, or None if not found
        """
        pass

    @abstractmethod
    async def find_entity_by_entity_id(self, entity_id: str) -> Optional[EntityModel]:
        """
        Find entity by entity ID

        Args:
            entity_id: Entity ID

        Returns:
            Entity model, or None if not found
        """
        pass

    @abstractmethod
    async def find_relationship_by_entity_ids(
        self, source_entity_id: str, target_entity_id: str
    ) -> Optional[RelationModel]:
        """
        Find relationship by entity IDs

        Args:
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID

        Returns:
            Relation model, or None if not found
        """
        pass


@service(name="fetch_memory_service", primary=True)
class FetchMemoryServiceImpl(FetchMemoryServiceInterface):
    """Real implementation of memory retrieval service

    Uses repository instances injected by DI framework for database access.
    """

    def __init__(self):
        """Initialize service"""
        self._episodic_repo = None
        self._core_repo = None
        self._entity_repo = None
        self._relationship_repo = None
        self._behavior_repo = None
        self._conversation_meta_repo = None
        self._event_log_repo = None
        self._foresight_record_repo = None
        self._user_profile_repo = None
        logger.info("FetchMemoryServiceImpl initialized")

    def _get_repositories(self):
        """Get repository instances"""
        if self._episodic_repo is None:
            self._episodic_repo = get_bean_by_type(EpisodicMemoryRawRepository)
        if self._core_repo is None:
            self._core_repo = get_bean_by_type(CoreMemoryRawRepository)
        if self._entity_repo is None:
            self._entity_repo = get_bean_by_type(EntityRawRepository)
        if self._relationship_repo is None:
            self._relationship_repo = get_bean_by_type(RelationshipRawRepository)
        if self._behavior_repo is None:
            self._behavior_repo = get_bean_by_type(BehaviorHistoryRawRepository)
        if self._conversation_meta_repo is None:
            self._conversation_meta_repo = get_bean_by_type(
                ConversationMetaRawRepository
            )
        if self._event_log_repo is None:
            self._event_log_repo = get_bean_by_type(EventLogRecordRawRepository)
        if self._foresight_record_repo is None:
            self._foresight_record_repo = get_bean_by_type(ForesightRecordRawRepository)
        if self._user_profile_repo is None:
            self._user_profile_repo = get_bean_by_type(UserProfileRawRepository)

    async def _get_employee_metadata(
        self,
        user_id: str,
        source: str,
        memory_type: str,
        limit: int = None,
        group_id: str = None,
    ) -> Metadata:
        """
        Get user information by user ID and create Metadata

        Retrieve user details from user_details in conversation-meta.
        Tables related to employee_org have been removed; user information now comes from conversation-meta.

        Args:
            user_id: User ID
            source: Data source
            memory_type: Memory type
            limit: Limit number (optional)
            group_id: Group ID (optional), if provided, retrieve user details from conversation-meta

        Returns:
            Metadata object
        """
        try:
            # If group_id is provided, try to get user details from conversation_meta
            if group_id:
                # Ensure repository is initialized
                if self._conversation_meta_repo is None:
                    self._get_repositories()

                # Query conversation metadata
                conversation_meta = await self._conversation_meta_repo.get_by_group_id(
                    group_id
                )

                # If conversation metadata is found and contains user details
                if conversation_meta and conversation_meta.user_details:
                    user_detail = conversation_meta.user_details.get(user_id)
                    if user_detail:
                        # Extract user information from user_details
                        # user_detail is a UserDetailModel object containing full_name, role, extra
                        return Metadata(
                            source=source,
                            user_id=user_id,
                            memory_type=memory_type,
                            limit=limit,
                            full_name=user_detail.full_name,
                            # Extract email and phone if available in extra
                            email=(
                                user_detail.extra.get("email")
                                if user_detail.extra
                                else None
                            ),
                            phone=(
                                user_detail.extra.get("phone")
                                if user_detail.extra
                                else None
                            ),
                        )

            # If no group_id or user information not found, return basic information
            return Metadata(
                source=source, user_id=user_id, memory_type=memory_type, limit=limit
            )

        except Exception as e:
            logger.warning(
                f"Failed to retrieve user information: {e}, creating Metadata with basic info"
            )
            return Metadata(
                source=source, user_id=user_id, memory_type=memory_type, limit=limit
            )

    def _render_readable_profile(self, profile_data: Dict[str, Any]) -> str:
        """Generate readable profile text from V2 profile data (no evidence/sources)."""
        try:
            profile = ProfileMemoryLife.from_dict(profile_data)
            return profile.to_readable_profile()
        except Exception:
            return "No profile data yet."

    async def _convert_core_memory(self, core_memory) -> CoreMemoryModel:
        """Convert core memory document to model"""
        metadata = await self._get_employee_metadata(
            user_id=core_memory.user_id,
            source="core_memory",
            memory_type=MemoryType.MULTIPLE.value,
        )

        return CoreMemoryModel(
            id=str(core_memory.id),
            user_id=core_memory.user_id,
            version=core_memory.version,
            is_latest=core_memory.is_latest,
            # BaseMemory fields
            user_name=core_memory.user_name,
            gender=core_memory.gender,
            position=core_memory.position,
            supervisor_user_id=core_memory.supervisor_user_id,
            team_members=core_memory.team_members,
            okr=core_memory.okr,
            base_location=core_memory.base_location,
            hiredate=core_memory.hiredate,
            age=core_memory.age,
            department=core_memory.department,
            # Profile fields
            hard_skills=core_memory.hard_skills,
            soft_skills=core_memory.soft_skills,
            output_reasoning=core_memory.output_reasoning,
            motivation_system=core_memory.motivation_system,
            fear_system=core_memory.fear_system,
            value_system=core_memory.value_system,
            humor_use=core_memory.humor_use,
            colloquialism=core_memory.colloquialism,
            personality=core_memory.personality,
            way_of_decision_making=core_memory.way_of_decision_making,
            projects_participated=core_memory.projects_participated,
            user_goal=core_memory.user_goal,
            work_responsibility=core_memory.work_responsibility,
            working_habit_preference=core_memory.working_habit_preference,
            interests=core_memory.interests,
            tendency=core_memory.tendency,
            # Common fields
            extend=core_memory.extend,
            created_at=core_memory.created_at,
            updated_at=core_memory.updated_at,
            metadata=metadata,
        )

    async def _convert_foresight(self, foresight) -> ForesightModel:
        """Convert foresight document to model"""
        metadata = await self._get_employee_metadata(
            user_id=foresight.user_id, source="user_input", memory_type="foresight"
        )

        return ForesightModel(
            id=str(foresight.id),
            user_id=foresight.user_id,
            concept=foresight.subject,
            definition=foresight.description,
            category="Foresight",
            related_concepts=[],
            confidence_score=1.0,
            source="User input",
            created_at=foresight.created_at,
            updated_at=foresight.updated_at,
            metadata=metadata,
        )

    def _convert_episodic_memory(self, episodic_memory) -> EpisodicMemoryModel:
        """Convert episodic memory document to model"""
        return EpisodicMemoryModel(
            id=str(episodic_memory.id),
            user_id=episodic_memory.user_id,
            episode_id=str(episodic_memory.event_id),
            title=episodic_memory.subject,
            summary=episodic_memory.summary,
            participants=episodic_memory.participants or [],
            location=(
                episodic_memory.extend.get("location", "")
                if episodic_memory.extend
                else ""
            ),
            key_events=episodic_memory.keywords or [],
            created_at=episodic_memory.created_at,
            updated_at=episodic_memory.updated_at,
            metadata=Metadata(
                source="episodic_memory",
                user_id=episodic_memory.user_id,
                memory_type=MemoryType.EPISODIC_MEMORY.value,
            ),
        )

    def _convert_entity(self, entity) -> EntityModel:
        """Convert entity document to model"""
        return EntityModel(
            id=str(entity.id),
            user_id="",  # Entity may not belong to a specific user
            entity_name=entity.name,
            entity_type=entity.type,
            description=f"Entity name: {entity.name} | Type: {entity.type}",
            attributes={},
            aliases=entity.aliases or [],
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            metadata=Metadata(
                source="entity",
                user_id=entity.user_id,
                memory_type=MemoryType.ENTITY.value,
            ),
        )

    def _convert_relationship(self, relationship) -> RelationModel:
        """Convert relationship document to model"""
        return RelationModel(
            id=str(relationship.id),
            user_id="",  # Relationship may not belong to a specific user
            source_entity_id=relationship.source_entity_id,
            target_entity_id=relationship.target_entity_id,
            relation_type=(
                relationship.relationship[0]["type"]
                if relationship.relationship
                else "Unknown relationship"
            ),
            relation_description=(
                relationship.relationship[0]["content"]
                if relationship.relationship
                else ""
            ),
            strength=0.8,  # Default strength
            created_at=relationship.created_at,
            updated_at=relationship.updated_at,
            metadata=Metadata(
                source="relationship",
                user_id=relationship.user_id,
                memory_type=MemoryType.RELATION.value,
            ),
        )

    def _convert_behavior_history(self, behavior) -> BehaviorHistoryModel:
        """Convert behavior history document to model"""
        return BehaviorHistoryModel(
            id=str(behavior.id),
            user_id=behavior.user_id,
            action_type=(
                behavior.behavior_type[0]
                if behavior.behavior_type
                else "Unknown behavior"
            ),
            action_description=f"Behavior type: {behavior.behavior_type}",
            context=behavior.meta or {},
            result="Success",
            session_id=behavior.event_id,
            created_at=behavior.created_at,
            updated_at=behavior.updated_at,
            metadata=Metadata(
                source="behavior_history",
                user_id=behavior.user_id,
                memory_type=MemoryType.BEHAVIOR_HISTORY.value,
            ),
        )

    def _convert_event_log(
        self, event_log: Union[EventLogRecord, EventLogRecordProjection]
    ) -> EventLogModel:
        """Convert event log document to model

        Supports both EventLogRecord and EventLogRecordShort types.
        EventLogRecordShort does not contain the vector field.
        """
        return EventLogModel(
            id=str(event_log.id),
            user_id=event_log.user_id,
            atomic_fact=event_log.atomic_fact,
            parent_episode_id=event_log.parent_episode_id,
            timestamp=event_log.timestamp,
            user_name=event_log.user_name,
            group_id=event_log.group_id,
            group_name=event_log.group_name,
            participants=event_log.participants,
            vector=getattr(
                event_log, 'vector', None
            ),  # EventLogRecordShort does not have vector field
            vector_model=event_log.vector_model,
            event_type=event_log.event_type,
            extend=event_log.extend,
            created_at=event_log.created_at,
            updated_at=event_log.updated_at,
            metadata=Metadata(
                source="event_log",
                user_id=event_log.user_id,
                memory_type=MemoryType.EVENT_LOG.value,
            ),
        )

    def _convert_foresight_record(
        self, foresight_record: Union[ForesightRecord, ForesightRecordProjection]
    ) -> ForesightRecordModel:
        """Convert foresight record document to model

        Supports both ForesightRecord and ForesightRecordProjection types.
        ForesightRecordProjection does not contain the vector field.
        """
        return ForesightRecordModel(
            id=str(foresight_record.id),
            content=foresight_record.content,
            parent_episode_id=foresight_record.parent_episode_id,
            user_id=foresight_record.user_id,
            user_name=foresight_record.user_name,
            group_id=foresight_record.group_id,
            group_name=foresight_record.group_name,
            start_time=foresight_record.start_time,
            end_time=foresight_record.end_time,
            duration_days=foresight_record.duration_days,
            participants=foresight_record.participants,
            vector=getattr(
                foresight_record, 'vector', None
            ),  # ForesightRecordProjection does not have vector field
            vector_model=foresight_record.vector_model,
            evidence=foresight_record.evidence,
            extend=foresight_record.extend,
            created_at=foresight_record.created_at,
            updated_at=foresight_record.updated_at,
            metadata=Metadata(
                source="foresight_record",
                user_id=foresight_record.user_id or "",
                memory_type=MemoryType.FORESIGHT.value,
            ),
        )

    async def find_by_user_id(
        self,
        user_id: str,
        memory_type: MemoryType,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        limit: int = 10,
    ) -> FetchMemResponse:
        """
        Find memories by user ID

        Args:
            user_id: User ID
            memory_type: Memory type
            version_range: Version range (start, end), closed interval [start, end].
                          If not provided or None, get the latest version (ordered by version descending)
            limit: Limit on number of returned items

        Returns:
            Memory query response
        """
        logger.debug(
            f"Fetching {memory_type} memories for user {user_id}, limit: {limit}"
        )

        try:
            self._get_repositories()
            memories = []

            match memory_type:
                case MemoryType.MULTIPLE:
                    # Multi-type query: get core_memory and convert to CoreMemoryModel
                    core_memory_result = await self._core_repo.get_by_user_id(
                        user_id, version_range=version_range
                    )
                    if core_memory_result:
                        # If version_range is None, core_memory_result is a single CoreMemory
                        # If version_range is not None, core_memory_result is List[CoreMemory]
                        if isinstance(core_memory_result, list):
                            memories = [
                                await self._convert_core_memory(core_memory)
                                for core_memory in core_memory_result
                            ]
                        else:
                            memories = [
                                await self._convert_core_memory(core_memory_result)
                            ]
                    else:
                        memories = []
                case MemoryType.FORESIGHT:
                    # Foresight: each user has only one foresight document
                    foresight_records = await self._foresight_record_repo.get_by_user_id(user_id, limit=limit)

                    memories = [
                        self._convert_foresight_record(record)
                        for record in foresight_records
                    ]

                case MemoryType.EPISODIC_MEMORY:
                    # Episodic memory: list of events sorted by time
                    episodic_memories = await self._episodic_repo.get_by_user_id(
                        user_id, limit=limit
                    )
                    memories = [
                        self._convert_episodic_memory(mem) for mem in episodic_memories
                    ]

                case MemoryType.BASE_MEMORY:
                    # Base memory: extract basic information from core memory
                    core_memory = await self._core_repo.get_by_user_id(user_id)
                    if core_memory:
                        base_info = self._core_repo.get_base(core_memory)
                        memories = [
                            BaseMemoryModel(
                                id=str(core_memory.id),
                                user_id=core_memory.user_id,
                                content=f"User: {base_info.get('user_name', 'Unknown')} | Position: {base_info.get('position', 'Unknown')} | Department: {base_info.get('department', 'Unknown')}",
                                created_at=core_memory.created_at,
                                updated_at=core_memory.updated_at,
                                metadata={
                                    "user_name": base_info.get('user_name', ''),
                                    "position": base_info.get('position', ''),
                                    "department": base_info.get('department', ''),
                                    "company": base_info.get('company', ''),
                                    "location": base_info.get('location', ''),
                                    "contact": base_info.get('contact', {}),
                                },
                            )
                        ]
                    else:
                        memories = []

                case MemoryType.PROFILE:
                    # Profile: get user profiles from user_profile_repo
                    user_profiles = await self._user_profile_repo.get_all_by_user(user_id, limit)
                    
                    memories = []
                    for up in user_profiles:
                        # Avoid mutating persisted dict instances (reduce hidden side effects)
                        profile_data = dict(up.profile_data or {})
                        # Generate readable_profile if not present (for life profiles)
                        if "readable_profile" not in profile_data and "explicit_info" in profile_data:
                            profile_data["readable_profile"] = self._render_readable_profile(profile_data)
                        memories.append({
                            "id": str(up.id),
                            "user_id": up.user_id,
                            "group_id": up.group_id,
                            "profile_data": profile_data,
                            "scenario": up.scenario,
                            "confidence": up.confidence,
                            "version": up.version,
                            "cluster_ids": up.cluster_ids,
                            "memcell_count": up.memcell_count,
                            "last_updated_cluster": up.last_updated_cluster,
                            "created_at": up.created_at.isoformat() if up.created_at else None,
                            "updated_at": up.updated_at.isoformat() if up.updated_at else None,
                        })

                case MemoryType.PREFERENCE:
                    # Preferences: extract preference settings from core memory
                    core_memory = await self._core_repo.get_by_user_id(user_id)
                    if core_memory:
                        preference_info = self._core_repo.get_preference(core_memory)
                        # Convert preference information into multiple PreferenceModel instances
                        memories = []
                        for key, value in preference_info.items():
                            memories.append(
                                PreferenceModel(
                                    id=f"{core_memory.id}_{key}",
                                    user_id=core_memory.user_id,
                                    category="Personal preference",
                                    preference_key=key,
                                    preference_value=str(value),
                                    confidence_score=1.0,
                                    created_at=core_memory.created_at,
                                    updated_at=core_memory.updated_at,
                                    metadata={
                                        "source": "core_memory",
                                        "original_key": key,
                                    },
                                )
                            )
                    else:
                        memories = []

                case MemoryType.ENTITY:
                    # Entity: query entities related to the user
                    entities = await self._entity_repo.get_by_type(
                        "Person", limit=limit
                    )
                    memories = [self._convert_entity(entity) for entity in entities]

                case MemoryType.RELATION:
                    # Relationship: query interpersonal relationships
                    relationships = (
                        await self._relationship_repo.get_by_relationship_type(
                            "Interpersonal relationship", limit=limit
                        )
                    )
                    memories = [
                        self._convert_relationship(rel) for rel in relationships
                    ]

                case MemoryType.BEHAVIOR_HISTORY:
                    # Behavior history: user behaviors sorted by time
                    behaviors = await self._behavior_repo.get_by_user_id(
                        user_id, limit=limit
                    )
                    memories = [
                        self._convert_behavior_history(behavior)
                        for behavior in behaviors
                    ]

                case MemoryType.EVENT_LOG:
                    # Event log: list of atomic facts
                    event_logs = await self._event_log_repo.get_by_user_id(
                        user_id, limit=limit, model=EventLogRecordProjection
                    )
                    memories = [
                        self._convert_event_log(event_log) for event_log in event_logs
                    ]

            # Create metadata containing employee information
            response_metadata = await self._get_employee_metadata(
                user_id=user_id,
                source="real_service",
                memory_type=memory_type.value,
                limit=limit,
            )
            print(f"---- response_metadata: {response_metadata}")

            return FetchMemResponse(
                memories=memories,
                total_count=len(memories),
                has_more=len(memories) == limit,
                metadata=response_metadata,
            )

        except Exception as e:
            logger.error(f"Error fetching memories for user {user_id}: {e}")
            # Even if error occurs, try to get employee information
            try:
                error_metadata = await self._get_employee_metadata(
                    user_id=user_id,
                    source="real_service",
                    memory_type=memory_type.value,
                    limit=limit,
                )
            except:
                error_metadata = Metadata(
                    source="real_service",
                    user_id=user_id,
                    memory_type=memory_type.value,
                    limit=limit,
                )

            return FetchMemResponse(
                memories=[], total_count=0, has_more=False, metadata=error_metadata
            )

    async def find_by_id(
        self, memory_id: str, memory_type: MemoryType
    ) -> Optional[MemoryModel]:
        """
        Find a single memory by memory ID

        Args:
            memory_id: Memory ID
            memory_type: Memory type

        Returns:
            Memory model, or None if not found
        """
        logger.debug(f"Fetching {memory_type} memory by ID: {memory_id}")

        try:
            self._get_repositories()

            match memory_type:
                case MemoryType.FORESIGHT:
                    # Foresight queried by user ID
                    foresight = await self._foresight_record_repo.get_by_user_id(
                        memory_id
                    )
                    if foresight:
                        return self._convert_foresight(foresight)

                case MemoryType.EPISODIC_MEMORY:
                    # Episodic memory queried by event ID, requires user ID
                    # Here we assume memory_id is event_id, requiring an additional user_id parameter
                    logger.warning(
                        "Episodic memory query by ID requires user_id, use find_episodic_by_event_id instead"
                    )
                    return None

                case MemoryType.ENTITY:
                    entity = await self._entity_repo.get_by_entity_id(memory_id)
                    if entity:
                        return self._convert_entity(entity)

                case MemoryType.RELATION:
                    # Relationship query requires source and target entity IDs
                    logger.warning(
                        "Relation query by ID requires source and target entity IDs, use find_relationship_by_entity_ids instead"
                    )
                    return None

                case MemoryType.BEHAVIOR_HISTORY:
                    # Behavior history queried by user ID
                    behaviors = await self._behavior_repo.get_by_user_id(
                        memory_id, limit=1
                    )
                    if behaviors:
                        return self._convert_behavior_history(behaviors[0])

                case MemoryType.EVENT_LOG:
                    # Event log queried by ID (using simplified version to reduce data transfer)
                    event_log = await self._event_log_repo.get_by_id(
                        memory_id, model=EventLogRecordProjection
                    )
                    if event_log:
                        return self._convert_event_log(event_log)

            return None

        except Exception as e:
            logger.error(f"Error fetching memory by ID {memory_id}: {e}")
            return None

    async def find_episodic_by_event_id(
        self, event_id: str, user_id: str
    ) -> Optional[EpisodicMemoryModel]:
        """
        Find episodic memory by event ID

        Args:
            event_id: Event ID
            user_id: User ID

        Returns:
            Episodic memory model, or None if not found
        """
        logger.debug(
            f"Fetching episodic memory by event_id: {event_id}, user_id: {user_id}"
        )

        try:
            self._get_repositories()
            episodic_memory = await self._episodic_repo.get_by_event_id(
                event_id, user_id
            )
            if episodic_memory:
                return self._convert_episodic_memory(episodic_memory)
            return None

        except Exception as e:
            logger.error(f"Error fetching episodic memory by event_id {event_id}: {e}")
            return None

    async def find_entity_by_entity_id(self, entity_id: str) -> Optional[EntityModel]:
        """
        Find entity by entity ID

        Args:
            entity_id: Entity ID

        Returns:
            Entity model, or None if not found
        """
        logger.debug(f"Fetching entity by entity_id: {entity_id}")

        try:
            self._get_repositories()
            entity = await self._entity_repo.get_by_entity_id(entity_id)
            if entity:
                return self._convert_entity(entity)
            return None

        except Exception as e:
            logger.error(f"Error fetching entity by entity_id {entity_id}: {e}")
            return None

    async def find_relationship_by_entity_ids(
        self, source_entity_id: str, target_entity_id: str
    ) -> Optional[RelationModel]:
        """
        Find relationship by entity IDs

        Args:
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID

        Returns:
            Relation model, or None if not found
        """
        logger.debug(
            f"Fetching relationship by entity_ids: {source_entity_id} -> {target_entity_id}"
        )

        try:
            self._get_repositories()
            relationship = await self._relationship_repo.get_by_entity_ids(
                source_entity_id, target_entity_id
            )
            if relationship:
                return self._convert_relationship(relationship)
            return None

        except Exception as e:
            logger.error(
                f"Error fetching relationship by entity_ids {source_entity_id} -> {target_entity_id}: {e}"
            )
            return None


def get_fetch_memory_service() -> FetchMemoryServiceInterface:
    """Get memory retrieval service instance

    Retrieve service instance via dependency injection framework, supporting singleton pattern.
    """
    return get_bean("fetch_memory_service")
