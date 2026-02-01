"""Profile memory extractor implementation."""

from __future__ import annotations

import asyncio
import ast
import json
import re
from datetime import datetime
from typing import AbstractSet, Any, Dict, List, Optional, Set

from core.observation.logger import get_logger

from memory_layer.llm.llm_provider import LLMProvider

from memory_layer.prompts import get_prompt_by
from api_specs.memory_types import MemoryType, MemCell
from memory_layer.memory_extractor.profile_memory.conversation import (
    annotate_relative_dates,
    build_conversation_text,
    build_episode_text,
    build_profile_prompt,
    extract_group_important_info,
    extract_user_mapping_from_memcells,
    is_important_to_user,
    merge_group_importance_evidence,
)
from memory_layer.memory_extractor.profile_memory.empty_evidence_completion import complete_missing_evidences
from memory_layer.memory_extractor.profile_memory.data_normalize import (
    accumulate_old_memory_entry,
    convert_projects_to_dataclass,
    merge_profiles,
    merge_projects_participated,
    profile_payload_to_memory,
    remove_evidences_from_profile,
)
from memory_layer.memory_extractor.profile_memory.evidence_utils import (
    filter_opinion_tendency_by_type,
    remove_entries_without_evidence,
)
from memory_layer.memory_extractor.profile_memory.project_helpers import filter_project_items_by_type
from memory_layer.memory_extractor.profile_memory.merger import convert_important_info_to_evidence
from memory_layer.memory_extractor.profile_memory.types import (
    GroupImportanceEvidence,
    ImportanceEvidence,
    ProfileMemory,
    ProfileMemoryExtractRequest,
    ProjectInfo,
)
from memory_layer.memory_extractor.base_memory_extractor import (
    MemoryExtractor,
    MemoryExtractRequest,
)

logger = get_logger(__name__)


class ProfileMemoryExtractor(MemoryExtractor):
    """Extractor for user profile information from conversations."""

    _conversation_date_map: Dict[str, str] = {}

    def __init__(self, llm_provider: LLMProvider | None = None):
        super().__init__(MemoryType.PROFILE)
        self.llm_provider = llm_provider

    async def extract_memory(
        self, request: ProfileMemoryExtractRequest
    ) -> Optional[List[ProfileMemory]]:
        """Extract profile memories from conversation memcells."""
        if not request.memcell_list:
            return None

        self.__class__._conversation_date_map = {}

        # Extract complete user_id to user_name mapping from all memcells and old memories
        user_id_to_name = extract_user_mapping_from_memcells(
            request.memcell_list, old_memory_list=request.old_memory_list
        )

        conversation_date_map = self.__class__._conversation_date_map
        all_conversation_text: List[str] = []
        all_episode_text: List[str] = []
        valid_conversation_ids: Set[str] = set()
        conversation_participants_map: Dict[str, Optional[AbstractSet[str]]] = {}
        for memcell in request.memcell_list:
            conversation_text, conversation_id = build_conversation_text(
                memcell, user_id_to_name
            )
            all_conversation_text.append(conversation_text)

            # episode_text, episode_id = build_episode_text(memcell, user_id_to_name)
            # all_episode_text.append(episode_text)

            timestamp_value = getattr(memcell, "timestamp", None)
            dt_value = self._parse_timestamp(timestamp_value)
            if dt_value is None:
                msg_timestamp = self._extract_first_message_timestamp(memcell)
                if msg_timestamp is not None:
                    dt_value = self._parse_timestamp(msg_timestamp)
            date_str: Optional[str] = None
            if dt_value:
                date_str = dt_value.date().isoformat()

            event_id_raw = getattr(memcell, "event_id", None)
            event_id_str = str(event_id_raw) if event_id_raw is not None else None
            participants_raw = getattr(memcell, "participants", None)
            participants_set: Optional[AbstractSet[str]] = None
            if participants_raw:
                normalized_participants = {
                    str(participant).strip()
                    for participant in participants_raw
                    if str(participant).strip()
                }
                if normalized_participants:
                    participants_set = frozenset(normalized_participants)

            if event_id_str is not None:
                conversation_participants_map[event_id_str] = participants_set

            if conversation_id:
                valid_conversation_ids.add(conversation_id)
                if date_str:
                    conversation_date_map.setdefault(conversation_id, date_str)
            if event_id_str:
                valid_conversation_ids.add(event_id_str)
                if date_str:
                    conversation_date_map.setdefault(event_id_str, date_str)

        resolved_group_id = request.group_id
        if not resolved_group_id:
            for memcell in request.memcell_list:
                candidate_group_id = getattr(memcell, "group_id", None)
                if candidate_group_id:
                    resolved_group_id = candidate_group_id
                    break
        resolved_group_id = resolved_group_id or ""

        participants_profile_list: List[Dict[str, Any]] = []
        participants_profile_list_no_evidences: List[Dict[str, Any]] = []
        participants_base_memory_map: Dict[str, Dict[str, Any]] = {}

        if request.old_memory_list:
            for mem in request.old_memory_list:
                if mem.memory_type == MemoryType.PROFILE:
                    accumulate_old_memory_entry(mem, participants_profile_list)
                    if participants_profile_list:
                        profile_obj_no_evidences = remove_evidences_from_profile(
                            participants_profile_list[-1]
                        )
                        participants_profile_list_no_evidences.append(
                            profile_obj_no_evidences
                        )
                elif mem.memory_type == MemoryType.BASE_MEMORY:
                    base_memory_obj: Dict[str, Any] = {"user_id": mem.user_id}

                    if getattr(mem, "position", None):
                        base_memory_obj["position"] = getattr(mem, "position", None)
                    if getattr(mem, "base_location", None):
                        base_memory_obj["base_location"] = getattr(
                            mem, "base_location", None
                        )
                    if getattr(mem, "department", None):
                        base_memory_obj["department"] = getattr(mem, "department", None)

                    if len(base_memory_obj) > 1:
                        participants_base_memory_map[mem.user_id] = base_memory_obj

        # Build two prompts (get via PromptManager)
        prompt_part1 = build_profile_prompt(
            get_prompt_by("CONVERSATION_PROFILE_PART1_EXTRACTION_PROMPT"),
            all_conversation_text,
            participants_profile_list_no_evidences,
            participants_base_memory_map,
            request,
        )
        prompt_part2 = build_profile_prompt(
            get_prompt_by("CONVERSATION_PROFILE_PART2_EXTRACTION_PROMPT"),
            all_conversation_text,
            participants_profile_list_no_evidences,
            participants_base_memory_map,
            request,
        )

        # Define async LLM invocation function
        async def invoke_llm(
            prompt: str, part_label: str
        ) -> Optional[List[Dict[str, Any]]]:
            extraction_attempts = 2
            response: Optional[str] = None
            parsed_profiles: Optional[List[Dict[str, Any]]] = None

            for attempt in range(extraction_attempts):
                try:
                    logger.info(
                        f"Starting {attempt+1} time {part_label} profile extraction"
                    )
                    response = await self.llm_provider.generate(prompt, temperature=0.3)
                    # Cannot batch convert relative dates anymore, because offline processing handles more than one day's data at once
                    annotated_response = response
                    parsed_profiles = self._extract_user_profiles_from_response(
                        annotated_response, part_label
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "%s profile extraction failed (attempt %s/%s): %s",
                        part_label,
                        attempt + 1,
                        extraction_attempts,
                        exc,
                    )
                    if response:
                        logger.warning(
                            "%s response preview (attempt %s): %s",
                            part_label,
                            attempt + 1,
                            response[:800],
                        )

                    if attempt < extraction_attempts - 1:
                        response = None
                        parsed_profiles = None
                        continue

                    repair_prompt = (
                        "The input string is in json format but has syntax errors. Please fix the syntax errors and output only the correctly formatted json ```json {}``` "
                        "Do not include any additional explanations or notes.\n Original string:\n"
                        + (response or "")
                    )

                    try:
                        logger.info(f"Starting to repair {part_label} profile response")
                        response = await self.llm_provider.generate(
                            repair_prompt, temperature=0
                        )
                        # Cannot batch convert relative dates anymore, because offline processing handles more than one day's data at once
                        annotated_response = response
                        parsed_profiles = self._extract_user_profiles_from_response(
                            annotated_response, part_label
                        )
                    except Exception as repair_exc:
                        logger.error(
                            "%s repair prompt failed: %s", part_label, repair_exc
                        )
                        if response:
                            logger.error(
                                "%s repair response preview: %s",
                                part_label,
                                response[:500],
                            )
                        return None

                    break

            return parsed_profiles

        # Serially invoke two LLMs
        profiles_part1 = await invoke_llm(prompt_part1, "personal profile part")
        profiles_part2 = await invoke_llm(prompt_part2, "project profile part")

        # Merge results
        if not profiles_part1 and not profiles_part2:
            logger.warning("Both parts returned no profiles")
            return None

        # Use pre-extracted user_id_to_name mapping for validation
        participant_user_ids: Set[str] = set(user_id_to_name.keys())

        part1_map: Dict[str, Dict[str, Any]] = {}
        if profiles_part1:
            for profile in profiles_part1:
                if not isinstance(profile, dict):
                    continue
                user_id = str(profile.get("user_id", "")).strip()
                if not user_id:
                    logger.info("LLM returned empty user_id in part1; skipping profile")
                    continue

                # Validate user_id against participants_profile_list
                if participant_user_ids and user_id not in participant_user_ids:
                    logger.debug(
                        "LLM returned user_id %s not found in participants_profile_list; skipping profile",
                        user_id,
                    )
                    continue
                part1_map[user_id] = profile

        part2_map: Dict[str, Dict[str, Any]] = {}
        if profiles_part2:
            for profile in profiles_part2:
                if not isinstance(profile, dict):
                    continue
                user_id = str(profile.get("user_id", "")).strip()
                if not user_id:
                    logger.info("LLM returned empty user_id in part2; skipping profile")
                    continue
                # Validate user_id against participants_profile_list
                if participant_user_ids and user_id not in participant_user_ids:
                    logger.debug(
                        "LLM returned user_id %s not found in participants_profile_list; skipping profile",
                        user_id,
                    )
                    continue
                part2_map[user_id] = profile

        # Merge data from both parts
        combined_user_ids = set(part1_map) | set(part2_map)
        if not combined_user_ids:
            logger.warning("No valid user_ids found in combined results")
            return None

        user_profiles_data: List[Dict[str, Any]] = []
        for user_id in combined_user_ids:
            combined_profile: Dict[str, Any] = {"user_id": user_id}

            # Merge part1 data (personal attributes: opinion_tendency, working_habit_preference, soft_skills, personality, way_of_decision_making, hard_skills)
            if user_id in part1_map:
                part1_profile = part1_map[user_id]
                for key, value in part1_profile.items():
                    if key != "user_id" and value:
                        combined_profile[key] = value

            # Merge part2 data (role_responsibility + opinion_tendency + project experience)
            if user_id in part2_map:
                part2_profile = part2_map[user_id]
                # Prefer user_name from part1, otherwise use part2
                if "user_name" not in combined_profile and "user_name" in part2_profile:
                    combined_profile["user_name"] = part2_profile["user_name"]
                # Extract role_responsibility from part2
                if "role_responsibility" in part2_profile:
                    combined_profile["role_responsibility"] = part2_profile[
                        "role_responsibility"
                    ]
                # Extract opinion_tendency from part2 and filter out invalid types
                if "opinion_tendency" in part2_profile:
                    filtered_opinion = filter_opinion_tendency_by_type(
                        part2_profile["opinion_tendency"]
                    )
                    if isinstance(filtered_opinion, list):
                        combined_profile["opinion_tendency"] = filtered_opinion
                # Extract projects_participated from part2 and filter out invalid types
                if "projects_participated" in part2_profile:
                    filtered_projects = filter_project_items_by_type(
                        part2_profile["projects_participated"]
                    )
                    if filtered_projects is not None:
                        combined_profile["projects_participated"] = filtered_projects

            user_profiles_data.append(combined_profile)

        # Filter out profiles with all key fields empty
        filtered_profiles_data: List[Dict[str, Any]] = []
        key_fields = [
            "hard_skills",
            "soft_skills",
            "output_reasoning",
            "motivation_system",
            "fear_system",
            "value_system",
            "humor_use",
            "colloquialism",
            "way_of_decision_making",
            "personality",
            "projects_participated",
            "user_goal",
            "work_responsibility",
            "working_habit_preference",
            "interests",
            "tendency",
        ]

        for profile_data in user_profiles_data:
            # Check if at least one key field has non-empty value
            has_valid_data = False
            for field in key_fields:
                value = profile_data.get(field)
                if value:  # Non-empty list/string/dict
                    has_valid_data = True
                    break

            if has_valid_data:
                filtered_profiles_data.append(profile_data)
            else:
                logger.debug(
                    "Filtering out profile for user %s: all key fields are empty",
                    profile_data.get("user_id"),
                )

        await complete_missing_evidences(
            filtered_profiles_data,
            conversation_lines=all_conversation_text,
            valid_conversation_ids=valid_conversation_ids,
            conversation_participants_map=conversation_participants_map,
            conversation_date_map=conversation_date_map,
            llm_provider=self.llm_provider,
            parse_payload=self._parse_profile_response_payload,
        )
        for profile_data in filtered_profiles_data:
            remove_entries_without_evidence(profile_data)

        profile_memories: List[ProfileMemory] = []
        for profile_data in filtered_profiles_data:
            if not isinstance(profile_data, dict):
                continue

            projects_participated = profile_data.get("projects_participated")
            project_payload_override: Optional[Dict[str, Any]] = None
            if isinstance(projects_participated, list):
                project_infos = convert_projects_to_dataclass(
                    projects_participated,
                    valid_conversation_ids=valid_conversation_ids,
                    conversation_date_map=conversation_date_map,
                )
                if len(project_infos) > 1:
                    logger.error(
                        "Unexpected multiple projects for user %s in group %s",
                        profile_data.get("user_id"),
                        request.group_id,
                    )
                    project_infos = merge_projects_participated(None, project_infos)
                if project_infos:
                    project_payload_override = {"projects_participated": project_infos}

            profile_memory = profile_payload_to_memory(
                profile_data,
                group_id=resolved_group_id,
                project_data=project_payload_override,
                valid_conversation_ids=valid_conversation_ids,
                conversation_date_map=conversation_date_map,
            )
            if profile_memory:
                profile_memories.append(profile_memory)

        merged_profiles = merge_profiles(
            profile_memories,
            participants_profile_list,
            group_id=resolved_group_id,
            valid_conversation_ids=valid_conversation_ids,
            conversation_date_map=conversation_date_map,
        )

        important_info = extract_group_important_info(
            request.memcell_list, request.group_id
        )
        new_evidence_list = convert_important_info_to_evidence(important_info)
        for profile in merged_profiles:
            old_evidence: Optional[GroupImportanceEvidence] = (
                profile.group_importance_evidence
            )
            new_evidence = merge_group_importance_evidence(
                old_evidence, new_evidence_list, user_id=profile.user_id
            )
            if new_evidence:
                new_evidence.is_important = is_important_to_user(
                    new_evidence.evidence_list
                )
                profile.group_importance_evidence = new_evidence

        return merged_profiles

    def _extract_user_profiles_from_response(
        self, response: str, part_label: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract user profiles from LLM response."""
        try:
            data = self._parse_profile_response_payload(response)
            user_profiles = data.get("user_profiles", [])
            if not user_profiles:
                logger.info(f"No user profiles found in {part_label}")
                return None
            return user_profiles
        except Exception as exc:
            logger.error(f"Failed to parse {part_label} llm response: {exc}")
            if response:
                logger.error(f"{part_label} llm response preview: {response[:500]}")
            return None

    @staticmethod
    def _parse_timestamp(timestamp: Any) -> Optional[datetime]:
        if isinstance(timestamp, datetime):
            return timestamp
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        if isinstance(timestamp, str):
            ts_value = timestamp.strip()
            iso_timestamp = (
                ts_value.replace("Z", "+00:00") if ts_value.endswith("Z") else ts_value
            )
            try:
                return datetime.fromisoformat(iso_timestamp)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_first_message_timestamp(memcell: MemCell) -> Optional[Any]:
        """Return the first available timestamp from a memcell's original data."""
        for message in getattr(memcell, "original_data", []) or []:
            if hasattr(message, "content"):
                ts_value = message.content.get("timestamp")
            else:
                ts_value = message.get("timestamp")
            if ts_value:
                return ts_value
        return None

    @staticmethod
    def _parse_profile_response_payload(response: str) -> Dict[str, Any]:
        """Best-effort JSON extraction from LLM responses with optional markdown fences."""
        if not response:
            raise ValueError("empty response")

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        parsed = ast.literal_eval(response)
        if isinstance(parsed, dict):
            return parsed
        elif isinstance(parsed, list):
            return {"user_profiles": parsed}
        return json.loads(parsed)

    async def extract_profile_companion(
        self, request: ProfileMemoryExtractRequest
    ) -> Optional[List[ProfileMemory]]:
        """Extract companion profile memories using Part3 prompts (90 personality dimensions).

        This function analyzes conversation memcells to extract comprehensive personality profiles
        based on 90 dimensions including psychological traits, AI alignment preferences,
        and content platform interests.

        Args:
            request: ProfileMemoryExtractRequest containing memcells and optional old memories

        Returns:
            Optional[List[ProfileMemory]]: List of extracted profile memories with 90-dimension analysis,
                                           or None if extraction failed
        """
        if not request.memcell_list:
            logger.warning(
                "[ProfileMemoryExtractor] No memcells provided for companion extraction"
            )
            print(f"[ProfileExtractor] ❌ memcell_list is empty")
            return None

        print(f"[ProfileExtractor] Received {len(request.memcell_list)} MemCells")
        print(f"[ProfileExtractor] request.user_id_list: {request.user_id_list}")
        print(f"[ProfileExtractor] request.group_id: {request.group_id}")

        # Extract user mapping from memcells and build conversation text
        user_id_to_name = extract_user_mapping_from_memcells(request.memcell_list)
        print(
            f"[ProfileExtractor] user_id_to_name (extracted from original_data): {user_id_to_name}"
        )

        # 🔧 If user_id_to_name is empty, extract from participants field
        if not user_id_to_name:
            print(
                f"[ProfileExtractor] user_id_to_name is empty, attempting to extract from participants"
            )
            for memcell in request.memcell_list:
                participants = getattr(memcell, "participants", None)
                if participants and isinstance(participants, list):
                    for user_id in participants:
                        if user_id and user_id not in user_id_to_name:
                            user_id_to_name[user_id] = (
                                user_id  # Use user_id as default name
                            )
                    print(
                        f"[ProfileExtractor] Extracted from participants: {list(participants)}"
                    )

        # 🔧 If still empty, use request.user_id_list
        if not user_id_to_name and request.user_id_list:
            print(f"[ProfileExtractor] Still empty, using request.user_id_list")
            for user_id in request.user_id_list:
                user_id_to_name[user_id] = user_id

        print(f"[ProfileExtractor] Final user_id_to_name: {user_id_to_name}")
        # Build conversation text from all memcells
        conversation_lines: List[str] = []
        user_profiles: Dict[str, Dict[str, Any]] = (
            {}
        )  # user_id -> {name, message_count}

        # Build evidence maps (date per conversation/event id) for evidences binding
        conversation_date_map: Dict[str, str] = {}
        valid_conversation_ids: Set[str] = set()
        default_date: Optional[str] = None

        for memcell in request.memcell_list:
            # 🔧 Directly use episode, because original_data is often empty
            episode_text, event_id = build_episode_text(memcell, user_id_to_name)

            if episode_text:
                conversation_lines.append(episode_text)
                print(f"[ProfileExtractor] Using episode_text: {episode_text[:200]}...")
                conversation_id = event_id
            else:
                print(
                    f"[ProfileExtractor] ⚠️  episode is empty, trying conversation_text fallback"
                )
                # Fallback: try conversation_text
                conversation_text, conversation_id = build_conversation_text(
                    memcell, user_id_to_name
                )
                if conversation_text and conversation_text.strip():
                    conversation_lines.append(conversation_text)
                    print(
                        f"[ProfileExtractor] Using conversation_text: {conversation_text[:200]}..."
                    )
                else:
                    print(
                        f"[ProfileExtractor] ❌ Both episode and conversation_text are empty!"
                    )

            # Collect user statistics
            # 🔧 Only count users in request.user_id_list (already filtered robot/assistant)
            target_user_ids = (
                set(request.user_id_list)
                if request.user_id_list
                else set(user_id_to_name.keys())
            )
            for user_id in user_id_to_name.keys():
                # 🔧 Only process users in the target user list
                if user_id not in target_user_ids:
                    continue

                if user_id not in user_profiles:
                    user_profiles[user_id] = {
                        "user_id": user_id,
                        "user_name": user_id_to_name.get(user_id, "Unknown"),
                        "message_count": 0,
                    }
                user_profiles[user_id]["message_count"] += 1

            # Evidence date mapping
            timestamp_value = getattr(memcell, "timestamp", None)
            dt_value = self._parse_timestamp(timestamp_value)
            if dt_value is None:
                msg_timestamp = self._extract_first_message_timestamp(memcell)
                if msg_timestamp is not None:
                    dt_value = self._parse_timestamp(msg_timestamp)
            date_str: Optional[str] = None
            if dt_value:
                date_str = dt_value.date().isoformat()
                default_date = date_str

            if conversation_id:
                valid_conversation_ids.add(conversation_id)
                if date_str:
                    conversation_date_map.setdefault(conversation_id, date_str)
            event_id = getattr(memcell, "event_id", None)
            if event_id:
                event_id_str = str(event_id)
                valid_conversation_ids.add(event_id_str)
                if date_str:
                    conversation_date_map.setdefault(event_id_str, date_str)

        if not conversation_lines:
            logger.warning(
                "[ProfileMemoryExtractor] No conversation text to analyze for companion profiles"
            )
            print(f"[ProfileExtractor] ❌ conversation_lines is empty")
            return None

        conversation_text = "\n".join(conversation_lines)
        print(
            f"[ProfileExtractor] conversation_text length: {len(conversation_text)} characters"
        )
        print(f"[ProfileExtractor] user_profiles: {user_profiles}")
        logger.info(
            f"[ProfileMemoryExtractor] Built companion conversation with {len(conversation_lines)} segments"
        )
        logger.info(
            f"[ProfileMemoryExtractor] Found {len(user_profiles)} unique users for companion analysis"
        )

        # Retrieve old profile information if available
        old_profiles_map: Dict[str, ProfileMemory] = {}
        if request.old_memory_list:
            for mem in request.old_memory_list:
                if mem.memory_type == MemoryType.PROFILE and hasattr(mem, 'user_id'):
                    old_profiles_map[mem.user_id] = mem

        # Extract Part3 profiles for each user (🚀 parallelize LLM calls)
        companion_profiles: List[ProfileMemory] = []

        # Define single-user extraction function
        async def extract_single_user_companion_profile(
            user_id: str, user_info: Dict[str, Any]
        ) -> List[ProfileMemory]:
            """Extract companion profile for a single user (parallelized)"""
            print(
                f"[ProfileExtractor] Starting to extract Profile for user {user_info['user_name']} (user_id={user_id})"
            )
            logger.info(
                f"[ProfileMemoryExtractor] Analyzing companion profile for: {user_info['user_name']} "
                f"({user_info['message_count']} messages)"
            )

            # Build Part3 prompt (retrieved via PromptManager)
            prompt = get_prompt_by("CONVERSATION_PROFILE_PART3_EXTRACTION_PROMPT")
            prompt += f"\n\n**Existing User Profile:**\n"
            prompt += f"User ID: {user_id}\n"
            prompt += f"User Name: {user_info['user_name']}\n"

            # Add old profile information if available
            if user_id in old_profiles_map:
                old_profile = old_profiles_map[user_id]
                prompt += f"\n**Previous Profile Summary:**\n"
                if hasattr(old_profile, 'personality') and old_profile.personality:
                    prompt += f"Personality: {old_profile.personality}\n"
                if hasattr(old_profile, 'soft_skills') and old_profile.soft_skills:
                    prompt += f"Soft Skills: {old_profile.soft_skills}\n"

            prompt += f"\n**New Conversation:**\n{conversation_text}\n"
            # Ask for structured JSON for companion fields (aligned with normalization schema)
            prompt += (
                f"\n**Task:** For user {user_info['user_name']}, extract ONLY these fields as JSON: "
                "personality, way_of_decision_making, working_habit_preference, interests, tendency, "
                "motivation_system, fear_system, value_system, humor_use, colloquialism, output_reasoning. "
                "For list-type fields, each item must be an object: {\\\"value\\\": string, \\\"evidences\\\": [string], \\\"level\\\": string?}. "
                "Use evidences referencing conversation ids when possible (e.g., [conversation_id:EVENT_ID] or EVENT_ID). "
                "Include user_id and user_name. Use ASCII quotes only. Return one fenced JSON block, no extra text.\n"
                "Exact response template:\n"
                "```json\n"
                "{\n"
                "  \"user_profiles\": [\n"
                "    {\n"
                "      \"user_id\": \"USER_ID\",\n"
                "      \"user_name\": \"USER_NAME\",\n"
                "      \"personality\": [],\n"
                "      \"way_of_decision_making\": [],\n"
                "      \"working_habit_preference\": [],\n"
                "      \"interests\": [],\n"
                "      \"tendency\": [],\n"
                "      \"motivation_system\": [],\n"
                "      \"fear_system\": [],\n"
                "      \"value_system\": [],\n"
                "      \"humor_use\": [],\n"
                "      \"colloquialism\": [],\n"
                "      \"output_reasoning\": \"\"\n"
                "    }\n"
                "  ]\n"
                "}\n"
                "```\n"
            )

            # Call LLM for analysis
            try:
                print(
                    f"[ProfileExtractor] Calling LLM to extract Profile for {user_info['user_name']}..."
                )
                response_text = await self.llm_provider.generate(
                    prompt, temperature=0.3
                )
                print(f"[ProfileExtractor] LLM returned: {response_text[:200]}...")
                logger.info(
                    f"[ProfileMemoryExtractor] ✅ Successfully extracted companion profile for {user_info['user_name']}"
                )

                # First try: structured JSON path compatible with existing normalization
                structured_profiles: Optional[List[Dict[str, Any]]] = None
                try:
                    # annotated = self._annotate_relative_dates(response_text)
                    annotated = response_text
                    structured_profiles = self._extract_user_profiles_from_response(
                        annotated, "companion profile"
                    )
                except Exception:
                    structured_profiles = None

                user_profiles_result: List[ProfileMemory] = []

                if structured_profiles:
                    print(
                        f"[ProfileExtractor] Parsed {len(structured_profiles)} structured Profiles"
                    )
                    # Ensure user_id/user_name present and add fallback evidences when missing
                    # Also route through profile_payload_to_memory for unified normalization
                    fallback_evidences: List[str] = []
                    # Prefer event_ids for fallback evidences
                    batch_event_ids: List[str] = [
                        str(mc.event_id)
                        for mc in request.memcell_list
                        if hasattr(mc, 'event_id') and mc.event_id
                    ]
                    for ev in batch_event_ids:
                        ev_date = conversation_date_map.get(ev) or default_date
                        fallback_evidences.append(f"{ev_date}|{ev}" if ev_date else ev)

                    for p in structured_profiles:
                        if not isinstance(p, dict):
                            continue
                        if not p.get("user_id"):
                            p["user_id"] = user_id
                        if not p.get("user_name"):
                            p["user_name"] = user_info["user_name"]

                        for field in (
                            "personality",
                            "way_of_decision_making",
                            "working_habit_preference",
                            "interests",
                            "tendency",
                        ):
                            items = p.get(field)
                            if isinstance(items, list):
                                for it in items:
                                    if not isinstance(it, dict):
                                        continue
                                    raw_evidences = it.get("evidences")
                                    normalized_evs: List[str] = []
                                    if isinstance(raw_evidences, list):
                                        for ev in raw_evidences:
                                            try:
                                                s = str(ev).strip()
                                            except Exception:
                                                s = ""
                                            if not s:
                                                continue
                                            # Try to extract conversation id from forms like "[conversation_id:ID]" or raw ID
                                            conv_id: Optional[str] = None
                                            if "conversation_id:" in s:
                                                conv_id = s.split("conversation_id:")[
                                                    -1
                                                ].strip("[]() ,.\t\n")
                                            else:
                                                conv_id = s
                                            if (
                                                conv_id
                                                and conv_id in valid_conversation_ids
                                            ):
                                                ev_date = (
                                                    conversation_date_map.get(conv_id)
                                                    or default_date
                                                )
                                                normalized_evs.append(
                                                    f"{ev_date}|{conv_id}"
                                                    if ev_date
                                                    else conv_id
                                                )
                                    # If after normalization nothing remains, fallback to batch evidences
                                    if not normalized_evs:
                                        normalized_evs = list(fallback_evidences)
                                    it["evidences"] = normalized_evs

                        mem = profile_payload_to_memory(
                            p,
                            group_id=request.group_id or "",
                            project_data=None,
                            valid_conversation_ids=valid_conversation_ids,
                            # default_date=default_date,
                            conversation_date_map=conversation_date_map,
                        )
                        if mem:
                            user_profiles_result.append(mem)
                            print(
                                f"[ProfileExtractor] ✅ Successfully converted Profile: user_id={mem.user_id}"
                            )
                        else:
                            print(
                                f"[ProfileExtractor] ⚠️  profile_payload_to_memory returned None"
                            )
                else:
                    print(
                        f"[ProfileExtractor] Failed to parse structured Profile, using fallback"
                    )
                    # Fallback: free-text analysis stored under personality with evidences bound
                    from datetime import datetime as _dt

                    fallback_evidences: List[str] = []
                    batch_event_ids: List[str] = [
                        str(mc.event_id)
                        for mc in request.memcell_list
                        if hasattr(mc, 'event_id') and mc.event_id
                    ]
                    for ev in batch_event_ids:
                        ev_date = conversation_date_map.get(ev) or default_date
                        fallback_evidences.append(f"{ev_date}|{ev}" if ev_date else ev)

                    profile_memory = ProfileMemory(
                        memory_type=MemoryType.PROFILE,
                        user_id=user_id,
                        timestamp=_dt.now(),
                        ori_event_id_list=[
                            mc.event_id
                            for mc in request.memcell_list
                            if hasattr(mc, 'event_id')
                        ],
                        group_id=request.group_id or "",
                        personality=[
                            {
                                "value": "90-dimension-analysis",
                                "evidences": fallback_evidences,
                                "analysis": response_text,
                            }
                        ],
                        working_habit_preference=None,
                        soft_skills=None,
                        hard_skills=None,
                        work_responsibility=None,
                        tendency=None,
                        way_of_decision_making=None,
                        projects_participated=None,
                        group_importance_evidence=None,
                    )
                    user_profiles_result.append(profile_memory)
                    print(
                        f"[ProfileExtractor] ✅ Using fallback Profile: user_id={user_id}"
                    )

                print(
                    f"[ProfileExtractor] Final result returns {len(user_profiles_result)} Profiles"
                )
                return user_profiles_result

            except Exception as exc:
                logger.error(
                    f"[ProfileMemoryExtractor] ❌ Failed to extract companion profile for "
                    f"{user_info['user_name']}: {exc}"
                )
                print(f"[ProfileExtractor] ❌ Extraction failed: {exc}")
                import traceback

                print(traceback.format_exc())
                return []

        # 🚀 Parallel execution of Profile extraction for all users
        logger.info(
            f"[ProfileMemoryExtractor] 🚀 Starting parallel extraction of companion profiles for {len(user_profiles)} users"
        )
        print(
            f"[ProfileExtractor] 🚀 Starting parallel extraction of companion profiles for {len(user_profiles)} users"
        )

        tasks = [
            extract_single_user_companion_profile(user_id, user_info)
            for user_id, user_info in user_profiles.items()
        ]

        print(f"[ProfileExtractor] Created {len(tasks)} extraction tasks")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        print(
            f"[ProfileExtractor] Parallel extraction completed, received {len(results)} results"
        )

        # Collect all successful profiles
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"[ProfileMemoryExtractor] Profile extraction task failed: {result}"
                )
                print(f"[ProfileExtractor] Task {i+1} failed: {result}")
                continue
            if isinstance(result, list):
                companion_profiles.extend(result)
                print(
                    f"[ProfileExtractor] Task {i+1} succeeded, returned {len(result)} Profiles"
                )

        if not companion_profiles:
            logger.warning(
                "[ProfileMemoryExtractor] No companion profiles were successfully extracted"
            )
            print(f"[ProfileExtractor] ❌ Final companion_profiles is empty")
            return None

        logger.info(
            f"[ProfileMemoryExtractor] Successfully extracted {len(companion_profiles)} companion profiles"
        )
        print(
            f"[ProfileExtractor] ✅ Finally successfully extracted {len(companion_profiles)} companion profiles"
        )
        return companion_profiles
