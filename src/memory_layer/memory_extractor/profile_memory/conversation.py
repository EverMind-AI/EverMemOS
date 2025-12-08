"""Conversation parsing utilities for profile memory extraction."""

from __future__ import annotations

import calendar
import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from core.observation.logger import get_logger

# 使用动态语言提示词导入（根据 MEMORY_LANGUAGE 环境变量自动选择）
from ...prompts import CONVERSATION_PROFILE_EVIDENCE_COMPLETION_PROMPT
from api_specs.memory_types import MemCell
from memory_layer.memory_extractor.profile_memory.types import (
    GroupImportanceEvidence,
    ImportanceEvidence,
    ProfileMemoryExtractRequest,
)

logger = get_logger(__name__)


def extract_user_mapping_from_memcells(
    memcells: Iterable[MemCell], old_memory_list: Optional[Iterable[Any]] = None
) -> Dict[str, str]:
    """Extract user_id to user_name mapping from memcells and old memories.

    Args:
        memcells: Iterable of MemCell objects
        old_memory_list: Optional iterable of Memory objects (ProfileMemory, etc.)

    Returns:
        Dictionary mapping user_id (str) to user_name (str)
    """
    user_id_to_name: Dict[str, str] = {}

    # Extract from memcells (speaker info only, referList has no name)
    for memcell in memcells:
        data_list: List[Any] = getattr(memcell, "original_data", []) or []

        for data in data_list:
            # Extract speaker info
            speaker_id = data.get("speaker_id")
            speaker_name = data.get("speaker_name")
            if speaker_id and speaker_name:
                user_id_to_name[str(speaker_id)] = speaker_name

            # Note: referList only has id, no name field, so we skip it

    # Extract from old_memory_list (ProfileMemory, BaseMemory, etc.)
    if old_memory_list:
        for memory in old_memory_list:
            user_id = getattr(memory, "user_id", None)
            user_name = getattr(memory, "user_name", None)
            if user_id and user_name:
                # Only add if not already present (memcells data takes priority)
                if str(user_id) not in user_id_to_name:
                    user_id_to_name[str(user_id)] = user_name

    return user_id_to_name


def _append_user_ids_to_content(
    content: Any, refer_list: Any, *, use_at_symbol: bool = True
) -> Any:
    """Generic function to append user IDs to name mentions in content.

    Args:
        content: The text content to process
        refer_list: List of user references with name and id
        use_at_symbol: If True, matches @name pattern; if False, matches name with word boundaries

    Returns:
        Content with user_id annotations added
    """
    if not isinstance(content, str) or not refer_list:
        return content

    updated_content = content
    processed_names: Set[str] = set()

    for refer in refer_list:
        if not isinstance(refer, dict):
            continue
        name = refer.get("name")
        if not name or name in processed_names:
            continue
        user_id = refer.get("_id") or refer.get("id")
        if user_id is None:
            continue

        user_id_str = str(user_id)

        # Build the pattern and replacement based on use_at_symbol flag
        if use_at_symbol:
            # Match @name that is not already followed by (user_id:
            annotated_form = f"@{name}(user_id:{user_id_str})"
            pattern = re.compile(rf"@{re.escape(str(name))}(?!\(user_id:)")
            replacement = f"@{name}(user_id:{user_id_str})"
        else:
            # Match name with word boundaries that is not already followed by (user_id:
            annotated_form = f"{name}(user_id:{user_id_str})"
            pattern = re.compile(rf"\b{re.escape(str(name))}(?!\(user_id:)")
            replacement = f"{name}(user_id:{user_id_str})"

        # Check if already annotated
        if annotated_form in updated_content:
            processed_names.add(name)
            continue

        # Apply the substitution
        updated_content, count = pattern.subn(
            lambda match: replacement, updated_content
        )
        if count:
            processed_names.add(name)

    return updated_content


def append_refer_user_ids(content: Any, refer_list: Any) -> Any:
    """Append user IDs to @mentions based on refer_list entries."""
    return _append_user_ids_to_content(content, refer_list, use_at_symbol=True)


def append_user_ids_to_names(content: Any, refer_list: Any) -> Any:
    """Append user IDs to name mentions (without @) in episode text based on refer_list entries."""
    return _append_user_ids_to_content(content, refer_list, use_at_symbol=False)


def build_conversation_text(
    memcell: MemCell,
    user_id_to_name: Dict[str, str],
) -> Tuple[str, Optional[str]]:
    """Convert raw data from a memcell into formatted conversation text.

    Args:
        memcell: The memcell containing conversation data
        user_id_to_name: Pre-extracted user_id to user_name mapping

    Returns:
        Tuple of (formatted conversation text, conversation_id)
    """
    conversation_id = getattr(memcell, "event_id", None)
    conversation_id_str = str(conversation_id) if conversation_id is not None else ""
    data_list: List[Any] = getattr(memcell, "original_data", []) or []

    lines: List[str] = []
    for data in data_list:
        messages = data.get("messages")
        for msg in messages:
            speaker_id = msg.get("extend").get("speaker_id")
            speaker_name = msg.get("extend").get("speaker_name")
            # Use extracted mapping as fallback if speaker_name is missing
            if not speaker_name and speaker_id:
                speaker_name = user_id_to_name.get(str(speaker_id), "")

            speaker = f"{speaker_name}(user_id:{speaker_id})" if speaker_id else speaker_name
            content = append_refer_user_ids(
                msg.get("content"),
                msg.get("referList"),
            )
            timestamp = msg.get("timestamp")

            if timestamp:
                lines.append(
                    f"[{timestamp}][conversation_id:{conversation_id_str}] {speaker}说: {content}"
                )
            else:
                lines.append(
                    f"[conversation_id:{conversation_id_str}] {speaker}说: {content}"
                )

    return "\n".join(lines), conversation_id_str or None


def build_episode_text(
    memcell: MemCell, user_id_to_name: Dict[str, str]
) -> Tuple[str, Optional[str]]:
    """Convert episode from a memcell into formatted text with user_id annotations.

    Args:
        memcell: The memcell containing episode data
        user_id_to_name: Pre-extracted user_id to user_name mapping

    Returns:
        Tuple of (formatted episode text, event_id)
    """
    event_id = getattr(memcell, "event_id", None)
    event_id_str = str(event_id) if event_id is not None else ""
    episode_content = getattr(memcell, "episode", None) or ""

    if not episode_content:
        return "", event_id_str or None

    # Get participants (list of user_ids)
    participants = getattr(memcell, "participants", None) or []

    # Build referList for append_user_ids_to_names using participants
    aggregated_refer_list: List[Dict[str, Any]] = []
    for user_id in participants:
        user_id_str = str(user_id)
        user_name = user_id_to_name.get(user_id_str, "")
        if user_name:
            aggregated_refer_list.append({"_id": user_id_str, "name": user_name})

    # Apply user_id annotations to episode content (without @ symbol)
    annotated_content = append_user_ids_to_names(episode_content, aggregated_refer_list)

    # Format with timestamp and event_id
    timestamp = getattr(memcell, "timestamp", None)
    return (
        f"[{timestamp}][episode_id:{event_id_str}] {annotated_content}",
        event_id_str or None,
    )


def annotate_relative_dates(text: str, base_date: Optional[str] = None) -> str:
    """Append absolute dates after relative date phrases in the LLM response.

    Args:
        text: The text to annotate
        base_date: ISO format date string (YYYY-MM-DD) to use as reference point

    Returns:
        Text with relative dates annotated with absolute dates
    """
    if not text or not base_date:
        return text

    try:
        reference_date = datetime.fromisoformat(base_date).date()
    except ValueError:
        return text

    def month_end(offset: int) -> date:
        year = reference_date.year
        month = reference_date.month + offset
        while month < 1:
            month += 12
            year -= 1
        while month > 12:
            month -= 12
            year += 1
        last_day = calendar.monthrange(year, month)[1]
        return datetime(year, month, last_day).date()

    def already_annotated(full_text: str, end_index: int) -> bool:
        tail = full_text[end_index:]
        return bool(re.match(r"^\s*[（\(]\d{4}-\d{2}-\d{2}", tail))

    def compute_month(offset: int) -> date:
        return month_end(offset)

    english_rules = {
        "today": lambda: reference_date,
        "tomorrow": lambda: reference_date + timedelta(days=1),
        "yesterday": lambda: reference_date - timedelta(days=1),
        "this week": lambda: reference_date,
        "last week": lambda: reference_date - timedelta(days=7),
        "next week": lambda: reference_date + timedelta(days=7),
        "this month": lambda: compute_month(0),
        "last month": lambda: compute_month(-1),
        "next month": lambda: compute_month(1),
    }

    chinese_rules = {
        "今天": english_rules["today"],
        "明天": english_rules["tomorrow"],
        "第二天": english_rules["tomorrow"],
        "昨天": english_rules["yesterday"],
        "本周": english_rules["this week"],
        "这周": english_rules["this week"],
        "上周": english_rules["last week"],
        "下周": english_rules["next week"],
        "本月": english_rules["this month"],
        "这个月": english_rules["this month"],
        "上个月": english_rules["last month"],
        "下个月": english_rules["next month"],
    }

    english_pattern = re.compile(
        r"\b(today|tomorrow|yesterday|this week|last week|next week|this month|last month|next month)\b",
        re.IGNORECASE,
    )
    chinese_pattern = re.compile(
        "(今天|明天|昨天|本周|这周|上周|下周|本月|这个月|上个月|下个月)"
    )

    def english_repl(match: re.Match[str]) -> str:
        original = match.group(0)
        normalized = re.sub(r"\s+", " ", original.lower())
        compute = english_rules.get(normalized)
        if not compute:
            return original
        if already_annotated(match.string, match.end()):
            return original
        absolute_date = compute().isoformat()
        return f"{original} ({absolute_date})"

    def chinese_repl(match: re.Match[str]) -> str:
        original = match.group(0)
        compute = chinese_rules.get(original)
        if not compute:
            return original
        if already_annotated(match.string, match.end()):
            return original
        absolute_date = compute().isoformat()
        return f"{original} ({absolute_date})"

    updated_text = english_pattern.sub(english_repl, text)
    updated_text = chinese_pattern.sub(chinese_repl, updated_text)
    return updated_text


def extract_group_important_info(
    memcells: Iterable[MemCell], group_id: str
) -> Dict[str, Any]:
    """Aggregate statistics used to determine user importance within a group."""
    group_data = {
        "group_id": group_id,
        "user_data": {},
        "group_data": {"total_messages": 0},
    }

    for memcell in memcells:
        for msg in getattr(memcell, "original_data", []) or []:
            group_data["group_data"]["total_messages"] += 1
            user_id = msg.get("speaker_id")
            name = msg.get("speaker_name")
            refer_list = msg.get("referList", [])
            if not isinstance(refer_list, list):
                refer_list = []
            for refer in refer_list:
                refer_id = refer.get("id")
                refer_name = refer.get("name")
                if refer_id not in group_data["user_data"]:
                    group_data["user_data"][refer_id] = {
                        "name": refer_name,
                        "at_count": 1,
                        "chat_count": 0,
                    }
                else:
                    group_data["user_data"][refer_id]["at_count"] += 1
            if user_id not in group_data["user_data"]:
                group_data["user_data"][user_id] = {
                    "name": name,
                    "at_count": 0,
                    "chat_count": 1,
                }
            else:
                group_data["user_data"][user_id]["chat_count"] += 1
    return group_data


def is_important_to_user(evidence_list: List[ImportanceEvidence]) -> bool:
    """Determine whether a group is important to a user based on evidence metrics."""
    speaker_sum = 0
    refer_sum = 0
    conversation_sum = 0
    for evidence in evidence_list:
        speaker_sum += evidence.speak_count
        refer_sum += evidence.refer_count
        conversation_sum += evidence.conversation_count
    if speaker_sum + refer_sum >= 5:
        return True
    if conversation_sum and speaker_sum / conversation_sum > 0.1:
        return True
    if refer_sum >= 2:
        return True
    return False


def merge_group_importance_evidence(
    existing_evidence: Optional[GroupImportanceEvidence],
    new_evidence_list: Optional[List[ImportanceEvidence]],
    *,
    user_id: str,
) -> Optional[GroupImportanceEvidence]:
    """Merge group importance evidences while limiting the total records."""
    if not existing_evidence and not new_evidence_list:
        return None

    matching_evidence: Optional[ImportanceEvidence] = None
    if new_evidence_list:
        for evidence in new_evidence_list:
            if evidence.user_id == user_id:
                matching_evidence = evidence
                break

    if not existing_evidence and not matching_evidence:
        return None

    if not existing_evidence:
        if not matching_evidence:
            return None
        return GroupImportanceEvidence(
            group_id=matching_evidence.group_id,
            evidence_list=[matching_evidence],
            is_important=False,
        )

    if not matching_evidence:
        return existing_evidence

    existing_evidence.evidence_list.append(matching_evidence)
    if len(existing_evidence.evidence_list) > 10:
        existing_evidence.evidence_list = existing_evidence.evidence_list[:10]
    return existing_evidence


def build_profile_prompt(
    prompt_template: str,
    conversation_lines: List[str],
    participants_profile_list_no_evidences: List[Dict[str, Any]],
    participants_base_memory_map: Dict[str, Dict[str, Any]],
    request: ProfileMemoryExtractRequest,
) -> str:
    """Construct a profile extraction prompt using shared conversation context."""
    return (
        prompt_template.replace("{conversation}", "\n".join(conversation_lines))
        .replace(
            "{participants_profile}",
            json.dumps(participants_profile_list_no_evidences, ensure_ascii=False),
        )
        .replace(
            "{participants_baseMemory}",
            json.dumps(participants_base_memory_map, ensure_ascii=False),
        )
        .replace("{project_name}", request.group_name or "")
        .replace("{project_id}", request.group_id or "")
    )


def build_evidence_completion_prompt(
    conversation_text: str, profiles_without_evidences: List[Dict[str, Any]]
) -> str:
    """Construct the evidence completion prompt for a batch of user profiles."""
    return CONVERSATION_PROFILE_EVIDENCE_COMPLETION_PROMPT.replace(
        "{conversation}", conversation_text
    ).replace(
        "{user_profiles_without_evidences}",
        json.dumps(profiles_without_evidences, ensure_ascii=False),
    )
