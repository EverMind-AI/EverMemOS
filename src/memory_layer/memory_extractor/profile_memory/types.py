"""Dataclasses and type definitions for profile memory extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...types import Memory, MemoryType, MemCell
from ..base_memory_extractor import MemoryExtractRequest


@dataclass
class ProjectInfo:
    """Project participation information."""

    project_id: str
    project_name: str
    entry_date: str
    subtasks: Optional[List[Dict[str, Any]]] = None
    user_objective: Optional[List[Dict[str, Any]]] = None
    contributions: Optional[List[Dict[str, Any]]] = None
    user_concerns: Optional[List[Dict[str, Any]]] = None


@dataclass
class ImportanceEvidence:
    """Aggregated evidence indicating user importance within a group."""

    user_id: str
    group_id: str
    speak_count: int = 0
    refer_count: int = 0
    conversation_count: int = 0


@dataclass
class GroupImportanceEvidence:
    """Group-level importance assessment for a user."""

    group_id: str
    evidence_list: List[ImportanceEvidence]
    is_important: bool


@dataclass
class ProfileMemory(Memory):
    """
    Profile memory result class.

    Contains user profile information extracted from conversations.
    All list attributes now contain dicts with 'value' and 'evidences' fields.
    """

    user_name: Optional[str] = None

    # Skills: [{"value": "Python", "level": "高级", "evidences": ["2024-01-01|conv_123"]}]
    # Legacy format: [{"skill": "Python", "level": "高级", "evidences": ["..."]}]
    hard_skills: Optional[List[Dict[str, Any]]] = None
    soft_skills: Optional[List[Dict[str, Any]]] = None

    output_reasoning: Optional[str] = None
    

    # Other attributes: [{"value": "xxx", "evidences": ["2024-01-01|conv_123"]}]
    way_of_decision_making: Optional[List[Dict[str, Any]]] = None
    personality: Optional[List[Dict[str, Any]]] = None
    projects_participated: Optional[List[ProjectInfo]] = None
    user_goal: Optional[List[Dict[str, Any]]] = None
    work_responsibility: Optional[List[Dict[str, Any]]] = None
    working_habit_preference: Optional[List[Dict[str, Any]]] = None
    interests: Optional[List[Dict[str, Any]]] = None
    tendency: Optional[List[Dict[str, Any]]] = None
    
    # Motivational attributes: [{"value": "achievement", "level": "high", "evidences": ["2024-01-01|conv_123"]}]
    motivation_system: Optional[List[Dict[str, Any]]] = None
    fear_system: Optional[List[Dict[str, Any]]] = None
    value_system: Optional[List[Dict[str, Any]]] = None
    humor_use: Optional[List[Dict[str, Any]]] = None
    colloquialism: Optional[List[Dict[str, Any]]] = None

    group_importance_evidence: Optional[GroupImportanceEvidence] = None

    def __post_init__(self) -> None:
        """Ensure the memory type is set to PROFILE."""
        self.memory_type = MemoryType.PROFILE
        super().__post_init__()
    
    def to_dict(self) -> Dict[str, Any]:
        """重写 to_dict() 以包含 ProfileMemory 的所有字段"""
        # 先获取基类的字段
        base_dict = super().to_dict()
        
        # 添加 ProfileMemory 特有的字段
        base_dict.update({
            "user_name": self.user_name,
            "hard_skills": self.hard_skills,
            "soft_skills": self.soft_skills,
            "output_reasoning": self.output_reasoning,
            "way_of_decision_making": self.way_of_decision_making,
            "personality": self.personality,
            "projects_participated": [
                p.to_dict() if hasattr(p, 'to_dict') else p
                for p in (self.projects_participated or [])
            ] if self.projects_participated else None,
            "user_goal": self.user_goal,
            "work_responsibility": self.work_responsibility,
            "working_habit_preference": self.working_habit_preference,
            "interests": self.interests,
            "tendency": self.tendency,
            "motivation_system": self.motivation_system,
            "fear_system": self.fear_system,
            "value_system": self.value_system,
            "humor_use": self.humor_use,
            "colloquialism": self.colloquialism,
            "group_importance_evidence": (
                self.group_importance_evidence.to_dict() 
                if hasattr(self.group_importance_evidence, 'to_dict')
                else self.group_importance_evidence
            ) if self.group_importance_evidence else None,
        })
        
        return base_dict


@dataclass
class ProfileMemoryExtractRequest(MemoryExtractRequest):
    """
    Request payload used by ProfileMemoryExtractor.
    
    Profile 提取需要处理多个 MemCell (来自聚类),因此覆盖基类的单个 memcell,
    使用 memcell_list 和 user_id_list
    """
    # 覆盖基类字段,设置为 None (Profile 不使用单个 memcell)
    memcell: Optional[MemCell] = None
    
    # Profile 特有字段
    memcell_list: List[MemCell] = None
    user_id_list: Optional[List[str]] = None
    
    def __post_init__(self):
        # 确保 memcell_list 不为 None
        if self.memcell_list is None:
            self.memcell_list = []
