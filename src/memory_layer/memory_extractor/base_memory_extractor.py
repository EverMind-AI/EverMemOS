"""
Simple Memory Extraction Base Class for EverMemOS

This module provides a simple base class for extracting memories
from boundary detection results (BoundaryResult).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import uuid

from ..types import MemoryType, Memory, MemCell


@dataclass
class MemoryExtractRequest:
    """
    记忆提取请求基类
    """
    memcell: MemCell
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    participants: Optional[List[str]] = None

    old_memory_list: Optional[List[Memory]] = None

    user_organization: Optional[List] = None


class MemoryExtractor(ABC):
    """
    Simple abstract base class for memory extraction.

    This class provides a minimal interface for extracting memories
    from boundary detection results.
    """

    def __init__(self, memory_type: MemoryType):
        """
        Initialize the memory extractor.

        Args:
            memory_type: The type of memory this extractor generates
        """
        self.memory_type = memory_type

    @abstractmethod
    async def extract_memory(self, request: MemoryExtractRequest) -> Optional[Memory]:
        """
        Extract memory from a boundary detection result.

        Args:
            boundary_result: The boundary detection result to extract from
            user_id: User ID for the memory

        Returns:
            MemoryExtractionResult if extraction is successful, None otherwise
        """
        pass

    def _generate_memory_id(self) -> str:
        """Generate a unique memory ID."""
        return str(uuid.uuid4())

    def __str__(self) -> str:
        """String representation of the extractor."""
        return f"{self.__class__.__name__}(type={self.memory_type.value})"
