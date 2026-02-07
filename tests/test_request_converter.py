"""Tests for request conversion utilities."""

from api_specs.memory_models import MemoryType
from api_specs.request_converter import (
    convert_dict_to_fetch_mem_request,
    convert_dict_to_retrieve_mem_request,
)


def test_convert_fetch_mem_request_strips_memory_type_whitespace():
    request = convert_dict_to_fetch_mem_request(
        {"user_id": "user_1", "memory_type": " episodic_memory "}
    )

    assert request.memory_type == MemoryType.EPISODIC_MEMORY


def test_convert_retrieve_mem_request_strips_memory_type_entries():
    request = convert_dict_to_retrieve_mem_request(
        {"user_id": "user_1", "memory_types": [" event_log ", "foresight"]}
    )

    assert request.memory_types == [MemoryType.EVENT_LOG, MemoryType.FORESIGHT]


def test_convert_retrieve_mem_request_strips_include_metadata_string():
    request = convert_dict_to_retrieve_mem_request(
        {"user_id": "user_1", "include_metadata": " true "}
    )

    assert request.include_metadata is True
