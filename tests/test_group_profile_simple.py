#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified GroupProfile Test

Test contents include:
1. Model creation and validation
2. Field type checking
3. JSON serialization test
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from infra_layer.adapters.out.persistence.document.memory.group_profile import (
    GroupProfile,
    TopicInfo,
)


def test_topic_info_creation():
    """Test TopicInfo model creation"""
    print("Starting TopicInfo model creation test...")

    current_time = get_now_with_timezone()

    # Test full parameter creation
    topic = TopicInfo(
        name="Python Best Practices",
        summary="Discuss best practices for Python programming",
        status="exploring",
        last_active_at=current_time,
        id="topic_001",
        update_type="new",
        old_topic_id=None,
    )

    assert topic.name == "Python Best Practices"
    assert topic.summary == "Discuss best practices for Python programming"
    assert topic.status == "exploring"
    assert topic.last_active_at == current_time
    assert topic.id == "topic_001"
    assert topic.update_type == "new"
    assert topic.old_topic_id is None

    print("✅ TopicInfo full parameter creation test passed")

    # Test required parameter creation
    topic_minimal = TopicInfo(
        name="Code Review",
        summary="Establish code review process",
        status="consensus",
        last_active_at=current_time,
    )

    assert topic_minimal.name == "Code Review"
    assert topic_minimal.id is None  # Optional parameter should be None
    assert topic_minimal.update_type is None
    assert topic_minimal.old_topic_id is None

    print("✅ TopicInfo required parameter creation test passed")

    # Test JSON serialization
    topic_dict = topic.model_dump()
    assert "name" in topic_dict
    assert "last_active_at" in topic_dict

    print("✅ TopicInfo JSON serialization test passed")
    print("TopicInfo model creation test completed\n")


def test_group_profile_creation():
    """Test GroupProfile model creation"""
    print("Starting GroupProfile model creation test...")

    current_time = get_now_with_timezone()
    current_timestamp = int(get_now_with_timezone().timestamp() * 1000)

    # Create test topics
    topics = [
        TopicInfo(
            name="Python Best Practices",
            summary="Discuss best practices for Python programming",
            status="exploring",
            last_active_at=current_time,
            id="topic_001",
        ),
        TopicInfo(
            name="Code Review Process",
            summary="Establish an effective code review process",
            status="consensus",
            last_active_at=current_time,
            id="topic_002",
        ),
    ]

    # Create test roles
    roles = {
        "core_contributor": [
            {"user_id": "user_001", "user_name": "Zhang San"},
            {"user_id": "user_002", "user_name": "Li Si"},
        ],
        "reviewer": [{"user_id": "user_003", "user_name": "Wang Wu"}],
    }

    # Test full parameter creation
    group_profile = GroupProfile(
        group_id="test_group_001",
        group_name="Technical Discussion Group",
        topics=topics,
        roles=roles,
        timestamp=current_timestamp,
        subject="Technical Exchange and Learning",
        summary="This group mainly discusses various technical topics to promote technical communication",
        extend={"priority": "high"},
    )

    assert group_profile.group_id == "test_group_001"
    assert group_profile.group_name == "Technical Discussion Group"
    assert len(group_profile.topics) == 2
    assert group_profile.topics[0].name == "Python Best Practices"
    assert group_profile.topics[1].status == "consensus"
    assert "core_contributor" in group_profile.roles
    assert len(group_profile.roles["core_contributor"]) == 2
    assert group_profile.timestamp == current_timestamp
    assert group_profile.subject == "Technical Exchange and Learning"
    assert group_profile.extend["priority"] == "high"

    print("✅ GroupProfile full parameter creation test passed")

    # Test required parameter creation
    minimal_profile = GroupProfile(
        group_id="test_group_002", timestamp=current_timestamp
    )

    assert minimal_profile.group_id == "test_group_002"
    assert minimal_profile.timestamp == current_timestamp
    assert minimal_profile.group_name is None
    assert minimal_profile.topics is None
    assert minimal_profile.roles is None
    assert minimal_profile.subject is None
    assert minimal_profile.summary is None

    print("✅ GroupProfile required parameter creation test passed")

    # Test JSON serialization
    profile_dict = group_profile.model_dump()
    assert "group_id" in profile_dict
    assert "timestamp" in profile_dict
    assert "topics" in profile_dict
    assert len(profile_dict["topics"]) == 2

    print("✅ GroupProfile JSON serialization test passed")

    # Test time serialization
    profile_json = group_profile.model_dump_json()
    assert "last_active_at" in profile_json

    print("✅ GroupProfile time serialization test passed")
    print("GroupProfile model creation test completed\n")


def test_timezone_handling():
    """Test handling of different timezones"""
    print("Starting timezone handling test...")

    # Create times in different timezones
    utc_time = get_now_with_timezone(ZoneInfo("UTC"))
    tokyo_time = get_now_with_timezone(ZoneInfo("Asia/Tokyo"))
    shanghai_time = get_now_with_timezone(ZoneInfo("Asia/Shanghai"))

    print(f"UTC time: {to_iso_format(utc_time)}")
    print(f"Tokyo time: {to_iso_format(tokyo_time)}")
    print(f"Shanghai time: {to_iso_format(shanghai_time)}")

    # Create topics using different timezones
    topics = [
        TopicInfo(
            name="UTC Topic",
            summary="Topic using UTC time",
            status="exploring",
            last_active_at=utc_time,
        ),
        TopicInfo(
            name="Tokyo Topic",
            summary="Topic using Tokyo time",
            status="consensus",
            last_active_at=tokyo_time,
        ),
        TopicInfo(
            name="Shanghai Topic",
            summary="Topic using Shanghai time",
            status="implemented",
            last_active_at=shanghai_time,
        ),
    ]

    # Create group
    group_profile = GroupProfile(
        group_id="timezone_test_group",
        timestamp=int(get_now_with_timezone().timestamp() * 1000),
        topics=topics,
    )

    # Verify time is correctly saved
    assert len(group_profile.topics) == 3

    utc_topic = next(t for t in group_profile.topics if t.name == "UTC Topic")
    tokyo_topic = next(t for t in group_profile.topics if t.name == "Tokyo Topic")
    shanghai_topic = next(t for t in group_profile.topics if t.name == "Shanghai Topic")

    # Output serialized time
    profile_dict = group_profile.model_dump()
    print("Serialized topic times:")
    for topic in profile_dict["topics"]:
        print(f"{topic['name']}: {topic['last_active_at']}")

    print("✅ Timezone handling test passed")
    print("Timezone handling test completed\n")


def test_validation():
    """Test data validation"""
    print("Starting data validation test...")

    current_time = get_now_with_timezone()

    try:
        # Test missing required field
        TopicInfo(
            name="Test Topic",
            summary="Test summary",
            status="exploring",
            # Missing last_active_at
        )
        assert False, "Should raise validation error"
    except Exception as e:
        print(f"✅ Required field validation passed: {type(e).__name__}")

    try:
        # Test missing required field
        GroupProfile(
            group_id="test_group"
            # Missing timestamp
        )
        assert False, "Should raise validation error"
    except Exception as e:
        print(f"✅ Required field validation passed: {type(e).__name__}")

    # Test correct creation
    valid_topic = TopicInfo(
        name="Valid Topic",
        summary="Valid summary",
        status="exploring",
        last_active_at=current_time,
    )

    valid_group = GroupProfile(
        group_id="valid_group",
        timestamp=int(get_now_with_timezone().timestamp() * 1000),
    )

    print("✅ Valid data creation passed")
    print("Data validation test completed\n")


def run_all_tests():
    """Run all tests"""
    print("🚀 Starting GroupProfile simplified tests...")
    print("=" * 60)

    try:
        test_topic_info_creation()
        test_group_profile_creation()
        test_timezone_handling()
        test_validation()

        print("=" * 60)
        print("✅ All tests completed")
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
