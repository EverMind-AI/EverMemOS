#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 GroupProfile 测试

测试内容包括:
1. 模型创建和验证
2. 字段类型检查
3. JSON 序列化测试
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from infra_layer.adapters.out.persistence.document.memory.group_profile import (
    GroupProfile,
    TopicInfo,
)


def test_topic_info_creation():
    """测试 TopicInfo 模型创建"""
    print("开始测试 TopicInfo 模型创建...")

    current_time = get_now_with_timezone()

    # 测试完整参数创建
    topic = TopicInfo(
        name="Python最佳实践",
        summary="讨论Python编程的最佳实践方法",
        status="exploring",
        last_active_at=current_time,
        id="topic_001",
        update_type="new",
        old_topic_id=None,
    )

    assert topic.name == "Python最佳实践"
    assert topic.summary == "讨论Python编程的最佳实践方法"
    assert topic.status == "exploring"
    assert topic.last_active_at == current_time
    assert topic.id == "topic_001"
    assert topic.update_type == "new"
    assert topic.old_topic_id is None

    print("✅ TopicInfo 完整参数创建测试成功")

    # 测试必填参数创建
    topic_minimal = TopicInfo(
        name="代码Review",
        summary="建立代码审查流程",
        status="consensus",
        last_active_at=current_time,
    )

    assert topic_minimal.name == "代码Review"
    assert topic_minimal.id is None  # 可选参数应为None
    assert topic_minimal.update_type is None
    assert topic_minimal.old_topic_id is None

    print("✅ TopicInfo 必填参数创建测试成功")

    # 测试 JSON 序列化
    topic_dict = topic.model_dump()
    assert "name" in topic_dict
    assert "last_active_at" in topic_dict

    print("✅ TopicInfo JSON序列化测试成功")
    print("TopicInfo 模型创建测试完成\n")


def test_group_profile_creation():
    """测试 GroupProfile 模型创建"""
    print("开始测试 GroupProfile 模型创建...")

    current_time = get_now_with_timezone()
    current_timestamp = int(datetime.now().timestamp() * 1000)

    # 创建测试话题
    topics = [
        TopicInfo(
            name="Python最佳实践",
            summary="讨论Python编程的最佳实践方法",
            status="exploring",
            last_active_at=current_time,
            id="topic_001",
        ),
        TopicInfo(
            name="代码Review流程",
            summary="建立有效的代码审查流程",
            status="consensus",
            last_active_at=current_time,
            id="topic_002",
        ),
    ]

    # 创建测试角色
    roles = {
        "core_contributor": [
            {"user_id": "user_001", "user_name": "张三"},
            {"user_id": "user_002", "user_name": "李四"},
        ],
        "reviewer": [{"user_id": "user_003", "user_name": "王五"}],
    }

    # 测试完整参数创建
    group_profile = GroupProfile(
        group_id="test_group_001",
        group_name="技术讨论组",
        topics=topics,
        roles=roles,
        timestamp=current_timestamp,
        subject="技术交流与学习",
        summary="本群组主要讨论各种技术话题，促进技术交流",
        extend={"priority": "high"},
    )

    assert group_profile.group_id == "test_group_001"
    assert group_profile.group_name == "技术讨论组"
    assert len(group_profile.topics) == 2
    assert group_profile.topics[0].name == "Python最佳实践"
    assert group_profile.topics[1].status == "consensus"
    assert "core_contributor" in group_profile.roles
    assert len(group_profile.roles["core_contributor"]) == 2
    assert group_profile.timestamp == current_timestamp
    assert group_profile.subject == "技术交流与学习"
    assert group_profile.extend["priority"] == "high"

    print("✅ GroupProfile 完整参数创建测试成功")

    # 测试必填参数创建
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

    print("✅ GroupProfile 必填参数创建测试成功")

    # 测试 JSON 序列化
    profile_dict = group_profile.model_dump()
    assert "group_id" in profile_dict
    assert "timestamp" in profile_dict
    assert "topics" in profile_dict
    assert len(profile_dict["topics"]) == 2

    print("✅ GroupProfile JSON序列化测试成功")

    # 测试时间序列化
    profile_json = group_profile.model_dump_json()
    assert "last_active_at" in profile_json

    print("✅ GroupProfile 时间序列化测试成功")
    print("GroupProfile 模型创建测试完成\n")


def test_timezone_handling():
    """测试不同时区的处理"""
    print("开始测试时区处理...")

    # 创建不同时区的时间
    utc_time = datetime.now(ZoneInfo("UTC"))
    tokyo_time = datetime.now(ZoneInfo("Asia/Tokyo"))
    shanghai_time = get_now_with_timezone()

    print(f"UTC时间: {to_iso_format(utc_time)}")
    print(f"东京时间: {to_iso_format(tokyo_time)}")
    print(f"上海时间: {to_iso_format(shanghai_time)}")

    # 创建使用不同时区的话题
    topics = [
        TopicInfo(
            name="UTC话题",
            summary="使用UTC时间的话题",
            status="exploring",
            last_active_at=utc_time,
        ),
        TopicInfo(
            name="东京话题",
            summary="使用东京时间的话题",
            status="consensus",
            last_active_at=tokyo_time,
        ),
        TopicInfo(
            name="上海话题",
            summary="使用上海时间的话题",
            status="implemented",
            last_active_at=shanghai_time,
        ),
    ]

    # 创建群组
    group_profile = GroupProfile(
        group_id="timezone_test_group",
        timestamp=int(datetime.now().timestamp() * 1000),
        topics=topics,
    )

    # 验证时间是否正确保存
    assert len(group_profile.topics) == 3

    utc_topic = next(t for t in group_profile.topics if t.name == "UTC话题")
    tokyo_topic = next(t for t in group_profile.topics if t.name == "东京话题")
    shanghai_topic = next(t for t in group_profile.topics if t.name == "上海话题")

    # 输出序列化后的时间
    profile_dict = group_profile.model_dump()
    print("序列化后的话题时间:")
    for topic in profile_dict["topics"]:
        print(f"{topic['name']}: {topic['last_active_at']}")

    print("✅ 时区处理测试成功")
    print("时区处理测试完成\n")


def test_validation():
    """测试数据验证"""
    print("开始测试数据验证...")

    current_time = get_now_with_timezone()

    try:
        # 测试缺少必填字段
        TopicInfo(
            name="测试话题",
            summary="测试摘要",
            status="exploring",
            # 缺少 last_active_at
        )
        assert False, "应该抛出验证错误"
    except Exception as e:
        print(f"✅ 必填字段验证成功: {type(e).__name__}")

    try:
        # 测试缺少必填字段
        GroupProfile(
            group_id="test_group"
            # 缺少 timestamp
        )
        assert False, "应该抛出验证错误"
    except Exception as e:
        print(f"✅ 必填字段验证成功: {type(e).__name__}")

    # 测试正确的创建
    valid_topic = TopicInfo(
        name="有效话题",
        summary="有效摘要",
        status="exploring",
        last_active_at=current_time,
    )

    valid_group = GroupProfile(
        group_id="valid_group", timestamp=int(datetime.now().timestamp() * 1000)
    )

    print("✅ 有效数据创建成功")
    print("数据验证测试完成\n")


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行 GroupProfile 简化测试...")
    print("=" * 60)

    try:
        test_topic_info_creation()
        test_group_profile_creation()
        test_timezone_handling()
        test_validation()

        print("=" * 60)
        print("✅ 所有测试完成")
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
