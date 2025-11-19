#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 MemCellRawRepository 的功能

测试内容包括:
1. 基于 event_id 的增删改查操作
2. 基于 user_id 的查询
3. 基于时间范围的查询（包括分段查询）
4. 基于 group_id 的查询
5. 基于参与者的查询
6. 基于关键词的查询
7. 批量删除操作
8. 统计和聚合查询
"""

import asyncio
from common_utils.datetime_utils import get_now_with_timezone
from datetime import timedelta, datetime
from bson import ObjectId
from pydantic import BaseModel, Field

from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell,
    DataTypeEnum,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


# ==================== 投影模型定义 ====================
class MemCellProjection(BaseModel):
    """
    MemCell 投影模型 - 用于测试字段投影功能
    只包含部分字段，排除了 original_data 等大字段
    """

    id: ObjectId = Field(alias="_id")
    user_id: str
    timestamp: datetime
    summary: str
    type: DataTypeEnum

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


async def test_basic_crud_operations():
    """测试基于 event_id 的基本增删改查操作"""
    logger.info("开始测试基于 event_id 的基本增删改查操作...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_001"

    try:
        # 先清理可能存在的测试数据
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 测试创建新 MemCell
        now = get_now_with_timezone()
        memcell = MemCell(
            user_id=user_id,
            timestamp=now,
            summary="这是一条测试记忆：讨论了项目的技术方案",
            type=DataTypeEnum.CONVERSATION,
            keywords=["技术方案", "项目讨论"],
            participants=["张三", "李四"],
        )

        created = await repo.append_memcell(memcell)
        assert created is not None
        assert created.user_id == user_id
        assert created.summary == "这是一条测试记忆：讨论了项目的技术方案"
        assert created.event_id is not None
        logger.info("✅ 测试创建新 MemCell 成功, event_id=%s", created.event_id)

        event_id = str(created.event_id)

        # 测试根据 event_id 查询
        queried = await repo.get_by_event_id(event_id)
        assert queried is not None
        assert queried.user_id == user_id
        assert str(queried.event_id) == event_id
        logger.info("✅ 测试根据 event_id 查询成功")

        # 测试更新 MemCell
        update_data = {
            "summary": "更新后的摘要：项目技术方案已确定",
            "keywords": ["技术方案", "项目讨论", "已确定"],
        }

        updated = await repo.update_by_event_id(event_id, update_data)
        assert updated is not None
        assert updated.summary == "更新后的摘要：项目技术方案已确定"
        assert len(updated.keywords) == 3
        logger.info("✅ 测试更新 MemCell 成功")

        # 测试删除 MemCell
        deleted = await repo.delete_by_event_id(event_id)
        assert deleted is True
        logger.info("✅ 测试删除 MemCell 成功")

        # 验证删除
        final_check = await repo.get_by_event_id(event_id)
        assert final_check is None, "记录应该已被删除"
        logger.info("✅ 验证删除成功")

    except Exception as e:
        logger.error("❌ 测试基本增删改查操作失败: %s", e)
        raise

    logger.info("✅ 基本增删改查操作测试完成")


async def test_find_by_user_id():
    """测试基于 user_id 的查询"""
    logger.info("开始测试基于 user_id 的查询...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_002"

    try:
        # 先清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建多条记录
        now = get_now_with_timezone()
        for i in range(5):
            memcell = MemCell(
                user_id=user_id,
                timestamp=now - timedelta(hours=i),
                summary=f"测试记忆 {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        logger.info("✅ 创建了 5 条测试记录")

        # 测试查询所有记录（降序）
        results = await repo.find_by_user_id(user_id, sort_desc=True)
        assert len(results) == 5
        assert results[0].summary == "测试记忆 1"  # 最新的
        logger.info("✅ 测试查询所有记录（降序）成功")

        # 测试查询所有记录（升序）
        results_asc = await repo.find_by_user_id(user_id, sort_desc=False)
        assert len(results_asc) == 5
        assert results_asc[0].summary == "测试记忆 5"  # 最早的
        logger.info("✅ 测试查询所有记录（升序）成功")

        # 测试限制数量
        limited_results = await repo.find_by_user_id(user_id, limit=2)
        assert len(limited_results) == 2
        logger.info("✅ 测试限制数量成功")

        # 测试跳过和限制
        skip_results = await repo.find_by_user_id(user_id, skip=2, limit=2)
        assert len(skip_results) == 2
        logger.info("✅ 测试跳过和限制成功")

        # 清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试基于 user_id 查询失败: %s", e)
        raise

    logger.info("✅ 基于 user_id 的查询测试完成")


async def test_find_by_time_range():
    """测试基于时间范围的查询（包括分段查询）"""
    logger.info("开始测试基于时间范围的查询...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_003"

    try:
        # 先清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建跨度较大的测试数据（10天）
        # 使用1990年的时间避免与现有数据冲突
        # 注意：必须使用带时区的时间，否则会与 MongoDB 存储的时区不匹配
        from common_utils.datetime_utils import get_timezone

        tz = get_timezone()
        start_time = datetime(1990, 1, 1, 0, 0, 0, tzinfo=tz)

        # 每天创建一条记录
        created_timestamps = []
        for i in range(10):
            ts = start_time + timedelta(days=i)
            created_timestamps.append(ts)
            memcell = MemCell(
                user_id=user_id,
                timestamp=ts,
                summary=f"第 {i+1} 天的记忆",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        logger.info("✅ 创建了 10 天的测试数据")
        logger.info(
            "   时间戳范围: %s 到 %s", created_timestamps[0], created_timestamps[-1]
        )

        # 测试小范围查询（3天，不触发分段）
        # 查询 day 0, 1, 2（共3条记录）
        small_start = start_time  # 1990-01-01 00:00:00
        small_end = start_time + timedelta(days=3)  # 1990-01-04 00:00:00（不包含）
        small_results = await repo.find_by_time_range(small_start, small_end)
        logger.info("   小范围查询返回了 %d 条记录（期望 3 条）", len(small_results))
        assert (
            len(small_results) == 3
        ), f"期望返回3条记录，实际返回 {len(small_results)} 条"
        logger.info("✅ 测试小范围查询（3天）成功，找到 %d 条记录", len(small_results))

        # 测试大范围查询（10天，触发分段查询）
        # 查询 day 0-9（共10条记录）
        # 最后一条记录是 1990-01-10 00:00:00，查询使用 $lt，所以结束时间必须 > 1990-01-10
        large_start = start_time  # 1990-01-01 00:00:00
        large_end = start_time + timedelta(
            days=10, seconds=1
        )  # 1990-01-11 00:00:01（确保包含 day 9）
        logger.info("   查询时间范围: %s 到 %s", large_start, large_end)
        large_results = await repo.find_by_time_range(large_start, large_end)
        logger.info("   大范围查询返回了 %d 条记录（期望 10 条）", len(large_results))

        # 打印返回的记录时间戳以便调试
        logger.info("   返回的记录详情:")
        for idx, mc in enumerate(large_results):
            logger.info("     [%d] %s - %s", idx, mc.timestamp, mc.summary)

        if len(large_results) != 10:
            logger.warning("   ⚠️ 记录数量不匹配！")
            logger.warning("   期望的时间戳:")
            for idx, ts in enumerate(created_timestamps):
                logger.warning("     [%d] %s", idx, ts)

            # 找出缺失的记录
            returned_timestamps = {mc.timestamp for mc in large_results}
            missing = [ts for ts in created_timestamps if ts not in returned_timestamps]
            if missing:
                logger.error("   ❌ 缺失的时间戳:")
                for ts in missing:
                    logger.error("     - %s", ts)

        assert (
            len(large_results) == 10
        ), f"期望返回10条记录，实际返回 {len(large_results)} 条"
        logger.info("✅ 测试大范围查询（10天）成功，找到 %d 条记录", len(large_results))

        # 测试降序查询
        desc_results = await repo.find_by_time_range(
            large_start, large_end, sort_desc=True
        )
        assert len(desc_results) == 10
        assert "第 10 天" in desc_results[0].summary  # 最新的在前
        logger.info("✅ 测试降序查询成功")

        # 测试升序查询
        asc_results = await repo.find_by_time_range(
            large_start, large_end, sort_desc=False
        )
        assert len(asc_results) == 10
        assert "第 1 天" in asc_results[0].summary  # 最早的在前
        logger.info("✅ 测试升序查询成功")

        # 测试分页
        page_results = await repo.find_by_time_range(large_start, large_end, limit=5)
        assert len(page_results) == 5
        logger.info("✅ 测试分页成功")

        # 清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试时间范围查询失败: %s", e)
        raise

    logger.info("✅ 时间范围查询测试完成")


async def test_find_by_user_and_time_range():
    """测试基于用户和时间范围的查询"""
    logger.info("开始测试基于用户和时间范围的查询...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_004"

    try:
        # 先清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建测试数据
        now = get_now_with_timezone()
        start_time = now - timedelta(days=5)

        for i in range(5):
            memcell = MemCell(
                user_id=user_id,
                timestamp=start_time + timedelta(days=i),
                summary=f"用户记忆 {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        logger.info("✅ 创建了 5 条测试数据")

        # 测试查询中间3天的数据
        query_start = start_time + timedelta(days=1)
        query_end = start_time + timedelta(days=4)
        results = await repo.find_by_user_and_time_range(
            user_id, query_start, query_end
        )

        assert len(results) == 3
        logger.info("✅ 测试用户和时间范围查询成功，找到 %d 条记录", len(results))

        # 清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试用户和时间范围查询失败: %s", e)
        raise

    logger.info("✅ 用户和时间范围查询测试完成")


async def test_find_by_group_id():
    """测试基于 group_id 的查询"""
    logger.info("开始测试基于 group_id 的查询...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_005"
    group_id = "test_group_001"

    try:
        # 先清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建群组记录
        now = get_now_with_timezone()
        for i in range(3):
            memcell = MemCell(
                user_id=user_id,
                group_id=group_id,
                timestamp=now - timedelta(hours=i),
                summary=f"群组记忆 {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        logger.info("✅ 创建了 3 条群组记录")

        # 测试查询
        results = await repo.find_by_group_id(group_id)
        assert len(results) == 3
        logger.info("✅ 测试根据 group_id 查询成功，找到 %d 条记录", len(results))

        # 清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试 group_id 查询失败: %s", e)
        raise

    logger.info("✅ group_id 查询测试完成")


async def test_find_by_participants():
    """测试基于参与者的查询"""
    logger.info("开始测试基于参与者的查询...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_006"

    try:
        # 先清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建测试数据
        now = get_now_with_timezone()

        # 记录1：张三、李四
        memcell1 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=1),
            summary="记录1：张三和李四的对话",
            participants=["张三", "李四"],
        )
        await repo.append_memcell(memcell1)

        # 记录2：张三、王五
        memcell2 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=2),
            summary="记录2：张三和王五的对话",
            participants=["张三", "王五"],
        )
        await repo.append_memcell(memcell2)

        # 记录3：李四、王五
        memcell3 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=3),
            summary="记录3：李四和王五的对话",
            participants=["李四", "王五"],
        )
        await repo.append_memcell(memcell3)

        logger.info("✅ 创建了 3 条测试记录")

        # 测试匹配任一参与者（包含"张三"）
        results_any = await repo.find_by_participants(["张三"], match_all=False)
        assert len(results_any) == 2
        logger.info("✅ 测试匹配任一参与者成功，找到 %d 条记录", len(results_any))

        # 测试匹配所有参与者（同时包含"张三"和"李四"）
        results_all = await repo.find_by_participants(["张三", "李四"], match_all=True)
        assert len(results_all) == 1
        logger.info("✅ 测试匹配所有参与者成功，找到 %d 条记录", len(results_all))

        # 清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试参与者查询失败: %s", e)
        raise

    logger.info("✅ 参与者查询测试完成")


async def test_search_by_keywords():
    """测试基于关键词的查询"""
    logger.info("开始测试基于关键词的查询...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_007"

    try:
        # 先清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建测试数据
        now = get_now_with_timezone()

        # 记录1：技术、Python
        memcell1 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=1),
            summary="记录1：Python技术讨论",
            keywords=["技术", "Python"],
        )
        await repo.append_memcell(memcell1)

        # 记录2：技术、Java
        memcell2 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=2),
            summary="记录2：Java技术讨论",
            keywords=["技术", "Java"],
        )
        await repo.append_memcell(memcell2)

        # 记录3：设计、架构
        memcell3 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=3),
            summary="记录3：架构设计讨论",
            keywords=["设计", "架构"],
        )
        await repo.append_memcell(memcell3)

        logger.info("✅ 创建了 3 条测试记录")

        # 测试匹配任一关键词（包含"技术"）
        results_any = await repo.search_by_keywords(["技术"], match_all=False)
        assert len(results_any) == 2
        logger.info("✅ 测试匹配任一关键词成功，找到 %d 条记录", len(results_any))

        # 测试匹配所有关键词（同时包含"技术"和"Python"）
        results_all = await repo.search_by_keywords(["技术", "Python"], match_all=True)
        assert len(results_all) == 1
        logger.info("✅ 测试匹配所有关键词成功，找到 %d 条记录", len(results_all))

        # 清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试关键词查询失败: %s", e)
        raise

    logger.info("✅ 关键词查询测试完成")


async def test_batch_delete_operations():
    """测试批量删除操作"""
    logger.info("开始测试批量删除操作...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_008"

    try:
        # 先清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建测试数据
        now = get_now_with_timezone()
        for i in range(10):
            memcell = MemCell(
                user_id=user_id,
                timestamp=now - timedelta(days=i),
                summary=f"测试记忆 {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        logger.info("✅ 创建了 10 条测试数据")

        # 测试删除时间范围内的记录（前5天）
        delete_start = now - timedelta(days=5)
        delete_end = now
        deleted_count = await repo.delete_by_time_range(
            delete_start, delete_end, user_id=user_id
        )
        assert deleted_count == 5
        logger.info("✅ 测试删除时间范围内的记录成功，删除了 %d 条", deleted_count)

        # 验证剩余记录
        remaining = await repo.find_by_user_id(user_id)
        assert len(remaining) == 5
        logger.info("✅ 验证剩余记录成功，还有 %d 条", len(remaining))

        # 测试删除用户所有记录
        total_deleted = await repo.delete_by_user_id(user_id)
        assert total_deleted == 5
        logger.info("✅ 测试删除用户所有记录成功，删除了 %d 条", total_deleted)

        # 验证全部删除
        final_check = await repo.find_by_user_id(user_id)
        assert len(final_check) == 0
        logger.info("✅ 验证全部删除成功")

    except Exception as e:
        logger.error("❌ 测试批量删除操作失败: %s", e)
        raise

    logger.info("✅ 批量删除操作测试完成")


async def test_statistics_and_aggregation():
    """测试统计和聚合查询"""
    logger.info("开始测试统计和聚合查询...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_009"

    try:
        # 先清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建不同类型的测试数据
        now = get_now_with_timezone()
        start_time = now - timedelta(days=7)

        # 创建6条对话记忆（注：原本是3条对话、2条邮件、1条文档，但现在只有CONVERSATION类型）
        for i in range(3):
            memcell = MemCell(
                user_id=user_id,
                timestamp=start_time + timedelta(days=i),
                summary=f"对话记忆 {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        for i in range(2):
            memcell = MemCell(
                user_id=user_id,
                timestamp=start_time + timedelta(days=i + 3),
                summary=f"邮件记忆 {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        memcell = MemCell(
            user_id=user_id,
            timestamp=start_time + timedelta(days=5),
            summary="文档记忆",
            type=DataTypeEnum.CONVERSATION,
        )
        await repo.append_memcell(memcell)

        logger.info("✅ 创建了 6 条测试数据（全部为CONVERSATION类型）")

        # 测试统计用户总记录数
        total_count = await repo.count_by_user_id(user_id)
        assert total_count == 6
        logger.info("✅ 测试统计用户总记录数成功，共 %d 条", total_count)

        # 测试统计时间范围内的记录数
        range_start = start_time
        range_end = start_time + timedelta(days=4)
        range_count = await repo.count_by_time_range(
            range_start, range_end, user_id=user_id
        )
        assert range_count == 4  # 前4天的记录（3条对话记忆 + 1条邮件记忆）
        logger.info("✅ 测试统计时间范围内的记录数成功，共 %d 条", range_count)

        # 测试获取用户最新记录
        latest = await repo.get_latest_by_user(user_id, limit=3)
        assert len(latest) == 3
        assert latest[0].summary == "文档记忆"  # 最新的
        logger.info("✅ 测试获取用户最新记录成功")

        # 测试获取用户活动摘要
        summary = await repo.get_user_activity_summary(user_id, start_time, now)
        assert summary["total_count"] == 6
        assert summary["user_id"] == user_id
        assert DataTypeEnum.CONVERSATION.value in summary["type_distribution"]
        assert (
            summary["type_distribution"][DataTypeEnum.CONVERSATION.value] == 6
        )  # 所有记录都是CONVERSATION类型
        logger.info("✅ 测试获取用户活动摘要成功")
        logger.info("   活动摘要: %s", summary)

        # 清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试统计和聚合查询失败: %s", e)
        raise

    logger.info("✅ 统计和聚合查询测试完成")


async def test_get_by_event_ids():
    """测试根据 event_ids 批量查询"""
    logger.info("开始测试根据 event_ids 批量查询...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_010"

    try:
        # 先清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建测试数据
        now = get_now_with_timezone()
        created_memcells = []

        for i in range(5):
            memcell = MemCell(
                user_id=user_id,
                timestamp=now - timedelta(hours=i),
                summary=f"测试记忆 {i+1}",
                episode=f"这是测试记忆 {i+1} 的详细内容",
                type=DataTypeEnum.CONVERSATION,
                keywords=[f"关键词{i+1}", "测试"],
            )
            created = await repo.append_memcell(memcell)
            created_memcells.append(created)

        logger.info("✅ 创建了 5 条测试数据")

        # 准备 event_ids
        event_ids = [str(mc.event_id) for mc in created_memcells[:3]]
        logger.info("   准备查询的 event_ids: %s", event_ids)

        # 测试1: 批量查询（不带投影）
        results = await repo.get_by_event_ids(event_ids)
        assert isinstance(results, dict), "返回结果应该是字典"
        assert len(results) == 3, f"应该返回3条记录，实际返回 {len(results)} 条"

        # 验证返回的是字典，key 是 event_id
        for event_id in event_ids:
            assert event_id in results, f"event_id {event_id} 应该在结果中"
            memcell = results[event_id]
            assert memcell.user_id == user_id
            assert memcell.episode is not None

        logger.info("✅ 测试批量查询（不带投影）成功，返回 %d 条记录", len(results))

        # 测试2: 批量查询（带字段投影）
        # 使用 Pydantic 投影模型，只返回指定的字段，排除 original_data 等大字段
        results_with_projection = await repo.get_by_event_ids(
            event_ids, projection_model=MemCellProjection
        )

        assert isinstance(results_with_projection, dict), "返回结果应该是字典"
        assert (
            len(results_with_projection) == 3
        ), f"应该返回3条记录，实际返回 {len(results_with_projection)} 条"

        # 验证投影效果：返回的应该是 MemCellProjection 实例
        for event_id, memcell_projection in results_with_projection.items():
            assert isinstance(
                memcell_projection, MemCellProjection
            ), "返回的应该是 MemCellProjection 实例"
            assert memcell_projection.summary is not None, "summary 字段应该存在"
            assert memcell_projection.timestamp is not None, "timestamp 字段应该存在"
            assert memcell_projection.type is not None, "type 字段应该存在"
            assert memcell_projection.user_id == user_id, "user_id 应该匹配"
            # 验证投影模型中没有定义的字段不会被包含
            assert not hasattr(
                memcell_projection, 'original_data'
            ), "original_data 字段不应该存在"
            assert not hasattr(memcell_projection, 'episode'), "episode 字段不应该存在"

        logger.info(
            "✅ 测试批量查询（带字段投影）成功，返回 %d 条记录",
            len(results_with_projection),
        )

        # 测试3: 查询部分有效的 event_ids（包含一个无效的）
        mixed_event_ids = event_ids[:2] + ["invalid_id_123", "507f1f77bcf86cd799439011"]
        results_mixed = await repo.get_by_event_ids(mixed_event_ids)

        # 应该只返回有效的2条记录
        assert (
            len(results_mixed) == 2
        ), f"应该返回2条记录，实际返回 {len(results_mixed)} 条"
        assert event_ids[0] in results_mixed
        assert event_ids[1] in results_mixed
        assert "invalid_id_123" not in results_mixed
        assert "507f1f77bcf86cd799439011" not in results_mixed

        logger.info(
            "✅ 测试查询部分有效的 event_ids 成功，返回 %d 条记录", len(results_mixed)
        )

        # 测试4: 空列表输入
        results_empty = await repo.get_by_event_ids([])
        assert isinstance(results_empty, dict), "返回结果应该是字典"
        assert len(results_empty) == 0, "空列表应该返回空字典"
        logger.info("✅ 测试空列表输入成功")

        # 测试5: 查询不存在的 event_ids
        non_existent_ids = ["507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012"]
        results_non_existent = await repo.get_by_event_ids(non_existent_ids)
        assert isinstance(results_non_existent, dict), "返回结果应该是字典"
        assert len(results_non_existent) == 0, "不存在的 event_ids 应该返回空字典"
        logger.info("✅ 测试查询不存在的 event_ids 成功")

        # 测试6: 验证返回的数据完整性
        first_event_id = event_ids[0]
        first_memcell = results[first_event_id]
        original_memcell = created_memcells[0]

        assert str(first_memcell.event_id) == str(original_memcell.event_id)
        assert first_memcell.summary == original_memcell.summary
        assert first_memcell.user_id == original_memcell.user_id
        logger.info("✅ 验证返回数据完整性成功")

        # 清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试根据 event_ids 批量查询失败: %s", e)
        import traceback

        logger.error("详细错误: %s", traceback.format_exc())
        raise

    logger.info("✅ 根据 event_ids 批量查询测试完成")


async def run_all_tests():
    """运行所有测试"""
    logger.info("🚀 开始运行 MemCellRawRepository 所有测试...")

    try:
        await test_basic_crud_operations()
        await test_find_by_user_id()
        await test_find_by_time_range()
        await test_find_by_user_and_time_range()
        await test_find_by_group_id()
        await test_find_by_participants()
        await test_search_by_keywords()
        await test_batch_delete_operations()
        await test_statistics_and_aggregation()
        await test_get_by_event_ids()
        logger.info("✅✅✅ 所有测试完成！")
    except Exception as e:
        logger.error("❌ 测试过程中出现错误: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
