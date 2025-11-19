#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ConversationStatusRawRepository 的功能

测试内容包括:
1. 基于group_id的查询和更新操作
2. 统计方法
"""

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from core.di import get_bean_by_type
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
    ConversationStatusRawRepository,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


def compare_datetime(dt1: datetime, dt2: datetime) -> bool:
    """比较两个datetime对象，只比较到秒级精度"""
    return dt1.replace(microsecond=0) == dt2.replace(microsecond=0)


async def test_group_operations():
    """测试群组相关操作"""
    logger.info("开始测试群组相关操作...")

    repo = get_bean_by_type(ConversationStatusRawRepository)
    group_id = "test_group_001"
    current_time = get_now_with_timezone()

    try:
        # 测试 upsert (创建新记录)
        update_data = {
            "old_msg_start_time": current_time,
            "new_msg_start_time": current_time,
            "last_memcell_time": current_time,
        }

        result = await repo.upsert_by_group_id(group_id, update_data)
        assert result is not None
        assert result.group_id == group_id
        logger.info("✅ 测试upsert创建新记录成功")

        # 测试根据group_id查询
        queried = await repo.get_by_group_id(group_id)
        assert queried is not None
        assert queried.group_id == group_id
        assert compare_datetime(queried.old_msg_start_time, current_time)
        assert compare_datetime(queried.new_msg_start_time, current_time)
        logger.info("✅ 测试根据group_id查询成功")

        # 测试 upsert (更新现有记录)
        new_time = get_now_with_timezone()
        update_data = {"old_msg_start_time": new_time, "new_msg_start_time": new_time}

        updated = await repo.upsert_by_group_id(group_id, update_data)
        assert updated is not None
        assert compare_datetime(updated.old_msg_start_time, new_time)
        assert compare_datetime(updated.new_msg_start_time, new_time)
        assert compare_datetime(
            updated.last_memcell_time, current_time
        )  # 未更新的字段应保持原值
        logger.info("✅ 测试upsert更新现有记录成功")

        # 再次查询验证更新
        queried_again = await repo.get_by_group_id(group_id)
        assert queried_again is not None
        assert compare_datetime(queried_again.old_msg_start_time, new_time)
        assert compare_datetime(queried_again.new_msg_start_time, new_time)
        assert compare_datetime(queried_again.last_memcell_time, current_time)
        logger.info("✅ 验证更新结果成功")

        # 清理测试数据
        await queried_again.delete()
        logger.info("✅ 清理测试数据成功")

        # 验证删除
        final_check = await repo.get_by_group_id(group_id)
        assert final_check is None, "记录应该已被删除"
        logger.info("✅ 验证删除成功")

    except Exception as e:
        logger.error("❌ 测试群组相关操作失败: %s", e)
        raise

    logger.info("✅ 群组相关操作测试完成")


async def test_statistics():
    """测试统计方法"""
    logger.info("开始测试统计方法...")

    repo = get_bean_by_type(ConversationStatusRawRepository)
    base_group_id = "test_group_stats"
    current_time = get_now_with_timezone()

    try:
        # 创建多条测试记录
        test_records = []
        for i in range(3):
            group_id = f"{base_group_id}_{i}"
            result = await repo.upsert_by_group_id(
                group_id=group_id,
                update_data={
                    "old_msg_start_time": current_time,
                    "new_msg_start_time": current_time,
                    "last_memcell_time": current_time,
                },
            )
            test_records.append(result)
        logger.info("✅ 创建测试记录成功")

        # 测试群组记录计数
        count = await repo.count_by_group_id(
            f"{base_group_id}_0"
        )  # 测试第一个群组的计数
        assert count == 1, "应该有1条记录，实际有%d条" % count
        logger.info("✅ 测试群组记录计数成功")

        # 测试总记录计数
        total = await repo.count_all()
        assert total >= 3, "总记录数应该至少为3，实际为%d" % total
        logger.info("✅ 测试总记录计数成功")

        # 清理测试数据
        for record in test_records:
            await record.delete()
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试统计方法失败: %s", e)
        raise

    logger.info("✅ 统计方法测试完成")


async def test_timezone_handling():
    """测试不同时区的datetime处理"""
    logger.info("开始测试时区处理...")

    repo = get_bean_by_type(ConversationStatusRawRepository)
    group_id = "test_timezone_001"

    try:
        # 创建UTC时间
        utc_time = datetime.now(ZoneInfo("UTC"))
        # 创建东京时间
        tokyo_time = datetime.now(ZoneInfo("Asia/Tokyo"))

        shanghai_time = datetime.now()

        # 使用两个不同时区的时间创建记录
        update_data = {
            "old_msg_start_time": utc_time,
            "new_msg_start_time": tokyo_time,
            "last_memcell_time": shanghai_time,  # 使用默认的上海时区
        }

        # 记录原始时间的ISO格式，用于比较
        logger.info("原始UTC时间: %s", to_iso_format(utc_time))
        logger.info("原始东京时间: %s", to_iso_format(tokyo_time))
        logger.info("原始上海时间: %s", to_iso_format(shanghai_time))

        # 插入数据库
        result = await repo.upsert_by_group_id(group_id, update_data)
        assert result is not None
        logger.info("✅ 插入不同时区的时间记录成功")

        # 从数据库获取并验证
        queried = await repo.get_by_group_id(group_id)
        assert queried is not None

        # 输出获取的时间信息
        logger.info("从数据库获取的时间:")
        logger.info(
            "old_msg_start_time (原UTC): %s", to_iso_format(queried.old_msg_start_time)
        )
        logger.info(
            "new_msg_start_time (原Tokyo): %s",
            to_iso_format(queried.new_msg_start_time),
        )
        logger.info(
            "last_memcell_time (原Shanghai): %s",
            to_iso_format(queried.last_memcell_time),
        )

        # 验证时间是否正确（转换到同一时区后应该相等）
        assert queried.old_msg_start_time.astimezone(ZoneInfo("UTC")).replace(
            microsecond=0
        ) == utc_time.replace(microsecond=0)
        assert queried.new_msg_start_time.astimezone(ZoneInfo("Asia/Tokyo")).replace(
            microsecond=0
        ) == tokyo_time.replace(microsecond=0)
        assert queried.last_memcell_time.replace(tzinfo=None).replace(
            microsecond=0
        ) == shanghai_time.replace(microsecond=0)
        logger.info("✅ 时区验证成功")

        # 清理测试数据
        # await queried.delete()
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试时区处理失败: %s", e)
        raise

    logger.info("✅ 时区处理测试完成")


async def run_all_tests():
    """运行所有测试"""
    logger.info("🚀 开始运行所有测试...")

    try:
        await test_group_operations()
        await test_statistics()
        await test_timezone_handling()
        logger.info("✅ 所有测试完成")
    except Exception as e:
        logger.error("❌ 测试过程中出现错误: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
