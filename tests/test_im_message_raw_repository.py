#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ImMessageRawRepository 的查询功能

测试内容包括:
1. 基于ID的查询操作
2. 基于房间ID的查询操作（支持消息类型和时间范围过滤）
3. 基于任务类型的查询操作
4. 各种查询参数组合验证
"""

import asyncio
from datetime import datetime, timedelta

from core.di import get_bean_by_type
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from infra_layer.adapters.out.persistence.repository.tanka.im_message_raw_repository import (
    ImMessageRawRepository,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


async def get_sample_data():
    """从数据库获取一些样本数据用于测试"""
    logger.info("获取样本数据...")

    repo = get_bean_by_type(ImMessageRawRepository)

    # 获取一些消息数据进行测试
    sample_messages = []

    # 尝试通过任务类型获取一些群聊消息
    group_messages = await repo.get_by_task_type(task_type=3, limit=5)  # 群聊
    if group_messages:
        sample_messages.extend(group_messages)
        logger.info("✅ 获取到 %d 条群聊消息", len(group_messages))

    # 尝试获取单聊消息
    private_messages = await repo.get_by_task_type(task_type=2, limit=3)  # 单聊
    if private_messages:
        sample_messages.extend(private_messages)
        logger.info("✅ 获取到 %d 条单聊消息", len(private_messages))

    # 获取bot消息
    bot_messages = await repo.get_by_task_type(task_type=4, limit=2)  # bot
    if bot_messages:
        sample_messages.extend(bot_messages)
        logger.info("✅ 获取到 %d 条bot消息", len(bot_messages))

    logger.info("✅ 总共获取到 %d 条样本消息", len(sample_messages))

    # 提取一些有用的测试数据
    room_ids = list(set([msg.roomId for msg in sample_messages if msg.roomId]))[:3]
    message_ids = [str(msg.id) for msg in sample_messages[:5]]
    msg_types = list(
        set([msg.msgType for msg in sample_messages if msg.msgType is not None])
    )[:3]

    logger.info("可用的房间ID: %s", room_ids)
    logger.info("可用的消息ID: %s", message_ids)
    logger.info("可用的消息类型: %s", msg_types)

    return {
        "sample_messages": sample_messages,
        "room_ids": room_ids,
        "message_ids": message_ids,
        "msg_types": msg_types,
    }


async def test_get_by_id():
    """测试根据ID获取消息"""
    logger.info("开始测试根据ID获取消息...")

    repo = get_bean_by_type(ImMessageRawRepository)
    sample_data = await get_sample_data()

    if not sample_data["message_ids"]:
        logger.warning("⚠️  没有可用的消息ID进行测试")
        return

    try:
        # 测试获取存在的消息
        for message_id in sample_data["message_ids"][:3]:
            message = await repo.get_by_id(message_id)
            if message:
                logger.info(
                    "✅ 成功获取消息: ID=%s, 内容=%s, 房间ID=%s",
                    message_id,
                    message.content[:50] if message.content else "无内容",
                    message.roomId,
                )
            else:
                logger.info("ℹ️  消息不存在: ID=%s", message_id)

        # 测试获取不存在的消息
        fake_id = "000000000000000000000000"  # 假的ObjectId
        fake_message = await repo.get_by_id(fake_id)
        assert fake_message is None, "不存在的消息应该返回None"
        logger.info("✅ 不存在的消息正确返回None")

    except Exception as e:
        logger.error("❌ 测试根据ID获取消息失败: %s", e)
        raise

    logger.info("✅ 根据ID获取消息测试完成")


async def test_get_by_room_id():
    """测试根据房间ID获取消息列表"""
    logger.info("开始测试根据房间ID获取消息列表...")

    repo = get_bean_by_type(ImMessageRawRepository)
    sample_data = await get_sample_data()

    if not sample_data["room_ids"]:
        logger.warning("⚠️  没有可用的房间ID进行测试")
        return

    try:
        room_id = sample_data["room_ids"][0]

        # 1. 测试基本查询（不带任何过滤）
        messages = await repo.get_by_room_id(room_id, limit=10)
        logger.info("✅ 房间 %s 基本查询: 找到 %d 条消息", room_id, len(messages))

        if messages:
            # 显示一些消息信息
            for i, msg in enumerate(messages[:3]):
                logger.info(
                    "  消息%d: 类型=%s, 时间=%s, 内容=%s",
                    i + 1,
                    msg.msgType,
                    to_iso_format(msg.createTime) if msg.createTime else "无时间",
                    msg.content[:30] if msg.content else "无内容",
                )

        # 2. 测试带消息类型过滤
        if sample_data["msg_types"]:
            msg_type = sample_data["msg_types"][0]
            filtered_messages = await repo.get_by_room_id(
                room_id, msg_type=msg_type, limit=5
            )
            logger.info(
                "✅ 房间 %s 按消息类型 %s 过滤: 找到 %d 条消息",
                room_id,
                msg_type,
                len(filtered_messages),
            )

        # 3. 测试时间范围过滤
        current_time = get_now_with_timezone()
        start_time = current_time - timedelta(days=30)  # 最近30天

        time_filtered = await repo.get_by_room_id(
            room_id, create_time_range=(start_time, current_time), limit=5
        )
        logger.info(
            "✅ 房间 %s 按时间范围过滤(最近30天): 找到 %d 条消息",
            room_id,
            len(time_filtered),
        )

        # 4. 测试组合过滤
        if sample_data["msg_types"]:
            msg_type = sample_data["msg_types"][0]
            combined_filtered = await repo.get_by_room_id(
                room_id,
                msg_type=msg_type,
                create_time_range=(start_time, current_time),
                limit=3,
            )
            logger.info(
                "✅ 房间 %s 组合过滤(类型+时间): 找到 %d 条消息",
                room_id,
                len(combined_filtered),
            )

        # 5. 测试分页
        page1 = await repo.get_by_room_id(room_id, limit=3, skip=0)
        page2 = await repo.get_by_room_id(room_id, limit=3, skip=3)

        logger.info("✅ 分页测试: 第1页 %d 条，第2页 %d 条", len(page1), len(page2))

        # 验证分页结果不重复（如果有足够数据）
        if len(page1) > 0 and len(page2) > 0:
            page1_ids = {str(msg.id) for msg in page1}
            page2_ids = {str(msg.id) for msg in page2}
            overlap = page1_ids & page2_ids
            if len(overlap) > 0:
                logger.warning("⚠️  分页结果有重复（可能因为排序不稳定）: %s", overlap)
            else:
                logger.info("✅ 分页结果无重复")

        # 6. 测试不存在的房间ID
        fake_room_id = "non_existent_room_12345"
        empty_result = await repo.get_by_room_id(fake_room_id, limit=5)
        assert len(empty_result) == 0, "不存在的房间ID应该返回空列表"
        logger.info("✅ 不存在的房间ID正确返回空列表")

    except Exception as e:
        logger.error("❌ 测试根据房间ID获取消息列表失败: %s", e)
        raise

    logger.info("✅ 根据房间ID获取消息列表测试完成")


async def test_get_by_task_type():
    """测试根据任务类型获取消息列表"""
    logger.info("开始测试根据任务类型获取消息列表...")

    repo = get_bean_by_type(ImMessageRawRepository)
    sample_data = await get_sample_data()

    try:
        # 测试各种任务类型
        task_types = [(2, "单聊"), (3, "群聊"), (4, "bot"), (9, "column")]

        for task_type, type_name in task_types:
            messages = await repo.get_by_task_type(task_type, limit=5)
            logger.info(
                "✅ 任务类型 %d (%s): 找到 %d 条消息",
                task_type,
                type_name,
                len(messages),
            )

            # 验证返回的消息确实是指定的任务类型
            for msg in messages:
                if msg.taskType is not None:
                    assert (
                        msg.taskType == task_type
                    ), f"消息任务类型不匹配: 期望{task_type}, 实际{msg.taskType}"

        # 测试结合房间ID的过滤
        if sample_data["room_ids"]:
            room_id = sample_data["room_ids"][0]

            # 获取该房间的群聊消息
            room_group_messages = await repo.get_by_task_type(
                3, room_id=room_id, limit=3
            )
            logger.info(
                "✅ 房间 %s 的群聊消息: 找到 %d 条", room_id, len(room_group_messages)
            )

            # 验证返回的消息确实属于指定房间和任务类型
            for msg in room_group_messages:
                if msg.roomId is not None:
                    assert (
                        msg.roomId == room_id
                    ), f"消息房间ID不匹配: 期望{room_id}, 实际{msg.roomId}"
                if msg.taskType is not None:
                    assert (
                        msg.taskType == 3
                    ), f"消息任务类型不匹配: 期望3, 实际{msg.taskType}"

        # 测试不存在的任务类型
        invalid_messages = await repo.get_by_task_type(999, limit=5)
        logger.info("✅ 不存在的任务类型 999: 找到 %d 条消息", len(invalid_messages))

    except Exception as e:
        logger.error("❌ 测试根据任务类型获取消息列表失败: %s", e)
        raise

    logger.info("✅ 根据任务类型获取消息列表测试完成")


async def test_time_range_filtering():
    """测试时间范围过滤功能的详细测试"""
    logger.info("开始测试时间范围过滤功能...")

    repo = get_bean_by_type(ImMessageRawRepository)
    sample_data = await get_sample_data()

    if not sample_data["room_ids"]:
        logger.warning("⚠️  没有可用的房间ID进行时间范围测试")
        return

    try:
        room_id = sample_data["room_ids"][0]
        current_time = get_now_with_timezone()

        # 测试不同的时间范围
        time_ranges = [
            (timedelta(days=1), "最近1天"),
            (timedelta(days=7), "最近7天"),
            (timedelta(days=30), "最近30天"),
            (timedelta(days=90), "最近90天"),
        ]

        for time_delta, description in time_ranges:
            start_time = current_time - time_delta

            messages = await repo.get_by_room_id(
                room_id, create_time_range=(start_time, current_time), limit=10
            )

            logger.info("✅ %s: 找到 %d 条消息", description, len(messages))

            # 验证返回的消息确实在指定时间范围内
            for msg in messages:
                if msg.createTime:
                    assert (
                        msg.createTime >= start_time
                    ), f"消息时间早于开始时间: {msg.createTime} < {start_time}"
                    assert (
                        msg.createTime <= current_time
                    ), f"消息时间晚于结束时间: {msg.createTime} > {current_time}"

        # 测试只有开始时间的情况
        start_only = current_time - timedelta(days=7)
        messages_start_only = await repo.get_by_room_id(
            room_id, create_time_range=(start_only, None), limit=5
        )
        logger.info(
            "✅ 只指定开始时间(7天前至今): 找到 %d 条消息", len(messages_start_only)
        )

        # 测试只有结束时间的情况
        end_only = current_time - timedelta(days=1)
        messages_end_only = await repo.get_by_room_id(
            room_id, create_time_range=(None, end_only), limit=5
        )
        logger.info(
            "✅ 只指定结束时间(1天前以前): 找到 %d 条消息", len(messages_end_only)
        )

    except Exception as e:
        logger.error("❌ 测试时间范围过滤功能失败: %s", e)
        raise

    logger.info("✅ 时间范围过滤功能测试完成")


async def test_edge_cases():
    """测试边界情况和异常处理"""
    logger.info("开始测试边界情况...")

    repo = get_bean_by_type(ImMessageRawRepository)

    try:
        # 1. 测试空字符串参数
        empty_room_messages = await repo.get_by_room_id("", limit=1)
        logger.info("✅ 空房间ID查询: 找到 %d 条消息", len(empty_room_messages))

        # 2. 测试极大的limit值
        large_limit_messages = await repo.get_by_room_id("test_room", limit=10000)
        logger.info("✅ 极大limit值查询: 找到 %d 条消息", len(large_limit_messages))

        # 3. 测试负数limit（应该被忽略或处理）
        try:
            negative_limit_messages = await repo.get_by_room_id("test_room", limit=-1)
            logger.info(
                "✅ 负数limit查询: 找到 %d 条消息", len(negative_limit_messages)
            )
        except ValueError as e:
            logger.info("ℹ️  负数limit查询异常（预期行为）: %s", str(e))
        except Exception as e:
            logger.info("ℹ️  负数limit查询其他异常: %s", str(e))

        # 4. 测试未来时间范围
        future_time = get_now_with_timezone() + timedelta(days=1)
        future_messages = await repo.get_by_room_id(
            "test_room",
            create_time_range=(future_time, future_time + timedelta(hours=1)),
            limit=5,
        )
        assert len(future_messages) == 0, "未来时间范围应该返回空列表"
        logger.info("✅ 未来时间范围查询正确返回空列表")

        # 5. 测试无效的时间范围（开始时间晚于结束时间）
        start_time = get_now_with_timezone()
        end_time = start_time - timedelta(hours=1)

        invalid_time_messages = await repo.get_by_room_id(
            "test_room", create_time_range=(start_time, end_time), limit=5
        )
        logger.info("✅ 无效时间范围查询: 找到 %d 条消息", len(invalid_time_messages))

    except Exception as e:
        logger.error("❌ 测试边界情况失败: %s", e)
        raise

    logger.info("✅ 边界情况测试完成")


async def test_performance_and_pagination():
    """测试性能和分页功能"""
    logger.info("开始测试性能和分页功能...")

    repo = get_bean_by_type(ImMessageRawRepository)
    sample_data = await get_sample_data()

    if not sample_data["room_ids"]:
        logger.warning("⚠️  没有可用的房间ID进行分页测试")
        return

    try:
        room_id = sample_data["room_ids"][0]

        # 测试大量数据的查询性能
        start_time = datetime.now()
        large_result = await repo.get_by_room_id(room_id, limit=100)
        end_time = datetime.now()

        query_time = (end_time - start_time).total_seconds()
        logger.info(
            "✅ 查询100条消息耗时: %.3f秒，返回%d条", query_time, len(large_result)
        )

        # 测试分页的一致性
        page_size = 10
        all_pages = []

        for page in range(3):  # 获取前3页
            page_messages = await repo.get_by_room_id(
                room_id, limit=page_size, skip=page * page_size
            )
            all_pages.extend(page_messages)
            logger.info("第%d页: %d条消息", page + 1, len(page_messages))

            if len(page_messages) < page_size:
                logger.info("已获取所有可用消息")
                break

        # 验证分页结果的唯一性
        all_ids = [str(msg.id) for msg in all_pages]
        unique_ids = set(all_ids)

        if len(all_ids) != len(unique_ids):
            duplicates = len(all_ids) - len(unique_ids)
            logger.warning("⚠️  分页结果中有 %d 条重复消息", duplicates)
        else:
            logger.info("✅ 分页结果无重复")

        # 验证排序的一致性（按createTime倒序）
        if len(all_pages) > 1:
            for i in range(len(all_pages) - 1):
                current_msg = all_pages[i]
                next_msg = all_pages[i + 1]

                if current_msg.createTime and next_msg.createTime:
                    assert (
                        current_msg.createTime >= next_msg.createTime
                    ), f"排序错误: {current_msg.createTime} < {next_msg.createTime}"

            logger.info("✅ 分页排序一致性验证通过")

    except Exception as e:
        logger.error("❌ 测试性能和分页功能失败: %s", e)
        raise

    logger.info("✅ 性能和分页功能测试完成")


async def run_all_tests():
    """运行所有查询测试"""
    logger.info("🚀 开始运行 ImMessage 查询接口测试...")

    try:
        await test_get_by_id()
        await test_get_by_room_id()
        await test_get_by_task_type()
        await test_time_range_filtering()
        await test_edge_cases()
        await test_performance_and_pagination()

        logger.info("✅ 所有查询接口测试完成")

    except Exception as e:
        logger.error("❌ 测试过程中出现错误: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
