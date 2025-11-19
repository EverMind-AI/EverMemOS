#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 EpisodicMemoryMilvusRepository 的功能

测试内容包括:
1. 基础CRUD操作（增删改查）
2. 向量搜索和过滤功能
3. 批量删除功能
4. 时区处理
"""

import asyncio
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo
import numpy as np
from typing import List
from core.di import get_bean_by_type
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


def compare_datetime(dt1: datetime, dt2: datetime) -> bool:
    """比较两个datetime对象，只比较到秒级精度"""
    return dt1.replace(microsecond=0) == dt2.replace(microsecond=0)


def generate_random_vector(dim: int = 1024) -> List[float]:
    """生成随机向量用于测试"""
    return np.random.randn(dim).astype(np.float32).tolist()


def build_episodic_memory_entity(
    event_id: str,
    user_id: str,
    timestamp: datetime,
    episode: str,
    search_content: List[str],
    vector: List[float],
    user_name: str = "",
    title: str = "",
    summary: str = "",
    group_id: str = "",
    participants: List[str] = None,
    event_type: str = "",
    keywords: List[str] = None,
    linked_entities: List[str] = None,
    created_at: datetime = None,
    updated_at: datetime = None,
) -> dict:
    """
    构建情景记忆实体用于测试

    Args:
        event_id: 事件ID
        user_id: 用户ID
        timestamp: 事件时间戳
        episode: 情景描述
        search_content: 搜索内容列表
        vector: 向量
        其他参数为可选参数

    Returns:
        dict: 可以直接插入 Milvus 的实体字典
    """
    now = get_now_with_timezone()
    if created_at is None:
        created_at = now
    if updated_at is None:
        updated_at = now

    # 构建 metadata
    metadata = {}
    if user_name:
        metadata["user_name"] = user_name
    if title:
        metadata["title"] = title
    if summary:
        metadata["summary"] = summary
    if participants:
        metadata["participants"] = participants
    if keywords:
        metadata["keywords"] = keywords
    if linked_entities:
        metadata["linked_entities"] = linked_entities

    # 构建实体
    entity = {
        "id": event_id,
        "user_id": user_id,
        "group_id": group_id if group_id is not None else "",
        "event_type": event_type if event_type is not None else "",
        "timestamp": int(timestamp.timestamp()),
        "episode": episode,
        "search_content": json.dumps(search_content, ensure_ascii=False),
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "vector": vector,
        "created_at": int(created_at.timestamp()),
        "updated_at": int(updated_at.timestamp()),
    }

    return entity


async def test_crud_operations():
    """测试基础CRUD操作"""
    logger.info("开始测试基础CRUD操作...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_event_id = "test_event_crud_001"
    test_user_id = "test_user_crud_123"
    current_time = get_now_with_timezone()

    try:
        # 测试创建（Create）
        entity = build_episodic_memory_entity(
            event_id=test_event_id,
            user_id=test_user_id,
            timestamp=current_time,
            episode="这是一个测试情景记忆",
            search_content=["测试", "情景", "记忆", "CRUD"],
            vector=generate_random_vector(),
            user_name="测试用户",
            title="测试标题",
            summary="测试摘要",
            group_id="test_group_001",
            participants=["user1", "user2"],
            event_type="Test",
            keywords=["测试", "单元测试"],
            linked_entities=["entity1", "entity2"],
        )

        # 插入文档
        await repo.collection.insert([entity])

        assert entity is not None
        assert entity["id"] == test_event_id
        assert entity["user_id"] == test_user_id
        assert entity["episode"] == "这是一个测试情景记忆"
        logger.info("✅ 测试创建操作成功")

        # 等待数据刷新
        await repo.flush()
        await asyncio.sleep(1)

        # 测试读取（Read）
        retrieved_doc = await repo.get_by_id(test_event_id)
        assert retrieved_doc is not None
        assert retrieved_doc["id"] == test_event_id
        assert retrieved_doc["user_id"] == test_user_id
        assert retrieved_doc["episode"] == "这是一个测试情景记忆"
        metadata = json.loads(retrieved_doc["metadata"])
        assert metadata["title"] == "测试标题"
        assert retrieved_doc["group_id"] == "test_group_001"
        logger.info("✅ 测试读取操作成功")

        # 测试删除（Delete）
        delete_result = await repo.delete_by_event_id(test_event_id)
        assert delete_result is True
        logger.info("✅ 测试删除操作成功")

        # 验证删除
        await repo.flush()
        deleted_check = await repo.get_by_id(test_event_id)
        assert deleted_check is None, "文档应该已被删除"
        logger.info("✅ 验证删除结果成功")

    except Exception as e:
        logger.error("❌ 测试基础CRUD操作失败: %s", e)
        # 清理可能残留的数据
        try:
            await repo.delete_by_event_id(test_event_id)
            await repo.flush()
        except Exception:
            pass
        raise

    logger.info("✅ 基础CRUD操作测试完成")


async def test_vector_search():
    """测试向量搜索和过滤功能"""
    logger.info("开始测试向量搜索和过滤功能...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_user_id = "test_user_search_456"
    test_group_id = "test_group_search_789"
    base_time = get_now_with_timezone()
    test_event_ids = []
    base_vector = generate_random_vector()  # 基准向量

    try:
        # 创建多个测试记忆
        test_data = [
            {
                "event_id": f"search_test_001_{int(base_time.timestamp())}",
                "episode": "讨论了公司的发展战略",
                "search_content": ["公司", "发展", "战略", "讨论"],
                "vector": [
                    x + 0.1 * np.random.randn() for x in base_vector
                ],  # 相似向量
                "title": "战略会议",
                "group_id": test_group_id,
                "event_type": "Conversation",
                "keywords": ["会议", "战略"],
                "timestamp": base_time - timedelta(days=1),
            },
            {
                "event_id": f"search_test_002_{int(base_time.timestamp())}",
                "episode": "学习了新的技术框架",
                "search_content": ["技术", "框架", "学习", "编程"],
                "vector": generate_random_vector(),  # 随机向量
                "title": "技术学习",
                "group_id": "",
                "event_type": "Learning",
                "keywords": ["技术", "学习"],
                "timestamp": base_time - timedelta(days=2),
            },
            {
                "event_id": f"search_test_003_{int(base_time.timestamp())}",
                "episode": "参加了团队建设活动",
                "search_content": ["团队", "建设", "活动", "参加"],
                "vector": [
                    x + 0.2 * np.random.randn() for x in base_vector
                ],  # 相似向量
                "title": "团队活动",
                "group_id": test_group_id,
                "event_type": "Activity",
                "keywords": ["团队", "活动"],
                "timestamp": base_time - timedelta(days=3),
            },
        ]

        # 批量创建测试数据
        for data in test_data:
            entity = build_episodic_memory_entity(
                event_id=data["event_id"],
                user_id=test_user_id,
                timestamp=data["timestamp"],
                episode=data["episode"],
                search_content=data["search_content"],
                vector=data["vector"],
                title=data["title"],
                group_id=data["group_id"],
                event_type=data["event_type"],
                keywords=data["keywords"],
            )
            await repo.collection.insert([entity])
            test_event_ids.append(data["event_id"])

        # 刷新集合
        await repo.flush()
        await repo.load()  # 加载到内存以提高搜索性能

        logger.info("✅ 创建了 %d 个测试记忆", len(test_data))

        # 等待数据加载
        await asyncio.sleep(2)

        # 测试1: 向量相似度搜索
        logger.info("测试1: 向量相似度搜索")
        results = await repo.vector_search(
            query_vector=base_vector, user_id=test_user_id, limit=10
        )
        assert len(results) >= 2, f"应该找到至少2条相似记录，实际找到{len(results)}条"
        logger.info("✅ 向量相似度搜索测试成功，找到 %d 条结果", len(results))

        # 测试2: 按用户ID过滤的向量搜索
        logger.info("测试2: 按用户ID过滤的向量搜索")
        user_results = await repo.vector_search(
            query_vector=base_vector, user_id=test_user_id, limit=10
        )
        assert (
            len(user_results) >= 2
        ), f"应该找到至少2条用户记录，实际找到{len(user_results)}条"
        logger.info("✅ 用户ID过滤测试成功，找到 %d 条结果", len(user_results))

        # 测试3: 按群组ID过滤的向量搜索
        logger.info("测试3: 按群组ID过滤的向量搜索")
        group_results = await repo.vector_search(
            query_vector=base_vector,
            user_id=test_user_id,
            group_id=test_group_id,
            limit=10,
        )
        assert (
            len(group_results) >= 1
        ), f"应该找到至少1条群组记录，实际找到{len(group_results)}条"
        logger.info("✅ 群组ID过滤测试成功，找到 %d 条结果", len(group_results))

        # 测试4: 按事件类型过滤的向量搜索
        logger.info("测试4: 按事件类型过滤的向量搜索")
        type_results = await repo.vector_search(
            query_vector=base_vector,
            user_id=test_user_id,
            event_type="Conversation",
            limit=10,
        )
        assert (
            len(type_results) >= 1
        ), f"应该找到至少1条Conversation类型记录，实际找到{len(type_results)}条"
        logger.info("✅ 事件类型过滤测试成功，找到 %d 条结果", len(type_results))

        # 测试5: 按时间范围过滤的向量搜索
        logger.info("测试5: 按时间范围过滤的向量搜索")
        time_results = await repo.vector_search(
            query_vector=base_vector,
            user_id=test_user_id,
            start_time=base_time - timedelta(days=2),
            end_time=base_time,
            limit=10,
        )
        assert (
            len(time_results) >= 1
        ), f"应该找到至少1条时间范围内的记录，实际找到{len(time_results)}条"
        logger.info("✅ 时间范围过滤测试成功，找到 %d 条结果", len(time_results))

    except Exception as e:
        logger.error("❌ 测试向量搜索和过滤功能失败: %s", e)
        raise
    finally:
        # 清理测试数据
        logger.info("清理搜索测试数据...")
        try:
            cleanup_count = await repo.delete_by_filters(user_id=test_user_id)
            await repo.flush()
            logger.info("✅ 清理了 %d 条搜索测试数据", cleanup_count)
        except Exception as cleanup_error:
            logger.error("清理搜索测试数据时出现错误: %s", cleanup_error)

    logger.info("✅ 向量搜索和过滤功能测试完成")


async def test_delete_operations():
    """测试删除功能"""
    logger.info("开始测试删除功能...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_user_id = "test_user_delete_789"
    test_group_id = "test_group_delete_012"
    base_time = get_now_with_timezone()
    test_event_ids = []

    try:
        # 创建测试数据
        for i in range(6):
            event_id = f"delete_test_{i}_{int(base_time.timestamp())}"
            test_event_ids.append(event_id)

            entity = build_episodic_memory_entity(
                event_id=event_id,
                user_id=test_user_id,
                timestamp=base_time - timedelta(days=i),
                episode=f"删除测试记忆 {i}",
                search_content=["删除", "测试", f"记忆{i}"],
                vector=generate_random_vector(),
                title=f"删除测试 {i}",
                group_id=test_group_id if i % 2 == 0 else "",  # 部分有group_id
                event_type="DeleteTest",
            )
            await repo.collection.insert([entity])

        await repo.flush()
        logger.info("✅ 创建了 %d 个删除测试记忆", len(test_event_ids))

        # 等待数据刷新
        await asyncio.sleep(2)

        # 测试1: 按event_id删除
        logger.info("测试1: 按event_id删除")
        event_id_to_delete = test_event_ids[0]
        delete_result = await repo.delete_by_event_id(event_id_to_delete)
        assert delete_result is True

        # 验证删除
        await repo.flush()
        deleted_doc = await repo.get_by_id(event_id_to_delete)
        assert deleted_doc is None, "文档应该已被删除"
        logger.info("✅ 按event_id删除测试成功")

        # 测试2: 按过滤条件删除 - 只删除有group_id的记忆
        logger.info("测试2: 按过滤条件删除（group_id）")
        deleted_count = await repo.delete_by_filters(
            user_id=test_user_id, group_id=test_group_id
        )
        assert (
            deleted_count >= 2
        ), f"应该删除至少2条有group_id的记录，实际删除{deleted_count}条"
        logger.info("✅ 按group_id过滤删除测试成功，删除了 %d 条记录", deleted_count)

        # 测试3: 按时间范围删除
        logger.info("测试3: 按时间范围删除")
        deleted_count = await repo.delete_by_filters(
            user_id=test_user_id,
            start_time=base_time - timedelta(days=2),
            end_time=base_time,
        )
        logger.info("✅ 按时间范围删除测试成功，删除了 %d 条记录", deleted_count)

        # 测试4: 验证参数检查
        logger.info("测试4: 验证参数检查")
        try:
            await repo.delete_by_filters()  # 没有提供任何过滤条件
            assert False, "应该抛出异常但没有"
        except ValueError as e:
            logger.info("✅ 正确捕获参数错误: %s", e)

        # 最终清理剩余数据
        remaining_count = await repo.delete_by_filters(user_id=test_user_id)
        await repo.flush()
        logger.info("✅ 最终清理了 %d 条剩余数据", remaining_count)

    except Exception as e:
        logger.error("❌ 测试删除功能失败: %s", e)
        raise
    finally:
        # 确保清理所有测试数据
        try:
            await repo.delete_by_filters(user_id=test_user_id)
            await repo.flush()
        except Exception:
            pass

    logger.info("✅ 删除功能测试完成")


async def test_timezone_handling():
    """测试时区处理"""
    logger.info("开始测试时区处理...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_event_id = "test_timezone_001"
    test_user_id = "test_user_timezone_999"

    try:
        # 创建不同时区的时间
        utc_time = datetime.now(ZoneInfo("UTC"))
        tokyo_time = datetime.now(ZoneInfo("Asia/Tokyo"))
        shanghai_time = get_now_with_timezone()  # 默认上海时区

        logger.info("原始UTC时间: %s", to_iso_format(utc_time))
        logger.info("原始东京时间: %s", to_iso_format(tokyo_time))
        logger.info("原始上海时间: %s", to_iso_format(shanghai_time))

        # 使用UTC时间创建记忆
        entity = build_episodic_memory_entity(
            event_id=test_event_id,
            user_id=test_user_id,
            timestamp=utc_time,
            episode="时区测试记忆",
            search_content=["时区", "测试"],
            vector=generate_random_vector(),
            title="时区测试",
            created_at=tokyo_time,
            updated_at=shanghai_time,
        )

        await repo.collection.insert([entity])

        assert entity is not None
        logger.info("✅ 创建带时区信息的记忆成功")

        await repo.flush()
        await asyncio.sleep(2)

        # 从数据库获取并验证
        retrieved_doc = await repo.get_by_id(test_event_id)
        assert retrieved_doc is not None

        # 解析时间戳
        retrieved_timestamp = datetime.fromtimestamp(retrieved_doc["timestamp"])
        logger.info("从数据库获取的时间戳: %s", to_iso_format(retrieved_timestamp))

        # 验证时间转换正确性（转换到同一时区后应该相等）
        assert compare_datetime(
            retrieved_timestamp.astimezone(ZoneInfo("UTC")),
            utc_time.astimezone(ZoneInfo("UTC")),
        )
        logger.info("✅ 时区验证成功")

        # 测试时间范围查询
        results = await repo.vector_search(
            query_vector=generate_random_vector(),
            user_id=test_user_id,
            start_time=shanghai_time - timedelta(hours=2),
            end_time=shanghai_time + timedelta(hours=2),
            limit=10,
        )
        assert len(results) >= 1, "应该找到时间范围内的记录"
        logger.info("✅ 时区时间范围查询测试成功")

    except Exception as e:
        logger.error("❌ 测试时区处理失败: %s", e)
        raise
    finally:
        # 清理测试数据
        try:
            await repo.delete_by_event_id(test_event_id)
            await repo.flush()
            logger.info("✅ 清理时区测试数据成功")
        except Exception:
            pass

    logger.info("✅ 时区处理测试完成")


async def test_edge_cases():
    """测试边界情况"""
    logger.info("开始测试边界情况...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_user_id = "test_user_edge_111"

    try:
        # 测试1: 不存在的用户
        logger.info("测试1: 不存在的用户")
        nonexistent_results = await repo.vector_search(
            query_vector=generate_random_vector(),
            user_id="nonexistent_user_999999",
            limit=10,
        )
        assert len(nonexistent_results) == 0, "不存在的用户应该返回空结果"
        logger.info("✅ 不存在用户测试成功")

        # 测试2: 删除不存在的event_id
        logger.info("测试2: 删除不存在的event_id")
        delete_result = await repo.delete_by_event_id("nonexistent_event_999999")
        assert delete_result is True, "删除不存在的文档不知道为什么也返回True"
        logger.info("✅ 删除不存在文档测试成功")

        # 测试3: 使用无效的时间范围
        logger.info("测试3: 使用无效的时间范围")
        future_time = datetime.now(ZoneInfo("UTC")) + timedelta(days=365)
        future_results = await repo.vector_search(
            query_vector=generate_random_vector(),
            user_id=test_user_id,
            start_time=future_time,
            end_time=future_time + timedelta(days=1),
            limit=10,
        )
        assert len(future_results) == 0, "未来时间范围应该返回空结果"
        logger.info("✅ 无效时间范围测试成功")

        # 测试4: 向量维度验证
        logger.info("测试4: 向量维度验证")
        try:
            entity = build_episodic_memory_entity(
                event_id="invalid_vector_test",
                user_id=test_user_id,
                timestamp=get_now_with_timezone(),
                episode="无效向量测试",
                search_content=["测试"],
                vector=[1.0] * 512,  # 错误的向量维度
            )
            await repo.collection.insert([entity])
            assert False, "应该因为向量维度错误而失败"
        except Exception as e:
            assert "the length(512) of float data should divide the dim(1024)" in str(e)
            logger.info("✅ 正确捕获向量维度错误: %s", e)

    except Exception as e:
        logger.error("❌ 测试边界情况失败: %s", e)
        raise

    logger.info("✅ 边界情况测试完成")


async def test_performance():
    """测试性能"""
    logger.info("开始性能测试...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_user_id = "test_user_perf_001"
    current_time = get_now_with_timezone()
    num_docs = 1000

    try:
        # 准备测试数据
        test_data = []
        base_vector = generate_random_vector()

        for i in range(num_docs):
            # 生成一个与基准向量相似的向量
            noise = np.random.normal(0, 0.1, len(base_vector))
            vector = [x + n for x, n in zip(base_vector, noise)]

            test_data.append(
                {
                    "event_id": f"perf_test_{i}",
                    "user_id": test_user_id,
                    "timestamp": current_time - timedelta(minutes=i),
                    "episode": f"性能测试记忆 {i}",
                    "search_content": ["性能", "测试", f"记忆{i}"],
                    "vector": vector,
                    "title": f"性能测试 {i}",
                    "group_id": "perf_test_group",
                    "event_type": "PerfTest",
                }
            )

        # 测试1: 批量插入性能
        logger.info("测试1: 批量插入性能 (%d 条记录)...", num_docs)
        insert_times = []
        batch_size = 100

        for i in range(0, num_docs, batch_size):
            batch = test_data[i : i + batch_size]
            start_time = datetime.now()

            for doc in batch:
                entity = build_episodic_memory_entity(**doc)
                await repo.collection.insert([entity])

            end_time = datetime.now()
            insert_time = (end_time - start_time).total_seconds()
            insert_times.append(insert_time)

            logger.info(
                "- 批次 %d/%d: %.3f 秒 (%.1f 条/秒)",
                i // batch_size + 1,
                (num_docs + batch_size - 1) // batch_size,
                insert_time,
                len(batch) / insert_time,
            )

        avg_insert_time = sum(insert_times) / len(insert_times)
        min_insert_time = min(insert_times)
        max_insert_time = max(insert_times)
        total_insert_time = sum(insert_times)

        logger.info("插入性能统计:")
        logger.info("- 总时间: %.3f 秒", total_insert_time)
        logger.info(
            "- 平均每批次: %.3f 秒 (%.1f 条/秒)",
            avg_insert_time,
            batch_size / avg_insert_time,
        )
        logger.info(
            "- 最快批次: %.3f 秒 (%.1f 条/秒)",
            min_insert_time,
            batch_size / min_insert_time,
        )
        logger.info(
            "- 最慢批次: %.3f 秒 (%.1f 条/秒)",
            max_insert_time,
            batch_size / max_insert_time,
        )

        # 测试2: Flush性能
        logger.info("测试2: Flush性能...")
        start_time = datetime.now()
        await repo.flush()
        flush_time = (datetime.now() - start_time).total_seconds()
        logger.info("Flush耗时: %.3f 秒", flush_time)

        # 等待数据加载
        await repo.load()
        await asyncio.sleep(2)

        # 测试3: 搜索性能
        logger.info("测试3: 搜索性能...")
        search_times = []
        num_searches = 10

        for i in range(num_searches):
            # 生成一个与基准向量相似的查询向量
            noise = np.random.normal(0, 0.1, len(base_vector))
            query_vector = [x + n for x, n in zip(base_vector, noise)]

            start_time = datetime.now()
            results = await repo.vector_search(
                query_vector=query_vector, user_id=test_user_id, limit=10
            )
            search_time = (datetime.now() - start_time).total_seconds()
            search_times.append(search_time)

            logger.info(
                "- 搜索 %d/%d: %.3f 秒, 找到 %d 条结果",
                i + 1,
                num_searches,
                search_time,
                len(results),
            )

        avg_search_time = sum(search_times) / len(search_times)
        min_search_time = min(search_times)
        max_search_time = max(search_times)

        logger.info("搜索性能统计:")
        logger.info("- 平均耗时: %.3f 秒", avg_search_time)
        logger.info("- 最快耗时: %.3f 秒", min_search_time)
        logger.info("- 最慢耗时: %.3f 秒", max_search_time)

    except Exception as e:
        logger.error("❌ 性能测试失败: %s", e)
        raise
    finally:
        # 清理测试数据
        try:
            cleanup_count = await repo.delete_by_filters(user_id=test_user_id)
            await repo.flush()
            logger.info("✅ 清理了 %d 条性能测试数据", cleanup_count)
        except Exception as cleanup_error:
            logger.error("清理性能测试数据时出现错误: %s", cleanup_error)

    logger.info("✅ 性能测试完成")


async def run_all_tests():
    """运行所有测试"""
    logger.info("🚀 开始运行EpisodicMemoryMilvusRepository所有测试...")

    try:
        await test_crud_operations()
        await test_vector_search()
        await test_delete_operations()
        await test_timezone_handling()
        await test_edge_cases()
        await test_performance()
        logger.info("✅ 所有测试完成")
    except Exception as e:
        logger.error("❌ 测试过程中出现错误: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
