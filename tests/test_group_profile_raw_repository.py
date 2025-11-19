#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 GroupProfileRawRepository 的版本管理功能

测试内容包括:
1. 基于group_id的增删改查操作（支持版本管理）
2. 版本管理相关功能测试
3. ensure_latest 方法测试
4. 批量查询的 only_latest 功能测试
"""

import asyncio
from datetime import datetime

from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.group_profile_raw_repository import (
    GroupProfileRawRepository,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


async def test_basic_crud_operations():
    """测试基本的增删改查操作（带版本管理）"""
    logger.info("开始测试基本的增删改查操作...")

    repo = get_bean_by_type(GroupProfileRawRepository)
    group_id = "test_group_001"
    current_timestamp = int(datetime.now().timestamp() * 1000)

    try:
        # 先清理可能存在的测试数据
        await repo.delete_by_group_id(group_id)
        logger.info("✅ 清理已存在的测试数据")

        # 测试创建新记录（必须提供version）
        group_data = {
            "version": "v1",
            "group_name": "技术讨论组",
            "subject": "技术交流与学习",
            "summary": "本群组主要讨论各种技术话题，促进技术交流",
        }

        result = await repo.upsert_by_group_id(group_id, group_data, current_timestamp)
        assert result is not None
        assert result.group_id == group_id
        assert result.group_name == "技术讨论组"
        assert result.version == "v1"
        assert result.is_latest == True
        assert result.timestamp == current_timestamp
        logger.info("✅ 测试创建新记录成功（version=v1, is_latest=True）")

        # 测试根据group_id查询（应该返回最新版本）
        queried = await repo.get_by_group_id(group_id)
        assert queried is not None
        assert queried.group_id == group_id
        assert queried.version == "v1"
        assert queried.is_latest == True
        logger.info("✅ 测试根据group_id查询成功")

        # 测试更新记录（不改变version）
        update_data = {"group_name": "高级技术讨论组", "summary": "更新后的群组描述"}

        updated = await repo.update_by_group_id(group_id, update_data)
        assert updated is not None
        assert updated.group_name == "高级技术讨论组"
        assert updated.summary == "更新后的群组描述"
        assert updated.version == "v1"  # 版本未变
        assert updated.subject == "技术交流与学习"  # 未更新的字段应保持原值
        logger.info("✅ 测试更新记录成功（版本未变）")

        # 测试删除特定版本
        deleted = await repo.delete_by_group_id(group_id, version="v1")
        assert deleted is True
        logger.info("✅ 测试删除特定版本成功")

        # 验证删除
        final_check = await repo.get_by_group_id(group_id)
        assert final_check is None, "记录应该已被删除"
        logger.info("✅ 验证删除成功")

    except Exception as e:
        logger.error("❌ 测试基本增删改查操作失败: %s", e)
        raise

    logger.info("✅ 基本增删改查操作测试完成")


async def test_version_management():
    """测试版本管理功能"""
    logger.info("开始测试版本管理功能...")

    repo = get_bean_by_type(GroupProfileRawRepository)
    group_id = "test_group_version_002"
    current_timestamp = int(datetime.now().timestamp() * 1000)

    try:
        # 先清理可能存在的测试数据
        await repo.delete_by_group_id(group_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建第一个版本
        v1_data = {"version": "202501", "group_name": "技术群v1", "subject": "初始版本"}

        v1_result = await repo.upsert_by_group_id(group_id, v1_data, current_timestamp)
        assert v1_result is not None
        assert v1_result.version == "202501"
        assert v1_result.is_latest == True
        logger.info("✅ 创建版本 202501 成功，is_latest=True")

        # 创建第二个版本
        v2_data = {"version": "202502", "group_name": "技术群v2", "subject": "第二版本"}

        v2_result = await repo.upsert_by_group_id(group_id, v2_data, current_timestamp)
        assert v2_result is not None
        assert v2_result.version == "202502"
        assert v2_result.is_latest == True
        logger.info("✅ 创建版本 202502 成功，is_latest=True")

        # 创建第三个版本
        v3_data = {"version": "202503", "group_name": "技术群v3", "subject": "第三版本"}

        v3_result = await repo.upsert_by_group_id(group_id, v3_data, current_timestamp)
        assert v3_result is not None
        assert v3_result.version == "202503"
        assert v3_result.is_latest == True
        logger.info("✅ 创建版本 202503 成功，is_latest=True")

        # 测试获取最新版本（不指定version_range）
        latest = await repo.get_by_group_id(group_id)
        assert latest is not None
        assert latest.version == "202503"
        assert latest.is_latest == True
        logger.info("✅ 获取最新版本成功: version=202503")

        # 测试版本范围查询（左闭右闭，返回范围内最新版本）
        v2_by_range = await repo.get_by_group_id(
            group_id, version_range=("202502", "202502")
        )
        assert v2_by_range is not None
        assert v2_by_range.version == "202502"
        logger.info("✅ 版本范围查询 [202502, 202502] 成功，返回 version=202502")

        # 测试多版本范围查询（返回范围内最新版本）
        v_multi_range = await repo.get_by_group_id(
            group_id, version_range=("202501", "202502")
        )
        assert v_multi_range is not None
        assert v_multi_range.version == "202502"  # 返回范围内最新的版本
        logger.info("✅ 版本范围查询 [202501, 202502] 成功，返回最新版本 202502")

        # 测试更新特定版本
        update_v2 = {"subject": "更新后的第二版本"}

        updated_v2 = await repo.update_by_group_id(
            group_id, update_v2, version="202502"
        )
        assert updated_v2 is not None
        assert updated_v2.version == "202502"
        assert updated_v2.subject == "更新后的第二版本"
        logger.info("✅ 更新特定版本 202502 成功")

        # 测试删除中间版本
        await repo.delete_by_group_id(group_id, version="202502")
        logger.info("✅ 删除版本 202502 成功")

        # 验证删除后最新版本仍然正确
        latest_after_delete = await repo.get_by_group_id(group_id)
        assert latest_after_delete is not None
        assert latest_after_delete.version == "202503"
        assert latest_after_delete.is_latest == True
        logger.info("✅ 删除中间版本后，最新版本仍正确")

        # 清理所有版本
        await repo.delete_by_group_id(group_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试版本管理功能失败: %s", e)
        raise

    logger.info("✅ 版本管理功能测试完成")


async def test_ensure_latest():
    """测试 ensure_latest 方法"""
    logger.info("开始测试 ensure_latest 方法...")

    repo = get_bean_by_type(GroupProfileRawRepository)
    group_id = "test_group_ensure_003"
    current_timestamp = int(datetime.now().timestamp() * 1000)

    try:
        # 先清理可能存在的测试数据
        await repo.delete_by_group_id(group_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建多个版本
        versions = ["202501", "202502", "202503", "202504"]
        for version in versions:
            data = {
                "version": version,
                "group_name": f"技术群{version}",
                "subject": f"版本{version}",
            }
            await repo.upsert_by_group_id(group_id, data, current_timestamp)

        logger.info("✅ 创建了 4 个版本")

        # 手动调用 ensure_latest
        result = await repo.ensure_latest(group_id)
        assert result is True
        logger.info("✅ ensure_latest 执行成功")

        # 验证最新版本
        latest = await repo.get_by_group_id(group_id)
        assert latest is not None
        assert latest.version == "202504"
        assert latest.is_latest == True
        logger.info("✅ 验证最新版本正确: version=202504, is_latest=True")

        # 验证旧版本的 is_latest 都是 False
        for old_version in ["202501", "202502", "202503"]:
            # 使用相同的起止版本来精确查询单个版本
            old_doc = await repo.get_by_group_id(
                group_id, version_range=(old_version, old_version)
            )
            assert old_doc is not None
            assert old_doc.is_latest == False
            logger.info("✅ 验证旧版本 %s 的 is_latest=False", old_version)

        # 测试幂等性：再次调用 ensure_latest
        result2 = await repo.ensure_latest(group_id)
        assert result2 is True
        logger.info("✅ ensure_latest 幂等性验证成功")

        # 清理测试数据
        await repo.delete_by_group_id(group_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试 ensure_latest 方法失败: %s", e)
        raise

    logger.info("✅ ensure_latest 方法测试完成")


async def test_batch_query_with_only_latest():
    """测试批量查询的 only_latest 功能"""
    logger.info("开始测试批量查询的 only_latest 功能...")

    repo = get_bean_by_type(GroupProfileRawRepository)
    base_group_id = "test_batch_group"
    current_timestamp = int(datetime.now().timestamp() * 1000)

    try:
        # 创建多个群组，每个群组有多个版本
        group_ids = [f"{base_group_id}_{i}" for i in range(1, 4)]

        # 先清理
        for gid in group_ids:
            await repo.delete_by_group_id(gid)
        logger.info("✅ 清理已存在的测试数据")

        # 为每个群组创建多个版本
        for gid in group_ids:
            for version in ["202501", "202502", "202503"]:
                data = {
                    "version": version,
                    "group_name": f"{gid}_{version}",
                    "subject": f"群组{gid}版本{version}",
                }
                await repo.upsert_by_group_id(gid, data, current_timestamp)

        logger.info("✅ 创建了 3 个群组，每个群组 3 个版本")

        # 测试 only_latest=True（默认）
        latest_results = await repo.find_by_group_ids(group_ids, only_latest=True)
        assert len(latest_results) == 3

        for result in latest_results:
            assert result.version == "202503"
            assert result.is_latest == True

        logger.info("✅ 批量查询 only_latest=True 成功，返回 3 个最新版本")

        # 测试 only_latest=False（返回所有版本）
        all_results = await repo.find_by_group_ids(group_ids, only_latest=False)
        assert len(all_results) == 9  # 3个群组 * 3个版本
        logger.info("✅ 批量查询 only_latest=False 成功，返回 9 个版本")

        # 清理测试数据
        for gid in group_ids:
            await repo.delete_by_group_id(gid)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试批量查询 only_latest 功能失败: %s", e)
        raise

    logger.info("✅ 批量查询 only_latest 功能测试完成")


async def test_create_without_version_should_fail():
    """测试创建时不提供 version 应该失败"""
    logger.info("开始测试创建时不提供 version 应该失败...")

    repo = get_bean_by_type(GroupProfileRawRepository)
    group_id = "test_no_version_004"
    current_timestamp = int(datetime.now().timestamp() * 1000)

    try:
        # 先清理
        await repo.delete_by_group_id(group_id)

        # 尝试创建不带 version 的记录
        data_without_version = {"group_name": "无版本群组", "subject": "这应该失败"}

        try:
            await repo.upsert_by_group_id(
                group_id, data_without_version, current_timestamp
            )
            assert False, "创建不带version的记录应该抛出异常"
        except ValueError as e:
            logger.info("✅ 正确抛出 ValueError: %s", str(e))
            assert "必须提供version字段" in str(e)

        logger.info("✅ 创建时不提供 version 正确失败")

    except AssertionError:
        raise
    except Exception as e:
        logger.error("❌ 测试创建不带version失败: %s", e)
        raise

    logger.info("✅ 创建不带version测试完成")


async def run_all_tests():
    """运行所有测试"""
    logger.info("🚀 开始运行 GroupProfile 所有测试...")

    try:
        await test_basic_crud_operations()
        await test_version_management()
        await test_ensure_latest()
        await test_batch_query_with_only_latest()
        await test_create_without_version_should_fail()
        logger.info("✅ 所有测试完成")
    except Exception as e:
        logger.error("❌ 测试过程中出现错误: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
