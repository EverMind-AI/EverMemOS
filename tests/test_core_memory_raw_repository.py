#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 CoreMemoryRawRepository 的版本管理功能

测试内容包括:
1. 基于user_id的增删改查操作（支持版本管理）
2. 版本管理相关功能测试
3. ensure_latest 方法测试
4. 批量查询的 only_latest 功能测试
"""

import asyncio

from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
    CoreMemoryRawRepository,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


async def test_basic_crud_operations():
    """测试基本的增删改查操作（带版本管理）"""
    logger.info("开始测试基本的增删改查操作...")

    repo = get_bean_by_type(CoreMemoryRawRepository)
    user_id = "test_user_001"

    try:
        # 先清理可能存在的测试数据
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 测试创建新记录（必须提供version）
        user_data = {
            "version": "v1",
            "user_name": "张三",
            "gender": "男",
            "position": "高级工程师",
            "department": "技术部",
        }

        result = await repo.upsert_by_user_id(user_id, user_data)
        assert result is not None
        assert result.user_id == user_id
        assert result.user_name == "张三"
        assert result.version == "v1"
        assert result.is_latest == True
        logger.info("✅ 测试创建新记录成功（version=v1, is_latest=True）")

        # 测试根据user_id查询（应该返回最新版本）
        queried = await repo.get_by_user_id(user_id)
        assert queried is not None
        assert queried.user_id == user_id
        assert queried.version == "v1"
        assert queried.is_latest == True
        logger.info("✅ 测试根据user_id查询成功")

        # 测试更新记录（不改变version）
        update_data = {"position": "资深工程师", "department": "研发部"}

        updated = await repo.update_by_user_id(user_id, update_data)
        assert updated is not None
        assert updated.position == "资深工程师"
        assert updated.department == "研发部"
        assert updated.version == "v1"  # 版本未变
        assert updated.user_name == "张三"  # 未更新的字段应保持原值
        logger.info("✅ 测试更新记录成功（版本未变）")

        # 测试删除特定版本
        deleted = await repo.delete_by_user_id(user_id, version="v1")
        assert deleted is True
        logger.info("✅ 测试删除特定版本成功")

        # 验证删除
        final_check = await repo.get_by_user_id(user_id)
        assert final_check is None, "记录应该已被删除"
        logger.info("✅ 验证删除成功")

    except Exception as e:
        logger.error("❌ 测试基本增删改查操作失败: %s", e)
        raise

    logger.info("✅ 基本增删改查操作测试完成")


async def test_version_management():
    """测试版本管理功能"""
    logger.info("开始测试版本管理功能...")

    repo = get_bean_by_type(CoreMemoryRawRepository)
    user_id = "test_user_version_002"

    try:
        # 先清理可能存在的测试数据
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建第一个版本
        v1_data = {"version": "202501", "user_name": "李四v1", "position": "工程师"}

        v1_result = await repo.upsert_by_user_id(user_id, v1_data)
        assert v1_result is not None
        assert v1_result.version == "202501"
        assert v1_result.is_latest == True
        logger.info("✅ 创建版本 202501 成功，is_latest=True")

        # 创建第二个版本
        v2_data = {"version": "202502", "user_name": "李四v2", "position": "高级工程师"}

        v2_result = await repo.upsert_by_user_id(user_id, v2_data)
        assert v2_result is not None
        assert v2_result.version == "202502"
        assert v2_result.is_latest == True
        logger.info("✅ 创建版本 202502 成功，is_latest=True")

        # 创建第三个版本
        v3_data = {"version": "202503", "user_name": "李四v3", "position": "资深工程师"}

        v3_result = await repo.upsert_by_user_id(user_id, v3_data)
        assert v3_result is not None
        assert v3_result.version == "202503"
        assert v3_result.is_latest == True
        logger.info("✅ 创建版本 202503 成功，is_latest=True")

        # 测试获取最新版本（不指定version_range）
        latest = await repo.get_by_user_id(user_id)
        assert latest is not None
        assert latest.version == "202503"
        assert latest.is_latest == True
        logger.info("✅ 获取最新版本成功: version=202503")

        # 测试版本范围查询（左闭右开）
        v2_by_range = await repo.get_by_user_id(
            user_id, version_range=("202502", "202503")
        )
        assert v2_by_range is not None
        assert v2_by_range.version == "202502"
        logger.info("✅ 版本范围查询 [202502, 202503) 成功，返回 version=202502")

        # 测试更新特定版本
        update_v2 = {"position": "更新后的高级工程师"}

        updated_v2 = await repo.update_by_user_id(user_id, update_v2, version="202502")
        assert updated_v2 is not None
        assert updated_v2.version == "202502"
        assert updated_v2.position == "更新后的高级工程师"
        logger.info("✅ 更新特定版本 202502 成功")

        # 测试删除中间版本
        await repo.delete_by_user_id(user_id, version="202502")
        logger.info("✅ 删除版本 202502 成功")

        # 验证删除后最新版本仍然正确
        latest_after_delete = await repo.get_by_user_id(user_id)
        assert latest_after_delete is not None
        assert latest_after_delete.version == "202503"
        assert latest_after_delete.is_latest == True
        logger.info("✅ 删除中间版本后，最新版本仍正确")

        # 清理所有版本
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试版本管理功能失败: %s", e)
        raise

    logger.info("✅ 版本管理功能测试完成")


async def test_ensure_latest():
    """测试 ensure_latest 方法"""
    logger.info("开始测试 ensure_latest 方法...")

    repo = get_bean_by_type(CoreMemoryRawRepository)
    user_id = "test_user_ensure_003"

    try:
        # 先清理可能存在的测试数据
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建多个版本
        versions = ["202501", "202502", "202503", "202504"]
        for version in versions:
            data = {
                "version": version,
                "user_name": f"王五{version}",
                "position": f"版本{version}",
            }
            await repo.upsert_by_user_id(user_id, data)

        logger.info("✅ 创建了 4 个版本")

        # 手动调用 ensure_latest
        result = await repo.ensure_latest(user_id)
        assert result is True
        logger.info("✅ ensure_latest 执行成功")

        # 验证最新版本
        latest = await repo.get_by_user_id(user_id)
        assert latest is not None
        assert latest.version == "202504"
        assert latest.is_latest == True
        logger.info("✅ 验证最新版本正确: version=202504, is_latest=True")

        # 验证旧版本的 is_latest 都是 False
        for old_version in ["202501", "202502", "202503"]:
            old_doc = await repo.get_by_user_id(
                user_id, version_range=(old_version, str(int(old_version) + 1))
            )
            assert old_doc is not None
            assert old_doc.is_latest == False
            logger.info("✅ 验证旧版本 %s 的 is_latest=False", old_version)

        # 测试幂等性：再次调用 ensure_latest
        result2 = await repo.ensure_latest(user_id)
        assert result2 is True
        logger.info("✅ ensure_latest 幂等性验证成功")

        # 清理测试数据
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试 ensure_latest 方法失败: %s", e)
        raise

    logger.info("✅ ensure_latest 方法测试完成")


async def test_batch_query_with_only_latest():
    """测试批量查询的 only_latest 功能"""
    logger.info("开始测试批量查询的 only_latest 功能...")

    repo = get_bean_by_type(CoreMemoryRawRepository)
    base_user_id = "test_batch_user"

    try:
        # 创建多个用户，每个用户有多个版本
        user_ids = [f"{base_user_id}_{i}" for i in range(1, 4)]

        # 先清理
        for uid in user_ids:
            await repo.delete_by_user_id(uid)
        logger.info("✅ 清理已存在的测试数据")

        # 为每个用户创建多个版本
        for uid in user_ids:
            for version in ["202501", "202502", "202503"]:
                data = {
                    "version": version,
                    "user_name": f"{uid}_{version}",
                    "position": f"用户{uid}版本{version}",
                }
                await repo.upsert_by_user_id(uid, data)

        logger.info("✅ 创建了 3 个用户，每个用户 3 个版本")

        # 测试 only_latest=True（默认）
        latest_results = await repo.find_by_user_ids(user_ids, only_latest=True)
        assert len(latest_results) == 3

        for result in latest_results:
            assert result.version == "202503"
            assert result.is_latest == True

        logger.info("✅ 批量查询 only_latest=True 成功，返回 3 个最新版本")

        # 测试 only_latest=False（返回所有版本）
        all_results = await repo.find_by_user_ids(user_ids, only_latest=False)
        assert len(all_results) == 9  # 3个用户 * 3个版本
        logger.info("✅ 批量查询 only_latest=False 成功，返回 9 个版本")

        # 清理测试数据
        for uid in user_ids:
            await repo.delete_by_user_id(uid)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试批量查询 only_latest 功能失败: %s", e)
        raise

    logger.info("✅ 批量查询 only_latest 功能测试完成")


async def test_profile_fields():
    """测试 profile 相关字段"""
    logger.info("开始测试 profile 相关字段...")

    repo = get_bean_by_type(CoreMemoryRawRepository)
    user_id = "test_user_profile_005"

    try:
        # 先清理
        await repo.delete_by_user_id(user_id)

        # 创建包含 profile 字段的记录
        user_data = {
            "version": "v1",
            "user_name": "测试用户",
            "hard_skills": [
                {"value": "Python", "level": "高级", "evidences": ["conv_001"]}
            ],
            "soft_skills": [
                {"value": "沟通能力", "level": "优秀", "evidences": ["conv_002"]}
            ],
            "personality": [{"value": "内向但善于沟通", "evidences": ["conv_003"]}],
            "interests": [{"value": "编程", "evidences": ["conv_004"]}],
        }

        result = await repo.upsert_by_user_id(user_id, user_data)
        assert result is not None
        assert result.hard_skills is not None
        assert len(result.hard_skills) == 1
        assert result.hard_skills[0]["value"] == "Python"
        logger.info("✅ 创建包含 profile 字段的记录成功")

        # 测试 get_profile 方法
        profile = repo.get_profile(result)
        assert profile is not None
        assert "hard_skills" in profile
        assert "soft_skills" in profile
        assert "personality" in profile
        assert "interests" in profile
        logger.info("✅ get_profile 方法测试成功")

        # 测试 get_base 方法
        base = repo.get_base(result)
        assert base is not None
        assert "user_name" in base
        logger.info("✅ get_base 方法测试成功")

        # 清理
        await repo.delete_by_user_id(user_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试 profile 字段失败: %s", e)
        raise

    logger.info("✅ profile 字段测试完成")


async def test_create_without_version_should_fail():
    """测试创建时不提供 version 应该失败"""
    logger.info("开始测试创建时不提供 version 应该失败...")

    repo = get_bean_by_type(CoreMemoryRawRepository)
    user_id = "test_no_version_006"

    try:
        # 先清理
        await repo.delete_by_user_id(user_id)

        # 尝试创建不带 version 的记录
        data_without_version = {"user_name": "无版本用户", "position": "这应该失败"}

        try:
            await repo.upsert_by_user_id(user_id, data_without_version)
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
    logger.info("🚀 开始运行 CoreMemory 所有测试...")

    try:
        await test_basic_crud_operations()
        await test_version_management()
        await test_ensure_latest()
        await test_batch_query_with_only_latest()
        await test_profile_fields()
        await test_create_without_version_should_fail()
        logger.info("✅ 所有测试完成")
    except Exception as e:
        logger.error("❌ 测试过程中出现错误: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
