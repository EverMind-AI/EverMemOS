#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 GroupUserProfileMemoryRawRepository 的版本管理功能

测试内容包括:
1. 基于user_id+group_id的增删改查操作（支持版本管理）
2. 版本管理相关功能测试
3. ensure_latest 方法测试
4. 批量查询的 only_latest 功能测试
"""

import asyncio

from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.group_user_profile_memory_raw_repository import (
    GroupUserProfileMemoryRawRepository,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


async def test_basic_crud_operations():
    """测试基本的增删改查操作（带版本管理）"""
    logger.info("开始测试基本的增删改查操作...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_user_001"
    group_id = "test_group_001"

    try:
        # 先清理可能存在的测试数据
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ 清理已存在的测试数据")

        # 测试创建新记录（必须提供version）
        profile_data = {
            "version": "v1",
            "user_name": "张三",
            "hard_skills": [
                {"value": "Python", "level": "高级", "evidences": ["conv_001"]}
            ],
            "personality": [{"value": "善于沟通", "evidences": ["conv_002"]}],
        }

        result = await repo.upsert_by_user_group(user_id, group_id, profile_data)
        assert result is not None
        assert result.user_id == user_id
        assert result.group_id == group_id
        assert result.user_name == "张三"
        assert result.version == "v1"
        assert result.is_latest == True
        logger.info("✅ 测试创建新记录成功（version=v1, is_latest=True）")

        # 测试根据user_id和group_id查询（应该返回最新版本）
        queried = await repo.get_by_user_group(user_id, group_id)
        assert queried is not None
        assert queried.user_id == user_id
        assert queried.group_id == group_id
        assert queried.version == "v1"
        assert queried.is_latest == True
        logger.info("✅ 测试根据user_id和group_id查询成功")

        # 测试更新记录（不改变version）
        update_data = {
            "user_name": "张三（更新）",
            "soft_skills": [
                {"value": "领导力", "level": "中级", "evidences": ["conv_003"]}
            ],
        }

        updated = await repo.update_by_user_group(user_id, group_id, update_data)
        assert updated is not None
        assert updated.user_name == "张三（更新）"
        assert updated.soft_skills is not None
        assert updated.version == "v1"  # 版本未变
        logger.info("✅ 测试更新记录成功（版本未变）")

        # 测试删除特定版本
        deleted = await repo.delete_by_user_group(user_id, group_id, version="v1")
        assert deleted is True
        logger.info("✅ 测试删除特定版本成功")

        # 验证删除
        final_check = await repo.get_by_user_group(user_id, group_id)
        assert final_check is None, "记录应该已被删除"
        logger.info("✅ 验证删除成功")

    except Exception as e:
        logger.error("❌ 测试基本增删改查操作失败: %s", e)
        raise

    logger.info("✅ 基本增删改查操作测试完成")


async def test_version_management():
    """测试版本管理功能"""
    logger.info("开始测试版本管理功能...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_user_version_002"
    group_id = "test_group_version_002"

    try:
        # 先清理可能存在的测试数据
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建第一个版本
        v1_data = {
            "version": "202501",
            "user_name": "李四v1",
            "personality": [{"value": "内向", "evidences": ["conv_001"]}],
        }

        v1_result = await repo.upsert_by_user_group(user_id, group_id, v1_data)
        assert v1_result is not None
        assert v1_result.version == "202501"
        assert v1_result.is_latest == True
        logger.info("✅ 创建版本 202501 成功，is_latest=True")

        # 创建第二个版本
        v2_data = {
            "version": "202502",
            "user_name": "李四v2",
            "personality": [{"value": "外向", "evidences": ["conv_002"]}],
        }

        v2_result = await repo.upsert_by_user_group(user_id, group_id, v2_data)
        assert v2_result is not None
        assert v2_result.version == "202502"
        assert v2_result.is_latest == True
        logger.info("✅ 创建版本 202502 成功，is_latest=True")

        # 创建第三个版本
        v3_data = {
            "version": "202503",
            "user_name": "李四v3",
            "personality": [{"value": "平衡", "evidences": ["conv_003"]}],
        }

        v3_result = await repo.upsert_by_user_group(user_id, group_id, v3_data)
        assert v3_result is not None
        assert v3_result.version == "202503"
        assert v3_result.is_latest == True
        logger.info("✅ 创建版本 202503 成功，is_latest=True")

        # 测试获取最新版本（不指定version_range）
        latest = await repo.get_by_user_group(user_id, group_id)
        assert latest is not None
        assert latest.version == "202503"
        assert latest.is_latest == True
        logger.info("✅ 获取最新版本成功: version=202503")

        # 测试版本范围查询（左闭右闭）
        v2_by_range = await repo.get_by_user_group(
            user_id, group_id, version_range=("202502", "202502")
        )
        assert v2_by_range is not None
        assert v2_by_range.version == "202502"
        logger.info("✅ 版本范围查询 [202502, 202502] 成功，返回 version=202502")

        # 测试更新特定版本
        update_v2 = {"user_name": "李四v2（更新）"}

        updated_v2 = await repo.update_by_user_group(
            user_id, group_id, update_v2, version="202502"
        )
        assert updated_v2 is not None
        assert updated_v2.version == "202502"
        assert updated_v2.user_name == "李四v2（更新）"
        logger.info("✅ 更新特定版本 202502 成功")

        # 测试删除中间版本
        await repo.delete_by_user_group(user_id, group_id, version="202502")
        logger.info("✅ 删除版本 202502 成功")

        # 验证删除后最新版本仍然正确
        latest_after_delete = await repo.get_by_user_group(user_id, group_id)
        assert latest_after_delete is not None
        assert latest_after_delete.version == "202503"
        assert latest_after_delete.is_latest == True
        logger.info("✅ 删除中间版本后，最新版本仍正确")

        # 清理所有版本
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试版本管理功能失败: %s", e)
        raise

    logger.info("✅ 版本管理功能测试完成")


async def test_ensure_latest():
    """测试 ensure_latest 方法"""
    logger.info("开始测试 ensure_latest 方法...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_user_ensure_003"
    group_id = "test_group_ensure_003"

    try:
        # 先清理可能存在的测试数据
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建多个版本
        versions = ["202501", "202502", "202503", "202504"]
        for version in versions:
            data = {"version": version, "user_name": f"王五{version}"}
            await repo.upsert_by_user_group(user_id, group_id, data)

        logger.info("✅ 创建了 4 个版本")

        # 手动调用 ensure_latest
        result = await repo.ensure_latest(user_id, group_id)
        assert result is True
        logger.info("✅ ensure_latest 执行成功")

        # 验证最新版本
        latest = await repo.get_by_user_group(user_id, group_id)
        assert latest is not None
        assert latest.version == "202504"
        assert latest.is_latest == True
        logger.info("✅ 验证最新版本正确: version=202504, is_latest=True")

        # 验证旧版本的 is_latest 都是 False
        for old_version in ["202501", "202502", "202503"]:
            old_doc = await repo.get_by_user_group(
                user_id, group_id, version_range=(old_version, old_version)
            )
            assert old_doc is not None
            assert old_doc.is_latest == False
            logger.info("✅ 验证旧版本 %s 的 is_latest=False", old_version)

        # 测试幂等性：再次调用 ensure_latest
        result2 = await repo.ensure_latest(user_id, group_id)
        assert result2 is True
        logger.info("✅ ensure_latest 幂等性验证成功")

        # 清理测试数据
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试 ensure_latest 方法失败: %s", e)
        raise

    logger.info("✅ ensure_latest 方法测试完成")


async def test_batch_query_with_only_latest():
    """测试批量查询的 only_latest 功能"""
    logger.info("开始测试批量查询的 only_latest 功能...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_batch_user"
    group_id = "test_batch_group"

    try:
        # 创建多个用户在同一个群组的多个版本
        user_ids = [f"{user_id}_{i}" for i in range(1, 4)]

        # 先清理
        for uid in user_ids:
            await repo.delete_by_user_group(uid, group_id)
        logger.info("✅ 清理已存在的测试数据")

        # 为每个用户创建多个版本
        for uid in user_ids:
            for version in ["202501", "202502", "202503"]:
                data = {"version": version, "user_name": f"{uid}_{version}"}
                await repo.upsert_by_user_group(uid, group_id, data)

        logger.info("✅ 创建了 3 个用户在同一群组的 3 个版本")

        # 测试 get_by_user_ids with only_latest=True（默认）
        latest_results = await repo.get_by_user_ids(
            user_ids, group_id=group_id, only_latest=True
        )
        assert len(latest_results) == 3

        for result in latest_results:
            assert result.version == "202503"
            assert result.is_latest == True

        logger.info("✅ get_by_user_ids only_latest=True 成功，返回 3 个最新版本")

        # 测试 get_by_user_ids with only_latest=False（返回所有版本）
        all_results = await repo.get_by_user_ids(
            user_ids, group_id=group_id, only_latest=False
        )
        assert len(all_results) == 9  # 3个用户 * 3个版本
        logger.info("✅ get_by_user_ids only_latest=False 成功，返回 9 个版本")

        # 测试 get_by_group_id with only_latest=True
        group_latest = await repo.get_by_group_id(group_id, only_latest=True)
        assert len(group_latest) == 3  # 3个用户的最新版本
        logger.info("✅ get_by_group_id only_latest=True 成功，返回 3 个用户的最新版本")

        # 测试 get_by_group_id with only_latest=False
        group_all = await repo.get_by_group_id(group_id, only_latest=False)
        assert len(group_all) == 9  # 所有版本
        logger.info("✅ get_by_group_id only_latest=False 成功，返回所有 9 个版本")

        # 清理测试数据
        for uid in user_ids:
            await repo.delete_by_user_group(uid, group_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试批量查询 only_latest 功能失败: %s", e)
        raise

    logger.info("✅ 批量查询 only_latest 功能测试完成")


async def test_get_profile_method():
    """测试 get_profile 方法"""
    logger.info("开始测试 get_profile 方法...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_user_profile_005"
    group_id = "test_group_profile_005"

    try:
        # 先清理
        await repo.delete_by_user_group(user_id, group_id)

        # 创建包含完整 profile 字段的记录
        profile_data = {
            "version": "v1",
            "user_name": "测试用户",
            "hard_skills": [
                {"value": "Python", "level": "高级", "evidences": ["conv_001"]}
            ],
            "soft_skills": [
                {"value": "沟通", "level": "优秀", "evidences": ["conv_002"]}
            ],
            "personality": [{"value": "外向", "evidences": ["conv_003"]}],
            "interests": [{"value": "编程", "evidences": ["conv_004"]}],
            "user_goal": [{"value": "成为技术专家", "evidences": ["conv_005"]}],
        }

        result = await repo.upsert_by_user_group(user_id, group_id, profile_data)
        assert result is not None
        logger.info("✅ 创建包含完整 profile 字段的记录成功")

        # 测试 get_profile 方法
        profile = repo.get_profile(result)
        assert profile is not None
        assert "hard_skills" in profile
        assert "soft_skills" in profile
        assert "personality" in profile
        assert "interests" in profile
        assert "user_goal" in profile
        assert "work_responsibility" in profile
        assert "working_habit_preference" in profile
        logger.info("✅ get_profile 方法测试成功，包含所有字段")

        # 清理
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试 get_profile 方法失败: %s", e)
        raise

    logger.info("✅ get_profile 方法测试完成")


async def test_create_without_version_should_fail():
    """测试创建时不提供 version 应该失败"""
    logger.info("开始测试创建时不提供 version 应该失败...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_no_version_006"
    group_id = "test_no_version_006"

    try:
        # 先清理
        await repo.delete_by_user_group(user_id, group_id)

        # 尝试创建不带 version 的记录
        data_without_version = {
            "user_name": "无版本用户",
            "personality": [{"value": "这应该失败", "evidences": ["test"]}],
        }

        try:
            await repo.upsert_by_user_group(user_id, group_id, data_without_version)
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


async def test_batch_get_by_user_groups():
    """测试批量获取群组用户档案功能"""
    logger.info("开始测试批量获取群组用户档案功能...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)

    # 准备测试数据
    test_data = [
        ("batch_user_001", "batch_group_001", "赵六"),
        ("batch_user_002", "batch_group_001", "钱七"),
        ("batch_user_003", "batch_group_002", "孙八"),
        ("batch_user_004", "batch_group_002", "李九"),
        ("batch_user_005", "batch_group_003", "周十"),
    ]

    try:
        # 先清理可能存在的测试数据
        for user_id, group_id, _ in test_data:
            await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ 清理已存在的测试数据")

        # 创建测试数据，每个用户创建多个版本
        for user_id, group_id, user_name in test_data:
            # 创建旧版本
            old_data = {
                "version": "v1",
                "user_name": f"{user_name}_v1",
                "personality": [{"value": "旧性格", "evidences": ["conv_old"]}],
            }
            await repo.upsert_by_user_group(user_id, group_id, old_data)

            # 创建最新版本
            new_data = {
                "version": "v2",
                "user_name": f"{user_name}_v2",
                "personality": [{"value": "新性格", "evidences": ["conv_new"]}],
                "group_importance_evidence": {
                    "evidence_list": [
                        {"speak_count": 10, "refer_count": 5, "conversation_count": 20}
                    ]
                },
            }
            await repo.upsert_by_user_group(user_id, group_id, new_data)

        logger.info("✅ 创建了 5 个用户的测试数据（每个用户 2 个版本）")

        # 测试 1: 批量获取所有用户档案（应该返回最新版本）
        user_group_pairs = [
            ("batch_user_001", "batch_group_001"),
            ("batch_user_002", "batch_group_001"),
            ("batch_user_003", "batch_group_002"),
            ("batch_user_004", "batch_group_002"),
            ("batch_user_005", "batch_group_003"),
        ]

        results = await repo.batch_get_by_user_groups(user_group_pairs)

        assert len(results) == 5, f"应该返回 5 个结果，实际返回 {len(results)} 个"
        logger.info("✅ 批量获取返回了 5 个结果")

        # 验证每个结果都是最新版本
        for (user_id, group_id), profile in results.items():
            assert (
                profile is not None
            ), f"用户 {user_id} 在群组 {group_id} 的档案不应为 None"
            assert (
                profile.version == "v2"
            ), f"应该返回最新版本 v2，实际返回 {profile.version}"
            assert profile.user_id == user_id
            assert profile.group_id == group_id
            assert profile.user_name.endswith("_v2"), "应该返回最新版本的用户名"
            logger.info(
                "✅ 验证 user_id=%s, group_id=%s: version=%s, user_name=%s",
                user_id,
                group_id,
                profile.version,
                profile.user_name,
            )

        # 测试 2: 包含不存在的用户-群组对
        pairs_with_nonexist = user_group_pairs + [
            ("nonexist_user", "nonexist_group"),
            ("batch_user_001", "nonexist_group"),
        ]

        results_with_none = await repo.batch_get_by_user_groups(pairs_with_nonexist)
        assert len(results_with_none) == 7, "应该返回 7 个结果（包括不存在的）"
        assert results_with_none[("nonexist_user", "nonexist_group")] is None
        assert results_with_none[("batch_user_001", "nonexist_group")] is None
        logger.info("✅ 正确处理不存在的用户-群组对，返回 None")

        # 测试 3: 测试去重功能
        duplicate_pairs = user_group_pairs + user_group_pairs[:2]  # 重复前两个
        results_dedup = await repo.batch_get_by_user_groups(duplicate_pairs)
        assert len(results_dedup) == 5, "去重后应该仍然是 5 个结果"
        logger.info("✅ 去重功能正常工作")

        # 测试 4: 空列表
        empty_results = await repo.batch_get_by_user_groups([])
        assert len(empty_results) == 0, "空列表应该返回空字典"
        logger.info("✅ 空列表返回空字典")

        # 测试 5: 验证 group_importance_evidence 字段
        user_001_profile = results[("batch_user_001", "batch_group_001")]
        assert hasattr(user_001_profile, "group_importance_evidence")
        assert user_001_profile.group_importance_evidence is not None
        assert "evidence_list" in user_001_profile.group_importance_evidence
        logger.info("✅ group_importance_evidence 字段正确获取")

        # 清理测试数据
        for user_id, group_id, _ in test_data:
            await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ 清理测试数据成功")

    except Exception as e:
        logger.error("❌ 测试批量获取群组用户档案功能失败: %s", e)
        # 确保清理
        for user_id, group_id, _ in test_data:
            try:
                await repo.delete_by_user_group(user_id, group_id)
            except:
                pass
        raise

    logger.info("✅ 批量获取群组用户档案功能测试完成")


async def run_all_tests():
    """运行所有测试"""
    logger.info("🚀 开始运行 GroupUserProfileMemory 所有测试...")

    try:
        await test_basic_crud_operations()
        await test_version_management()
        await test_ensure_latest()
        await test_batch_query_with_only_latest()
        await test_get_profile_method()
        await test_create_without_version_should_fail()
        await test_batch_get_by_user_groups()
        logger.info("✅ 所有测试完成")
    except Exception as e:
        logger.error("❌ 测试过程中出现错误: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
