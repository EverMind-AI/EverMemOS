#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Memories 数据迁移脚本

功能1: 将 core_memories 和 group_core_profile_memory 表中每条记录的 user_goal 值复制到 work_responsibility 字段，
       并将 user_goal 字段置为 null。

功能2: 将 core_memories 和 group_core_profile_memory 表中 soft_skills 和 hard_skills 字段里的 "value" 键重命名为 "value" 键。

功能选择:
    1. 仅执行 user_goal 数据迁移
    2. 仅执行技能字段重命名
    3. 执行所有功能
    4. 退出

使用方法：
    python src/bootstrap.py tests/migrate_user_goal_to_work_responsibility.py
"""

import asyncio
from typing import List, Optional, Dict, Any

# 导入依赖注入相关模块
from core.di.utils import get_bean_by_type, get_bean
from core.observation.logger import get_logger

# 导入 MongoDB 相关模块
from infra_layer.adapters.out.persistence.document.memory.core_memory import CoreMemory
from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
    CoreMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.group_user_profile_memory import (
    GroupUserProfileMemory,
)
from infra_layer.adapters.out.persistence.repository.group_user_profile_memory_raw_repository import (
    GroupUserProfileMemoryRawRepository,
)

# 获取日志记录器
logger = get_logger(__name__)


async def get_core_memory_repository():
    """获取 CoreMemory 仓库实例"""
    try:
        # 通过类型获取Bean（推荐）
        core_memory_repo = get_bean_by_type(CoreMemoryRawRepository)
        logger.info(f"✅ 成功获取 CoreMemoryRawRepository: {type(core_memory_repo)}")
        return core_memory_repo
    except Exception as e:
        logger.error(f"❌ 获取 CoreMemoryRawRepository 失败: {e}")
        raise


async def get_group_user_profile_memory_repository():
    """获取 GroupUserProfileMemory 仓库实例"""
    try:
        # 通过类型获取Bean（推荐）
        group_repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
        logger.info(
            f"✅ 成功获取 GroupUserProfileMemoryRawRepository: {type(group_repo)}"
        )
        return group_repo
    except Exception as e:
        logger.error(f"❌ 获取 GroupUserProfileMemoryRawRepository 失败: {e}")
        raise


async def preview_migration_data(repo: CoreMemoryRawRepository):
    """预览需要迁移的数据"""
    print("\n" + "=" * 60)
    print("📊 预览迁移数据")
    print("=" * 60)

    try:
        # 分批查询数据，避免一次性加载所有数据到内存
        batch_size = 2000
        all_records = []
        total_count = 0
        batch_num = 0

        print("🔍 正在分批查询数据...")

        while True:
            batch_num += 1
            print(f"📦 查询批次 {batch_num}...")

            # 分批查询，使用排序确保数据一致性，使用 skip 和 limit
            batch_records = (
                await CoreMemory.find({"user_goal": {"$ne": None, "$exists": True}})
                .sort([("_id", 1)])
                .skip(total_count)
                .limit(batch_size)
                .to_list()
            )

            if not batch_records:
                break

            all_records.extend(batch_records)
            total_count += len(batch_records)
            print(f"   批次 {batch_num} 找到 {len(batch_records)} 条记录")

            # 如果返回的记录数少于批次大小，说明已经查询完毕
            if len(batch_records) < batch_size:
                break

        print(f"📈 总共找到 {total_count} 条包含 user_goal 数据的记录")

        if total_count > 0:
            print("\n📋 前5条记录预览:")
            for i, record in enumerate(all_records[:5]):
                print(f"  {i+1}. User ID: {record.user_id}")
                print(f"     user_goal: {record.user_goal}")
                print(f"     work_responsibility: {record.work_responsibility}")
                print()

        return all_records

    except Exception as e:
        logger.error(f"❌ 预览数据失败: {e}")
        raise


async def migrate_user_goal_to_work_responsibility(
    repo: CoreMemoryRawRepository, records: List[CoreMemory]
):
    """执行数据迁移：将 user_goal 复制到 work_responsibility，并将 user_goal 置为 null"""
    print("\n" + "=" * 60)
    print("🔄 开始数据迁移")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for record in records:
        try:
            # 检查是否有 user_goal 数据
            if not record.user_goal:
                logger.warning(f"⚠️  记录 {record.user_id} 的 user_goal 为空，跳过")
                continue

            # 直接修改记录属性
            record.work_responsibility = (
                record.user_goal
            )  # 复制 user_goal 到 work_responsibility
            record.user_goal = None  # 将 user_goal 置为 null

            # 保存更新后的记录
            await record.save()

            # 更新成功
            success_count += 1
            logger.info(f"✅ 成功迁移记录 {record.user_id}")
            print(f"✅ 迁移成功: {record.user_id}")

        except Exception as e:
            error_count += 1
            logger.error(f"❌ 迁移记录 {record.user_id} 时出错: {e}")
            print(f"❌ 迁移出错: {record.user_id} - {e}")

    print(f"\n📊 迁移完成统计:")
    print(f"   ✅ 成功: {success_count} 条")
    print(f"   ❌ 失败: {error_count} 条")
    print(f"   📈 总计: {len(records)} 条")

    return success_count, error_count


async def verify_migration_results(repo: CoreMemoryRawRepository):
    """验证迁移结果"""
    print("\n" + "=" * 60)
    print("🔍 验证迁移结果")
    print("=" * 60)

    try:
        # 检查是否还有 user_goal 不为 null 的记录
        remaining_user_goals = await CoreMemory.find(
            {"user_goal": {"$ne": None}}
        ).to_list()

        # 检查有多少记录的 work_responsibility 有数据
        records_with_work_responsibility = await CoreMemory.find(
            {"work_responsibility": {"$ne": None, "$exists": True}}
        ).to_list()

        print(f"📊 验证结果:")
        print(f"   剩余 user_goal 不为 null 的记录: {len(remaining_user_goals)}")
        print(
            f"   work_responsibility 有数据的记录: {len(records_with_work_responsibility)}"
        )

        if len(remaining_user_goals) == 0:
            print("✅ 所有 user_goal 字段已成功置为 null")
        else:
            print("⚠️  仍有部分 user_goal 字段未置为 null")
            for record in remaining_user_goals[:3]:  # 显示前3条
                print(f"   - {record.user_id}: {record.user_goal}")

        return len(remaining_user_goals) == 0

    except Exception as e:
        logger.error(f"❌ 验证迁移结果失败: {e}")
        raise


async def preview_group_migration_data():
    """预览群组用户档案记忆需要迁移的数据"""
    print("\n" + "=" * 60)
    print("📊 预览群组用户档案记忆迁移数据")
    print("=" * 60)

    try:
        # 分批查询数据，避免一次性加载所有数据到内存
        batch_size = 2000
        all_records = []
        total_count = 0
        batch_num = 0

        print("🔍 正在分批查询群组数据...")

        while True:
            batch_num += 1
            print(f"📦 查询批次 {batch_num}...")

            # 分批查询，使用排序确保数据一致性，使用 skip 和 limit
            batch_records = (
                await GroupUserProfileMemory.find(
                    {"user_goal": {"$ne": None, "$exists": True}}
                )
                .sort([("_id", 1)])
                .skip(total_count)
                .limit(batch_size)
                .to_list()
            )

            if not batch_records:
                break

            all_records.extend(batch_records)
            total_count += len(batch_records)
            print(f"   批次 {batch_num} 找到 {len(batch_records)} 条记录")

            # 如果返回的记录数少于批次大小，说明已经查询完毕
            if len(batch_records) < batch_size:
                break

        print(f"📈 总共找到 {total_count} 条包含 user_goal 数据的群组用户档案记录")

        if total_count > 0:
            print("\n📋 前5条记录预览:")
            for i, record in enumerate(all_records[:5]):
                print(
                    f"  {i+1}. User ID: {record.user_id}, Group ID: {record.group_id}"
                )
                print(f"     user_goal: {record.user_goal}")
                print(f"     work_responsibility: {record.work_responsibility}")
                print()

        return all_records

    except Exception as e:
        logger.error(f"❌ 预览群组数据失败: {e}")
        raise


async def migrate_group_user_goal_to_work_responsibility(
    records: List[GroupUserProfileMemory],
):
    """执行群组用户档案记忆数据迁移：将 user_goal 复制到 work_responsibility，并将 user_goal 置为 null"""
    print("\n" + "=" * 60)
    print("🔄 开始群组用户档案记忆数据迁移")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for record in records:
        try:
            # 检查是否有 user_goal 数据
            if not record.user_goal:
                logger.warning(
                    f"⚠️  群组记录 {record.user_id}-{record.group_id} 的 user_goal 为空，跳过"
                )
                continue

            # 直接修改记录属性
            record.work_responsibility = (
                record.user_goal
            )  # 复制 user_goal 到 work_responsibility
            record.user_goal = None  # 将 user_goal 置为 null

            # 保存更新后的记录
            await record.save()

            # 更新成功
            success_count += 1
            logger.info(f"✅ 成功迁移群组记录 {record.user_id}-{record.group_id}")
            print(f"✅ 群组迁移成功: {record.user_id}-{record.group_id}")

        except Exception as e:
            error_count += 1
            logger.error(
                f"❌ 迁移群组记录 {record.user_id}-{record.group_id} 时出错: {e}"
            )
            print(f"❌ 群组迁移出错: {record.user_id}-{record.group_id} - {e}")

    print(f"\n📊 群组迁移完成统计:")
    print(f"   ✅ 成功: {success_count} 条")
    print(f"   ❌ 失败: {error_count} 条")
    print(f"   📈 总计: {len(records)} 条")

    return success_count, error_count


async def verify_group_migration_results():
    """验证群组用户档案记忆迁移结果"""
    print("\n" + "=" * 60)
    print("🔍 验证群组用户档案记忆迁移结果")
    print("=" * 60)

    try:
        # 检查是否还有 user_goal 不为 null 的记录
        remaining_user_goals = await GroupUserProfileMemory.find(
            {"user_goal": {"$ne": None}}
        ).to_list()

        # 检查有多少记录的 work_responsibility 有数据
        records_with_work_responsibility = await GroupUserProfileMemory.find(
            {"work_responsibility": {"$ne": None, "$exists": True}}
        ).to_list()

        print(f"📊 群组验证结果:")
        print(f"   剩余 user_goal 不为 null 的记录: {len(remaining_user_goals)}")
        print(
            f"   work_responsibility 有数据的记录: {len(records_with_work_responsibility)}"
        )

        if len(remaining_user_goals) == 0:
            print("✅ 所有群组 user_goal 字段已成功置为 null")
        else:
            print("⚠️  仍有部分群组 user_goal 字段未置为 null")
            for record in remaining_user_goals[:3]:  # 显示前3条
                print(f"   - {record.user_id}-{record.group_id}: {record.user_goal}")

        return len(remaining_user_goals) == 0

    except Exception as e:
        logger.error(f"❌ 验证群组迁移结果失败: {e}")
        raise


async def preview_skills_rename_data():
    """预览需要重命名技能字段的数据"""
    print("\n" + "=" * 60)
    print("📊 预览技能字段重命名数据")
    print("=" * 60)

    try:
        # 分批查询 CoreMemory 中包含 skill 字段的记录
        print("🔍 正在分批查询 CoreMemory 技能字段数据...")
        core_batch_size = 2000
        core_all_records = []
        core_total_count = 0
        core_batch_num = 0

        while True:
            core_batch_num += 1
            print(f"📦 查询 CoreMemory 批次 {core_batch_num}...")

            batch_records = (
                await CoreMemory.find(
                    {
                        "$or": [
                            {"soft_skills.skill": {"$exists": True}},
                            {"hard_skills.skill": {"$exists": True}},
                        ]
                    }
                )
                .sort([("_id", 1)])
                .skip(core_total_count)
                .limit(core_batch_size)
                .to_list()
            )

            if not batch_records:
                break

            core_all_records.extend(batch_records)
            core_total_count += len(batch_records)
            print(f"   批次 {core_batch_num} 找到 {len(batch_records)} 条记录")

            if len(batch_records) < core_batch_size:
                break

        # 分批查询 GroupUserProfileMemory 中包含 skill 字段的记录
        print("🔍 正在分批查询 GroupUserProfileMemory 技能字段数据...")
        group_batch_size = 2000
        group_all_records = []
        group_total_count = 0
        group_batch_num = 0

        while True:
            group_batch_num += 1
            print(f"📦 查询 GroupUserProfileMemory 批次 {group_batch_num}...")

            batch_records = (
                await GroupUserProfileMemory.find(
                    {
                        "$or": [
                            {"soft_skills.skill": {"$exists": True}},
                            {"hard_skills.skill": {"$exists": True}},
                        ]
                    }
                )
                .sort([("_id", 1)])
                .skip(group_total_count)
                .limit(group_batch_size)
                .to_list()
            )

            if not batch_records:
                break

            group_all_records.extend(batch_records)
            group_total_count += len(batch_records)
            print(f"   批次 {group_batch_num} 找到 {len(batch_records)} 条记录")

            if len(batch_records) < group_batch_size:
                break

        print(f"📈 CoreMemory 表总共找到 {core_total_count} 条包含 skill 字段的记录")
        print(
            f"📈 GroupUserProfileMemory 表总共找到 {group_total_count} 条包含 skill 字段的记录"
        )

        if core_all_records:
            print("\n📋 CoreMemory 前3条记录预览:")
            for i, record in enumerate(core_all_records[:3]):
                print(f"  {i+1}. User ID: {record.user_id}")
                if hasattr(record, 'soft_skills') and record.soft_skills:
                    print(f"     soft_skills: {record.soft_skills}")
                if hasattr(record, 'hard_skills') and record.hard_skills:
                    print(f"     hard_skills: {record.hard_skills}")
                print()

        if group_all_records:
            print("\n📋 GroupUserProfileMemory 前3条记录预览:")
            for i, record in enumerate(group_all_records[:3]):
                print(
                    f"  {i+1}. User ID: {record.user_id}, Group ID: {record.group_id}"
                )
                if hasattr(record, 'soft_skills') and record.soft_skills:
                    print(f"     soft_skills: {record.soft_skills}")
                if hasattr(record, 'hard_skills') and record.hard_skills:
                    print(f"     hard_skills: {record.hard_skills}")
                print()

        return core_all_records, group_all_records

    except Exception as e:
        logger.error(f"❌ 预览技能字段数据失败: {e}")
        raise


async def rename_skill_to_value_in_core_memory(records: List[CoreMemory]):
    """重命名 CoreMemory 中技能字段的 skill 键为 value"""
    print("\n" + "=" * 60)
    print("🔄 开始重命名 CoreMemory 技能字段")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for record in records:
        try:
            updated = False

            # 处理 soft_skills 字段
            if hasattr(record, 'soft_skills') and record.soft_skills:
                if isinstance(record.soft_skills, list):
                    for skill_item in record.soft_skills:
                        if isinstance(skill_item, dict) and 'skill' in skill_item:
                            skill_item['value'] = skill_item.pop('skill')
                            updated = True
                elif (
                    isinstance(record.soft_skills, dict)
                    and 'skill' in record.soft_skills
                ):
                    record.soft_skills['value'] = record.soft_skills.pop('skill')
                    updated = True

            # 处理 hard_skills 字段
            if hasattr(record, 'hard_skills') and record.hard_skills:
                if isinstance(record.hard_skills, list):
                    for skill_item in record.hard_skills:
                        if isinstance(skill_item, dict) and 'skill' in skill_item:
                            skill_item['value'] = skill_item.pop('skill')
                            updated = True
                elif (
                    isinstance(record.hard_skills, dict)
                    and 'skill' in record.hard_skills
                ):
                    record.hard_skills['value'] = record.hard_skills.pop('skill')
                    updated = True

            if updated:
                # 保存更新后的记录
                await record.save()
                success_count += 1
                logger.info(
                    f"✅ 成功重命名 CoreMemory 记录 {record.user_id} 的技能字段"
                )
                print(f"✅ 重命名成功: {record.user_id}")
            else:
                logger.warning(
                    f"⚠️  CoreMemory 记录 {record.user_id} 没有需要重命名的技能字段"
                )

        except Exception as e:
            error_count += 1
            logger.error(f"❌ 重命名 CoreMemory 记录 {record.user_id} 时出错: {e}")
            print(f"❌ 重命名出错: {record.user_id} - {e}")

    print(f"\n📊 CoreMemory 重命名完成统计:")
    print(f"   ✅ 成功: {success_count} 条")
    print(f"   ❌ 失败: {error_count} 条")
    print(f"   📈 总计: {len(records)} 条")

    return success_count, error_count


async def rename_skill_to_value_in_group_memory(records: List[GroupUserProfileMemory]):
    """重命名 GroupUserProfileMemory 中技能字段的 skill 键为 value"""
    print("\n" + "=" * 60)
    print("🔄 开始重命名 GroupUserProfileMemory 技能字段")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for record in records:
        try:
            updated = False

            # 处理 soft_skills 字段
            if hasattr(record, 'soft_skills') and record.soft_skills:
                if isinstance(record.soft_skills, list):
                    for skill_item in record.soft_skills:
                        if isinstance(skill_item, dict) and 'skill' in skill_item:
                            skill_item['value'] = skill_item.pop('skill')
                            updated = True
                elif (
                    isinstance(record.soft_skills, dict)
                    and 'skill' in record.soft_skills
                ):
                    record.soft_skills['value'] = record.soft_skills.pop('skill')
                    updated = True

            # 处理 hard_skills 字段
            if hasattr(record, 'hard_skills') and record.hard_skills:
                if isinstance(record.hard_skills, list):
                    for skill_item in record.hard_skills:
                        if isinstance(skill_item, dict) and 'skill' in skill_item:
                            skill_item['value'] = skill_item.pop('skill')
                            updated = True
                elif (
                    isinstance(record.hard_skills, dict)
                    and 'skill' in record.hard_skills
                ):
                    record.hard_skills['value'] = record.hard_skills.pop('skill')
                    updated = True

            if updated:
                # 保存更新后的记录
                await record.save()
                success_count += 1
                logger.info(
                    f"✅ 成功重命名 GroupUserProfileMemory 记录 {record.user_id}-{record.group_id} 的技能字段"
                )
                print(f"✅ 群组重命名成功: {record.user_id}-{record.group_id}")
            else:
                logger.warning(
                    f"⚠️  GroupUserProfileMemory 记录 {record.user_id}-{record.group_id} 没有需要重命名的技能字段"
                )

        except Exception as e:
            error_count += 1
            logger.error(
                f"❌ 重命名 GroupUserProfileMemory 记录 {record.user_id}-{record.group_id} 时出错: {e}"
            )
            print(f"❌ 群组重命名出错: {record.user_id}-{record.group_id} - {e}")

    print(f"\n📊 GroupUserProfileMemory 重命名完成统计:")
    print(f"   ✅ 成功: {success_count} 条")
    print(f"   ❌ 失败: {error_count} 条")
    print(f"   📈 总计: {len(records)} 条")

    return success_count, error_count


async def verify_skills_rename_results():
    """验证技能字段重命名结果"""
    print("\n" + "=" * 60)
    print("🔍 验证技能字段重命名结果")
    print("=" * 60)

    try:
        # 检查 CoreMemory 中是否还有 skill 字段的记录
        core_remaining_skill = await CoreMemory.find(
            {
                "$or": [
                    {"soft_skills.skill": {"$exists": True}},
                    {"hard_skills.skill": {"$exists": True}},
                ]
            }
        ).to_list()

        # 检查 GroupUserProfileMemory 中是否还有 skill 字段的记录
        group_remaining_skill = await GroupUserProfileMemory.find(
            {
                "$or": [
                    {"soft_skills.skill": {"$exists": True}},
                    {"hard_skills.skill": {"$exists": True}},
                ]
            }
        ).to_list()

        # 检查有多少记录的 value 字段有数据
        core_records_with_value = await CoreMemory.find(
            {
                "$or": [
                    {"soft_skills.value": {"$exists": True}},
                    {"hard_skills.value": {"$exists": True}},
                ]
            }
        ).to_list()

        group_records_with_value = await GroupUserProfileMemory.find(
            {
                "$or": [
                    {"soft_skills.value": {"$exists": True}},
                    {"hard_skills.value": {"$exists": True}},
                ]
            }
        ).to_list()

        print(f"📊 技能字段重命名验证结果:")
        print(f"   CoreMemory 剩余 skill 字段记录: {len(core_remaining_skill)}")
        print(
            f"   GroupUserProfileMemory 剩余 skill 字段记录: {len(group_remaining_skill)}"
        )
        print(f"   CoreMemory 有 value 字段记录: {len(core_records_with_value)}")
        print(
            f"   GroupUserProfileMemory 有 value 字段记录: {len(group_records_with_value)}"
        )

        if len(core_remaining_skill) == 0 and len(group_remaining_skill) == 0:
            print("✅ 所有技能字段的 skill 键已成功重命名为 value")
        else:
            print("⚠️  仍有部分技能字段的 skill 键未重命名")
            if core_remaining_skill:
                print("   CoreMemory 剩余记录:")
                for record in core_remaining_skill[:2]:
                    print(f"   - {record.user_id}")
            if group_remaining_skill:
                print("   GroupUserProfileMemory 剩余记录:")
                for record in group_remaining_skill[:2]:
                    print(f"   - {record.user_id}-{record.group_id}")

        return len(core_remaining_skill) == 0 and len(group_remaining_skill) == 0

    except Exception as e:
        logger.error(f"❌ 验证技能字段重命名结果失败: {e}")
        raise


async def main():
    """主函数"""
    print("🚀 Core Memories 和 Group User Profile Memory 数据迁移脚本启动")
    print(f"📝 当前脚本: {__file__}")
    print("📋 涉及表: core_memories, group_core_profile_memory")

    # 功能选择菜单
    print("\n" + "=" * 60)
    print("📋 请选择要执行的功能:")
    print("=" * 60)
    print("1. user_goal 数据迁移到 work_responsibility 字段")
    print("2. 技能字段重命名 (skill -> value)")
    print("3. 执行所有功能")
    print("4. 退出")
    print("=" * 60)

    try:
        # 获取用户选择
        while True:
            try:
                choice = input("\n请输入选择 (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    break
                else:
                    print("❌ 无效选择，请输入 1-4 之间的数字")
            except KeyboardInterrupt:
                print("\n\n👋 用户取消操作，退出程序")
                return
            except EOFError:
                print("\n\n👋 输入结束，退出程序")
                return

        # 根据选择执行相应功能
        if choice == '4':
            print("👋 退出程序")
            return

        print(f"\n✅ 已选择功能: {choice}")

        # 执行选中的功能
        if choice == '1':
            await execute_user_goal_migration()
        elif choice == '2':
            await execute_skills_rename()
        elif choice == '3':
            await execute_all_functions()

    except Exception as e:
        logger.error(f"❌ 迁移脚本执行失败: {e}")
        print(f"\n❌ 迁移失败: {e}")
        raise


async def execute_user_goal_migration():
    """执行 user_goal 数据迁移功能"""
    print("\n" + "=" * 80)
    print("🏠 开始执行 user_goal 数据迁移")
    print("=" * 80)

    try:
        # ==================== 1. Core Memories 迁移 ====================
        print("\n" + "=" * 80)
        print("🏠 开始处理 Core Memories 表")
        print("=" * 80)

        # 1.1 获取仓库实例
        core_memory_repo = await get_core_memory_repository()

        # 1.2 预览需要迁移的数据
        core_records_to_migrate = await preview_migration_data(core_memory_repo)

        if core_records_to_migrate:
            print(f"\n⚠️  即将迁移 {len(core_records_to_migrate)} 条 Core Memories 记录")
            print("   此操作将:")
            print("   - 将 user_goal 的值复制到 work_responsibility")
            print("   - 将 user_goal 字段置为 null")
            print("   - 此操作不可逆，请确认继续")

            # 1.3 执行 Core Memories 数据迁移
            core_success_count, core_error_count = (
                await migrate_user_goal_to_work_responsibility(
                    core_memory_repo, core_records_to_migrate
                )
            )

            # 1.4 验证 Core Memories 迁移结果
            core_migration_success = await verify_migration_results(core_memory_repo)
        else:
            print("ℹ️  Core Memories 表没有需要迁移的数据")
            core_success_count, core_error_count = 0, 0
            core_migration_success = True

        # ==================== 2. Group User Profile Memory 迁移 ====================
        print("\n" + "=" * 80)
        print("👥 开始处理 Group User Profile Memory 表")
        print("=" * 80)

        # 2.1 获取群组仓库实例
        group_repo = await get_group_user_profile_memory_repository()

        # 2.2 预览需要迁移的群组数据
        group_records_to_migrate = await preview_group_migration_data()

        if group_records_to_migrate:
            print(
                f"\n⚠️  即将迁移 {len(group_records_to_migrate)} 条 Group User Profile Memory 记录"
            )
            print("   此操作将:")
            print("   - 将 user_goal 的值复制到 work_responsibility")
            print("   - 将 user_goal 字段置为 null")
            print("   - 此操作不可逆，请确认继续")

            # 2.3 执行群组数据迁移
            group_success_count, group_error_count = (
                await migrate_group_user_goal_to_work_responsibility(
                    group_records_to_migrate
                )
            )

            # 2.4 验证群组迁移结果
            group_migration_success = await verify_group_migration_results()
        else:
            print("ℹ️  Group User Profile Memory 表没有需要迁移的数据")
            group_success_count, group_error_count = 0, 0
            group_migration_success = True

        # ==================== 3. 总结报告 ====================
        print("\n" + "=" * 80)
        print("📊 user_goal 迁移完成总结报告")
        print("=" * 80)

        print(f"🏠 Core Memories 表:")
        print(f"   ✅ 成功: {core_success_count} 条")
        print(f"   ❌ 失败: {core_error_count} 条")
        print(
            f"   📈 总计: {len(core_records_to_migrate) if core_records_to_migrate else 0} 条"
        )

        print(f"\n👥 Group User Profile Memory 表:")
        print(f"   ✅ 成功: {group_success_count} 条")
        print(f"   ❌ 失败: {group_error_count} 条")
        print(
            f"   📈 总计: {len(group_records_to_migrate) if group_records_to_migrate else 0} 条"
        )

        total_success = core_success_count + group_success_count
        total_error = core_error_count + group_error_count

        print(f"\n🎯 总体结果:")
        print(f"   ✅ 总成功: {total_success} 条")
        print(f"   ❌ 总失败: {total_error} 条")

        if core_migration_success and group_migration_success and total_error == 0:
            print("\n🎉 user_goal 数据迁移完全成功！")
        elif total_success > 0:
            print("\n⚠️  user_goal 数据迁移部分成功，请检查错误日志")
        else:
            print("\n❌ user_goal 数据迁移失败，请检查错误日志")
        print("=" * 80)

    except Exception as e:
        logger.error(f"❌ user_goal 迁移脚本执行失败: {e}")
        print(f"\n❌ user_goal 迁移失败: {e}")
        raise


async def execute_skills_rename():
    """执行技能字段重命名功能"""
    print("\n" + "=" * 80)
    print("🔧 开始执行技能字段重命名")
    print("=" * 80)

    try:
        # ==================== 1. 预览技能字段重命名数据 ====================
        print("\n" + "=" * 80)
        print("🔧 开始处理技能字段重命名")
        print("=" * 80)

        # 1.1 预览需要重命名的技能字段数据
        core_skill_records, group_skill_records = await preview_skills_rename_data()

        # ==================== 2. 处理 CoreMemory 技能字段重命名 ====================
        core_skill_success_count = 0
        core_skill_error_count = 0
        if core_skill_records:
            print(f"\n⚠️  即将重命名 {len(core_skill_records)} 条 CoreMemory 技能字段")
            print("   此操作将:")
            print("   - 将 soft_skills 和 hard_skills 中的 'skill' 键重命名为 'value'")
            print("   - 此操作不可逆，请确认继续")

            core_skill_success_count, core_skill_error_count = (
                await rename_skill_to_value_in_core_memory(core_skill_records)
            )
        else:
            print("ℹ️  CoreMemory 表没有需要重命名的技能字段")

        # ==================== 3. 处理 GroupUserProfileMemory 技能字段重命名 ====================
        group_skill_success_count = 0
        group_skill_error_count = 0
        if group_skill_records:
            print(
                f"\n⚠️  即将重命名 {len(group_skill_records)} 条 GroupUserProfileMemory 技能字段"
            )
            print("   此操作将:")
            print("   - 将 soft_skills 和 hard_skills 中的 'skill' 键重命名为 'value'")
            print("   - 此操作不可逆，请确认继续")

            group_skill_success_count, group_skill_error_count = (
                await rename_skill_to_value_in_group_memory(group_skill_records)
            )
        else:
            print("ℹ️  GroupUserProfileMemory 表没有需要重命名的技能字段")

        # ==================== 4. 验证技能字段重命名结果 ====================
        skills_rename_success = await verify_skills_rename_results()

        # ==================== 5. 总结报告 ====================
        print("\n" + "=" * 80)
        print("📊 技能字段重命名完成总结报告")
        print("=" * 80)

        print(f"🔧 CoreMemory 表:")
        print(f"   ✅ 成功: {core_skill_success_count} 条")
        print(f"   ❌ 失败: {core_skill_error_count} 条")
        print(f"   📈 总计: {len(core_skill_records) if core_skill_records else 0} 条")

        print(f"\n🔧 GroupUserProfileMemory 表:")
        print(f"   ✅ 成功: {group_skill_success_count} 条")
        print(f"   ❌ 失败: {group_skill_error_count} 条")
        print(
            f"   📈 总计: {len(group_skill_records) if group_skill_records else 0} 条"
        )

        total_success = core_skill_success_count + group_skill_success_count
        total_error = core_skill_error_count + group_skill_error_count

        print(f"\n🎯 总体结果:")
        print(f"   ✅ 总成功: {total_success} 条")
        print(f"   ❌ 总失败: {total_error} 条")

        if skills_rename_success and total_error == 0:
            print("\n🎉 技能字段重命名完全成功！")
        elif total_success > 0:
            print("\n⚠️  技能字段重命名部分成功，请检查错误日志")
        else:
            print("\n❌ 技能字段重命名失败，请检查错误日志")
        print("=" * 80)

    except Exception as e:
        logger.error(f"❌ 技能字段重命名脚本执行失败: {e}")
        print(f"\n❌ 技能字段重命名失败: {e}")
        raise


async def execute_all_functions():
    """执行所有功能"""
    print("\n" + "=" * 80)
    print("🚀 开始执行所有功能")
    print("=" * 80)

    try:
        # 执行 user_goal 迁移
        # await execute_user_goal_migration()

        # 执行技能字段重命名
        await execute_skills_rename()

        print("\n🎉 所有功能执行完成！")

    except Exception as e:
        logger.error(f"❌ 执行所有功能失败: {e}")
        print(f"\n❌ 执行所有功能失败: {e}")
        raise


if __name__ == "__main__":
    # 当直接运行此脚本时执行
    # 注意：通过 bootstrap.py 运行时，环境已经初始化完成
    asyncio.run(main())
