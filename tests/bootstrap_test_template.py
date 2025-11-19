#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap 测试模板

展示如何在脚本中使用依赖注入获取单例对象和调用 MongoDB 仓库。
这个模板可以作为编写测试脚本的参考。

使用方法：
    python src/bootstrap.py tests/bootstrap_test_template.py
"""

import asyncio
from typing import List, Optional

# 导入依赖注入相关模块
from core.di.utils import get_bean_by_type, get_bean
from core.observation.logger import get_logger

# 导入 MongoDB 相关模块
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell,
    DataTypeEnum,
)

# 导入其他可能需要的服务
from memory_layer.memory_manager import MemoryManager

# 获取日志记录器
logger = get_logger(__name__)


async def test_di_container():
    """测试依赖注入容器的使用"""
    print("=" * 60)
    print("🔧 测试依赖注入容器")
    print("=" * 60)

    try:
        # 方式1: 通过类型获取Bean（推荐）
        memcell_repo = get_bean_by_type(MemCellRawRepository)
        print(f"✅ 通过类型获取 MemCellRawRepository: {type(memcell_repo)}")

        # 方式2: 通过名称获取Bean
        memcell_repo_by_name = get_bean("memcell_raw_repository")
        print(f"✅ 通过名称获取 MemCellRawRepository: {type(memcell_repo_by_name)}")

        # 验证是否为同一个单例实例
        is_singleton = memcell_repo is memcell_repo_by_name
        print(f"✅ 单例验证: {is_singleton}")

        # 获取其他服务示例
        return memcell_repo

    except Exception as e:
        logger.error(f"❌ 依赖注入测试失败: {e}")
        raise


async def test_mongodb_repository(repo: MemCellRawRepository):
    """测试 MongoDB 仓库操作"""
    print("\n" + "=" * 60)
    print("🗄️  测试 MongoDB 仓库操作")
    print("=" * 60)

    try:
        # 1. 测试统计总数
        total_count = await repo.count_all()
        print(f"✅ MemCell 总数: {total_count}")

        memcell = await repo.get_by_event_id("evt_20250911_sample_002")
        print(memcell)
        print(f"✅ 仓库集合名称: {repo.get_collection_name()}")
        print(f"✅ 仓库模型名称: {repo.get_model_name()}")

    except Exception as e:
        logger.error(f"❌ MongoDB 仓库测试失败: {e}")
        raise


async def main():
    """主函数"""
    print("🚀 Bootstrap 测试模板启动")
    print(f"📝 当前脚本: {__file__}")

    try:
        # 1. 测试依赖注入
        memcell_repo = await test_di_container()

        # 2. 测试 MongoDB 仓库
        await test_mongodb_repository(memcell_repo)

        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")
        print("=" * 60)

    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        print(f"\n❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    # 当直接运行此脚本时执行
    # 注意：通过 bootstrap.py 运行时，环境已经初始化完成
    asyncio.run(main())
