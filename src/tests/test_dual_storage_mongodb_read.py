#!/usr/bin/env python3
"""
测试双存储模式下从 MongoDB 读取数据的完整性

本测试完全模仿 sync 脚本的读取方式，验证：
1. 通过 Repository.model.find().to_list() 读取的数据是否包含完整字段
2. 5个集合：episodic_memories, event_log_records, foresight_records, conversation_meta, memcells

预期结果：
- episodic_memories: 应包含 subject, summary, episode
- event_log_records: 应包含 atomic_fact
- foresight_records: 应包含 content/foresight
- conversation_meta: 应包含完整数据
- memcells: 应包含 summary, original_data

这验证了 DualStorageQueryProxy 能正确从 KV-Storage 加载完整数据
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# IMPORTANT: Must setup environment and DI BEFORE importing repositories
from common_utils.load_env import setup_environment
setup_environment(load_env_file_name=".env", check_env_var="MONGODB_HOST")

# Setup all (DI container, etc.) - same as run.py
from application_startup import setup_all
setup_all(load_entrypoints=False)  # Don't load addons for testing

from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)


async def test_episodic_memory_read():
    """
    测试 episodic_memories 集合读取

    完全模仿 milvus_sync_episodic_memory_docs.py 的读取方式
    """
    print("\n" + "="*80)
    print("测试 1: Episodic Memory 读取")
    print("="*80)

    from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
        EpisodicMemoryRawRepository,
    )

    # 获取 Repository
    mongo_repo = get_bean_by_type(EpisodicMemoryRawRepository)

    # 使用和 sync 脚本完全相同的读取方式
    # 注意：为了测试新数据，按 created_at 降序排列（最新的数据在前）
    query = mongo_repo.model.find({}).sort("-created_at")  # Descending order to get newest
    mongo_docs = await query.limit(3).to_list()

    print(f"\n📊 读取到 {len(mongo_docs)} 条文档")

    if mongo_docs:
        print("\n检查第一条文档的字段完整性：")
        doc = mongo_docs[0]

        # 检查关键字段
        fields_to_check = {
            "id": getattr(doc, 'id', None),
            "subject": getattr(doc, 'subject', None),
            "summary": getattr(doc, 'summary', None),
            "episode": getattr(doc, 'episode', None),
            "user_id": getattr(doc, 'user_id', None),
            "group_id": getattr(doc, 'group_id', None),
            "timestamp": getattr(doc, 'timestamp', None),
            "vector": getattr(doc, 'vector', None),
        }

        for field_name, field_value in fields_to_check.items():
            has_value = field_value is not None
            value_preview = ""
            if has_value:
                if field_name == "vector":
                    value_preview = f"(向量长度: {len(field_value)})" if field_value else ""
                elif isinstance(field_value, str) and len(field_value) > 50:
                    value_preview = f"'{field_value[:50]}...'"
                else:
                    value_preview = f"'{field_value}'"

            status = "✅" if has_value else "❌"
            print(f"  {status} {field_name:15s}: {'有值' if has_value else '空值'} {value_preview}")

        # 关键验证
        print("\n🎯 关键验证:")
        if doc.subject and doc.summary and doc.episode:
            print("  ✅ PASS - 包含完整内容字段 (subject, summary, episode)")
        else:
            print("  ❌ FAIL - 缺少内容字段！这说明读取到的是 Lite 数据")

    else:
        print("⚠️  集合为空，无法测试")


async def test_event_log_read():
    """
    测试 event_log_records 集合读取

    完全模仿 sync 脚本的读取方式
    """
    print("\n" + "="*80)
    print("测试 2: Event Log 读取")
    print("="*80)

    from infra_layer.adapters.out.persistence.repository.event_log_record_raw_repository import (
        EventLogRecordRawRepository,
    )

    # 获取 Repository
    mongo_repo = get_bean_by_type(EventLogRecordRawRepository)

    # 使用和 sync 脚本完全相同的读取方式
    # 注意：为了测试新数据，按 created_at 降序排列（最新的数据在前）
    query = mongo_repo.model.find({}).sort("-created_at")  # Descending order to get newest
    mongo_docs = await query.limit(3).to_list()

    print(f"\n📊 读取到 {len(mongo_docs)} 条文档")

    if mongo_docs:
        print("\n检查第一条文档的字段完整性：")
        doc = mongo_docs[0]

        # 检查关键字段
        fields_to_check = {
            "id": getattr(doc, 'id', None),
            "atomic_fact": getattr(doc, 'atomic_fact', None),
            "parent_type": getattr(doc, 'parent_type', None),
            "parent_id": getattr(doc, 'parent_id', None),
            "user_id": getattr(doc, 'user_id', None),
            "group_id": getattr(doc, 'group_id', None),
            "timestamp": getattr(doc, 'timestamp', None),
            "vector": getattr(doc, 'vector', None),
        }

        for field_name, field_value in fields_to_check.items():
            has_value = field_value is not None
            value_preview = ""
            if has_value:
                if field_name == "vector":
                    value_preview = f"(向量长度: {len(field_value)})" if field_value else ""
                elif isinstance(field_value, str) and len(field_value) > 50:
                    value_preview = f"'{field_value[:50]}...'"
                else:
                    value_preview = f"'{field_value}'"

            status = "✅" if has_value else "❌"
            print(f"  {status} {field_name:15s}: {'有值' if has_value else '空值'} {value_preview}")

        # 关键验证
        print("\n🎯 关键验证:")
        if doc.atomic_fact:
            print("  ✅ PASS - 包含完整内容字段 (atomic_fact)")
        else:
            print("  ❌ FAIL - 缺少 atomic_fact 字段！这说明读取到的是 Lite 数据")

    else:
        print("⚠️  集合为空，无法测试")


async def test_foresight_read():
    """
    测试 foresight_records 集合读取

    完全模仿 sync 脚本的读取方式
    """
    print("\n" + "="*80)
    print("测试 3: Foresight 读取")
    print("="*80)

    from infra_layer.adapters.out.persistence.repository.foresight_record_repository import (
        ForesightRecordRawRepository,
    )

    # 获取 Repository
    mongo_repo = get_bean_by_type(ForesightRecordRawRepository)

    # 使用和 sync 脚本完全相同的读取方式
    # 注意：为了测试新数据，按 created_at 降序排列（最新的数据在前）
    query = mongo_repo.model.find({}).sort("-created_at")  # Descending order to get newest
    mongo_docs = await query.limit(3).to_list()

    print(f"\n📊 读取到 {len(mongo_docs)} 条文档")

    if mongo_docs:
        print("\n检查第一条文档的字段完整性：")
        doc = mongo_docs[0]

        # 检查关键字段
        fields_to_check = {
            "id": getattr(doc, 'id', None),
            "content": getattr(doc, 'content', None),
            "evidence": getattr(doc, 'evidence', None),
            "parent_type": getattr(doc, 'parent_type', None),
            "parent_id": getattr(doc, 'parent_id', None),
            "user_id": getattr(doc, 'user_id', None),
            "group_id": getattr(doc, 'group_id', None),
            "start_time": getattr(doc, 'start_time', None),
            "vector": getattr(doc, 'vector', None),
        }

        for field_name, field_value in fields_to_check.items():
            has_value = field_value is not None
            value_preview = ""
            if has_value:
                if field_name == "vector":
                    value_preview = f"(向量长度: {len(field_value)})" if field_value else ""
                elif isinstance(field_value, str) and len(field_value) > 50:
                    value_preview = f"'{field_value[:50]}...'"
                else:
                    value_preview = f"'{field_value}'"

            status = "✅" if has_value else "❌"
            print(f"  {status} {field_name:15s}: {'有值' if has_value else '空值'} {value_preview}")

        # 关键验证
        print("\n🎯 关键验证:")
        if doc.content:
            print("  ✅ PASS - 包含完整内容字段 (content)")
        else:
            print("  ❌ FAIL - 缺少 content 字段！这说明读取到的是 Lite 数据")

    else:
        print("⚠️  集合为空，无法测试")


async def test_conversation_meta_read():
    """
    测试 conversation_meta 集合读取

    完全模仿 sync 脚本的读取方式
    """
    print("\n" + "="*80)
    print("测试 4: Conversation Meta 读取")
    print("="*80)

    from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
        ConversationMetaRawRepository,
    )

    # 获取 Repository
    mongo_repo = get_bean_by_type(ConversationMetaRawRepository)

    # 使用和 sync 脚本完全相同的读取方式
    # 注意：为了测试新数据，按 created_at 降序排列（最新的数据在前）
    query = mongo_repo.model.find({}).sort("-created_at")  # Descending order to get newest
    mongo_docs = await query.limit(3).to_list()

    print(f"\n📊 读取到 {len(mongo_docs)} 条文档")

    if mongo_docs:
        print("\n检查第一条文档的字段完整性：")
        doc = mongo_docs[0]

        # 检查关键字段
        fields_to_check = {
            "id": getattr(doc, 'id', None),
            "group_id": getattr(doc, 'group_id', None),
            "name": getattr(doc, 'name', None),
            "description": getattr(doc, 'description', None),
            "user_details": getattr(doc, 'user_details', None),
            "tags": getattr(doc, 'tags', None),
            "created_at": getattr(doc, 'created_at', None),
        }

        for field_name, field_value in fields_to_check.items():
            has_value = field_value is not None
            value_preview = ""
            if has_value:
                if field_name == "user_details":
                    value_preview = f"(字典长度: {len(field_value)})" if isinstance(field_value, dict) else ""
                elif field_name == "tags":
                    value_preview = f"(列表长度: {len(field_value)})" if isinstance(field_value, list) else ""
                elif isinstance(field_value, str) and len(field_value) > 50:
                    value_preview = f"'{field_value[:50]}...'"
                else:
                    value_preview = f"'{field_value}'"

            status = "✅" if has_value else "❌"
            print(f"  {status} {field_name:20s}: {'有值' if has_value else '空值'} {value_preview}")

        # 关键验证
        print("\n🎯 关键验证:")
        has_description = getattr(doc, 'description', None) is not None and doc.description
        has_user_details = getattr(doc, 'user_details', None) is not None and doc.user_details
        has_tags = getattr(doc, 'tags', None) is not None and doc.tags

        if has_description or has_user_details or has_tags:
            print(f"  ✅ PASS - 包含完整数据字段 (description: {has_description}, user_details: {has_user_details}, tags: {has_tags})")
        else:
            print("  ❌ FAIL - 缺少数据字段！")

    else:
        print("⚠️  集合为空，无法测试")


async def test_memcell_read():
    """
    测试 memcells 集合读取

    完全模仿 sync 脚本的读取方式
    """
    print("\n" + "="*80)
    print("测试 5: MemCell 读取")
    print("="*80)

    from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
        MemCellRawRepository,
    )

    # 获取 Repository
    mongo_repo = get_bean_by_type(MemCellRawRepository)

    # 使用和 sync 脚本完全相同的读取方式
    # 注意：为了测试新数据，按 created_at 降序排列（最新的数据在前）
    query = mongo_repo.model.find({}).sort("-created_at")  # Descending order to get newest
    mongo_docs = await query.limit(3).to_list()

    print(f"\n📊 读取到 {len(mongo_docs)} 条文档")

    if mongo_docs:
        print("\n检查第一条文档的字段完整性：")
        doc = mongo_docs[0]

        # 检查关键字段
        fields_to_check = {
            "id": getattr(doc, 'id', None),
            "user_id": getattr(doc, 'user_id', None),
            "group_id": getattr(doc, 'group_id', None),
            "timestamp": getattr(doc, 'timestamp', None),
            "summary": getattr(doc, 'summary', None),
            "original_data": getattr(doc, 'original_data', None),
            "subject": getattr(doc, 'subject', None),
            "episode": getattr(doc, 'episode', None),
            "participants": getattr(doc, 'participants', None),
        }

        for field_name, field_value in fields_to_check.items():
            has_value = field_value is not None
            value_preview = ""
            if has_value:
                if field_name == "original_data":
                    value_preview = f"(列表长度: {len(field_value)})" if isinstance(field_value, list) else ""
                elif field_name == "participants":
                    value_preview = f"(列表长度: {len(field_value)})" if isinstance(field_value, list) else ""
                elif isinstance(field_value, str) and len(field_value) > 50:
                    value_preview = f"'{field_value[:50]}...'"
                else:
                    value_preview = f"'{field_value}'"

            status = "✅" if has_value else "❌"
            print(f"  {status} {field_name:20s}: {'有值' if has_value else '空值'} {value_preview}")

        # 关键验证
        print("\n🎯 关键验证:")
        has_summary = getattr(doc, 'summary', None) is not None and doc.summary
        has_original_data = getattr(doc, 'original_data', None) is not None and doc.original_data

        if has_summary or has_original_data:
            print(f"  ✅ PASS - 包含完整数据字段 (summary: {has_summary}, original_data: {has_original_data})")
        else:
            print("  ❌ FAIL - 缺少数据字段！")

    else:
        print("⚠️  集合为空，无法测试")


async def main():
    """主测试函数"""
    print("\n" + "🔬" * 40)
    print("双存储模式 MongoDB 读取完整性测试")
    print("模仿 sync 脚本的读取方式验证数据完整性")
    print("🔬" * 40)

    try:
        # 测试所有集合
        await test_episodic_memory_read()
        await test_event_log_read()
        await test_foresight_read()
        await test_conversation_meta_read()
        await test_memcell_read()

        # 最终总结
        print("\n" + "="*80)
        print("测试完成总结")
        print("="*80)
        print("""
如果所有测试都显示 ✅ PASS：
  → DualStorageQueryProxy 正确工作，从 KV-Storage 加载了完整数据
  → Sync 脚本能正确读取完整数据并同步到 Milvus/ES

如果任何测试显示 ❌ FAIL：
  → DualStorageQueryProxy 可能有问题
  → 或者数据是在双存储启用前创建的（只有 Lite 数据）
  → 建议：重新运行 demo 创建新数据后再测试
        """)

    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
