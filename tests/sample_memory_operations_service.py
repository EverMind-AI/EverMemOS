from core.di import get_bean_by_type, enable_mock_mode, scan_packages
from infra_layer.adapters.out.persistence.repository.semantic_memory_raw_repository import (
    SemanticMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.semantic_memory import (
    SemanticMemory,
)
from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
)
from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
    CoreMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.core_memory import CoreMemory
from infra_layer.adapters.out.persistence.repository.entity_raw_repository import (
    EntityRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.entity import Entity
from infra_layer.adapters.out.persistence.repository.relationship_raw_repository import (
    RelationshipRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.relationship import (
    Relationship,
)
from infra_layer.adapters.out.persistence.repository.behavior_history_raw_repository import (
    BehaviorHistoryRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.behavior_history import (
    BehaviorHistory,
)
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell,
    DataTypeEnum,
    RawData,
    Message,
)
from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
    ConversationStatusRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
    ConversationStatus,
)
from infra_layer.adapters.out.persistence.repository.employee_organization_raw_repository import (
    EmployeeOrganizationRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.employee_organization import (
    EmployeeOrganization,
)
import uuid
from datetime import datetime
import os
import logging

# 配置logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 如果没有处理器，添加一个控制台处理器
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


async def test_semantic_memory_operations():
    """
    测试语义记忆的插入和查询操作

    该函数用于验证 SemanticMemoryRawRepository 的基本功能
    """
    try:
        logger.debug("🧪 开始测试语义记忆操作...")

        # 获取语义记忆仓库
        repository = get_bean_by_type(SemanticMemoryRawRepository)
        if not repository:
            logger.error("❌ 无法获取 SemanticMemoryRawRepository 实例")
            return

        # 测试数据
        test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        test_subject = "测试主题：Python编程"
        test_description = "Python是一种高级编程语言，语法简洁，适合数据科学和Web开发"
        test_links = ["evt_test_001", "raw_data_test_001"]

        logger.debug(f"📝 测试用户ID: {test_user_id}")

        # 1. 测试创建语义记忆
        logger.debug("1️⃣ 测试创建语义记忆...")
        semantic_memory = SemanticMemory(
            user_id=test_user_id,
            subject=test_subject,
            description=test_description,
            link=test_links,
        )

        # 插入到数据库
        await semantic_memory.insert()
        logger.debug(f"✅ 成功创建语义记忆: {test_subject}")

        # 2. 测试根据用户ID查询
        logger.debug("2️⃣ 测试根据用户ID查询...")
        found_memory = await repository.get_by_user_id(test_user_id)
        if found_memory:
            logger.debug(
                f"✅ 查询成功: {found_memory.subject} - {found_memory.description}"
            )
            logger.debug(f"   来源链接: {found_memory.link}")
        else:
            logger.error("❌ 查询失败：未找到语义记忆")
            return

        # 3. 测试更新操作
        logger.debug("3️⃣ 测试更新操作...")
        update_data = {
            "description": "Python是一种功能强大的高级编程语言，语法简洁优雅，广泛应用于数据科学、Web开发和人工智能领域",
            "link": ["evt_test_001", "raw_data_test_001", "evt_test_002"],
        }

        updated_memory = await repository.update_by_user_id(test_user_id, update_data)
        if updated_memory:
            logger.debug(f"✅ 更新成功: {updated_memory.description}")
            logger.debug(f"   更新后的来源链接: {updated_memory.link}")
        else:
            logger.error("❌ 更新失败")

        # 4. 清理测试数据
        # logger.debug("4️⃣ 清理测试数据...")
        # deleted = await repository.delete_by_user_id(test_user_id)
        # if deleted:
        #     logger.debug("✅ 测试数据清理完成")
        # else:
        #     logger.debug("❌ 清理失败")

        logger.debug("🎉 语义记忆操作测试完成！")

    except Exception as e:
        logger.error(f"❌ 语义记忆操作测试失败: {e}")
        import traceback

        traceback.print_exc()


async def test_episodic_memory_operations():
    """
    测试情景记忆的插入和查询操作

    该函数用于验证 EpisodicMemoryRawRepository 的基本功能
    """
    try:
        logger.debug("🧪 开始测试情景记忆操作...")

        # 获取情景记忆仓库
        repository = get_bean_by_type(EpisodicMemoryRawRepository)
        if not repository:
            logger.error("❌ 无法获取 EpisodicMemoryRawRepository 实例")
            return

        # 测试数据
        test_event_id = f"evt_test_{uuid.uuid4().hex[:8]}"
        test_user_id = f"user_test_{uuid.uuid4().hex[:8]}"
        test_group_id = "group_test"
        test_timestamp = int(datetime.now().timestamp())
        test_participants = ["张三", "李四"]
        test_summary = "测试记忆单元：项目进度讨论"
        test_subject = "项目会议"
        test_episode = "在会议室进行了项目进度讨论，确定了下周的开发任务分配，讨论了技术难点和解决方案"
        test_type = "Conversation"
        test_keywords = ["项目", "进度", "会议", "开发"]
        test_linked_entities = ["proj_001", "task_123", "user_456"]
        test_extend = {"priority": "high", "location": "会议室A", "duration": 60}

        logger.debug(f"📝 测试事件ID: {test_event_id}")
        logger.debug(f"📝 测试用户ID: {test_user_id}")

        # 1. 测试创建情景记忆
        logger.debug("1️⃣ 测试创建情景记忆...")
        episodic_memory = EpisodicMemory(
            user_id=test_user_id,
            group_id=test_group_id,
            timestamp=test_timestamp,
            participants=test_participants,
            summary=test_summary,
            subject=test_subject,
            episode=test_episode,
            type=test_type,
            keywords=test_keywords,
            linked_entities=test_linked_entities,
            extend=test_extend,
        )

        # 插入到数据库
        await episodic_memory.insert()
        logger.debug(f"✅ 成功创建情景记忆: {test_subject}")

        # 获取生成的事件ID
        actual_event_id = str(episodic_memory.id)
        logger.debug(f"📝 生成的事件ID: {actual_event_id}")

        # 2. 测试根据事件ID和用户ID查询
        logger.debug("2️⃣ 测试根据事件ID和用户ID查询...")
        found_memory = await repository.get_by_event_id(actual_event_id, test_user_id)
        if found_memory:
            logger.debug(
                f"✅ 查询成功: {found_memory.subject} - {found_memory.summary}"
            )
            logger.debug(f"   参与者: {found_memory.participants}")
            logger.debug(f"   情景: {found_memory.episode[:50]}...")
            logger.debug(f"   类型: {found_memory.type}")
            logger.debug(f"   关键词: {found_memory.keywords}")
            logger.debug(f"   关联实体: {found_memory.linked_entities}")
            logger.debug(f"   扩展字段: {found_memory.extend}")
        else:
            logger.error("❌ 查询失败：未找到情景记忆")
            return

        # 3. 测试根据用户ID查询列表
        logger.debug("3️⃣ 测试根据用户ID查询列表...")
        user_memories = await repository.get_by_user_id(test_user_id)
        logger.debug(f"✅ 用户 {test_user_id} 有 {len(user_memories)} 条情景记忆")
        for i, memory in enumerate(user_memories[:3]):  # 只显示前3条
            logger.debug(f"   {i+1}. {memory.subject}: {memory.summary}")

        # 4. 测试追加操作
        logger.debug("4️⃣ 测试追加操作...")
        additional_memory = EpisodicMemory(
            user_id=test_user_id,
            group_id=test_group_id,
            timestamp=test_timestamp + 3600,  # 1小时后
            participants=["张三", "李四", "王五"],  # 不同的参与者
            summary="追加的记忆单元：技术方案讨论",
            subject="技术会议",
            episode="在会议室进行了技术方案讨论，确定了架构设计方向，讨论了性能优化策略和代码规范",
            type="Conversation",  # 修复：使用字符串而不是列表
            keywords=["技术", "架构", "性能", "优化"],
            linked_entities=["proj_002", "tech_001"],
            extend={"priority": "medium", "location": "会议室B", "duration": 45},
        )

        appended_memory = await repository.append_episodic_memory(additional_memory)
        if appended_memory:
            logger.debug(f"✅ 追加成功: {appended_memory.summary}")
            logger.debug(f"   追加的事件ID: {appended_memory.event_id}")
            logger.debug(f"   追加的参与者: {appended_memory.participants}")
        else:
            logger.error("❌ 追加失败")

        # 5. 测试创建更多情景记忆
        logger.debug("5️⃣ 测试创建更多情景记忆...")
        for i in range(2):
            additional_memory = EpisodicMemory(
                user_id=test_user_id,
                group_id=test_group_id,
                timestamp=test_timestamp + i * 3600,  # 每小时一个
                participants=test_participants,
                summary=f"测试记忆单元 {i+2}：技术讨论",
                subject=f"技术会议 {i+2}",
                episode=f"这是第 {i+2} 个测试情景记忆，用于验证批量查询功能",
                type="Conversation",  # 修复：使用字符串而不是列表
                keywords=["技术", "讨论", f"测试{i+2}"],
                linked_entities=[f"entity_{i+2}"],
                extend={"test_id": i + 2, "priority": "medium"},
            )
            await additional_memory.insert()
            logger.debug(f"   ✅ 创建第 {i+2} 个情景记忆")

        # 6. 测试批量查询
        logger.debug("6️⃣ 测试批量查询...")
        all_user_memories = await repository.get_by_user_id(test_user_id)
        logger.debug(
            f"✅ 用户 {test_user_id} 总共有 {len(all_user_memories)} 条情景记忆"
        )

        # 按时间戳排序显示
        sorted_memories = sorted(
            all_user_memories, key=lambda x: x.timestamp, reverse=True
        )
        for i, memory in enumerate(sorted_memories[:5]):  # 显示前5条
            logger.debug(
                f"   {i+1}. [{memory.timestamp}] {memory.subject}: {memory.summary}"
            )

        # 7. 测试限制查询
        logger.debug("7️⃣ 测试限制查询...")
        limited_memories = await repository.get_by_user_id(test_user_id, limit=2)
        logger.debug(f"✅ 限制查询返回 {len(limited_memories)} 条记录")

        # # 8. 清理测试数据
        # logger.debug("8️⃣ 清理测试数据...")
        # # 删除所有测试数据
        # deleted_count = await repository.delete_by_user_id(test_user_id)
        # logger.debug(f"✅ 清理完成，删除了 {deleted_count} 条测试数据")

        # # 验证清理结果
        # remaining_memories = await repository.get_by_user_id(test_user_id)
        # logger.debug(f"✅ 验证清理结果：剩余 {len(remaining_memories)} 条记录")

        logger.debug("🎉 情景记忆操作测试完成！")

    except Exception as e:
        logger.error(f"❌ 情景记忆操作测试失败: {e}")
        import traceback

        traceback.print_exc()


async def test_core_memory_operations():
    """
    测试核心记忆的插入和查询操作

    该函数用于验证 CoreMemoryRawRepository 的基本功能
    """
    try:
        logger.debug("🧪 开始测试核心记忆操作...")

        # 获取核心记忆仓库
        repository = get_bean_by_type(CoreMemoryRawRepository)
        if not repository:
            logger.error("❌ 无法获取 CoreMemoryRawRepository 实例")
            return

        # 测试数据
        test_user_id = f"user_core_{uuid.uuid4().hex[:8]}"

        logger.debug(f"📝 测试用户ID: {test_user_id}")

        # 1. 测试创建核心记忆（包含所有三种类型的数据）
        logger.debug("1️⃣ 测试创建核心记忆...")
        core_memory = CoreMemory(
            user_id=test_user_id,
            # 基础信息字段
            user_name="张三",
            gender="男",
            position="高级工程师",
            supervisor_user_id="supervisor_001",
            team_members=["user_002", "user_003"],
            okr=[
                {"objective": "提升系统性能", "key_result": "响应时间减少50%"},
                {"objective": "团队协作", "key_result": "完成3个重要项目"},
            ],
            base_location="北京",
            hiredate="2022-01-15",
            age=30,
            department="技术部",
            # 个人档案字段
            personality=["内向但善于沟通", "喜欢深度思考", "注重细节"],
            hard_skills={"Python": "高级", "SQL": "专家", "产品设计": "中级"},
            soft_skills={
                "沟通能力": "清晰表达想法，有效倾听他人意见",
                "团队合作": "与他人协作实现共同目标，增强团队凝聚力",
            },
            projects_participated=[
                {
                    "project_id": "PROJ_A",
                    "project_name": "电商平台重构",
                    "role": "技术负责人",
                    "贡献": "负责整体架构设计和技术选型",
                    "进入项目日期": "2023-01-01",
                }
            ],
            # 个人偏好字段
            working_habit_preference=["远程工作", "弹性时间", "技术挑战"],
            interests=["编程", "阅读", "运动"],
            tendency=["技术导向", "质量优先", "持续学习"],
            user_goal=["成为技术专家", "提升领导力", "开源贡献"],
            # 通用字段
            extend={"priority": "high", "level": "senior", "expertise": "微服务架构"},
        )

        # 插入到数据库
        await core_memory.insert()
        logger.debug(f"✅ 成功创建核心记忆: {core_memory.user_name}")

        # 2. 测试根据用户ID查询
        logger.debug("2️⃣ 测试根据用户ID查询...")
        found_memory = await repository.get_by_user_id(test_user_id)
        if found_memory:
            logger.debug(
                f"✅ 查询成功: {found_memory.user_name} - {found_memory.position}"
            )
            logger.debug(f"   部门: {found_memory.department}")
            logger.debug(f"   性格: {found_memory.personality}")
            logger.debug(f"   硬技能数量: {len(found_memory.hard_skills or {})}")
            logger.debug(f"   工作偏好: {found_memory.working_habit_preference}")
            logger.debug(f"   个人目标: {found_memory.user_goal}")
        else:
            logger.error("❌ 查询失败：未找到核心记忆")
            return

        # 3. 测试字段提取方法
        logger.debug("3️⃣ 测试字段提取方法...")
        base_info = repository.get_base(found_memory)
        profile_info = repository.get_profile(found_memory)

        logger.debug(
            f"✅ 基础信息提取: {base_info['user_name']} - {base_info['position']}"
        )
        logger.debug(f"✅ 个人档案提取: {profile_info['personality']}")
        logger.debug(f"✅ 个人偏好提取: {profile_info['working_habit_preference']}")

        # 4. 测试分类更新方法
        logger.debug("4️⃣ 测试分类更新方法...")

        # 更新基础信息
        base_update_data = {
            "user_name": "张三（更新）",
            "position": "技术专家",
            "age": 31,
            "department": "技术部",
        }
        updated_memory = await repository.update_base(test_user_id, base_update_data)
        if updated_memory:
            logger.debug(
                f"✅ 基础信息更新成功: {updated_memory.user_name} - {updated_memory.position}"
            )

        # 更新个人档案
        profile_update_data = {
            "personality": ["外向且善于沟通", "喜欢团队协作"],
            "hard_skills": {"Python": "专家", "Go": "高级", "Kubernetes": "中级"},
        }
        updated_memory = await repository.update_profile(
            test_user_id, profile_update_data
        )
        if updated_memory:
            logger.debug(f"✅ 个人档案更新成功: {updated_memory.personality}")
            logger.debug(
                f"   更新后的硬技能: {len(updated_memory.hard_skills or {})} 项"
            )

        # 更新个人偏好
        preference_update_data = {
            "working_habit_preference": [
                "远程工作",
                "弹性时间",
                "技术挑战",
                "创新项目",
            ],
            "user_goal": ["成为技术专家", "提升领导力", "开源贡献", "技术分享"],
        }
        updated_memory = await repository.update_profile(
            test_user_id, preference_update_data
        )
        if updated_memory:
            logger.debug(
                f"✅ 个人偏好更新成功: {updated_memory.working_habit_preference}"
            )
            logger.debug(f"   更新后的目标: {updated_memory.user_goal}")

        # 6. 测试混合数据更新（验证字段过滤）
        logger.debug("6️⃣ 测试混合数据更新...")
        mixed_data = {
            "user_name": "张三（混合更新）",  # 基础信息字段
            "personality": ["混合更新后的性格"],  # 个人档案字段
            "working_habit_preference": ["混合更新偏好"],  # 个人偏好字段
            "invalid_field": "无效字段",  # 无效字段
        }

        # 使用 update_base 只会更新基础信息字段
        updated_memory = await repository.update_base(test_user_id, mixed_data)
        if updated_memory:
            logger.debug(
                f"✅ 混合数据更新成功（只更新基础信息）: {updated_memory.user_name}"
            )
            logger.debug(f"   性格未变: {updated_memory.personality}")
            logger.debug(f"   偏好未变: {updated_memory.working_habit_preference}")

        # # 7. 清理测试数据
        # logger.debug("7️⃣ 清理测试数据...")
        # deleted = await repository.delete_by_user_id(test_user_id)
        # if deleted:
        #     logger.debug("✅ 测试数据清理完成")
        # else:
        #     logger.debug("❌ 清理失败")

        # # 验证清理结果
        # remaining_memory = await repository.get_by_user_id(test_user_id)
        # if remaining_memory is None:
        #     logger.debug("✅ 验证清理结果：数据已完全删除")
        # else:
        #     logger.debug("❌ 验证清理结果：数据未完全删除")

        logger.debug("🎉 核心记忆操作测试完成！")

    except Exception as e:
        logger.error(f"❌ 核心记忆操作测试失败: {e}")
        import traceback

        traceback.print_exc()


async def test_entity_operations():
    """
    测试实体库的插入和查询操作

    该函数用于验证 EntityRawRepository 的基本功能
    """
    try:
        logger.debug("🧪 开始测试实体库操作...")

        # 获取实体库仓库
        repository = get_bean_by_type(EntityRawRepository)
        if not repository:
            logger.error("❌ 无法获取 EntityRawRepository 实例")
            return

        # 测试数据
        test_entity_id = f"entity_test_{uuid.uuid4().hex[:8]}"

        logger.debug(f"📝 测试实体ID: {test_entity_id}")

        # 1. 测试创建实体
        logger.debug("1️⃣ 测试创建实体...")
        entity = Entity(
            entity_id=test_entity_id,
            name="测试用户",
            type="Person",
            aliases=["测试", "test_user", "用户测试"],
            extend={"department": "测试部", "level": "测试工程师"},
        )

        # 插入到数据库
        await entity.insert()
        logger.debug(f"✅ 成功创建实体: {entity.name}")

        # 2. 测试根据实体ID查询
        logger.debug("2️⃣ 测试根据实体ID查询...")
        found_entity = await repository.get_by_entity_id(test_entity_id)
        if found_entity:
            logger.debug(f"✅ 查询成功: {found_entity.name} - {found_entity.type}")
            logger.debug(f"   别名: {found_entity.aliases}")
            logger.debug(f"   扩展信息: {found_entity.extend}")
        else:
            logger.error("❌ 查询失败：未找到实体")
            return

        # 5. 测试根据别名查询
        logger.debug("5️⃣ 测试根据别名查询...")
        alias_entities = await repository.get_by_alias("测试")
        logger.debug(f"✅ 根据别名查询成功: 找到 {len(alias_entities)} 个实体")

        # 6. 测试更新操作
        logger.debug("6️⃣ 测试更新操作...")
        update_data = {
            "name": "测试用户（更新）",
            "aliases": ["测试", "test_user", "用户测试", "更新测试"],
            "extend": {
                "department": "测试部",
                "level": "高级测试工程师",
                "updated": True,
            },
        }

        updated_entity = await repository.update_by_entity_id(
            test_entity_id, update_data
        )
        if updated_entity:
            logger.debug(f"✅ 更新成功: {updated_entity.name}")
            logger.debug(f"   更新后的别名: {updated_entity.aliases}")
            logger.debug(f"   更新后的扩展信息: {updated_entity.extend}")
        else:
            logger.error("❌ 更新失败")

        # 8. 测试统计功能
        logger.debug("8️⃣ 测试统计功能...")
        person_count = await repository.count_by_type("Person")
        total_count = await repository.count_all()
        logger.debug(
            f"✅ 统计成功: Person类型 {person_count} 个，总计 {total_count} 个实体"
        )

        # 9. 清理测试数据
        # logger.debug("9️⃣ 清理测试数据...")
        # deleted = await repository.delete_by_entity_id(test_entity_id)
        # if deleted:
        #     logger.debug("✅ 测试数据清理完成")
        # else:
        #     logger.debug("❌ 清理失败")

        logger.debug("🎉 实体库操作测试完成！")

    except Exception as e:
        logger.error(f"❌ 实体库操作测试失败: {e}")
        import traceback

        traceback.print_exc()


async def test_relationship_operations():
    """
    测试关系库的插入和查询操作

    该函数用于验证 RelationshipRawRepository 的基本功能
    """
    try:
        logger.debug("🧪 开始测试关系库操作...")

        # 获取关系库仓库
        repository = get_bean_by_type(RelationshipRawRepository)
        if not repository:
            logger.error("❌ 无法获取 RelationshipRawRepository 实例")
            return

        # 测试数据
        test_source_id = f"entity_source_{uuid.uuid4().hex[:8]}"
        test_target_id = f"entity_target_{uuid.uuid4().hex[:8]}"

        logger.debug(f"📝 测试源实体ID: {test_source_id}")
        logger.debug(f"📝 测试目标实体ID: {test_target_id}")

        # 1. 测试创建关系
        logger.debug("1️⃣ 测试创建关系...")
        relationship = Relationship(
            source_entity_id=test_source_id,
            target_entity_id=test_target_id,
            relationship=[
                {
                    "type": "人际关系",
                    "content": "项目协作",
                    "detail": "在测试项目中有合作关系",
                },
                {"type": "工作关系", "content": "同事", "detail": "同属一个团队"},
            ],
            extend={"strength": "medium", "context": "工作环境"},
        )

        # 插入到数据库
        await relationship.insert()
        logger.debug(f"✅ 成功创建关系: {test_source_id} -> {test_target_id}")

        # 2. 测试根据实体ID查询关系
        logger.debug("2️⃣ 测试根据实体ID查询关系...")
        found_relationship = await repository.get_by_entity_ids(
            test_source_id, test_target_id
        )
        if found_relationship:
            logger.debug(f"✅ 查询成功: 找到关系")
            logger.debug(
                f"   关系类型: {[r['type'] for r in found_relationship.relationship]}"
            )
            logger.debug(
                f"   关系内容: {[r['content'] for r in found_relationship.relationship]}"
            )
        else:
            logger.error("❌ 查询失败：未找到关系")
            return

        # 3. 测试根据源实体查询
        logger.debug("3️⃣ 测试根据源实体查询...")
        source_relationships = await repository.get_by_source_entity(test_source_id)
        logger.debug(f"✅ 根据源实体查询成功: 找到 {len(source_relationships)} 个关系")

        # 4. 测试根据目标实体查询
        logger.debug("4️⃣ 测试根据目标实体查询...")
        target_relationships = await repository.get_by_target_entity(test_target_id)
        logger.debug(
            f"✅ 根据目标实体查询成功: 找到 {len(target_relationships)} 个关系"
        )

        # 5. 测试根据关系类型查询（方法已移除）
        logger.debug("5️⃣ 测试根据关系类型查询...")
        logger.debug("✅ 关系类型查询功能测试跳过（方法已移除）")

        # 6. 测试更新操作
        logger.debug("6️⃣ 测试更新操作...")
        update_data = {
            "relationship": [
                {
                    "type": "人际关系",
                    "content": "项目协作",
                    "detail": "在测试项目中有深度合作关系",
                },
                {
                    "type": "工作关系",
                    "content": "同事",
                    "detail": "同属一个团队，经常协作",
                },
                {"type": "朋友关系", "content": "好友", "detail": "工作之外也是好朋友"},
            ],
            "extend": {"strength": "strong", "context": "工作环境", "updated": True},
        }

        updated_relationship = await repository.update_by_entity_ids(
            test_source_id, test_target_id, update_data
        )
        if updated_relationship:
            logger.debug(
                f"✅ 更新成功: 关系数量 {len(updated_relationship.relationship)}"
            )
            logger.debug(
                f"   更新后的关系类型: {[r['type'] for r in updated_relationship.relationship]}"
            )
        else:
            logger.error("❌ 更新失败")

        # 7. 测试搜索功能（已移除search_by_relationship_content方法）
        logger.debug("7️⃣ 测试搜索功能...")
        logger.debug("✅ 搜索功能测试跳过（方法已移除）")

        # 8. 测试统计功能
        logger.debug("8️⃣ 测试统计功能...")
        source_count = await repository.count_by_entity(test_source_id)
        total_count = await repository.count_all()
        logger.debug(
            f"✅ 统计成功: 源实体关系 {source_count} 个，总计 {total_count} 个关系"
        )

        # 9. 清理测试数据
        # logger.debug("9️⃣ 清理测试数据...")
        # deleted = await repository.delete_by_entity_ids(test_source_id, test_target_id)
        # if deleted:
        #     logger.debug("✅ 测试数据清理完成")
        # else:
        #     logger.debug("❌ 清理失败")

        logger.debug("🎉 关系库操作测试完成！")

    except Exception as e:
        logger.debug(f"❌ 关系库操作测试失败: {e}")
        import traceback

        traceback.print_exc()


async def test_behavior_history_operations():
    """
    测试行为历史的插入和查询操作

    该函数用于验证 BehaviorHistoryRawRepository 的基本功能
    """
    try:
        logger.debug("🧪 开始测试行为历史操作...")

        # 获取行为历史仓库
        repository = get_bean_by_type(BehaviorHistoryRawRepository)
        if not repository:
            logger.error("❌ 无法获取 BehaviorHistoryRawRepository 实例")
            return

        # 测试数据
        test_user_id = f"user_behavior_{uuid.uuid4().hex[:8]}"
        test_timestamp = int(datetime.now().timestamp())

        logger.debug(f"📝 测试用户ID: {test_user_id}")
        logger.debug(f"📝 测试时间戳: {test_timestamp}")

        # 1. 测试创建行为历史
        logger.debug("1️⃣ 测试创建行为历史...")
        behavior_history = BehaviorHistory(
            user_id=test_user_id,
            timestamp=test_timestamp,
            behavior_type=["chat", "follow-up"],
            event_id="evt_test_001",
            meta={
                "conversation_id": "conv_test_001",
                "message_count": 3,
                "duration_minutes": 10,
                "topics": ["技术讨论", "测试"],
            },
            extend={"priority": "high", "location": "office"},
        )

        # 插入到数据库
        await behavior_history.insert()
        logger.debug(f"✅ 成功创建行为历史: {behavior_history.behavior_type}")

        # 2. 测试根据用户ID和时间戳查询（方法已移除）
        logger.debug("2️⃣ 测试根据用户ID和时间戳查询...")
        logger.debug("✅ 查询功能测试跳过（方法已移除）")

        # 3. 测试根据用户ID查询列表
        logger.debug("3️⃣ 测试根据用户ID查询列表...")
        user_behaviors = await repository.get_by_user_id(test_user_id, limit=5)
        logger.debug(f"✅ 根据用户ID查询成功: 找到 {len(user_behaviors)} 个行为历史")

        # 4. 测试根据行为类型查询（方法已移除）
        logger.debug("4️⃣ 测试根据行为类型查询...")
        logger.debug("✅ 行为类型查询功能测试跳过（方法已移除）")

        # 5. 测试根据事件ID查询（方法已移除）
        logger.debug("5️⃣ 测试根据事件ID查询...")
        logger.debug("✅ 事件ID查询功能测试跳过（方法已移除）")

        # 6. 测试根据时间范围查询
        logger.debug("6️⃣ 测试根据时间范围查询...")
        start_time = test_timestamp - 3600  # 1小时前
        end_time = test_timestamp + 3600  # 1小时后
        time_range_behaviors = await repository.get_by_time_range(
            start_time, end_time, test_user_id
        )
        logger.debug(
            f"✅ 根据时间范围查询成功: 找到 {len(time_range_behaviors)} 个行为历史"
        )

        # 7. 测试追加操作
        logger.debug("7️⃣ 测试追加操作...")
        additional_behavior = BehaviorHistory(
            user_id=test_user_id,
            timestamp=test_timestamp + 1,  # 不同的时间戳
            behavior_type=["Smart-Reply", "Vote"],
            event_id="evt_test_002",
            meta={
                "conversation_id": "conv_test_002",
                "message_count": 2,
                "duration_minutes": 5,
                "topics": ["追加测试", "新行为"],
            },
            extend={"priority": "medium", "location": "remote"},
        )

        appended_behavior = await repository.append_behavior(additional_behavior)
        if appended_behavior:
            logger.debug(f"✅ 追加成功: {appended_behavior.behavior_type}")
            logger.debug(f"   追加的行为时间戳: {appended_behavior.timestamp}")
        else:
            logger.error("❌ 追加失败")

        # 8. 测试搜索功能（方法已移除）
        logger.debug("8️⃣ 测试搜索功能...")
        logger.debug("✅ 搜索功能测试跳过（方法已移除）")

        # 9. 测试统计功能
        logger.debug("9️⃣ 测试统计功能...")
        user_count = await repository.count_by_user(test_user_id)
        logger.debug(f"✅ 统计成功: 用户行为 {user_count} 个")
        # 10. 清理测试数据（方法已移除）
        logger.debug("🔟 清理测试数据...")
        logger.debug("✅ 测试数据清理跳过（方法已移除）")

        logger.debug("🎉 行为历史操作测试完成！")

    except Exception as e:
        logger.debug(f"❌ 行为历史操作测试失败: {e}")
        import traceback

        traceback.print_exc()


async def test_memcell_operations():
    """
    测试 MemCell 的插入和查询操作

    该函数用于验证 MemCellRawRepository 的基本功能
    """
    try:
        logger.debug("🧪 开始测试 MemCell 操作...")

        # 获取 MemCell 仓库
        repository = get_bean_by_type(MemCellRawRepository)
        if not repository:
            logger.error("❌ 无法获取 MemCellRawRepository 实例")
            return

        # 测试数据
        test_event_id = f"evt_memcell_{uuid.uuid4().hex[:8]}"
        test_user_id = f"user_memcell_{uuid.uuid4().hex[:8]}"
        test_group_id = f"group_memcell_{uuid.uuid4().hex[:8]}"
        test_timestamp = datetime.now()

        logger.debug(f"📝 测试事件ID: {test_event_id}")
        logger.debug(f"📝 测试用户ID: {test_user_id}")
        logger.debug(f"📝 测试群组ID: {test_group_id}")

        # 1. 测试创建 MemCell
        logger.debug("1️⃣ 测试创建 MemCell...")

        # 创建原始数据
        raw_data = RawData(
            data_type=DataTypeEnum.CONVERSATION,
            messages=[
                Message(
                    content="今天的会议讨论了新功能的设计方案，大家都很积极",
                    files=["https://example.com/design_doc.pdf"],
                    extend={
                        "sender": "张三",
                        "message_id": "msg_001",
                        "platform": "WeChat",
                    },
                ),
                Message(
                    content="我觉得这个方案很有前景，可以继续推进",
                    extend={
                        "sender": "李四",
                        "message_id": "msg_002",
                        "platform": "WeChat",
                    },
                ),
            ],
            meta={
                "chat_id": "chat_12345",
                "platform": "WeChat",
                "conversation_type": "group",
            },
        )

        memcell = MemCell(
            user_id=test_user_id,
            group_id=test_group_id,
            timestamp=test_timestamp,
            summary="团队讨论新功能设计方案，获得积极反馈",
            original_data=[raw_data],
            participants=["张三", "李四", "王五"],
            type=DataTypeEnum.CONVERSATION,
            keywords=["新功能", "设计方案", "会议", "讨论"],
            linked_entities=["project_001", "feature_002", "user_003"],
        )

        # 插入到数据库
        await memcell.insert()
        logger.debug(f"✅ 成功创建 MemCell: {memcell.summary}")

        # 2. 测试根据事件ID查询
        logger.debug("2️⃣ 测试根据事件ID查询...")
        actual_event_id = str(memcell.id)
        found_memcell = await repository.get_by_event_id(actual_event_id)
        if found_memcell:
            logger.debug(f"✅ 查询成功: {found_memcell.summary}")
            logger.debug(f"   用户ID: {found_memcell.user_id}")
            logger.debug(f"   群组ID: {found_memcell.group_id}")
            logger.debug(f"   参与者: {found_memcell.participants}")
            logger.debug(f"   关键词: {found_memcell.keywords}")
            logger.debug(f"   原始数据数量: {len(found_memcell.original_data)}")
        else:
            logger.error("❌ 查询失败：未找到 MemCell")
            return

        # 3. 测试根据用户ID查询
        logger.debug("3️⃣ 测试根据用户ID查询...")
        user_memcells = await repository.find_by_user_id(test_user_id, limit=5)
        logger.debug(f"✅ 根据用户ID查询成功: 找到 {len(user_memcells)} 个 MemCell")
        for i, memcell in enumerate(user_memcells[:3]):  # 只显示前3个
            logger.debug(f"   {i+1}. {memcell.summary[:50]}...")

        # 4. 测试根据群组ID查询
        logger.debug("4️⃣ 测试根据群组ID查询...")
        group_memcells = await repository.find_by_group_id(test_group_id, limit=5)
        logger.debug(f"✅ 根据群组ID查询成功: 找到 {len(group_memcells)} 个 MemCell")

        # 5. 测试根据参与者查询
        logger.debug("5️⃣ 测试根据参与者查询...")
        participant_memcells = await repository.find_by_participants(
            ["张三"], match_all=False, limit=5
        )
        logger.debug(
            f"✅ 根据参与者查询成功: 找到 {len(participant_memcells)} 个 MemCell"
        )

        # 6. 测试根据关键词查询
        logger.debug("6️⃣ 测试根据关键词查询...")
        keyword_memcells = await repository.search_by_keywords(
            ["新功能"], match_all=False, limit=5
        )
        logger.debug(f"✅ 根据关键词查询成功: 找到 {len(keyword_memcells)} 个 MemCell")

        # 7. 测试时间范围查询
        logger.debug("7️⃣ 测试时间范围查询...")
        start_time = test_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = test_timestamp.replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
        time_range_memcells = await repository.find_by_user_and_time_range(
            test_user_id, start_time, end_time, limit=5
        )
        logger.debug(
            f"✅ 根据时间范围查询成功: 找到 {len(time_range_memcells)} 个 MemCell"
        )

        # 8. 测试更新操作
        logger.debug("8️⃣ 测试更新操作...")
        update_data = {
            "summary": "团队讨论新功能设计方案，获得积极反馈（已更新）",
            "keywords": ["新功能", "设计方案", "会议", "讨论", "更新"],
            "linked_entities": ["project_001", "feature_002", "user_003", "update_001"],
        }

        updated_memcell = await repository.update_by_event_id(
            actual_event_id, update_data
        )
        if updated_memcell:
            logger.debug(f"✅ 更新成功: {updated_memcell.summary}")
            logger.debug(f"   更新后的关键词: {updated_memcell.keywords}")
            logger.debug(f"   更新后的关联实体: {updated_memcell.linked_entities}")
        else:
            logger.error("❌ 更新失败")

        # 9. 测试统计功能
        logger.debug("9️⃣ 测试统计功能...")
        user_count = await repository.count_by_user_id(test_user_id)
        time_range_count = await repository.count_by_time_range(
            start_time, end_time, test_user_id
        )
        logger.debug(
            f"✅ 统计成功: 用户 MemCell {user_count} 个，时间范围内 {time_range_count} 个"
        )

        # 10. 测试获取最新记录
        logger.debug("🔟 测试获取最新记录...")
        latest_memcells = await repository.get_latest_by_user(test_user_id, limit=3)
        logger.debug(f"✅ 获取最新记录成功: 返回 {len(latest_memcells)} 个 MemCell")
        for i, memcell in enumerate(latest_memcells):
            logger.debug(f"   {i+1}. [{memcell.timestamp}] {memcell.summary[:50]}...")

        # 11. 测试用户活动摘要
        logger.debug("1️⃣1️⃣ 测试用户活动摘要...")
        activity_summary = await repository.get_user_activity_summary(
            test_user_id, start_time, end_time
        )
        if activity_summary:
            logger.debug(
                f"✅ 活动摘要成功: 总计 {activity_summary.get('total_count', 0)} 条记录"
            )
            logger.debug(
                f"   类型分布: {activity_summary.get('type_distribution', {})}"
            )
            logger.debug(
                f"   最新活动: {activity_summary.get('latest_activity', 'N/A')}"
            )
        else:
            logger.debug("❌ 活动摘要失败")

        # 12. 测试创建更多 MemCell（用于批量测试）
        logger.debug("1️⃣2️⃣ 测试创建更多 MemCell...")
        for i in range(2):
            additional_memcell = MemCell(
                user_id=test_user_id,
                group_id=test_group_id,
                timestamp=datetime.now(),
                summary=f"测试 MemCell {i+2}：技术讨论和方案评审",
                original_data=[
                    RawData(
                        data_type=DataTypeEnum.MEETING,
                        messages=[
                            Message(
                                content=f"第 {i+2} 个测试会议，讨论技术实现细节",
                                extend={
                                    "sender": f"参与者{i+2}",
                                    "meeting_id": f"meeting_{i+2}",
                                },
                            )
                        ],
                        meta={"meeting_type": "technical", "duration": 60},
                    )
                ],
                participants=["张三", "李四", f"参与者{i+2}"],
                type=DataTypeEnum.MEETING,
                keywords=["技术", "讨论", f"测试{i+2}"],
                linked_entities=[f"meeting_{i+2}", f"tech_{i+2}"],
            )
            await additional_memcell.insert()
            logger.debug(f"   ✅ 创建第 {i+2} 个 MemCell")

        # 13. 测试批量查询
        logger.debug("1️⃣3️⃣ 测试批量查询...")
        all_user_memcells = await repository.find_by_user_id(test_user_id)
        logger.debug(
            f"✅ 用户 {test_user_id} 总共有 {len(all_user_memcells)} 个 MemCell"
        )

        # 按时间戳排序显示
        sorted_memcells = sorted(
            all_user_memcells, key=lambda x: x.timestamp, reverse=True
        )
        for i, memcell in enumerate(sorted_memcells[:5]):  # 显示前5个
            logger.debug(f"   {i+1}. [{memcell.timestamp}] {memcell.summary[:50]}...")

        # 14. 测试限制查询
        logger.debug("1️⃣4️⃣ 测试限制查询...")
        limited_memcells = await repository.find_by_user_id(test_user_id, limit=2)
        logger.debug(f"✅ 限制查询返回 {len(limited_memcells)} 条记录")

        # # 15. 清理测试数据
        # logger.debug("1️⃣5️⃣ 清理测试数据...")
        # deleted_count = await repository.delete_by_user_id(test_user_id)
        # logger.debug(f"✅ 清理完成，删除了 {deleted_count} 条测试数据")

        # # 验证清理结果
        # remaining_memcells = await repository.find_by_user_id(test_user_id)
        # logger.debug(f"✅ 验证清理结果：剩余 {len(remaining_memcells)} 条记录")

        logger.debug("🎉 MemCell 操作测试完成！")

    except Exception as e:
        logger.error(f"❌ MemCell 操作测试失败: {e}")
        import traceback

        traceback.print_exc()


async def test_conversation_status_operations():
    """
    测试对话状态的插入和查询操作

    该函数用于验证 ConversationStatusRawRepository 的基本功能
    """
    try:
        logger.debug("🧪 开始测试对话状态操作...")

        # 获取对话状态仓库
        repository = get_bean_by_type(ConversationStatusRawRepository)
        if not repository:
            logger.error("❌ 无法获取 ConversationStatusRawRepository 实例")
            return

        # 测试数据
        test_conversation_id = f"conv_test_{uuid.uuid4().hex[:8]}"
        test_group_id = f"group_test_{uuid.uuid4().hex[:8]}"
        test_timestamp = int(datetime.now().timestamp())

        logger.debug(f"📝 测试对话ID: {test_conversation_id}")
        logger.debug(f"📝 测试群组ID: {test_group_id}")

        # 1. 测试创建对话状态
        logger.debug("1️⃣ 测试创建对话状态...")
        conversation_status = ConversationStatus(
            conversation_id=test_conversation_id,
            group_id=test_group_id,
            old_msg_start_time=test_timestamp - 3600,  # 1小时前
            new_msg_start_time=test_timestamp - 1800,  # 30分钟前
            last_memcell_time=test_timestamp - 900,  # 15分钟前
        )

        # 插入到数据库
        await conversation_status.create()
        logger.debug(f"✅ 成功创建对话状态: {conversation_status.conversation_id}")

        # 2. 测试根据群组ID查询
        logger.debug("2️⃣ 测试根据群组ID查询...")
        found_status = await repository.get_by_group_id(test_group_id)
        if found_status:
            logger.debug(f"✅ 查询成功: {found_status.conversation_id}")
            logger.debug(f"   群组ID: {found_status.group_id}")
            logger.debug(f"   旧消息起始时间: {found_status.old_msg_start_time}")
            logger.debug(f"   新消息起始时间: {found_status.new_msg_start_time}")
            logger.debug(f"   最后MemCell时间: {found_status.last_memcell_time}")
            logger.debug(f"   创建时间: {found_status.created_at}")
            logger.debug(f"   更新时间: {found_status.updated_at}")
        else:
            logger.error("❌ 查询失败：未找到对话状态")
            return

        # 3. 测试根据群组ID查询
        logger.debug("3️⃣ 测试根据群组ID查询...")
        found_by_group = await repository.get_by_group_id(test_group_id)
        if found_by_group:
            logger.debug(f"✅ 根据群组ID查询成功: {found_by_group.conversation_id}")
            logger.debug(f"   群组ID: {found_by_group.group_id}")
            logger.debug(f"   旧消息起始时间: {found_by_group.old_msg_start_time}")
        else:
            logger.error("❌ 根据群组ID查询失败")

        # 4. 测试更新操作
        logger.debug("4️⃣ 测试更新操作...")
        update_data = {
            "old_msg_start_time": test_timestamp - 1800,  # 更新为30分钟前
            "new_msg_start_time": test_timestamp - 600,  # 更新为10分钟前
            "last_memcell_time": test_timestamp - 300,  # 更新为5分钟前
        }

        updated_status = await repository.upsert_by_group_id(test_group_id, update_data)
        if updated_status:
            logger.debug(f"✅ 更新成功: {updated_status.conversation_id}")
            logger.debug(
                f"   更新后的旧消息起始时间: {updated_status.old_msg_start_time}"
            )
            logger.debug(
                f"   更新后的新消息起始时间: {updated_status.new_msg_start_time}"
            )
            logger.debug(
                f"   更新后的最后MemCell时间: {updated_status.last_memcell_time}"
            )
        else:
            logger.error("❌ 更新失败")

        # 5. 测试 upsert 操作（更新现有记录）
        logger.debug("5️⃣ 测试 upsert 操作（更新现有记录）...")
        upsert_data = {
            "old_msg_start_time": test_timestamp - 1200,  # 更新为20分钟前
            "new_msg_start_time": test_timestamp - 300,  # 更新为5分钟前
            "last_memcell_time": test_timestamp - 60,  # 更新为1分钟前
        }

        upserted_status = await repository.upsert_by_group_id(
            test_group_id, upsert_data
        )
        if upserted_status:
            logger.debug(f"✅ upsert 成功: {upserted_status.conversation_id}")
            logger.debug(
                f"   更新后的旧消息起始时间: {upserted_status.old_msg_start_time}"
            )
            logger.debug(
                f"   更新后的新消息起始时间: {upserted_status.new_msg_start_time}"
            )
            logger.debug(
                f"   更新后的最后MemCell时间: {upserted_status.last_memcell_time}"
            )
        else:
            logger.error("❌ upsert 失败")

        # 6. 测试创建新的对话状态（用于测试 upsert 创建新记录）
        logger.debug("6️⃣ 测试创建新的对话状态...")
        new_group_id = f"group_new_{uuid.uuid4().hex[:8]}"
        new_upsert_data = {
            "old_msg_start_time": test_timestamp + 3600,  # 1小时后
            "new_msg_start_time": test_timestamp + 3600,  # 1小时后
            "last_memcell_time": test_timestamp + 3600,  # 1小时后
        }

        new_upserted_status = await repository.upsert_by_group_id(
            new_group_id, new_upsert_data
        )
        if new_upserted_status:
            logger.debug(
                f"✅ 新对话状态创建成功: {new_upserted_status.conversation_id}"
            )
            logger.debug(f"   群组ID: {new_upserted_status.group_id}")
            logger.debug(f"   旧消息起始时间: {new_upserted_status.old_msg_start_time}")
            logger.debug(f"   新消息起始时间: {new_upserted_status.new_msg_start_time}")
            logger.debug(f"   最后MemCell时间: {new_upserted_status.last_memcell_time}")
        else:
            logger.error("❌ 新对话状态创建失败")

        # 7. 测试统计功能
        logger.debug("7️⃣ 测试统计功能...")
        group_count = await repository.count_by_group_id(test_group_id)
        total_count = await repository.count_all()
        logger.debug(
            f"✅ 统计成功: 群组对话状态 {group_count} 个，总计 {total_count} 个"
        )

        # 8. 测试删除操作
        logger.debug("8️⃣ 测试删除操作...")
        deleted = await repository.delete_by_group_id(test_group_id)
        if deleted:
            logger.debug(f"✅ 删除成功: {test_group_id}")
        else:
            logger.error("❌ 删除失败")

        # 验证删除结果
        deleted_status = await repository.get_by_group_id(test_group_id)
        if deleted_status is None:
            logger.debug("✅ 验证删除结果：记录已成功删除")
        else:
            logger.debug("❌ 验证删除结果：记录未完全删除")

        # 9. 测试批量创建和查询
        logger.debug("9️⃣ 测试批量创建和查询...")
        for i in range(3):
            batch_conversation_id = f"conv_batch_{i}_{uuid.uuid4().hex[:8]}"
            batch_group_id = f"group_batch_{i}_{uuid.uuid4().hex[:8]}"

            batch_status = ConversationStatus(
                conversation_id=batch_conversation_id,
                group_id=batch_group_id,
                old_msg_start_time=test_timestamp + i * 1800,  # 每30分钟一个
                new_msg_start_time=test_timestamp + i * 1800 + 900,  # 15分钟后
                last_memcell_time=test_timestamp + i * 1800 + 1800,  # 30分钟后
            )

            await batch_status.create()
            logger.debug(f"   ✅ 创建批量对话状态 {i+1}: {batch_conversation_id}")

        # 10. 测试最终统计
        logger.debug("🔟 测试最终统计...")
        final_total_count = await repository.count_all()
        logger.debug(f"✅ 最终统计: 总计 {final_total_count} 个对话状态")

        # # 11. 清理测试数据
        # logger.debug("1️⃣1️⃣ 清理测试数据...")
        # # 清理批量创建的测试数据
        # for i in range(3):
        #     batch_conversation_id = f"conv_batch_{i}_{uuid.uuid4().hex[:8]}"
        #     await repository.delete_by_conversation_id(batch_conversation_id)
        #
        # # 清理新创建的对话状态
        # await repository.delete_by_conversation_id(new_upserted_status.conversation_id)
        #
        # logger.debug("✅ 测试数据清理完成")

        logger.debug("🎉 对话状态操作测试完成！")

    except Exception as e:
        logger.error(f"❌ 对话状态操作测试失败: {e}")
        import traceback

        traceback.print_exc()


async def test_employee_organization_operations():
    """
    测试员工组织架构关系的插入和查询操作

    该函数用于验证 EmployeeOrganizationRawRepository 的基本功能
    """
    try:
        logger.debug("🧪 开始测试员工组织架构关系操作...")

        # 获取员工组织架构关系仓库
        repository = get_bean_by_type(EmployeeOrganizationRawRepository)
        if not repository:
            logger.error("❌ 无法获取 EmployeeOrganizationRawRepository 实例")
            return

        # 测试数据
        test_user_id = f"user_org_{uuid.uuid4().hex[:8]}"
        test_email = f"test_{uuid.uuid4().hex[:8]}@company.com"
        test_direct_manager_id = f"manager_{uuid.uuid4().hex[:8]}"
        test_org_id = f"org_{uuid.uuid4().hex[:8]}"

        logger.debug(f"📝 测试用户ID: {test_user_id}")
        logger.debug(f"📝 测试邮箱: {test_email}")

        # 1. 测试创建员工组织架构关系
        logger.debug("1️⃣ 测试创建员工组织架构关系...")
        employee_organization = EmployeeOrganization(
            full_name="Shan ZHANG",
            team="Tanka",
            user_id=test_user_id,
            email=test_email,
            direct_manager="Si LI",
            direct_manager_id=test_direct_manager_id,
            skip_level_manager="Wu WANG",
            phone="13800138000",
            org_id=test_org_id,
        )

        # 插入到数据库
        await employee_organization.insert()
        logger.debug(f"✅ 成功创建员工组织架构关系: {employee_organization.full_name}")

        # 获取生成的ID
        actual_id = str(employee_organization.id)
        logger.debug(f"📝 生成的员工组织架构关系ID: {actual_id}")

        # 2. 测试根据ID查询
        logger.debug("2️⃣ 测试根据ID查询...")
        found_by_id = await repository.get_by_id(actual_id)
        if found_by_id:
            logger.debug(
                f"✅ 根据ID查询成功: {found_by_id.full_name} - {found_by_id.team}"
            )
            logger.debug(f"   用户ID: {found_by_id.user_id}")
            logger.debug(f"   邮箱: {found_by_id.email}")
            logger.debug(f"   直属经理: {found_by_id.direct_manager}")
            logger.debug(f"   隔级经理: {found_by_id.skip_level_manager}")
            logger.debug(f"   组织ID: {found_by_id.org_id}")
        else:
            logger.error("❌ 根据ID查询失败：未找到员工组织架构关系")
            return

        # 3. 测试根据用户ID查询
        logger.debug("3️⃣ 测试根据用户ID查询...")
        found_by_user_id = await repository.get_by_user_id(test_user_id)
        if found_by_user_id:
            logger.debug(f"✅ 根据用户ID查询成功: {found_by_user_id.full_name}")
            logger.debug(f"   团队: {found_by_user_id.team}")
            logger.debug(f"   邮箱: {found_by_user_id.email}")
        else:
            logger.error("❌ 根据用户ID查询失败：未找到员工组织架构关系")
            return

        # 4. 测试根据多个用户ID批量查询
        logger.debug("4️⃣ 测试根据多个用户ID批量查询...")

        # 创建更多测试数据
        test_user_ids = [test_user_id]
        for i in range(2):
            additional_user_id = f"user_org_{uuid.uuid4().hex[:8]}"
            additional_email = f"test_{uuid.uuid4().hex[:8]}@company.com"

            additional_employee = EmployeeOrganization(
                full_name=f"Employee {i+2}",
                team="Tanka",
                user_id=additional_user_id,
                email=additional_email,
                direct_manager=f"Manager {i+2}",
                phone=f"1380013800{i+1}",
                org_id=test_org_id,
            )

            await additional_employee.insert()
            test_user_ids.append(additional_user_id)
            logger.debug(
                f"   ✅ 创建第 {i+2} 个员工组织架构关系: {additional_employee.full_name}"
            )

        # 批量查询
        batch_results = await repository.get_by_user_ids(test_user_ids)
        logger.debug(f"✅ 批量查询成功: 找到 {len(batch_results)} 个员工组织架构关系")
        for i, emp in enumerate(batch_results):
            logger.debug(f"   {i+1}. {emp.full_name} - {emp.team} - {emp.user_id}")

        # 5. 测试空列表查询
        logger.debug("5️⃣ 测试空列表查询...")
        empty_results = await repository.get_by_user_ids([])
        logger.debug(f"✅ 空列表查询成功: 返回 {len(empty_results)} 个结果")

        # 6. 测试不存在的用户ID查询
        logger.debug("6️⃣ 测试不存在的用户ID查询...")
        non_existent_user_id = f"non_existent_{uuid.uuid4().hex[:8]}"
        found_non_existent = await repository.get_by_user_id(non_existent_user_id)
        if found_non_existent is None:
            logger.debug("✅ 不存在的用户ID查询正确返回 None")
        else:
            logger.error("❌ 不存在的用户ID查询应该返回 None")

        # 7. 测试不存在的ID查询
        logger.debug("7️⃣ 测试不存在的ID查询...")
        non_existent_id = "507f1f77bcf86cd799439011"  # 有效的ObjectId格式但不存在
        found_non_existent_id = await repository.get_by_id(non_existent_id)
        if found_non_existent_id is None:
            logger.debug("✅ 不存在的ID查询正确返回 None")
        else:
            logger.error("❌ 不存在的ID查询应该返回 None")

        # 8. 测试无效ID格式查询
        logger.debug("8️⃣ 测试无效ID格式查询...")
        try:
            invalid_id = "invalid_id_format"
            found_invalid_id = await repository.get_by_id(invalid_id)
            logger.debug(f"✅ 无效ID格式查询处理成功: {found_invalid_id}")
        except Exception as e:
            logger.debug(f"✅ 无效ID格式查询正确抛出异常: {e}")

        # 9. 测试统计功能（通过批量查询结果）
        logger.debug("9️⃣ 测试统计功能...")
        all_employees = await repository.get_by_user_ids(test_user_ids)
        logger.debug(
            f"✅ 统计成功: 测试用户组共有 {len(all_employees)} 个员工组织架构关系"
        )

        # 按团队分组统计
        team_stats = {}
        for emp in all_employees:
            team = emp.team
            team_stats[team] = team_stats.get(team, 0) + 1
        logger.debug(f"   团队分布: {team_stats}")

        # 10. 测试数据完整性验证
        logger.debug("🔟 测试数据完整性验证...")
        for emp in all_employees:
            # 验证必填字段
            assert emp.full_name, "员工全名不能为空"
            assert emp.team, "团队不能为空"
            assert emp.user_id, "用户ID不能为空"
            assert emp.email, "邮箱不能为空"

            # 验证邮箱格式
            assert "@" in emp.email, "邮箱格式不正确"

            logger.debug(f"   ✅ 数据完整性验证通过: {emp.full_name}")

        # # 11. 清理测试数据
        # logger.debug("1️⃣1️⃣ 清理测试数据...")
        # for emp in all_employees:
        #     await emp.delete()
        #     logger.debug(f"   ✅ 删除员工组织架构关系: {emp.full_name}")
        # logger.debug("✅ 测试数据清理完成")

        logger.debug("🎉 员工组织架构关系操作测试完成！")

    except Exception as e:
        logger.error(f"❌ 员工组织架构关系操作测试失败: {e}")
        import traceback

        traceback.print_exc()


async def run_all_tests():
    """
    运行所有内存操作测试

    包括：
    - 语义记忆测试
    - 情景记忆测试
    - 核心记忆测试
    - 实体库测试
    - 关系库测试
    - 行为历史测试
    - MemCell 测试
    - 对话状态测试
    - 员工组织架构关系测试
    """
    try:
        logger.info("🚀 开始运行所有内存操作测试...")

        # 运行各个测试
        await test_semantic_memory_operations()
        await test_episodic_memory_operations()
        await test_core_memory_operations()
        await test_entity_operations()
        await test_relationship_operations()
        await test_behavior_history_operations()
        await test_memcell_operations()
        await test_conversation_status_operations()
        await test_employee_organization_operations()

        logger.info("🎉 所有内存操作测试完成！")

    except Exception as e:
        logger.error(f"❌ 运行所有测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    """
    直接运行此文件时执行所有测试
    """
    import asyncio

    # 运行所有测试
    asyncio.run(run_all_tests())
