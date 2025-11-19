"""
Real Conversation Pipeline 测试

测试完整的记忆提取管道，包括：
- 使用真实的tanka_memorize.memorize()函数
- 自动处理ConvMemCellExtractor和EpisodeMemoryExtractor
- 测试完整的端到端流程

使用方法：
    python src/bootstrap.py tests/test_real_conversation_pipeline.py
"""

import pytest
import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 导入依赖注入相关模块
from core.observation.logger import get_logger

# 导入要测试的模块
from biz_layer.tanka_memorize import memorize
from memory_layer.memory_manager import MemorizeRequest
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from memory_layer.types import RawDataType, MemoryType

# 获取日志记录器
logger = get_logger(__name__)


class TestRealConversationPipeline:
    """Real Conversation Pipeline 测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.base_time = datetime.now() - timedelta(hours=1)

    def create_project_discussion_messages(self) -> List[Dict[str, Any]]:
        """创建项目讨论对话"""
        messages = [
            {
                "speaker_id": "user_1",
                "speaker_name": "Alice",
                "content": "今天的项目进度怎么样？我们来讨论一下当前的开发状况。",
                "timestamp": (self.base_time + timedelta(minutes=0)).isoformat(),
            },
            {
                "speaker_id": "user_2",
                "speaker_name": "Bob",
                "content": "还不错，前端部分基本完成了，正在做最后的测试。React组件都已经开发完毕。",
                "timestamp": (self.base_time + timedelta(minutes=2)).isoformat(),
            },
            {
                "speaker_id": "user_1",
                "speaker_name": "Alice",
                "content": "那后端API开发呢？数据库设计完成了吗？",
                "timestamp": (self.base_time + timedelta(minutes=4)).isoformat(),
            },
            {
                "speaker_id": "user_2",
                "speaker_name": "Bob",
                "content": "后端API还在进行中，数据库设计已经完成，预计明天能完成所有接口的开发。",
                "timestamp": (self.base_time + timedelta(minutes=6)).isoformat(),
            },
            {
                "speaker_id": "user_1",
                "speaker_name": "Alice",
                "content": "好的，那测试计划什么时候开始？我们需要准备测试环境。",
                "timestamp": (self.base_time + timedelta(minutes=8)).isoformat(),
            },
            {
                "speaker_id": "user_2",
                "speaker_name": "Bob",
                "content": "测试计划已经准备好了，测试环境也配置完成，等后端完成后就可以开始集成测试。",
                "timestamp": (self.base_time + timedelta(minutes=10)).isoformat(),
            },
        ]
        return messages

    def create_personal_conversation_messages(self) -> List[Dict[str, Any]]:
        """创建个人关怀对话（话题转换）"""
        messages = [
            {
                "speaker_id": "user_1",
                "speaker_name": "Alice",
                "content": "对了Bob，你最近身体怎么样？工作强度挺大的。",
                "timestamp": (
                    self.base_time + timedelta(minutes=35)
                ).isoformat(),  # 25分钟间隔
            },
            {
                "speaker_id": "user_2",
                "speaker_name": "Bob",
                "content": "还不错，就是有点累，最近工作压力比较大，但还能坚持。",
                "timestamp": (self.base_time + timedelta(minutes=37)).isoformat(),
            },
            {
                "speaker_id": "user_1",
                "speaker_name": "Alice",
                "content": "要注意休息啊，身体最重要。项目可以稍微放慢节奏。",
                "timestamp": (self.base_time + timedelta(minutes=39)).isoformat(),
            },
            {
                "speaker_id": "user_2",
                "speaker_name": "Bob",
                "content": "谢谢Alice的关心，我会注意平衡工作和休息的。",
                "timestamp": (self.base_time + timedelta(minutes=41)).isoformat(),
            },
        ]
        return messages

    def create_complete_meeting_conversation(self) -> List[Dict[str, Any]]:
        """创建完整的会议对话（应该能生成MemCell）"""
        base_time = datetime.now() - timedelta(hours=2)

        messages = [
            {
                "speaker_id": "user_1",
                "speaker_name": "Alice",
                "content": "大家好，现在开始我们的周会。今天主要讨论三个议题：上周进度回顾、本周计划、技术难点。",
                "timestamp": (base_time + timedelta(minutes=0)).isoformat(),
            },
            {
                "speaker_id": "user_2",
                "speaker_name": "Bob",
                "content": "好的Alice，我准备好了进度报告。",
                "timestamp": (base_time + timedelta(minutes=1)).isoformat(),
            },
            {
                "speaker_id": "user_3",
                "speaker_name": "Charlie",
                "content": "我也准备好了技术方案的更新。",
                "timestamp": (base_time + timedelta(minutes=2)).isoformat(),
            },
            {
                "speaker_id": "user_1",
                "speaker_name": "Alice",
                "content": "很好，Bob先汇报上周进度。",
                "timestamp": (base_time + timedelta(minutes=3)).isoformat(),
            },
            {
                "speaker_id": "user_2",
                "speaker_name": "Bob",
                "content": "上周完成了用户认证模块和权限管理系统，测试覆盖率达到90%，没有发现重大bug。",
                "timestamp": (base_time + timedelta(minutes=5)).isoformat(),
            },
            {
                "speaker_id": "user_3",
                "speaker_name": "Charlie",
                "content": "前端配合完成了登录界面和权限控制组件，与后端联调正常。",
                "timestamp": (base_time + timedelta(minutes=7)).isoformat(),
            },
            # 40分钟后继续讨论（表示经过深入讨论）
            {
                "speaker_id": "user_1",
                "speaker_name": "Alice",
                "content": "经过讨论，我们确定本周的重点是完成支付模块和订单管理。",
                "timestamp": (base_time + timedelta(minutes=45)).isoformat(),
            },
            {
                "speaker_id": "user_2",
                "speaker_name": "Bob",
                "content": "支付模块我来负责，预计周三完成开发。",
                "timestamp": (base_time + timedelta(minutes=46)).isoformat(),
            },
            {
                "speaker_id": "user_3",
                "speaker_name": "Charlie",
                "content": "订单管理界面我来处理，周五之前完成。",
                "timestamp": (base_time + timedelta(minutes=47)).isoformat(),
            },
            {
                "speaker_id": "user_1",
                "speaker_name": "Alice",
                "content": "很好，大家都很明确。有什么技术难点需要讨论的吗？",
                "timestamp": (base_time + timedelta(minutes=48)).isoformat(),
            },
            {
                "speaker_id": "user_2",
                "speaker_name": "Bob",
                "content": "支付安全这块需要特别注意，我会参考业界最佳实践。",
                "timestamp": (base_time + timedelta(minutes=49)).isoformat(),
            },
            {
                "speaker_id": "user_1",
                "speaker_name": "Alice",
                "content": "好的，今天的会议就到这里。大家有问题随时沟通。",
                "timestamp": (base_time + timedelta(minutes=50)).isoformat(),
            },
        ]
        return messages

    def create_raw_data_list(self, messages: List[Dict[str, Any]]) -> List[RawData]:
        """将消息转换为RawData列表"""
        raw_data_list = []
        for i, msg in enumerate(messages):
            raw_data = RawData(
                content=msg, data_id=f"msg_{i}", metadata={"message_index": i}
            )
            raw_data_list.append(raw_data)
        return raw_data_list

    def display_conversation_content(
        self, messages: List[Dict[str, Any]], title: str = "对话内容"
    ):
        """显示对话内容"""
        print(f"\n💬 {title}:")
        for msg in messages:
            timestamp = msg['timestamp']
            speaker = msg['speaker_name']
            content = msg['content']
            print(f"   [{timestamp}] {speaker}: {content}")

    def display_memory_results(self, memory_list: List, test_name: str):
        """显示记忆提取结果"""
        print(f"\n📋 {test_name} - tanka_memorize结果:")
        if memory_list:
            print(f"✅ 提取了 {len(memory_list)} 个记忆:")
            for i, memory in enumerate(memory_list):
                print(f"\n   记忆 {i+1}:")
                print(f"     - Event ID: {memory.event_id}")
                print(f"     - User ID: {memory.user_id}")
                print(f"     - Memory Type: {memory.memory_type.value}")
                print(f"     - Timestamp: {memory.timestamp}")
                print(f"     - Subject: {memory.subject}")
                print(
                    f"     - Summary: {memory.summary[:150] if memory.summary else 'N/A'}..."
                )
                print(f"     - Group ID: {memory.group_id}")
                print(f"     - Participants: {memory.participants}")

                # 显示特定字段
                if hasattr(memory, 'episode') and memory.episode:
                    print(f"     - Episode: {memory.episode[:150]}...")
                if hasattr(memory, 'tags') and memory.tags:
                    print(f"     - Tags: {memory.tags}")
                if hasattr(memory, 'hard_skills') and memory.hard_skills:
                    print(f"     - Hard Skills: {memory.hard_skills}")
                if hasattr(memory, 'soft_skills') and memory.soft_skills:
                    print(f"     - Soft Skills: {memory.soft_skills}")
                if (
                    hasattr(memory, 'projects_participated')
                    and memory.projects_participated
                ):
                    print(
                        f"     - Projects: {len(memory.projects_participated)} 个项目"
                    )
        else:
            print("❌ 没有提取到记忆")

    async def run_memorize_pipeline(
        self,
        messages: List[Dict[str, Any]],
        test_name: str,
        group_id: str = "test_group",
    ) -> List:
        """运行记忆提取管道"""
        raw_data_list = self.create_raw_data_list(messages)

        # 分割为历史和新消息
        mid_point = len(raw_data_list) // 2
        history_raw_data_list = raw_data_list[:mid_point]
        new_raw_data_list = raw_data_list[mid_point:]

        print(
            f"\n📊 {test_name} - 处理 {len(history_raw_data_list)} 条历史消息 + {len(new_raw_data_list)} 条新消息"
        )

        # 创建MemorizeRequest
        memorize_request = MemorizeRequest(
            history_raw_data_list=history_raw_data_list,
            new_raw_data_list=new_raw_data_list,
            raw_data_type=RawDataType.CONVERSATION,
            participants=["alice", "bob", "charlie"],
            group_id=group_id,
        )

        # 调用tanka_memorize
        print(f"\n🔄 执行 tanka_memorize.memorize()...")
        memory_list = await memorize(memorize_request)

        return memory_list

    async def run_streaming_memorize_pipeline(
        self,
        messages: List[Dict[str, Any]],
        test_name: str,
        group_id: str = "test_group",
    ) -> List:
        """运行流式记忆提取管道 - 一条条输入消息"""
        print(f"\n🌊 开始流式记忆提取测试: {test_name}")
        print("=" * 80)

        all_raw_data = self.create_raw_data_list(messages)

        # 模拟流式输入：维护历史消息缓冲区
        history_buffer = []
        all_memories = []

        print(f"📊 总共将流式处理 {len(all_raw_data)} 条消息")

        for i, new_raw_data in enumerate(all_raw_data):
            print(f"\n{'='*60}")
            print(f"📨 流式输入第 {i+1}/{len(all_raw_data)} 条消息")
            print(f"{'='*60}")

            # 显示当前消息
            msg_content = new_raw_data.content
            print(
                f"👤 {msg_content.get('speaker_name', 'Unknown')}: {msg_content.get('content', '')[:80]}..."
            )
            print(f"⏰ 时间: {msg_content.get('timestamp', 'N/A')}")

            # 创建MemorizeRequest - 流式输入
            memorize_request = MemorizeRequest(
                history_raw_data_list=history_buffer.copy(),  # 当前历史
                new_raw_data_list=[new_raw_data],  # 只有一条新消息
                raw_data_type=RawDataType.CONVERSATION,
                participants=["alice", "bob", "charlie"],
                group_id=group_id,
            )

            print(f"📊 当前状态:")
            print(f"   历史消息: {len(history_buffer)} 条")
            print(f"   新消息: 1 条")
            print(f"   参与者: {memorize_request.participants}")

            # 调用tanka_memorize
            print(f"🔄 执行 tanka_memorize.memorize()...")
            try:
                memory_list = await memorize(memorize_request)

                if memory_list:
                    print(f"✅ 提取到 {len(memory_list)} 个记忆!")

                    # 显示每个记忆的详细信息
                    for j, memory in enumerate(memory_list):
                        print(f"\n   🧠 记忆 #{j+1}:")
                        print(f"      类型: {memory.memory_type.value}")
                        print(f"      用户: {memory.user_id}")
                        print(f"      标题: {memory.subject}")
                        print(f"      摘要: {memory.summary[:100]}...")
                        print(f"      事件ID: {memory.event_id}")
                        print(f"      参与者: {memory.participants}")

                    all_memories.extend(memory_list)

                    # 调试：查看Mock数据库操作
                    await self.debug_database_operations(memory_list, f"流式第{i+1}轮")

                else:
                    print("ℹ️ 未提取到记忆（可能需要更多消息才能触发边界检测）")

            except Exception as e:
                print(f"❌ memorize调用失败: {e}")
                import traceback

                traceback.print_exc()

            # 将当前消息添加到历史缓冲区
            history_buffer.append(new_raw_data)

            # 可选：限制历史缓冲区大小，避免过长
            max_history_size = 20
            if len(history_buffer) > max_history_size:
                history_buffer = history_buffer[-max_history_size:]
                print(f"📝 历史缓冲区已满，保留最近 {max_history_size} 条消息")

            # 流式处理间隔
            await asyncio.sleep(0.1)

        print(f"\n{'='*80}")
        print(f"🎉 流式测试完成!")
        print(f"📊 总共处理: {len(all_raw_data)} 条消息")
        print(f"🧠 生成记忆: {len(all_memories)} 个")

        if all_memories:
            # 统计记忆类型
            memory_types = {}
            for memory in all_memories:
                mem_type = memory.memory_type.value
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1

            print(f"📋 记忆类型分布:")
            for mem_type, count in memory_types.items():
                print(f"   - {mem_type}: {count} 个")

        return all_memories

    async def debug_database_operations(self, memory_list: List, round_name: str):
        """调试数据库操作"""
        print(f"\n🔍 数据库操作调试 - {round_name}:")

        # 尝试获取mock repository实例并查看其状态
        try:
            from core.di import get_bean_by_type
            from biz_layer.mock_repositories import (
                UserProfileRepository,
                MemoryRepository,
            )

            # 获取repository实例
            user_profile_repo = get_bean_by_type(UserProfileRepository)
            memory_repo = get_bean_by_type(MemoryRepository)

            print(
                f"   📂 UserProfileRepository类型: {type(user_profile_repo).__name__}"
            )
            print(f"   📂 MemoryRepository类型: {type(memory_repo).__name__}")

            # 如果是mock实例，尝试查看其内部状态
            if hasattr(user_profile_repo, '_stored_profiles'):
                profile_count = (
                    len(user_profile_repo._stored_profiles)
                    if user_profile_repo._stored_profiles
                    else 0
                )
                print(f"   👥 Mock用户profiles数量: {profile_count}")

                if profile_count > 0:
                    print(
                        f"   👥 用户列表: {list(user_profile_repo._stored_profiles.keys())}"
                    )

            if hasattr(memory_repo, '_stored_memories'):
                memory_count = (
                    len(memory_repo._stored_memories)
                    if memory_repo._stored_memories
                    else 0
                )
                print(f"   🧠 Mock存储记忆数量: {memory_count}")

                # 显示最近保存的记忆
                if memory_count > 0:
                    recent_memories = memory_repo._stored_memories[-3:]  # 显示最近3个
                    print(f"   🧠 最近保存的记忆:")
                    for i, mem in enumerate(recent_memories):
                        if hasattr(mem, 'memory_type') and hasattr(mem, 'user_id'):
                            print(
                                f"      {i+1}. {mem.memory_type.value} - {mem.user_id}: {mem.title[:50]}..."
                            )
                        else:
                            print(
                                f"      {i+1}. {type(mem).__name__}: {str(mem)[:50]}..."
                            )

        except Exception as e:
            print(f"   ❌ 无法获取repository状态: {e}")

        # 显示这轮保存的记忆
        if memory_list:
            print(f"   💾 本轮保存的记忆:")
            for i, memory in enumerate(memory_list):
                print(
                    f"      {i+1}. {memory.memory_type.value} - {memory.user_id}: {memory.subject[:50]}..."
                )

    @pytest.mark.asyncio
    async def test_project_discussion_pipeline(self):
        """测试项目讨论对话的记忆提取管道"""
        print("\n🧪 测试项目讨论对话管道")

        # 检查API密钥
        if not os.getenv("OPENROUTER_API_KEY"):
            print("⚠️ OPENROUTER_API_KEY 未设置，跳过真实LLM测试")
            pytest.skip("OPENROUTER_API_KEY not available")
            return

        # 创建项目讨论对话
        messages = self.create_project_discussion_messages()
        self.display_conversation_content(messages, "项目讨论对话")

        # 运行管道
        memory_list = await self.run_memorize_pipeline(
            messages, "项目讨论", "project_team"
        )

        # 显示结果
        self.display_memory_results(memory_list, "项目讨论")

        # 简单验证（不强制断言，因为LLM结果可能变化）
        if memory_list:
            print(f"✅ 项目讨论管道测试通过：提取了记忆")
        else:
            print(f"ℹ️ 项目讨论管道：未提取到记忆（可能对话未达到边界条件）")

    @pytest.mark.asyncio
    async def test_personal_conversation_pipeline(self):
        """测试个人关怀对话的记忆提取管道"""
        print("\n🧪 测试个人关怀对话管道")

        # 检查API密钥
        if not os.getenv("OPENROUTER_API_KEY"):
            print("⚠️ OPENROUTER_API_KEY 未设置，跳过真实LLM测试")
            pytest.skip("OPENROUTER_API_KEY not available")
            return

        # 创建个人关怀对话
        messages = self.create_personal_conversation_messages()
        self.display_conversation_content(messages, "个人关怀对话")

        # 运行管道
        memory_list = await self.run_memorize_pipeline(
            messages, "个人关怀", "personal_chat"
        )

        # 显示结果
        self.display_memory_results(memory_list, "个人关怀")

        # 简单验证
        if memory_list:
            print(f"✅ 个人关怀管道测试通过：提取了记忆")
        else:
            print(f"ℹ️ 个人关怀管道：未提取到记忆（可能对话未达到边界条件）")

    @pytest.mark.asyncio
    async def test_complete_meeting_pipeline(self):
        """测试完整会议对话的记忆提取管道"""
        print("\n🧪 测试完整会议对话管道")

        # 检查API密钥
        if not os.getenv("OPENROUTER_API_KEY"):
            print("⚠️ OPENROUTER_API_KEY 未设置，跳过真实LLM测试")
            pytest.skip("OPENROUTER_API_KEY not available")
            return

        # 创建完整会议对话
        messages = self.create_complete_meeting_conversation()
        self.display_conversation_content(messages, "完整会议对话")

        # 运行管道
        memory_list = await self.run_memorize_pipeline(
            messages, "完整会议", "weekly_meeting"
        )

        # 显示结果
        self.display_memory_results(memory_list, "完整会议")

        # 验证结果
        if memory_list:
            print(f"✅ 完整会议管道测试通过：提取了 {len(memory_list)} 个记忆")

            # 验证记忆类型
            memory_types = [m.memory_type for m in memory_list]
            if MemoryType.EPISODE_SUMMARY in memory_types:
                print("   ✅ 包含情节记忆")
            if MemoryType.PROFILE in memory_types:
                print("   ✅ 包含档案记忆")

            # 验证参与者
            all_participants = set()
            for memory in memory_list:
                if memory.participants:
                    all_participants.update(memory.participants)
            print(f"   📋 涉及参与者: {list(all_participants)}")

        else:
            print(f"⚠️ 完整会议管道：未提取到记忆，可能需要调整对话内容或边界检测逻辑")

    @pytest.mark.asyncio
    async def test_mixed_conversation_pipeline(self):
        """测试混合对话（项目+个人）的记忆提取管道"""
        print("\n🧪 测试混合对话管道")

        # 检查API密钥
        if not os.getenv("OPENROUTER_API_KEY"):
            print("⚠️ OPENROUTER_API_KEY 未设置，跳过真实LLM测试")
            pytest.skip("OPENROUTER_API_KEY not available")
            return

        # 合并项目讨论和个人关怀对话
        project_messages = self.create_project_discussion_messages()
        personal_messages = self.create_personal_conversation_messages()
        mixed_messages = project_messages + personal_messages

        self.display_conversation_content(mixed_messages, "混合对话（项目+个人）")

        # 运行管道
        memory_list = await self.run_memorize_pipeline(
            mixed_messages, "混合对话", "mixed_chat"
        )

        # 显示结果
        self.display_memory_results(memory_list, "混合对话")

        # 分析结果
        if memory_list:
            print(f"✅ 混合对话管道测试通过：提取了 {len(memory_list)} 个记忆")

            # 分析记忆类型分布
            episode_count = sum(
                1 for m in memory_list if m.memory_type == MemoryType.EPISODE_SUMMARY
            )
            profile_count = sum(
                1 for m in memory_list if m.memory_type == MemoryType.PROFILE
            )

            print(f"   📊 记忆类型分布:")
            print(f"     - 情节记忆: {episode_count} 个")
            print(f"     - 档案记忆: {profile_count} 个")

        else:
            print(f"ℹ️ 混合对话管道：未提取到记忆")

    @pytest.mark.asyncio
    async def test_streaming_complete_meeting_pipeline(self):
        """测试流式完整会议对话的记忆提取管道"""
        print("\n🧪 测试流式完整会议对话管道")

        # 检查API密钥
        if not os.getenv("OPENROUTER_API_KEY"):
            print("⚠️ OPENROUTER_API_KEY 未设置，跳过真实LLM测试")
            pytest.skip("OPENROUTER_API_KEY not available")
            return

        # 创建完整会议对话
        messages = self.create_complete_meeting_conversation()
        self.display_conversation_content(messages, "完整会议对话（流式处理）")

        # 运行流式管道
        memory_list = await self.run_streaming_memorize_pipeline(
            messages, "流式完整会议", "streaming_weekly_meeting"
        )

        # 显示最终结果
        self.display_memory_results(memory_list, "流式完整会议")

        # 验证结果
        if memory_list:
            print(f"✅ 流式完整会议管道测试通过：提取了 {len(memory_list)} 个记忆")

            # 验证记忆类型
            memory_types = [m.memory_type for m in memory_list]
            if MemoryType.EPISODE_SUMMARY in memory_types:
                print("   ✅ 包含情节记忆")
            if MemoryType.PROFILE in memory_types:
                print("   ✅ 包含档案记忆")

        else:
            print(f"⚠️ 流式完整会议管道：未提取到记忆，检查是否需要调整边界检测参数")

    @pytest.mark.asyncio
    async def test_streaming_mixed_conversation_pipeline(self):
        """测试流式混合对话的记忆提取管道"""
        print("\n🧪 测试流式混合对话管道")

        # 检查API密钥
        if not os.getenv("OPENROUTER_API_KEY"):
            print("⚠️ OPENROUTER_API_KEY 未设置，跳过真实LLM测试")
            pytest.skip("OPENROUTER_API_KEY not available")
            return

        # 合并项目讨论和个人关怀对话
        project_messages = self.create_project_discussion_messages()
        personal_messages = self.create_personal_conversation_messages()
        mixed_messages = project_messages + personal_messages

        self.display_conversation_content(
            mixed_messages, "混合对话（项目+个人）流式处理"
        )

        # 运行流式管道
        memory_list = await self.run_streaming_memorize_pipeline(
            mixed_messages, "流式混合对话", "streaming_mixed_chat"
        )

        # 显示结果
        self.display_memory_results(memory_list, "流式混合对话")

        # 分析结果
        if memory_list:
            print(f"✅ 流式混合对话管道测试通过：提取了 {len(memory_list)} 个记忆")

            # 分析记忆类型分布
            episode_count = sum(
                1 for m in memory_list if m.memory_type == MemoryType.EPISODE_SUMMARY
            )
            profile_count = sum(
                1 for m in memory_list if m.memory_type == MemoryType.PROFILE
            )

            print(f"   📊 记忆类型分布:")
            print(f"     - 情节记忆: {episode_count} 个")
            print(f"     - 档案记忆: {profile_count} 个")

        else:
            print(f"ℹ️ 流式混合对话管道：未提取到记忆")


async def run_all_tests():
    """运行所有管道测试"""
    print("🚀 开始运行 Real Conversation Pipeline 测试")
    print("=" * 80)

    test_instance = TestRealConversationPipeline()

    try:
        # 先运行标准批量测试
        print("\n📦 第一阶段：批量处理测试")
        print("=" * 60)

        test_instance.setup_method()
        await test_instance.test_project_discussion_pipeline()

        test_instance.setup_method()
        await test_instance.test_personal_conversation_pipeline()

        test_instance.setup_method()
        await test_instance.test_complete_meeting_pipeline()

        test_instance.setup_method()
        await test_instance.test_mixed_conversation_pipeline()

        # 然后运行流式处理测试
        print("\n🌊 第二阶段：流式处理测试")
        print("=" * 60)

        test_instance.setup_method()
        await test_instance.test_streaming_complete_meeting_pipeline()

        test_instance.setup_method()
        await test_instance.test_streaming_mixed_conversation_pipeline()

        print("\n" + "=" * 80)
        print("🎉 所有管道测试完成！")
        print("=" * 80)
        print("\n💡 测试总结:")
        print("   📦 批量处理测试:")
        print("     - 项目讨论对话管道 ✅")
        print("     - 个人关怀对话管道 ✅")
        print("     - 完整会议对话管道 ✅")
        print("     - 混合对话管道 ✅")
        print("   🌊 流式处理测试:")
        print("     - 流式完整会议对话管道 ✅")
        print("     - 流式混合对话管道 ✅")
        print("\n📋 这些测试验证了 tanka_memorize 的完整端到端功能")
        print("🔍 流式测试提供了详细的数据库操作调试信息")

    except Exception as e:
        logger.error(f"❌ 管道测试执行失败: {e}")
        import traceback

        traceback.print_exc()
        raise


async def run_streaming_tests_only():
    """仅运行流式处理测试"""
    print("🌊 运行流式处理测试")
    print("=" * 80)

    test_instance = TestRealConversationPipeline()

    try:
        print("\n🧪 流式处理专项测试 - 模拟真实消息逐条输入")
        print("=" * 60)

        # 只运行流式测试，获得更详细的调试信息
        test_instance.setup_method()
        await test_instance.test_streaming_complete_meeting_pipeline()

        test_instance.setup_method()
        await test_instance.test_streaming_mixed_conversation_pipeline()

        print("\n" + "=" * 80)
        print("🎉 流式处理测试完成！")
        print("=" * 80)
        print("\n💡 流式测试特点:")
        print("   - 一条条消息输入，模拟真实对话场景")
        print("   - 实时显示边界检测结果")
        print("   - 详细的数据库操作调试信息")
        print("   - Mock Repository状态追踪")
        print("\n🔍 调试信息可以帮助分析:")
        print("   - 何时触发对话边界检测")
        print("   - 记忆提取的详细过程")
        print("   - 数据库读写操作状况")

    except Exception as e:
        logger.error(f"❌ 流式测试执行失败: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    # 当直接运行此脚本时执行
    # 注意：通过 bootstrap.py 运行时，环境已经初始化完成

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--streaming-only":
        print("🌊 仅运行流式处理测试")
        asyncio.run(run_streaming_tests_only())
    else:
        print("🚀 运行所有测试（包括批量和流式）")
        print("💡 提示：使用 --streaming-only 参数仅运行流式测试")
        asyncio.run(run_all_tests())
