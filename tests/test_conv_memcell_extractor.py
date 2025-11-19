"""
ConvMemCellExtractor 测试

测试对话边界检测功能，包括：
- 对话边界检测逻辑
- MemCell生成
- 状态判断

使用方法：
    python src/bootstrap.py tests/test_conv_memcell_extractor.py
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 导入依赖注入相关模块
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

# 导入要测试的模块
from memory_layer.memcell_extractor.conv_memcell_extractor import (
    ConvMemCellExtractor,
    ConversationMemCellExtractRequest,
)
from memory_layer.memcell_extractor.base_memcell_extractor import RawData, MemCell
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.llm.openai_provider import OpenAIProvider
from memory_layer.memory_manager import RawDataType

# 获取日志记录器
logger = get_logger(__name__)


def get_llm_provider() -> LLMProvider:
    """获取LLM Provider，先尝试DI容器，失败则直接创建"""
    try:
        # 尝试从DI容器获取
        return get_bean_by_type(LLMProvider)
    except:
        # 如果DI容器中没有，则直接创建
        logger.info("DI容器中未找到LLMProvider，直接创建...")
        return LLMProvider("openai")


class TestConvMemCellExtractor:
    """ConvMemCellExtractor 测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.base_time = datetime.now() - timedelta(hours=1)

    def create_test_messages(
        self,
        count: int,
        sender: str = "Alice",
        time_offset_minutes: int = 0,
        content_prefix: str = "Test message",
    ) -> List[Dict[str, Any]]:
        """创建测试消息"""
        messages = []
        for i in range(count):
            messages.append(
                {
                    "speaker_id": f"user_{i}",
                    "speaker_name": sender if i % 2 == 0 else "Bob",
                    "content": f"{content_prefix} {i+1}: This is a test conversation.",
                    "timestamp": (
                        self.base_time + timedelta(minutes=time_offset_minutes + i)
                    ).isoformat(),
                }
            )
        return messages

    def create_raw_data_list(self, messages: List[Dict[str, Any]]) -> List[RawData]:
        """将消息转换为RawData列表"""
        raw_data_list = []
        for i, msg in enumerate(messages):
            raw_data = RawData(
                content=msg, data_id=f"test_data_{i}", metadata={"message_index": i}
            )
            raw_data_list.append(raw_data)
        return raw_data_list

    def create_realistic_conversation(self) -> tuple[List[RawData], List[RawData]]:
        """创建真实的对话场景"""
        # 历史对话 - 项目讨论
        history_messages = [
            {
                "speaker_name": "Alice",
                "content": "大家好，我们开始今天的项目会议",
                "offset": 0,
            },
            {
                "speaker_name": "Bob",
                "content": "好的，我来汇报一下后端开发进度",
                "offset": 2,
            },
            {
                "speaker_name": "Charlie",
                "content": "前端这边也有一些更新要分享",
                "offset": 4,
            },
            {"speaker_name": "Alice", "content": "很好，Bob你先说", "offset": 6},
            {
                "speaker_name": "Bob",
                "content": "后端API已经完成了80%，数据库设计基本确定",
                "offset": 8,
            },
        ]

        # 新对话 - 继续讨论
        new_messages = [
            {
                "speaker_name": "Charlie",
                "content": "前端界面已经完成了主要页面的设计",
                "offset": 30,
            },
            {
                "speaker_name": "Alice",
                "content": "太好了，那我们什么时候可以开始联调？",
                "offset": 32,
            },
            {
                "speaker_name": "Bob",
                "content": "我预计下周可以提供稳定的API",
                "offset": 34,
            },
            {
                "speaker_name": "Charlie",
                "content": "那正好，我下周也可以开始集成测试",
                "offset": 36,
            },
            {
                "speaker_name": "Alice",
                "content": "完美！那我们就这样安排",
                "offset": 38,
            },
        ]

        def create_raw_data_from_msgs(msgs: List[Dict], prefix: str) -> List[RawData]:
            raw_data_list = []
            for i, msg in enumerate(msgs):
                timestamp = (
                    self.base_time + timedelta(minutes=msg["offset"])
                ).isoformat()
                raw_data = RawData(
                    content={
                        "speaker_id": f"user_{msg['speaker_name'].lower()}",
                        "speaker_name": msg["speaker_name"],
                        "content": msg["content"],
                        "timestamp": timestamp,
                    },
                    data_id=f"{prefix}_{i}",
                    metadata={"message_index": i},
                )
                raw_data_list.append(raw_data)
            return raw_data_list

        history_raw_data = create_raw_data_from_msgs(history_messages, "history")
        new_raw_data = create_raw_data_from_msgs(new_messages, "new")

        return history_raw_data, new_raw_data

    @pytest.mark.asyncio
    async def test_conv_boundary_detection_basic(self):
        """测试基础对话边界检测"""
        print("\n🧪 测试基础对话边界检测")

        # 获取LLM Provider
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        # 创建测试数据
        history_messages = self.create_test_messages(3, "Alice", 0, "历史消息")
        new_messages = self.create_test_messages(2, "Bob", 30, "新消息")

        history_raw_data = self.create_raw_data_list(history_messages)
        new_raw_data = self.create_raw_data_list(new_messages)

        # 创建请求
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=history_raw_data,
            new_raw_data_list=new_raw_data,
            user_id_list=["alice", "bob"],
            participants=["alice", "bob"],
            group_id="test_group",
        )

        print(
            f"📋 请求数据: {len(history_raw_data)} 条历史 + {len(new_raw_data)} 条新消息"
        )

        # 执行测试
        result = await extractor.extract_memcell(request)

        # 验证结果
        assert result is not None, "边界检测结果不应该为None"
        memcell, status_result = result

        print(f"✅ 边界检测完成:")
        print(f"   - MemCell: {memcell is not None}")
        print(f"   - should_wait: {status_result.should_wait}")

        if memcell:
            assert memcell.event_id is not None
            assert len(memcell.user_id_list) > 0
            assert memcell.summary is not None

            print(f"\n📄 MemCell详细信息:")
            print(f"   - event_id: {memcell.event_id}")
            print(f"   - user_id_list: {memcell.user_id_list}")
            print(f"   - participants: {memcell.participants}")
            print(f"   - group_id: {memcell.group_id}")
            print(f"   - timestamp: {memcell.timestamp}")
            print(f"   - summary: {memcell.summary}")
            print(
                f"   - original_data条数: {len(memcell.original_data) if memcell.original_data else 0}"
            )

            if memcell.original_data:
                print(f"\n💬 原始对话内容:")
                for i, msg in enumerate(memcell.original_data[:3]):  # 只显示前3条
                    speaker = msg.get('speaker_name', '未知')
                    content = msg.get('content', '')
                    timestamp = msg.get('timestamp', '')
                    print(f"     {i+1}. [{timestamp}] {speaker}: {content}")
                if len(memcell.original_data) > 3:
                    print(f"     ... 还有 {len(memcell.original_data) - 3} 条消息")
        else:
            print(f"⚠️ 没有生成MemCell")

    @pytest.mark.asyncio
    async def test_realistic_conversation_scenario(self):
        """测试真实对话场景"""
        print("\n🧪 测试真实对话场景")

        # 获取LLM Provider
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        # 创建真实对话数据
        history_raw_data, new_raw_data = self.create_realistic_conversation()

        # 创建请求
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=history_raw_data,
            new_raw_data_list=new_raw_data,
            user_id_list=["alice", "bob", "charlie"],
            participants=["alice", "bob", "charlie"],
            group_id="project_team",
        )

        print(f"📋 真实对话场景:")
        print(f"   - 历史消息: {len(history_raw_data)} 条")
        print(f"   - 新消息: {len(new_raw_data)} 条")
        print(f"   - 参与者: {request.participants}")

        # 执行测试
        result = await extractor.extract_memcell(request)

        # 分析结果
        if result is None:
            print("⚠️ 没有检测到对话边界（这可能是正常的）")
        else:
            memcell, status_result = result
            print(f"✅ 边界检测返回结果:")
            print(f"   - MemCell: {memcell is not None}")
            print(f"   - should_wait: {status_result.should_wait}")

            if memcell:
                print(f"\n📄 真实对话MemCell详细信息:")
                print(f"   - event_id: {memcell.event_id}")
                print(f"   - user_id_list: {memcell.user_id_list}")
                print(f"   - 参与者: {memcell.participants}")
                print(f"   - 群组: {memcell.group_id}")
                print(f"   - timestamp: {memcell.timestamp}")
                print(f"   - 摘要: {memcell.summary}")
                print(
                    f"   - 原始数据条数: {len(memcell.original_data) if memcell.original_data else 0}"
                )

                # 显示完整的对话内容
                if memcell.original_data:
                    print(f"\n💬 完整对话记录:")
                    for i, msg in enumerate(memcell.original_data):
                        speaker = msg.get('speaker_name', '未知')
                        content = msg.get('content', '')
                        timestamp = msg.get('timestamp', '')
                        print(f"     {i+1}. [{timestamp}] {speaker}: {content}")

                # 验证基本字段
                assert memcell.event_id is not None
                assert len(memcell.user_id_list) == 3
                assert "alice" in memcell.user_id_list
                assert "bob" in memcell.user_id_list
                assert "charlie" in memcell.user_id_list
                assert memcell.group_id == "project_team"
            else:
                print("   - MemCell为None，可能对话还没有完整的边界")

            print(f"\n📊 边界检测状态:")
            print(f"   - should_wait: {status_result.should_wait}")
            if status_result.should_wait:
                print("   - 含义: 需要等待更多消息")
            else:
                print("   - 含义: 不需要等待，可以继续处理")

    @pytest.mark.asyncio
    async def test_insufficient_data_scenario(self):
        """测试数据不足的场景"""
        print("\n🧪 测试数据不足场景")

        # 获取LLM Provider
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        # 创建很少的消息
        history_messages = self.create_test_messages(1, "Alice", 0, "简短历史")
        new_messages = self.create_test_messages(1, "Bob", 1, "简短新消息")

        history_raw_data = self.create_raw_data_list(history_messages)
        new_raw_data = self.create_raw_data_list(new_messages)

        # 创建请求
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=history_raw_data,
            new_raw_data_list=new_raw_data,
            user_id_list=["alice", "bob"],
            participants=["alice", "bob"],
            group_id="test_group",
        )

        print(
            f"📋 数据不足场景: {len(history_raw_data)} 条历史 + {len(new_raw_data)} 条新消息"
        )

        # 执行测试
        result = await extractor.extract_memcell(request)

        # 验证结果 - 可能返回None或should_wait=True
        if result is None:
            print("✅ 正确处理数据不足情况：返回None")
        else:
            memcell, status_result = result
            print(f"✅ 状态判断: should_wait={status_result.should_wait}")

            if memcell:
                print(f"\n📄 数据不足场景MemCell信息:")
                print(f"   - event_id: {memcell.event_id}")
                print(f"   - summary: {memcell.summary}")
                print(f"   - user_id_list: {memcell.user_id_list}")
                print(
                    f"   - original_data条数: {len(memcell.original_data) if memcell.original_data else 0}"
                )
            else:
                print("   - MemCell: None")

            if status_result.should_wait:
                print("✅ 正确识别需要等待更多数据")
            else:
                print("ℹ️ 不需要等待更多数据")

    @pytest.mark.asyncio
    async def test_conversation_should_end_scenario(self):
        """测试应该结束的完整对话场景"""
        print("\n🧪 测试应该结束的完整对话场景")

        # 获取LLM Provider
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        # 构造一个完整的会议对话，从开始到明确结束
        complete_conversation = self.create_complete_meeting_conversation()
        history_raw_data, new_raw_data = complete_conversation

        # 创建请求
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=history_raw_data,
            new_raw_data_list=new_raw_data,
            user_id_list=["alice", "bob", "charlie"],
            participants=["alice", "bob", "charlie"],
            group_id="complete_meeting",
        )

        print(f"📋 完整会议对话场景:")
        print(f"   - 历史消息: {len(history_raw_data)} 条")
        print(f"   - 新消息: {len(new_raw_data)} 条")
        print(f"   - 参与者: {request.participants}")
        print(f"   - 总消息数: {len(history_raw_data) + len(new_raw_data)} 条")

        # 显示对话内容预览
        print(f"\n💬 对话内容预览:")
        all_messages = []
        for data in history_raw_data + new_raw_data:
            all_messages.append(data.content)

        for i, msg in enumerate(all_messages[:3]):
            speaker = msg.get('speaker_name', '未知')
            content = msg.get('content', '')
            print(f"   开始: {speaker}: {content}")

        print(f"   ... (中间 {len(all_messages) - 6} 条消息)")

        for i, msg in enumerate(all_messages[-3:]):
            speaker = msg.get('speaker_name', '未知')
            content = msg.get('content', '')
            print(f"   结束: {speaker}: {content}")

        # 执行测试
        print(f"\n🔄 开始边界检测...")
        result = await extractor.extract_memcell(request)

        # 分析结果
        if result is None:
            print("❌ 意外：完整对话没有检测到边界")
        else:
            memcell, status_result = result
            print(f"✅ 完整对话边界检测结果:")
            print(f"   - MemCell: {memcell is not None}")
            print(f"   - should_wait: {status_result.should_wait}")

            if memcell:
                print(f"\n📄 完整对话MemCell详细信息:")
                print(f"   - event_id: {memcell.event_id}")
                print(f"   - user_id_list: {memcell.user_id_list}")
                print(f"   - 参与者: {memcell.participants}")
                print(f"   - 群组: {memcell.group_id}")
                print(f"   - timestamp: {memcell.timestamp}")
                print(f"   - 摘要: {memcell.summary}")
                print(
                    f"   - 原始数据条数: {len(memcell.original_data) if memcell.original_data else 0}"
                )

                # 显示完整的对话内容
                if memcell.original_data:
                    print(f"\n💬 MemCell中包含的对话记录:")
                    for i, msg in enumerate(memcell.original_data):
                        speaker = msg.get('speaker_name', '未知')
                        content = msg.get('content', '')
                        timestamp = msg.get('timestamp', '')
                        print(f"     {i+1}. [{timestamp}] {speaker}: {content}")

                # 验证这是一个完整的对话
                assert memcell.event_id is not None
                assert len(memcell.user_id_list) == 3
                assert memcell.group_id == "complete_meeting"
                print(f"\n✅ 验证通过：这是一个完整的会议对话MemCell")

            else:
                print("⚠️ MemCell为None，可能对话判断逻辑需要调整")

            print(f"\n📊 边界检测状态分析:")
            print(f"   - should_wait: {status_result.should_wait}")
            if status_result.should_wait:
                print("   - 含义: 需要等待更多消息（可能判断不够完整）")
            else:
                print("   - 含义: 对话已完整，可以处理（符合预期）")

            if memcell and not status_result.should_wait:
                print(f"\n🎉 成功：检测到完整对话边界！")
            elif not memcell and not status_result.should_wait:
                print(f"\n🤔 部分成功：判断对话完整但未生成MemCell")
            else:
                print(f"\n📝 需要优化：对话判断逻辑可能需要调整")

    def create_complete_meeting_conversation(
        self,
    ) -> tuple[List[RawData], List[RawData]]:
        """创建一个完整的会议对话，从开始到明确结束"""
        base_time = datetime.now() - timedelta(hours=2)  # 2小时前开始

        # 第一阶段：会议开始和议题介绍 (历史消息)
        meeting_start = [
            {
                "speaker_name": "Alice",
                "content": "大家好，现在开始我们的项目评审会议。今天主要讨论三个议题：项目进度、技术方案确认、下一步计划。",
                "offset": 0,
            },
            {
                "speaker_name": "Bob",
                "content": "好的Alice，我已经准备好项目进度报告了。",
                "offset": 1,
            },
            {
                "speaker_name": "Charlie",
                "content": "技术方案文档我也已经更新完成。",
                "offset": 2,
            },
            {
                "speaker_name": "Alice",
                "content": "很好，那我们按顺序来。Bob，请先汇报项目进度。",
                "offset": 3,
            },
            {
                "speaker_name": "Bob",
                "content": "好的。本周我们完成了用户登录模块的开发和测试，进度符合预期。数据库设计也已经完成，下周开始接口开发。",
                "offset": 5,
            },
            {
                "speaker_name": "Alice",
                "content": "不错，有遇到什么技术难题吗？",
                "offset": 6,
            },
            {
                "speaker_name": "Bob",
                "content": "主要是在用户权限管理这块，不过已经找到解决方案了。",
                "offset": 7,
            },
        ]

        # 第二阶段：技术讨论和决策 + 会议总结和结束 (新消息，时间间隔较长表示经过深入讨论)
        meeting_end = [
            {
                "speaker_name": "Alice",
                "content": "好的，现在Charlie来介绍技术方案的调整。",
                "offset": 45,
            },  # 45分钟后，表示中间有深入讨论
            {
                "speaker_name": "Charlie",
                "content": "经过这段时间的分析，我建议我们采用微服务架构，这样可以更好地支持后续扩展。",
                "offset": 46,
            },
            {
                "speaker_name": "Bob",
                "content": "我赞成Charlie的方案，这样确实更灵活。我们需要调整一下开发计划吗？",
                "offset": 47,
            },
            {
                "speaker_name": "Alice",
                "content": "需要的。我们重新评估一下时间线。整体项目可能会延后一周，但质量会更好。",
                "offset": 48,
            },
            {
                "speaker_name": "Charlie",
                "content": "我可以在下周提供详细的架构设计文档。",
                "offset": 49,
            },
            {
                "speaker_name": "Bob",
                "content": "那我这边也会配合调整开发计划。",
                "offset": 50,
            },
            {
                "speaker_name": "Alice",
                "content": "很好。那我们今天的三个议题都讨论完了。总结一下：项目进度正常，技术方案调整为微服务架构，时间线调整为延后一周。",
                "offset": 52,
            },
            {"speaker_name": "Alice", "content": "大家还有其他问题吗？", "offset": 53},
            {"speaker_name": "Bob", "content": "我没有其他问题了。", "offset": 54},
            {"speaker_name": "Charlie", "content": "我也没有。", "offset": 55},
            {
                "speaker_name": "Alice",
                "content": "好的，那今天的会议就到这里。谢谢大家的参与，我会整理会议纪要发给大家。散会！",
                "offset": 56,
            },
        ]

        def create_raw_data_from_msgs(msgs: List[Dict], prefix: str) -> List[RawData]:
            raw_data_list = []
            for i, msg in enumerate(msgs):
                timestamp = (base_time + timedelta(minutes=msg["offset"])).isoformat()
                raw_data = RawData(
                    content={
                        "speaker_id": f"user_{msg['speaker_name'].lower()}",
                        "speaker_name": msg["speaker_name"],
                        "content": msg["content"],
                        "timestamp": timestamp,
                    },
                    data_id=f"{prefix}_{i}",
                    metadata={"message_index": i, "meeting_phase": prefix},
                )
                raw_data_list.append(raw_data)
            return raw_data_list

        history_raw_data = create_raw_data_from_msgs(meeting_start, "meeting_start")
        new_raw_data = create_raw_data_from_msgs(meeting_end, "meeting_end")

        print(f"🏗️ 构造完整会议对话:")
        print(f"   - 会议开始阶段: {len(meeting_start)} 条消息")
        print(f"   - 会议结束阶段: {len(meeting_end)} 条消息")
        print(
            f"   - 时间跨度: {meeting_start[0]['offset']} 到 {meeting_end[-1]['offset']} 分钟"
        )
        print(f"   - 特点: 明确的开始、讨论、决策、总结、结束")

        return history_raw_data, new_raw_data

    @pytest.mark.asyncio
    async def test_data_processing_internal(self):
        """测试内部数据处理逻辑"""
        print("\n🧪 测试内部数据处理")

        # 获取LLM Provider
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        # 创建测试数据
        test_message = {
            "speaker_id": "user_alice",
            "speaker_name": "Alice",
            "content": "这是一条测试消息",
            "timestamp": self.base_time.isoformat(),
        }

        raw_data = RawData(
            content=test_message, data_id="test_data", metadata={"test": True}
        )

        # 测试内部数据处理方法
        processed_data = extractor._data_process(raw_data)

        print(f"📋 数据处理测试:")
        print(f"   - 原始数据: {test_message}")
        print(f"   - 处理后: {processed_data}")

        # 验证处理结果
        assert processed_data is not None
        assert isinstance(processed_data, dict)
        assert "speaker_name" in processed_data
        assert "content" in processed_data


async def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行ConvMemCellExtractor测试")
    print("=" * 60)

    test_instance = TestConvMemCellExtractor()

    try:
        # 运行测试方法
        test_instance.setup_method()
        await test_instance.test_conv_boundary_detection_basic()

        test_instance.setup_method()
        await test_instance.test_realistic_conversation_scenario()

        test_instance.setup_method()
        await test_instance.test_insufficient_data_scenario()

        test_instance.setup_method()
        await test_instance.test_conversation_should_end_scenario()

        test_instance.setup_method()
        await test_instance.test_data_processing_internal()

        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")

    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    # 当直接运行此脚本时执行
    # 注意：通过 bootstrap.py 运行时，环境已经初始化完成
    asyncio.run(run_all_tests())
