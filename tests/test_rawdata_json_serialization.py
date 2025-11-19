"""
测试 RawData 的 JSON 序列化和反序列化功能
包含基础功能测试和改进后的字段名启发式判断测试
"""

import pytest
from datetime import datetime
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from common_utils.datetime_utils import get_now_with_timezone


class TestRawDataJsonSerialization:
    """RawData JSON序列化测试类"""

    def test_basic_serialization(self):
        """测试基本的序列化和反序列化"""
        # 创建测试数据
        original_data = RawData(
            content={
                "speaker_id": "user_001",
                "speaker_name": "张三",
                "content": "这是一条测试消息",
                "msgType": 1,
                "roomId": "room_123",
            },
            data_id="msg_001",
            data_type="Conversation",
            metadata={"source": "test", "version": "1.0"},
        )

        # 序列化
        json_str = original_data.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # 反序列化
        restored_data = RawData.from_json_str(json_str)

        # 验证数据一致性
        assert restored_data.content == original_data.content
        assert restored_data.data_id == original_data.data_id
        assert restored_data.data_type == original_data.data_type
        assert restored_data.metadata == original_data.metadata

    def test_datetime_serialization(self):
        """测试包含datetime对象的序列化"""
        # 创建带时间的测试数据
        test_time = get_now_with_timezone()

        original_data = RawData(
            content={
                "timestamp": test_time,
                "createTime": test_time,
                "updateTime": test_time,
                "content": "包含时间的消息",
            },
            data_id="msg_datetime",
            data_type="Conversation",
            metadata={"created_at": test_time, "processed_at": test_time},
        )

        # 序列化
        json_str = original_data.to_json()

        # 反序列化
        restored_data = RawData.from_json_str(json_str)

        # 验证时间字段被正确恢复为datetime对象
        assert isinstance(restored_data.content["timestamp"], datetime)
        assert isinstance(restored_data.content["createTime"], datetime)
        assert isinstance(restored_data.content["updateTime"], datetime)
        assert isinstance(restored_data.metadata["created_at"], datetime)
        assert isinstance(restored_data.metadata["processed_at"], datetime)

        # 验证时间值的准确性（允许微小的精度差异）
        assert abs((restored_data.content["timestamp"] - test_time).total_seconds()) < 1
        assert (
            abs((restored_data.metadata["created_at"] - test_time).total_seconds()) < 1
        )

    def test_nested_structure_serialization(self):
        """测试嵌套结构的序列化"""
        original_data = RawData(
            content={
                "replyInfo": {
                    "originalMessage": "原始消息",
                    "timestamp": get_now_with_timezone(),
                    "author": {
                        "id": "user_001",
                        "name": "张三",
                        "settings": {"notify": True, "theme": "dark"},
                    },
                },
                "attachments": [
                    {"type": "image", "url": "http://example.com/img1.jpg"},
                    {"type": "file", "name": "document.pdf", "size": 1024},
                ],
            },
            data_id="complex_msg",
            data_type="Conversation",
        )

        # 序列化和反序列化
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # 验证嵌套结构
        assert restored_data.content["replyInfo"]["originalMessage"] == "原始消息"
        assert isinstance(restored_data.content["replyInfo"]["timestamp"], datetime)
        assert restored_data.content["replyInfo"]["author"]["id"] == "user_001"
        assert (
            restored_data.content["replyInfo"]["author"]["settings"]["notify"] is True
        )
        assert len(restored_data.content["attachments"]) == 2
        assert restored_data.content["attachments"][0]["type"] == "image"

    def test_email_data_serialization(self):
        """测试邮件数据的序列化（模拟email_mapper的输出）"""
        original_data = RawData(
            content={
                'user_id_list': ["user_001", "user_002"],
                'id': 'email_123',
                'source': 'gmail',
                'mail_address': 'test@example.com',
                'thread_id': 'thread_456',
                'is_delete': False,
                'is_read': True,
                'is_draft': False,
                'importance': 'high',
                'sent_timestamp': get_now_with_timezone(),
                'received_timestamp': get_now_with_timezone(),
                'labels': ['inbox', 'important'],
                'sender_name': '发送者姓名',
                'sender_address': 'sender@example.com',
                'receiver': ['receiver@example.com'],
                'cc': [],
                'bcc': [],
                'subject': '邮件主题',
                'body_type': 'html',
                'body_content': '<p>邮件内容</p>',
                'attachments': [],
                'create_timestamp': get_now_with_timezone(),
                'last_update_timestamp': get_now_with_timezone(),
                'message_id': 'msg_789',
            },
            data_id="email_123",
            data_type="Email",
            metadata={'original_id': 'email_123', 'source': 'email_mapper'},
        )

        # 序列化和反序列化
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # 验证邮件特定字段
        assert restored_data.content['user_id_list'] == ["user_001", "user_002"]
        assert restored_data.content['source'] == 'gmail'
        assert restored_data.content['is_read'] is True
        assert isinstance(restored_data.content['sent_timestamp'], datetime)
        assert isinstance(restored_data.content['received_timestamp'], datetime)
        assert restored_data.content['labels'] == ['inbox', 'important']
        assert restored_data.data_type == "Email"

    def test_linkdoc_data_serialization(self):
        """测试文档数据的序列化（模拟linkdoc_mapper的输出）"""
        original_data = RawData(
            content={
                'user_id_list': ["user_001"],
                'title': '文档标题',
                'content': '文档内容',
                'is_delete': False,
                'download_url': 'https://example.com/doc.pdf',
                'participants': ["user_001", "user_002"],
                'modify_timestamp': get_now_with_timezone(),
                'file_type': "pdf",
                'source_type': 'notion',
            },
            data_id="doc_456",
            data_type="LinkDoc",
            metadata={'original_id': 'doc_456', 'source': 'linkdoc_mapper'},
        )

        # 序列化和反序列化
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # 验证文档特定字段
        assert restored_data.content['title'] == '文档标题'
        assert restored_data.content['source_type'] == 'notion'
        assert isinstance(restored_data.content['modify_timestamp'], datetime)
        assert restored_data.content['participants'] == ["user_001", "user_002"]
        assert restored_data.data_type == "LinkDoc"

    def test_conversation_data_serialization(self):
        """测试对话数据的序列化（模拟format_transfer的输出）"""
        test_time = get_now_with_timezone()

        original_data = RawData(
            content={
                "speaker_name": "张三",
                "receiverId": "room_123",
                "roomId": "room_123",
                "groupName": "项目讨论组",
                "userIdList": ["user_001", "user_002", "user_003"],
                "referList": [],
                "content": "大家好，今天我们讨论一下项目进度",
                "timestamp": test_time,
                "createBy": "user_001",
                "updateTime": test_time,
                "orgId": "org_456",
                "speaker_id": "user_001",
                "msgType": 1,
            },
            data_id="conv_789",
            data_type="Conversation",
            metadata={
                "original_id": "conv_789",
                "createTime": test_time,
                "updateTime": test_time,
                "createBy": "user_001",
                "orgId": "org_456",
            },
        )

        # 序列化和反序列化
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # 验证对话特定字段
        assert restored_data.content["speaker_name"] == "张三"
        assert restored_data.content["groupName"] == "项目讨论组"
        assert restored_data.content["userIdList"] == [
            "user_001",
            "user_002",
            "user_003",
        ]
        assert isinstance(restored_data.content["timestamp"], datetime)
        assert isinstance(restored_data.content["updateTime"], datetime)
        assert isinstance(restored_data.metadata["createTime"], datetime)

    def test_empty_and_none_values(self):
        """测试空值和None值的处理"""
        original_data = RawData(
            content={
                "required_field": "有值",
                "empty_string": "",
                "empty_list": [],
                "empty_dict": {},
                "none_value": None,
                "zero_value": 0,
                "false_value": False,
            },
            data_id="empty_test",
            data_type=None,  # 测试可选字段为None
            metadata=None,  # 测试可选字段为None
        )

        # 序列化和反序列化
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # 验证各种空值的正确处理
        assert restored_data.content["required_field"] == "有值"
        assert restored_data.content["empty_string"] == ""
        assert restored_data.content["empty_list"] == []
        assert restored_data.content["empty_dict"] == {}
        assert restored_data.content["none_value"] is None
        assert restored_data.content["zero_value"] == 0
        assert restored_data.content["false_value"] is False
        assert restored_data.data_type is None
        assert restored_data.metadata is None

    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效JSON
        with pytest.raises(ValueError, match="JSON格式错误"):
            RawData.from_json_str("invalid json")

        # 测试非对象JSON
        with pytest.raises(ValueError, match="JSON必须是一个对象"):
            RawData.from_json_str('"string"')

        # 测试缺少必需字段
        with pytest.raises(ValueError, match="JSON缺少必需字段"):
            RawData.from_json_str('{"data_id": "test"}')  # 缺少content

        with pytest.raises(ValueError, match="JSON缺少必需字段"):
            RawData.from_json_str('{"content": {}}')  # 缺少data_id

    def test_round_trip_consistency(self):
        """测试多次序列化反序列化的一致性"""
        # 创建复杂的测试数据
        original_data = RawData(
            content={
                "mixed_types": {
                    "string": "文本",
                    "number": 42,
                    "float": 3.14,
                    "boolean": True,
                    "null": None,
                    "datetime": get_now_with_timezone(),
                    "list": [1, "two", {"three": 3}],
                    "nested": {
                        "deep": {
                            "value": "深层嵌套",
                            "timestamp": get_now_with_timezone(),
                        }
                    },
                }
            },
            data_id="round_trip_test",
            data_type="Test",
            metadata={
                "test_metadata": {"created": get_now_with_timezone(), "version": 1.0}
            },
        )

        # 进行多次序列化和反序列化
        data = original_data
        for _ in range(3):
            json_str = data.to_json()
            data = RawData.from_json_str(json_str)

        # 验证最终结果与原始数据一致
        assert data.content["mixed_types"]["string"] == "文本"
        assert data.content["mixed_types"]["number"] == 42
        assert data.content["mixed_types"]["boolean"] is True
        assert isinstance(data.content["mixed_types"]["datetime"], datetime)
        assert data.content["mixed_types"]["nested"]["deep"]["value"] == "深层嵌套"
        assert isinstance(
            data.content["mixed_types"]["nested"]["deep"]["timestamp"], datetime
        )
        assert isinstance(data.metadata["test_metadata"]["created"], datetime)

    # ==================== 改进后的字段名启发式判断测试 ====================

    def test_datetime_field_recognition(self):
        """测试时间字段的识别逻辑"""
        raw_data = RawData(content={}, data_id="test")

        # 测试应该被识别为时间字段的名称
        datetime_fields = [
            'timestamp',
            'createTime',
            'updateTime',
            'create_time',
            'update_time',
            'sent_timestamp',
            'received_timestamp',
            'create_timestamp',
            'last_update_timestamp',
            'modify_timestamp',
            'readUpdateTime',
            'created_at',
            'updated_at',
            'joinTime',
            'leaveTime',
            'lastOnlineTime',
            'sync_time',
            'processed_at',
            'custom_time',
            'event_timestamp',
            'process_at',
            'end_date',
            'datetime',
            'created',
            'updated',
            'start_time',
            'end_time',
            'event_time',
            'build_timestamp',
        ]

        for field in datetime_fields:
            # 使用受保护方法进行测试 - 这是测试内部逻辑的必要方式
            assert raw_data._is_datetime_field(
                field
            ), f"字段 '{field}' 应该被识别为时间字段"  # pylint: disable=protected-access

        # 测试不应该被识别为时间字段的名称
        non_datetime_fields = [
            'content',
            'message',
            'user_id',
            'room_id',
            'title',
            'description',
            'count',
            'size',
            'type',
            'status',
            'version',
            'id',
            'name',
            'timeout',
            'runtime',
            'timeline',
            'timestamp_format',
            'time_zone',
            'time_limit',
            'timestamp_count',
            'timestamp_enabled',
            'time_sync',
        ]

        for field in non_datetime_fields:
            # 使用受保护方法进行测试 - 这是测试内部逻辑的必要方式
            assert not raw_data._is_datetime_field(
                field
            ), f"字段 '{field}' 不应该被识别为时间字段"  # pylint: disable=protected-access

    def test_datetime_content_vs_field_name(self):
        """测试基于字段名而非内容的时间判断"""
        test_time = get_now_with_timezone()
        iso_time_str = test_time.isoformat()

        # 创建包含ISO格式字符串但字段名不是时间字段的数据
        original_data = RawData(
            content={
                "timestamp": test_time,  # 时间字段，应该被转换
                "createTime": test_time,  # 时间字段，应该被转换
                "message_content": iso_time_str,  # 内容是时间格式但字段名不是时间字段，不应该被转换
                "description": f"事件发生在 {iso_time_str}",  # 包含时间格式的描述，不应该被转换
                "user_id": "2024-01-01T10:00:00Z",  # 看起来像时间但不是时间字段，不应该被转换
                "version": "2024-01-01T10:00:00.123456+08:00",  # 版本号恰好是时间格式，不应该被转换
            },
            data_id="field_test",
            data_type="Test",
        )

        # 序列化和反序列化
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # 验证时间字段被正确转换
        assert isinstance(restored_data.content["timestamp"], datetime)
        assert isinstance(restored_data.content["createTime"], datetime)

        # 验证非时间字段保持字符串
        assert isinstance(restored_data.content["message_content"], str)
        assert isinstance(restored_data.content["description"], str)
        assert isinstance(restored_data.content["user_id"], str)
        assert isinstance(restored_data.content["version"], str)

        # 验证字符串内容不变
        assert restored_data.content["message_content"] == iso_time_str
        assert restored_data.content["description"] == f"事件发生在 {iso_time_str}"
        assert restored_data.content["user_id"] == "2024-01-01T10:00:00Z"
        assert restored_data.content["version"] == "2024-01-01T10:00:00.123456+08:00"

    def test_real_world_conversation_data_improved(self):
        """测试真实的对话数据场景（改进版）"""
        test_time = get_now_with_timezone()

        # 模拟 format_transfer.py 的真实输出
        original_data = RawData(
            content={
                "speaker_name": "张三",
                "receiverId": "room_123",
                "roomId": "room_123",
                "groupName": "技术讨论组",
                "userIdList": ["user_001", "user_002"],
                "referList": [],
                "content": "会议时间定在2024-01-01T10:00:00Z，大家记得参加",  # 包含时间格式的消息内容
                "timestamp": test_time,  # 真正的时间字段
                "createBy": "user_001",
                "updateTime": test_time,  # 真正的时间字段
                "orgId": "org_456",
                "speaker_id": "user_001",
                "msgType": 1,
                "readUpdateTime": test_time,  # 真正的时间字段
            },
            data_id="conv_001",
            data_type="Conversation",
            metadata={
                "original_id": "conv_001",
                "createTime": test_time,  # 真正的时间字段
                "updateTime": test_time,  # 真正的时间字段
                "createBy": "user_001",
                "orgId": "org_456",
            },
        )

        # 序列化和反序列化
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # 验证时间字段被正确转换
        assert isinstance(restored_data.content["timestamp"], datetime)
        assert isinstance(restored_data.content["updateTime"], datetime)
        assert isinstance(restored_data.content["readUpdateTime"], datetime)
        assert isinstance(restored_data.metadata["createTime"], datetime)
        assert isinstance(restored_data.metadata["updateTime"], datetime)

        # 验证消息内容中的时间格式字符串保持不变
        assert isinstance(restored_data.content["content"], str)
        assert "2024-01-01T10:00:00Z" in restored_data.content["content"]

        # 验证其他字段类型正确
        assert isinstance(restored_data.content["speaker_name"], str)
        assert isinstance(restored_data.content["userIdList"], list)
        assert isinstance(restored_data.content["msgType"], int)

    def test_real_world_email_data_improved(self):
        """测试真实的邮件数据场景（改进版）"""
        test_time = get_now_with_timezone()

        # 模拟 email_mapper.py 的真实输出
        original_data = RawData(
            content={
                'user_id_list': ["user_001"],
                'id': 'email_123',
                'source': 'gmail',
                'subject': '会议安排：2024-01-01T14:00:00Z开始',  # 主题包含时间格式
                'body_content': '会议将在2024-01-01T14:00:00+08:00开始，请准时参加',  # 正文包含时间格式
                'sent_timestamp': test_time,  # 真正的时间字段
                'received_timestamp': test_time,  # 真正的时间字段
                'create_timestamp': test_time,  # 真正的时间字段
                'last_update_timestamp': test_time,  # 真正的时间字段
                'sender_name': '李四',
                'sender_address': 'lisi@company.com',
            },
            data_id="email_123",
            data_type="Email",
        )

        # 序列化和反序列化
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # 验证时间字段被正确转换
        assert isinstance(restored_data.content['sent_timestamp'], datetime)
        assert isinstance(restored_data.content['received_timestamp'], datetime)
        assert isinstance(restored_data.content['create_timestamp'], datetime)
        assert isinstance(restored_data.content['last_update_timestamp'], datetime)

        # 验证文本内容中的时间格式字符串保持不变
        assert isinstance(restored_data.content['subject'], str)
        assert isinstance(restored_data.content['body_content'], str)
        assert '2024-01-01T14:00:00Z' in restored_data.content['subject']
        assert '2024-01-01T14:00:00+08:00' in restored_data.content['body_content']

    def test_real_world_document_data_improved(self):
        """测试真实的文档数据场景（改进版）"""
        test_time = get_now_with_timezone()

        # 模拟 linkdoc_mapper.py 的真实输出
        original_data = RawData(
            content={
                'user_id_list': ["user_001"],
                'title': '项目计划 - 截止时间2024-12-31T23:59:59Z',  # 标题包含时间格式
                'content': '# 项目计划\n\n开始时间：2024-01-01T09:00:00+08:00\n结束时间：2024-12-31T18:00:00+08:00',  # 内容包含时间格式
                'modify_timestamp': test_time,  # 真正的时间字段
                'last_update_timestamp': test_time,  # 真正的时间字段
                'source_type': 'notion',
                'file_type': 'markdown',
            },
            data_id="doc_123",
            data_type="LinkDoc",
        )

        # 序列化和反序列化
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # 验证时间字段被正确转换
        assert isinstance(restored_data.content['modify_timestamp'], datetime)
        assert isinstance(restored_data.content['last_update_timestamp'], datetime)

        # 验证文本内容中的时间格式字符串保持不变
        assert isinstance(restored_data.content['title'], str)
        assert isinstance(restored_data.content['content'], str)
        assert '2024-12-31T23:59:59Z' in restored_data.content['title']
        assert '2024-01-01T09:00:00+08:00' in restored_data.content['content']
        assert '2024-12-31T18:00:00+08:00' in restored_data.content['content']

    def test_nested_structure_with_mixed_content(self):
        """测试嵌套结构中的混合内容"""
        test_time = get_now_with_timezone()

        original_data = RawData(
            content={
                "message_info": {
                    "content": "定时任务将在2024-01-01T10:00:00Z执行",  # 非时间字段但包含时间格式
                    "timestamp": test_time,  # 时间字段
                    "createTime": test_time,  # 时间字段
                    "author": {
                        "name": "张三",
                        "last_login_time": test_time,  # 时间字段
                        "profile_description": "用户注册时间：2024-01-01T08:00:00+08:00",  # 非时间字段但包含时间格式
                    },
                },
                "system_info": {
                    "version": "2024.01.01T10.00.00",  # 版本号，不是时间字段
                    "build_timestamp": test_time,  # 时间字段
                    "config": {
                        "timeout": "2024-01-01T10:00:00Z",  # 超时配置，不是时间字段
                        "start_time": test_time,  # 时间字段
                        "schedule": "每天2024-01-01T10:00:00Z执行",  # 调度描述，不是时间字段
                    },
                },
            },
            data_id="nested_test",
            data_type="Test",
        )

        # 序列化和反序列化
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # 验证时间字段被正确转换
        assert isinstance(restored_data.content["message_info"]["timestamp"], datetime)
        assert isinstance(restored_data.content["message_info"]["createTime"], datetime)
        assert isinstance(
            restored_data.content["message_info"]["author"]["last_login_time"], datetime
        )
        assert isinstance(
            restored_data.content["system_info"]["build_timestamp"], datetime
        )
        assert isinstance(
            restored_data.content["system_info"]["config"]["start_time"], datetime
        )

        # 验证非时间字段保持字符串
        assert isinstance(restored_data.content["message_info"]["content"], str)
        assert isinstance(
            restored_data.content["message_info"]["author"]["profile_description"], str
        )
        assert isinstance(restored_data.content["system_info"]["version"], str)
        assert isinstance(
            restored_data.content["system_info"]["config"]["timeout"], str
        )
        assert isinstance(
            restored_data.content["system_info"]["config"]["schedule"], str
        )

        # 验证字符串内容不变
        assert (
            "2024-01-01T10:00:00Z" in restored_data.content["message_info"]["content"]
        )
        assert (
            "2024-01-01T08:00:00+08:00"
            in restored_data.content["message_info"]["author"]["profile_description"]
        )
        assert restored_data.content["system_info"]["version"] == "2024.01.01T10.00.00"
        assert (
            restored_data.content["system_info"]["config"]["timeout"]
            == "2024-01-01T10:00:00Z"
        )
        assert (
            "2024-01-01T10:00:00Z"
            in restored_data.content["system_info"]["config"]["schedule"]
        )

    def test_edge_cases_improved(self):
        """测试边界情况（改进版）"""
        test_time = get_now_with_timezone()

        original_data = RawData(
            content={
                # 空字符串和None值
                "timestamp": test_time,
                "empty_timestamp": "",
                "null_timestamp": None,
                # 数字字段名包含时间关键词
                "timestamp_count": 5,
                "time_limit": 3600,
                # 布尔字段名包含时间关键词
                "timestamp_enabled": True,
                "time_sync": False,
                # 列表中的时间字段
                "events": [
                    {
                        "name": "事件1",
                        "timestamp": test_time,  # 应该被转换
                        "description": "发生在2024-01-01T10:00:00Z",  # 不应该被转换
                    },
                    {
                        "name": "事件2",
                        "event_time": test_time,  # 应该被转换
                        "note": "计划在2024-01-02T14:00:00+08:00执行",  # 不应该被转换
                    },
                ],
            },
            data_id="edge_test",
            data_type="Test",
        )

        # 序列化和反序列化
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # 验证时间字段正确处理
        assert isinstance(restored_data.content["timestamp"], datetime)

        # 验证空值保持不变
        assert restored_data.content["empty_timestamp"] == ""
        assert restored_data.content["null_timestamp"] is None

        # 验证非字符串值保持原类型
        assert isinstance(restored_data.content["timestamp_count"], int)
        assert isinstance(restored_data.content["time_limit"], int)
        assert isinstance(restored_data.content["timestamp_enabled"], bool)
        assert isinstance(restored_data.content["time_sync"], bool)

        # 验证列表中的嵌套处理
        assert isinstance(restored_data.content["events"][0]["timestamp"], datetime)
        assert isinstance(restored_data.content["events"][1]["event_time"], datetime)
        assert isinstance(restored_data.content["events"][0]["description"], str)
        assert isinstance(restored_data.content["events"][1]["note"], str)

        # 验证字符串内容不变
        assert (
            "2024-01-01T10:00:00Z" in restored_data.content["events"][0]["description"]
        )
        assert "2024-01-02T14:00:00+08:00" in restored_data.content["events"][1]["note"]


if __name__ == "__main__":
    # 运行测试
    test_instance = TestRawDataJsonSerialization()

    print("开始测试 RawData JSON 序列化功能...")

    try:
        # 基础功能测试
        test_instance.test_basic_serialization()
        print("✅ 基本序列化测试通过")

        test_instance.test_datetime_serialization()
        print("✅ datetime序列化测试通过")

        test_instance.test_nested_structure_serialization()
        print("✅ 嵌套结构序列化测试通过")

        test_instance.test_email_data_serialization()
        print("✅ 邮件数据序列化测试通过")

        test_instance.test_linkdoc_data_serialization()
        print("✅ 文档数据序列化测试通过")

        test_instance.test_conversation_data_serialization()
        print("✅ 对话数据序列化测试通过")

        test_instance.test_empty_and_none_values()
        print("✅ 空值处理测试通过")

        test_instance.test_error_handling()
        print("✅ 错误处理测试通过")

        test_instance.test_round_trip_consistency()
        print("✅ 往返一致性测试通过")

        # 改进后的字段名启发式判断测试
        print("\n--- 改进后的字段名启发式判断测试 ---")

        test_instance.test_datetime_field_recognition()
        print("✅ 时间字段识别测试通过")

        test_instance.test_datetime_content_vs_field_name()
        print("✅ 字段名vs内容判断测试通过")

        test_instance.test_real_world_conversation_data_improved()
        print("✅ 真实对话数据测试（改进版）通过")

        test_instance.test_real_world_email_data_improved()
        print("✅ 真实邮件数据测试（改进版）通过")

        test_instance.test_real_world_document_data_improved()
        print("✅ 真实文档数据测试（改进版）通过")

        test_instance.test_nested_structure_with_mixed_content()
        print("✅ 嵌套结构混合内容测试通过")

        test_instance.test_edge_cases_improved()
        print("✅ 边界情况测试（改进版）通过")

        print("\n🎉 所有测试通过！RawData JSON序列化功能正常工作。")
        print("\n主要功能和改进:")
        print("- ✅ 完整的JSON序列化和反序列化支持")
        print("- ✅ 基于字段名而非内容的智能时间字段识别")
        print("- ✅ 避免误将消息内容中的时间格式字符串转换为datetime")
        print("- ✅ 支持项目中所有常见的时间字段命名模式")
        print("- ✅ 正确处理嵌套结构中的混合内容类型")
        print("- ✅ 完善的错误处理和边界情况处理")

    except (AssertionError, ValueError, TypeError) as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
