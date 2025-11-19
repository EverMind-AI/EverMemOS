"""
单元测试：测试 AgenticLayer 类的功能
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from agentic_layer.agentic_layer import AgenticLayer
from agentic_layer.schemas import Mode, RequestType, AppType
from agentic_layer.converter import convert_rest_to_request, convert_mq_to_request


class TestAgenticLayer:
    """测试 AgenticLayer 类"""

    @pytest.fixture
    def agentic_layer(self):
        """创建 AgenticLayer 实例"""
        return AgenticLayer(mode=Mode.WORK)

    @pytest.mark.asyncio
    async def test_handle_rest_request_success(self, agentic_layer):
        """测试 REST 请求处理成功"""
        # 模拟 FastAPI Request
        mock_fastapi_request = Mock()
        mock_fastapi_request.json = AsyncMock(
            return_value={
                "mode": "work",
                "request_type": "memorize",
                "memorize_request": {
                    "messages": [
                        {
                            "_id": "msg_1",
                            "fullName": "测试用户",
                            "roomId": "test_room",
                            "content": "这是一条测试消息",
                            "createTime": "2024-01-01T10:00:00Z",
                            "createBy": "test_user",
                        }
                    ],
                    "participants": ["test_user"],
                    "group_id": "test_room",
                },
                # data_type 字段已移除
                # payload 字段已移除
            }
        )

        with patch.object(agentic_layer.dispatcher, 'handle') as mock_handle:
            mock_handle.return_value = {"status": "ok", "created": []}

            result = await agentic_layer.handle_rest_request(mock_fastapi_request)

            # 验证 _handle 被调用
            mock_handle.assert_called_once()

            # 验证返回结果
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_handle_rest_request_json_error(self, agentic_layer):
        """测试 REST 请求 JSON 解析错误"""
        # 模拟 FastAPI Request
        mock_fastapi_request = Mock()
        mock_fastapi_request.json = AsyncMock(
            side_effect=json.JSONDecodeError("test", "doc", 0)
        )

        with pytest.raises(ValueError, match="无效的 JSON 格式"):
            await agentic_layer.handle_rest_request(mock_fastapi_request)

    @pytest.mark.asyncio
    async def test_handle_mq_request_with_consumer_record(self, agentic_layer):
        """测试 MQ 请求处理 - ConsumerRecord 格式"""
        # 模拟 Kafka ConsumerRecord
        mock_consumer_record = Mock()
        mock_consumer_record.value = {
            "mode": "work",
            "request_type": "retrieve_dynamic_mem_bm25",
            "query": "test query",
            "source": "smart_reply",
        }
        mock_consumer_record.key = "test_key"

        with patch.object(agentic_layer.dispatcher, 'handle') as mock_handle:
            mock_handle.return_value = {"status": "ok", "memories": []}

            result = await agentic_layer.handle_mq_request(mock_consumer_record)

            # 验证 _handle 被调用
            mock_handle.assert_called_once()

            # 验证返回结果
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_handle_mq_request_with_dict(self, agentic_layer):
        """测试 MQ 请求处理 - 字典格式"""
        message_dict = {
            "mode": "work",
            "request_type": "memorize",
            "memorize_request": {
                "messages": [
                    {
                        "_id": "msg_1",
                        "fullName": "测试用户",
                        "roomId": "test_room",
                        "content": "这是一条测试消息",
                        "createTime": "2024-01-01T10:00:00Z",
                        "createBy": "test_user",
                    }
                ],
                "participants": ["test_user"],
                "group_id": "test_room",
            },
            # data_type 字段已移除
            # payload 字段已移除
        }

        with patch.object(agentic_layer.dispatcher, 'handle') as mock_handle:
            mock_handle.return_value = {"status": "ok", "created": []}

            result = await agentic_layer.handle_mq_request(message_dict)

            # 验证 _handle 被调用
            mock_handle.assert_called_once()

            # 验证返回结果
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_handle_mq_request_invalid_type(self, agentic_layer):
        """测试 MQ 请求处理 - 无效类型"""
        invalid_message = "invalid_string_message"

        with pytest.raises(ValueError, match="不支持的 MQ 消息类型"):
            await agentic_layer.handle_mq_request(invalid_message)

    @pytest.mark.asyncio
    async def test_handle_mq_request_invalid_data(self, agentic_layer):
        """测试 MQ 请求处理 - 无效数据格式"""
        # 模拟 ConsumerRecord 但 value 不是字典
        mock_consumer_record = Mock()
        mock_consumer_record.value = "invalid_data"
        mock_consumer_record.key = "test_key"

        with pytest.raises(ValueError, match="MQ 消息数据必须是字典格式"):
            await agentic_layer.handle_mq_request(mock_consumer_record)

    def test_convert_rest_to_request(self):
        """测试 REST 请求转换"""
        body = {
            "mode": "work",
            "request_type": "memorize",
            "memorize_request": {
                "messages": [
                    {
                        "_id": "msg_1",
                        "fullName": "测试用户",
                        "roomId": "test_room",
                        "content": "这是一条测试消息",
                        "createTime": "2024-01-01T10:00:00Z",
                        "createBy": "test_user",
                    }
                ],
                "participants": ["test_user"],
                "group_id": "test_room",
            },
            # data_type 字段已移除
            # payload 字段已移除,
            "query": "test query",
            "source": "smart_reply",
        }
        mock_fastapi_request = Mock()

        request = convert_rest_to_request(body, mock_fastapi_request)

        assert request.mode == Mode.WORK
        assert request.request_type == RequestType.MEMORIZE
        # data_type 字段已移除，检查 request_type
        # payload 字段已移除
        assert request.query == "test query"
        assert request.source == AppType.SMART_REPLY

    def test_convert_rest_to_request_minimal(self):
        """测试 REST 请求转换 - 最小字段"""
        body = {}
        mock_fastapi_request = Mock()

        request = convert_rest_to_request(body, mock_fastapi_request)

        assert request.mode == Mode.WORK  # 默认值
        assert request.request_type is None
        # data_type 字段已移除
        # payload 字段已移除
        assert request.query is None
        assert request.source == AppType.UNKNOWN  # 默认值

    def test_convert_mq_to_request_consumer_record(self):
        """测试 MQ 消息转换 - ConsumerRecord"""
        mock_consumer_record = Mock()
        mock_consumer_record.value = {
            "mode": "work",
            "request_type": "retrieve_dynamic_mem_vector",
            "query": "test query",
            "source": "smart_vote",
        }
        mock_consumer_record.key = "test_key"

        request = convert_mq_to_request(mock_consumer_record)

        assert request.mode == Mode.WORK
        assert request.request_type == RequestType.RETRIEVE_DYNAMIC_MEM_VECTOR
        assert request.query == "test query"
        assert request.source == AppType.SMART_VOTE

    def test_convert_mq_to_request_dict(self):
        """测试 MQ 消息转换 - 字典"""
        message_dict = {
            "mode": "test",
            "request_type": "memorize",
            "memorize_request": {
                "messages": [
                    {
                        "_id": "msg_1",
                        "fullName": "测试用户",
                        "roomId": "test_room",
                        "content": "这是一条测试消息",
                        "createTime": "2024-01-01T10:00:00Z",
                        "createBy": "test_user",
                    }
                ],
                "participants": ["test_user"],
                "group_id": "test_room",
            },
            # data_type 字段已移除
            # payload 字段已移除
        }

        request = convert_mq_to_request(message_dict)

        assert request.mode == Mode.TEST
        assert request.request_type == RequestType.MEMORIZE
        # data_type 字段已移除，检查 request_type
        # payload 字段已移除

    def test_integration_with_dispatcher(self, agentic_layer):
        """测试与 Dispatcher 的集成"""
        # 测试 AgenticLayer 正确初始化了 Dispatcher
        assert agentic_layer.dispatcher is not None
        assert agentic_layer.memory_manager is not None

        # 测试模式设置
        assert agentic_layer.mode == Mode.WORK


if __name__ == "__main__":
    pytest.main([__file__])
