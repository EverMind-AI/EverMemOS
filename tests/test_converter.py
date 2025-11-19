"""
单元测试：测试 converter.py 中的转换函数
"""

import pytest
from unittest.mock import Mock

from agentic_layer.schemas import Mode, RequestType, AppType
from agentic_layer.converter import convert_rest_to_request, convert_mq_to_request


class TestConverter:
    """测试转换器函数"""

    def test_convert_rest_to_request_full(self):
        """测试 REST 请求转换 - 完整字段"""
        body = {
            "mode": "work",
            "request_type": "memorize",
            "memorize_request": {"test": "data"},  # 添加必需的 memorize_request
            # data_type 字段已移除
            # payload 字段已移除
            "query": "test query",
            "override_keys": ["key1", "key2"],
            "source": "smart_reply",
        }
        mock_fastapi_request = Mock()

        request = convert_rest_to_request(body, mock_fastapi_request)

        assert request.mode == Mode.WORK
        assert request.request_type == RequestType.MEMORIZE
        # data_type 字段已移除，检查 request_type
        # payload 字段已移除
        assert request.query == "test query"
        assert request.override_keys == ["key1", "key2"]
        assert request.source == AppType.SMART_REPLY

    def test_convert_rest_to_request_minimal(self):
        """测试 REST 请求转换 - 最小字段"""
        body = {}

        request = convert_rest_to_request(body)

        assert request.mode == Mode.WORK  # 默认值
        assert request.request_type is None
        # data_type 字段已移除
        # payload 字段已移除
        assert request.query is None
        assert request.override_keys is None
        assert request.source == AppType.UNKNOWN  # 默认值

    def test_convert_rest_to_request_partial(self):
        """测试 REST 请求转换 - 部分字段"""
        body = {
            "mode": "test",
            "request_type": "retrieve_dynamic_mem_bm25",
            "query": "search query",
        }

        request = convert_rest_to_request(body)

        assert request.mode == Mode.TEST
        assert request.request_type == RequestType.RETRIEVE_DYNAMIC_MEM_BM25
        # data_type 字段已移除
        # payload 字段已移除
        assert request.query == "search query"
        assert request.override_keys is None
        assert request.source == AppType.UNKNOWN

    def test_convert_mq_to_request_consumer_record(self):
        """测试 MQ 消息转换 - ConsumerRecord 格式"""
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
        # data_type 字段已移除
        # payload 字段已移除
        assert request.query == "test query"
        assert request.override_keys is None
        assert request.source == AppType.SMART_VOTE

    def test_convert_mq_to_request_dict(self):
        """测试 MQ 消息转换 - 字典格式"""
        message_dict = {
            "mode": "test",
            "request_type": "memorize",
            "memorize_request": {"test": "data"},  # 添加必需的 memorize_request
            # data_type 字段已移除
            # payload 字段已移除
            "override_keys": ["email_key"],
        }

        request = convert_mq_to_request(message_dict)

        assert request.mode == Mode.TEST
        assert request.request_type == RequestType.MEMORIZE
        # data_type 字段已移除，检查 request_type
        # payload 字段已移除
        assert request.query is None
        assert request.override_keys == ["email_key"]
        assert request.source == AppType.UNKNOWN

    def test_convert_mq_to_request_dict_with_key(self):
        """测试 MQ 消息转换 - 字典格式带 key"""
        message_dict = {
            "key": "message_key",
            "mode": "work",
            "request_type": "fetch_mem",
            "query": "fetch query",
            "source": "outputs",
        }

        request = convert_mq_to_request(message_dict)

        assert request.mode == Mode.WORK
        assert request.request_type == RequestType.FETCH_MEM
        assert request.query == "fetch query"
        assert request.source == AppType.OUTPUTS

    def test_convert_mq_to_request_invalid_type(self):
        """测试 MQ 消息转换 - 无效类型"""
        invalid_message = "invalid_string_message"

        with pytest.raises(ValueError, match="不支持的 MQ 消息类型"):
            convert_mq_to_request(invalid_message)

    def test_convert_mq_to_request_invalid_data(self):
        """测试 MQ 消息转换 - 无效数据格式"""
        mock_consumer_record = Mock()
        mock_consumer_record.value = "invalid_data"  # 不是字典
        mock_consumer_record.key = "test_key"

        with pytest.raises(ValueError, match="MQ 消息数据必须是字典格式"):
            convert_mq_to_request(mock_consumer_record)

    def test_convert_mq_to_request_object_without_value_key(self):
        """测试 MQ 消息转换 - 对象没有 value 和 key 属性"""
        mock_object = Mock()
        # 删除 value 和 key 属性
        del mock_object.value
        del mock_object.key

        # 这应该被当作字典处理，但会失败因为它不是字典
        with pytest.raises(ValueError, match="不支持的 MQ 消息类型"):
            convert_mq_to_request(mock_object)

    def test_convert_rest_to_request_with_all_enum_values(self):
        """测试 REST 请求转换 - 测试所有枚举值"""
        body = {
            "mode": "companion",
            "request_type": "retrieve_static_mem_hipporag",
            # data_type 字段已移除
            "query": "test query",  # 添加必需的 query 字段
            "source": "follow_up",
        }

        request = convert_rest_to_request(body)

        assert request.mode == Mode.COMPANION
        assert request.request_type == RequestType.RETRIEVE_STATIC_MEM_HIPPORAG
        # data_type 字段已移除，检查 request_type
        assert request.source == AppType.FOLLOW_UP

    def test_convert_mq_to_request_with_all_request_types(self):
        """测试 MQ 消息转换 - 测试所有数据类型"""
        message_dict = {
            "mode": "test",
            "request_type": "memorize",
            "memorize_request": {"test": "data"},  # 添加必需的 memorize_request
            # data_type 字段已移除
            # payload 字段已移除
        }

        request = convert_mq_to_request(message_dict)

        assert request.mode == Mode.TEST
        assert request.request_type == RequestType.MEMORIZE
        # data_type 字段已移除，检查 request_type
        # payload 字段已移除


if __name__ == "__main__":
    pytest.main([__file__])
