"""
单元测试：测试 Dispatcher 类的功能
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from agentic_layer.dispatcher import Dispatcher
from agentic_layer.agentic_layer import AgenticLayer
from agentic_layer.schemas import (
    Request,
    RequestEntrypointType,
    RequestType,
    Mode,
    AppType,
    MemoryCell,
)


class TestDispatcher:
    """测试 Dispatcher 类"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        manager = Mock()
        # 模拟 memorize 方法返回值（异步）
        manager.memorize = AsyncMock(
            return_value=[
                MemoryCell(key=1, value="test_value", metadata={"test": "data"})
            ]
        )
        # 模拟 fetch_mem 方法返回值（异步）
        from agentic_layer.memory_models import MemoryQueryResponse

        manager.fetch_mem = AsyncMock(
            return_value=MemoryQueryResponse(
                memories=[{"content": "test memory", "score": 0.9}], total_count=1
            )
        )
        # 模拟 retrieve_mem 方法返回值
        manager.retrieve_mem = AsyncMock(
            return_value=[{"content": "retrieved memory", "score": 0.8}]
        )
        return manager

    @pytest.fixture
    def dispatcher(self, mock_memory_manager):
        """创建 Dispatcher 实例"""
        return Dispatcher(mock_memory_manager)

    @pytest.mark.asyncio
    async def test_handle_memorize_request(self, dispatcher, mock_memory_manager):
        """测试记忆化请求处理"""
        request = Request(
            request_entrypoint_type=RequestEntrypointType.REST,
            mode=Mode.WORK,
            request_type=RequestType.MEMORIZE,
            memorize_request={"test": "data"},  # 添加必需的 memorize_request
            # data_type 字段已移除
            # payload 字段已移除
        )

        with patch('src.agentic_layer.dispatcher.route_request') as mock_route:
            mock_route.return_value = (RequestType.MEMORIZE, None)

            result = await dispatcher.handle(request)

            # 验证调用
            mock_memory_manager.memorize.assert_called_once_with(request)

            # 验证返回结果
            assert result["status"] == "ok"
            assert "saved_memories" in result

    @pytest.mark.asyncio
    async def test_handle_fetch_mem_request(self, dispatcher, mock_memory_manager):
        """测试获取记忆请求处理"""
        request = Request(
            request_entrypoint_type=RequestEntrypointType.REST,
            mode=Mode.WORK,
            request_type=RequestType.FETCH_MEM,
            query="test query",
            source=AppType.SMART_REPLY,
        )

        with (
            patch('src.agentic_layer.dispatcher.route_request') as mock_route,
            patch('src.agentic_layer.dispatcher.get_keys_for_source') as mock_get_keys,
        ):

            mock_route.return_value = (RequestType.FETCH_MEM, None)
            mock_get_keys.return_value = ["key1", "key2"]

            result = await dispatcher.handle(request)

            # 验证调用 - 现在 fetch_mem 接受 MemoryQueryRequest
            # mock_get_keys.assert_called_once_with(AppType.SMART_REPLY, Mode.WORK)
            mock_memory_manager.fetch_mem.assert_called_once()

            # 验证返回结果
            assert result["status"] == "ok"
            assert "memory_response" in result

    @pytest.mark.asyncio
    async def test_handle_retrieve_request(self, dispatcher, mock_memory_manager):
        """测试检索请求处理"""
        request = Request(
            mode=Mode.WORK,
            request_entrypoint_type=RequestEntrypointType.REST,
            request_type=RequestType.RETRIEVE_DYNAMIC_MEM_VECTOR,
            query="test query",
            source=AppType.SMART_REPLY,
        )

        with patch('src.agentic_layer.dispatcher.route_request') as mock_route:
            mock_route.return_value = (RequestType.RETRIEVE_DYNAMIC_MEM_BM25, "bm25")

            result = await dispatcher.handle(request)

            # 验证调用
            mock_memory_manager.retrieve_mem.assert_called_once_with(request, "bm25")

            # 验证返回结果
            assert result["status"] == "ok"
            assert "memories" in result

    # REST 和 MQ 请求处理测试现在移到 AgenticLayer 测试中

    # 转换方法测试现在移到 AgenticLayer 测试中

    @pytest.mark.asyncio
    async def test_handle_with_override_keys(self, dispatcher, mock_memory_manager):
        """测试使用覆盖键的请求处理"""
        request = Request(
            mode=Mode.WORK,
            request_entrypoint_type=RequestEntrypointType.REST,
            request_type=RequestType.FETCH_MEM,
            query="test query",
            override_keys=["custom_key1", "custom_key2"],
            source=AppType.SMART_REPLY,
        )

        with patch('src.agentic_layer.dispatcher.route_request') as mock_route:
            mock_route.return_value = (RequestType.FETCH_MEM, None)

            result = await dispatcher.handle(request)

            # 验证调用了 fetch_mem - 现在接受 MemoryQueryRequest
            mock_memory_manager.fetch_mem.assert_called_once()

            # 验证返回结果
            assert result["status"] == "ok"
            assert "memory_response" in result


if __name__ == "__main__":
    pytest.main([__file__])
