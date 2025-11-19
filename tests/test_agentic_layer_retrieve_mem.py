#!/usr/bin/env python3
"""
Agentic Layer Retrieve Memory 测试

测试流程：
1. 建设测试用例，生成对应的Request
2. Request中重点关注retrieve_mem_request建设，其user_id为caroline，memory_types为['episodic_memory']
3. 调用agentic_layer.py 中的handle方法，传入Request，成功从memory_base检索回记忆
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

from agentic_layer.schemas import (
    Request,
    RequestEntrypointType,
    RequestType,
    Mode,
    AppType,
)
from agentic_layer.dtos.memory_query import RetrieveMemRequest, MemoryType

from agentic_layer.agentic_layer import AgenticLayer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgenticLayerRetrieveTest:
    """Agentic Layer Retrieve Memory 测试类"""

    def __init__(self):
        """初始化测试类"""
        # 创建AgenticLayer实例
        self.agentic_layer = AgenticLayer(mode=Mode.WORK)

    def create_retrieve_request(
        self,
        query: str,
        user_id: str = "c896cd10-99a0-4a40-8959-350a212e5a05",
        memory_types: List[MemoryType] = None,
        top_k: int = 10,
    ) -> Request:
        """
        创建检索请求

        Args:
            query: 查询字符串
            user_id: 用户ID
            memory_types: 记忆类型列表 (默认为['episodic_memory'])
            top_k: 返回结果数量

        Returns:
            Request对象
        """
        if memory_types is None:
            memory_types = [MemoryType.EPISODIC_MEMORY]

        # 创建RetrieveMemRequest
        retrieve_mem_request = RetrieveMemRequest(
            user_id=user_id,
            memory_types=memory_types,
            top_k=top_k,
            filters={},
            include_metadata=True,
        )

        # 创建Request对象
        request = Request(
            mode=Mode.WORK,
            request_entrypoint_type=RequestEntrypointType.REST,
            request_type=RequestType.RETRIEVE_DYNAMIC_MEM_KEYWORD,
            retrieve_mem_request=retrieve_mem_request,
            source=AppType.UNKNOWN,
        )

        return request

    async def test_agentic_layer_retrieve(
        self, query: str, user_id: str = "c896cd10-99a0-4a40-8959-350a212e5a05"
    ) -> Dict[str, Any]:
        """
        测试通过agentic_layer的handle方法进行检索

        Args:
            query: 查询字符串

        Returns:
            检索结果
        """
        logger.info(f"开始测试agentic_layer检索，查询: {query}")

        try:
            # 创建Request对象
            request = self.create_retrieve_request(
                query=query,
                user_id=user_id,
                memory_types=[MemoryType.EPISODIC_MEMORY],
                top_k=5,
            )

            logger.info(
                f"创建的Request: user_id={request.retrieve_mem_request.user_id}, "
                f"memory_types={request.retrieve_mem_request.memory_types}"
            )

            # 调用agentic_layer的handle方法
            result = await self.agentic_layer.handle(request)

            logger.info(f"agentic_layer检索完成，返回结果: {result}")

            return {
                "method": "agentic_layer",
                "query": query,
                "result": result,
                "success": True,
            }

        except Exception as e:
            logger.error(f"agentic_layer检索失败: {e}")
            import traceback

            traceback.print_exc()
            return {
                "method": "agentic_layer",
                "query": query,
                "error": str(e),
                "success": False,
            }

    def print_results(self, results: List[Dict[str, Any]], test_name: str):
        """
        打印测试结果

        Args:
            results: 测试结果列表
            test_name: 测试名称
        """
        print(f"\n{'='*80}")
        print(f"测试结果: {test_name}")
        print(f"{'='*80}")

        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            if result.get("success", False):
                print(f"  方法: {result.get('method', 'N/A')}")
                print(f"  查询: {result.get('query', 'N/A')}")
                if 'user_id' in result:
                    print(f"  用户ID: {result.get('user_id', 'N/A')}")

                # 打印agentic_layer返回的结果
                agentic_result = result.get('result', {})
                if isinstance(agentic_result, dict):
                    print(f"  状态: {agentic_result.get('status', 'N/A')}")
                    memories_response = agentic_result.get('retrieve_memories', None)
                    if hasattr(memories_response, 'memories'):
                        memories = memories_response.memories
                        print(f"  记忆数量: {len(memories)}")

                        # 打印记忆内容（前3条）
                        for j, memory in enumerate(memories[:3]):
                            if isinstance(memory, dict):
                                print(f"    记忆 {j+1}: {memory}")
                            else:
                                print(f"    记忆 {j+1}: {str(memory)[:200]}...")
                    else:
                        print(f"  记忆数量: 0")
                else:
                    print(f"  结果: {str(agentic_result)[:200]}...")
            else:
                print(f"  用户ID: {result.get('user_id', 'N/A')}")
                print(f"  错误: {result.get('error', 'Unknown error')}")

    async def test_user_id_retrieval(
        self, query: str = "Jie GONG参与2025年9月11日下午项目进展会议安排"
    ) -> List[Dict[str, Any]]:
        """
        测试不同user_id的检索结果

        Args:
            query: 查询字符串

        Returns:
            不同user_id的检索结果列表
        """
        logger.info("开始测试不同user_id的检索结果")

        # 测试不同的user_id
        user_ids = ["c896cd10-99a0-4a40-8959-350a212e5a05", "jon_1", "NonExistentUser"]
        results = []

        for user_id in user_ids:
            logger.info(f"测试user_id: {user_id}")

            try:
                # 创建Request对象
                request = self.create_retrieve_request(
                    query=query,
                    user_id=user_id,
                    memory_types=[MemoryType.EPISODIC_MEMORY],
                    top_k=5,
                )

                logger.info(
                    f"创建的Request: user_id={request.retrieve_mem_request.user_id}, "
                    f"memory_types={request.retrieve_mem_request.memory_types}"
                )

                # 调用agentic_layer的handle方法
                response = await self.agentic_layer.handle(request)

                # 处理响应
                if (
                    response
                    and isinstance(response, dict)
                    and response.get("status") == "ok"
                ):
                    memories_response = response.get("retrieve_memories")
                    result = {
                        "success": True,
                        "method": "agentic_layer",
                        "query": query,
                        "user_id": user_id,
                        "result": {
                            "status": "ok",
                            "retrieve_memories": memories_response,
                        },
                    }
                else:
                    result = {
                        "success": False,
                        "method": "agentic_layer",
                        "query": query,
                        "user_id": user_id,
                        "error": f"No response or invalid response format: {response}",
                    }

                results.append(result)
                logger.info(f"user_id {user_id} 检索完成，返回结果: {result}")

            except Exception as e:
                logger.error(f"user_id {user_id} 检索失败: {e}")
                result = {
                    "success": False,
                    "method": "agentic_layer",
                    "query": query,
                    "user_id": user_id,
                    "error": str(e),
                }
                results.append(result)

        return results

    async def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行Agentic Layer Retrieve Memory测试")

        # 测试1: 基本检索
        logger.info("测试1: 基本检索")
        basic_result = await self.test_agentic_layer_retrieve(
            "Jie GONG参与2025年9月11日下午项目进展会议安排"
        )
        self.print_results([basic_result], "基本检索")

        # 测试2: 不同user_id的检索区分测试
        logger.info("测试2: 不同user_id的检索区分测试")
        user_id_results = await self.test_user_id_retrieval(
            "Jie GONG参与2025年9月11日下午项目进展会议安排"
        )
        self.print_results(user_id_results, "不同user_id的检索区分测试")

        logger.info("所有测试完成")


async def main():
    """主函数"""
    try:
        # 创建测试实例
        test = AgenticLayerRetrieveTest()

        # 运行所有测试
        await test.run_all_tests()

    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())
