"""
Memory Controller API 测试脚本
验证 /api/v1/memories 下的所有接口的输入输出结构

使用方法:
    # 运行所有测试
    python tests/test_memory_controller.py
    
    # 指定API地址
    python tests/test_memory_controller.py --base-url http://localhost:1995
    
    # 指定测试用户
    python tests/test_memory_controller.py --base-url http://dev-server:1995 --user-id test_user_123
    
    # 单独测试某个方法
    python tests/test_memory_controller.py --test-method memorize
    python tests/test_memory_controller.py --test-method fetch_episodic
    python tests/test_memory_controller.py --test-method fetch_event_log
    python tests/test_memory_controller.py --test-method search_keyword
"""

import argparse
import json
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

# 使用上海时区
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


class MemoryControllerTester:
    """Memory Controller API 测试类"""

    def __init__(self, base_url: str, user_id: str, group_id: str, timeout: int = 180):
        """
        初始化测试器

        Args:
            base_url: API基础URL
            user_id: 测试用户ID
            group_id: 测试群组ID
            timeout: 请求超时时间(秒)，默认180秒(3分钟)
        """
        self.base_url = base_url
        self.api_prefix = "/api/v1/memories"
        self.user_id = user_id
        self.group_id = group_id
        self.timeout = timeout

    def print_section(self, title: str):
        """打印分隔线"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def call_post_api(self, endpoint: str, data: dict):
        """
        调用 POST API 并打印结果

        Args:
            endpoint: API端点
            data: 请求数据

        Returns:
            (status_code, response_json)
        """
        # 如果是 memorize 接口且没有提供 sender，则随机生成一个
        if endpoint == "" and "sender" not in data:
            data["sender"] = f"user_{uuid.uuid4().hex[:12]}"
            print(f"⚠️  未提供 sender，自动生成: {data['sender']}")

        url = f"{self.base_url}{self.api_prefix}{endpoint}"
        print(f"\n📍 URL: POST {url}")
        print("📤 请求数据:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        try:
            response = requests.post(url, json=data, timeout=self.timeout)
            print(f"\n📥 响应状态码: {response.status_code}")
            print("📥 响应数据:")
            response_json = response.json()
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            return response.status_code, response_json
        except Exception as e:  # noqa: BLE001 需要捕获所有异常以保证脚本继续运行
            print(f"\n❌ 请求失败: {e}")
            return None, None

    def call_get_api(self, endpoint: str, params: dict = None):
        """
        调用 GET API 并打印结果

        Args:
            endpoint: API端点
            params: 查询参数

        Returns:
            (status_code, response_json)
        """
        url = f"{self.base_url}{self.api_prefix}{endpoint}"
        print(f"\n📍 URL: GET {url}")
        if params:
            print("📤 查询参数:")
            print(json.dumps(params, indent=2, ensure_ascii=False))

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            print(f"\n📥 响应状态码: {response.status_code}")
            print("📥 响应数据:")
            response_json = response.json()
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            return response.status_code, response_json
        except Exception as e:  # noqa: BLE001 需要捕获所有异常以保证脚本继续运行
            print(f"\n❌ 请求失败: {e}")
            return None, None

    def call_get_with_body_api(self, endpoint: str, data: dict):
        """
        调用 GET API（带 body）并打印结果

        虽然不常见，但某些搜索接口（如 Elasticsearch）使用 GET + body 的方式传递复杂参数

        Args:
            endpoint: API端点
            data: 请求数据（放在 body 中）

        Returns:
            (status_code, response_json)
        """
        url = f"{self.base_url}{self.api_prefix}{endpoint}"
        print(f"\n📍 URL: GET {url} (with body)")
        print("📤 请求数据:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        try:
            # GET 请求带 body（requests 库支持，但不常用）
            response = requests.request("GET", url, json=data, timeout=self.timeout)
            print(f"\n📥 响应状态码: {response.status_code}")
            print("📥 响应数据:")
            response_json = response.json()
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            return response.status_code, response_json
        except Exception as e:  # noqa: BLE001 需要捕获所有异常以保证脚本继续运行
            print(f"\n❌ 请求失败: {e}")
            return None, None

    def call_patch_api(self, endpoint: str, data: dict):
        """
        调用 PATCH API 并打印结果

        Args:
            endpoint: API端点
            data: 请求数据

        Returns:
            (status_code, response_json)
        """
        url = f"{self.base_url}{self.api_prefix}{endpoint}"
        print(f"\n📍 URL: PATCH {url}")
        print("📤 请求数据:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        try:
            response = requests.patch(url, json=data, timeout=self.timeout)
            print(f"\n📥 响应状态码: {response.status_code}")
            print("📥 响应数据:")
            response_json = response.json()
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            return response.status_code, response_json
        except Exception as e:  # noqa: BLE001 需要捕获所有异常以保证脚本继续运行
            print(f"\n❌ 请求失败: {e}")
            return None, None

    def test_memorize_single_message(self):
        """测试1: POST /api/v1/memories - 存储对话记忆（发送多条消息以触发边界检测）"""
        self.print_section("测试1: POST /api/v1/memories - 存储对话记忆")

        # 准备一段简单的对话，模拟用户和助手的交互
        # 发送多条消息可以触发边界检测并提取记忆
        base_time = datetime.now(SHANGHAI_TZ)

        # 构建对话序列，通过以下方式触发边界检测：
        # 1. 第一个场景：关于咖啡偏好的对话（4条消息）
        # 2. 第二个场景：开启新话题（通过时间间隔+主题切换触发边界）
        messages = [
            # 场景1：讨论咖啡偏好（完整的对话情节）
            {
                "group_id": self.group_id,
                "group_name": "测试群组",
                "message_id": "msg_001",
                "create_time": base_time.isoformat(),
                "sender": self.user_id,
                "sender_name": "测试用户",
                "content": "我最近想养成喝咖啡的习惯，你有什么建议吗？",
                "refer_list": [],
            },
            {
                "group_id": self.group_id,
                "group_name": "测试群组",
                "message_id": "msg_002",
                "create_time": (base_time + timedelta(seconds=30)).isoformat(),
                "sender": "assistant_001",
                "sender_name": "AI助手",
                "content": "当然可以！咖啡有很多种类，从浓郁的意式浓缩到温和的美式，您可以根据口味选择。建议从美式咖啡开始尝试。",
                "refer_list": [],
            },
            {
                "group_id": self.group_id,
                "group_name": "测试群组",
                "message_id": "msg_003",
                "create_time": (base_time + timedelta(minutes=1)).isoformat(),
                "sender": self.user_id,
                "sender_name": "测试用户",
                "content": "我喜欢喝美式咖啡，不加糖不加奶，越浓越好。",
                "refer_list": [],
            },
            {
                "group_id": self.group_id,
                "group_name": "测试群组",
                "message_id": "msg_004",
                "create_time": (
                    base_time + timedelta(minutes=1, seconds=30)
                ).isoformat(),
                "sender": "assistant_001",
                "sender_name": "AI助手",
                "content": "了解您的偏好了！纯黑美式咖啡确实能完整体验咖啡豆的风味。建议选择深度烘焙的咖啡豆会更浓郁。",
                "refer_list": [],
            },
            # 场景2：开启新话题（通过较长时间间隔+主题切换触发边界）
            # 根据边界检测规则：时间间隔超过4小时且内容无关联会触发边界
            {
                "group_id": self.group_id,
                "group_name": "测试群组",
                "message_id": "msg_005",
                "create_time": (base_time + timedelta(hours=24)).isoformat(),
                "sender": self.user_id,
                "sender_name": "测试用户",
                "content": "对了，周末的项目进展如何？",
                "refer_list": [],
            },
            {
                "group_id": self.group_id,
                "group_name": "测试群组",
                "message_id": "msg_006",
                "create_time": (
                    base_time + timedelta(hours=24, seconds=30)
                ).isoformat(),
                "sender": "assistant_001",
                "sender_name": "AI助手",
                "content": "项目进展顺利，主要功能已经完成了80%，预计下周可以提交测试。",
                "refer_list": [],
            },
        ]

        # 逐条发送消息
        print("\n📨 开始发送对话序列...")
        print("💡 策略说明：前4条消息构成完整对话场景1（咖啡偏好讨论）")
        print("💡 第5条消息通过5小时时间间隔+新主题触发边界检测")
        print("💡 这样可以确保场景1的记忆被成功提取")

        last_response = None
        for i, msg in enumerate(messages, 1):
            if i == 5:
                print(
                    f"\n🔄 --- 场景切换：发送第 {i}/{len(messages)} 条消息（触发边界） ---"
                )
            else:
                print(f"\n--- 发送第 {i}/{len(messages)} 条消息 ---")

            status_code, response = self.call_post_api("", msg)

            # 验证每条消息都成功处理
            assert (
                status_code == 200
            ), f"第 {i} 条消息状态码应该是 200，实际: {status_code}"
            assert response.get("status") == "ok", f"第 {i} 条消息状态应该是 ok"

            last_response = response

        # 使用最后一条消息的响应进行验证
        status_code = 200
        response = last_response

        # 断言：验证结果结构
        print("\n📊 验证对话记忆提取结果...")
        assert "result" in response, "成功响应应包含 result 字段"
        result = response["result"]
        assert "saved_memories" in result, "result 应包含 saved_memories 字段"
        assert "count" in result, "result 应包含 count 字段"
        assert "status_info" in result, "result 应包含 status_info 字段"

        # 验证 saved_memories 是列表
        assert isinstance(result["saved_memories"], list), "saved_memories 应该是列表"
        assert result["count"] >= 0, "count 应该 >= 0"
        assert result["status_info"] in [
            "accumulated",
            "extracted",
        ], "status_info 应该是 accumulated 或 extracted"

        # 如果有提取的记忆，验证每条记忆的结构
        if result["count"] > 0:
            print(f"\n✅ 成功提取 {result['count']} 条记忆！")
            print(f"✅ 边界检测成功：通过时间间隔(5小时)+主题切换触发")
            for idx, memory in enumerate(result["saved_memories"], 1):
                assert isinstance(memory, dict), f"第 {idx} 条记忆应该是字典"
                # 注意：不同的记忆类型可能有不同的字段结构
                # 这里只验证基本的字段存在
                memory_type = memory.get('memory_type', 'unknown')
                summary = memory.get('summary', memory.get('content', 'no summary'))[
                    :50
                ]
                print(f"  记忆 {idx}: {memory_type} - {summary}...")
        else:
            print(
                f"\n⚠️  消息已累积，等待边界检测（status_info: {result['status_info']}）"
            )
            print(f"   已发送 {len(messages)} 条消息，但可能未达到边界检测条件")
            print(f"   💡 提示：边界检测需要满足以下条件之一：")
            print(f"      1. 跨天（新消息与上一条消息日期不同）")
            print(f"      2. 长时间中断（超过4小时）+ 主题切换")
            print(f"      3. 明确的场景/主题切换信号")

        print(f"\n✅ Memorize 测试完成")
        return status_code, response

    def test_fetch_episodic(self):
        """测试2: GET /api/v1/memories - 获取用户情景记忆（episodic_memory类型）"""
        self.print_section("测试2: GET /api/v1/memories - 获取用户情景记忆")

        params = {
            "user_id": self.user_id,
            "memory_type": "episodic_memory",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_api("", params)

        # 断言：精确验证响应结构
        assert status_code == 200, f"状态码应该是 200，实际: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"状态应该是 ok，实际: {response.get('status')}"
        assert "result" in response, "响应应包含 result 字段"

        result = response["result"]
        assert "memories" in result, "result 应包含 memories 字段"
        assert "total_count" in result, "result 应包含 total_count 字段"
        assert "has_more" in result, "result 应包含 has_more 字段"
        assert "metadata" in result, "result 应包含 metadata 字段"

        # 验证数据类型
        assert isinstance(result["memories"], list), "memories 应该是列表"
        assert result["total_count"] >= 0, "total_count 应该 >= 0"
        assert isinstance(result["has_more"], bool), "has_more 应该是布尔值"

        # 验证 metadata 结构
        metadata = result["metadata"]
        assert isinstance(metadata, dict), "metadata 应该是字典"
        assert "source" in metadata, "metadata 应包含 source 字段"
        assert "user_id" in metadata, "metadata 应包含 user_id 字段"
        assert "memory_type" in metadata, "metadata 应包含 memory_type 字段"
        assert metadata.get("user_id") == self.user_id, "metadata 的 user_id 应该匹配"

        # 如果有记忆，深度验证结构
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert isinstance(memory, dict), f"第 {idx} 条记忆应该是字典"
                assert "user_id" in memory, f"第 {idx} 条记忆应包含 user_id"
                assert "timestamp" in memory, f"第 {idx} 条记忆应包含 timestamp"
                assert (
                    memory.get("user_id") == self.user_id
                ), f"第 {idx} 条记忆的 user_id 应该匹配"

            print(
                f"✅ Fetch Episodic 成功，返回 {result['total_count']} 条情景记忆，已验证深度结构"
            )
        else:
            print(f"✅ Fetch Episodic 成功，返回 {result['total_count']} 条情景记忆")

        return status_code, response

    def test_fetch_personal_semantic_memory(self):
        """测试3: GET /api/v1/memories - 获取个人语义记忆（personal_semantic_memory类型）"""
        self.print_section("测试3: GET /api/v1/memories - 获取个人语义记忆")

        params = {
            "user_id": self.user_id,
            "memory_type": "personal_semantic_memory",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_api("", params)

        # 断言：精确验证响应结构
        assert status_code == 200, f"状态码应该是 200，实际: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"状态应该是 ok，实际: {response.get('status')}"
        assert "result" in response, "响应应包含 result 字段"

        result = response["result"]
        assert "memories" in result, "result 应包含 memories 字段"
        assert "total_count" in result, "result 应包含 total_count 字段"
        assert "has_more" in result, "result 应包含 has_more 字段"
        assert "metadata" in result, "result 应包含 metadata 字段"

        # 验证数据类型
        assert isinstance(result["memories"], list), "memories 应该是列表"
        assert result["total_count"] >= 0, "total_count 应该 >= 0"
        assert isinstance(result["has_more"], bool), "has_more 应该是布尔值"

        # 验证 metadata 结构
        metadata = result["metadata"]
        assert isinstance(metadata, dict), "metadata 应该是字典"
        assert "source" in metadata, "metadata 应包含 source 字段"
        assert "user_id" in metadata, "metadata 应包含 user_id 字段"
        assert "memory_type" in metadata, "metadata 应包含 memory_type 字段"
        assert metadata.get("user_id") == self.user_id, "metadata 的 user_id 应该匹配"

        # 如果有记忆，深度验证结构
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert isinstance(memory, dict), f"第 {idx} 条记忆应该是字典"
                assert "content" in memory, f"第 {idx} 条记忆应包含 content"
                assert (
                    "parent_episode_id" in memory
                ), f"第 {idx} 条记忆应包含 parent_episode_id"
                # 个人语义记忆的user_id可能为None（群组场景），所以不强制检查

            print(
                f"✅ Fetch Personal Semantic Memory 成功，返回 {result['total_count']} 条个人语义记忆，已验证深度结构"
            )
        else:
            print(
                f"✅ Fetch Personal Semantic Memory 成功，返回 {result['total_count']} 条个人语义记忆"
            )

        return status_code, response

    def test_fetch_event_log(self):
        """测试4: GET /api/v1/memories - 获取用户事件日志（personal_event_log类型）"""
        self.print_section("测试4: GET /api/v1/memories - 获取用户事件日志")

        params = {
            "user_id": self.user_id,
            "memory_type": "personal_event_log",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_api("", params)

        # 断言：精确验证响应结构
        assert status_code == 200, f"状态码应该是 200，实际: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"状态应该是 ok，实际: {response.get('status')}"
        assert "result" in response, "响应应包含 result 字段"

        result = response["result"]
        assert "memories" in result, "result 应包含 memories 字段"
        assert "total_count" in result, "result 应包含 total_count 字段"
        assert "has_more" in result, "result 应包含 has_more 字段"
        assert "metadata" in result, "result 应包含 metadata 字段"

        # 验证数据类型
        assert isinstance(result["memories"], list), "memories 应该是列表"
        assert result["total_count"] >= 0, "total_count 应该 >= 0"
        assert isinstance(result["has_more"], bool), "has_more 应该是布尔值"

        # 验证 metadata 结构
        metadata = result["metadata"]
        assert isinstance(metadata, dict), "metadata 应该是字典"
        assert "source" in metadata, "metadata 应包含 source 字段"
        assert "user_id" in metadata, "metadata 应包含 user_id 字段"
        assert "memory_type" in metadata, "metadata 应包含 memory_type 字段"
        assert metadata.get("user_id") == self.user_id, "metadata 的 user_id 应该匹配"

        # 如果有事件日志，深度验证结构
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert isinstance(memory, dict), f"第 {idx} 条记忆应该是字典"
                assert "atomic_fact" in memory, f"第 {idx} 条记忆应包含 atomic_fact"
                assert "timestamp" in memory, f"第 {idx} 条记忆应包含 timestamp"
                assert "user_id" in memory, f"第 {idx} 条记忆应包含 user_id"
                assert (
                    memory.get("user_id") == self.user_id
                ), f"第 {idx} 条记忆的 user_id 应该匹配"

            print(
                f"✅ Fetch Event Log 成功，返回 {result['total_count']} 条事件日志，已验证深度结构"
            )
        else:
            print(f"✅ Fetch Event Log 成功，返回 {result['total_count']} 条事件日志")

        return status_code, response

    def test_search_memories_keyword(self):
        """测试5: GET /api/v1/memories/search - 关键词检索（通过 body 传参）"""
        self.print_section("测试5: GET /api/v1/memories/search - 关键词检索")

        # 注意：虽然路由定义是 GET，但实际实现从 body 读取参数
        # 类似 Elasticsearch 的搜索 API，GET 请求可以带 body
        data = {
            "user_id": self.user_id,
            "query": "咖啡",
            "top_k": 10,
            "retrieve_method": "keyword",
        }

        status_code, response = self.call_get_with_body_api("/search", data)

        # 断言：精确验证响应结构
        assert status_code == 200, f"状态码应该是 200，实际: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"状态应该是 ok，实际: {response.get('status')}"
        assert "result" in response, "响应应包含 result 字段"

        result = response["result"]
        assert "memories" in result, "result 应包含 memories 字段"
        assert "scores" in result, "result 应包含 scores 字段"
        assert "total_count" in result, "result 应包含 total_count 字段"
        assert "has_more" in result, "result 应包含 has_more 字段"
        assert "metadata" in result, "result 应包含 metadata 字段"

        # 验证数据类型
        assert isinstance(result["memories"], list), "memories 应该是列表"
        assert isinstance(result["scores"], list), "scores 应该是列表"
        assert result["total_count"] >= 0, "total_count 应该 >= 0"

        # 验证 metadata
        metadata = result["metadata"]
        assert metadata.get("user_id") == self.user_id, "metadata 的 user_id 应该匹配"

        # 如果有结果，深度验证嵌套结构
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            # 验证 memories 和 scores 数量一致
            assert len(result["memories"]) == len(
                result["scores"]
            ), "memories 和 scores 数量应该一致"

            # 遍历每个群组的记忆
            for group_idx, memory_group in enumerate(result["memories"]):
                assert isinstance(
                    memory_group, dict
                ), f"第 {group_idx} 个 memory_group 应该是字典"

                # 遍历群组内的记忆列表
                for group_id, memory_list in memory_group.items():
                    assert isinstance(group_id, str), f"group_id 应该是字符串"
                    assert isinstance(
                        memory_list, list
                    ), f"群组 {group_id} 的 memory_list 应该是列表"

                    # 验证每条记忆的基本字段
                    for mem_idx, mem in enumerate(memory_list):
                        assert isinstance(mem, dict), f"第 {mem_idx} 条记忆应该是字典"
                        assert (
                            "memory_type" in mem
                        ), f"第 {mem_idx} 条记忆应包含 memory_type"
                        assert "user_id" in mem, f"第 {mem_idx} 条记忆应包含 user_id"
                        assert (
                            "timestamp" in mem
                        ), f"第 {mem_idx} 条记忆应包含 timestamp"

            print(f"✅ Search Keyword 成功，返回 {result['total_count']} 个群组的记忆")
        else:
            print(f"✅ Search Keyword 成功，返回 {result['total_count']} 个群组的记忆")

        return status_code, response

    def test_search_memories_vector(self):
        """测试6: GET /api/v1/memories/search - 向量检索（通过 body 传参）"""
        self.print_section("测试6: GET /api/v1/memories/search - 向量检索")

        data = {
            "user_id": self.user_id,
            "query": "用户的饮食偏好",
            "top_k": 10,
            "retrieve_method": "vector",
        }

        status_code, response = self.call_get_with_body_api("/search", data)

        # 断言：精确验证响应结构
        assert status_code == 200, f"状态码应该是 200，实际: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"状态应该是 ok，实际: {response.get('status')}"
        assert "result" in response, "响应应包含 result 字段"

        result = response["result"]
        assert "memories" in result, "result 应包含 memories 字段"
        assert "scores" in result, "result 应包含 scores 字段"
        assert "total_count" in result, "result 应包含 total_count 字段"
        assert "has_more" in result, "result 应包含 has_more 字段"
        assert "metadata" in result, "result 应包含 metadata 字段"

        # 向量检索应该有 importance_scores
        if result["total_count"] > 0:
            assert (
                "importance_scores" in result
            ), "向量检索 result 应包含 importance_scores 字段"
            assert isinstance(
                result["importance_scores"], list
            ), "importance_scores 应该是列表"

        print(f"✅ Search Vector 成功，返回 {result['total_count']} 个群组的记忆")

        return status_code, response

    def test_search_memories_hybrid(self):
        """测试7: GET /api/v1/memories/search - 混合检索（通过 body 传参）"""
        self.print_section("测试7: GET /api/v1/memories/search - 混合检索")

        now = datetime.now(SHANGHAI_TZ)
        data = {
            "user_id": self.user_id,
            "query": "咖啡偏好",
            "top_k": 10,
            "retrieve_method": "hybrid",
            "start_time": (now - timedelta(days=60)).isoformat(),
            "end_time": now.isoformat(),
        }

        status_code, response = self.call_get_with_body_api("/search", data)

        # 断言：精确验证响应结构
        assert status_code == 200, f"状态码应该是 200，实际: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"状态应该是 ok，实际: {response.get('status')}"
        assert "result" in response, "响应应包含 result 字段"

        result = response["result"]
        assert "memories" in result, "result 应包含 memories 字段"
        assert "scores" in result, "result 应包含 scores 字段"
        assert "total_count" in result, "result 应包含 total_count 字段"
        assert "has_more" in result, "result 应包含 has_more 字段"
        assert "metadata" in result, "result 应包含 metadata 字段"

        # 混合检索应该有 importance_scores
        if result["total_count"] > 0:
            assert (
                "importance_scores" in result
            ), "混合检索 result 应包含 importance_scores 字段"
            assert isinstance(
                result["importance_scores"], list
            ), "importance_scores 应该是列表"

            # 验证 metadata 中的 source
            metadata = result["metadata"]
            assert (
                metadata.get("source") == "hybrid_retrieval"
            ), "混合检索的 source 应该是 hybrid_retrieval"

        print(f"✅ Search Hybrid 成功，返回 {result['total_count']} 个群组的记忆")

        return status_code, response

    def test_save_conversation_meta(self):
        """测试8: POST /api/v1/memories/conversation-meta - 保存对话元数据"""
        self.print_section(
            "测试8: POST /api/v1/memories/conversation-meta - 保存对话元数据"
        )

        now = datetime.now(SHANGHAI_TZ)
        data = {
            "version": "1.0",
            "scene": "assistant",
            "scene_desc": {
                "description": "项目协作群聊",
                "bot_ids": ["bot_001"],
                "extra": {"category": "test"},
            },
            "name": "测试项目讨论组",
            "description": "用于测试的项目讨论群组",
            "group_id": self.group_id,
            "created_at": now.isoformat(),
            "default_timezone": "Asia/Shanghai",
            "user_details": {
                self.user_id: {
                    "full_name": "测试用户",
                    "role": "developer",
                    "extra": {"department": "技术部"},
                }
            },
            "tags": ["测试", "项目"],
        }

        status_code, response = self.call_post_api("/conversation-meta", data)

        # 断言：精确验证响应结构
        assert status_code == 200, f"状态码应该是 200，实际: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"状态应该是 ok，实际: {response.get('status')}"
        assert "result" in response, "响应应包含 result 字段"

        result = response["result"]
        assert "id" in result, "result 应包含 id 字段"
        assert "group_id" in result, "result 应包含 group_id 字段"
        assert "scene" in result, "result 应包含 scene 字段"
        assert "name" in result, "result 应包含 name 字段"
        assert "version" in result, "result 应包含 version 字段"

        # 验证值的正确性
        assert result["group_id"] == self.group_id, "返回的 group_id 应该匹配"
        assert result["scene"] == "assistant", "返回的 scene 应该匹配"
        assert result["name"] == "测试项目讨论组", "返回的 name 应该匹配"

        print(f"✅ Save Conversation Meta 成功，id={result['id']}")

        return status_code, response

    def test_patch_conversation_meta(self):
        """测试9: PATCH /api/v1/memories/conversation-meta - 局部更新对话元数据"""
        self.print_section(
            "测试9: PATCH /api/v1/memories/conversation-meta - 局部更新对话元数据"
        )

        data = {
            "group_id": self.group_id,
            "name": "更新后的测试项目讨论组",
            "tags": ["测试", "项目", "更新"],
        }

        status_code, response = self.call_patch_api("/conversation-meta", data)

        # 断言：精确验证响应结构
        if status_code == 200:
            assert (
                response.get("status") == "ok"
            ), f"状态应该是 ok，实际: {response.get('status')}"
            assert "result" in response, "响应应包含 result 字段"

            result = response["result"]
            assert "id" in result, "result 应包含 id 字段"
            assert "group_id" in result, "result 应包含 group_id 字段"
            assert "updated_fields" in result, "result 应包含 updated_fields 字段"

            # 验证更新的字段
            assert result["group_id"] == self.group_id, "返回的 group_id 应该匹配"
            assert isinstance(
                result["updated_fields"], list
            ), "updated_fields 应该是列表"

            if len(result["updated_fields"]) > 0:
                print(
                    f"✅ Patch Conversation Meta 成功，更新了 {len(result['updated_fields'])} 个字段: {result['updated_fields']}"
                )
            else:
                print("✅ Patch Conversation Meta 成功，没有字段需要更新")
        elif status_code == 404:
            print(
                f"⚠️  Patch Conversation Meta: 对话元数据不存在（需要先调用 POST 创建）"
            )
        else:
            print(
                f"⚠️  Patch Conversation Meta 失败: {response.get('message', 'Unknown error')}"
            )

        return status_code, response

    def run_all_tests(self, test_method: str = "all"):
        """
        运行测试

        Args:
            test_method: 指定要运行的测试方法，可选值：
                - all: 运行所有测试
                - memorize: 测试存储对话记忆
                - fetch_episodic: 测试获取情景记忆
                - fetch_event_log: 测试获取事件日志
                - fetch_profile: 测试获取用户画像
                - search_keyword: 测试关键词检索
                - search_vector: 测试向量检索
                - search_hybrid: 测试混合检索
                - save_meta: 测试保存对话元数据
                - patch_meta: 测试更新对话元数据
        """
        print("\n" + "=" * 80)
        print("  开始执行 Memory Controller API 测试")
        print("=" * 80)
        print(f"  API地址: {self.base_url}")
        print(f"  测试用户: {self.user_id}")
        print(f"  测试群组: {self.group_id}")
        print(f"  测试方法: {test_method}")
        print("=" * 80)

        # 定义测试方法映射
        test_methods = {
            "memorize": self.test_memorize_single_message,
            "fetch_episodic": self.test_fetch_episodic,
            "fetch_semantic": self.test_fetch_personal_semantic_memory,
            "fetch_event_log": self.test_fetch_event_log,
            "search_keyword": self.test_search_memories_keyword,
            "search_vector": self.test_search_memories_vector,
            "search_hybrid": self.test_search_memories_hybrid,
            "save_meta": self.test_save_conversation_meta,
            "patch_meta": self.test_patch_conversation_meta,
        }

        # 执行测试
        try:
            if test_method == "all":
                # 运行所有测试
                for method in test_methods.values():
                    method()
            elif test_method in test_methods:
                # 运行指定的单个测试
                test_methods[test_method]()
            else:
                print(f"\n❌ 未知的测试方法: {test_method}")
                return
        except AssertionError as e:
            print(f"\n❌ 测试失败: {e}")
            raise
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ 测试异常: {e}")
            raise

        # 测试完成
        self.print_section("测试完成")
        if test_method == "all":
            print("\n✅ 所有接口结构验证通过！")
        else:
            print(f"\n✅ 测试方法 [{test_method}] 验证通过！")
        print("💡 提示: 如果某个接口失败，请检查输入输出结构是否发生变化\n")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Memory Controller API 测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认配置测试本地服务
  python tests/test_memory_controller.py

  # 指定API地址
  python tests/test_memory_controller.py --base-url http://localhost:1995

  # 指定API地址和测试用户
  python tests/test_memory_controller.py --base-url http://dev-server:1995 --user-id test_user_123

  # 单独测试某个方法
  python tests/test_memory_controller.py --test-method memorize
  python tests/test_memory_controller.py --test-method fetch_episodic
  python tests/test_memory_controller.py --test-method fetch_event_log
  python tests/test_memory_controller.py --test-method search_keyword

  # 指定所有参数
  python tests/test_memory_controller.py --base-url http://dev-server:1995 --user-id test_user --group-id test_group --timeout 60
        """,
    )

    parser.add_argument(
        "--base-url",
        default="http://localhost:1995",
        help="API基础URL (默认: http://localhost:1995)",
    )

    parser.add_argument("--user-id", default=None, help="测试用户ID (默认: 随机生成)")

    parser.add_argument("--group-id", default=None, help="测试群组ID (默认: 随机生成)")

    parser.add_argument(
        "--timeout", type=int, default=180, help="请求超时时间(秒) (默认: 180)"
    )

    parser.add_argument(
        "--test-method",
        default="all",
        choices=[
            "all",
            "memorize",
            "fetch_episodic",
            "fetch_semantic",
            "fetch_event_log",
            "fetch_profile",
            "search_keyword",
            "search_vector",
            "search_hybrid",
            "save_meta",
            "patch_meta",
        ],
        help="指定要运行的测试方法 (默认: all 运行所有测试)",
    )

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 如果未提供 user_id，随机生成一个
    user_id = args.user_id if args.user_id else f"user_{uuid.uuid4().hex[:12]}"

    # 如果未提供 group_id，随机生成一个
    group_id = args.group_id if args.group_id else f"group_{uuid.uuid4().hex[:12]}"

    # 输出使用的ID信息
    if not args.user_id:
        print(f"⚠️  未提供 --user-id，自动生成: {user_id}")
    if not args.group_id:
        print(f"⚠️  未提供 --group-id，自动生成: {group_id}")

    # 创建测试器实例
    tester = MemoryControllerTester(
        base_url=args.base_url, user_id=user_id, group_id=group_id, timeout=args.timeout
    )

    # 运行测试（根据参数决定运行全部还是单个）
    tester.run_all_tests(test_method=args.test_method)


if __name__ == "__main__":
    main()
