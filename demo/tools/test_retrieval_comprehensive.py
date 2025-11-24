"""全面的记忆检索测试

测试所有检索模式的组合：
- 数据源：episode、event_log、semantic_memory
- 记忆范围：personal、group、all
- 检索模式：bm25、embedding、rrf
- Profile 数据源：仅测试固定的 user_id + group_id 组合（不区分 memory_scope / 检索模式）

使用方法：
    # 确保 API 服务器已启动
    uv run python src/bootstrap.py src/run.py --port 8001
    
    # 在另一个终端运行测试
    uv run python src/bootstrap.py demo/tools/test_retrieval_comprehensive.py
"""

import asyncio
import httpx
from typing import List, Dict, Any
from datetime import datetime
import time


class RetrievalTester:
    """全面的检索测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """初始化测试器
        
        Args:
            base_url: API 服务器地址
        """
        self.base_url = base_url
        self.retrieve_url = f"{base_url}/api/v3/agentic/retrieve_lightweight"
        
        # 测试配置
        self.data_sources = ["episode", "event_log", "semantic_memory", "profile"]
        self.memory_scopes = ["personal", "group"]
        self.retrieval_modes = ["embedding", "bm25", "rrf"]
        
        # 测试结果统计
        self.total_tests = 0
        self.successful_tests = 0
        self.failed_tests = 0
        self.test_results = []
        
        # 计时统计
        self.start_time = None
        self.end_time = None
        self.total_request_time = 0.0  # 所有请求的累计耗时
        self.max_latency = 0.0  # 最大延迟
        self.min_latency = float('inf')  # 最小延迟
    
    async def test_retrieval(
        self,
        query: str,
        data_source: str,
        memory_scope: str,
        retrieval_mode: str,
        user_id: str = "test_user",
        group_id: str = None,
        top_k: int = 5,
        current_time: str = None,
        allow_empty: bool = False,
    ) -> Dict[str, Any]:
        """执行单次检索测试
        
        Args:
            query: 查询文本
            data_source: 数据源（episode/event_log/semantic_memory/profile）
            memory_scope: 记忆范围（personal/group）
            retrieval_mode: 检索模式（embedding/bm25/rrf）
            user_id: 用户ID
            group_id: 群组ID
            top_k: 返回结果数量
            current_time: 当前时间（仅对 semantic_memory 有效）
            
        Returns:
            测试结果字典
        """
        self.total_tests += 1
        
        # 记录单次请求开始时间
        request_start_time = time.time()
        
        # 构建请求参数
        payload = {
            "query": query,
            "user_id": user_id,
            "group_id": group_id,
            "top_k": top_k,
            "data_source": data_source,
            "retrieval_mode": retrieval_mode,
        }
        
        # 添加可选参数
        if current_time and data_source == "semantic_memory":
            payload["current_time"] = current_time
        
        test_name = f"{data_source}_{memory_scope}_{retrieval_mode}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.retrieve_url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # 计算单次请求耗时
                request_elapsed = (time.time() - request_start_time) * 1000  # 转为毫秒
                self.total_request_time += request_elapsed
                
                if result.get("status") == "ok":
                    memories = result.get("result", {}).get("memories", [])
                    metadata = result.get("result", {}).get("metadata", {})
                    latency = metadata.get("total_latency_ms", 0)
                    
                    # 更新最大/最小延迟
                    if latency > 0:
                        self.max_latency = max(self.max_latency, latency)
                        self.min_latency = min(self.min_latency, latency)
                    
                    if len(memories) == 0:
                        if allow_empty:
                            self.successful_tests += 1
                            info_msg = f"{test_name}: 允许空结果（耗时 {latency:.2f}ms）"
                            print(f"  ✅ {info_msg}")
                            empty_result = {
                                "test_name": test_name,
                                "status": "✅ 成功",
                                "query": query,
                                "data_source": data_source,
                                "retrieval_mode": retrieval_mode,
                                "count": 0,
                                "latency_ms": latency,
                                "metadata": metadata,
                                "memories": [],
                                "note": "allow_empty",
                            }
                            return empty_result
                        # 将 0 条结果视为失败，方便定位问题
                        self.failed_tests += 1
                        warning_msg = f"{test_name}: 返回 0 条记忆（耗时 {latency:.2f}ms）"
                        print(f"  ⚠️ {warning_msg}")
                        return {
                            "test_name": test_name,
                            "status": "⚠️ 空结果",
                            "query": query,
                            "data_source": data_source,
                            "retrieval_mode": retrieval_mode,
                            "count": 0,
                            "latency_ms": latency,
                            "metadata": metadata,
                            "memories": [],
                        }
                    
                    self.successful_tests += 1
                    test_result = {
                        "test_name": test_name,
                        "status": "✅ 成功",
                        "query": query,
                        "data_source": data_source,
                        "retrieval_mode": retrieval_mode,
                        "count": len(memories),
                        "latency_ms": latency,
                        "request_time_ms": request_elapsed,  # 添加完整请求耗时
                        "metadata": metadata,
                        "memories": memories[:3],  # 只保存前3条
                    }
                    
                    # 打印分数（前3条）
                    score_info = ""
                    scores = [f"{m.get('score', 0):.4f}" for m in memories[:3]]
                    score_info = f"，分数: [{', '.join(scores)}]"
                    
                    print(f"  ✅ {test_name}: 找到 {len(memories)} 条记忆，API耗时 {latency:.2f}ms，总耗时 {request_elapsed:.2f}ms{score_info}")
                    
                    if data_source == "profile" and memories:
                        profile_entry = memories[0]
                        profile_data = profile_entry.get("profile") or {}
                        print("    👤 Profile 详情（第一条样例）:")
                        print(
                            f"      user_id={profile_entry.get('user_id')}, "
                            f"group_id={profile_entry.get('group_id')}, "
                            f"version={profile_entry.get('version')}, "
                            f"scenario={profile_entry.get('scenario')}, "
                            f"updated_at={profile_entry.get('updated_at')}"
                        )
                        summary_text = profile_data.get("summary") or profile_data.get("output_reasoning")
                        if summary_text:
                            short_summary = summary_text[:80] + ("..." if len(summary_text) > 80 else "")
                            print(f"      摘要: {short_summary}")
                        interests = profile_data.get("interests") or []
                        if interests:
                            interest_names = ", ".join(
                                [
                                    item.get("value")
                                    for item in interests[:3]
                                    if isinstance(item, dict) and item.get("value")
                                ]
                            )
                            if interest_names:
                                print(f"      兴趣: {interest_names}")
                    
                    return test_result
                else:
                    self.failed_tests += 1
                    error_msg = result.get('message', '未知错误')
                    print(f"  ❌ {test_name}: 检索失败 - {error_msg}")
                    return {
                        "test_name": test_name,
                        "status": "❌ 失败",
                        "error": error_msg,
                    }
                    
        except httpx.ConnectError:
            self.failed_tests += 1
            print(f"  ❌ {test_name}: 无法连接到 API 服务器")
            return {
                "test_name": test_name,
                "status": "❌ 连接失败",
                "error": "无法连接到 API 服务器",
            }
        except Exception as e:
            self.failed_tests += 1
            print(f"  ❌ {test_name}: 异常 - {e}")
            return {
                "test_name": test_name,
                "status": "❌ 异常",
                "error": str(e),
            }
    
    async def run_comprehensive_test(
        self,
        query: str,
        user_id: str = "test_user",
        group_id: str = None,
        current_time: str = None,
        query_overrides: Dict[str, str] | None = None,
        profile_group_id: str | None = None,
    ):
        """运行全面的检索测试
        
        Args:
            query: 查询文本
            user_id: 用户ID
            group_id: 群组ID
            current_time: 当前时间（YYYY-MM-DD格式）
        """
        # 记录测试开始时间
        if self.start_time is None:
            self.start_time = time.time()
        
        print("\n" + "="*80)
        print(f"🧪 开始全面检索测试")
        print(f"   查询: {query}")
        print(f"   用户ID: {user_id}")
        print(f"   群组ID: {group_id or '无'}")
        print(f"   当前时间: {current_time or '无'}")
        print("="*80)
        
        # 遍历所有组合
        query_overrides = query_overrides or {}
        for data_source in self.data_sources:
            print(f"\n📊 数据源: {data_source}")
            print("-"*80)
            
            if data_source == "profile":
                profile_gid = profile_group_id or group_id
                if not profile_gid:
                    print("  ⚠️ 跳过 profile 测试：缺少 group_id")
                    continue
                
                effective_query = query_overrides.get(data_source, query)
                print("\n  📁 记忆范围: user_id + group_id（固定）")
                result = await self.test_retrieval(
                    query=effective_query or "",
                    data_source="profile",
                    memory_scope="group",
                    retrieval_mode="rrf",
                    user_id="user_001",
                    group_id=profile_gid,
                    current_time=current_time,
                )
                self.test_results.append(result)
                continue
            
            for memory_scope in self.memory_scopes:
                if memory_scope == "group":
                    user_id = None
                    group_id = "chat_user_001_assistant"
                if memory_scope == "personal":
                    user_id = "user_001"
                    group_id = "chat_user_001_assistant"
                print(f"\n  📁 记忆范围: {memory_scope}")
                
                for retrieval_mode in self.retrieval_modes:
                    effective_query = query_overrides.get(data_source, query)
                    effective_group_id = group_id
                    if data_source == "profile":
                        effective_group_id = profile_group_id or group_id
                        if effective_group_id is None:
                            print("  ⚠️ 跳过 profile 测试：缺少 group_id")
                            continue
                    result = await self.test_retrieval(
                        query=effective_query,
                        memory_scope=memory_scope,
                        data_source=data_source,
                        retrieval_mode=retrieval_mode,
                        user_id=user_id,
                        group_id=effective_group_id,
                        current_time=current_time,
                    )
                    self.test_results.append(result)
                    
                    # 短暂延迟，避免请求过快
    
    def print_summary(self):
        """打印测试总结"""
        # 记录测试结束时间
        if self.end_time is None:
            self.end_time = time.time()
        
        total_elapsed = self.end_time - self.start_time if self.start_time else 0
        
        print("\n" + "="*80)
        print("📊 测试总结")
        print("="*80)
        print(f"总测试数: {self.total_tests}")
        print(f"成功: {self.successful_tests} ✅")
        print(f"失败: {self.failed_tests} ❌")
        print(f"成功率: {(self.successful_tests/self.total_tests*100):.1f}%")
        
        # ⏱️ 计时统计
        print("\n⏱️  性能统计:")
        print(f"  总测试耗时: {total_elapsed:.2f}秒")
        print(f"  总请求耗时: {self.total_request_time/1000:.2f}秒")
        print(f"  平均请求耗时: {self.total_request_time/self.total_tests:.2f}ms" if self.total_tests > 0 else "  平均请求耗时: N/A")
        print(f"  最大API延迟: {self.max_latency:.2f}ms" if self.max_latency > 0 else "  最大API延迟: N/A")
        print(f"  最小API延迟: {self.min_latency:.2f}ms" if self.min_latency != float('inf') else "  最小API延迟: N/A")
        
        # 按数据源分组统计
        print("\n📈 按数据源分组:")
        for data_source in self.data_sources:
            source_results = [r for r in self.test_results if r.get("data_source") == data_source]
            success = len([r for r in source_results if r.get("status") == "✅ 成功"])
            total = len(source_results)
            avg_count = sum(r.get("count", 0) for r in source_results if r.get("count")) / total if total > 0 else 0
            avg_latency = sum(r.get("latency_ms", 0) for r in source_results if r.get("latency_ms")) / total if total > 0 else 0
            print(f"  {data_source}: {success}/{total} 成功，平均 {avg_count:.1f} 条，平均API耗时 {avg_latency:.2f}ms")
        
        # 按检索模式分组统计
        print("\n🔍 按检索模式分组:")
        for mode in self.retrieval_modes:
            mode_results = [r for r in self.test_results if r.get("retrieval_mode") == mode]
            success = len([r for r in mode_results if r.get("status") == "✅ 成功"])
            total = len(mode_results)
            avg_latency = sum(r.get("latency_ms", 0) for r in mode_results if r.get("latency_ms")) / total if total > 0 else 0
            print(f"  {mode}: {success}/{total} 成功，平均耗时 {avg_latency:.2f}ms")
        
        # 按记忆范围分组统计
        print("\n📁 按记忆范围分组:")
        for scope in self.memory_scopes:
            scope_results = [r for r in self.test_results if r.get("memory_scope") == scope]
            success = len([r for r in scope_results if r.get("status") == "✅ 成功"])
            total = len(scope_results)
            avg_count = sum(r.get("count", 0) for r in scope_results if r.get("count")) / total if total > 0 else 0
            print(f"  {scope}: {success}/{total} 成功，平均返回 {avg_count:.1f} 条记忆")
        
        # 失败的测试详情
        failed_results = [r for r in self.test_results if r.get("status") != "✅ 成功"]
        if failed_results:
            print("\n❌ 失败的测试:")
            for r in failed_results:
                print(f"  - {r.get('test_name')}: {r.get('error', '未知错误')}")
    
    def export_results(self, output_file: str = "demo/results/retrieval_test_results.json"):
        """导出测试结果到 JSON 文件
        
        Args:
            output_file: 输出文件路径
        """
        import json
        from pathlib import Path
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 导出数据添加计时信息
        export_data = {
            "test_time": datetime.now().isoformat(),
            "summary": {
                "total_tests": self.total_tests,
                "successful_tests": self.successful_tests,
                "failed_tests": self.failed_tests,
                "success_rate": f"{(self.successful_tests/self.total_tests*100):.1f}%" if self.total_tests > 0 else "0%",
                "total_elapsed_seconds": round(self.end_time - self.start_time, 2) if self.start_time and self.end_time else 0,
                "total_request_time_ms": round(self.total_request_time, 2),
                "avg_request_time_ms": round(self.total_request_time / self.total_tests, 2) if self.total_tests > 0 else 0,
                "max_latency_ms": round(self.max_latency, 2) if self.max_latency > 0 else 0,
                "min_latency_ms": round(self.min_latency, 2) if self.min_latency != float('inf') else 0,
            },
            "test_results": self.test_results,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 测试结果已保存到: {output_file}")


async def main():
    """主测试函数"""
    
    # 记录整体测试开始时间
    overall_start_time = time.time()
    
    print("="*80)
    print("🧪 全面的记忆检索测试")
    print("="*80)
    print("\n本测试将系统地测试所有检索模式的组合：")
    print("  - 数据源: episode, event_log, semantic_memory（全量 3×3×3 组合）")
    print("  - Profile 数据源: 仅固定 user_id + group_id 的 direct 检索")
    print("  - 检索模式: embedding, bm25, rrf（仅适用于非 profile 数据源）")
    print(f"\n总测试数: 3 × 3 × 3 + profile(1) = 28 种组合（profile 若缺少 group_id 将跳过）")
    print("\n⚠️  请确保 API 服务器已启动: uv run python src/bootstrap.py src/run.py --port 8001")
    print("\n按 Enter 继续...")
    input()
    
    # 创建测试器
    tester = RetrievalTester()
    
    # ========== 测试 1: 个人记忆查询 ==========
    print("\n" + "🔬"*40)
    print("测试场景 1: 个人记忆查询")
    print("🔬"*40)
    test1_start = time.time()
    
    await tester.run_comprehensive_test(
        query="运动",
        user_id="user_001",  # 使用实际数据库中的 user_id
        group_id=None,  # 不指定 group_id
        current_time=None,  # 不传 current_time,避免过滤掉已过期的群组语义记忆
        query_overrides={
            # "event_log": "Beijing travel and food recommendation",  # 注释掉英文查询
            "profile": "profile summary",
        },
        profile_group_id="chat_user_001_assistant",
    )
    test1_elapsed = time.time() - test1_start
    print(f"\n⏱️  场景 1 耗时: {test1_elapsed:.2f}秒")
    
    # ========== 测试 2: 群组记忆查询 ==========
    print("\n" + "🔬"*40)
    print("测试场景 2: 群组记忆查询")
    print("🔬"*40)
    test2_start = time.time()
    
    await tester.run_comprehensive_test(
        query="运动",
        user_id="user_001",  # 使用实际数据库中的 user_id
        group_id="chat_user_001_assistant",  # 使用实际数据库中的 group_id
        current_time=None,  # 不传 current_time,避免过滤掉已过期的群组语义记忆
        query_overrides={
            # "event_log": "Beijing food and travel",  # 注释掉英文查询
            "profile": "profile summary",
        },
        profile_group_id="chat_user_001_assistant",
    )
    test2_elapsed = time.time() - test2_start
    print(f"\n⏱️  场景 2 耗时: {test2_elapsed:.2f}秒")
    
    # ========== 测试 3: 语义记忆专项测试（有效期过滤） ==========
    print("\n" + "🔬"*40)
    print("测试场景 3: 语义记忆有效期过滤")
    print("🔬"*40)
    test3_start = time.time()
    
    # 测试当前有效的语义记忆
    print("\n  📅 子测试 3.1: 检索当前有效的语义记忆")
    result_current = await tester.test_retrieval(
        query="运动",
        data_source="semantic_memory",
        memory_scope="personal",
        retrieval_mode="rrf",
        user_id="user_001",  # 使用实际数据库中的 user_id
        current_time=datetime.now().strftime("%Y-%m-%d"),
    )
    
    # 测试未来时间（应该返回更多记忆）
    print("\n  📅 子测试 3.2: 检索未来时间的语义记忆（包含更长期的预测）")
    result_future = await tester.test_retrieval(
        query="运动",
        data_source="semantic_memory",
        memory_scope="personal",
        retrieval_mode="rrf",
        user_id="user_001",  # 使用实际数据库中的 user_id
        current_time="2027-12-31",  # 未来时间
        allow_empty=True,
    )
    
    # 测试过去时间（应该返回较少记忆）
    print("\n  📅 子测试 3.3: 检索过去时间的语义记忆（已过期的记忆）")
    result_past = await tester.test_retrieval(
        query="运动",
        data_source="semantic_memory",
        memory_scope="personal",
        retrieval_mode="rrf",
        user_id="user_001",  # 使用实际数据库中的 user_id
        current_time="2024-01-01",  # 过去时间
        allow_empty=True,
    )
    
    test3_elapsed = time.time() - test3_start
    
    print(f"\n  📊 时间过滤效果对比:")
    print(f"     过去时间(2024-01-01): {result_past.get('count', 0)} 条")
    print(f"     当前时间({datetime.now().strftime('%Y-%m-%d')}): {result_current.get('count', 0)} 条")
    print(f"     未来时间(2027-12-31): {result_future.get('count', 0)} 条")
    print(f"\n⏱️  场景 3 耗时: {test3_elapsed:.2f}秒")
    
    # ========== 打印总结 ==========
    tester.print_summary()
    
    # 总体耗时
    overall_elapsed = time.time() - overall_start_time
    print(f"\n⏱️  总体测试耗时: {overall_elapsed:.2f}秒")
    print(f"   场景 1: {test1_elapsed:.2f}秒 ({test1_elapsed/overall_elapsed*100:.1f}%)")
    print(f"   场景 2: {test2_elapsed:.2f}秒 ({test2_elapsed/overall_elapsed*100:.1f}%)")
    print(f"   场景 3: {test3_elapsed:.2f}秒 ({test3_elapsed/overall_elapsed*100:.1f}%)")
    
    # ========== 导出结果 ==========
    tester.export_results()
    
    print("\n" + "="*80)
    print("✅ 全面检索测试完成！")
    print("="*80)


async def demo_semantic_memory_evidence():
    """演示语义记忆的 evidence 字段用法"""
    
    print("\n" + "="*80)
    print("💡 语义记忆 Evidence 字段演示")
    print("="*80)
    
    base_url = "http://localhost:8001"
    retrieve_url = f"{base_url}/api/v3/agentic/retrieve_lightweight"
    
    print("\n📖 场景说明:")
    print("   用户拔了智齿 → 系统生成语义记忆：'会优先选择软质食物'")
    print("   Evidence 字段存储原因：'刚拔除智齿'")
    print("   当用户查询'推荐食物'时，可以看到推荐依据")
    
    payload = {
        "query": "运动",
        "user_id": "robot_001",  # 使用实际数据库中的 user_id
        "data_source": "semantic_memory",
        "retrieval_mode": "rrf",
        "top_k": 5,
        "current_time": datetime.now().strftime("%Y-%m-%d"),
    }
    
    print(f"\n🔍 查询: {payload['query']}")
    print(f"   数据源: semantic_memory")
    print(f"   当前时间: {payload['current_time']}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(retrieve_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "ok":
                memories = result.get("result", {}).get("memories", [])
                metadata = result.get("result", {}).get("metadata", {})
                
                print(f"\n✅ 检索成功: 找到 {len(memories)} 条语义记忆")
                print(f"   耗时: {metadata.get('total_latency_ms', 0):.2f}ms")
                
                if memories:
                    print("\n📝 语义记忆详情（包含 evidence）:")
                    for i, mem in enumerate(memories[:5], 1):
                        print(f"\n  [{i}] 相关度: {mem.get('score', 0):.4f}")
                        print(f"      内容: {mem.get('episode', '')[:100]}")
                        
                        # 重点展示 evidence 字段
                        evidence = mem.get('evidence', '')
                        if evidence:
                            print(f"      🔍 证据: {evidence}")
                        
                        # 展示时间范围
                        timestamp = mem.get('timestamp', '')
                        if timestamp:
                            if isinstance(timestamp, str):
                                print(f"      ⏰ 时间: {timestamp[:10]}")
                            else:
                                print(f"      ⏰ 时间: {timestamp}")
                        
                        # 展示元数据
                        metadata_detail = mem.get('metadata', {})
                        if metadata_detail:
                            print(f"      📋 元数据: {metadata_detail}")
                else:
                    print("\n  💡 未找到相关语义记忆")
                    print("     可能原因:")
                    print("     1. 还没有生成语义记忆（需要先运行 extract_memory.py）")
                    print("     2. 查询与现有语义记忆不相关")
                    print("     3. 语义记忆已过期（end_time < current_time）")
            else:
                print(f"\n❌ 检索失败: {result.get('message')}")
                
    except httpx.ConnectError:
        print(f"\n❌ 无法连接到 API 服务器 ({base_url})")
        print("   请先启动服务: uv run python src/bootstrap.py src/run.py --port 8001")
    except Exception as e:
        print(f"\n❌ 异常: {e}")


async def main_menu():
    """主菜单"""
    
    print("\n" + "="*80)
    print("🧪 记忆检索测试工具")
    print("="*80)
    print("\n选择测试模式:")
    print("  1. 全面检索测试（27种组合）")
    print("  2. 语义记忆 Evidence 演示")
    print("  3. 两者都运行")
    print("\n⚠️  注意: 请确保已有测试数据（运行过 extract_memory.py）")
    print("\n请输入选项 (1/2/3): ", end="")
    
    choice = input().strip()
    
    if choice == "1":
        await main()
    elif choice == "2":
        await demo_semantic_memory_evidence()
    elif choice == "3":
        await main()
        await demo_semantic_memory_evidence()
    else:
        print("❌ 无效选项，请重新运行")


if __name__ == "__main__":
    asyncio.run(main_menu())

