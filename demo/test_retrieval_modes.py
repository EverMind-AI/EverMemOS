"""测试不同的检索模式和数据源

演示 V3 API 的多种检索配置：
1. 检索模式：embedding / bm25 / rrf
2. 数据源：memcell / event_log
"""

import asyncio

from core.di import get_bean_by_type
from infra_layer.adapters.input.api.v3.agentic_v3_controller import (
    AgenticV3Controller,
)
from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
    ConversationMetaRawRepository,
)


class MockRequest:
    def __init__(self, data):
        self.data = data
    
    async def json(self):
        return self.data


async def test_all_modes():
    """测试所有检索模式和数据源组合"""
    print("=" * 80)
    print("测试不同的检索模式和数据源")
    print("=" * 80)
    
    # 通过依赖注入获取 ConversationMetaRawRepository
    repository = get_bean_by_type(ConversationMetaRawRepository)
    controller = AgenticV3Controller(repository)
    query = "北京旅游美食"
    group_id = "assistant"
    
    # 测试配置矩阵
    test_configs = [
        ("rrf", "memcell", "RRF 融合 + MemCell Episode"),
        ("embedding", "memcell", "纯向量 + MemCell Episode"),
        ("bm25", "memcell", "纯BM25 + MemCell Episode"),
        ("rrf", "event_log", "RRF 融合 + Event Log"),
        ("embedding", "event_log", "纯向量 + Event Log"),
        ("bm25", "event_log", "纯BM25 + Event Log"),
    ]
    
    results_summary = []
    
    for retrieval_mode, data_source, desc in test_configs:
        print(f"\n{'='*80}")
        print(f"[测试] {desc}")
        print(f"  - retrieval_mode: {retrieval_mode}")
        print(f"  - data_source: {data_source}")
        print(f"{'='*80}")
        
        request = MockRequest({
            "query": query,
            "group_id": group_id,
            "top_k": 5,
            "retrieval_mode": retrieval_mode,
            "data_source": data_source,
        })
        
        response = await controller.retrieve_lightweight(request)
        result = response.get("result", {})
        metadata = result.get("metadata", {})
        memories = result.get("memories", [])
        
        print(f"\n结果:")
        print(f"  - 找到: {len(memories)} 条")
        print(f"  - 耗时: {metadata.get('total_latency_ms', 0):.2f} ms")
        
        if retrieval_mode == "embedding":
            print(f"  - Embedding 候选: {metadata.get('emb_count', 0)} 条")
        elif retrieval_mode == "bm25":
            print(f"  - BM25 候选: {metadata.get('bm25_count', 0)} 条")
        else:  # rrf
            print(f"  - Embedding 候选: {metadata.get('emb_count', 0)} 条")
            print(f"  - BM25 候选: {metadata.get('bm25_count', 0)} 条")
        
        if memories:
            print(f"\n  Top 3:")
            for i, mem in enumerate(memories[:3], 1):
                score = mem.get('score', 0)
                subject = mem.get('subject', 'N/A')
                episode = mem.get('episode', '')[:80]
                
                # 根据检索模式显示合适的分数格式
                if retrieval_mode == "embedding":
                    score_desc = f"余弦相似度={score:.4f}"
                elif retrieval_mode == "bm25":
                    score_desc = f"BM25分数={score:.4f}"
                else:
                    score_desc = f"RRF分数={score:.4f}"
                
                print(f"    [{i}] {score_desc}")
                print(f"        {subject}")
                print(f"        {episode}...")
        
        results_summary.append({
            "config": desc,
            "count": len(memories),
            "latency_ms": metadata.get('total_latency_ms', 0),
        })
    
    # 汇总对比
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)
    print(f"\n{'配置':<40} {'结果数':<10} {'耗时(ms)':<15}")
    print("-" * 80)
    for summary in results_summary:
        print(f"{summary['config']:<40} {summary['count']:<10} {summary['latency_ms']:<15.2f}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    
    print("\n💡 使用建议:")
    print("  - **纯向量 (embedding)**: 语义理解最强，适合模糊查询")
    print("  - **纯BM25 (bm25)**: 关键词精确匹配，适合专有名词查询")
    print("  - **RRF融合 (rrf)**: 综合最佳，推荐默认使用")
    print("\n  - **MemCell检索**: 查找完整对话，适合获取上下文")
    print("  - **Event Log检索**: 查找具体事实，适合精确信息提取")


if __name__ == "__main__":
    asyncio.run(test_all_modes())

