"""
Memory Retrieval Module

提供以下功能:
1. lightweight_retrieval: 轻量级 BM25-only 检索 (无 LLM 调用, 纯本地计算)
2. reranker_search: Reranker 重排序 (被 scene_retrieval 调用)

核心检索模式 (agentic) 在 scene_retrieval.py 中实现。
底层搜索算法 (BM25, Embedding, RRF, MaxSim) 在 retrieval_utils.py 中实现。
"""

from typing import List, Tuple
import time
import asyncio

from evaluation.src.adapters.evermemos.config import ExperimentConfig
from evaluation.src.adapters.evermemos.retrieval_utils import search_with_bm25_index
from agentic_layer import rerank_service


# ==========================================================================
# Lightweight Retrieval (BM25-only)
# ==========================================================================

async def lightweight_retrieval(
    query: str,
    emb_index,
    bm25,
    docs,
    config: ExperimentConfig,
    fact_to_doc_idx: list = None,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """
    轻量级 BM25-only 检索 (无 LLM 调用, 纯词法匹配)。

    适用场景:
    - 延迟敏感
    - 预算受限
    - 简单关键词查询

    Args:
        query: 用户查询
        emb_index: Embedding 索引 (未使用, 保留接口兼容)
        bm25: BM25 索引
        docs: 文档列表
        config: 实验配置
        fact_to_doc_idx: fact 索引到文档索引的映射 (用于 MaxSim 聚合)

    Returns:
        (final_results, metadata)
    """
    start_time = time.time()

    metadata = {
        "retrieval_mode": "lightweight",
        "bm25_count": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }

    bm25_results = await asyncio.to_thread(
        search_with_bm25_index,
        query,
        bm25,
        docs,
        config.lightweight_bm25_top_n,
        fact_to_doc_idx,
    )
    metadata["bm25_count"] = len(bm25_results)
    final_results = bm25_results[: config.lightweight_final_top_n]

    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000

    return final_results, metadata


# ==========================================================================
# Reranker Search
# ==========================================================================

async def reranker_search(
    query: str,
    results: List[Tuple[dict, float]],
    top_n: int = 20,
    reranker_instruction: str = None,
    batch_size: int = 10,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    timeout: float = 30.0,
    fallback_threshold: float = 0.3,
    config: ExperimentConfig = None,
):
    """
    使用 Reranker 模型对检索结果重排序 (批量并发 + 重试 + 回退)。

    Rerank 统一使用 episode memory 字段 (非 atomic facts)。

    策略:
    - 按 batch_size 分批处理
    - 每批支持重试 + 指数退避
    - 成功率过低时自动降级为原始排序
    - 单批超时保护

    Args:
        query: 用户查询
        results: 初始检索结果 [(doc, score), ...]
        top_n: 返回结果数
        reranker_instruction: Reranker 指令
        batch_size: 每批文档数
        max_retries: 每批最大重试次数
        retry_delay: 基础重试延迟 (秒, 指数退避)
        timeout: 单批超时 (秒)
        fallback_threshold: 成功率低于此值时回退

    Returns:
        重排序后的 Top-N 结果列表
    """
    if not results:
        return []

    # Step 1: 提取 episode 文本
    doc_texts = []
    original_indices = []

    for idx, (doc, score) in enumerate(results):
        episode_text = doc.get("episode")
        if episode_text:
            doc_texts.append(episode_text)
            original_indices.append(idx)

    if not doc_texts:
        return []

    hybrid_reranker = rerank_service.get_rerank_service()
    reranker = hybrid_reranker.get_service()
    print(f"Reranking query: {query}")
    print(f"Reranking {len(doc_texts)} documents in batches of {batch_size}...")

    # Step 2: 分批处理
    batches = []
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i : i + batch_size]
        batches.append((i, batch))

    async def process_batch_with_retry(start_idx: int, batch_texts: List[str]):
        """处理单批 (重试 + 超时 + 指数退避)。"""
        for attempt in range(max_retries):
            try:
                batch_results = await asyncio.wait_for(
                    reranker.rerank_documents(
                        query, batch_texts, instruction=reranker_instruction
                    ),
                    timeout=timeout,
                )
                for item in batch_results["results"]:
                    item["global_index"] = start_idx + item["index"]
                if attempt > 0:
                    print(f"  Batch at {start_idx} succeeded on attempt {attempt + 1}")
                return batch_results["results"]

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"  Batch at {start_idx} timeout (attempt {attempt + 1}), retrying in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  Batch at {start_idx} timeout after {max_retries} attempts")
                    return []

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"  Batch at {start_idx} failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  Batch at {start_idx} failed after {max_retries} attempts: {e}")
                    return []

    # 控制并发 (保守策略)
    max_concurrent = getattr(config, 'reranker_concurrent_batches', 2)
    batch_results_list = []
    successful_batches = 0

    for group_start in range(0, len(batches), max_concurrent):
        group_batches = batches[group_start : group_start + max_concurrent]

        tasks = [
            process_batch_with_retry(start_idx, batch)
            for start_idx, batch in group_batches
        ]
        group_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in group_results:
            if isinstance(result, list) and result:
                batch_results_list.append(result)
                successful_batches += 1
            else:
                batch_results_list.append([])

        if group_start + max_concurrent < len(batches):
            await asyncio.sleep(0.3)

    # Step 3: 合并 + 回退策略
    all_rerank_results = []
    for batch_results in batch_results_list:
        all_rerank_results.extend(batch_results)

    success_rate = successful_batches / len(batches) if batches else 0.0
    print(f"Reranker success rate: {success_rate:.1%} ({successful_batches}/{len(batches)} batches)")

    if not all_rerank_results:
        print("Warning: All reranker batches failed, using original ranking as fallback")
        return results[:top_n]

    if success_rate < fallback_threshold:
        print(f"Warning: Reranker success rate too low ({success_rate:.1%}), using original ranking")
        return results[:top_n]

    # Step 4: 按 reranker 分数排序, 返回 Top-N
    sorted_results = sorted(
        all_rerank_results, key=lambda x: x["score"], reverse=True
    )[:top_n]

    final_results = [
        (results[original_indices[item["global_index"]]][0], item["score"])
        for item in sorted_results
    ]

    return final_results
