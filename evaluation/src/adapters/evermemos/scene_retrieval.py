"""
Scene Retrieval — 默认 Agentic 检索模式 (93% on LoCoMo)

两级检索架构:
- Level 1: 场景选择 (RRF: Embedding + BM25 → MaxSim 聚合 → Top-K Scenes)
- Level 2: 场景内 Rerank → 充分性检查 → Multi-Query 扩展 (不充分时)
- Round 2 在全部 memcells 中搜索 (不受场景限制), 确保补充遗漏信息

超图记忆架构:
- 节点: MemCells (episode, event_log, atomic_facts)
- 超边: Scenes (聚类结果, 连接语义相关的 MemCells)
"""

import asyncio
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from evaluation.src.adapters.evermemos.config import ExperimentConfig

# 底层搜索算法 (retrieval_utils)
from evaluation.src.adapters.evermemos.retrieval_utils import (
    filter_docs_by_memcell_ids,
    filter_emb_index_by_memcell_ids,
    search_with_emb_index,
    search_with_bm25_index,
    reciprocal_rank_fusion,
    multi_rrf_fusion,
)

# Reranker (stage3)
from evaluation.src.adapters.evermemos.stage3_memory_retrivel import reranker_search

# Agentic 工具 (充分性检查, Multi-Query 生成)
from evaluation.src.adapters.evermemos.tools import agentic_utils

from agentic_layer import vectorize_service


async def agentic_retrieval(
    query: str,
    scene_index: Dict[str, Any],
    emb_index: List[Dict[str, Any]],
    docs: List[Dict[str, Any]],
    config: ExperimentConfig,
    bm25=None,
    llm_provider=None,
    llm_config: dict = None,
    fact_to_doc_idx: List[int] = None,
    question_date: Optional[str] = None,
) -> Tuple[List[Tuple[Dict[str, Any], float]], Dict[str, Any]]:
    """
    Agentic 两级场景检索 (默认模式, 93% on LoCoMo)。

    流程:
    ┌─────────────────────────────────────────────────────────┐
    │ Level 1: 场景选择                                       │
    │   Embedding(全库) + BM25(全库) → RRF 融合               │
    │   → MaxSim 聚合到 Scene → Top-K Scenes                 │
    │   → 展开得到约 40+ memcells                             │
    ├─────────────────────────────────────────────────────────┤
    │ Level 2: 场景内精排                                      │
    │   场景内所有 memcells → Reranker → Top K                │
    │   → LLM 充分性检查                                      │
    │     ├─ 充分 (68%) → 直接返回                            │
    │     └─ 不充分 (32%) → Round 2                           │
    ├─────────────────────────────────────────────────────────┤
    │ Round 2: Multi-Query 全库补充 (仅不充分时)               │
    │   LLM 生成 3 个改写查询                                 │
    │   → 每个查询在 全部 memcells 中 Hybrid Search           │
    │   → Multi-RRF 融合 → 合并 Round 1 → Final Rerank       │
    └─────────────────────────────────────────────────────────┘

    Args:
        query: 用户查询
        scene_index: 场景索引 (scenes, memcell_to_scene)
        emb_index: Embedding 索引 (全部 memcells)
        docs: 文档列表 (全部 MemCells)
        config: 实验配置
        bm25: BM25 索引
        llm_provider: LLM provider (充分性检查 + Multi-Query)
        llm_config: LLM 配置
        fact_to_doc_idx: fact 索引 → 文档索引映射
        question_date: 问题时间 (用于时序推理)

    Returns:
        (results, metadata)
    """
    start_time = time.time()

    # 校验
    if not scene_index or not scene_index.get("memcell_to_scene"):
        raise ValueError("Scene index is empty or missing memcell_to_scene mapping.")

    memcell_to_scene = scene_index["memcell_to_scene"]
    scenes = scene_index.get("scenes", [])
    scene_dict_map = {s["scene_id"]: s for s in scenes}

    metadata = {
        "retrieval_mode": "agentic",
        "scene_top_k": config.scene_top_k,
        "response_top_k": config.response_top_k,
        "is_multi_round": False,
        "is_sufficient": None,
        "reasoning": None,
        "level1_scene_count": 0,
        "total_memcells_in_scenes": 0,
        "round1_count": 0,
        "round1_reranked_count": 0,
        "round2_count": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }

    print(f"\n{'='*60}")
    print(f"Agentic Retrieval: {query[:50]}...")
    print(f"{'='*60}")

    # ==================================================================
    # Level 1: 场景选择 (RRF: Embedding + BM25 → MaxSim → Top-K Scenes)
    # ==================================================================
    print(f"  [Level 1] Scene Selection (RRF + MaxSim)...")

    query_embedding = np.array(
        await vectorize_service.get_vectorize_service().get_embedding(query)
    )

    # Step 1: Embedding 搜索 (全库)
    emb_results = await search_with_emb_index(
        query=query,
        emb_index=emb_index,
        top_n=config.level1_emb_candidates,
        query_embedding=query_embedding,
    )
    print(f"    Embedding: {len(emb_results)} candidates")

    # Step 2: BM25 搜索 (全库)
    bm25_results = []
    if bm25 is not None:
        bm25_results = await asyncio.to_thread(
            search_with_bm25_index,
            query=query,
            bm25=bm25,
            docs=docs,
            top_n=config.level1_bm25_candidates,
            fact_to_doc_idx=fact_to_doc_idx,
        )
        print(f"    BM25: {len(bm25_results)} candidates")

    # Step 3: RRF 融合
    if emb_results and bm25_results:
        rrf_results = reciprocal_rank_fusion(emb_results, bm25_results, k=config.level1_rrf_k)
        print(f"    RRF fusion: {len(rrf_results)} unique memcells")
    elif emb_results:
        rrf_results = emb_results
    else:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    # Step 4: MaxSim 聚合到场景 (动态扩展, 直到覆盖 scene_top_k 个场景)
    scene_scores: Dict[str, float] = {}
    selected_memcell_count = 0

    for doc, score in rrf_results:
        event_id = doc.get("event_id", "")
        scene_id = memcell_to_scene.get(event_id)
        selected_memcell_count += 1

        if scene_id:
            if scene_id not in scene_scores or score > scene_scores[scene_id]:
                scene_scores[scene_id] = score

        if len(scene_scores) >= config.scene_top_k:
            break

    metadata["level1_rrf_count"] = selected_memcell_count
    print(f"    Scanned {selected_memcell_count} memcells, covered {len(scene_scores)} scenes")

    if not scene_scores:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    # 选出 Top-K 场景
    sorted_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
    top_k_scenes = sorted_scenes[:config.scene_top_k]
    metadata["level1_scene_count"] = len(top_k_scenes)

    # 展开场景内的所有 memcell IDs
    all_memcell_ids = set()
    for scene_id, _ in top_k_scenes:
        scene_dict = scene_dict_map.get(scene_id, {})
        all_memcell_ids.update(scene_dict.get("memcell_ids", []))

    metadata["total_memcells_in_scenes"] = len(all_memcell_ids)
    print(f"    Selected {len(top_k_scenes)} scenes, {len(all_memcell_ids)} memcells")

    # ==================================================================
    # Level 2: 场景内 Rerank → 充分性检查
    # ==================================================================
    print(f"  [Level 2] Scene Rerank...")

    filtered_docs = filter_docs_by_memcell_ids(docs, all_memcell_ids)
    scene_memcells_for_rerank = [(doc, 1.0) for doc in filtered_docs]

    if not scene_memcells_for_rerank:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    # Rerank 场景内所有 memcell → Top K
    round1_top_k = config.response_top_k
    print(f"    [Rerank] {len(scene_memcells_for_rerank)} memcells -> Top {round1_top_k}...")

    if config.use_reranker:
        reranked_results = await reranker_search(
            query=query,
            results=scene_memcells_for_rerank,
            top_n=round1_top_k,
            reranker_instruction=config.reranker_instruction,
            batch_size=config.reranker_batch_size,
            max_retries=config.reranker_max_retries,
            retry_delay=config.reranker_retry_delay,
            timeout=config.reranker_timeout,
            fallback_threshold=config.reranker_fallback_threshold,
            config=config,
        )
    else:
        # 无 Reranker 回退: 场景内 Embedding 排序
        filtered_emb_index = filter_emb_index_by_memcell_ids(emb_index, all_memcell_ids)
        emb_fallback = await search_with_emb_index(
            query=query,
            emb_index=filtered_emb_index,
            top_n=50,
            query_embedding=query_embedding,
        )
        reranked_results = emb_fallback[:round1_top_k]

    round1_results = reranked_results
    metadata["round1_count"] = len(round1_results)
    metadata["round1_reranked_count"] = len(reranked_results)

    if not reranked_results:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    # LLM 充分性检查
    print(f"    [LLM] Sufficiency Check...")
    is_sufficient, reasoning, missing_info, key_info = await agentic_utils.check_sufficiency(
        query=query,
        results=reranked_results,
        llm_provider=llm_provider,
        llm_config=llm_config,
        max_docs=config.sufficiency_max_docs,
        question_date=question_date,
    )

    metadata["is_sufficient"] = is_sufficient
    metadata["reasoning"] = reasoning
    metadata["key_information_found"] = key_info

    print(f"      {'Sufficient' if is_sufficient else 'Insufficient'}")

    if is_sufficient:
        final_results = reranked_results[:config.response_top_k]
        metadata["final_count"] = len(final_results)
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        print(f"  [Complete] {len(final_results)} results | {metadata['total_latency_ms']:.0f}ms")
        return final_results, metadata

    # ==================================================================
    # Round 2: Multi-Query 全库补充 (仅不充分时触发)
    # ==================================================================
    if not config.use_multi_query:
        final_results = reranked_results[:config.response_top_k]
        metadata["final_count"] = len(final_results)
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return final_results, metadata

    metadata["is_multi_round"] = True
    metadata["missing_info"] = missing_info
    print(f"    [Round 2] Multi-Query Expansion...")

    # 生成改写查询
    refined_queries, query_strategy = await agentic_utils.generate_multi_queries(
        original_query=query,
        results=reranked_results,
        missing_info=missing_info,
        llm_provider=llm_provider,
        llm_config=llm_config,
        key_info=key_info,
        max_docs=config.response_top_k,
        num_queries=config.multi_query_num,
        question_date=question_date,
    )

    metadata["refined_queries"] = refined_queries
    metadata["query_strategy"] = query_strategy
    print(f"      Generated {len(refined_queries)} queries")

    # 每个查询在 全部 memcells 中 Hybrid Search (不受场景限制)
    print(f"      Searching ALL memcells (not limited to scenes)...")

    multi_query_results = []
    for q in refined_queries:
        q_emb = await search_with_emb_index(
            query=q,
            emb_index=emb_index,
            top_n=config.hybrid_emb_candidates,
        )

        q_bm25 = []
        if bm25 is not None:
            q_bm25 = await asyncio.to_thread(
                search_with_bm25_index, q, bm25, docs,
                config.hybrid_bm25_candidates, fact_to_doc_idx,
            )

        if q_emb and q_bm25:
            q_results = reciprocal_rank_fusion(q_emb, q_bm25, k=config.hybrid_rrf_k)
        elif q_emb:
            q_results = q_emb
        else:
            q_results = q_bm25

        multi_query_results.append(q_results[:50])

    # Multi-RRF 融合多个查询结果
    round2_results = multi_rrf_fusion(multi_query_results, k=config.hybrid_rrf_k)[:40]
    metadata["round2_count"] = len(round2_results)
    print(f"      Round 2: {len(round2_results)} candidates (from all memcells)")

    # 合并 Round 1 + Round 2 (去重, Round 1 优先)
    round1_ids = {doc.get("event_id") for doc, _ in round1_results}
    round2_unique = [(d, s) for d, s in round2_results if d.get("event_id") not in round1_ids]

    combined = list(round1_results)
    combined.extend(round2_unique[:40 - len(combined)])
    print(f"      Combined: {len(combined)} candidates")

    # Final Rerank
    round2_top_k = config.round2_response_top_k
    if config.use_reranker and combined:
        print(f"    [Final Rerank] {len(combined)} -> Top {round2_top_k}...")
        final_results = await reranker_search(
            query=query,
            results=combined,
            top_n=round2_top_k,
            reranker_instruction=config.reranker_instruction,
            batch_size=config.reranker_batch_size,
            max_retries=config.reranker_max_retries,
            retry_delay=config.reranker_retry_delay,
            timeout=config.reranker_timeout,
            fallback_threshold=config.reranker_fallback_threshold,
            config=config,
        )
    else:
        final_results = combined[:round2_top_k]

    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000

    print(f"  [Complete] {len(final_results)} results | {metadata['total_latency_ms']:.0f}ms")
    return final_results, metadata
