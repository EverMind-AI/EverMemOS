"""
检索公共工具函数模块

提供以下功能:
1. 相似度计算: cosine_similarity, cosine_similarity_batch, compute_maxsim_score
2. 文本处理: tokenize, ensure_nltk_data
3. 结果融合: reciprocal_rank_fusion, multi_rrf_fusion
4. 索引搜索: search_with_bm25_index, search_with_emb_index
5. 过滤工具: filter_docs_by_memcell_ids, filter_emb_index_by_memcell_ids
6. 场景工具: maxsim_aggregate_to_scenes, collect_memcells_from_scenes

这个模块提供统一的检索工具函数，被 scene_retrieval.py 和 stage3_memory_retrivel.py 调用。
"""

import asyncio
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from agentic_layer import vectorize_service


# ==============================================================================
# NLTK 初始化
# ==============================================================================

def ensure_nltk_data():
    """
    确保 NLTK 数据已下载
    
    检查并下载:
    - punkt: 分词器
    - punkt_tab: 分词器表格数据
    - stopwords: 停用词列表
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading punkt...")
        nltk.download("punkt", quiet=True)
    
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading punkt_tab...")
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("Downloading stopwords...")
        nltk.download("stopwords", quiet=True)
    
    # 验证 stopwords 可用性
    try:
        test_stopwords = stopwords.words("english")
        if not test_stopwords:
            raise ValueError("Stopwords is empty")
    except Exception as e:
        print(f"Warning: NLTK stopwords error: {e}")
        print("Re-downloading stopwords...")
        nltk.download("stopwords", quiet=False, force=True)


# ==============================================================================
# 相似度计算
# ==============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
    
    Returns:
        余弦相似度 (float, 范围 [-1, 1])
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def cosine_similarity_batch(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    批量计算余弦相似度 (query vs 多个文档)
    
    Args:
        query_vec: 查询向量 (1D numpy array)
        doc_vecs: 文档向量矩阵 (2D numpy array, 每行一个文档)
    
    Returns:
        相似度数组 (1D numpy array)
    """
    if doc_vecs.size == 0:
        return np.array([])
    
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(len(doc_vecs))
    
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    # 避免除零
    doc_norms[doc_norms == 0] = 1e-9
    
    dot_products = np.dot(doc_vecs, query_vec)
    return dot_products / (query_norm * doc_norms)


def compute_maxsim_score(query_emb: np.ndarray, fact_embs: List[np.ndarray]) -> float:
    """
    MaxSim 策略: 计算 query 与多个 atomic facts 的最大相似度
    
    核心思想:
    - 找到与 query 最相关的单个 fact
    - 如果任意一个 fact 与 query 高度相关，则认为整个 event_log 相关
    - 避免被无关 facts 稀释分数
    - 适合记忆检索场景（用户通常只关注某个方面）
    
    优化: 使用向量化矩阵运算，相比循环提升 2-3x 速度
    
    Args:
        query_emb: 查询向量 (1D numpy array)
        fact_embs: atomic_fact 向量列表
    
    Returns:
        最大相似度 (float, 范围 [-1, 1], 通常 [0, 1])
    """
    if not fact_embs:
        return 0.0
    
    query_norm = np.linalg.norm(query_emb)
    if query_norm == 0:
        return 0.0
    
    try:
        # 优化: 使用矩阵运算代替循环 (2-3x 加速)
        # 转换为矩阵: shape = (n_facts, embedding_dim)
        fact_matrix = np.array(fact_embs)
        
        # 批量计算所有 fact 的范数
        fact_norms = np.linalg.norm(fact_matrix, axis=1)
        
        # 过滤零向量
        valid_mask = fact_norms > 0
        if not np.any(valid_mask):
            return 0.0
        
        # 向量化计算所有相似度
        dot_products = np.dot(fact_matrix[valid_mask], query_emb)
        sims = dot_products / (query_norm * fact_norms[valid_mask])
        
        # 返回最大相似度
        return float(np.max(sims))
    
    except Exception:
        # 兼容性回退: 使用循环方法
        similarities = []
        for fact_emb in fact_embs:
            fact_norm = np.linalg.norm(fact_emb)
            if fact_norm == 0:
                continue
            sim = np.dot(query_emb, fact_emb) / (query_norm * fact_norm)
            similarities.append(sim)
        return max(similarities) if similarities else 0.0


# ==============================================================================
# 文本处理
# ==============================================================================

def tokenize(text: str, stemmer=None, stop_words: Set[str] = None) -> List[str]:
    """
    NLTK 分词 + 词干提取 + 停用词过滤
    
    必须与索引构建时使用的分词方式完全一致
    
    Args:
        text: 输入文本
        stemmer: NLTK 词干提取器 (默认 PorterStemmer)
        stop_words: 停用词集合 (默认英文停用词)
    
    Returns:
        处理后的词列表
    """
    if not text:
        return []
    
    if stemmer is None:
        stemmer = PorterStemmer()
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
    
    tokens = word_tokenize(text.lower())
    
    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token.isalpha() and len(token) >= 2 and token not in stop_words
    ]
    
    return processed_tokens


# ==============================================================================
# 结果融合 (RRF)
# ==============================================================================

def reciprocal_rank_fusion(
    emb_results: List[Tuple[dict, float]],
    bm25_results: List[Tuple[dict, float]],
    k: int = 60,
    id_field: str = "event_id"
) -> List[Tuple[dict, float]]:
    """
    RRF (Reciprocal Rank Fusion) 融合 Embedding 和 BM25 检索结果
    
    RRF 是一种无需归一化的融合策略，对排名位置敏感。
    公式: RRF_score(doc) = Σ(1 / (k + rank_i))
    
    优点:
    1. 无需归一化分数 (Embedding 和 BM25 分数范围不同)
    2. 简单有效，已在工业界广泛验证 (Elasticsearch 等)
    3. 对头部结果更敏感 (高排名贡献更大)
    4. 无需调参 (k=60 是经验最优值)
    
    Args:
        emb_results: Embedding 检索结果 [(doc, score), ...]
        bm25_results: BM25 检索结果 [(doc, score), ...]
        k: RRF 常数，通常取 60 (经验值)
        id_field: 文档 ID 字段名 (用于去重)
    
    Returns:
        融合结果 [(doc, rrf_score), ...]，按 RRF 分数降序排列
    
    Example:
        emb_results = [(doc1, 0.92), (doc2, 0.87), (doc3, 0.81)]
        bm25_results = [(doc2, 15.3), (doc1, 12.7), (doc4, 10.2)]
        
        Doc1: 1/(60+1) + 1/(60+2) = 0.0323
        Doc2: 1/(60+2) + 1/(60+1) = 0.0323  
        Doc3: 1/(60+3) + 0        = 0.0159
        Doc4: 0        + 1/(60+3) = 0.0159
        
        结果: [(doc1, 0.0323), (doc2, 0.0323), (doc3, 0.0159), (doc4, 0.0159)]
    """
    doc_rrf_scores: Dict[Any, float] = {}
    doc_map: Dict[Any, dict] = {}
    
    # 处理 Embedding 检索结果
    for rank, (doc, score) in enumerate(emb_results, start=1):
        doc_id = doc.get(id_field) or doc.get("turn_id") or id(doc)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # 处理 BM25 检索结果
    for rank, (doc, score) in enumerate(bm25_results, start=1):
        doc_id = doc.get(id_field) or doc.get("turn_id") or id(doc)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # 按 RRF 分数排序
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 转换回 (doc, score) 格式
    return [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]


def multi_rrf_fusion(
    results_list: List[List[Tuple[dict, float]]],
    k: int = 60,
    id_field: str = "event_id"
) -> List[Tuple[dict, float]]:
    """
    多路 RRF 融合 (支持多个查询的结果)
    
    与双路 RRF 类似，但支持融合任意数量的检索结果。
    每个结果集贡献分数: 1 / (k + rank)
    
    原理:
    - 在多个查询中排名靠前的文档会累积高分，最终排名靠前
    - 相当于"投票机制": 被多个查询认为相关的文档更可能真正相关
    
    Args:
        results_list: 多个检索结果列表 [
            [(doc1, score), (doc2, score), ...],  # 查询 1 结果
            [(doc3, score), (doc1, score), ...],  # 查询 2 结果
            [(doc4, score), (doc2, score), ...],  # 查询 3 结果
        ]
        k: RRF 常数 (默认 60)
        id_field: 文档 ID 字段名
    
    Returns:
        融合结果 [(doc, rrf_score), ...]，按 RRF 分数降序排列
    
    Example:
        查询 1 结果: [(doc_A, 0.9), (doc_B, 0.8), (doc_C, 0.7)]
        查询 2 结果: [(doc_B, 0.88), (doc_D, 0.82), (doc_A, 0.75)]
        查询 3 结果: [(doc_A, 0.92), (doc_E, 0.85), (doc_B, 0.80)]
        
        RRF 分数:
        doc_A: 1/(60+1) + 1/(60+3) + 1/(60+1) = 高分  <- 出现在 Q1,Q2,Q3
        doc_B: 1/(60+2) + 1/(60+1) + 1/(60+3) = 高分  <- 出现在 Q1,Q2,Q3
        doc_C: 1/(60+3) + 0        + 0        = 低分  <- 仅在 Q1
        
        最终: doc_A 和 doc_B 排名最高 (被多个查询认可)
    """
    if not results_list:
        return []
    
    # 如果只有一个结果集，直接返回
    if len(results_list) == 1:
        return results_list[0]
    
    doc_rrf_scores: Dict[Any, float] = {}
    doc_map: Dict[Any, dict] = {}
    
    # 遍历每个查询的检索结果
    for query_results in results_list:
        for rank, (doc, score) in enumerate(query_results, start=1):
            doc_id = doc.get(id_field) or doc.get("turn_id") or id(doc)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            # 累加 RRF 分数
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # 按 RRF 分数排序
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 转换回 (doc, score) 格式
    return [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]


# ==============================================================================
# 索引搜索
# ==============================================================================

def search_with_bm25_index(
    query: str,
    bm25,
    docs: List[dict],
    top_n: int = 50,
    fact_to_doc_idx: List[int] = None
) -> List[Tuple[dict, float]]:
    """
    BM25 搜索 (支持 MaxSim 聚合)
    
    MaxSim 策略:
    1. 计算每个 atomic_fact 的 BM25 分数
    2. 对于每个文档，取其所有 facts 中的最大分数
    3. 返回按 MaxSim 分数排序的文档
    
    这确保了拥有一个高度相关 fact 的文档排名高于拥有多个中等相关 facts 的文档。
    
    Args:
        query: 搜索查询
        bm25: BM25 索引 (在 fact 级别语料上构建)
        docs: 原始文档列表
        top_n: 返回结果数量
        fact_to_doc_idx: fact 索引到文档索引的映射 (用于 MaxSim 聚合)
    
    Returns:
        [(doc, score), ...] 按 MaxSim 分数降序排列
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    tokenized_query = tokenize(query, stemmer, stop_words)

    if not tokenized_query:
        print("Warning: Query is empty after tokenization.")
        return []

    # 获取所有 facts 的 BM25 分数
    fact_scores = bm25.get_scores(tokenized_query)
    
    # 检查是否有 MaxSim 索引 (带 fact_to_doc_idx)
    if fact_to_doc_idx is not None:
        # MaxSim 聚合: 每个文档取其 facts 中的最大分数
        doc_max_scores: Dict[int, float] = {}
        
        for fact_idx, score in enumerate(fact_scores):
            doc_idx = fact_to_doc_idx[fact_idx]
            if doc_idx not in doc_max_scores:
                doc_max_scores[doc_idx] = score
            else:
                doc_max_scores[doc_idx] = max(doc_max_scores[doc_idx], score)
        
        # 构建结果列表
        results_with_scores = [
            (docs[doc_idx], max_score) 
            for doc_idx, max_score in doc_max_scores.items()
        ]
    else:
        # 传统模式: 直接使用文档级别分数 (向后兼容)
        results_with_scores = list(zip(docs, fact_scores))
    
    # 按分数降序排列并返回 top_n
    sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


async def search_with_emb_index(
    query: str,
    emb_index: List[Dict[str, Any]],
    top_n: int = 50,
    query_embedding: Optional[np.ndarray] = None
) -> List[Tuple[dict, float]]:
    """
    Embedding 搜索 (使用 MaxSim 策略)
    
    对于包含 atomic_facts 的文档:
    - 计算 query 与每个 atomic_fact 的相似度
    - 取最大相似度作为文档分数 (MaxSim 策略)
    
    对于传统文档:
    - 回退到使用 subject/summary/episode 字段
    - 取这些字段中的最大相似度
    
    优化: 支持预计算的 query embedding 以避免重复 API 调用
    
    Args:
        query: 查询文本
        emb_index: 预构建的 embedding 索引
        top_n: 返回结果数量
        query_embedding: 可选的预计算 query embedding (避免重复计算)
    
    Returns:
        [(doc, score), ...] 排序列表
    """
    # 获取 query embedding (如果未提供则调用 API)
    if query_embedding is not None:
        query_vec = query_embedding
    else:
        query_vec = np.array(await vectorize_service.get_vectorize_service().get_embedding(query))
    
    query_norm = np.linalg.norm(query_vec)
    
    # 如果 query 向量为零，返回空结果
    if query_norm == 0:
        return []
    
    # 存储每个文档的 MaxSim 分数
    doc_scores = []
    
    for item in emb_index:
        doc = item.get("doc")
        embeddings = item.get("embeddings", {})
        
        if not embeddings:
            continue
        
        # 优先使用 atomic_facts (MaxSim 策略)
        if "atomic_facts" in embeddings:
            atomic_fact_embs = embeddings["atomic_facts"]
            if atomic_fact_embs:
                # 使用 MaxSim 计算分数
                score = compute_maxsim_score(query_vec, atomic_fact_embs)
                doc_scores.append((doc, score))
                continue
        
        # 回退到传统字段 (保持向后兼容)
        # 对于传统字段，也使用 MaxSim 策略 (取最大值)
        field_scores = []
        for field in ["episode", "subject", "summary"]:
            if field in embeddings:
                field_emb = embeddings[field]
                field_norm = np.linalg.norm(field_emb)
                
                if field_norm > 0:
                    sim = np.dot(query_vec, field_emb) / (query_norm * field_norm)
                    field_scores.append(sim)
        
        if field_scores:
            score = max(field_scores)
            doc_scores.append((doc, score))
    
    if not doc_scores:
        return []
    
    # 按分数降序排列并返回 Top-N
    sorted_results = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


# ==============================================================================
# 过滤工具
# ==============================================================================

def filter_docs_by_memcell_ids(
    docs: List[Dict[str, Any]],
    memcell_ids: Set[str]
) -> List[Dict[str, Any]]:
    """
    按 memcell ID 过滤文档
    
    Args:
        docs: 文档列表
        memcell_ids: 要保留的 memcell event_id 集合
    
    Returns:
        过滤后的文档列表
    """
    return [doc for doc in docs if doc.get("event_id") in memcell_ids]


def filter_emb_index_by_memcell_ids(
    emb_index: List[Dict[str, Any]],
    memcell_ids: Set[str]
) -> List[Dict[str, Any]]:
    """
    按 memcell ID 过滤 embedding 索引
    
    Args:
        emb_index: Embedding 索引 (列表格式: {"doc": ..., "embeddings": ...})
        memcell_ids: 要保留的 memcell event_id 集合
    
    Returns:
        过滤后的 embedding 索引
    """
    return [
        item for item in emb_index 
        if item.get("doc", {}).get("event_id") in memcell_ids
    ]


# ==============================================================================
# 场景工具
# ==============================================================================

def maxsim_aggregate_to_scenes(
    rrf_results: List[Tuple[dict, float]],
    memcell_to_scene: Dict[str, str],
    target_scene_count: int
) -> Tuple[Dict[str, float], int]:
    """
    MaxSim 聚合到场景
    
    从 RRF 结果中逐步选择 memcell，直到覆盖足够的场景。
    每个场景的分数 = 其包含的 memcell 中的最高分数 (MaxSim)
    
    Args:
        rrf_results: RRF 融合后的结果 [(doc, score), ...]
        memcell_to_scene: memcell event_id -> scene_id 映射
        target_scene_count: 目标场景数量
    
    Returns:
        (scene_scores, selected_memcell_count)
        - scene_scores: {scene_id: max_score}
        - selected_memcell_count: 已处理的 memcell 数量
    """
    scene_scores: Dict[str, float] = {}
    selected_count = 0
    
    for doc, score in rrf_results:
        event_id = doc.get("event_id", "")
        scene_id = memcell_to_scene.get(event_id)
        selected_count += 1
        
        if scene_id:
            # MaxSim: 每个场景取最高分
            if scene_id not in scene_scores or score > scene_scores[scene_id]:
                scene_scores[scene_id] = score
        
        # 一旦覆盖了足够场景，立即停止
        if len(scene_scores) >= target_scene_count:
            break
    
    return scene_scores, selected_count


def collect_memcells_from_scenes(
    top_scenes: List[Tuple[str, float]],
    scene_dict_map: Dict[str, Dict]
) -> Set[str]:
    """
    从选中的场景中收集所有 memcell IDs
    
    Args:
        top_scenes: 选中的场景列表 [(scene_id, score), ...]
        scene_dict_map: scene_id -> scene_dict 映射
    
    Returns:
        所有 memcell event_id 的集合
    """
    all_memcell_ids: Set[str] = set()
    for scene_id, _ in top_scenes:
        scene_dict = scene_dict_map.get(scene_id, {})
        memcell_ids = scene_dict.get("memcell_ids", [])
        all_memcell_ids.update(memcell_ids)
    return all_memcell_ids


# ==============================================================================
# 索引加载工具
# ==============================================================================

def load_bm25_index(index_dir: Path, conv_index: str) -> Tuple[Any, List[dict], Optional[List[int]]]:
    """
    加载 BM25 索引
    
    Args:
        index_dir: 索引目录
        conv_index: 会话索引 (数字字符串)
    
    Returns:
        (bm25, docs, fact_to_doc_idx)
    """
    bm25_file = index_dir / f"bm25_index_conv_{conv_index}.pkl"
    if not bm25_file.exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_file}")
    
    with open(bm25_file, "rb") as f:
        data = pickle.load(f)
    
    return data.get("bm25"), data.get("docs"), data.get("fact_to_doc_idx")


def load_emb_index(index_dir: Path, conv_index: str) -> Optional[List[Dict[str, Any]]]:
    """
    加载 Embedding 索引
    
    Args:
        index_dir: 索引目录
        conv_index: 会话索引 (数字字符串)
    
    Returns:
        Embedding 索引，如果不存在则返回 None
    """
    emb_file = index_dir / f"embedding_index_conv_{conv_index}.pkl"
    if not emb_file.exists():
        return None
    
    with open(emb_file, "rb") as f:
        return pickle.load(f)


def load_scene_index(index_dir: Path, conv_index: str) -> Dict[str, Any]:
    """
    加载场景索引
    
    Args:
        index_dir: 索引目录
        conv_index: 会话索引 (数字字符串)
    
    Returns:
        场景索引字典
    
    Raises:
        FileNotFoundError: 如果索引文件不存在
    """
    scene_file = index_dir / f"scene_index_conv_{conv_index}.pkl"
    if not scene_file.exists():
        raise FileNotFoundError(f"Scene index not found: {scene_file}")
    
    with open(scene_file, "rb") as f:
        return pickle.load(f)

