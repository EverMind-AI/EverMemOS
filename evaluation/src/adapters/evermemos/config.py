import os
from dotenv import load_dotenv

load_dotenv()


class ExperimentConfig:
    # ========== Basic Configuration ==========
    experiment_name: str = "locomo_evaluation"
    datase_path: str = "data/locomo10.json"
    num_conv: int = 10

    # ========== MemCell Extraction ==========
    enable_foresight_extraction: bool = False
    enable_clustering: bool = True
    enable_profile_extraction: bool = False

    # Clustering
    cluster_similarity_threshold: float = 0.70
    cluster_max_time_gap_days: float = 7.0

    # Profile
    profile_scenario: str = "assistant"
    profile_min_confidence: float = 0.6
    profile_min_memcells: int = 1
    profile_extraction_mode: str = "conversation"  # "conversation" | "scene"
    profile_life_max_items: int = 25

    # ========== Retrieval Mode ==========
    # "agentic" (default): 两级场景检索 + Multi-Query (93% on LoCoMo)
    #   Level 1: RRF (Embedding + BM25) → MaxSim → Top-K Scenes
    #   Level 2: Rerank → Sufficiency Check → Multi-Query (全库搜索)
    # "lightweight": 纯 BM25 检索 (无 LLM 调用, 最快)
    retrieval_mode: str = "agentic"

    # ========== Agentic Retrieval ==========
    use_emb: bool = True            # 是否使用 Embedding (影响索引构建)
    use_reranker: bool = True       # 是否使用 Reranker
    use_multi_query: bool = True    # 不充分时是否启用 Multi-Query 扩展

    # 最终返回数量
    response_top_k: int = 10
    round2_response_top_k: int = 10

    # 充分性检查
    sufficiency_max_docs: int = 10

    # Multi-Query (Round 2) 参数
    multi_query_num: int = 3            # 生成的扩展查询数量
    hybrid_emb_candidates: int = 50     # 每个查询的 Embedding 候选数
    hybrid_bm25_candidates: int = 50    # 每个查询的 BM25 候选数
    hybrid_rrf_k: int = 40              # RRF 融合常数

    # ========== Scene Selection (Level 1) ==========
    enable_scene_retrieval: bool = True
    scene_top_k: int = 10               # 选择的场景数
    level1_emb_candidates: int = 50     # Level 1 Embedding 候选数
    level1_bm25_candidates: int = 50    # Level 1 BM25 候选数
    level1_rrf_k: int = 40              # Level 1 RRF 融合常数

    # ========== Lightweight Retrieval (BM25-only) ==========
    lightweight_bm25_top_n: int = 50
    lightweight_final_top_n: int = 20

    # ========== Reranker ==========
    reranker_batch_size: int = 32
    reranker_max_retries: int = 3
    reranker_retry_delay: float = 3.0
    reranker_timeout: float = 60.0
    reranker_fallback_threshold: float = 0.3
    reranker_concurrent_batches: int = 2

    reranker_instruction: str = (
        "Determine if the passage contains specific facts, entities (names, dates, locations), "
        "or details that directly answer the question."
    )

    # ========== LLM ==========
    llm_service: str = "openai"
    llm_config: dict = {
        "openai": {
            "llm_provider": "openai",
            "model": "openai/gpt-4.1-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("LLM_API_KEY"),
            "temperature": 0.3,
            "max_tokens": 16384,
        },
        "vllm": {
            "llm_provider": "openai",
            "model": "Qwen3-30B",
            "base_url": "http://0.0.0.0:8000/v1",
            "api_key": "123",
            "temperature": 0,
            "max_tokens": 16384,
        },
    }

    max_retries: int = 5
    max_concurrent_requests: int = 10

    # ========== Answer ==========
    use_profile_in_answer: bool = False
    use_episode_in_answer: bool = True
    use_profile_classifier: bool = False

