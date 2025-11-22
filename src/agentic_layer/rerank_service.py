"""
Rerank Service
重排序服务

This module provides methods to call DeepInfra or vLLM API for reranking retrieved memories.
该模块提供调用DeepInfra或vLLM API对检索到的记忆进行重排序的方法。
"""

from __future__ import annotations

import os
import asyncio
import aiohttp
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import numpy as np

from core.di import get_bean, service

logger = logging.getLogger(__name__)


class RerankProvider(str, Enum):
    """Rerank服务提供商枚举"""
    DEEPINFRA = "deepinfra"
    VLLM = "vllm"


@dataclass
@service(name="rerank_config", primary=True)
class RerankConfig:
    """Rerank API配置类"""
    
    provider: RerankProvider = RerankProvider.DEEPINFRA
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5

    def __post_init__(self):
        """初始化后从环境变量加载配置值"""
        # 处理 provider
        env_provider = os.getenv("RERANK_PROVIDER")
        if env_provider:
            provider_str = env_provider.lower()
            try:
                self.provider = RerankProvider(provider_str)
            except ValueError:
                logger.error(f"Invalid rerank provider '{provider_str}', expected one of {[p.value for p in RerankProvider]}")
                raise ValueError(f"Invalid rerank provider '{provider_str}', expected one of {[p.value for p in RerankProvider]}")

        if not self.api_key:
            self.api_key = os.getenv("RERANK_API_KEY", "")
            if self.provider == RerankProvider.VLLM and not self.api_key:
                self.api_key = "EMPTY"

        if not self.base_url:
            # vLLM/Local default might be http://localhost:12000/score
            default_url = "https://api.deepinfra.com/v1/inference" if self.provider == RerankProvider.DEEPINFRA else "http://localhost:12000/score"
            self.base_url = os.getenv("RERANK_BASE_URL", default_url)

        if not self.model:
            self.model = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
        
        if self.timeout == 30:
            self.timeout = int(os.getenv("RERANK_TIMEOUT", "30"))
        if self.max_retries == 3:
            self.max_retries = int(os.getenv("RERANK_MAX_RETRIES", "3"))
        if self.batch_size == 10:
            self.batch_size = int(os.getenv("RERANK_BATCH_SIZE", "10"))
        if self.max_concurrent_requests == 5:
            self.max_concurrent_requests = int(os.getenv("RERANK_MAX_CONCURRENT", "5"))


class RerankError(Exception):
    """Rerank API错误异常类"""
    pass


@dataclass
class RerankMemResponse:
    """重排序后的记忆检索响应"""
    memories: List[Dict[str, List[Any]]] = field(default_factory=list)
    scores: List[Dict[str, List[float]]] = field(default_factory=list)
    rerank_scores: List[Dict[str, List[float]]] = field(default_factory=list)
    importance_scores: List[float] = field(default_factory=list)
    original_data: List[Dict[str, List[Dict[str, Any]]]] = field(default_factory=list)
    total_count: int = 0
    has_more: bool = False
    query_metadata: Any = field(default_factory=dict)
    metadata: Any = field(default_factory=dict)


class RerankServiceInterface(ABC):
    """重排序服务接口"""

    @abstractmethod
    async def rerank_memories(
        self, query: str, retrieve_response: Any, instruction: str = None
    ) -> Union[RerankMemResponse, List[Dict[str, Any]]]:
        pass


@service(name="rerank_service", primary=True)
class RerankService(RerankServiceInterface):
    """
    重排序服务类 (支持 DeepInfra, vLLM)
    """

    def __init__(self, config: Optional[RerankConfig] = None):
        if config is None:
            try:
                config = get_bean("rerank_config")
            except Exception:
                config = self._load_config_from_env()

        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        logger.info(f"Initialized Rerank Service | provider={config.provider.value} | model={config.model}")

    def _load_config_from_env(self) -> RerankConfig:
        return RerankConfig()

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def _make_rerank_request(
        self, query: str, documents: List[str], instruction: str = None
    ) -> Dict[str, Any]:
        if not documents:
            return {"results": []}

        # 拆分成批次
        batch_size = self.config.batch_size
        if batch_size <= 0: batch_size = 10
        
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

        batch_tasks = []
        for i, batch in enumerate(batches):
            start_index = i * batch_size
            batch_tasks.append(self._send_rerank_request_batch(query, batch, start_index, instruction))

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        all_scores = []
        total_input_tokens = 0
        last_response = None

        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Rerank batch {i} failed: {result}")
                batch_len = len(batches[i])
                all_scores.extend([-100.0] * batch_len)
                continue

            scores = result.get("scores", [])
            all_scores.extend(scores)
            total_input_tokens += result.get("input_tokens", 0)
            last_response = result

        if not last_response and not all_scores:
             pass 

        combined_response = {
            "scores": all_scores,
            "input_tokens": total_input_tokens,
            "request_id": last_response.get("request_id") if last_response else None,
        }
        return self._convert_response_format(combined_response, len(documents))

    def _format_rerank_texts(self, query: str, documents: List[str], instruction: Optional[str] = None):
        """构建 Rerank 请求文本（Qwen-Reranker 通用格式）"""
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        instruction = instruction or "Given a question and a passage, determine if the passage contains information relevant to answering the question."
        
        formatted_query = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        formatted_docs = [f"<Document>: {doc}{suffix}" for doc in documents]
        
        return [formatted_query] * len(documents), formatted_docs

    async def _send_rerank_request_batch(
        self, query: str, documents: List[str], start_index: int, instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """根据 provider 发送请求"""
        await self._ensure_session()

        # 统一格式化文本（Qwen-Reranker 模型格式）
        queries, formatted_docs = self._format_rerank_texts(query, documents, instruction)
        
        url = self.config.base_url
        
        if self.config.provider == RerankProvider.DEEPINFRA:
            if not url.endswith(self.config.model):
                url = f"{url}/{self.config.model}"
            request_data = {
                "queries": queries,
                "documents": formatted_docs,
            }
        else:
            # vLLM
            request_data = {
                "model": self.config.model,
                "text_1": queries,
                "text_2": formatted_docs,
            }

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    async with self.session.post(url, json=request_data) as response:
                        if response.status == 200:
                            json_body = await response.json()
                            return self._parse_provider_response(json_body)
                        else:
                            error_text = await response.text()
                            logger.error(f"Rerank API error ({self.config.provider.value}) {response.status}: {error_text}")
                            if attempt < self.config.max_retries - 1:
                                await asyncio.sleep(2**attempt)
                                continue
                            raise RerankError(f"API failed: {response.status} - {error_text}")
                except Exception as e:
                    logger.error(f"Rerank Exception ({self.config.provider.value}): {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise RerankError(f"Exception: {e}")

    def _parse_provider_response(self, json_body: Dict[str, Any]) -> Dict[str, Any]:
        """解析不同 provider 的响应为统一格式: {scores: [], input_tokens: int, ...}"""
        scores = []
        if self.config.provider == RerankProvider.DEEPINFRA:
            if "results" in json_body:
                results = json_body["results"]
                results.sort(key=lambda x: x.get("index", 0))
                scores = [item.get("relevance_score", 0.0) for item in results]
            elif "scores" in json_body:
                scores = json_body["scores"]
        else:
            if "data" in json_body:
                scores = [item.get("score", 0.0) for item in json_body["data"]]
            elif "scores" in json_body:
                scores = json_body["scores"]
        
        return {
            "scores": scores,
            "input_tokens": json_body.get("usage", {}).get("prompt_tokens", 0) or json_body.get("input_tokens", 0),
            "request_id": json_body.get("id") or json_body.get("request_id"),
        }

    def _convert_response_format(
        self, combined_response: Dict[str, Any], num_documents: int
    ) -> Dict[str, Any]:
        scores = combined_response.get("scores", [])
        if len(scores) < num_documents:
            scores.extend([0.0] * (num_documents - len(scores)))
        scores = scores[:num_documents]

        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (original_index, score) in enumerate(indexed_scores):
            results.append(
                {"index": original_index, "relevance_score": score, "rank": rank}
            )

        return {
            "results": results,
            "input_tokens": combined_response.get("input_tokens", 0),
            "request_id": combined_response.get("request_id"),
        }
    
    def _extract_memory_text(self, memory: Any) -> str:
        if hasattr(memory, 'episode') and memory.episode:
            return memory.episode
        elif hasattr(memory, 'summary') and memory.summary:
            return memory.summary
        elif hasattr(memory, 'subject') and memory.subject:
            return memory.subject
        return str(memory)
    
    def _extract_text_from_hit(self, hit: Dict[str, Any]) -> str:
        source = hit.get('_source', hit)
        if source.get('episode'): return source['episode']
        if source.get('summary'): return source['summary']
        if source.get('subject'): return source['subject']
        return str(hit)
        
    async def rerank_memories(
        self, query: str, retrieve_response: Any, instruction: str = None
    ) -> Union[RerankMemResponse, List[Dict[str, Any]]]:
        
        # 1. Handle List of hits (raw dicts)
        if isinstance(retrieve_response, list):
            all_hits = retrieve_response
            if not all_hits: return []
            all_texts = [self._extract_text_from_hit(hit) for hit in all_hits]
            
            try:
                rerank_result = await self._make_rerank_request(query, all_texts, instruction)
                
                results_meta = rerank_result.get("results", [])
                
                reranked_hits = []
                for item in results_meta:
                    idx = item["index"]
                    score = item["relevance_score"]
                    if 0 <= idx < len(all_hits):
                        hit = all_hits[idx].copy()
                        hit['_rerank_score'] = score
                        reranked_hits.append(hit)
                
                return reranked_hits

            except Exception as e:
                logger.error(f"Rerank list failed: {e}")
                return all_hits

        # 2. Handle RetrieveMemResponse object
        if not hasattr(retrieve_response, 'memories') or not retrieve_response.memories:
             return RerankMemResponse(memories=[], scores=[])

        all_memories_meta = [] 
        all_texts = []
        
        for group_idx, memory_dict_by_group in enumerate(retrieve_response.memories):
            for group_id, memory_list in memory_dict_by_group.items():
                for mem_idx, memory in enumerate(memory_list):
                    all_memories_meta.append((group_idx, group_id, mem_idx, memory))
                    all_texts.append(self._extract_memory_text(memory))
        
        if not all_texts:
             return RerankMemResponse(
                 memories=retrieve_response.memories,
                 scores=retrieve_response.scores,
                 total_count=getattr(retrieve_response, 'total_count', 0)
             )

        try:
            rerank_result = await self._make_rerank_request(query, all_texts, instruction)
            results_meta = rerank_result.get("results", []) 

            group_data_map = {} 
            
            for item in results_meta:
                original_idx = item["index"]
                score = item["relevance_score"]
                
                group_idx, group_id, mem_idx, memory = all_memories_meta[original_idx]
                
                if group_id not in group_data_map:
                    group_data_map[group_id] = {
                        "memories": [], "scores": [], "rerank_scores": [], "original_data": []
                    }
                
                orig_score = 0.0
                if group_idx < len(retrieve_response.scores):
                     g_scores = retrieve_response.scores[group_idx].get(group_id, [])
                     if mem_idx < len(g_scores): orig_score = g_scores[mem_idx]
                
                orig_data = {}
                if hasattr(retrieve_response, 'original_data') and group_idx < len(retrieve_response.original_data):
                    g_data = retrieve_response.original_data[group_idx].get(group_id, [])
                    if mem_idx < len(g_data): orig_data = g_data[mem_idx]

                group_data_map[group_id]["memories"].append(memory)
                group_data_map[group_id]["scores"].append(orig_score)
                group_data_map[group_id]["rerank_scores"].append(score)
                group_data_map[group_id]["original_data"].append(orig_data)

            final_memories = [ {gid: d["memories"] for gid, d in group_data_map.items()} ]
            final_scores = [ {gid: d["scores"] for gid, d in group_data_map.items()} ]
            final_rerank_scores = [ {gid: d["rerank_scores"] for gid, d in group_data_map.items()} ]
            final_original_data = [ {gid: d["original_data"] for gid, d in group_data_map.items()} ]
            
            return RerankMemResponse(
                memories=final_memories,
                scores=final_scores,
                rerank_scores=final_rerank_scores,
                original_data=final_original_data,
                total_count=len(all_texts),
                has_more=getattr(retrieve_response, 'has_more', False),
                metadata=getattr(retrieve_response, 'metadata', {})
            )

        except Exception as e:
            logger.error(f"Rerank object failed: {e}")
            return RerankMemResponse(
                memories=retrieve_response.memories,
                scores=retrieve_response.scores,
                total_count=getattr(retrieve_response, 'total_count', 0)
            )

def get_rerank_service() -> RerankServiceInterface:
    return get_bean("rerank_service")
