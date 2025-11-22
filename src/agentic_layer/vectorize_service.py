"""
Vectorize Service
向量化服务

This module provides methods to call DeepInfra or vLLM API for getting text embeddings.
该模块提供调用DeepInfra或vLLM API获取文本embedding向量的方法。
"""

from __future__ import annotations

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from openai import AsyncOpenAI, BadRequestError

from core.di.utils import get_bean
from core.di.decorators import service

logger = logging.getLogger(__name__)


class VectorizeProvider(str, Enum):
    """向量化服务提供商枚举"""
    DEEPINFRA = "deepinfra"
    VLLM = "vllm"


@dataclass
@service(name="vectorize_config", primary=True)
class VectorizeConfig:
    """Vectorize API配置类"""

    provider: VectorizeProvider = VectorizeProvider.DEEPINFRA
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5
    encoding_format: str = "float"
    dimensions: int = 1024

    def __post_init__(self):
        """初始化后从环境变量加载配置值"""
        # 处理 provider
        env_provider = os.getenv("VECTORIZE_PROVIDER")
        if env_provider:
            provider_str = env_provider.lower()
            try:
                self.provider = VectorizeProvider(provider_str)
            except ValueError:
                logger.error(f"Invalid provider '{provider_str}', expected one of {[p.value for p in VectorizeProvider]}")
                raise ValueError(f"Invalid provider '{provider_str}', expected one of {[p.value for p in VectorizeProvider]}")

        if not self.api_key:
            self.api_key = os.getenv("VECTORIZE_API_KEY") or os.getenv("DEEPINFRA_API_KEY", "")
            if self.provider == VectorizeProvider.VLLM and not self.api_key:
                 self.api_key = "EMPTY"

        if not self.base_url:
            default_url = "https://api.deepinfra.com/v1/openai"
            if self.provider == VectorizeProvider.VLLM:
                 default_url = "http://localhost:8000/v1" # Standard vLLM port
            
            self.base_url = os.getenv("VECTORIZE_BASE_URL", default_url)

        if not self.model:
            self.model = os.getenv("VECTORIZE_MODEL", "Qwen/Qwen3-Embedding-4B")

        if self.timeout == 30:
            self.timeout = int(os.getenv("VECTORIZE_TIMEOUT", "30"))
        if self.max_retries == 3:
            self.max_retries = int(os.getenv("VECTORIZE_MAX_RETRIES", "3"))
        if self.batch_size == 10:
            self.batch_size = int(os.getenv("VECTORIZE_BATCH_SIZE") or os.getenv("DEEPINFRA_BATCH_SIZE", "10"))
        if self.max_concurrent_requests == 5:
            self.max_concurrent_requests = int(
                os.getenv("VECTORIZE_MAX_CONCURRENT", "5")
            )
        if self.encoding_format == "float":
            self.encoding_format = os.getenv("VECTORIZE_ENCODING_FORMAT", "float")
        if self.dimensions == 1024:
            self.dimensions = int(os.getenv("VECTORIZE_DIMENSIONS", "1024"))


class VectorizeError(Exception):
    """Vectorize API错误异常类"""
    pass


@dataclass
class UsageInfo:
    """Token使用情况信息"""

    prompt_tokens: int
    total_tokens: int

    @classmethod
    def from_openai_usage(cls, usage) -> "UsageInfo":
        """从OpenAI usage对象创建UsageInfo对象"""
        return cls(prompt_tokens=usage.prompt_tokens, total_tokens=usage.total_tokens)


class VectorizeServiceInterface(ABC):
    """向量化服务接口"""

    @abstractmethod
    async def get_embedding(
        self, text: str, instruction: Optional[str] = None
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def get_embedding_with_usage(
        self, text: str, instruction: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[UsageInfo]]:
        pass

    @abstractmethod
    async def get_embeddings(
        self, texts: List[str], instruction: Optional[str] = None
    ) -> List[np.ndarray]:
        pass

    @abstractmethod
    async def get_embeddings_batch(
        self, text_batches: List[List[str]], instruction: Optional[str] = None
    ) -> List[List[np.ndarray]]:
        pass


@service(name="vectorize_service", primary=True)
class VectorizeService(VectorizeServiceInterface):
    """
    通用向量化服务类 (支持 DeepInfra, vLLM 等 OpenAI 兼容接口)
    """

    def __init__(self, config: Optional[VectorizeConfig] = None):
        if config is None:
            try:
                from core.di import get_bean
                config = get_bean("vectorize_config")
                logger.info("Vectorize config source: DI bean 'vectorize_config'")
            except Exception:
                config = self._load_config_from_env()
                logger.info("Vectorize config source: env")

        # 规范化配置
        base_url = config.base_url or ""
        if base_url and not (base_url.startswith("http://") or base_url.startswith("https://")):
             # 默认根据provider推断，或者 defaulting to https
            base_url = f"https://{base_url}"
            
        config.base_url = base_url
        
        self.config = config
        self.client: Optional[AsyncOpenAI] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        logger.info(
            f"Initialized Vectorize Service | provider={config.provider.value} | model={config.model} | base_url={config.base_url}"
        )

    def _load_config_from_env(self) -> VectorizeConfig:
        """从环境变量加载配置"""
        return VectorizeConfig()

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_client(self):
        if self.client is None:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )

    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None

    async def _make_request(self, texts: List[str], instruction: Optional[str] = None, is_query: bool = False):
        await self._ensure_client()
        if not self.config.model:
            raise VectorizeError("Embedding model is not configured.")

        # 如果 is_query=True，使用 instruction 格式包装文本
        if is_query:
            default_instruction = "Given a search query, retrieve relevant passages that answer the query"
            final_instruction = instruction if instruction is not None else default_instruction
            formatted_texts = [f"Instruct: {final_instruction}\nQuery: {text}" for text in texts]
        else:
            formatted_texts = texts

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    request_kwargs = {
                        "model": self.config.model,
                        "input": formatted_texts,
                        "encoding_format": self.config.encoding_format,
                    }
                    
                    # Only add dimensions parameter if provider is NOT vllm
                    # vLLM typically doesn't support 'dimensions' param in OpenAI API compatibility layer yet
                    # We handle vLLM via client-side truncation in _parse_embeddings_response
                    if self.config.dimensions and self.config.dimensions > 0:
                        if self.config.provider != VectorizeProvider.VLLM:
                            request_kwargs["dimensions"] = self.config.dimensions

                    response = await self.client.embeddings.create(**request_kwargs)
                    return response

                except Exception as e:
                    logger.error(f"Vectorize API error (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    else:
                        raise VectorizeError(f"API request failed: {e}")

    async def get_embedding(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> np.ndarray:
        response = await self._make_request([text], instruction, is_query)
        if not response.data:
            raise VectorizeError("Invalid API response: missing data")
        return np.array(self._parse_embeddings_response(response)[0], dtype=np.float32)

    async def get_embedding_with_usage(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> Tuple[np.ndarray, Optional[UsageInfo]]:
        response = await self._make_request([text], instruction, is_query)
        if not response.data:
            raise VectorizeError("Invalid API response: missing data")
        
        embeddings = self._parse_embeddings_response(response)
        embedding = np.array(embeddings[0], dtype=np.float32)
        usage_info = UsageInfo.from_openai_usage(response.usage) if response.usage else None
        return embedding, usage_info

    async def get_embeddings(
        self, texts: List[str], instruction: Optional[str] = None, is_query: bool = False
    ) -> List[np.ndarray]:
        if not texts:
            return []

        if len(texts) <= self.config.batch_size:
            response = await self._make_request(texts, instruction, is_query)
            return self._parse_embeddings_response(response)

        embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            response = await self._make_request(batch_texts, instruction, is_query)
            embeddings.extend(self._parse_embeddings_response(response))
            if i + self.config.batch_size < len(texts):
                await asyncio.sleep(0.1)
        return embeddings

    def _parse_embeddings_response(self, response) -> List[np.ndarray]:
        if not response.data:
            raise VectorizeError("Invalid API response: missing data")
        
        embeddings = []
        for item in response.data:
             emb = np.array(item.embedding, dtype=np.float32)
             # Client-side truncation if configured and necessary
             if self.config.dimensions and self.config.dimensions > 0 and len(emb) > self.config.dimensions:
                 emb = emb[:self.config.dimensions]
                 # Optional: Re-normalize if needed. Usually safer to re-normalize after truncation
                 # to maintain unit length for cosine similarity.
                 norm = np.linalg.norm(emb)
                 if norm > 0:
                     emb = emb / norm
             
             embeddings.append(emb)
        return embeddings

    async def get_embeddings_batch(
        self, text_batches: List[List[str]], instruction: Optional[str] = None, is_query: bool = False
    ) -> List[List[np.ndarray]]:
        tasks = [self.get_embeddings(batch, instruction, is_query) for batch in text_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        embeddings_batches = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing batch {i}: {result}")
                embeddings_batches.append([])
            else:
                embeddings_batches.append(result)
        return embeddings_batches

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "timeout": self.config.timeout,
            "batch_size": self.config.batch_size,
            "max_concurrent": self.config.max_concurrent_requests,
            "encoding_format": self.config.encoding_format,
        }


def get_vectorize_service() -> VectorizeServiceInterface:
    return get_bean("vectorize_service")


# 便捷函数
async def get_text_embedding(text: str, instruction: Optional[str] = None, is_query: bool = False) -> np.ndarray:
    return await get_vectorize_service().get_embedding(text, instruction, is_query)

async def get_text_embeddings(texts: List[str], instruction: Optional[str] = None, is_query: bool = False) -> List[np.ndarray]:
    return await get_vectorize_service().get_embeddings(texts, instruction, is_query)

async def get_text_embeddings_batch(text_batches: List[List[str]], instruction: Optional[str] = None, is_query: bool = False) -> List[List[np.ndarray]]:
    return await get_vectorize_service().get_embeddings_batch(text_batches, instruction, is_query)

async def get_text_embedding_with_usage(text: str, instruction: Optional[str] = None, is_query: bool = False) -> Tuple[np.ndarray, Optional[UsageInfo]]:
    return await get_vectorize_service().get_embedding_with_usage(text, instruction, is_query)
