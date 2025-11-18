"""
DeepInfra Vectorize Service
DeepInfra向量化服务

This module provides methods to call DeepInfra API for getting text embeddings.
该模块提供调用DeepInfra API获取文本embedding向量的方法。
"""

from __future__ import annotations

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from openai import AsyncOpenAI

from core.di.utils import get_bean
from core.di.decorators import service

logger = logging.getLogger(__name__)


@dataclass
@service(name="deepinfra_config", primary=True)
class DeepInfraConfig:
    """DeepInfra API配置类"""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5
    encoding_format: str = "float"  # 编码格式
    dimensions: int = 1024

    def __post_init__(self):
        """初始化后从环境变量加载配置值"""
        if not self.api_key:
            self.api_key = os.getenv("DEEPINFRA_API_KEY", "")
        if not self.base_url:
            self.base_url = os.getenv(
                "DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"
            )
        if not self.model:
            self.model = os.getenv(
                "DEEPINFRA_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B"
            )
        if self.timeout == 30:  # 使用默认值时才从环境变量读取
            self.timeout = int(os.getenv("DEEPINFRA_TIMEOUT", "30"))
        if self.max_retries == 3:  # 使用默认值时才从环境变量读取
            self.max_retries = int(os.getenv("DEEPINFRA_MAX_RETRIES", "3"))
        if self.batch_size == 10:  # 使用默认值时才从环境变量读取
            self.batch_size = int(os.getenv("DEEPINFRA_BATCH_SIZE", "10"))
        if self.max_concurrent_requests == 5:  # 使用默认值时才从环境变量读取
            self.max_concurrent_requests = int(
                os.getenv("DEEPINFRA_MAX_CONCURRENT", "5")
            )
        if self.encoding_format == "float":  # 使用默认值时才从环境变量读取
            self.encoding_format = os.getenv("DEEPINFRA_ENCODING_FORMAT", "float")
        if self.dimensions == 1024:  # 使用默认值时才从环境变量读取
            self.dimensions = int(os.getenv("DEEPINFRA_DIMENSIONS", "1024"))


class DeepInfraError(Exception):
    """DeepInfra API错误异常类"""

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


class DeepInfraVectorizeServiceInterface(ABC):
    """DeepInfra向量化服务接口"""

    @abstractmethod
    async def get_embedding(
        self, text: str, instruction: Optional[str] = None
    ) -> np.ndarray:
        """
        获取单个文本的embedding向量

        Args:
            text: 要向量化的文本
            instruction: 可选的指令文本

        Returns:
            embedding向量（numpy数组）
        """
        pass

    @abstractmethod
    async def get_embedding_with_usage(
        self, text: str, instruction: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[UsageInfo]]:
        """
        获取单个文本的embedding向量和使用情况

        Args:
            text: 要向量化的文本
            instruction: 可选的指令文本

        Returns:
            (embedding向量, 使用情况信息)
        """
        pass

    @abstractmethod
    async def get_embeddings(
        self, texts: List[str], instruction: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        获取多个文本的embedding向量

        Args:
            texts: 要向量化的文本列表
            instruction: 可选的指令文本

        Returns:
            embedding向量列表（numpy数组列表）
        """
        pass

    @abstractmethod
    async def get_embeddings_batch(
        self, text_batches: List[List[str]], instruction: Optional[str] = None
    ) -> List[List[np.ndarray]]:
        """
        批量获取多个文本批次的embedding向量

        Args:
            text_batches: 文本批次列表，每个批次包含多个文本
            instruction: 可选的指令文本

        Returns:
            每个批次的embedding向量列表
        """
        pass


@service(name="vectorize_service", primary=True)
class DeepInfraVectorizeService(DeepInfraVectorizeServiceInterface):
    """
    DeepInfra向量化服务类

    提供调用DeepInfra API获取文本embedding向量的方法
    """

    def __init__(self, config: Optional[DeepInfraConfig] = None):
        """
        初始化DeepInfra向量化服务

        Args:
            config: DeepInfra配置，如果为None则尝试从依赖注入获取，最后从环境变量读取
        """
        if config is None:
            try:
                # 尝试从依赖注入获取配置
                from core.di import get_bean

                config = get_bean("deepinfra_config")
                logger.info("DeepInfra config source: DI bean 'deepinfra_config'")
            except Exception:
                # 如果依赖注入失败，从环境变量读取
                config = self._load_config_from_env()
                # 打印.env传入情况（不输出密钥内容）
                logger.info(
                    "DeepInfra config source: env | DEEPINFRA_API_KEY set=%s | DEEPINFRA_BASE_URL=%s | DEEPINFRA_EMBEDDING_MODEL=%s | DEEPINFRA_DIMENSIONS=%s",
                    bool(os.getenv("DEEPINFRA_API_KEY")),
                    os.getenv("DEEPINFRA_BASE_URL"),
                    os.getenv("DEEPINFRA_EMBEDDING_MODEL"),
                    os.getenv("DEEPINFRA_DIMENSIONS"),
                )

        # 规范化配置，避免后续请求异常
        # 确保 base_url 包含协议
        base_url = config.base_url or ""
        if base_url and not (
            base_url.startswith("http://") or base_url.startswith("https://")
        ):
            base_url = f"https://{base_url}"
        # 确保模型非空
        model = config.model or os.getenv(
            "DEEPINFRA_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B"
        )

        # 写回配置对象
        config.base_url = base_url
        config.model = model

        self.config = config
        self.client: Optional[AsyncOpenAI] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        logger.info(
            f"Initialized DeepInfra Vectorize Service | model={config.model} | base_url={config.base_url}"
        )

    def get_model_name(self) -> str:
        """
        获取当前使用的embedding模型名称

        Returns:
            模型名称字符串
        """
        return self.config.model

    def _load_config_from_env(self) -> DeepInfraConfig:
        """从环境变量加载配置"""
        return DeepInfraConfig(
            api_key=os.getenv("DEEPINFRA_API_KEY", ""),
            base_url=os.getenv(
                "DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"
            ),
            model=os.getenv("DEEPINFRA_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B"),
            timeout=int(os.getenv("DEEPINFRA_TIMEOUT", "30")),
            max_retries=int(os.getenv("DEEPINFRA_MAX_RETRIES", "3")),
            batch_size=int(os.getenv("DEEPINFRA_BATCH_SIZE", "10")),
            max_concurrent_requests=int(os.getenv("DEEPINFRA_MAX_CONCURRENT", "5")),
            encoding_format=os.getenv("DEEPINFRA_ENCODING_FORMAT", "float"),
            dimensions=int(os.getenv("DEEPINFRA_DIMENSIONS", "1024")),
        )

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def _ensure_client(self):
        """确保OpenAI客户端已创建"""
        if self.client is None:
            # 规范化 base_url，防止缺少协议导致 httpx.UnsupportedProtocol
            base_url = self.config.base_url or ""
            if base_url and not (
                base_url.startswith("http://") or base_url.startswith("https://")
            ):
                # 默认使用 https
                base_url = f"https://{base_url}"

            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=base_url,
                timeout=self.config.timeout,
            )

    async def close(self):
        """关闭OpenAI客户端"""
        if self.client:
            await self.client.close()
            self.client = None

    async def _make_request(self, texts: List[str], instruction: Optional[str] = None):
        """
        向DeepInfra API发送请求

        Args:
            texts: 要向量化的文本列表
            instruction: 可选的指令文本（DeepInfra可能不支持，但保留接口兼容性）

        Returns:
            OpenAI embeddings响应对象

        Raises:
            DeepInfraError: API请求失败时抛出
        """
        await self._ensure_client()
        # 确保模型在请求时有效，避免传空字符串导致供应商报错
        if not self.config.model:
            raise DeepInfraError("Embedding model is not configured.")

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    # 使用OpenAI客户端调用DeepInfra API
                    response = await self.client.embeddings.create(
                        model=self.config.model,
                        input=texts,
                        encoding_format=self.config.encoding_format,
                        dimensions=self.config.dimensions,
                    )
                    return response

                except Exception as e:
                    logger.error(f"DeepInfra API error (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)  # 指数退避
                        continue
                    else:
                        raise DeepInfraError(f"API request failed: {e}")

    async def get_embedding(
        self, text: str, instruction: Optional[str] = None
    ) -> np.ndarray:
        """
        获取单个文本的embedding向量

        Args:
            text: 要向量化的文本
            instruction: 可选的指令文本

        Returns:
            embedding向量（numpy数组）

        Raises:
            DeepInfraError: API请求失败时抛出
        """
        response = await self._make_request([text], instruction)

        if not response.data or len(response.data) == 0:
            raise DeepInfraError("Invalid API response: missing data field")

        embedding_data = response.data[0].embedding
        return np.array(embedding_data, dtype=np.float32)

    async def get_embedding_with_usage(
        self, text: str, instruction: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[UsageInfo]]:
        """
        获取单个文本的embedding向量和使用情况

        Args:
            text: 要向量化的文本
            instruction: 可选的指令文本

        Returns:
            (embedding向量, 使用情况信息)

        Raises:
            DeepInfraError: API请求失败时抛出
        """
        response = await self._make_request([text], instruction)

        if not response.data or len(response.data) == 0:
            raise DeepInfraError("Invalid API response: missing data field")

        embedding_data = response.data[0].embedding
        embedding = np.array(embedding_data, dtype=np.float32)

        # 提取usage信息
        usage_info = None
        if response.usage:
            usage_info = UsageInfo.from_openai_usage(response.usage)

        return embedding, usage_info

    async def get_embeddings(
        self, texts: List[str], instruction: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        获取多个文本的embedding向量

        Args:
            texts: 要向量化的文本列表
            instruction: 可选的指令文本

        Returns:
            embedding向量列表（numpy数组列表）

        Raises:
            DeepInfraError: API请求失败时抛出
        """
        if not texts:
            return []

        # 如果文本数量较少，直接请求
        if len(texts) <= self.config.batch_size:
            response = await self._make_request(texts, instruction)
            return self._parse_embeddings_response(response)

        # 批量处理大量文本
        embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            response = await self._make_request(batch_texts, instruction)
            batch_embeddings = self._parse_embeddings_response(response)
            embeddings.extend(batch_embeddings)

            # 避免请求过于频繁
            if i + self.config.batch_size < len(texts):
                await asyncio.sleep(0.1)

        return embeddings

    def _parse_embeddings_response(self, response) -> List[np.ndarray]:
        """
        解析API响应中的embedding数据

        Args:
            response: OpenAI embeddings响应对象

        Returns:
            embedding向量列表

        Raises:
            DeepInfraError: 响应格式错误时抛出
        """
        if not response.data:
            raise DeepInfraError("Invalid API response: missing data field")

        embeddings = []
        for i, item in enumerate(response.data):
            if not hasattr(item, 'embedding'):
                raise DeepInfraError(
                    f"Invalid API response: data[{i}] missing 'embedding' field"
                )

            embedding_data = item.embedding
            if not isinstance(embedding_data, list):
                raise DeepInfraError(
                    f"Invalid API response: data[{i}].embedding should be a list"
                )

            if not embedding_data:
                raise DeepInfraError(
                    f"Invalid API response: data[{i}].embedding is empty"
                )

            # 验证embedding数据都是数字
            try:
                embedding_array = np.array(embedding_data, dtype=np.float32)
            except (ValueError, TypeError) as e:
                raise DeepInfraError(
                    f"Invalid API response: data[{i}].embedding contains non-numeric values: {e}"
                )

            embeddings.append(embedding_array)

        return embeddings

    async def get_embeddings_batch(
        self, text_batches: List[List[str]], instruction: Optional[str] = None
    ) -> List[List[np.ndarray]]:
        """
        批量获取多个文本批次的embedding向量

        Args:
            text_batches: 文本批次列表，每个批次包含多个文本
            instruction: 可选的指令文本

        Returns:
            每个批次的embedding向量列表
        """
        tasks = []
        for batch in text_batches:
            task = self.get_embeddings(batch, instruction)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        embeddings_batches = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing batch {i}: {result}")
                # 返回空列表作为占位符
                embeddings_batches.append([])
            else:
                embeddings_batches.append(result)

        return embeddings_batches

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前使用的模型信息

        Returns:
            模型信息字典
        """
        return {
            "model": self.config.model,
            "base_url": self.config.base_url,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
            "batch_size": self.config.batch_size,
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "encoding_format": self.config.encoding_format,
        }


def get_vectorize_service() -> DeepInfraVectorizeServiceInterface:
    """获取向量化服务实例

    通过依赖注入框架获取服务实例，支持单例模式。
    """
    return get_bean("vectorize_service")


# 便捷函数
async def get_text_embedding(
    text: str, instruction: Optional[str] = None
) -> np.ndarray:
    """
    便捷函数：获取单个文本的embedding向量

    Args:
        text: 要向量化的文本
        instruction: 可选的指令文本

    Returns:
        embedding向量（numpy数组）
    """
    service = get_vectorize_service()
    return await service.get_embedding(text, instruction)


async def get_text_embeddings(
    texts: List[str], instruction: Optional[str] = None
) -> List[np.ndarray]:
    """
    便捷函数：获取多个文本的embedding向量

    Args:
        texts: 要向量化的文本列表
        instruction: 可选的指令文本

    Returns:
        embedding向量列表（numpy数组列表）
    """
    service = get_vectorize_service()
    return await service.get_embeddings(texts, instruction)


async def get_text_embeddings_batch(
    text_batches: List[List[str]], instruction: Optional[str] = None
) -> List[List[np.ndarray]]:
    """
    便捷函数：批量获取多个文本批次的embedding向量

    Args:
        text_batches: 文本批次列表，每个批次包含多个文本
        instruction: 可选的指令文本

    Returns:
        每个批次的embedding向量列表
    """
    service = get_vectorize_service()
    return await service.get_embeddings_batch(text_batches, instruction)


async def get_text_embedding_with_usage(
    text: str, instruction: Optional[str] = None
) -> Tuple[np.ndarray, Optional[UsageInfo]]:
    """
    便捷函数：获取单个文本的embedding向量和使用情况

    Args:
        text: 要向量化的文本
        instruction: 可选的指令文本

    Returns:
        (embedding向量, 使用情况信息)
    """
    service = get_vectorize_service()
    return await service.get_embedding_with_usage(text, instruction)
