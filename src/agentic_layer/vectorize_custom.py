"""
Custom (Self-Deployed) Vectorize Service Implementation

This module provides vectorization service for self-deployed embedding servers,
such as vLLM, Ollama, or other OpenAI-compatible endpoints.
"""

import os
import asyncio
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from openai import AsyncOpenAI

from agentic_layer.vectorize_interface import (
    VectorizeServiceInterface,
    VectorizeError,
    UsageInfo,
)

logger = logging.getLogger(__name__)


@dataclass
class CustomVectorizeConfig:
    """Configuration for custom self-deployed vectorization service"""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"  # Many self-deployed services don't require API key
    model: str = "Qwen/Qwen3-Embedding-4B"
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5
    encoding_format: str = "float"
    dimensions: int = 1024  # Client-side truncation target

    def __post_init__(self):
        """Load configuration from environment variables"""
        # Always read from environment variables, they override defaults
        self.base_url = os.getenv("CUSTOM_EMBEDDING_URL", self.base_url)
        self.api_key = os.getenv("CUSTOM_EMBEDDING_API_KEY", self.api_key)
        self.model = os.getenv("CUSTOM_EMBEDDING_MODEL", self.model)

        self.timeout = int(os.getenv("VECTORIZE_TIMEOUT", str(self.timeout)))
        self.max_retries = int(os.getenv("VECTORIZE_MAX_RETRIES", str(self.max_retries)))
        self.batch_size = int(os.getenv("VECTORIZE_BATCH_SIZE", str(self.batch_size)))
        self.max_concurrent_requests = int(
            os.getenv("VECTORIZE_MAX_CONCURRENT", str(self.max_concurrent_requests))
        )
        self.encoding_format = os.getenv("VECTORIZE_ENCODING_FORMAT", self.encoding_format)
        self.dimensions = int(os.getenv("VECTORIZE_DIMENSIONS", str(self.dimensions)))


class CustomVectorizeService(VectorizeServiceInterface):
    """
    Custom self-deployed embedding service implementation
    
    Supports:
    - vLLM (https://github.com/vllm-project/vllm)
    - Ollama (https://ollama.ai)
    - Any OpenAI-compatible embedding endpoint
    """

    def __init__(self, config: Optional[CustomVectorizeConfig] = None):
        if config is None:
            config = CustomVectorizeConfig()

        self.config = config
        self.client: Optional[AsyncOpenAI] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        logger.info(
            f"Initialized CustomVectorizeService | model={config.model} | base_url={config.base_url}"
        )

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_client(self):
        """Ensure OpenAI client is initialized"""
        if self.client is None:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )

    async def close(self):
        """Close the client connection"""
        if self.client:
            await self.client.close()
            self.client = None

    async def _make_request(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        is_query: bool = False,
    ):
        """Make embedding request to custom service"""
        await self._ensure_client()
        if not self.config.model:
            raise VectorizeError("Embedding model is not configured.")

        # Format texts with instruction if needed
        if is_query:
            default_instruction = (
                "Given a search query, retrieve relevant passages that answer the query"
            )
            final_instruction = (
                instruction if instruction is not None else default_instruction
            )
            formatted_texts = [
                f"Instruct: {final_instruction}\nQuery: {text}" for text in texts
            ]
        else:
            formatted_texts = texts

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    request_kwargs = {
                        "model": self.config.model,
                        "input": formatted_texts,
                        "encoding_format": self.config.encoding_format,
                        # Never pass dimensions parameter - always use client-side truncation
                    }

                    response = await self.client.embeddings.create(**request_kwargs)
                    return response

                except Exception as e:
                    error_msg = str(e)
                    logger.error(
                        f"Custom service API error (attempt {attempt + 1}/{self.config.max_retries}): {error_msg}"
                    )
                    
                    # Log detailed error for debugging
                    if "Connection" in error_msg or "timeout" in error_msg.lower():
                        logger.warning(
                            f"Network issue connecting to {self.config.base_url}: {error_msg}"
                        )
                    
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    else:
                        raise VectorizeError(f"Custom service API request failed after {self.config.max_retries} attempts: {error_msg}")

    def _parse_embeddings_response(self, response) -> List[np.ndarray]:
        """Parse embeddings from API response"""
        if not response.data:
            raise VectorizeError("Invalid API response: missing data")

        embeddings = []
        for item in response.data:
            emb = np.array(item.embedding, dtype=np.float32)

            # Client-side truncation (simple truncation without re-normalization)
            if (
                self.config.dimensions
                and self.config.dimensions > 0
                and len(emb) > self.config.dimensions
            ):
                logger.debug(
                    f"Client-side truncation: {len(emb)}D → {self.config.dimensions}D"
                )
                emb = emb[: self.config.dimensions]
            embeddings.append(emb)
        return embeddings

    async def get_embedding(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> np.ndarray:
        """Get embedding for a single text"""
        response = await self._make_request([text], instruction, is_query)
        if not response.data:
            raise VectorizeError("Invalid API response: missing data")
        return np.array(self._parse_embeddings_response(response)[0], dtype=np.float32)

    async def get_embedding_with_usage(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> Tuple[np.ndarray, Optional[UsageInfo]]:
        """Get embedding with usage information"""
        response = await self._make_request([text], instruction, is_query)
        if not response.data:
            raise VectorizeError("Invalid API response: missing data")

        embeddings = self._parse_embeddings_response(response)
        embedding = np.array(embeddings[0], dtype=np.float32)
        usage_info = (
            UsageInfo.from_openai_usage(response.usage) if response.usage else None
        )
        return embedding, usage_info

    async def get_embeddings(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        is_query: bool = False,
    ) -> List[np.ndarray]:
        """Get embeddings for multiple texts"""
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

    async def get_embeddings_batch(
        self,
        text_batches: List[List[str]],
        instruction: Optional[str] = None,
        is_query: bool = False,
    ) -> List[List[np.ndarray]]:
        """Get embeddings for multiple batches"""
        tasks = [
            self.get_embeddings(batch, instruction, is_query) for batch in text_batches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        embeddings_batches = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing batch {i}: {result}")
                embeddings_batches.append([])
            else:
                embeddings_batches.append(result)
        return embeddings_batches

    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.config.model

