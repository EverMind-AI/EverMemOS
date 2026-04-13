"""
OpenAI-compatible LLM provider implementation.

This provider uses a caller-supplied API key and base URL.
"""

from math import log
import os
import time
import json
import urllib.request
import urllib.parse
import urllib.error
import aiohttp
from typing import Optional
import asyncio
import random

from memory_layer.llm.protocol import LLMProvider, LLMError
from core.observation.logger import get_logger
from core.di.utils import get_bean_by_type
from core.component.token_usage_collector import TokenUsageCollector

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI-compatible LLM provider.

    This provider expects the caller to supply API key and base URL.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = 100 * 1024,
        enable_stats: bool = False,  # New: optional statistics feature, disabled by default
        provider_type: str | None = None,  # Provider type: "openai" or "openrouter"
        **kwargs,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
            api_key: API key (required by caller)
            base_url: API base URL (required by caller)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            enable_stats: Enable usage statistics accumulation (default: False)
            provider_type: Provider type ("openai" or "openrouter")
            **kwargs: Additional arguments (ignored for now)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_stats = enable_stats
        self.provider_type = (provider_type or "openrouter").lower()
        self.api_key = api_key
        self.base_url = base_url

        # Validate model whitelist from env: {PROVIDER}_WHITE_LIST
        # If whitelist is empty or not set, no restriction is applied.
        self._validate_model_whitelist(self.provider_type, model)

        # Optional per-call statistics (disabled by default)
        if self.enable_stats:
            self.current_call_stats = None  # Store statistics for current call

    @staticmethod
    def _validate_model_whitelist(provider_type: str, model: str) -> None:
        """
        Validate model against the provider's whitelist from environment variable.

        Reads {PROVIDER}_WHITE_LIST env var (comma-separated model names).
        If the env var is not set or empty, no restriction is applied.
        """
        env_key = f"{provider_type.upper()}_WHITE_LIST"
        raw = os.getenv(env_key, "").strip()
        if not raw:
            return
        allowed_models = {m.strip() for m in raw.split(",") if m.strip()}
        if not allowed_models:
            return
        if model not in allowed_models:
            raise ValueError(
                f"Provider '{provider_type}' only supports: {', '.join(sorted(allowed_models))}. Got: '{model}'."
            )

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt
            temperature: Override temperature for this request
            max_tokens: Override max tokens for this request

        Returns:
            Generated response text

        Raises:
            LLMError: If generation fails
        """
        # Use time.perf_counter() for more precise time measurement
        start_time = time.perf_counter()
        # Prepare request data
        if os.getenv("LLM_OPENROUTER_PROVIDER", "default") != "default":
            provider_str = os.getenv('LLM_OPENROUTER_PROVIDER')
            provider_list = [p.strip() for p in provider_str.split(',')]
            openrouter_provider = {"order": provider_list, "allow_fallbacks": False}
        else:
            openrouter_provider = None
        # Prepare request data
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "provider": openrouter_provider,
            "response_format": response_format,
        }
        # print(data)
        # print(data["extra_body"])
        # Add max_tokens if specified
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        elif self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens

        # Merge per-call extra_body into request data
        if extra_body:
            data.update(extra_body)

        # Use asynchronous aiohttp instead of synchronous urllib
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        max_retries = 5
        for retry_num in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=600)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions", json=data, headers=headers
                    ) as response:
                        chunks = []
                        async for chunk in response.content.iter_any():
                            chunks.append(chunk)
                        test = b"".join(chunks).decode()
                        response_data = json.loads(test)
                        # print(response_data)
                        # Handle error responses
                        if response.status != 200:
                            error_msg = response_data.get('error', {}).get(
                                'message', f"HTTP {response.status}"
                            )
                            logger.error(
                                f"❌ [OpenAI-{self.model}] HTTP error {response.status}:"
                            )
                            logger.error(f"   💬 Error message: {error_msg}")

                            # Retryable errors: rate limit, server errors
                            if response.status in (429, 500, 502, 503, 504):
                                logger.warning(
                                    f"Retryable error {response.status}, retry {retry_num + 1}/{max_retries}"
                                )
                                await asyncio.sleep(random.randint(5, 20))
                                if retry_num < max_retries - 1:
                                    continue  # Retry
                                # Last retry failed
                                raise LLMError(
                                    f"HTTP Error {response.status}: {error_msg} (after {max_retries} retries)"
                                )

                            # Non-retryable errors (401, 403, 404, 400, etc.)
                            raise LLMError(f"HTTP Error {response.status}: {error_msg}")

                        # Use time.perf_counter() for more precise time measurement
                        end_time = time.perf_counter()

                        # Extract finish_reason
                        finish_reason = response_data.get('choices', [{}])[0].get(
                            'finish_reason', ''
                        )
                        if finish_reason == 'stop':
                            logger.debug(
                                f"[OpenAI-{self.model}] Finish reason: {finish_reason}"
                            )
                        else:
                            logger.warning(
                                f"[OpenAI-{self.model}] Finish reason: {finish_reason}"
                            )

                        # Extract token usage information
                        usage = response_data.get('usage', {})
                        prompt_tokens = usage.get('prompt_tokens', 0)
                        completion_tokens = usage.get('completion_tokens', 0)
                        total_tokens = usage.get('total_tokens', 0)

                        # Print detailed usage information

                        logger.debug(f"[OpenAI-{self.model}] API call completed:")
                        logger.debug(
                            f"[OpenAI-{self.model}] Duration: {end_time - start_time:.2f}s"
                        )
                        # If the duration is too long
                        if end_time - start_time > 30:
                            logger.warning(
                                f"[OpenAI-{self.model}] Duration too long: {end_time - start_time:.2f}s"
                            )
                        logger.debug(
                            f"[OpenAI-{self.model}] Prompt Tokens: {prompt_tokens:,}"
                        )
                        logger.debug(
                            f"[OpenAI-{self.model}] Completion Tokens: {completion_tokens:,}"
                        )
                        logger.debug(
                            f"[OpenAI-{self.model}] Total Tokens: {total_tokens:,}"
                        )

                        # Report token usage to collector
                        try:
                            collector = get_bean_by_type(TokenUsageCollector)
                            collector.add(
                                self.model,
                                prompt_tokens,
                                completion_tokens,
                                call_type="llm",
                            )
                        except Exception:
                            pass

                        # New: record statistics for current call (if statistics enabled)
                        if self.enable_stats:
                            self.current_call_stats = {
                                'prompt_tokens': prompt_tokens,
                                'completion_tokens': completion_tokens,
                                'total_tokens': total_tokens,
                                'duration': end_time - start_time,
                                'timestamp': time.time(),
                            }

                        message = response_data['choices'][0]['message']
                        reasoning = message.get('reasoning_content') or message.get('reasoning') or message.get('thinking')
                        if reasoning:
                            logger.debug(
                                f"[OpenAI-{self.model}] "
                                f"🧠 Thinking detected: "
                                f"{len(reasoning)} chars"
                            )
                        else:
                            logger.debug(
                                f"[OpenAI-{self.model}] "
                                f"💭 No thinking in response"
                            )
                        return message['content']

            except aiohttp.ClientError as e:
                error_time = time.perf_counter()
                logger.error("aiohttp.ClientError: %s", e)
                # logger.error(f"❌ [OpenAI-{self.model}] Request failed:")
                logger.error(f"   ⏱️  Duration: {error_time - start_time:.2f}s")
                logger.error(f"   💬 Error message: {str(e)}")
                logger.error(f"retry_num: {retry_num}")
                # raise LLMError(f"Request failed: {str(e)}")
                if retry_num == max_retries - 1:
                    raise LLMError(f"Request failed: {str(e)}")
            except Exception as e:
                error_time = time.perf_counter()
                logger.error("Exception: %s", e)
                logger.error(f"   ⏱️  Duration: {error_time - start_time:.2f}s")
                logger.error(f"   💬 Error message: {str(e)}")
                logger.error(f"retry_num: {retry_num}")
                if retry_num == max_retries - 1:
                    raise LLMError(f"Request failed: {str(e)}")

    async def test_connection(self) -> bool:
        """
        Test the connection to the OpenRouter API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"🔗 [OpenAI-{self.model}] Testing API connection...")
            # Try a simple generation to test connection
            test_response = await self.generate("Hello", temperature=0.1)
            success = len(test_response) > 0
            if success:
                logger.info(f"✅ [OpenAI-{self.model}] API connection test succeeded")
            else:
                logger.error(
                    f"❌ [OpenAI-{self.model}] API connection test failed: Empty response"
                )
            return success
        except Exception as e:
            logger.error(f"❌ [OpenAI-{self.model}] API connection test failed: {e}")
            return False

    def get_current_call_stats(self) -> Optional[dict]:
        if self.enable_stats:
            return self.current_call_stats
        return None

    def __repr__(self) -> str:
        """String representation of the provider."""
        return (
            "OpenAIProvider("
            f"provider_type={self.provider_type}, model={self.model}, base_url={self.base_url}"
            ")"
        )
