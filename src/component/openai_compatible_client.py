import asyncio
import os
from typing import Dict, Any, List, Optional, AsyncGenerator, Union

from core.di.decorators import component
from core.observation.logger import get_logger
from component.config_provider import ConfigProvider

from component.llm_adapter.llm.message import ChatMessage
from component.llm_adapter.llm.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from component.llm_adapter.llm.llm_backend_adapter import LLMBackendAdapter
from component.llm_adapter.llm.openai_adapter import OpenAIAdapter
from component.llm_adapter.llm.anthropic_adapter import AnthropicAdapter
from component.llm_adapter.llm.gemini_adapter import GeminiAdapter

logger = get_logger(__name__)


@component(name="openai_compatible_client", primary=True)
class OpenAICompatibleClient:
    """
    OpenAI兼容API客户端。
    该客户端作为一个外观（Facade），管理多个LLM后端适配器。
    """

    def __init__(self, config_provider: ConfigProvider):
        """
        初始化客户端。
        Args:
            config_provider: 配置提供者，用于加载 llm_backends.yaml。
        """
        self.config_provider = config_provider
        self._adapters: Dict[str, LLMBackendAdapter] = {}
        self._config: Dict[str, Any] = self.config_provider.get_config("llm_backends")
        self._init_locks: Dict[str, asyncio.Lock] = {}  # 每个后端一个锁
        self._lock_creation_lock = asyncio.Lock()  # 用于创建锁的锁

    async def _get_adapter(self, backend_name: str) -> LLMBackendAdapter:
        """
        按需异步初始化并获取指定后端的适配器
        使用锁确保并发安全，避免重复初始化
        """
        # 如果适配器已存在，直接返回
        if backend_name in self._adapters:
            return self._adapters[backend_name]

        # 确保每个后端都有对应的锁
        async with self._lock_creation_lock:
            if backend_name not in self._init_locks:
                self._init_locks[backend_name] = asyncio.Lock()

        # 使用后端特定的锁来确保并发安全
        async with self._init_locks[backend_name]:
            # 再次检查，因为可能在等待锁的过程中已经被其他协程初始化了
            if backend_name in self._adapters:
                return self._adapters[backend_name]

            llm_backends = self._config.get("llm_backends", {})
            if backend_name not in llm_backends:
                raise ValueError(
                    f"Backend '{backend_name}' not found in configuration."
                )

            backend_config = llm_backends[backend_name]
            provider = backend_config.get("provider", "openai")

            try:
                adapter: LLMBackendAdapter
                if provider in ["openai", "azure", "custom", "ollama"]:
                    adapter = OpenAIAdapter(backend_config)
                elif provider == "anthropic":
                    adapter = AnthropicAdapter(backend_config)
                elif provider == "gemini":
                    adapter = GeminiAdapter(backend_config)
                else:
                    raise ValueError(f"Unsupported provider type: {provider}")

                self._adapters[backend_name] = adapter
                return adapter
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize adapter for backend '{backend_name}': {e}"
                ) from e

    def _get_param_with_priority(
        self,
        param_name: str,
        passed_value: Any,
        default_settings: dict,
        backend_config: dict,
    ) -> Any:
        """
        获取参数优先级：传入参数 > backend_config > default_settings
        """
        if passed_value is not None:
            return passed_value
        if backend_config.get(param_name) is not None:
            return backend_config.get(param_name)
        if default_settings.get(param_name) is not None:
            return default_settings[param_name]
        return None

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        thinking_budget: Optional[int] = None,  # 添加thinking_budget参数支持
        stream: bool = False,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """
        执行聊天完成。
        Args:
            messages: 聊天消息列表
            backend: 后端名称，若不指定则使用默认后端
            ... other params
            thinking_budget: 思考预算（用于支持think功能）
            stream: 是否流式响应
        Returns:
            聊天完成响应或流式生成器
        """
        # 选择后端
        backend_name = backend or self._config.get("default_backend", "openai")
        default_settings = self._config.get("default_settings", {})
        backend_config = self._config.get("llm_backends", {}).get(backend_name, {})

        # 统一参数优先级处理
        final_params = {}
        param_definitions = {
            # 下面backend_config有默认值
            "model": model,
            # 下面default_settings有默认值
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "thinking_budget": thinking_budget,
        }
        for name, value in param_definitions.items():
            final_params[name] = self._get_param_with_priority(
                name, value, default_settings, backend_config
            )

        # 组装请求
        request = ChatCompletionRequest(
            messages=messages, stream=stream, **final_params
        )

        adapter = await self._get_adapter(backend_name)
        return await adapter.chat_completion(request)

    def chat_completion_sync(
        self,
        messages: List[ChatMessage],
        backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: bool = False,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """
        执行聊天完成的同步版本。

        注意：此方法不再支持同步调用，因为某些LLM适配器（如Gemini）
        内部会绑定事件循环，创建新的线程和事件循环会导致问题。

        请使用异步版本 chat_completion() 方法。
        """
        raise NotImplementedError(
            "同步版本的聊天完成不再支持，因为某些LLM适配器（如Gemini）内部会绑定事件循环。"
            "请使用异步版本 chat_completion() 方法。"
        )

    def get_available_backends(self) -> List[str]:
        """获取可用后端列表"""
        return list(self._config.get("llm_backends", {}).keys())

    async def get_available_models(self, backend: Optional[str] = None) -> List[str]:
        """获取指定后端的可用模型列表"""
        backend_name = backend or self._config.get("default_backend", "openai")
        try:
            adapter = await self._get_adapter(backend_name)
            return adapter.get_available_models()
        except (ValueError, RuntimeError):
            return []

    def get_available_models_sync(self, backend: Optional[str] = None) -> List[str]:
        """
        获取指定后端的可用模型列表的同步版本。

        注意：此方法不再支持同步调用，因为某些LLM适配器（如Gemini）
        内部会绑定事件循环，创建新的线程和事件循环会导致问题。

        请使用异步版本 get_available_models() 方法。
        """
        raise NotImplementedError(
            "同步版本的模型获取不再支持，因为某些LLM适配器（如Gemini）内部会绑定事件循环。"
            "请使用异步版本 get_available_models() 方法。"
        )

    def get_backend_info(self, backend: str) -> Optional[Dict[str, Any]]:
        """获取后端信息，隐藏敏感信息"""
        config = self._config.get("llm_backends", {}).get(backend)
        if config:
            safe_config = config.copy()
            if "api_key" in safe_config:
                safe_config["api_key"] = (
                    f"***{safe_config['api_key'][-4:]}"
                    if len(safe_config.get('api_key', '')) > 4
                    else "***"
                )
            return safe_config
        return None

    def reload_config(self):
        """重新加载配置并清空现有适配器实例和锁"""
        self._config = self.config_provider.get_config("llm_backends")
        self._adapters.clear()
        self._init_locks.clear()

    async def close(self):
        """关闭所有适配器的HTTP客户端连接"""
        for adapter in self._adapters.values():
            if hasattr(adapter, 'close'):
                await adapter.close()  # type: ignore

    def close_sync(self):
        """
        关闭所有适配器的HTTP客户端连接的同步版本。

        注意：此方法不再支持同步调用，因为某些LLM适配器（如Gemini）
        内部会绑定事件循环，创建新的线程和事件循环会导致问题。

        请使用异步版本 close() 方法。
        """
        raise NotImplementedError(
            "同步版本的关闭操作不再支持，因为某些LLM适配器（如Gemini）内部会绑定事件循环。"
            "请使用异步版本 close() 方法。"
        )
