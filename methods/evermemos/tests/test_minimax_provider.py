"""Unit tests for MiniMax provider support via OpenAIProvider."""

import os
import pytest
from unittest.mock import AsyncMock, patch

from memory_layer.llm.openai_provider import OpenAIProvider
from memory_layer.llm.llm_provider import LLMProvider, resolve_provider_env
from memory_layer.llm.api_key_rotator import ApiKeyRotator


@pytest.fixture(autouse=True)
def _reset_shared_rotator():
    """Ensure each test starts with a clean singleton state."""
    ApiKeyRotator._shared = None
    yield
    ApiKeyRotator._shared = None


def _success_body(content: str = "Hello from MiniMax!") -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


class TestMiniMaxProviderInstantiation:
    """Test that OpenAIProvider can be configured for MiniMax."""

    def test_creates_instance_with_minimax_config(self) -> None:
        provider = OpenAIProvider(
            model="MiniMax-M2.7",
            api_key="test-minimax-key",
            base_url="https://api.minimax.io/v1",
            provider_type="minimax",
        )
        assert provider is not None
        assert provider.model == "MiniMax-M2.7"
        assert provider.base_url == "https://api.minimax.io/v1"
        assert provider.provider_type == "minimax"

    def test_creates_instance_with_highspeed_model(self) -> None:
        provider = OpenAIProvider(
            model="MiniMax-M2.7-highspeed",
            api_key="test-minimax-key",
            base_url="https://api.minimax.io/v1",
            provider_type="minimax",
        )
        assert provider.model == "MiniMax-M2.7-highspeed"

    def test_default_temperature_is_applied(self) -> None:
        provider = OpenAIProvider(
            model="MiniMax-M2.7",
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
            provider_type="minimax",
        )
        # MiniMax temperature range is (0.0, 1.0] - default should not be 0
        assert provider.temperature > 0.0
        assert provider.temperature <= 1.0


class TestMiniMaxRequestBuilding:
    """Test that requests built for MiniMax don't include null/unsupported fields."""

    def test_request_data_excludes_null_provider_field(self) -> None:
        provider = OpenAIProvider(
            model="MiniMax-M2.7",
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
            provider_type="minimax",
        )
        with patch.dict(os.environ, {}, clear=False):
            # Remove the OpenRouter provider env var if set
            os.environ.pop("LLM_OPENROUTER_PROVIDER", None)
            data = provider._build_request_data("Hello", None, None, None)

        assert "provider" not in data, "MiniMax requests must not contain 'provider' field"
        assert "response_format" not in data, (
            "MiniMax requests must not contain 'response_format' field when None"
        )

    def test_request_data_includes_required_fields(self) -> None:
        provider = OpenAIProvider(
            model="MiniMax-M2.7",
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
            provider_type="minimax",
        )
        data = provider._build_request_data("Test prompt", 0.7, 1024, None)

        assert data["model"] == "MiniMax-M2.7"
        assert data["messages"] == [{"role": "user", "content": "Test prompt"}]
        assert data["temperature"] == 0.7
        assert data["max_tokens"] == 1024

    def test_request_data_includes_response_format_when_provided(self) -> None:
        provider = OpenAIProvider(
            model="MiniMax-M2.7",
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
            provider_type="minimax",
        )
        response_format = {"type": "json_object"}
        data = provider._build_request_data("Test", None, None, response_format)

        assert data["response_format"] == response_format

    def test_request_url_uses_minimax_endpoint(self) -> None:
        provider = OpenAIProvider(
            model="MiniMax-M2.7",
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
            provider_type="minimax",
        )
        assert provider.base_url == "https://api.minimax.io/v1"


class TestMiniMaxGenerate:
    """Test MiniMax generate call (mock HTTP)."""

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        provider = OpenAIProvider(
            model="MiniMax-M2.7",
            api_key="test-minimax-key",
            base_url="https://api.minimax.io/v1",
            provider_type="minimax",
        )

        async def mock_do_request(data: dict, api_key: str) -> tuple[int, dict]:
            assert api_key == "test-minimax-key"
            assert data["model"] == "MiniMax-M2.7"
            assert "provider" not in data
            return 200, _success_body("Hello from MiniMax!")

        provider._do_request = mock_do_request
        result = await provider.generate("Say hello")
        assert result == "Hello from MiniMax!"

    @pytest.mark.asyncio
    async def test_generate_m2_7_highspeed(self) -> None:
        provider = OpenAIProvider(
            model="MiniMax-M2.7-highspeed",
            api_key="test-minimax-key",
            base_url="https://api.minimax.io/v1",
            provider_type="minimax",
        )

        async def mock_do_request(data: dict, api_key: str) -> tuple[int, dict]:
            assert data["model"] == "MiniMax-M2.7-highspeed"
            return 200, _success_body("Fast response!")

        provider._do_request = mock_do_request
        result = await provider.generate("Test highspeed model")
        assert result == "Fast response!"

    @pytest.mark.asyncio
    async def test_generate_respects_custom_temperature(self) -> None:
        provider = OpenAIProvider(
            model="MiniMax-M2.7",
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
            provider_type="minimax",
        )
        captured_data: list[dict] = []

        async def capture_request(data: dict, api_key: str) -> tuple[int, dict]:
            captured_data.append(data)
            return 200, _success_body()

        provider._do_request = capture_request
        await provider.generate("Test", temperature=0.8)
        assert captured_data[0]["temperature"] == 0.8


class TestMiniMaxEnvConfig:
    """Test MiniMax provider resolution via environment variables."""

    def test_resolve_minimax_api_key_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {"MINIMAX_API_KEY": "env-minimax-key", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
        ):
            api_key, base_url = resolve_provider_env("minimax")
            assert api_key == "env-minimax-key"
            assert base_url == "https://api.minimax.io/v1"

    def test_resolve_minimax_env_case_insensitive(self) -> None:
        with patch.dict(
            os.environ,
            {"MINIMAX_API_KEY": "test-key", "MINIMAX_BASE_URL": "https://api.minimax.io/v1"},
        ):
            api_key, base_url = resolve_provider_env("minimax")
            assert api_key == "test-key"
            assert base_url == "https://api.minimax.io/v1"


class TestMiniMaxWhitelist:
    """Test MiniMax model whitelist enforcement."""

    def test_whitelist_allows_m2_7(self) -> None:
        with patch.dict(os.environ, {"MINIMAX_WHITE_LIST": "MiniMax-M2.7,MiniMax-M2.7-highspeed"}):
            # Should not raise
            OpenAIProvider._validate_model_whitelist("minimax", "MiniMax-M2.7")
            OpenAIProvider._validate_model_whitelist("minimax", "MiniMax-M2.7-highspeed")

    def test_whitelist_rejects_unknown_model(self) -> None:
        with patch.dict(os.environ, {"MINIMAX_WHITE_LIST": "MiniMax-M2.7,MiniMax-M2.7-highspeed"}):
            with pytest.raises(ValueError, match="only supports"):
                OpenAIProvider._validate_model_whitelist("minimax", "gpt-4o")

    def test_no_whitelist_allows_any_model(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINIMAX_WHITE_LIST", None)
            # Should not raise for any model name
            OpenAIProvider._validate_model_whitelist("minimax", "MiniMax-M2.7")
            OpenAIProvider._validate_model_whitelist("minimax", "any-other-model")
