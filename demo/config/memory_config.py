import os
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import Optional


class ScenarioType(Enum):
    ASSISTANT = "assistant"
    GROUP_CHAT = "group_chat"


@dataclass
class LLMConfig:
    """LLM 配置 - 从环境变量自动加载"""
    provider: str = "openai"
    model: Optional[str] = field(default_factory=lambda: os.getenv("LLM_MODEL"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    temperature: float = 0.3
    max_tokens: int = 16384


@dataclass
class EmbeddingConfig:
    base_url: str = field(default_factory=lambda: os.getenv("EMB_BASE_URL", "http://0.0.0.0:11000/v1/embeddings"))
    model: str = field(default_factory=lambda: os.getenv("EMB_MODEL", "Qwen3-Embedding-4B"))


@dataclass
class MongoDBConfig:
    """MongoDB 配置 - 支持通过 URI 参数追加认证信息"""
    uri: str = field(default_factory=lambda: os.getenv("MONGODB_URI", "mongodb://localhost:27017/memsys"))
    database: str = field(default_factory=lambda: os.getenv("MONGODB_DATABASE", "memsys"))
    
    def __post_init__(self):
        uri_params = os.getenv("MONGODB_URI_PARAMS", "").strip()
        if uri_params:
            separator = '&' if '?' in self.uri else '?'
            self.uri = f"{self.uri}{separator}{uri_params}"


@dataclass
class ExtractModeConfig:
    
    scenario_type: ScenarioType = ScenarioType.GROUP_CHAT
    language: str = "zh"
    
    # 可选覆盖
    data_file: Optional[Path] = None
    output_dir: Optional[Path] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    enable_profile_extraction: bool = True

    def __post_init__(self):
        # 自动设置输出目录
        if self.output_dir is None:
            self.output_dir = Path(__file__).parent.parent / "memcell_outputs" / f"{self.scenario_type.value}_{self.language}"
        
        # 根据场景设置默认值
        if self.scenario_type == ScenarioType.GROUP_CHAT:
            self.group_id = self.group_id or "group_chat_001"
            self.group_name = self.group_name or "Project Discussion Group"
            self.enable_semantic_extraction = False
        else:
            self.group_id = self.group_id or "assistant"
            self.group_name = self.group_name or "Personal Assistant"
            self.enable_semantic_extraction = True
        
        # 向后兼容
        self.prompt_language = self.language


@dataclass
class ChatModeConfig:
    """对话系统配置 - 使用合理默认值"""
    
    # 核心参数（通常不需要修改）
    api_base_url: str = field(default_factory=lambda: os.getenv("API_BASE_URL", "http://localhost:8001"))
    top_k_memories: int = 20
    conversation_history_size: int = 10
    time_range_days: int = 365
    show_retrieved_memories: bool = True
    
    # 路径（自动设置）
    chat_history_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "chat_history")
    memcell_output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "memcell_outputs")

    def __post_init__(self):
        self.chat_history_dir.mkdir(parents=True, exist_ok=True)
