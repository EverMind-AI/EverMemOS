"""对话会话管理

管理单个群组的对话会话，提供记忆检索和 LLM 对话功能。
"""

import json
import httpx
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from demo.config import ChatModeConfig, LLMConfig, ScenarioType
from demo.utils import query_memcells_by_group_and_time
from demo.ui import I18nTexts
from memory_layer.llm.llm_provider import LLMProvider
from common_utils.datetime_utils import get_now_with_timezone


class ChatSession:
    """对话会话管理器"""
    
    def __init__(
        self,
        group_id: str,
        config: ChatModeConfig,
        llm_config: LLMConfig,
        scenario_type: ScenarioType,
        retrieval_mode: str,  # "rrf" / "embedding" / "bm25"
        data_source: str,     # "episode" / "event_log"
        texts: I18nTexts,
    ):
        """初始化对话会话
        
        Args:
            group_id: 群组 ID
            config: 对话模式配置
            llm_config: LLM 配置
            scenario_type: 场景类型
            retrieval_mode: 检索模式（rrf/embedding/bm25）
            data_source: 数据源（episode/event_log）
            texts: 国际化文本对象
        """
        self.group_id = group_id
        self.config = config
        self.llm_config = llm_config
        self.scenario_type = scenario_type
        self.retrieval_mode = retrieval_mode
        self.data_source = data_source
        self.texts = texts
        
        # 会话状态
        self.conversation_history: List[Tuple[str, str]] = []
        self.memcell_count: int = 0
        
        # 服务
        self.llm_provider: Optional[LLMProvider] = None
        
        # API 配置
        self.api_base_url = config.api_base_url
        self.retrieve_lightweight_url = f"{self.api_base_url}/api/v3/agentic/retrieve_lightweight"
        self.retrieve_agentic_url = f"{self.api_base_url}/api/v3/agentic/retrieve_agentic"
        
        # 最后一次检索元数据
        self.last_retrieval_metadata: Optional[Dict[str, Any]] = None
    
    async def initialize(self, load_history: bool = True) -> bool:
        """初始化会话
        
        Args:
            load_history: 是否加载历史记录
            
        Returns:
            初始化是否成功
        """
        try:
            display_name = "group_chat" if self.group_id == "AI产品群" else self.group_id
            print(f"\n[{self.texts.get('loading_label')}] {self.texts.get('loading_group_data', name=display_name)}")
            
            # 检查 API 服务器健康状态
            await self._check_api_server()
            
            # 统计 MemCell 数量
            now = get_now_with_timezone()
            start_date = now - timedelta(days=self.config.time_range_days)
            memcells = await query_memcells_by_group_and_time(self.group_id, start_date, now)
            self.memcell_count = len(memcells)
            print(f"[{self.texts.get('loading_label')}] {self.texts.get('loading_memories_success', count=self.memcell_count)} ✅")
            
            # 根据用户选择加载对话历史
            if load_history:
                loaded_history_count = await self.load_conversation_history()
                if loaded_history_count > 0:
                    print(f"[{self.texts.get('loading_label')}] {self.texts.get('loading_history_success', count=loaded_history_count)} ✅")
                else:
                    print(f"[{self.texts.get('loading_label')}] {self.texts.get('loading_history_new')} ✅")
            else:
                self.conversation_history = []
                print(f"[{self.texts.get('loading_label')}] {self.texts.get('history_cleared')} ✅")
            
            # 创建 LLM Provider
            self.llm_provider = LLMProvider(
                self.llm_config.provider,
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            
            print(f"\n[{self.texts.get('hint_label')}] {self.texts.get('loading_help_hint')}\n")
            return True
        
        except Exception as e:
            print(f"\n[{self.texts.get('error_label')}] {self.texts.get('session_init_error', error=str(e))}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _check_api_server(self) -> None:
        """检查 API 服务器是否运行
        
        Raises:
            ConnectionError: 如果服务器未运行
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # 尝试访问健康检查端点或任何端点
                response = await client.get(f"{self.api_base_url}/docs")
                if response.status_code >= 500:
                    raise ConnectionError("API 服务器返回错误")
        except (httpx.ConnectError, httpx.TimeoutException, ConnectionError) as e:
            error_msg = (
                f"\n❌ 无法连接到 API 服务器: {self.api_base_url}\n\n"
                f"请先启动 V3 API 服务器：\n"
                f"  uv run python src/bootstrap.py src/run.py --port 8001\n\n"
                f"然后在另一个终端运行聊天应用。\n"
            )
            raise ConnectionError(error_msg) from e
    
    async def load_conversation_history(self) -> int:
        """从文件加载对话历史
        
        Returns:
            加载的对话轮数
        """
        try:
            display_name = "group_chat" if self.group_id == "AI产品群" else self.group_id
            history_files = sorted(
                self.config.chat_history_dir.glob(f"{display_name}_*.json"),
                reverse=True
            )
            
            if not history_files:
                return 0
            
            latest_file = history_files[0]
            with latest_file.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            
            history = data.get("conversation_history", [])
            self.conversation_history = [
                (item["user_input"], item["assistant_response"])
                for item in history[-self.config.conversation_history_size:]
            ]
            
            return len(self.conversation_history)
        
        except Exception as e:
            print(f"[{self.texts.get('warning_label')}] {self.texts.get('loading_history_new')}: {e}")
            return 0
    
    async def save_conversation_history(self) -> None:
        """保存对话历史到文件"""
        try:
            display_name = "group_chat" if self.group_id == "AI产品群" else self.group_id
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename = f"{display_name}_{timestamp}.json"
            filepath = self.config.chat_history_dir / filename
            
            data = {
                "group_id": self.group_id,
                "last_updated": datetime.now().isoformat(),
                "conversation_history": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "user_input": user_q,
                        "assistant_response": assistant_a,
                    }
                    for user_q, assistant_a in self.conversation_history
                ],
            }
            
            with filepath.open("w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
            
            print(f"[{self.texts.get('save_label')}] {filename} ✅")
        
        except Exception as e:
            print(f"[{self.texts.get('error_label')}] {e}")
    
    async def retrieve_memories(self, query: str) -> List[Dict[str, Any]]:
        """检索相关记忆 - 通过 HTTP API 调用
        
        Args:
            query: 用户查询
            
        Returns:
            检索到的记忆列表
        """
        # 🔥 根据检索模式选择不同的 HTTP API 端点
        if self.retrieval_mode == "agentic":
            # Agentic 检索 API
            result = await self._call_retrieve_agentic_api(query)
        else:
            # Lightweight 检索 API
            result = await self._call_retrieve_lightweight_api(query)
        
        # 提取结果和元数据
        memories = result.get("memories", [])
        metadata = result.get("metadata", {})
        
        # 保存元数据（用于 UI 显示）
        self.last_retrieval_metadata = metadata
        
        return memories
    
    async def _call_retrieve_lightweight_api(self, query: str) -> Dict[str, Any]:
        """调用 Lightweight 检索 API（与 test_v3_retrieve_http.py 对齐）
        
        Args:
            query: 用户查询
            
        Returns:
            检索结果字典
        """
        # 🔥 关键：与 test_v3_retrieve_http.py 完全对齐
        payload = {
            "query": query,
            "user_id": "user_001",  # 与 test 保持一致
            "top_k": self.config.top_k_memories,
            "data_source": self.data_source,  # episode / event_log
            "retrieval_mode": self.retrieval_mode,  # rrf / embedding / bm25
            "memory_scope": "all",  # 检索所有记忆（个人 + 群组）
        }
        
        # 调试日志（仅在开发环境显示）
        # print(f"\n[DEBUG] Lightweight 检索请求:")
        # print(f"  - API URL: {self.retrieve_lightweight_url}")
        # print(f"  - query: {query}")
        # print(f"  - user_id: user_001")
        # print(f"  - retrieval_mode: {self.retrieval_mode}")
        # print(f"  - data_source: {self.data_source}")
        # print(f"  - memory_scope: all")
        # print(f"  - top_k: {self.config.top_k_memories}")
        
        try:
            # 🔥 与 test_v3_retrieve_http.py 完全一致：verify=False, timeout=30.0
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.post(self.retrieve_lightweight_url, json=payload)
                response.raise_for_status()
                api_response = response.json()
                
                # 检查 API 响应状态
                if api_response.get("status") == "ok":
                    result = api_response.get("result", {"memories": [], "metadata": {}})
                    # memories_count = len(result.get("memories", []))
                    # print(f"  ✅ 检索成功: {memories_count} 条记忆")
                    return result
                else:
                    error_msg = api_response.get('message', '未知错误')
                    # print(f"  ❌ API 返回错误: {error_msg}")
                    raise RuntimeError(f"API 返回错误: {error_msg}")
        
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            raise RuntimeError(error_msg)
        except httpx.TimeoutException:
            error_msg = "请求超时（超过30秒）"
            raise RuntimeError(error_msg)
        except httpx.ConnectError as e:
            error_msg = f"连接失败: 无法连接到 {self.api_base_url}\n请确保 V3 API 服务已启动: uv run python src/bootstrap.py src/run.py --port 8001"
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"检索失败: {type(e).__name__}: {e}"
            raise RuntimeError(error_msg)
    
    async def _call_retrieve_agentic_api(self, query: str) -> Dict[str, Any]:
        """调用 Agentic 检索 API（与 test_v3_retrieve_http.py 对齐）
        
        Args:
            query: 用户查询
            
        Returns:
            检索结果字典
        """
        # 🔥 关键：与 test_v3_retrieve_http.py 完全对齐
        payload = {
            "query": query,
            "user_id": "user_001",  # 与 test 保持一致
            "top_k": self.config.top_k_memories,
            "time_range_days": self.config.time_range_days,  # 使用配置的时间范围
        }
        
        # 调试日志（仅在开发环境显示）
        # print(f"\n[DEBUG] Agentic 检索请求:")
        # print(f"  - API URL: {self.retrieve_agentic_url}")
        # print(f"  - query: {query}")
        # print(f"  - user_id: user_001")
        # print(f"  - top_k: {self.config.top_k_memories}")
        # print(f"  - time_range_days: {self.config.time_range_days}")
        
        # 显示友好的等待提示
        print(f"\n⏳ 正在检索记忆...")
        # print(f"   涉及：LLM 充分性判断 → 多轮检索 → 结果融合")
        
        try:
            # 🔥 Agentic 检索需要更长时间：增加到 180 秒（3分钟）
            # 因为涉及 LLM 调用、充分性判断、多轮检索等复杂操作
            async with httpx.AsyncClient(timeout=180.0, verify=False) as client:
                response = await client.post(self.retrieve_agentic_url, json=payload)
                response.raise_for_status()
                api_response = response.json()
                
                # 检查 API 响应状态
                if api_response.get("status") == "ok":
                    result = api_response.get("result", {"memories": [], "metadata": {}})
                    # memories_count = len(result.get("memories", []))
                    # print(f"  ✅ 检索成功: {memories_count} 条记忆")
                    return result
                else:
                    error_msg = api_response.get('message', '未知错误')
                    # print(f"  ❌ API 返回错误: {error_msg}")
                    raise RuntimeError(f"API 返回错误: {error_msg}")
        
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            raise RuntimeError(error_msg)
        except httpx.TimeoutException:
            error_msg = "请求超时（超过180秒）\n提示：Agentic 检索涉及 LLM 调用和多轮检索，耗时较长\n建议：使用 RRF/Embedding/BM25 检索模式（更快）"
            raise RuntimeError(error_msg)
        except httpx.ConnectError as e:
            error_msg = f"连接失败: 无法连接到 {self.api_base_url}\n请确保 V3 API 服务已启动: uv run python src/bootstrap.py src/run.py --port 8001"
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Agentic 检索失败: {type(e).__name__}: {e}"
            raise RuntimeError(error_msg)
    
    def build_prompt(self, user_query: str, memories: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """构建 Prompt
        
        Args:
            user_query: 用户查询
            memories: 检索到的记忆列表
            
        Returns:
            Chat Messages 列表
        """
        messages = []
        
        # System Message
        lang_key = "zh" if self.texts.language == "zh" else "en"
        system_content = self.texts.get(f"prompt_system_role_{lang_key}")
        messages.append({"role": "system", "content": system_content})
        
        # Retrieved Memories
        if memories:
            memory_lines = []
            for i, mem in enumerate(memories, start=1):
                timestamp = mem.get("timestamp", "")[:10]
                subject = mem.get("subject", "")
                summary = mem.get("summary", "")
                episode = mem.get("episode", "")
                
                parts = [f"[{i}] {self.texts.get('prompt_memory_date', date=timestamp)}"]
                if subject:
                    parts.append(self.texts.get("prompt_memory_subject", subject=subject))
                if summary:
                    parts.append(self.texts.get("prompt_memory_content", content=summary))
                if episode:
                    parts.append(self.texts.get("prompt_memory_episode", episode=episode))
                
                memory_lines.append(" | ".join(parts))
            
            memory_content = self.texts.get("prompt_memories_prefix") + "\n".join(memory_lines)
            messages.append({"role": "system", "content": memory_content})
        
        # Conversation History
        for user_q, assistant_a in self.conversation_history[-self.config.conversation_history_size:]:
            messages.append({"role": "user", "content": user_q})
            messages.append({"role": "assistant", "content": assistant_a})
        
        # Current Question
        messages.append({"role": "user", "content": user_query})
        
        return messages
    
    async def chat(self, user_input: str) -> str:
        """核心对话逻辑
        
        Args:
            user_input: 用户输入
            
        Returns:
            助手回答
        """
        from .ui import ChatUI
        
        # 检索记忆
        memories = await self.retrieve_memories(user_input)
        
        # 显示检索结果
        if self.config.show_retrieved_memories and memories:
            ChatUI.print_retrieved_memories(
                memories[:5],
                texts=self.texts,
                retrieval_metadata=self.last_retrieval_metadata,
            )
        
        # 构建 Prompt
        messages = self.build_prompt(user_input, memories)
        
        # 显示生成进度
        ChatUI.print_generating_indicator(self.texts)
        
        # 调用 LLM
        try:
            if hasattr(self.llm_provider, 'provider') and hasattr(
                self.llm_provider.provider, 'chat_with_messages'
            ):
                raw_response = await self.llm_provider.provider.chat_with_messages(messages)
            else:
                prompt_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                
                prompt = "\n\n".join(prompt_parts)
                raw_response = await self.llm_provider.generate(prompt)
            
            raw_response = raw_response.strip()
            
            # 清除生成进度
            ChatUI.print_generation_complete(self.texts)
            
            assistant_response = raw_response
        
        except Exception as e:
            ChatUI.clear_progress_indicator()
            error_msg = f"[{self.texts.get('error_label')}] {self.texts.get('chat_llm_error', error=str(e))}"
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg
        
        # 更新对话历史
        self.conversation_history.append((user_input, assistant_response))
        
        if len(self.conversation_history) > self.config.conversation_history_size:
            self.conversation_history = self.conversation_history[-self.config.conversation_history_size:]
        
        return assistant_response
    
    def clear_history(self) -> None:
        """清空对话历史"""
        from .ui import ChatUI
        count = len(self.conversation_history)
        self.conversation_history = []
        ChatUI.print_info(self.texts.get("cmd_clear_done", count=count), self.texts)
    
    async def reload_data(self) -> None:
        """重新加载记忆数据"""
        from .ui import ChatUI
        from common_utils.cli_ui import CLIUI
        
        display_name = "group_chat" if self.group_id == "AI产品群" else self.group_id
        
        ui = CLIUI()
        print()
        ui.note(self.texts.get("cmd_reload_refreshing", name=display_name), icon="🔄")
        
        # 重新统计 MemCell 数量
        now = get_now_with_timezone()
        start_date = now - timedelta(days=self.config.time_range_days)
        memcells = await query_memcells_by_group_and_time(self.group_id, start_date, now)
        self.memcell_count = len(memcells)
        
        print()
        ui.success(f"✓ {self.texts.get('cmd_reload_complete', users=0, memories=self.memcell_count)}")
        print()

