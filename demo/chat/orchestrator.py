"""聊天应用编排器

负责整个聊天应用的流程编排：
1. 初始化配置
2. 用户交互（语言、场景、群组、检索模式选择）
3. 会话管理
4. 对话循环
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from demo.config import ChatModeConfig, LLMConfig, MongoDBConfig
from demo.utils import ensure_mongo_beanie_ready
from demo.ui import I18nTexts
from common_utils.cli_ui import CLIUI

from .session import ChatSession
from .ui import ChatUI
from .selectors import LanguageSelector, ScenarioSelector, GroupSelector


class ChatOrchestrator:
    """聊天应用编排器"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "demo" / ".chat_history"
        self._configure_logging()
    
    def _configure_logging(self):
        """配置日志 - 隐藏第三方库的 DEBUG 日志"""
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
        logging.getLogger().setLevel(logging.WARNING)
        
        # 禁用常见第三方库日志
        for logger_name in ['jieba', 'elasticsearch', 'urllib3', 'pymongo', 'pymilvus']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    def setup_readline(self):
        """配置 readline 历史记录"""
        try:
            import readline
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
            readline.set_history_length(1000)
        except Exception:
            pass
    
    def save_readline_history(self):
        """保存 readline 历史记录"""
        try:
            import readline
            readline.write_history_file(str(self.history_file))
        except Exception:
            pass
    
    async def select_language(self) -> I18nTexts:
        """语言选择"""
        language = LanguageSelector.select_language()
        return I18nTexts(language)
    
    async def select_scenario(self, texts: I18nTexts) -> Optional[str]:
        """场景选择"""
        ChatUI.clear_screen()
        ChatUI.print_banner(texts)
        
        scenario_type = ScenarioSelector.select_scenario(texts)
        if not scenario_type:
            ChatUI.print_info(texts.get("groups_not_selected_exit"), texts)
            return None
        
        return scenario_type
    
    async def initialize_database(self, texts: I18nTexts) -> bool:
        """初始化数据库连接"""
        mongo_config = MongoDBConfig()
        print(mongo_config) 
        ui = CLIUI()
        ui.note(texts.get("mongodb_connecting"), icon="🔌")
        
        try:
            await ensure_mongo_beanie_ready(mongo_config)
            # 清除连接提示
            print("\r\033[K", end="")
            print("\033[A\033[K", end="")
            return True
        except Exception as e:
            ChatUI.print_error(texts.get("mongodb_init_failed", error=str(e)), texts)
            return False
    
    async def select_group(self, texts: I18nTexts) -> Optional[str]:
        """群组选择"""
        groups = await GroupSelector.list_available_groups()
        selected_group_id = await GroupSelector.select_group(groups, texts)
        
        if not selected_group_id:
            ChatUI.print_info(texts.get("groups_not_selected_exit"), texts)
            return None
        
        return selected_group_id
    
    async def select_retrieval_mode(self) -> str:
        """检索模式选择"""
        ui = CLIUI()
        print()
        ui.section_heading("🔍 检索模式选择")
        print()
        print("  [1] RRF 融合（推荐） - Embedding + BM25 融合")
        print("  [2] 纯向量检索 - 语义理解最强")
        print("  [3] 纯 BM25 检索 - 关键词精确匹配")
        print("  [4] Agentic 检索 - LLM 引导的多轮检索（实验性）")
        print()
        
        mode_map = {1: "rrf", 2: "embedding", 3: "bm25", 4: "agentic"}
        mode_desc = {1: "RRF 融合", 2: "纯向量检索", 3: "纯 BM25", 4: "Agentic 检索"}
        
        while True:
            try:
                choice = input("请选择检索模式 [1-4]: ").strip()
                if not choice:
                    continue
                
                index = int(choice)
                if index in mode_map:
                    # 特殊提示：Agentic 模式需要 LLM
                    if index == 4:
                        print()
                        ui.note("⚠️  Agentic 检索将使用 LLM API，可能产生额外费用", icon="💰")
                        print()
                    
                    ui.success(f"✓ 已选择: {mode_desc[index]}")
                    return mode_map[index]
                else:
                    ui.error("✗ 请输入 1-4")
            except ValueError:
                ui.error("✗ 请输入有效数字")
            except KeyboardInterrupt:
                print("\n")
                raise
    
    def verify_api_key(self, llm_config: LLMConfig, texts: I18nTexts) -> bool:
        """验证 API Key 是否配置"""
        import os
        api_key_present = any([
            llm_config.api_key,
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ])
        
        if not api_key_present:
            ChatUI.print_error(texts.get("config_api_key_missing"), texts)
            print(f"{texts.get('config_api_key_hint')}\n")
            return False
        
        return True
    
    def select_history_mode(self, texts: I18nTexts) -> bool:
        """选择历史记录模式
        
        Args:
            texts: 国际化文本对象
            
        Returns:
            True - 加载历史记录，False - 新建会话
        """
        ui = CLIUI()
        print()
        ui.section_heading(texts.get("history_mode_selection_title"))
        print()
        print(f"  [1] {texts.get('history_mode_load')}")
        print(f"  [2] {texts.get('history_mode_new')}")
        print()
        
        while True:
            try:
                choice = input(f"{texts.get('history_mode_prompt')}: ").strip()
                if choice == "1":
                    ui.success(f"✓ {texts.get('history_mode_load_selected')}")
                    return True
                elif choice == "2":
                    ui.success(f"✓ {texts.get('history_mode_new_selected')}")
                    return False
                else:
                    ui.error(f"✗ {texts.get('invalid_input_number')}")
            except KeyboardInterrupt:
                print("\n")
                raise
    
    async def create_session(
        self,
        group_id: str,
        scenario_type: str,
        retrieval_mode: str,
        texts: I18nTexts,
        load_history: bool = True,
    ) -> Optional[ChatSession]:
        """创建并初始化会话"""
        chat_config = ChatModeConfig()
        llm_config = LLMConfig()
        
        session = ChatSession(
            group_id=group_id,
            config=chat_config,
            llm_config=llm_config,
            scenario_type=scenario_type,
            retrieval_mode=retrieval_mode,
            data_source="episode",  # 固定使用 episode
            texts=texts,
        )
        
        if not await session.initialize(load_history=load_history):
            ChatUI.print_error(texts.get("session_init_failed"), texts)
            return None
        
        return session
    
    async def run_chat_loop(self, session: ChatSession, texts: I18nTexts):
        """运行对话循环"""
        # 刷新屏幕，进入干净的对话界面
        ChatUI.clear_screen()
        ChatUI.print_banner(texts)
        
        # 显示开始提示
        ui = CLIUI()
        print()
        ui.rule()
        ui.note(texts.get("chat_start_note"), icon="💬")
        ui.rule()
        print()
        
        while True:
            try:
                user_input = input(texts.get("chat_input_prompt")).strip()
                
                if not user_input:
                    continue
                
                command = user_input.lower()
                
                # 处理命令
                if command == "exit":
                    await self._handle_exit(session, texts)
                    break
                elif command == "clear":
                    session.clear_history()
                    continue
                elif command == "reload":
                    await session.reload_data()
                    continue
                elif command == "help":
                    ChatUI.print_help(texts)
                    continue
                
                # 执行对话
                response = await session.chat(user_input)
                ChatUI.print_assistant_response(response, texts)
            
            except KeyboardInterrupt:
                await self._handle_interrupt(session, texts)
                break
            
            except Exception as e:
                ChatUI.print_error(texts.get("chat_error", error=str(e)), texts)
                import traceback
                traceback.print_exc()
                print()
    
    async def _handle_exit(self, session: ChatSession, texts: I18nTexts):
        """处理退出命令"""
        ui = CLIUI()
        print()
        ui.note(texts.get("cmd_exit_saving"), icon="💾")
        await session.save_conversation_history()
        print()
        ui.success(f"✓ {texts.get('cmd_exit_complete')}")
        print()
    
    async def _handle_interrupt(self, session: ChatSession, texts: I18nTexts):
        """处理中断信号"""
        ui = CLIUI()
        print("\n")
        ui.note(texts.get("cmd_interrupt_saving"), icon="⚠️")
        await session.save_conversation_history()
        print()
        ui.success(f"✓ {texts.get('cmd_exit_complete')}")
        print()
    
    async def run(self):
        """运行聊天应用主流程"""
        # 1. 初始化 readline
        self.setup_readline()
        
        # 2. 语言选择
        texts = await self.select_language()
        
        # 3. 场景选择
        scenario_type = await self.select_scenario(texts)
        if not scenario_type:
            return
        
        # 4. 刷新屏幕
        ChatUI.clear_screen()
        ChatUI.print_banner(texts)
        
        # 5. 验证 API Key
        llm_config = LLMConfig()
        if not self.verify_api_key(llm_config, texts):
            return
        
        # 6. 初始化数据库
        if not await self.initialize_database(texts):
            return
        
        # 7. 群组选择
        group_id = await self.select_group(texts)
        if not group_id:
            return
        
        # 8. 检索模式选择
        try:
            retrieval_mode = await self.select_retrieval_mode()
        except KeyboardInterrupt:
            print("\n")
            return
        
        # 9. 历史记录模式选择
        try:
            load_history = self.select_history_mode(texts)
        except KeyboardInterrupt:
            print("\n")
            return
        
        # 10. 创建会话
        session = await self.create_session(group_id, scenario_type, retrieval_mode, texts, load_history)
        if not session:
            return
        
        # 11. 运行对话循环
        await self.run_chat_loop(session, texts)
        
        # 12. 保存历史
        self.save_readline_history()

