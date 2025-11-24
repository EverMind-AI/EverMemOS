"""
Memory Layer 重构后的完整测试

测试流程：
1. 加载测试数据（assistant_chat_zh.json）
2. 提取 MemCell（边界检测）
3. 提取群组 Episode
4. 提取个人 Episode（为每个用户）
5. 提取 Semantic Memory（基于 Episode）
6. 提取 Event Log（基于 Episode）
7. 验证每个步骤的输入输出

环境变量从 .env 加载
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(project_root))

from memory_layer.memory_manager import MemoryManager
from memory_layer.types import MemoryType, RawDataType, MemCell, Memory
from memory_layer.memcell_extractor.base_memcell_extractor import RawData


# ============ LLM 输入输出记录器 ============
class LLMLogger:
    """记录所有 LLM 调用的输入和输出"""
    
    def __init__(self, log_dir="test_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.call_count = 0
        
        # 创建本次测试的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"llm_calls_{timestamp}.jsonl"
        self.summary_file = self.log_dir / f"llm_summary_{timestamp}.txt"
        
        print(f"📝 LLM 调用日志将保存到: {self.log_file}")
        print(f"📝 LLM 调用摘要将保存到: {self.summary_file}")
    
    def log_call(self, stage: str, prompt: str, response: str, metadata: dict = None):
        """记录一次 LLM 调用"""
        self.call_count += 1
        
        log_entry = {
            "call_id": self.call_count,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
        }
        
        # 追加到 JSONL 文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # 打印摘要
        print(f"\n{'='*80}")
        print(f"🤖 LLM 调用 #{self.call_count} - {stage}")
        print(f"{'='*80}")
        print(f"📥 输入 (prompt 前200字):\n{prompt[:200]}...")
        print(f"\n📤 输出 (response 前200字):\n{response[:200]}...")
        print(f"{'='*80}\n")
    
    def write_summary(self):
        """写入调用摘要"""
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write(f"LLM 调用总结\n")
            f.write(f"{'='*80}\n")
            f.write(f"总调用次数: {self.call_count}\n\n")
            
            # 读取所有调用记录
            if self.log_file.exists():
                with open(self.log_file, 'r', encoding='utf-8') as log_f:
                    for line in log_f:
                        entry = json.loads(line)
                        f.write(f"\n{'='*80}\n")
                        f.write(f"调用 #{entry['call_id']} - {entry['stage']}\n")
                        f.write(f"时间: {entry['timestamp']}\n")
                        f.write(f"{'-'*80}\n")
                        f.write(f"输入:\n{entry['prompt']}\n")
                        f.write(f"{'-'*80}\n")
                        f.write(f"输出:\n{entry['response']}\n")
                        f.write(f"{'='*80}\n")
        
        print(f"\n✅ LLM 调用摘要已保存到: {self.summary_file}")


# 全局 LLM Logger
llm_logger = LLMLogger()


# ============ 猴子补丁：拦截 LLM 调用 ============
def patch_llm_provider():
    """给 LLMProvider 打补丁，记录所有调用"""
    from memory_layer.llm.llm_provider import LLMProvider
    import traceback
    import re
    
    original_generate = LLMProvider.generate
    
    async def logged_generate(self, prompt: str, **kwargs):
        """带日志的 generate"""
        # 获取调用栈，识别是哪个阶段
        stack = traceback.extract_stack()
        stage = "unknown"
        
        # 从调用栈中识别阶段
        for frame in reversed(stack):
            filename = frame.filename
            function = frame.name
            
            # 边界检测
            if 'conv_memcell_extractor' in filename and 'extract_memcell' in function:
                stage = "边界检测 (MemCell)"
                break
            # Episode 提取
            elif 'episode_memory_extractor' in filename:
                if '_extract_episode' in function or 'extract_group_episode' in function or 'extract_personal_episode' in function:
                    # 从调用栈中判断是群组还是个人
                    if 'extract_group_episode' in function:
                        stage = "Episode提取 (群组)"
                    elif 'extract_personal_episode' in function:
                        stage = "Episode提取 (个人)"
                    else:
                        stage = "Episode提取"
                    break
            # Semantic 提取
            elif 'semantic_memory_extractor' in filename:
                stage = "语义记忆提取 (Semantic)"
                break
            # Event Log 提取
            elif 'event_log_extractor' in filename:
                stage = "事件日志提取 (EventLog)"
                break
            # Profile 提取
            elif 'profile_memory_extractor' in filename:
                stage = "个人档案提取 (Profile)"
                break
            elif 'group_profile_memory_extractor' in filename:
                stage = "群组档案提取 (GroupProfile)"
                break
        
        # 调用原始方法
        response = await original_generate(self, prompt, **kwargs)
        
        # 记录日志
        llm_logger.log_call(
            stage=stage,
            prompt=prompt,
            response=response,
            metadata={
                "model": self.provider.model if hasattr(self, 'provider') else 'unknown',
                "temperature": kwargs.get('temperature') or (self.provider.temperature if hasattr(self, 'provider') else None),
            }
        )
        
        return response
    
    # 替换方法
    LLMProvider.generate = logged_generate
    print("✅ LLM Provider 已打补丁，所有调用将被记录\n")


# 应用补丁
patch_llm_provider()


class TestMemoryLayerRefactored:
    """Memory Layer 重构后的完整测试"""
    
    def __init__(self):
        self.test_data_path = Path("/Users/admin/Documents/Projects/opensource/memsys-opensource/data/assistant_chat_zh.json")
        self.test_data = self.load_test_data()
        self.memory_manager = MemoryManager()
        self.raw_data_list = self.convert_to_raw_data()
        self.group_info = self.get_group_info()
        
        # 模拟对话历史缓存（类似 Redis 中的历史）
        self.conversation_history = []
    
    def load_test_data(self):
        """加载测试数据"""
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def convert_to_raw_data(self):
        """转换测试数据为 RawData 列表"""
        conversation_list = self.test_data.get("conversation_list", [])
        raw_data_list = []
        
        for msg in conversation_list:
            raw_data = RawData(
                data_id=msg.get("message_id"),
                content={
                    "speaker_id": msg.get("sender"),
                    "speaker_name": msg.get("sender_name"),
                    "content": msg.get("content"),
                    "timestamp": msg.get("create_time"),
                },
                data_type="conversation",
                metadata={
                    "timestamp": datetime.fromisoformat(msg.get("create_time").replace('+08:00', '+00:00'))
                }
            )
            raw_data_list.append(raw_data)
        
        return raw_data_list
    
    def get_group_info(self):
        """获取群组信息"""
        meta = self.test_data.get("conversation_meta", {})
        user_details = meta.get("user_details", {})
        
        return {
            "group_id": meta.get("group_id"),
            "group_name": meta.get("name"),
            "user_ids": list(user_details.keys()),
            "user_details": user_details,
        }
    
    async def test_01_extract_memcell(self):
        """
        测试1：提取 MemCell（模拟完整的对话历史管理流程）
        
        流程：
        1. 模拟分批发送消息（每次1条）
        2. 每次都用历史+新消息做边界检测
        3. 如果到边界，提取 MemCell 并清空历史
        4. 如果未到边界，将新消息追加到历史继续累积
        """
        print("\n" + "="*80)
        print("测试1：提取 MemCell（边界检测 + 对话历史管理）")
        print("="*80)
        
        memcell = None
        total_messages = len(self.raw_data_list)
        
        for i in range(total_messages):
            new_message = [self.raw_data_list[i]]  # 每次发送1条消息
            
            print(f"\n--- 消息 {i+1}/{total_messages} ---")
            print(f"   当前历史: {len(self.conversation_history)} 条")
            print(f"   发送者: {new_message[0].content.get('speaker_name')}")
            print(f"   内容: {new_message[0].content.get('content')[:50]}...")
            
            # 边界检测（使用历史 + 新消息）
            result = await self.memory_manager.extract_memcell(
                history_raw_data_list=self.conversation_history,
                new_raw_data_list=new_message,
                raw_data_type=RawDataType.CONVERSATION,
                group_id=self.group_info["group_id"],
                group_name=self.group_info["group_name"],
                user_id_list=self.group_info["user_ids"],
            )
            
            extracted_memcell, status_result = result
            
            if extracted_memcell:
                # 到边界了！
                print(f"   ✅ 检测到边界！提取 MemCell")
                print(f"      - event_id: {extracted_memcell.event_id}")
                print(f"      - 包含消息数: {len(extracted_memcell.original_data)}")
                
                memcell = extracted_memcell
                
                # 清空历史（模拟 mem_memorize.py 的行为）
                print(f"   🗑️  清空对话历史，重新开始累积")
                self.conversation_history = []
                
                # 找到第一个 MemCell 就停止
                break
            else:
                # 未到边界
                if status_result and status_result.should_wait:
                    print(f"   ⏳ 无法判断边界，继续等待")
                else:
                    print(f"   ⏸️  非边界，继续累积")
                
                # 将新消息追加到历史
                self.conversation_history.extend(new_message)
                print(f"   📝 追加到历史，当前历史: {len(self.conversation_history)} 条")
        
        # 验证结果
        assert memcell is not None, "应该至少提取到一个 MemCell"
        assert isinstance(memcell, MemCell), f"应该返回 MemCell，实际类型: {type(memcell)}"
        assert memcell.event_id is not None, "event_id 不应为 None"
        assert memcell.original_data is not None, "original_data 不应为 None"
        assert len(memcell.original_data) > 0, "original_data 应该包含数据"
        
        print(f"\n✅ MemCell 提取完成:")
        print(f"   - event_id: {memcell.event_id}")
        print(f"   - 包含消息数: {len(memcell.original_data)}")
        print(f"   - timestamp: {memcell.timestamp}")
        print(f"   - group_id: {memcell.group_id}")
        
        return memcell
    
    async def test_02_extract_group_episode(self, memcell):
        """测试2：提取群组 Episode"""
        print("\n" + "="*80)
        print("测试2：提取群组 Episode")
        print("="*80)
        
        # 提取群组 Episode
        group_episode = await self.memory_manager.extract_memory(
            memcell=memcell,
            memory_type=MemoryType.EPISODIC_MEMORY,
            user_id=None,  # None 表示群组
            group_id=self.group_info["group_id"],
        )
        
        # 验证结果
        assert group_episode is not None, "群组 Episode 不应为 None"
        assert isinstance(group_episode, Memory), f"应该返回 Memory，实际类型: {type(group_episode)}"
        assert group_episode.user_id is None, "群组 Episode 的 user_id 应该为 None"
        assert group_episode.episode is not None, "episode 内容不应为 None"
        assert group_episode.subject is not None, "subject 不应为 None"
        assert group_episode.memory_type == MemoryType.EPISODIC_MEMORY, "memory_type 应该是 EPISODIC_MEMORY"
        
        # 验证 embedding
        assert hasattr(group_episode, 'extend'), "应该有 extend 字段"
        if group_episode.extend:
            assert 'embedding' in group_episode.extend, "extend 应该包含 embedding"
            assert 'vector_model' in group_episode.extend, "extend 应该包含 vector_model"
        
        print(f"✅ 群组 Episode 提取成功:")
        print(f"   - subject: {group_episode.subject}")
        print(f"   - user_id: {group_episode.user_id} (应该为 None)")
        print(f"   - episode 长度: {len(group_episode.episode)}")
        print(f"   - summary: {group_episode.summary[:100]}...")
        print(f"   - 包含 embedding: {bool(group_episode.extend and 'embedding' in group_episode.extend)}")
        
        return group_episode
    
    async def test_03_extract_personal_episodes(self, memcell):
        """测试3：提取个人 Episode（为每个用户）- 并发提取"""
        print("\n" + "="*80)
        print("测试3：提取个人 Episode（并发）")
        print("="*80)
        
        # 并发提取所有用户的个人 Episode
        tasks = []
        for user_id in self.group_info["user_ids"]:
            user_name = self.group_info["user_details"].get(user_id, {}).get("full_name", user_id)
            print(f"准备提取用户 {user_name} ({user_id}) 的个人 Episode...")
            
            task = self.memory_manager.extract_memory(
                memcell=memcell,
                memory_type=MemoryType.EPISODIC_MEMORY,
                user_id=user_id,  # 有值表示个人
                group_id=self.group_info["group_id"],
            )
            tasks.append((user_id, user_name, task))
        
        # 并发执行所有任务
        print(f"\n开始并发提取 {len(tasks)} 个用户的个人 Episode...")
        results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
        
        # 处理结果
        personal_episodes = []
        for i, (user_id, user_name, _) in enumerate(tasks):
            result = results[i]
            
            if isinstance(result, Exception):
                print(f"   ❌ 用户 {user_name} ({user_id}) 提取失败: {result}")
            elif result:
                assert isinstance(result, Memory), f"应该返回 Memory，实际类型: {type(result)}"
                assert result.user_id == user_id, f"user_id 应该是 {user_id}"
                assert result.episode is not None, "episode 内容不应为 None"
                assert result.subject is not None, "subject 不应为 None"
                
                print(f"   ✅ 用户 {user_name} ({user_id}) Episode 提取成功:")
                print(f"      - subject: {result.subject}")
                print(f"      - episode 长度: {len(result.episode)}")
                
                personal_episodes.append(result)
            else:
                print(f"   ⚠️  用户 {user_id} 的 Episode 提取返回 None")
        
        assert len(personal_episodes) > 0, "至少应该提取到一个个人 Episode"
        
        print(f"\n✅ 并发提取完成: 成功 {len(personal_episodes)}/{len(tasks)} 个")
        
        return personal_episodes
    
    async def test_04_extract_semantic_memory(self, memcell, episode):
        """测试4：提取 Semantic Memory（基于 Episode）"""
        print("\n" + "="*80)
        print("测试4：提取 Semantic Memory")
        print("="*80)
        
        # 提取 Semantic Memory
        semantic_memories = await self.memory_manager.extract_memory(
            memcell=memcell,
            memory_type=MemoryType.SEMANTIC_MEMORY,
            episode_memory=episode,
        )
        
        # 验证结果
        assert semantic_memories is not None, "Semantic memories 不应为 None"
        assert isinstance(semantic_memories, list), f"应该返回 list，实际类型: {type(semantic_memories)}"
        
        if len(semantic_memories) > 0:
            print(f"✅ Semantic Memory 提取成功:")
            print(f"   - 提取了 {len(semantic_memories)} 条语义记忆")
            
            for i, sem in enumerate(semantic_memories[:3], 1):
                print(f"   {i}. {sem.content}")
                assert sem.embedding is not None, f"第{i}条 semantic memory 应该有 embedding"
                assert sem.content is not None, f"第{i}条 semantic memory 应该有 content"
        else:
            print("⚠️  没有提取到 Semantic Memory")
        
        return semantic_memories
    
    async def test_05_extract_event_log(self, memcell, episode):
        """测试5：提取 Event Log（基于 Episode）"""
        print("\n" + "="*80)
        print("测试5：提取 Event Log")
        print("="*80)
        
        # 提取 Event Log
        event_log = await self.memory_manager.extract_memory(
            memcell=memcell,
            memory_type=MemoryType.PERSONAL_EVENT_LOG,
            episode_memory=episode,
        )
        
        # 验证结果
        if event_log:
            assert event_log.time is not None, "time 不应为 None"
            assert event_log.atomic_fact is not None, "atomic_fact 不应为 None"
            assert isinstance(event_log.atomic_fact, list), "atomic_fact 应该是 list"
            assert len(event_log.atomic_fact) > 0, "atomic_fact 不应该为空"
            
            # 验证 embedding
            assert event_log.fact_embeddings is not None, "fact_embeddings 不应为 None"
            assert len(event_log.fact_embeddings) == len(event_log.atomic_fact), "embedding 数量应该和 fact 数量一致"
            
            print(f"✅ Event Log 提取成功:")
            print(f"   - time: {event_log.time}")
            print(f"   - 包含 {len(event_log.atomic_fact)} 个原子事实")
            
            for i, fact in enumerate(event_log.atomic_fact[:3], 1):
                print(f"   {i}. {fact}")
        else:
            print("⚠️  Event Log 提取失败")
        
        return event_log
    
    async def run_all_tests(self):
        """运行所有测试"""
        try:
            # 测试1: 提取 MemCell
            memcell = await self.test_01_extract_memcell()
            
            # 测试2&3: 并发提取群组 Episode 和所有个人 Episode
            print("\n" + "="*80)
            print("测试2&3：并发提取群组 + 个人 Episode")
            print("="*80)
            
            # 准备所有任务
            tasks = []
            
            # 任务1: 群组 Episode
            print("准备提取群组 Episode...")
            group_task = self.memory_manager.extract_memory(
                memcell=memcell,
                memory_type=MemoryType.EPISODIC_MEMORY,
                user_id=None,  # None 表示群组
                group_id=self.group_info["group_id"],
            )
            tasks.append(("group", None, group_task))
            
            # 任务2-N: 个人 Episode
            for user_id in self.group_info["user_ids"]:
                user_name = self.group_info["user_details"].get(user_id, {}).get("full_name", user_id)
                print(f"准备提取用户 {user_name} ({user_id}) 的个人 Episode...")
                
                personal_task = self.memory_manager.extract_memory(
                    memcell=memcell,
                    memory_type=MemoryType.EPISODIC_MEMORY,
                    user_id=user_id,
                    group_id=self.group_info["group_id"],
                )
                tasks.append(("personal", user_id, personal_task))
            
            # 并发执行所有 Episode 提取
            print(f"\n开始并发提取 {len(tasks)} 个 Episode (1个群组 + {len(tasks)-1}个个人)...")
            results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
            
            # 处理结果
            group_episode = None
            personal_episodes = []
            
            for i, (task_type, user_id, _) in enumerate(tasks):
                result = results[i]
                
                if isinstance(result, Exception):
                    print(f"   ❌ {'群组' if task_type == 'group' else f'用户 {user_id}'} Episode 提取失败: {result}")
                elif result:
                    if task_type == "group":
                        # 验证群组 Episode
                        assert isinstance(result, Memory), f"应该返回 Memory，实际类型: {type(result)}"
                        assert result.user_id is None, "群组 Episode 的 user_id 应该为 None"
                        assert result.episode is not None, "episode 内容不应为 None"
                        
                        group_episode = result
                        print(f"   ✅ 群组 Episode 提取成功:")
                        print(f"      - subject: {result.subject}")
                        print(f"      - episode 长度: {len(result.episode)}")
                    else:
                        # 验证个人 Episode
                        assert isinstance(result, Memory), f"应该返回 Memory，实际类型: {type(result)}"
                        assert result.user_id == user_id, f"user_id 应该是 {user_id}"
                        assert result.episode is not None, "episode 内容不应为 None"
                        
                        user_name = self.group_info["user_details"].get(user_id, {}).get("full_name", user_id)
                        print(f"   ✅ 用户 {user_name} ({user_id}) Episode 提取成功:")
                        print(f"      - subject: {result.subject}")
                        print(f"      - episode 长度: {len(result.episode)}")
                        
                        personal_episodes.append(result)
            
            assert group_episode is not None, "群组 Episode 不应为 None"
            assert len(personal_episodes) > 0, "至少应该提取到一个个人 Episode"
            
            print(f"\n✅ 并发提取完成: 群组✓, 个人 {len(personal_episodes)}/{len(tasks)-1}")
            
            # 测试4: 提取 Semantic Memory（使用第一个个人 Episode）
            if personal_episodes:
                semantic_memories = await self.test_04_extract_semantic_memory(memcell, personal_episodes[0])
            
            # 测试5: 提取 Event Log（使用第一个个人 Episode）
            if personal_episodes:
                event_log = await self.test_05_extract_event_log(memcell, personal_episodes[0])
            
            # 写入 LLM 调用摘要
            llm_logger.write_summary()
            
            # 总结
            print("\n" + "="*80)
            print("✅ 所有测试通过！")
            print("="*80)
            print(f"总结:")
            print(f"  - MemCell: ✓")
            print(f"  - 群组 Episode: ✓")
            print(f"  - 个人 Episode: ✓ ({len(personal_episodes)} 个)")
            if personal_episodes:
                print(f"  - Semantic Memory: ✓ ({len(semantic_memories) if semantic_memories else 0} 条)")
                print(f"  - Event Log: {'✓' if event_log else '✗'}")
            print(f"  - LLM 调用总数: {llm_logger.call_count}")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """主函数"""
    print("="*80)
    print("Memory Layer 重构后的完整测试")
    print("="*80)
    print(f"测试数据: assistant_chat_zh.json")
    print(f"环境变量: 从 .env 加载")
    print("="*80)
    
    tester = TestMemoryLayerRefactored()
    success = await tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())


