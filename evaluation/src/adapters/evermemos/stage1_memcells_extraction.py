from typing import Dict, List
import json
import os
import sys
import uuid
import asyncio
import time
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console


from common_utils.datetime_utils import (
    to_iso_format,
    from_iso_format,
    from_timestamp,
    get_now_with_timezone,
)
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memcell_extractor.base_memcell_extractor import RawData, MemCell
from memory_layer.memcell_extractor.conv_memcell_extractor import (
    ConvMemCellExtractor,
    ConversationMemCellExtractRequest,
)
from memory_layer.memory_extractor.episode_memory_extractor import (
    EpisodeMemoryExtractRequest,
    EpisodeMemoryExtractor,
)
from memory_layer.memory_extractor.event_log_extractor import EventLogExtractor
from memory_layer.types import RawDataType

# 新增：聚类和 Profile 管理组件
from memory_layer.cluster_manager import (
    ClusterManager,
    ClusterManagerConfig,
    InMemoryClusterStorage,
)
from memory_layer.profile_manager import (
    ProfileManager,
    ProfileManagerConfig,
    ScenarioType,
    InMemoryProfileStorage,
)

from evaluation.src.adapters.evermemos.config import ExperimentConfig
from datetime import datetime, timedelta
from pathlib import Path


def parse_locomo_timestamp(timestamp_str: str) -> datetime:
    """Parse LoComo timestamp format to datetime object."""
    timestamp_str = timestamp_str.replace("\\s+", " ").strip()
    dt = datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
    return dt


def raw_data_load(locomo_data_path: str) -> Dict[str, List[RawData]]:
    with open(locomo_data_path, "r") as f:
        data = json.load(f)

    # data = [data[2]]
    # data = [data[0], data[1], data[2]]
    raw_data_dict = {}

    conversations = [data[i]['conversation'] for i in range(len(data))]
    print(f"   📅 Found {len(conversations)} conversations")
    for con_id, conversation in enumerate(conversations):
        messages = []
        # print(conversation.keys())
        session_keys = sorted(
            [
                key
                for key in conversation
                if key.startswith("session_") and not key.endswith("_date_time")
            ],
            key=lambda x: int(x.replace("session_", ""))
        )

        print(f"   📅 Found {len(session_keys)} sessions")
        print(
            f"   🎭 Speakers: {conversation.get('speaker_a', 'Unknown')} & {conversation.get('speaker_b', 'Unknown')}"
        )
        speaker_name_to_id = {}
        for session_key in session_keys:
            session_messages = conversation[session_key]
            session_time_key = f"{session_key}_date_time"

            if session_time_key in conversation:
                # Parse session timestamp
                session_time = parse_locomo_timestamp(conversation[session_time_key])

                # Process each message in this session
                for i, msg in enumerate(session_messages):
                    # Generate timestamp for this message (session time + message offset)
                    msg_timestamp = session_time + timedelta(
                        seconds=i * 30
                    )  # 30 seconds between messages
                    iso_timestamp = to_iso_format(msg_timestamp)

                    # Generate unique speaker_id for this conversation
                    speaker_name = msg["speaker"]
                    if speaker_name not in speaker_name_to_id:
                        # Generate unique ID: {name}_{conversation_index}
                        unique_id = f"{speaker_name.lower().replace(' ', '_')}_{con_id}"
                        speaker_name_to_id[speaker_name] = unique_id

                    # Process content with image information if present
                    content = msg["text"]
                    if msg.get("img_url"):
                        blip_caption = msg.get("blip_caption", "an image")
                        content = f"[{speaker_name} shared an image: {blip_caption}] {content}"

                    message = {
                        "speaker_id": speaker_name_to_id[speaker_name],
                        "user_name": speaker_name,
                        "speaker_name": speaker_name,
                        "content": content,
                        "timestamp": iso_timestamp,
                        "original_timestamp": conversation[session_time_key],
                        "dia_id": msg["dia_id"],
                        "session": session_key,
                    }
                    # Add optional fields if present
                    for optional_field in ["img_url", "blip_caption", "query"]:
                        if optional_field in msg:
                            message[optional_field] = msg[optional_field]
                    messages.append(message)
            # messages = messages[:30]
        raw_data_dict[str(con_id)] = messages

        print(
            f"   ✅ Converted {len(messages)} messages from {len(session_keys)} sessions"
        )

    return raw_data_dict


def convert_conversation_to_raw_data_list(conversation: list) -> List[RawData]:
    raw_data_list = []
    for msg in conversation:
        raw_data_list.append(RawData(content=msg, data_id=str(uuid.uuid4())))
    return raw_data_list


async def memcell_extraction_from_conversation(
    raw_data_list: List[RawData],
    llm_provider: LLMProvider = None,
    memcell_extractor: ConvMemCellExtractor = None,
    smart_mask: bool = True,
    conv_id: str = None,  # 添加会话ID用于进度条描述
    progress: Progress = None,  # 添加进度条对象
    task_id: int = None,  # 添加任务ID
    use_semantic_extraction: bool = False,  # 新增：是否启用语义记忆提取
) -> list:

    episode_extractor = EpisodeMemoryExtractor(llm_provider=llm_provider, use_eval_prompts=True)
    memcell_list = []
    speakers = {
        raw_data.content["speaker_id"]
        for raw_data in raw_data_list
        if isinstance(raw_data.content, dict) and "speaker_id" in raw_data.content
    }
    history_raw_data_list = []
    # raw_data_list = raw_data_list[:100]

    # 处理消息
    total_messages = len(raw_data_list)
    smart_mask_flag = False

    for idx, raw_data in enumerate(raw_data_list):
        # 更新进度条（在处理前更新，显示正在处理第几条）
        if progress and task_id is not None:
            progress.update(task_id, completed=idx)

        if history_raw_data_list == [] or len(history_raw_data_list) == 1:
            history_raw_data_list.append(raw_data)
            continue

        if smart_mask and len(history_raw_data_list) > 5:
            smart_mask_flag = True
            # analysis_history = history_raw_data_list[:-1]
        else:
            # analysis_history = history_raw_data_list
            smart_mask_flag = False
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=history_raw_data_list,
            new_raw_data_list=[raw_data],
            user_id_list=list(speakers),
            smart_mask_flag=smart_mask_flag,
            # group_id="group_1",
        )
        for i in range(5):
            try:
                result = await memcell_extractor.extract_memcell(
                    request,
                    use_semantic_extraction=use_semantic_extraction  # 传递开关
                )
                break
            except Exception as e:
                print('retry: ', i)
                if i == 4:
                    raise Exception("Memcell extraction failed")
                continue
        memcell_result = result[0]
        # print(f"   ✅ Memcell result: {memcell_result}")  # 注释掉避免干扰进度条
        if memcell_result is None:
            history_raw_data_list.append(raw_data)
        elif isinstance(memcell_result, MemCell):
            if smart_mask_flag:
                history_raw_data_list = [history_raw_data_list[-1], raw_data]
            else:
                history_raw_data_list = [raw_data]
            memcell_result.summary = memcell_result.episode[:200] + "..."
            memcell_list.append(memcell_result)
        else:
            console = Console()
            console.print("--------------------------------")
            console.print(f"   ❌ Memcell result: {memcell_result}", style="bold red")
            raise Exception("Memcell extraction failed")

    # 处理完成，更新进度为100%
    if progress and task_id is not None:
        progress.update(task_id, completed=total_messages)

    if history_raw_data_list:
        memcell = MemCell(
            type=RawDataType.CONVERSATION,
            event_id=str(uuid.uuid4()),
            user_id_list=list(speakers),
            original_data=history_raw_data_list,
            timestamp=(memcell_list[-1].timestamp),
            summary="111",
        )
        episode_request = EpisodeMemoryExtractRequest(
            memcell_list=[memcell],
            user_id_list=request.user_id_list,
            participants=list(speakers),
            group_id=request.group_id,
        )

        episode_result = await episode_extractor.extract_memory(
            episode_request, use_group_prompt=True
        )
        memcell.episode = episode_result.episode
        memcell.subject = episode_result.subject
        memcell.summary = episode_result.episode[:200] + "..."
        memcell.original_data = episode_extractor.get_conversation_text(
            history_raw_data_list
        )
        original_data_list = []
        for raw_data in history_raw_data_list:
            original_data_list.append(memcell_extractor._data_process(raw_data))
        memcell.original_data = original_data_list
        memcell_list.append(memcell)

    return memcell_list


async def process_single_conversation(
    conv_id: str,
    conversation: list,
    save_dir: str,
    llm_provider: LLMProvider = None,
    event_log_extractor: EventLogExtractor = None,
    progress_counter: dict = None,
    progress: Progress = None,
    task_id: int = None,
    config: ExperimentConfig = None,  # 新增：传入配置
) -> tuple:
    """处理单个会话并返回结果（新增：聚类和 Profile 提取）

    Args:
        conv_id: 会话ID
        conversation: 会话数据
        save_dir: 保存目录
        llm_provider: 共享的LLM提供者实例
        event_log_extractor: 事件日志提取器实例
        progress: 进度条对象
        task_id: 进度任务ID
        config: 实验配置（用于读取开关）

    Returns:
        tuple: (conv_id, memcell_list)
    """
    try:
        # 更新状态为处理中
        if progress and task_id is not None:
            progress.update(task_id, status="处理中")

        # ===== 根据配置创建组件 =====
        cluster_mgr = None
        profile_mgr = None
        
        # 创建 MemCellExtractor
        raw_data_list = convert_conversation_to_raw_data_list(conversation)
        memcell_extractor = ConvMemCellExtractor(llm_provider=llm_provider, use_eval_prompts=True)
        
        # 条件创建：聚类管理器（每个对话独立）
        if config and config.enable_clustering:
            cluster_storage = InMemoryClusterStorage(
                enable_persistence=True,
                persist_dir=Path(save_dir) / "clusters" / f"conv_{conv_id}"
            )
            cluster_config = ClusterManagerConfig(
                similarity_threshold=config.cluster_similarity_threshold,
                max_time_gap_days=config.cluster_max_time_gap_days,
                enable_persistence=True,
                persist_dir=str(Path(save_dir) / "clusters" / f"conv_{conv_id}"),
                clustering_algorithm="centroid"
            )
            cluster_mgr = ClusterManager(config=cluster_config, storage=cluster_storage)
            cluster_mgr.attach_to_extractor(memcell_extractor)
        
        # 条件创建：Profile 管理器
        if config and config.enable_profile_extraction and cluster_mgr:
            profile_storage = InMemoryProfileStorage(
                enable_persistence=True,
                persist_dir=Path(save_dir) / "profiles" / f"conv_{conv_id}",
                enable_versioning=True
            )
            
            # 动态设置场景类型
            scenario = ScenarioType.ASSISTANT if config.profile_scenario.lower() == "assistant" else ScenarioType.GROUP_CHAT
            
            profile_config = ProfileManagerConfig(
                scenario=scenario,
                min_confidence=config.profile_min_confidence,
                enable_versioning=True,
                auto_extract=True,
                batch_size=50,
            )
            
            profile_mgr = ProfileManager(
                llm_provider=llm_provider,
                config=profile_config,
                storage=profile_storage,
                group_id=f"locomo_conv_{conv_id}",
                group_name=f"LoComo Conversation {conv_id}"
            )
            
            # 设置最小 MemCells 阈值
            profile_mgr._min_memcells_threshold = config.profile_min_memcells
            
            # 连接组件
            profile_mgr.attach_to_cluster_manager(cluster_mgr)
        
        # 提取 MemCells（根据配置决定是否启用语义记忆）
        use_semantic = config.enable_semantic_extraction if config else False
        memcell_list = await memcell_extraction_from_conversation(
            raw_data_list,
            llm_provider=llm_provider,
            memcell_extractor=memcell_extractor,
            conv_id=conv_id,
            progress=progress,
            task_id=task_id,
            use_semantic_extraction=use_semantic,  # 传递语义记忆开关
        )
        # print(f"   ✅ 会话 {conv_id}: {len(memcell_list)} memcells extracted")  # 注释掉避免干扰进度条

        # 在保存前转换时间戳为 datetime 对象
        for memcell in memcell_list:
            if hasattr(memcell, 'timestamp'):
                ts = memcell.timestamp
                if isinstance(ts, (int, float)):
                    # 将 int/float 时间戳转换为带时区的 datetime
                    memcell.timestamp = from_timestamp(ts)
                elif isinstance(ts, str):
                    # 将字符串时间戳转换为带时区的 datetime
                    memcell.timestamp = from_iso_format(ts)
                elif not isinstance(ts, datetime):
                    # 如果不是预期的类型，使用当前时间
                    memcell.timestamp = get_now_with_timezone()

        # 🔥 优化：并发生成 event log（提升速度 10-20 倍）
        if event_log_extractor:
            # 准备所有需要提取 event log 的 memcells
            memcells_with_episode = [
                (idx, memcell) 
                for idx, memcell in enumerate(memcell_list)
                if hasattr(memcell, 'episode') and memcell.episode
            ]
            
            # 定义单个 event log 提取任务
            async def extract_single_event_log(idx: int, memcell):
                try:
                    event_log = await event_log_extractor.extract_event_log(
                        episode_text=memcell.episode, 
                        timestamp=memcell.timestamp
                    )
                    return idx, event_log
                except Exception as e:
                    console = Console()
                    console.print(
                        f"\n⚠️  生成event log失败 (Conv {conv_id}, Memcell {idx}): {e}",
                        style="yellow",
                    )
                    return idx, None
            
            # 🔥 并发提取所有 event logs（使用 Semaphore 控制并发数）
            sem = asyncio.Semaphore(20)  # 限制并发数为 20（避免 API 限流）
            
            async def extract_with_semaphore(idx, memcell):
                async with sem:
                    return await extract_single_event_log(idx, memcell)
            
            print(f"\n🔥 开始并发提取 {len(memcells_with_episode)} 个 event logs...")
            event_log_tasks = [
                extract_with_semaphore(idx, memcell) 
                for idx, memcell in memcells_with_episode
            ]
            event_log_results = await asyncio.gather(*event_log_tasks)
            
            # 将 event logs 关联回对应的 memcells
            for original_idx, event_log in event_log_results:
                if event_log:
                    memcell_list[original_idx].event_log = event_log
            
            print(f"✅ Event log 提取完成: {sum(1 for _, el in event_log_results if el)}/{len(event_log_results)} 成功")

        # 保存单个会话的结果
        memcell_dicts = []
        for memcell in memcell_list:
            memcell_dict = memcell.to_dict()
            # 如果有event_log，添加到字典中
            if hasattr(memcell, 'event_log') and memcell.event_log:
                memcell_dict['event_log'] = memcell.event_log.to_dict()
            memcell_dicts.append(memcell_dict)

        memcell_dicts = [memcell_dict for memcell_dict in memcell_dicts]
        # print(memcell_dicts)  # 注释掉大量输出
        output_file = os.path.join(save_dir, f"memcell_list_conv_{conv_id}.json")
        with open(output_file, "w") as f:
            json.dump(memcell_dicts, f, ensure_ascii=False, indent=2)

        # ===== 条件导出：聚类和 Profile 结果 =====
        cluster_stats = {}
        profile_stats = {}
        profile_count = 0
        
        if cluster_mgr or profile_mgr:
            await asyncio.sleep(2)  # 给异步任务时间完成
        
        # 导出聚类结果（如果启用）
        if cluster_mgr:
            cluster_output_dir = Path(save_dir) / "clusters" / f"conv_{conv_id}"
            cluster_output_dir.mkdir(parents=True, exist_ok=True)
            await cluster_mgr.export_clusters(cluster_output_dir)
            cluster_stats = cluster_mgr.get_stats()
        
        # 导出 Profiles（如果启用）
        if profile_mgr:
            profile_output_dir = Path(save_dir) / "profiles" / f"conv_{conv_id}"
            profile_count = await profile_mgr.export_profiles(profile_output_dir, include_history=True)
            profile_stats = profile_mgr.get_stats()
        
        # 保存统计信息
        stats_output = {
            "conv_id": conv_id,
            "memcells": len(memcell_list),
            "clustering_enabled": config.enable_clustering if config else False,
            "profile_enabled": config.enable_profile_extraction if config else False,
            "semantic_enabled": config.enable_semantic_extraction if config else False,
        }
        
        if cluster_stats:
            stats_output["clustering"] = cluster_stats
        if profile_stats:
            stats_output["profiles"] = profile_stats
            stats_output["profile_count"] = profile_count
        
        stats_file = Path(save_dir) / "stats" / f"conv_{conv_id}_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w") as f:
            json.dump(stats_output, f, ensure_ascii=False, indent=2)

        # 更新进度（静默，避免干扰进度条）
        if progress_counter:
            progress_counter['completed'] += 1
            # 不打印，避免干扰进度条

        return conv_id, memcell_list

    except Exception as e:
        # 显示错误信息，这样我们能知道具体问题
        console = Console()
        console.print(f"\n❌ 处理会话 {conv_id} 时出错: {e}", style="bold red")
        if progress_counter:
            progress_counter['completed'] += 1
            progress_counter['failed'] += 1
        import traceback

        traceback.print_exc()
        return conv_id, []


async def main():
    """主函数 - 并发处理所有会话"""

    config = ExperimentConfig()
    llm_service = config.llm_service
    dataset_path = config.datase_path
    raw_data_dict = raw_data_load(dataset_path)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 🔥 修正：实际文件在 locomo_evaluation/ 目录下，而不是 results/ 目录
    os.makedirs(os.path.join(CURRENT_DIR, config.experiment_name), exist_ok=True)
    os.makedirs(
        os.path.join(CURRENT_DIR, config.experiment_name, "memcells"),
        exist_ok=True,
    )
    save_dir = os.path.join(CURRENT_DIR, config.experiment_name, "memcells")

    console = Console()
    
    # 打印配置信息
    console.print("\n" + "=" * 60, style="bold cyan")
    console.print("实验配置", style="bold cyan")
    console.print("=" * 60, style="bold cyan")
    console.print(f"实验名称: {config.experiment_name}", style="cyan")
    console.print(f"数据路径: {config.datase_path}", style="cyan")
    console.print(f"\n功能开关:", style="bold yellow")
    console.print(f"  - 语义记忆提取: {'✅ 启用' if config.enable_semantic_extraction else '❌ 禁用'}", 
                  style="green" if config.enable_semantic_extraction else "dim")
    console.print(f"  - 聚类: {'✅ 启用' if config.enable_clustering else '❌ 禁用'}", 
                  style="green" if config.enable_clustering else "dim")
    console.print(f"  - Profile 提取: {'✅ 启用' if config.enable_profile_extraction else '❌ 禁用'}", 
                  style="green" if config.enable_profile_extraction else "dim")
    
    if config.enable_clustering:
        console.print(f"\n聚类配置:", style="bold")
        console.print(f"  - 相似度阈值: {config.cluster_similarity_threshold}", style="dim")
        console.print(f"  - 最大时间间隔: {config.cluster_max_time_gap_days} 天", style="dim")
    
    if config.enable_profile_extraction:
        console.print(f"\nProfile 配置:", style="bold")
        console.print(f"  - 场景: {config.profile_scenario}", style="dim")
        console.print(f"  - 最小置信度: {config.profile_min_confidence}", style="dim")
        console.print(f"  - 最小 MemCells: {config.profile_min_memcells}", style="dim")
    console.print("=" * 60 + "\n", style="bold cyan")
    
    # 🔥 断点续传：检查已完成的对话
    completed_convs = set()
    for conv_id in raw_data_dict.keys():
        output_file = os.path.join(save_dir, f"memcell_list_conv_{conv_id}.json")
        if os.path.exists(output_file):
            # 验证文件有效性（非空且可解析）
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                    if data and len(data) > 0:  # 确保有数据
                        completed_convs.add(conv_id)
                        console.print(f"✅ 跳过已完成的会话: {conv_id} ({len(data)} memcells)", style="green")
            except Exception as e:
                console.print(f"⚠️  会话 {conv_id} 文件损坏，将重新处理: {e}", style="yellow")
    
    # 过滤出需要处理的对话
    pending_raw_data_dict = {
        conv_id: conv_data 
        for conv_id, conv_data in raw_data_dict.items() 
        if conv_id not in completed_convs
    }
    
    console.print(f"\n📊 总共发现 {len(raw_data_dict)} 个会话", style="bold cyan")
    console.print(f"✅ 已完成: {len(completed_convs)} 个", style="bold green")
    console.print(f"⏳ 待处理: {len(pending_raw_data_dict)} 个", style="bold yellow")
    
    if len(pending_raw_data_dict) == 0:
        console.print(f"\n🎉 所有会话已完成，无需处理！", style="bold green")
        return
    
    total_messages = sum(len(conv) for conv in pending_raw_data_dict.values())
    console.print(f"📝 待处理消息数: {total_messages}", style="bold blue")
    console.print(f"🚀 开始并发处理剩余会话...\n", style="bold green")

    # 创建共享的 LLM Provider 和 MemCell Extractor 实例（解决连接竞争问题）
    console.print("⚙️ 初始化 LLM Provider...", style="yellow")
    console.print(f"   模型: {config.llm_config[llm_service]['model']}", style="dim")
    console.print(
        f"   Base URL: {config.llm_config[llm_service]['base_url']}", style="dim"
    )

    shared_llm_provider = LLMProvider(
        provider_type="openai",
        model=config.llm_config[llm_service]["model"],
        api_key=config.llm_config[llm_service]["api_key"],
        base_url=config.llm_config[llm_service]["base_url"],
        temperature=config.llm_config[llm_service]["temperature"],
        max_tokens=config.llm_config[llm_service]["max_tokens"],
    )

    # 创建共享的 Event Log Extractor（使用评估专用提示词）
    console.print("⚙️ 初始化 Event Log Extractor...", style="yellow")
    shared_event_log_extractor = EventLogExtractor(
        llm_provider=shared_llm_provider,
        use_eval_prompts=True  # 评估系统使用 eval/ 提示词
    )

    # 🔥 使用待处理的对话字典（断点续传）
    # 创建进度计数器
    progress_counter = {'total': len(pending_raw_data_dict), 'completed': 0, 'failed': 0}

    # 使用 Rich 进度条
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),  # 显示 "3/10" 格式
        TextColumn("•"),
        TaskProgressColumn(),  # 显示百分比
        TextColumn("•"),
        TimeElapsedColumn(),  # 已用时间
        TextColumn("•"),
        TimeRemainingColumn(),  # 预计剩余时间
        TextColumn("•"),
        TextColumn("[bold blue]{task.fields[status]}"),
        console=console,
        transient=False,
        refresh_per_second=1,
    ) as progress:
        # 创建主进度任务
        main_task = progress.add_task(
            "[bold cyan]🎯 总进度",
            total=len(raw_data_dict),
            completed=len(completed_convs),  # 🔥 已完成的数量
            status="处理中",
        )

        # 🔥 先添加已完成的会话到进度条（显示为已完成）
        conversation_tasks = {}
        for conv_id in completed_convs:
            conv_task_id = progress.add_task(
                f"[green]Conv-{conv_id}",
                total=len(raw_data_dict[conv_id]),
                completed=len(raw_data_dict[conv_id]),  # 100%
                status="✅ (已跳过)",
            )
            conversation_tasks[conv_id] = conv_task_id

        # 🔥 为待处理的会话创建进度条任务
        updated_tasks = []
        for conv_id, conversation in pending_raw_data_dict.items():
            # 创建每个会话的进度条
            conv_task_id = progress.add_task(
                f"[yellow]Conv-{conv_id}",  # 简化名称
                total=len(conversation),  # 消息总数
                completed=0,  # 初始化为0
                status="等待",
            )
            conversation_tasks[conv_id] = conv_task_id

            # 创建处理任务
            task = process_single_conversation(
                conv_id,
                conversation,
                save_dir,
                llm_provider=shared_llm_provider,
                event_log_extractor=shared_event_log_extractor,
                progress_counter=progress_counter,
                progress=progress,
                task_id=conv_task_id,
                config=config,  # 传入配置
            )
            updated_tasks.append(task)

        # 定义完成时更新函数
        async def run_with_completion(task, conv_id):
            result = await task
            progress.update(
                conversation_tasks[conv_id],
                status="✅",
                completed=progress.tasks[conversation_tasks[conv_id]].total,
            )
            progress.update(main_task, advance=1)
            return result

        # 🔥 并发执行所有待处理的任务
        if updated_tasks:
            results = await asyncio.gather(
                *[
                    run_with_completion(task, conv_id)
                    for (conv_id, _), task in zip(pending_raw_data_dict.items(), updated_tasks)
                ]
            )
        else:
            results = []
        # with open(os.path.join(save_dir, "response_info.json"), "w") as f:
        #     json.dump(shared_llm_provider.provider.response_info, f, ensure_ascii=False, indent=2)
        # 更新主进度为完成
        progress.update(main_task, status="✅ 完成")

    end_time = time.time()

    # 统计结果
    all_memcells = []
    successful_convs = 0
    for conv_id, memcell_list in results:
        if memcell_list:
            successful_convs += 1
            all_memcells.extend(memcell_list)

    console.print("\n" + "=" * 60, style="dim")
    console.print("📊 处理完成统计:", style="bold")
    console.print(
        f"   ✅ 成功处理会话数: {successful_convs}/{len(raw_data_dict)}", style="green"
    )
    console.print(f"   📝 总共提取的 memcells: {len(all_memcells)}", style="blue")
    console.print(f"   ⏱️  总耗时: {end_time - start_time:.2f} 秒", style="yellow")
    console.print(
        f"   🚀 平均每会话耗时: {(end_time - start_time)/len(raw_data_dict):.2f} 秒",
        style="cyan",
    )
    console.print("=" * 60, style="dim")

    # 保存汇总结果
    all_memcells_dicts = [memcell.to_dict() for memcell in all_memcells]
    summary_file = os.path.join(save_dir, "memcell_list_all.json")
    with open(summary_file, "w") as f:
        json.dump(all_memcells_dicts, f, ensure_ascii=False, indent=2)
    console.print(f"\n💾 汇总结果已保存到: {summary_file}", style="green")

    # ===== 新增：汇总聚类和 Profile 统计 =====
    # 统计所有会话的聚类和 Profile 信息
    total_clusters = 0
    total_profiles = 0
    cluster_stats_list = []
    profile_stats_list = []
    
    stats_dir = Path(save_dir) / "stats"
    if stats_dir.exists():
        for stats_file in stats_dir.glob("conv_*_stats.json"):
            try:
                with open(stats_file) as f:
                    conv_stats = json.load(f)
                total_clusters += conv_stats.get("clustering", {}).get("total_clusters", 0)
                total_profiles += conv_stats.get("profile_count", 0)
                cluster_stats_list.append(conv_stats.get("clustering", {}))
                profile_stats_list.append(conv_stats.get("profiles", {}))
            except Exception:
                pass
    
    # 保存处理摘要（新增聚类和 Profile 统计）
    summary = {
        "total_conversations": len(raw_data_dict),
        "successful_conversations": successful_convs,
        "total_memcells": len(all_memcells),
        "total_clusters": total_clusters,
        "total_profiles": total_profiles,
        "processing_time_seconds": end_time - start_time,
        "average_time_per_conversation": (end_time - start_time) / len(raw_data_dict),
        "conversation_results": {
            conv_id: len(memcell_list) for conv_id, memcell_list in results
        },
        "clustering_summary": {
            "total_clusters": total_clusters,
            "avg_clusters_per_conv": total_clusters / successful_convs if successful_convs > 0 else 0,
        },
        "profile_summary": {
            "total_profiles": total_profiles,
            "avg_profiles_per_conv": total_profiles / successful_convs if successful_convs > 0 else 0,
        },
    }
    summary_info_file = os.path.join(save_dir, "processing_summary.json")
    with open(summary_info_file, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    console.print(f"📊 处理摘要已保存到: {summary_info_file}", style="green")
    
    # 打印聚类和 Profile 统计
    console.print(f"\n📊 聚类统计:", style="bold cyan")
    console.print(f"   - 总聚类数: {total_clusters}", style="cyan")
    console.print(f"   - 平均每会话: {total_clusters / successful_convs if successful_convs > 0 else 0:.1f}", style="cyan")
    console.print(f"\n👤 Profile 统计:", style="bold green")
    console.print(f"   - 总 Profiles: {total_profiles}", style="green")
    console.print(f"   - 平均每会话: {total_profiles / successful_convs if successful_convs > 0 else 0:.1f}\n", style="green")


if __name__ == "__main__":
    asyncio.run(main())
