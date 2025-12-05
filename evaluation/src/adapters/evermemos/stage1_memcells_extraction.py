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
    from_datetime_str_strict,
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
from api_specs.memory_types import RawDataType

# Clustering and Profile management components
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
            key=lambda x: int(x.replace("session_", "")),
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
                    # Priority 1: Use message-level timestamp if available (e.g., evermembench)
                    if 'time' in msg and msg['time']:
                        # Parse message-level timestamp (strict parsing, raises on error)
                        msg_datetime = from_datetime_str_strict(msg['time'])
                        iso_timestamp = to_iso_format(msg_datetime)
                    else:
                        # Priority 2: Generate timestamp from session time (e.g., locomo)
                        msg_timestamp = session_time + timedelta(seconds=i * 30)
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
    conv_id: str = None,  # Add conversation ID for progress bar description
    progress: Progress = None,  # Add progress bar object
    task_id: int = None,  # Add task ID
    use_semantic_extraction: bool = False,  # Enable semantic memory extraction
) -> list:

    episode_extractor = EpisodeMemoryExtractor(
        llm_provider=llm_provider, use_eval_prompts=True
    )
    memcell_list = []
    speakers = {
        raw_data.content["speaker_id"]
        for raw_data in raw_data_list
        if isinstance(raw_data.content, dict) and "speaker_id" in raw_data.content
    }
    history_raw_data_list = []
    # raw_data_list = raw_data_list[:100]

    # Process messages
    total_messages = len(raw_data_list)
    smart_mask_flag = False

    for idx, raw_data in enumerate(raw_data_list):
        # Update progress bar (before processing, showing which message is being processed)
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
        for i in range(10):
            try:
                result = await memcell_extractor.extract_memcell(
                    request,
                    use_semantic_extraction=use_semantic_extraction,  # Pass flag
                )
                break
            except Exception as e:
                print('retry: ', i)
                if i == 9:
                    raise Exception("Memcell extraction failed")
                continue
        memcell_result = result[0]
        # print(f"   ✅ Memcell result: {memcell_result}")  # Commented to avoid interrupting progress bar
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

    # Processing complete, update progress to 100%
    if progress and task_id is not None:
        progress.update(task_id, completed=total_messages)

    if history_raw_data_list:
        # Determine timestamp: use last memcell's timestamp if available, otherwise use last message's timestamp
        if memcell_list:
            last_timestamp = memcell_list[-1].timestamp
        else:
            # Fallback: use the last raw data's timestamp if memcell_list is empty
            # RawData.content["timestamp"] is ISO format string, guaranteed by raw_data_load
            last_raw_data = history_raw_data_list[-1]
            if isinstance(last_raw_data.content, dict) and "timestamp" in last_raw_data.content:
                # Convert ISO format string to datetime
                last_timestamp = from_iso_format(last_raw_data.content["timestamp"])
            else:
                # Defensive fallback (should not happen after raw_data_load fix)
                last_timestamp = get_now_with_timezone()
        
        memcell = MemCell(
            type=RawDataType.CONVERSATION,
            event_id=str(uuid.uuid4()),
            user_id_list=list(speakers),
            original_data=history_raw_data_list,
            timestamp=last_timestamp,
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
    config: ExperimentConfig = None,  # Pass configuration
) -> tuple:
    """Process single conversation and return results (with clustering and Profile extraction).

    Args:
        conv_id: Conversation ID
        conversation: Conversation data
        save_dir: Save directory
        llm_provider: Shared LLM provider instance
        event_log_extractor: Event log extractor instance
        progress: Progress bar object
        task_id: Progress task ID
        config: Experiment configuration (for reading feature flags)

    Returns:
        tuple: (conv_id, memcell_list)
    """
    try:
        # Update status to processing
        if progress and task_id is not None:
            progress.update(task_id, status="Processing")

        # Create components based on configuration
        cluster_mgr = None
        profile_mgr = None

        # Create MemCellExtractor
        raw_data_list = convert_conversation_to_raw_data_list(conversation)
        memcell_extractor = ConvMemCellExtractor(
            llm_provider=llm_provider, use_eval_prompts=True
        )

        # Conditional creation: Cluster manager (per-conversation)
        if config and config.enable_clustering:
            cluster_storage = InMemoryClusterStorage(
                enable_persistence=True,
                persist_dir=Path(save_dir) / "clusters" / f"conv_{conv_id}",
            )
            cluster_config = ClusterManagerConfig(
                similarity_threshold=config.cluster_similarity_threshold,
                max_time_gap_days=config.cluster_max_time_gap_days,
                enable_persistence=True,
                persist_dir=str(Path(save_dir) / "clusters" / f"conv_{conv_id}"),
                clustering_algorithm="centroid",
            )
            cluster_mgr = ClusterManager(config=cluster_config, storage=cluster_storage)
            cluster_mgr.attach_to_extractor(memcell_extractor)

        # Conditional creation: Profile manager
        if config and config.enable_profile_extraction and cluster_mgr:
            profile_storage = InMemoryProfileStorage(
                enable_persistence=True,
                persist_dir=Path(save_dir) / "profiles" / f"conv_{conv_id}",
                enable_versioning=True,
            )

            # Set scenario type dynamically
            scenario = (
                ScenarioType.ASSISTANT
                if config.profile_scenario.lower() == "assistant"
                else ScenarioType.GROUP_CHAT
            )

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
                group_name=f"LoComo Conversation {conv_id}",
            )

            # Set minimum MemCells threshold
            profile_mgr._min_memcells_threshold = config.profile_min_memcells

            # Connect components
            profile_mgr.attach_to_cluster_manager(cluster_mgr)

        # Extract MemCells (enable semantic memory based on config)
        use_semantic = config.enable_semantic_extraction if config else False
        memcell_list = await memcell_extraction_from_conversation(
            raw_data_list,
            llm_provider=llm_provider,
            memcell_extractor=memcell_extractor,
            conv_id=conv_id,
            progress=progress,
            task_id=task_id,
            use_semantic_extraction=use_semantic,  # Pass semantic memory flag
        )
        # print(f"   ✅ Conv {conv_id}: {len(memcell_list)} memcells extracted")  # Commented to avoid interrupting progress bar

        # Convert timestamps to datetime objects before saving
        for memcell in memcell_list:
            if hasattr(memcell, 'timestamp'):
                ts = memcell.timestamp
                if isinstance(ts, (int, float)):
                    # Convert int/float timestamp to datetime with timezone
                    memcell.timestamp = from_timestamp(ts)
                elif isinstance(ts, str):
                    # Convert string timestamp to datetime with timezone
                    memcell.timestamp = from_iso_format(ts)
                elif not isinstance(ts, datetime):
                    # If unexpected type, use current time
                    memcell.timestamp = get_now_with_timezone()

        # Optimization: concurrent event log generation (10-20x faster)
        if event_log_extractor:
            # Prepare all memcells needing event log extraction
            memcells_with_episode = [
                (idx, memcell)
                for idx, memcell in enumerate(memcell_list)
                if hasattr(memcell, 'episode') and memcell.episode
            ]

            # Define single event log extraction task
            async def extract_single_event_log(idx: int, memcell):
                try:
                    event_log = await event_log_extractor.extract_event_log(
                        episode_text=memcell.episode, timestamp=memcell.timestamp
                    )
                    return idx, event_log
                except Exception as e:
                    console = Console()
                    console.print(
                        f"\n⚠️  Event log generation failed (Conv {conv_id}, Memcell {idx}): {e}",
                        style="yellow",
                    )
                    return idx, None

            # Concurrent extraction of all event logs (using Semaphore to control concurrency)
            sem = asyncio.Semaphore(
                20
            )  # Limit concurrency to 20 (avoid API rate limits)

            async def extract_with_semaphore(idx, memcell):
                async with sem:
                    return await extract_single_event_log(idx, memcell)

            print(
                f"\n🔥 Starting concurrent extraction of {len(memcells_with_episode)} event logs..."
            )
            event_log_tasks = [
                extract_with_semaphore(idx, memcell)
                for idx, memcell in memcells_with_episode
            ]
            event_log_results = await asyncio.gather(*event_log_tasks)

            # Link event logs back to corresponding memcells
            for original_idx, event_log in event_log_results:
                if event_log:
                    memcell_list[original_idx].event_log = event_log

            print(
                f"✅ Event log extraction complete: {sum(1 for _, el in event_log_results if el)}/{len(event_log_results)} succeeded"
            )

        # Save single conversation results
        memcell_dicts = []
        for memcell in memcell_list:
            memcell_dict = memcell.to_dict()
            # If has event_log, add to dict
            if hasattr(memcell, 'event_log') and memcell.event_log:
                memcell_dict['event_log'] = memcell.event_log.to_dict()
            memcell_dicts.append(memcell_dict)

        memcell_dicts = [memcell_dict for memcell_dict in memcell_dicts]
        # print(memcell_dicts)  # Commented to avoid large output
        output_file = os.path.join(save_dir, f"memcell_list_conv_{conv_id}.json")
        with open(output_file, "w") as f:
            json.dump(memcell_dicts, f, ensure_ascii=False, indent=2)

        # Conditional export: clustering and Profile results
        cluster_stats = {}
        profile_stats = {}
        profile_count = 0

        if cluster_mgr or profile_mgr:
            await asyncio.sleep(2)  # Give async tasks time to complete

        # Export clustering results (if enabled)
        if cluster_mgr:
            cluster_output_dir = Path(save_dir) / "clusters" / f"conv_{conv_id}"
            cluster_output_dir.mkdir(parents=True, exist_ok=True)
            await cluster_mgr.export_clusters(cluster_output_dir)
            cluster_stats = cluster_mgr.get_stats()

        # Export Profiles (if enabled)
        if profile_mgr:
            profile_output_dir = Path(save_dir) / "profiles" / f"conv_{conv_id}"
            profile_count = await profile_mgr.export_profiles(
                profile_output_dir, include_history=True
            )
            profile_stats = profile_mgr.get_stats()

        # Save statistics
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

        # Update progress (silent, avoid interrupting progress bar)
        if progress_counter:
            progress_counter['completed'] += 1
            # No printing to avoid interrupting progress bar

        return conv_id, memcell_list

    except Exception as e:
        # Show error message so we know the specific issue
        console = Console()
        console.print(
            f"\n❌ Error processing conversation {conv_id}: {e}", style="bold red"
        )
        if progress_counter:
            progress_counter['completed'] += 1
            progress_counter['failed'] += 1
        import traceback

        traceback.print_exc()
        return conv_id, []


async def main():
    """Main function - concurrent processing of all conversations."""

    config = ExperimentConfig()
    llm_service = config.llm_service
    dataset_path = config.datase_path
    raw_data_dict = raw_data_load(dataset_path)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(CURRENT_DIR, config.experiment_name), exist_ok=True)
    os.makedirs(
        os.path.join(CURRENT_DIR, config.experiment_name, "memcells"), exist_ok=True
    )
    save_dir = os.path.join(CURRENT_DIR, config.experiment_name, "memcells")

    console = Console()

    # Print configuration info
    console.print("\n" + "=" * 60, style="bold cyan")
    console.print("Experiment Configuration", style="bold cyan")
    console.print("=" * 60, style="bold cyan")
    console.print(f"Experiment name: {config.experiment_name}", style="cyan")
    console.print(f"Dataset path: {config.datase_path}", style="cyan")
    console.print(f"\nFeature flags:", style="bold yellow")
    console.print(
        f"  - Semantic extraction: {'✅ Enabled' if config.enable_semantic_extraction else '❌ Disabled'}",
        style="green" if config.enable_semantic_extraction else "dim",
    )
    console.print(
        f"  - Clustering: {'✅ Enabled' if config.enable_clustering else '❌ Disabled'}",
        style="green" if config.enable_clustering else "dim",
    )
    console.print(
        f"  - Profile extraction: {'✅ Enabled' if config.enable_profile_extraction else '❌ Disabled'}",
        style="green" if config.enable_profile_extraction else "dim",
    )

    if config.enable_clustering:
        console.print(f"\nClustering config:", style="bold")
        console.print(
            f"  - Similarity threshold: {config.cluster_similarity_threshold}",
            style="dim",
        )
        console.print(
            f"  - Max time gap: {config.cluster_max_time_gap_days} days", style="dim"
        )

    if config.enable_profile_extraction:
        console.print(f"\nProfile config:", style="bold")
        console.print(f"  - Scenario: {config.profile_scenario}", style="dim")
        console.print(
            f"  - Min confidence: {config.profile_min_confidence}", style="dim"
        )
        console.print(f"  - Min MemCells: {config.profile_min_memcells}", style="dim")
    console.print("=" * 60 + "\n", style="bold cyan")

    # Checkpoint resume: check completed conversations
    completed_convs = set()
    for conv_id in raw_data_dict.keys():
        output_file = os.path.join(save_dir, f"memcell_list_conv_{conv_id}.json")
        if os.path.exists(output_file):
            # Validate file (non-empty and parseable)
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                    if data and len(data) > 0:  # Ensure data available
                        completed_convs.add(conv_id)
                        console.print(
                            f"✅ Skipping completed conversation: {conv_id} ({len(data)} memcells)",
                            style="green",
                        )
            except Exception as e:
                console.print(
                    f"⚠️  Conversation {conv_id} file corrupted, will reprocess: {e}",
                    style="yellow",
                )

    # Filter conversations needing processing
    pending_raw_data_dict = {
        conv_id: conv_data
        for conv_id, conv_data in raw_data_dict.items()
        if conv_id not in completed_convs
    }

    console.print(
        f"\n📊 Total conversations found: {len(raw_data_dict)}", style="bold cyan"
    )
    console.print(f"✅ Completed: {len(completed_convs)}", style="bold green")
    console.print(f"⏳ Pending: {len(pending_raw_data_dict)}", style="bold yellow")

    if len(pending_raw_data_dict) == 0:
        console.print(
            f"\n🎉 All conversations completed, nothing to process!", style="bold green"
        )
        return

    total_messages = sum(len(conv) for conv in pending_raw_data_dict.values())
    console.print(f"📝 Pending messages: {total_messages}", style="bold blue")
    console.print(
        f"🚀 Starting concurrent processing of remaining conversations...\n",
        style="bold green",
    )

    # Create shared LLM Provider and MemCell Extractor instances (solve connection race issue)
    console.print("⚙️ Initializing LLM Provider...", style="yellow")
    console.print(f"   Model: {config.llm_config[llm_service]['model']}", style="dim")
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
        use_eval_prompts=True,  # 评估系统使用 eval/ 提示词
    )

    # 🔥 Use pending conversation dict (checkpoint resume)
    # Create progress counter
    progress_counter = {
        'total': len(pending_raw_data_dict),
        'completed': 0,
        'failed': 0,
    }

    # Use Rich progress bar
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),  # Show "3/10" format
        TextColumn("•"),
        TaskProgressColumn(),  # Show percentage
        TextColumn("•"),
        TimeElapsedColumn(),  # Elapsed time
        TextColumn("•"),
        TimeRemainingColumn(),  # Remaining time
        TextColumn("•"),
        TextColumn("[bold blue]{task.fields[status]}"),
        console=console,
        transient=False,
        refresh_per_second=1,
    ) as progress:
        # Create main progress task
        main_task = progress.add_task(
            "[bold cyan]🎯 Overall Progress",
            total=len(raw_data_dict),
            completed=len(completed_convs),  # Completed count
            status="Processing",
        )

        # First add completed conversations to progress bar (show as completed)
        conversation_tasks = {}
        for conv_id in completed_convs:
            conv_task_id = progress.add_task(
                f"[green]Conv-{conv_id}",
                total=len(raw_data_dict[conv_id]),
                completed=len(raw_data_dict[conv_id]),  # 100%
                status="✅ (Skipped)",
            )
            conversation_tasks[conv_id] = conv_task_id

        # Create progress bar tasks for pending conversations
        updated_tasks = []
        for conv_id, conversation in pending_raw_data_dict.items():
            # Create progress bar for each conversation
            conv_task_id = progress.add_task(
                f"[yellow]Conv-{conv_id}",  # Simplified name
                total=len(conversation),  # Total messages
                completed=0,  # Initialize to 0
                status="Waiting",
            )
            conversation_tasks[conv_id] = conv_task_id

            # Create processing task
            task = process_single_conversation(
                conv_id,
                conversation,
                save_dir,
                llm_provider=shared_llm_provider,
                event_log_extractor=shared_event_log_extractor,
                progress_counter=progress_counter,
                progress=progress,
                task_id=conv_task_id,
                config=config,  # Pass configuration
            )
            updated_tasks.append(task)

        # Define completion update function
        async def run_with_completion(task, conv_id):
            result = await task
            progress.update(
                conversation_tasks[conv_id],
                status="✅",
                completed=progress.tasks[conversation_tasks[conv_id]].total,
            )
            progress.update(main_task, advance=1)
            return result

        # Execute all pending tasks concurrently
        if updated_tasks:
            results = await asyncio.gather(
                *[
                    run_with_completion(task, conv_id)
                    for (conv_id, _), task in zip(
                        pending_raw_data_dict.items(), updated_tasks
                    )
                ]
            )
        else:
            results = []
        # with open(os.path.join(save_dir, "response_info.json"), "w") as f:
        #     json.dump(shared_llm_provider.provider.response_info, f, ensure_ascii=False, indent=2)
        # Update main progress to complete
        progress.update(main_task, status="✅ Complete")

    end_time = time.time()

    # Gather statistics
    all_memcells = []
    successful_convs = 0
    for conv_id, memcell_list in results:
        if memcell_list:
            successful_convs += 1
            all_memcells.extend(memcell_list)

    console.print("\n" + "=" * 60, style="dim")
    console.print("📊 Processing completion statistics:", style="bold")
    console.print(
        f"   ✅ Successfully processed: {successful_convs}/{len(raw_data_dict)}",
        style="green",
    )
    console.print(f"   📝 Total memcells extracted: {len(all_memcells)}", style="blue")
    console.print(f"   ⏱️  Total time: {end_time - start_time:.2f}s", style="yellow")
    console.print(
        f"   🚀 Average per conversation: {(end_time - start_time)/len(raw_data_dict):.2f}s",
        style="cyan",
    )
    console.print("=" * 60, style="dim")

    # Save summary results
    all_memcells_dicts = [memcell.to_dict() for memcell in all_memcells]
    summary_file = os.path.join(save_dir, "memcell_list_all.json")
    with open(summary_file, "w") as f:
        json.dump(all_memcells_dicts, f, ensure_ascii=False, indent=2)
    console.print(f"\n💾 Summary results saved to: {summary_file}", style="green")

    # Aggregate clustering and Profile statistics
    # Collect clustering and Profile info from all conversations
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
                total_clusters += conv_stats.get("clustering", {}).get(
                    "total_clusters", 0
                )
                total_profiles += conv_stats.get("profile_count", 0)
                cluster_stats_list.append(conv_stats.get("clustering", {}))
                profile_stats_list.append(conv_stats.get("profiles", {}))
            except Exception:
                pass

    # Save processing summary (with clustering and Profile statistics)
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
            "avg_clusters_per_conv": (
                total_clusters / successful_convs if successful_convs > 0 else 0
            ),
        },
        "profile_summary": {
            "total_profiles": total_profiles,
            "avg_profiles_per_conv": (
                total_profiles / successful_convs if successful_convs > 0 else 0
            ),
        },
    }
    summary_info_file = os.path.join(save_dir, "processing_summary.json")
    with open(summary_info_file, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    console.print(f"📊 Processing summary saved to: {summary_info_file}", style="green")

    # Print clustering and Profile statistics
    console.print(f"\n📊 Clustering statistics:", style="bold cyan")
    console.print(f"   - Total clusters: {total_clusters}", style="cyan")
    console.print(
        f"   - Average per conversation: {total_clusters / successful_convs if successful_convs > 0 else 0:.1f}",
        style="cyan",
    )
    console.print(f"\n👤 Profile statistics:", style="bold green")
    console.print(f"   - Total Profiles: {total_profiles}", style="green")
    console.print(
        f"   - Average per conversation: {total_profiles / successful_convs if successful_convs > 0 else 0:.1f}\n",
        style="green",
    )


if __name__ == "__main__":
    asyncio.run(main())
