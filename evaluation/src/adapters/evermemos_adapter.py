"""
EverMemOS Adapter - connects evaluation framework with EverMemOS implementation.
"""

import asyncio
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

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

from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult
from common_utils.datetime_utils import to_iso_format

# Import EverMemOS implementation
from evaluation.src.adapters.evermemos import (
    stage1_memcells_extraction,
    stage2_index_building,
    stage3_memory_retrivel,
    stage4_response,
)
from evaluation.src.adapters.evermemos.prompts.answer_prompts_personamem import (
    PERSONAMEM_MCQ_PROMPT,
)
from evaluation.src.adapters.evermemos.prompts.answer_prompts_lme import (
    get_lme_prompt_template,
)
from evaluation.src.adapters.evermemos.prompts.profile.profile_cls_prompt import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT_TEMPLATE,
)

# Import Memory Layer components
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.event_log_extractor import EventLogExtractor


@register_adapter("evermemos")
class EverMemOSAdapter(BaseAdapter):
    """
    EverMemOS adapter.

    Responsibilities:
    1. Receive calls from evaluation framework
    2. Convert data formats (evaluation framework ↔ EverMemOS)
    3. Call stage*.py implementations
    4. Return results in evaluation framework format

    Implementation details:
    - MemCell extraction (stage1)
    - Index building (stage2)
    - Retrieval logic (stage3)
    - Answer generation (stage4)
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")

        # Initialize LLM Provider (shared across all stages)
        # Read from YAML llm configuration
        llm_config = config.get("llm", {})

        self.llm_provider = LLMProvider(
            provider_type=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 32768),
        )

        # Initialize Event Log Extractor
        self.event_log_extractor = EventLogExtractor(llm_provider=self.llm_provider)

        # Ensure NLTK data is available
        stage2_index_building.ensure_nltk_data()

        print(f"✅ EverMemOS Adapter initialized")
        print(f"   LLM Model: {llm_config.get('model')}")
        print(f"   Output Dir: {self.output_dir}")

    @staticmethod
    def _extract_conv_index(conversation_id: str) -> str:
        """
        Extract numeric index part from conversation_id.

        Examples:
        - "locomo_0" -> "0"
        - "personamem_42" -> "42"
        - "123" -> "123"
        - "test_abc_5" -> "5"

        Strategy: Take the part after the last underscore, or return original if no underscore
        """
        if "_" in conversation_id:
            return conversation_id.split("_")[-1]
        return conversation_id

    def _check_missing_indexes(
        self, index_dir: Path, num_conv: int, index_type: str = "bm25"
    ) -> List[int]:
        """
        Check for missing index files.

        Args:
            index_dir: Index directory
            num_conv: Total number of conversations
            index_type: Index type ("bm25" or "embedding")

        Returns:
            List of conversation indices with missing indexes
        """
        missing_indexes = []

        for i in range(num_conv):
            if index_type == "bm25":
                index_file = index_dir / f"bm25_index_conv_{i}.pkl"
            else:  # embedding
                index_file = index_dir / f"embedding_index_conv_{i}.pkl"

            if not index_file.exists():
                missing_indexes.append(i)

        return missing_indexes

    def _load_profile_text(self, conversation_id: str) -> str:
        conv_index = self._extract_conv_index(conversation_id)
        profile_file = self.output_dir / "profiles" / f"profile_conv_{conv_index}.json"
        if not profile_file.exists():
            return ""

        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            profiles = data.get("profiles", [])
        except Exception as e:
            print(f"Warning: Failed to load profile file {profile_file}: {e}")
            return ""

        return self._format_profiles(profiles)

    def _format_profiles(self, profiles: List[Dict[str, Any]]) -> str:
        if not profiles:
            return ""

        def _format_value_list(items: Any) -> List[str]:
            values = []
            if not isinstance(items, list):
                return values
            for item in items:
                if isinstance(item, dict):
                    value = (
                        item.get("value")
                        or item.get("skill")
                        or item.get("trait")
                        or item.get("description")
                    )
                    if value:
                        values.append(str(value))
                elif isinstance(item, str):
                    values.append(item)
            return values

        lines: List[str] = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue

            user_id = profile.get("user_id")
            if user_id:
                lines.append(f"User: {user_id}")

            explicit_info = profile.get("explicit_info") or []
            implicit_traits = profile.get("implicit_traits") or []

            if explicit_info or implicit_traits:
                if explicit_info:
                    lines.append("[Explicit Info]")
                    for item in explicit_info:
                        if not isinstance(item, dict):
                            continue
                        desc = item.get("description", "").strip()
                        category = item.get("category", "").strip()
                        if category and desc:
                            lines.append(f"- {category}: {desc}")
                        elif desc:
                            lines.append(f"- {desc}")
                if implicit_traits:
                    if lines:
                        lines.append("")
                    lines.append("[Implicit Traits]")
                    for item in implicit_traits:
                        if not isinstance(item, dict):
                            continue
                        trait = item.get("trait", "").strip()
                        desc = item.get("description", "").strip()
                        if trait and desc:
                            lines.append(f"- {trait}: {desc}")
                        elif trait:
                            lines.append(f"- {trait}")
                if lines and lines[-1] != "":
                    lines.append("")
                continue

            # Fallback for non-life profiles
            fallback_sections = [
                ("interests", "Interests"),
                ("personality", "Personality"),
                ("tendency", "Tendency"),
                ("hard_skills", "Hard Skills"),
                ("soft_skills", "Soft Skills"),
            ]
            for key, label in fallback_sections:
                values = _format_value_list(profile.get(key))
                if values:
                    lines.append(f"[{label}]")
                    for value in values:
                        lines.append(f"- {value}")
                    lines.append("")

        return "\n".join(line.rstrip() for line in lines).strip()

    @staticmethod
    def _format_options(options: Dict[str, str]) -> str:
        if not options:
            return ""
        sorted_items = sorted(options.items(), key=lambda x: x[0])
        return "\n".join([f"{key} {value}" for key, value in sorted_items])

    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        if not response:
            return {
                "classification": "no_profile",
                "reasoning": "Empty response from classifier",
                "key_evidence": "",
            }

        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return {
                "classification": "no_profile",
                "reasoning": "Classifier response missing JSON",
                "key_evidence": response[:200],
            }

        try:
            parsed = json.loads(response[start_idx : end_idx + 1])
        except json.JSONDecodeError:
            return {
                "classification": "no_profile",
                "reasoning": "Failed to parse classifier JSON",
                "key_evidence": response[start_idx : end_idx + 1][:200],
            }

        return {
            "classification": parsed.get("classification", "no_profile"),
            "reasoning": parsed.get("reasoning", ""),
            "key_evidence": parsed.get("key_evidence", ""),
        }

    async def classify_profile_dependency(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_query = payload.get("user_query", "")
        correct_answer = payload.get("correct_answer", "")
        incorrect_answers = payload.get("incorrect_answers", [])
        preference = payload.get("preference", "")
        pref_type = payload.get("pref_type", "")
        related_snippet = payload.get("related_conversation_snippet", "")

        if not user_query or not correct_answer:
            return {
                "classification": "no_profile",
                "reasoning": "Missing user_query or correct_answer",
                "key_evidence": "",
            }

        prompt = (
            CLASSIFICATION_SYSTEM_PROMPT
            + "\n\n"
            + CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
                user_query=user_query,
                correct_answer=correct_answer,
                incorrect_answers=incorrect_answers,
                preference=preference,
                pref_type=pref_type,
                related_conversation_snippet=related_snippet,
            )
        )

        try:
            response = await self.llm_provider.generate(
                prompt=prompt, temperature=0.0, max_tokens=800
            )
        except Exception as e:
            return {
                "classification": "no_profile",
                "reasoning": f"Classifier failed: {e}",
                "key_evidence": "",
            }

        return self._parse_classification_response(response)

    async def add(
        self,
        conversations: List[Conversation],
        output_dir: Path = None,
        checkpoint_manager=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add stage: Extract MemCells and build indexes.

        Call flow:
        1. Stage 1: Extract MemCells (stage1_memcells_extraction.py) - concurrent processing
        2. Stage 2: Build BM25 and Embedding indexes (stage2_index_building.py)

        Returns: Index metadata (Plan A: lazy loading)
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        memcells_dir = output_dir / "memcells"
        memcells_dir.mkdir(parents=True, exist_ok=True)
        bm25_index_dir = output_dir / "bm25_index"
        emb_index_dir = output_dir / "vectors"
        bm25_index_dir.mkdir(parents=True, exist_ok=True)
        emb_index_dir.mkdir(parents=True, exist_ok=True)

        console = Console()

        # ========== Stage 1: MemCell Extraction (concurrent processing) ==========
        console.print(f"\n{'='*60}", style="bold cyan")
        console.print(f"Stage 1: MemCell Extraction", style="bold cyan")
        console.print(f"{'='*60}", style="bold cyan")

        # Convert data format: evaluation framework → EverMemOS
        raw_data_dict = {}
        for conv in conversations:
            conv_id = conv.conversation_id
            raw_data = []

            for idx, msg in enumerate(conv.messages):
                # Handle timestamp: if None, use index-based pseudo timestamp
                if msg.timestamp is not None:
                    timestamp_str = to_iso_format(msg.timestamp)
                else:
                    # Generate pseudo timestamp using message index (maintain relative order)
                    # Base time: 2023-01-01 00:00:00, 30 seconds interval per message
                    from datetime import datetime, timedelta

                    base_time = datetime(2023, 1, 1, 0, 0, 0)
                    pseudo_time = base_time + timedelta(seconds=idx * 30)
                    timestamp_str = to_iso_format(pseudo_time)

                message_dict = {
                    "speaker_id": msg.speaker_id,
                    "user_name": msg.speaker_name or msg.speaker_id,
                    "speaker_name": msg.speaker_name or msg.speaker_id,
                    "content": msg.content,
                    "timestamp": timestamp_str,
                }

                # Add optional fields
                for optional_field in ["img_url", "blip_caption", "query"]:
                    if (
                        optional_field in msg.metadata
                        and msg.metadata[optional_field] is not None
                    ):
                        message_dict[optional_field] = msg.metadata[optional_field]

                raw_data.append(message_dict)

            raw_data_dict[conv_id] = raw_data

        # Check completed conversations (checkpoint resume)
        # Use extracted index to check files (stage1 saves using extracted index)
        completed_convs = set()
        if checkpoint_manager:
            all_conv_indices = [
                self._extract_conv_index(conv.conversation_id) for conv in conversations
            ]
            completed_indices = checkpoint_manager.load_add_progress(
                memcells_dir, all_conv_indices
            )
            # Map completed indices back to original conversation_id
            for conv in conversations:
                if self._extract_conv_index(conv.conversation_id) in completed_indices:
                    completed_convs.add(conv.conversation_id)

        # Filter conversations to process
        pending_conversations = [
            conv
            for conv in conversations
            if conv.conversation_id not in completed_convs
        ]

        console.print(
            f"\n📊 Total conversations: {len(conversations)}", style="bold cyan"
        )
        console.print(f"✅ Completed: {len(completed_convs)}", style="bold green")
        console.print(f"⏳ Pending: {len(pending_conversations)}", style="bold yellow")

        if len(pending_conversations) == 0:
            console.print(
                f"\n🎉 All conversations completed, skipping MemCell extraction!",
                style="bold green",
            )
        else:
            total_messages = sum(
                len(raw_data_dict[c.conversation_id]) for c in pending_conversations
            )
            console.print(f"📝 Pending messages: {total_messages}", style="bold blue")
            console.print(f"🚀 Starting concurrent processing...\n", style="bold green")

            # Use Rich progress bar for concurrent processing
            start_time = time.time()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                TextColumn("•"),
                TextColumn("[bold blue]{task.fields[status]}"),
                console=console,
                transient=False,
                refresh_per_second=1,
            ) as progress:
                # Create main progress task
                main_task = progress.add_task(
                    "[bold cyan]🎯 Overall Progress",
                    total=len(conversations),
                    completed=len(completed_convs),
                    status="Processing",
                )

                # Create progress bars for completed conversations (show as complete)
                conversation_tasks = {}
                for conv_id in completed_convs:
                    conv_index = self._extract_conv_index(conv_id)
                    conv_task_id = progress.add_task(
                        f"[green]Conv-{conv_index}",
                        total=len(raw_data_dict.get(conv_id, [])),
                        completed=len(raw_data_dict.get(conv_id, [])),
                        status="✅ (Skipped)",
                    )
                    conversation_tasks[conv_id] = conv_task_id

                # Create progress bars and tasks for pending conversations
                processing_tasks = []
                for conv in pending_conversations:
                    conv_id = conv.conversation_id
                    conv_index = self._extract_conv_index(
                        conv_id
                    )  # Extract numeric index
                    conv_task_id = progress.add_task(
                        f"[yellow]Conv-{conv_index}",
                        total=len(raw_data_dict[conv_id]),
                        completed=0,
                        status="Waiting",
                    )
                    conversation_tasks[conv_id] = conv_task_id

                    # Create processing task, pass extracted index
                    task = stage1_memcells_extraction.process_single_conversation(
                        conv_id=conv_index,  # Use extracted index
                        conversation=raw_data_dict[conv_id],  # Data uses original ID
                        save_dir=str(memcells_dir),
                        llm_provider=self.llm_provider,
                        event_log_extractor=self.event_log_extractor,
                        progress_counter=None,
                        progress=progress,
                        task_id=conv_task_id,
                        config=self._convert_config_to_experiment_config(),
                    )
                    processing_tasks.append((conv_id, task))

                # Define completion update function
                async def run_with_completion(conv_id, task):
                    result = await task
                    progress.update(
                        conversation_tasks[conv_id],
                        status="✅",
                        completed=progress.tasks[conversation_tasks[conv_id]].total,
                    )
                    progress.update(main_task, advance=1)
                    return result

                # Execute all pending tasks concurrently
                if processing_tasks:
                    results = await asyncio.gather(
                        *[
                            run_with_completion(conv_id, task)
                            for conv_id, task in processing_tasks
                        ]
                    )
                else:
                    results = []

                progress.update(main_task, status="✅ Complete")

            end_time = time.time()
            elapsed = end_time - start_time

            # Statistics
            successful_convs = sum(1 for _, memcell_list in results if memcell_list)
            total_memcells = sum(len(memcell_list) for _, memcell_list in results)

            console.print("\n" + "=" * 60, style="dim")
            console.print("📊 MemCell Extraction Statistics:", style="bold")
            console.print(
                f"   ✅ Successfully processed: {successful_convs}/{len(pending_conversations)}",
                style="green",
            )
            console.print(f"   📝 Total memcells: {total_memcells}", style="blue")
            console.print(f"   ⏱️  Total time: {elapsed:.2f}s", style="yellow")
            if len(pending_conversations) > 0:
                console.print(
                    f"   🚀 Average per conversation: {elapsed/len(pending_conversations):.2f}s",
                    style="cyan",
                )
            console.print("=" * 60, style="dim")

        # ========== Stage 2: Index Building ==========
        console.print(f"\n{'='*60}", style="bold cyan")
        console.print(f"Stage 2: Index Building", style="bold cyan")
        console.print(f"{'='*60}", style="bold cyan")

        # Call stage2 implementation to build indexes
        exp_config = self._convert_config_to_experiment_config()
        exp_config.num_conv = len(conversations)  # Set conversation count

        # Smart skip logic: check existing index files
        bm25_need_build = self._check_missing_indexes(
            index_dir=bm25_index_dir, num_conv=len(conversations), index_type="bm25"
        )

        emb_need_build = []
        use_hybrid = self.config.get("search", {}).get("use_hybrid_search", True)
        if use_hybrid:
            emb_need_build = self._check_missing_indexes(
                index_dir=emb_index_dir,
                num_conv=len(conversations),
                index_type="embedding",
            )

        # Statistics
        total_convs = len(conversations)
        bm25_to_build = len(bm25_need_build)
        emb_to_build = len(emb_need_build) if use_hybrid else 0

        console.print(f"\n📊 Index Building Statistics:")
        console.print(f"   Total conversations: {total_convs}")
        console.print(
            f"   BM25 index: need to build {bm25_to_build}, existing {total_convs - bm25_to_build}"
        )
        if use_hybrid:
            console.print(
                f"   Embedding index: need to build {emb_to_build}, existing {total_convs - emb_to_build}"
            )

        # Build BM25 index
        if bm25_to_build > 0:
            console.print(
                f"\n🔨 Building BM25 index ({bm25_to_build} conversations)...",
                style="yellow",
            )
            stage2_index_building.build_bm25_index(
                config=exp_config, data_dir=memcells_dir, bm25_save_dir=bm25_index_dir
            )
            console.print("✅ BM25 index building completed", style="green")
        else:
            console.print("✅ All BM25 indexes exist, skipping build", style="green")

        # Build Embedding index (if enabled)
        if use_hybrid:
            if emb_to_build > 0:
                console.print(
                    f"\n🔨 Building Embedding index ({emb_to_build} conversations)...",
                    style="yellow",
                )
                await stage2_index_building.build_emb_index(
                    config=exp_config, data_dir=memcells_dir, emb_save_dir=emb_index_dir
                )
                console.print("✅ Embedding index building completed", style="green")
            else:
                console.print(
                    "✅ All Embedding indexes exist, skipping build", style="green"
                )

        # Build scene index from cluster data (for agentic retrieval mode)
        if exp_config.enable_scene_retrieval and exp_config.enable_clustering:
            scenes_dir = output_dir / "scenes"
            scenes_dir.mkdir(parents=True, exist_ok=True)
            console.print(
                f"\n🔨 Building Scene index from cluster data...",
                style="yellow",
            )
            stage2_index_building.build_scene_index(
                config=exp_config,
                data_dir=memcells_dir,
                cluster_dir=memcells_dir / "clusters",
                scene_save_dir=scenes_dir,
            )
            console.print("✅ Scene index building completed", style="green")

        # Build scene-based profiles (PersonaMem-style, time-ordered by scene)
        if (
            exp_config.enable_profile_extraction
            and exp_config.enable_clustering
            and getattr(exp_config, "profile_extraction_mode", "conversation") == "scene"
        ):
            console.print(f"\n{'='*60}", style="bold cyan")
            console.print(f"Stage 2.5: Scene Profile Extraction", style="bold cyan")
            console.print(f"{'='*60}", style="bold cyan")

            scene_profile_dir = output_dir / "scene_profiles"
            profile_output_dir = output_dir / "profiles"
            scene_profile_dir.mkdir(parents=True, exist_ok=True)
            profile_output_dir.mkdir(parents=True, exist_ok=True)

            await stage2_index_building.build_scene_profiles(
                config=exp_config,
                data_dir=memcells_dir,
                cluster_dir=memcells_dir / "clusters",
                scene_profile_dir=scene_profile_dir,
                profile_output_dir=profile_output_dir,
            )

        # ========== Plan A: Return index metadata (lazy loading) ==========
        # Don't load indexes into memory, only return paths and metadata
        index_metadata = {
            "type": "lazy_load",  # Mark as lazy loading
            "memcells_dir": str(memcells_dir),
            "bm25_index_dir": str(bm25_index_dir),
            "emb_index_dir": str(emb_index_dir),
            "scenes_dir": str(output_dir / "scenes"),
            "conversation_ids": [conv.conversation_id for conv in conversations],
            "use_hybrid_search": use_hybrid,
            "total_conversations": len(conversations),
        }

        console.print(f"\n{'='*60}", style="dim")
        console.print(f"✅ Add stage completed", style="bold green")
        console.print(f"   📁 MemCells: {memcells_dir}", style="dim")
        console.print(f"   📁 BM25 index: {bm25_index_dir}", style="dim")
        if use_hybrid:
            console.print(f"   📁 Embedding index: {emb_index_dir}", style="dim")
        console.print(
            f"   💡 Using lazy loading strategy (memory-friendly)", style="cyan"
        )
        console.print(f"{'='*60}\n", style="dim")

        return index_metadata

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """
        Search stage: Retrieve relevant MemCells.

        Lazy loading: Load indexes from files on demand (memory-friendly).
        """
        # Lazy loading - read indexes from files
        bm25_index_dir = Path(index["bm25_index_dir"])
        emb_index_dir = Path(index["emb_index_dir"])

        # Extract numeric index from conversation_id to find index files
        # Example: conversation_id = "locomo_0" -> conv_index = "0"
        conv_index = self._extract_conv_index(conversation_id)

        # Load BM25 index on demand (using numeric index)
        bm25_file = bm25_index_dir / f"bm25_index_conv_{conv_index}.pkl"
        if not bm25_file.exists():
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"error": f"BM25 index not found: {bm25_file.name}"},
            )

        with open(bm25_file, "rb") as f:
            bm25_index_data = pickle.load(f)

        bm25 = bm25_index_data.get("bm25")
        docs = bm25_index_data.get("docs")

        # Load Embedding index on demand (using numeric index)
        emb_index = None
        if index.get("use_hybrid_search"):
            emb_file = emb_index_dir / f"embedding_index_conv_{conv_index}.pkl"
            if emb_file.exists():
                with open(emb_file, "rb") as f:
                    emb_index = pickle.load(f)

        # Call stage3 retrieval implementation
        search_config = self.config.get("search", {})
        retrieval_mode = search_config.get("mode", "agentic")

        exp_config = self._convert_config_to_experiment_config()
        # Get correct format llm_config from exp_config
        llm_config = exp_config.llm_config.get(exp_config.llm_service, {})

        if retrieval_mode == "agentic":
            # 默认 Agentic 两级场景检索 (93% on LoCoMo)
            from evaluation.src.adapters.evermemos import scene_retrieval

            # Load fact_to_doc_idx from BM25 index (needed for BM25 MaxSim aggregation)
            fact_to_doc_idx = bm25_index_data.get("fact_to_doc_idx")

            # Load or build scene_index for this conversation
            scene_index_data = self._load_or_build_scene_index(
                conv_index=conv_index,
                index=index,
                exp_config=exp_config,
            )
            top_results, metadata = await scene_retrieval.agentic_retrieval(
                query=query,
                scene_index=scene_index_data,
                emb_index=emb_index,
                docs=docs,
                config=exp_config,
                bm25=bm25,
                llm_provider=self.llm_provider,
                llm_config=llm_config,
                fact_to_doc_idx=fact_to_doc_idx,
                question_date=kwargs.get("question_date"),
            )
        elif retrieval_mode == "lightweight":
            # Lightweight BM25-only retrieval (fastest, no LLM calls)
            top_results, metadata = await stage3_memory_retrivel.lightweight_retrieval(
                query=query,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                config=exp_config,
            )
        else:
            raise ValueError(
                f"Unsupported retrieval mode: '{retrieval_mode}'. "
                f"Supported modes: 'agentic', 'lightweight'"
            )

        # Get response_top_k from config (use early for consistency)
        response_top_k = exp_config.response_top_k

        # If Round 2 (insufficient), use round2_response_top_k for more results
        is_multi_round = metadata.get("is_multi_round", False)
        if is_multi_round:
            actual_top_k = getattr(exp_config, 'round2_response_top_k', response_top_k)
        else:
            actual_top_k = response_top_k

        # Convert to evaluation framework format
        results = []
        for doc, score in top_results[:actual_top_k]:
            results.append(
                {
                    "content": doc.get("episode", ""),
                    "score": float(score),
                    "metadata": {
                        "subject": doc.get("subject", ""),
                        "summary": doc.get("summary", ""),
                    },
                }
            )

        # Build formatted_context
        formatted_context = ""
        conversation = kwargs.get("conversation")
        if conversation and top_results:
            # Get speaker information
            speaker_a = conversation.metadata.get("speaker_a", "Speaker A")
            speaker_b = conversation.metadata.get("speaker_b", "Speaker B")

            # Build context using actual_top_k (Round 2 may use more)
            retrieved_docs_text = []
            for doc, score in top_results[:actual_top_k]:
                subject = doc.get('subject', 'N/A')
                episode = doc.get('episode', 'N/A')
                doc_text = f"{subject}: {episode}\n---"
                retrieved_docs_text.append(doc_text)

            speaker_memories = "\n\n".join(retrieved_docs_text)

            TEMPLATE = """Episodes memories for conversation between {speaker_1} and {speaker_2}:

    {speaker_memories}
"""
            formatted_context = TEMPLATE.format(
                speaker_1=speaker_a,
                speaker_2=speaker_b,
                speaker_memories=speaker_memories,
            )

        # Add formatted_context to metadata
        metadata["formatted_context"] = formatted_context

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata=metadata,
        )

    async def answer(self, query: str, context: str, **kwargs) -> str:
        """
        Answer stage: Generate answer.

        Calls stage4_response.py implementation.
        """
        exp_config = self._convert_config_to_experiment_config()

        options = kwargs.get("options") or {}
        dataset_name = kwargs.get("dataset_name") or ""
        use_mcq_prompt = bool(options) and dataset_name in {
            "personamem",
            "personamemv2",
        }

        # Load profile if enabled
        profile_text = ""
        if exp_config.use_profile_in_answer:
            conversation_id = kwargs.get("conversation_id", "")
            if conversation_id:
                profile_text = self._load_profile_text(conversation_id)

        # Determine episode context based on config
        episode_context = context if exp_config.use_episode_in_answer else "(none)"

        if use_mcq_prompt:
            options_text = self._format_options(options)
            profile_block = (
                f"User Profile:\n{profile_text}" if profile_text else "User Profile:\n(none)"
            )
            answer = await stage4_response.locomo_response(
                llm_provider=self.llm_provider,
                context=episode_context,
                question=query,
                experiment_config=exp_config,
                prompt_template=PERSONAMEM_MCQ_PROMPT,
                options_text=options_text,
                profile_text=profile_block,
            )
        else:
            if profile_text:
                episode_context = episode_context + "\n\nUser Profile:\n\n" + profile_text

            # Use LME-specific prompt when question_date is available (LongMemEval temporal reasoning)
            question_date = kwargs.get("question_date")
            if question_date:
                lme_template = get_lme_prompt_template(current_time=question_date)
                answer = await stage4_response.locomo_response(
                    llm_provider=self.llm_provider,
                    context=episode_context,
                    question=query,
                    experiment_config=exp_config,
                    prompt_template=lme_template,
                )
            else:
                answer = await stage4_response.locomo_response(
                    llm_provider=self.llm_provider,
                    context=episode_context,
                    question=query,
                    experiment_config=exp_config,
                )

        return answer

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "EverMemOS",
            "version": "1.0",
            "description": "EverMemOS memory system with agentic retrieval",
            "adapter": "Adapter connecting framework to EverMemOS implementation",
        }

    def _convert_config_to_experiment_config(self):
        """
        Convert evaluation framework config to ExperimentConfig format.
        """
        from evaluation.src.adapters.evermemos.config import ExperimentConfig
        import os

        exp_config = ExperimentConfig()

        # Map LLM configuration: convert YAML llm to ExperimentConfig llm_config format
        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "openai")

        exp_config.llm_service = provider
        exp_config.llm_config = {
            provider: {
                "llm_provider": provider,
                "model": llm_cfg.get("model", "gpt-4o-mini"),
                "api_key": llm_cfg.get("api_key") or os.getenv("LLM_API_KEY", ""),
                "base_url": llm_cfg.get("base_url")
                or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                "temperature": llm_cfg.get("temperature", 0.3),
                "max_tokens": llm_cfg.get("max_tokens", 32768),
            }
        }

        # Map Add stage configuration (only override explicitly specified in YAML)
        add_config = self.config.get("add", {})
        if "enable_foresight_extraction" in add_config:
            exp_config.enable_foresight_extraction = add_config[
                "enable_foresight_extraction"
            ]
        if "enable_clustering" in add_config:
            exp_config.enable_clustering = add_config["enable_clustering"]
        if "cluster_similarity_threshold" in add_config:
            exp_config.cluster_similarity_threshold = add_config["cluster_similarity_threshold"]
        if "cluster_max_time_gap_days" in add_config:
            exp_config.cluster_max_time_gap_days = add_config["cluster_max_time_gap_days"]
        if "enable_profile_extraction" in add_config:
            exp_config.enable_profile_extraction = add_config[
                "enable_profile_extraction"
            ]
        if "profile_scenario" in add_config:
            exp_config.profile_scenario = add_config["profile_scenario"]
        if "profile_min_confidence" in add_config:
            exp_config.profile_min_confidence = add_config["profile_min_confidence"]
        if "profile_min_memcells" in add_config:
            exp_config.profile_min_memcells = add_config["profile_min_memcells"]
        if "profile_extraction_mode" in add_config:
            exp_config.profile_extraction_mode = add_config[
                "profile_extraction_mode"
            ]
        if "profile_life_max_items" in add_config:
            exp_config.profile_life_max_items = add_config[
                "profile_life_max_items"
            ]

        # Map Search stage configuration (only override explicitly specified in YAML)
        search_config = self.config.get("search", {})
        if "mode" in search_config:
            exp_config.retrieval_mode = search_config["mode"]

        # Map Answer stage configuration
        answer_config = self.config.get("answer", {})
        if "use_profile_in_answer" in answer_config:
            exp_config.use_profile_in_answer = answer_config["use_profile_in_answer"]
        if "use_episode_in_answer" in answer_config:
            exp_config.use_episode_in_answer = answer_config["use_episode_in_answer"]
        if "use_profile_classifier" in answer_config:
            exp_config.use_profile_classifier = answer_config["use_profile_classifier"]

        return exp_config

    def build_lazy_index(
        self, conversations: List[Conversation], output_dir: Any
    ) -> Dict[str, Any]:
        """
        Build EverMemOS lazy-load index metadata.

        EverMemOS specifics:
        - Local indexes (memcells, bm25, embeddings)
        - Lazy loading (only save metadata, don't load actual index files)

        Args:
            conversations: Conversation list
            output_dir: Output directory

        Returns:
            Index metadata dict
        """
        return {
            "type": "lazy_load",
            "memcells_dir": str(output_dir / "memcells"),
            "bm25_index_dir": str(output_dir / "bm25_index"),
            "emb_index_dir": str(output_dir / "vectors"),
            "scenes_dir": str(output_dir / "scenes"),
            "conversation_ids": [conv.conversation_id for conv in conversations],
            "use_hybrid_search": True,
            "total_conversations": len(conversations),
        }

    def _load_or_build_scene_index(
        self, conv_index: str, index: Dict[str, Any], exp_config
    ) -> Dict[str, Any]:
        """
        Load scene_index from file, or build all scene indexes on first miss.

        On first call where scene_index files don't exist, builds indexes
        for ALL conversations at once (efficient). Subsequent calls just load from file.

        Args:
            conv_index: Conversation numeric index (e.g., "0")
            index: Lazy-load index metadata
            exp_config: Experiment configuration

        Returns:
            Scene index dict with 'scenes', 'memcell_to_scene', etc.
        """
        # Derive scenes_dir: from index metadata, or fallback to sibling of memcells_dir
        scenes_dir_str = index.get("scenes_dir")
        if not scenes_dir_str:
            scenes_dir_str = str(Path(index["memcells_dir"]).parent / "scenes")
        scenes_dir = Path(scenes_dir_str)
        scene_index_path = scenes_dir / f"scene_index_conv_{conv_index}.pkl"

        # Try loading existing scene_index
        if scene_index_path.exists():
            with open(scene_index_path, "rb") as f:
                return pickle.load(f)

        # Build scene_index for ALL conversations from cluster data
        memcells_dir = Path(index["memcells_dir"])
        cluster_dir = memcells_dir / "clusters"

        print(f"Scene index not found, building for all conversations from cluster data...")
        scenes_dir.mkdir(parents=True, exist_ok=True)

        stage2_index_building.build_scene_index(
            config=exp_config,
            data_dir=memcells_dir,
            cluster_dir=cluster_dir,
            scene_save_dir=scenes_dir,
        )

        # Load the newly built scene_index
        if scene_index_path.exists():
            with open(scene_index_path, "rb") as f:
                return pickle.load(f)

        raise FileNotFoundError(
            f"Failed to build scene index for conv_{conv_index}. "
            f"Check that cluster data exists at {cluster_dir}/conv_{conv_index}/"
        )
