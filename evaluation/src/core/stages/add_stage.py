"""
Add stage - ingest conversation data and build index.
"""
from pathlib import Path
from typing import List, Any, Optional
from logging import Logger

from evaluation.src.core.data_models import Conversation, Dataset
from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.utils.checkpoint import CheckpointManager
from evaluation.src.core.benchmark_context import LatencyRecorder, NULL_RECORDER


async def run_add_stage(
    adapter: BaseAdapter,
    dataset: Dataset,
    output_dir: Path,
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
    console: Any,
    completed_stages: set,
    latency_recorder: Optional[LatencyRecorder] = None,
) -> dict:
    """
    Execute Add stage.

    Args:
        adapter: System adapter
        dataset: Standard format dataset
        output_dir: Output directory
        checkpoint_manager: Checkpoint manager for resume
        logger: Logger
        console: Console object
        completed_stages: Set of completed stages
        latency_recorder: Optional harness-level latency recorder
            (see docs/latency-alignment.md / Phase 1).

    Returns:
        Dict containing index
    """
    recorder = latency_recorder or NULL_RECORDER

    # Adapter.add() is still a batch operation over all conversations.
    # Phase 3 of the latency-alignment plan un-batches this so each
    # conversation gets its own Layer-1 record; for now the whole-batch
    # wall_ms is recorded under unit_id="all". Per-conversation latency
    # remains available via adapter-written add_summary.json (aggregated
    # by diagnostics.py), so no signal is lost in the interim.
    async with recorder.measure("add", unit_id="all") as ctx:
        index = await adapter.add(
            conversations=dataset.conversations,
            output_dir=output_dir,
            checkpoint_manager=checkpoint_manager,
            benchmark_ctx=ctx,
        )

    # Index metadata (lazy load, no need to persist)
    logger.info("✅ Stage 1 completed")

    # Save checkpoint
    completed_stages.add("add")
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(completed_stages)

    return {"index": index}

