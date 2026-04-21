"""
Answer stage - generate answers.
"""
import asyncio
import time
from typing import List, Optional
from logging import Logger
from tqdm import tqdm

from evaluation.src.core.data_models import QAPair, SearchResult, AnswerResult
from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.utils.checkpoint import CheckpointManager
from evaluation.src.core.benchmark_context import (
    LatencyRecorder,
    NULL_RECORDER,
    OUTCOME_FAILED_OTHER,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
    max_retries_for,
)


# Tokenizer is loaded lazily so importing answer_stage stays cheap and the
# dependency on tiktoken is only paid when tests / pipelines actually call
# estimate_tokens().
_TOKEN_ENCODING = None


def estimate_tokens(text: str) -> int:
    """Rough token count for latency/context diagnostics.

    Uses tiktoken's o200k_base encoding (matches gpt-4o / gpt-4o-mini). Falls
    back to whitespace splitting if tiktoken is unavailable so the pipeline
    keeps working in stripped-down environments.
    """
    if not text:
        return 0
    global _TOKEN_ENCODING
    if _TOKEN_ENCODING is None:
        try:
            import tiktoken

            _TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")
        except Exception:
            _TOKEN_ENCODING = "fallback"
    if _TOKEN_ENCODING == "fallback":
        return len(text.split())
    return len(_TOKEN_ENCODING.encode(text))


def build_context(search_result: SearchResult) -> str:
    """
    Build context from search results.
    
    Prefer pre-formatted context (dual-speaker scenarios), else use simple numbering (single-speaker scenarios).
    
    Args:
        search_result: Search result
        
    Returns:
        Context string
    """
    # Prefer pre-formatted context (provided by adapter)
    formatted_context = search_result.retrieval_metadata.get("formatted_context", "")
    if formatted_context:
        return formatted_context
    
    # Single speaker scenario: simple formatting
    context_parts = []
    
    # Get top_k from retrieval_metadata, default to len(results) if not specified
    top_k = search_result.retrieval_metadata.get("top_k", len(search_result.results))
    
    # Add memory content (use top_k instead of hardcoded 10)
    for idx, result in enumerate(search_result.results[:top_k], 1):
        content = result.get("content", "")
        context_parts.append(f"{idx}. {content}")
    
    context = "\n\n".join(context_parts)
    
    # For systems supporting preferences (e.g., Memos), add formatted pref_string
    preferences = search_result.retrieval_metadata.get("preferences", {})
    pref_string = preferences.get("pref_string", "")
    
    if pref_string:
        context += "\n\n" + pref_string
    
    return context


async def run_answer_stage(
    adapter: BaseAdapter,
    qa_pairs: List[QAPair],
    search_results: List[SearchResult],
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
    latency_recorder: Optional[LatencyRecorder] = None,
) -> List[AnswerResult]:
    """
    Generate answers with fine-grained checkpointing.
    
    Save checkpoint every SAVE_INTERVAL questions.
    
    Args:
        adapter: System adapter
        qa_pairs: List of QA pairs
        search_results: List of search results
        checkpoint_manager: Checkpoint manager for resume
        logger: Logger
        
    Returns:
        List of answer results
    """
    print(f"\n{'='*60}")
    print(f"Stage 3/4: Answer")
    print(f"{'='*60}")
    
    SAVE_INTERVAL = 400  # Save every 400 tasks
    MAX_CONCURRENT = 50  # Max concurrency
    
    # Load fine-grained checkpoint
    all_answer_results = {}
    if checkpoint_manager:
        loaded_results = checkpoint_manager.load_answer_progress()
        # Convert to {question_id: AnswerResult} format
        for result in loaded_results.values():
            all_answer_results[result["question_id"]] = result
    
    total_qa_count = len(qa_pairs)
    processed_count = len(all_answer_results)
    
    print(f"Total questions: {total_qa_count}")
    if processed_count > 0:
        print(f"Already processed: {processed_count} questions (from checkpoint)")
        print(f"Remaining: {total_qa_count - processed_count} questions")
    
    # Pair qa with its search_result by question_id (stashed by
    # search_stage in retrieval_metadata) with positional fallback for
    # SearchResults that never saw search_stage. Shared helper keeps
    # the logic in lockstep with retrieval_metrics / content_overlap.
    from evaluation.src.metrics.pairing import pair_by_question_id

    search_by_id, _ = pair_by_question_id(qa_pairs, search_results)

    pending_tasks = []
    for qa in qa_pairs:
        if qa.question_id in all_answer_results:
            continue
        sr = search_by_id.get(qa.question_id)
        if sr is None:
            # No retrieval output for this question - build an empty one
            # so the stage proceeds (answer_stage is resilient to empty
            # context but we must not drop the qa).
            sr = SearchResult(
                query=qa.question,
                conversation_id=qa.metadata.get("conversation_id", ""),
                results=[],
                retrieval_metadata={"question_id": qa.question_id,
                                    "error": "no search_result for question_id"},
            )
        pending_tasks.append((qa, sr))
    
    if not pending_tasks:
        print(f"✅ All questions already processed!")
        # Convert to AnswerResult object list (original order)
        results = []
        for qa in qa_pairs:
            if qa.question_id in all_answer_results:
                result_dict = all_answer_results[qa.question_id]
                results.append(AnswerResult(
                    question_id=result_dict["question_id"],
                    question=result_dict["question"],
                    answer=result_dict["answer"],
                    golden_answer=result_dict["golden_answer"],
                    category=result_dict.get("category"),
                    conversation_id=result_dict.get("conversation_id", ""),
                    formatted_context=result_dict.get("formatted_context", ""),  # Load formatted_context
                    # search_results not loaded to save space
                ))
        return results
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    completed = processed_count
    failed = 0
    start_time = time.time()

    recorder = latency_recorder or NULL_RECORDER
    answer_max_retries = max_retries_for(recorder.retry_policy)
    
    # Use tqdm progress bar
    pbar = tqdm(
        total=total_qa_count,
        initial=processed_count,
        desc="💬 Answer Progress",
        unit="qa"
    )
    
    async def answer_single_with_tracking(qa, search_result):
        nonlocal completed, failed

        async with semaphore:
            context = ""
            context_chars = 0
            context_tokens = 0
            answer_latency_ms = None
            answer = "Error: Failed to generate answer"

            try:
                # Build context
                context = build_context(search_result)
                context_chars = len(context)
                context_tokens = estimate_tokens(context)

                # Detect multiple-choice and enhance question if needed
                query = qa.question
                if "all_options" in qa.metadata:
                    options = qa.metadata["all_options"]
                    options_text = "\n".join([f"{key} {value}" for key, value in options.items()])

                    # Integrate options and requirements into question
                    query = f"""{qa.question}

OPTIONS:
{options_text}

IMPORTANT: This is a multiple-choice question. You MUST analyze the context and select the BEST option. In your FINAL ANSWER, return ONLY the option letter like (a), (b), (c), or (d), nothing else."""

                # Call adapter's answer method with timeout and retry.
                # Every attempt is recorded to the LatencyRecorder so
                # Layer-1 four-view aggregation can distinguish clean
                # first-hit latency from retry-inflated wall time.
                # max_retries is set by the pipeline's retry_policy; the
                # strict_no_retry value of 1 disables retries entirely.
                max_retries = answer_max_retries
                timeout_seconds = 120.0  # 2 minutes timeout per attempt
                retry_wait_seconds = 2.0

                async with recorder.measure("answer", qa.question_id) as ctx:
                    for attempt in range(max_retries):
                        # Skip retries once the harness-level deadline
                        # elapsed so one adapter call can't burn all
                        # of another sample's budget. fallback=true
                        # excludes the sample from clean stats.
                        if attempt > 0 and ctx.deadline_exceeded():
                            tqdm.write(
                                f"  ⏹️  Answer deadline exceeded before attempt {attempt + 1}; "
                                f"stopping retries for {qa.question_id}."
                            )
                            ctx.record_fallback()
                            answer = "Error: deadline exceeded before retry"
                            failed += 1
                            break
                        t_start = time.perf_counter()
                        try:
                            answer = await asyncio.wait_for(
                                adapter.answer(
                                    query=query,
                                    context=context,
                                    conversation_id=search_result.conversation_id,
                                    question_id=qa.question_id,
                                    benchmark_ctx=ctx,
                                ),
                                timeout=timeout_seconds
                            )
                            answer_latency_ms = (time.perf_counter() - t_start) * 1000.0
                            ctx.record_attempt(attempt + 1, answer_latency_ms, OUTCOME_SUCCESS)
                            answer = answer.strip()
                            break  # Success, exit retry loop

                        except asyncio.TimeoutError:
                            attempt_ms = (time.perf_counter() - t_start) * 1000.0
                            is_last = attempt >= max_retries - 1
                            ctx.record_attempt(
                                attempt + 1, attempt_ms, OUTCOME_TIMEOUT,
                                wait_ms_before_next=(0.0 if is_last else retry_wait_seconds * 1000.0),
                            )
                            if not is_last:
                                tqdm.write(f"  ⏱️  Timeout ({timeout_seconds}s) for {qa.question_id}, retry {attempt + 1}/{max_retries}...")
                                await asyncio.sleep(retry_wait_seconds)
                            else:
                                tqdm.write(f"  ❌ Timeout after {max_retries} attempts for {qa.question_id}: {qa.question[:50]}...")
                                ctx.record_fallback()
                                answer = "Error: Answer generation timeout after retries"
                                failed += 1

            except Exception as e:
                tqdm.write(f"  ⚠️ Answer generation failed for {qa.question_id}: {e}")
                answer = "Error: Failed to generate answer"
                failed += 1

            retrieval_meta = search_result.retrieval_metadata or {}
            metadata = {
                **qa.metadata,
                "answer_latency_ms": answer_latency_ms,
                "final_context_chars": context_chars,
                "final_context_tokens": context_tokens,
                "retrieval_latency_ms": retrieval_meta.get("retrieval_latency_ms"),
                "retrieval_route": retrieval_meta.get("retrieval_route"),
                "backend_mode": retrieval_meta.get("backend_mode"),
            }

            result = AnswerResult(
                question_id=qa.question_id,
                question=qa.question,
                answer=answer,
                golden_answer=qa.answer,
                category=qa.category,
                conversation_id=search_result.conversation_id,
                formatted_context=context,  # Save actual context used
                metadata=metadata,
            )

            # Save result
            all_answer_results[qa.question_id] = {
                "question_id": result.question_id,
                "question": result.question,
                "answer": result.answer,
                "golden_answer": result.golden_answer,
                "category": result.category,
                "conversation_id": result.conversation_id,
                "formatted_context": result.formatted_context,  # Save formatted_context
                "metadata": result.metadata,  # Save metadata (contains all_options + latency)
            }
            
            completed += 1
            pbar.update(1)  # Update progress bar
            
            # Save checkpoint periodically
            if checkpoint_manager and (completed % SAVE_INTERVAL == 0 or completed == total_qa_count):
                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                eta = (total_qa_count - completed) / speed if speed > 0 else 0
                
                tqdm.write(f"Progress: {completed}/{total_qa_count} ({completed/total_qa_count*100:.1f}%) | "
                          f"Speed: {speed:.1f} qa/s | Failed: {failed} | ETA: {eta/60:.1f} min")
                
                checkpoint_manager.save_answer_progress(all_answer_results, completed, total_qa_count)
            
            return result
    
    # Create all pending tasks
    tasks = [
        answer_single_with_tracking(qa, sr)
        for qa, sr in pending_tasks
    ]
    
    # Execute concurrently
    await asyncio.gather(*tasks)
    
    # Close progress bar
    pbar.close()
    
    # Statistics
    elapsed_time = time.time() - start_time
    success_rate = (completed - failed) / completed * 100 if completed > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"✅ All responses generated!")
    print(f"   - Total questions: {total_qa_count}")
    print(f"   - Successful: {completed - failed}")
    print(f"   - Failed: {failed}")
    print(f"   - Success rate: {success_rate:.1f}%")
    print(f"   - Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")
    print(f"   - Average speed: {total_qa_count/elapsed_time:.1f} qa/s")
    print(f"{'='*60}\n")
    
    # Delete fine-grained checkpoints after completion
    if checkpoint_manager:
        checkpoint_manager.delete_answer_checkpoints()
    
    # Convert to AnswerResult object list (original order)
    results = []
    for qa in qa_pairs:
        if qa.question_id in all_answer_results:
            result_dict = all_answer_results[qa.question_id]
            results.append(AnswerResult(
                question_id=result_dict["question_id"],
                question=result_dict["question"],
                answer=result_dict["answer"],
                golden_answer=result_dict["golden_answer"],
                category=result_dict.get("category"),
                conversation_id=result_dict.get("conversation_id", ""),
                formatted_context=result_dict.get("formatted_context", ""),
                search_results=result_dict.get("search_results", []),
                metadata=result_dict.get("metadata", {}),  # Restore metadata
            ))
    
    return results

