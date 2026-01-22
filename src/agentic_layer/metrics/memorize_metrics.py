"""
Memorize Pipeline Metrics

Metrics for monitoring the memory ingestion (add memory) pipeline including:
- Request processing
- Stage-level performance (conversion, save_logs, memorize_process)
- Memory extraction statistics

Usage:
    from agentic_layer.metrics.memorize_metrics import (
        record_memorize_request,
        record_memorize_stage,
        record_memorize_error,
    )
    
    # Record successful memorize request
    record_memorize_request(
        status='success',
        duration_seconds=0.5,
        memories_extracted=3,
    )
    
    # Record stage duration
    record_memorize_stage(
        stage='conversion',
        duration_seconds=0.01,
    )
"""

from core.observation.metrics import Counter, Histogram, HistogramBuckets


# ============================================================
# Counter Metrics
# ============================================================

MEMORIZE_REQUESTS_TOTAL = Counter(
    name='memorize_requests_total',
    description='Total number of memorize requests',
    labelnames=['status'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Memorize requests counter

Labels:
- status: success, error, accumulated, extracted
  - success: Request processed successfully (with or without memory extraction)
  - error: Request failed
  - accumulated: No memory extracted, message queued
  - extracted: Memories extracted successfully
"""


MEMORIZE_ERRORS_TOTAL = Counter(
    name='memorize_errors_total',
    description='Total number of memorize errors',
    labelnames=['stage', 'error_type'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Memorize errors counter

Labels:
- stage: conversion, save_logs, memorize_process
- error_type: validation_error, timeout, connection_error, unknown
"""


# ============================================================
# Histogram Metrics
# ============================================================

MEMORIZE_DURATION_SECONDS = Histogram(
    name='memorize_duration_seconds',
    description='End-to-end duration of memorize operation in seconds',
    labelnames=['status'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=HistogramBuckets.API_CALL,  # 10ms - 30s for API calls
)
"""
End-to-end memorize duration histogram

Labels:
- status: success, error

Buckets: 10ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s, 30s
"""


MEMORIZE_STAGE_DURATION_SECONDS = Histogram(
    name='memorize_stage_duration_seconds',
    description='Duration of individual memorize stages in seconds',
    labelnames=['stage'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=HistogramBuckets.DATABASE,  # 1ms - 5s for database operations
)
"""
Stage-specific duration histogram

Labels:
- stage: conversion, save_logs, memorize_process, boundary_detection, memory_extraction

Buckets: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s
"""


MEMORIZE_MEMORIES_EXTRACTED = Histogram(
    name='memorize_memories_extracted',
    description='Number of memories extracted per memorize request',
    labelnames=[],
    namespace='evermemos',
    subsystem='agentic',
    buckets=(0, 1, 2, 3, 5, 10, 20, 50, 100),
)
"""
Memories extracted per request histogram

Buckets: 0, 1, 2, 3, 5, 10, 20, 50, 100 memories
"""


MEMORIZE_MESSAGES_TOTAL = Counter(
    name='memorize_messages_total',
    description='Total number of messages processed for memorization',
    labelnames=['status'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Messages processed counter

Labels:
- status: received, saved, processed
"""


# ============================================================
# Boundary Detection Metrics
# ============================================================

BOUNDARY_DETECTION_TOTAL = Counter(
    name='boundary_detection_total',
    description='Total number of boundary detection results',
    labelnames=['result', 'trigger_type'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Boundary detection results counter

Labels:
- result: should_end, should_wait, error, force_split
- trigger_type: llm, token_limit, message_limit, first_message
"""


BOUNDARY_DETECTION_DURATION_SECONDS = Histogram(
    name='boundary_detection_duration_seconds',
    description='Duration of boundary detection in seconds',
    labelnames=['trigger_type'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=HistogramBuckets.ML_INFERENCE,  # LLM inference buckets
)
"""
Boundary detection duration histogram

Labels:
- trigger_type: llm, token_limit, message_limit, first_message

Buckets: 10ms - 10s for ML inference
"""


MEMCELL_EXTRACTED_TOTAL = Counter(
    name='memcell_extracted_total',
    description='Total number of MemCells extracted',
    labelnames=['trigger_type'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
MemCell extraction counter

Labels:
- trigger_type: llm, token_limit, message_limit
"""


# ============================================================
# Memory Extraction Metrics
# ============================================================

MEMORY_EXTRACTION_STAGE_DURATION_SECONDS = Histogram(
    name='memory_extraction_stage_duration_seconds',
    description='Duration of individual memory extraction stages in seconds',
    labelnames=['stage'],
    namespace='evermemos',
    subsystem='agentic',
    buckets=HistogramBuckets.ML_INFERENCE,  # LLM inference buckets
)
"""
Memory extraction stage duration histogram

Labels:
- stage: init_state, extract_episodes, extract_foresights, extract_event_logs, 
         update_memcell_cluster, process_memories

Buckets: 10ms - 10s for ML inference
"""


MEMORY_EXTRACTED_TOTAL = Counter(
    name='memory_extracted_total',
    description='Total number of memories extracted by type',
    labelnames=['memory_type'],
    namespace='evermemos',
    subsystem='agentic',
)
"""
Memory extraction counter by type

Labels:
- memory_type: episode, foresight, event_log
"""


EXTRACT_MEMORY_REQUESTS_TOTAL = Counter(
    name='extract_memory_requests_total',
    description='Total number of extract_memory calls by memory type',
    labelnames=['memory_type', 'status'],
    namespace='evermemos',
    subsystem='memory_layer',
)
"""
extract_memory call counter

Labels:
- memory_type: episodic_memory, foresight, event_log, profile, group_profile
- status: success, error, empty_result
"""


EXTRACT_MEMORY_DURATION_SECONDS = Histogram(
    name='extract_memory_duration_seconds',
    description='Duration of extract_memory calls by memory type in seconds',
    labelnames=['memory_type'],
    namespace='evermemos',
    subsystem='memory_layer',
    buckets=HistogramBuckets.ML_INFERENCE,  # LLM inference buckets
)
"""
extract_memory duration histogram

Labels:
- memory_type: episodic_memory, foresight, event_log, profile, group_profile

Buckets: 10ms - 10s for ML inference
"""


# ============================================================
# Helper Functions
# ============================================================

def record_memorize_request(
    status: str,
    duration_seconds: float,
    memories_extracted: int = 0,
) -> None:
    """
    Helper function to record memorize request metrics
    
    Args:
        status: Request status (success, error, accumulated, extracted)
        duration_seconds: Total operation duration in seconds
        memories_extracted: Number of memories extracted (0 if accumulated)
    
    Example:
        record_memorize_request(
            status='extracted',
            duration_seconds=0.5,
            memories_extracted=3,
        )
    """
    # Counter
    MEMORIZE_REQUESTS_TOTAL.labels(status=status).inc()
    
    # Duration histogram (use simplified status for duration)
    duration_status = 'success' if status in ('success', 'accumulated', 'extracted') else 'error'
    MEMORIZE_DURATION_SECONDS.labels(status=duration_status).observe(duration_seconds)
    
    # Memories extracted histogram
    MEMORIZE_MEMORIES_EXTRACTED.observe(memories_extracted)


def record_memorize_stage(
    stage: str,
    duration_seconds: float,
) -> None:
    """
    Helper function to record stage-specific duration
    
    Args:
        stage: Memorize stage (conversion, save_logs, memorize_process, 
               boundary_detection, memory_extraction)
        duration_seconds: Stage duration in seconds
    
    Example:
        record_memorize_stage(
            stage='conversion',
            duration_seconds=0.01,
        )
    """
    MEMORIZE_STAGE_DURATION_SECONDS.labels(stage=stage).observe(duration_seconds)


def record_memorize_error(
    stage: str,
    error_type: str,
) -> None:
    """
    Helper function to record memorize error
    
    Args:
        stage: Stage where error occurred (conversion, save_logs, memorize_process)
        error_type: Error type (validation_error, timeout, connection_error, unknown)
    
    Example:
        record_memorize_error(
            stage='conversion',
            error_type='validation_error',
        )
    """
    MEMORIZE_ERRORS_TOTAL.labels(stage=stage, error_type=error_type).inc()


def record_memorize_message(status: str, count: int = 1) -> None:
    """
    Helper function to record message processing
    
    Args:
        status: Message status (received, saved, processed)
        count: Number of messages
    
    Example:
        record_memorize_message(status='received', count=1)
    """
    MEMORIZE_MESSAGES_TOTAL.labels(status=status).inc(count)


def classify_memorize_error(error: Exception) -> str:
    """
    Classify error type for metrics
    
    Args:
        error: Exception instance
    
    Returns:
        Error type string for metrics label
    """
    error_type = type(error).__name__.lower()
    
    if 'timeout' in error_type:
        return 'timeout'
    elif 'validation' in error_type or 'value' in error_type:
        return 'validation_error'
    elif 'connection' in error_type or 'network' in error_type:
        return 'connection_error'
    elif 'http' in error_type:
        return 'http_error'
    else:
        return 'unknown'


def record_boundary_detection(
    result: str,
    trigger_type: str,
    duration_seconds: float,
) -> None:
    """
    Helper function to record boundary detection metrics
    
    Args:
        result: Detection result (should_end, should_wait, error, force_split)
        trigger_type: What triggered the detection (llm, token_limit, message_limit, first_message)
        duration_seconds: Detection duration in seconds
    
    Example:
        record_boundary_detection(
            result='should_end',
            trigger_type='llm',
            duration_seconds=1.5,
        )
    """
    BOUNDARY_DETECTION_TOTAL.labels(result=result, trigger_type=trigger_type).inc()
    BOUNDARY_DETECTION_DURATION_SECONDS.labels(trigger_type=trigger_type).observe(duration_seconds)


def record_memcell_extracted(trigger_type: str) -> None:
    """
    Helper function to record MemCell extraction
    
    Args:
        trigger_type: What triggered the extraction (llm, token_limit, message_limit)
    
    Example:
        record_memcell_extracted(trigger_type='llm')
    """
    MEMCELL_EXTRACTED_TOTAL.labels(trigger_type=trigger_type).inc()


def record_extraction_stage(stage: str, duration_seconds: float) -> None:
    """
    Helper function to record memory extraction stage duration
    
    Args:
        stage: Extraction stage (init_state, extract_episodes, extract_foresights, 
               extract_event_logs, update_memcell_cluster, process_memories)
        duration_seconds: Stage duration in seconds
    
    Example:
        record_extraction_stage(
            stage='extract_episodes',
            duration_seconds=2.5,
        )
    """
    MEMORY_EXTRACTION_STAGE_DURATION_SECONDS.labels(stage=stage).observe(duration_seconds)


def record_memory_extracted(memory_type: str, count: int = 1) -> None:
    """
    Helper function to record extracted memory count by type
    
    Args:
        memory_type: Memory type (episode, foresight, event_log)
        count: Number of memories extracted
    
    Example:
        record_memory_extracted(memory_type='episode', count=3)
    """
    MEMORY_EXTRACTED_TOTAL.labels(memory_type=memory_type).inc(count)


def record_extract_memory_call(
    memory_type: str,
    status: str,
    duration_seconds: float,
) -> None:
    """
    Helper function to record extract_memory call metrics
    
    Args:
        memory_type: Memory type (episodic_memory, foresight, event_log, profile, group_profile)
        status: Call status (success, error, empty_result)
        duration_seconds: Call duration in seconds
    
    Example:
        record_extract_memory_call(
            memory_type='episodic_memory',
            status='success',
            duration_seconds=2.5,
        )
    """
    EXTRACT_MEMORY_REQUESTS_TOTAL.labels(memory_type=memory_type, status=status).inc()
    EXTRACT_MEMORY_DURATION_SECONDS.labels(memory_type=memory_type).observe(duration_seconds)
