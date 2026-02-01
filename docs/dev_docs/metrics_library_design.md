# Metrics Library Design

## 🎯 Design Goals

1. **Business isolation from third-party dependencies**: Business code only references its own metrics library, no direct dependency on `prometheus_client`
2. **Lightweight auto-refresh**: Each Gauge instance manages its own refresh task, no global scheduler needed
3. **Unified inheritance pattern**: All Gauges inherit from BaseGauge and override the refresh method
4. **Unified interface**: Counter, Histogram, and Gauge use consistent wrappers

---

## 📐 Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              src/core/metrics/ (wrapper layer)               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Counter   │  │  Histogram  │  │  BaseGauge  │         │
│  │  (wrapper)  │  │  (wrapper)  │  │(wrapper+ref)│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                  │                │
│         └────────────────┴──────────────────┘                │
│                          │                                   │
│                 wraps prometheus_client                      │
└─────────────────────────────────────────────────────────────┘
                          ↑ imports
┌─────────────────────────────────────────────────────────────┐
│           Business code (only imports core.metrics)          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  from core.metrics import Counter, Histogram, BaseGauge     │
│                                                               │
│  # Unified inheritance pattern                               │
│  class QueueSizeGauge(BaseGauge):                           │
│      def __init__(self, queue):                             │
│          super().__init__('queue_size', 'Queue size')       │
│          self.queue = queue                                 │
│                                                               │
│      def refresh(self, labels: dict) -> float:              │
│          return self.queue.qsize()                          │
│                                                               │
│  # Usage                                                     │
│  gauge = QueueSizeGauge(queue)                              │
│  gauge.labels(name='main').start_refresh()  # default 5 sec │
│  # or manual set                                             │
│  gauge.labels(name='main').set(42)                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Core Implementation

### 1. Counter Wrapper

**File: `src/core/metrics/counter.py`**

```python
"""
Counter Wrapper

Provides a unified Counter interface, isolating prometheus_client
"""
from prometheus_client import Counter as PrometheusCounter
from typing import Sequence
from .registry import get_metrics_registry


class Counter:
    """
    Counter metric wrapper

    Features:
    - Monotonically increasing counter
    - Suitable for total requests, total errors, etc.
    - Business code doesn't need to import prometheus_client directly

    Usage example:
        from core.metrics import Counter

        requests_total = Counter(
            name='http_requests_total',
            description='Total HTTP requests',
            labelnames=['method', 'path', 'status']
        )

        # Usage
        requests_total.labels(method='GET', path='/api', status='200').inc()
    """

    def __init__(
        self,
        name: str,
        description: str,
        labelnames: Sequence[str] = (),
        namespace: str = '',
        subsystem: str = '',
        unit: str = '',
    ):
        """
        Args:
            name: Metric name
            description: Metric description
            labelnames: List of label names
            namespace: Namespace (optional)
            subsystem: Subsystem (optional)
            unit: Unit (optional)
        """
        registry = get_metrics_registry()
        
        self._counter = PrometheusCounter(
            name=name,
            documentation=description,
            labelnames=labelnames,
            namespace=namespace,
            subsystem=subsystem,
            unit=unit,
            registry=registry,
        )
    
    def labels(self, **labels):
        """
        Return a labeled Counter

        Returns:
            LabeledCounter instance
        """
        labeled = self._counter.labels(**labels)
        return LabeledCounter(labeled)
    
    def inc(self, amount: float = 1) -> None:
        """
        Increment counter (unlabeled version)

        Args:
            amount: Increment amount, default 1
        """
        self._counter.inc(amount)


class LabeledCounter:
    """Labeled Counter"""

    def __init__(self, labeled_counter):
        self._counter = labeled_counter

    def inc(self, amount: float = 1) -> None:
        """
        Increment counter

        Args:
            amount: Increment amount, default 1
        """
        self._counter.inc(amount)
```

---

### 2. Histogram Wrapper

**File: `src/core/metrics/histogram.py`**

```python
"""
Histogram Wrapper

Provides a unified Histogram interface, isolating prometheus_client
"""
from prometheus_client import Histogram as PrometheusHistogram
from typing import Sequence
from .registry import get_metrics_registry


class Histogram:
    """
    Histogram metric wrapper

    Features:
    - Distribution statistics of observed values
    - Suitable for latency, size, and other distribution data
    - Automatically calculates quantiles, mean, and sum

    Usage example:
        from core.metrics import Histogram
        
        request_duration = Histogram(
            name='http_request_duration_seconds',
            description='HTTP request duration',
            labelnames=['method', 'path'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
        )
        
        # Usage
        request_duration.labels(method='GET', path='/api').observe(0.123)
    """

    def __init__(
        self,
        name: str,
        description: str,
        labelnames: Sequence[str] = (),
        namespace: str = '',
        subsystem: str = '',
        unit: str = '',
        buckets: Sequence[float] = (
            0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
            0.75, 1.0, 2.5, 5.0, 7.5, 10.0
        ),
    ):
        """
        Args:
            name: Metric name
            description: Metric description
            labelnames: List of label names
            namespace: Namespace (optional)
            subsystem: Subsystem (optional)
            unit: Unit (optional)
            buckets: Histogram bucket boundaries
        """
        registry = get_metrics_registry()
        
        self._histogram = PrometheusHistogram(
            name=name,
            documentation=description,
            labelnames=labelnames,
            namespace=namespace,
            subsystem=subsystem,
            unit=unit,
            buckets=buckets,
            registry=registry,
        )
    
    def labels(self, **labels):
        """
        Return a labeled Histogram

        Returns:
            LabeledHistogram instance
        """
        labeled = self._histogram.labels(**labels)
        return LabeledHistogram(labeled)
    
    def observe(self, amount: float) -> None:
        """
        Record an observation (unlabeled version)

        Args:
            amount: Observed value
        """
        self._histogram.observe(amount)


class LabeledHistogram:
    """Labeled Histogram"""

    def __init__(self, labeled_histogram):
        self._histogram = labeled_histogram

    def observe(self, amount: float) -> None:
        """
        Record an observation

        Args:
            amount: Observed value
        """
        self._histogram.observe(amount)
```

---

### 3. BaseGauge Base Class (Core)

**File: `src/core/metrics/gauge.py`**

```python
"""
Gauge Wrapper

Provides a unified Gauge interface with built-in auto-refresh capability
"""
from prometheus_client import Gauge as PrometheusGauge
from typing import Sequence, Optional, Callable, Any
import asyncio
import logging
from abc import ABC

logger = logging.getLogger(__name__)


class BaseGauge(ABC):
    """
    Gauge base class

    Features:
    - Instant value that can increase or decrease
    - Built-in auto-refresh capability (default 5 seconds)
    - Must inherit and override refresh() method
    - Each instance manages its own refresh task
    - Supports manual set() method

    Usage - inherit and override refresh method:
        class KafkaPendingMessagesGauge(BaseGauge):
            def __init__(self, kafka_consumer):
                super().__init__(
                    name='kafka_pending_messages',
                    description='Number of pending messages',
                    labelnames=['job_name']
                )
                self.kafka_consumer = kafka_consumer
            
            def refresh(self, labels: dict) -> float:
                '''Return current value'''
                return len(self.kafka_consumer.pending_messages)

        # Usage 1: Auto-refresh (default 5 seconds)
        gauge = KafkaPendingMessagesGauge(kafka_consumer)
        gauge.labels(job_name='tanka').start_refresh()

        # Usage 2: Custom refresh interval
        gauge.labels(job_name='tanka').start_refresh(interval_seconds=10)

        # Usage 3: Manual set (no auto-refresh)
        gauge.labels(job_name='tanka').set(42)
    """

    def __init__(
        self,
        name: str,
        description: str,
        labelnames: Sequence[str] = (),
        namespace: str = '',
        subsystem: str = '',
        unit: str = '',
    ):
        """
        Args:
            name: Metric name
            description: Metric description
            labelnames: List of label names
            namespace: Namespace (optional)
            subsystem: Subsystem (optional)
            unit: Unit (optional)
        """
        from .registry import get_metrics_registry
        registry = get_metrics_registry()
        
        self._gauge = PrometheusGauge(
            name=name,
            documentation=description,
            labelnames=labelnames,
            namespace=namespace,
            subsystem=subsystem,
            unit=unit,
            registry=registry,
        )
        
        self._name = name
        self._labelnames = labelnames
        
        # Store refresh task for each label combination
        # key: label 值的 tuple, value: RefreshTask
        self._refresh_tasks: dict[tuple, 'RefreshTask'] = {}
    
    def labels(self, **labels) -> 'LabeledGauge':
        """
        Return a labeled Gauge
        
        Returns:
            LabeledGauge 实例
        """
        labeled_gauge = self._gauge.labels(**labels)
        label_key = self._make_label_key(**labels)
        
        return LabeledGauge(
            base_gauge=self,
            labeled_gauge=labeled_gauge,
            label_key=label_key,
            label_dict=labels,
        )
    
    def set(self, value: float) -> None:
        """Set value (unlabeled version)"""
        self._gauge.set(value)
    
    def inc(self, amount: float = 1) -> None:
        """Increment value (unlabeled version)"""
        self._gauge.inc(amount)
    
    def dec(self, amount: float = 1) -> None:
        """Decrement value (unlabeled version)"""
        self._gauge.dec(amount)
    
    def refresh(self, labels: dict) -> float:
        """
        刷新方法（子类必须重写）
        
        Args:
            labels: 标签字典
        
        Returns:
            当前 Gauge 值
        
        说明：
            - 子类必须重写此方法来实现自定义刷新逻辑
            - 此方法会被自动刷新任务定期调用（默认 5 秒）
            - 可以返回任何 float 值，会自动更新到 Gauge
        
        示例：
            class QueueSizeGauge(BaseGauge):
                def __init__(self, queue):
                    super().__init__('queue_size', 'Queue size')
                    self.queue = queue
                
                def refresh(self, labels: dict) -> float:
                    return self.queue.qsize()
        """
        raise NotImplementedError(
            f"Gauge '{self._name}' must override refresh() method"
        )
    
    def _make_label_key(self, **labels) -> tuple:
        """Generate label key"""
        if self._labelnames:
            return tuple(labels.get(name, '') for name in self._labelnames)
        return ()
    
    async def _stop_all_refresh_tasks(self) -> None:
        """Stop all refresh tasks"""
        for task in self._refresh_tasks.values():
            await task.stop()
        self._refresh_tasks.clear()


class LabeledGauge:
    """
    带标签的 Gauge
    
    提供和原生 Gauge 一致的接口，同时支持自动刷新
    """
    
    def __init__(
        self,
        base_gauge: BaseGauge,
        labeled_gauge: Any,
        label_key: tuple,
        label_dict: dict,
    ):
        self._base_gauge = base_gauge
        self._labeled_gauge = labeled_gauge
        self._label_key = label_key
        self._label_dict = label_dict
    
    def set(self, value: float) -> None:
        """Set value"""
        self._labeled_gauge.set(value)
    
    def inc(self, amount: float = 1) -> None:
        """Increment value"""
        self._labeled_gauge.inc(amount)
    
    def dec(self, amount: float = 1) -> None:
        """Decrement value"""
        self._labeled_gauge.dec(amount)
    
    def set_to_current_time(self) -> None:
        """Set to current timestamp"""
        self._labeled_gauge.set_to_current_time()
    
    def start_refresh(
        self,
        interval_seconds: int = 5,
        enable_async: bool = True,
    ) -> 'LabeledGauge':
        """
        启动自动刷新
        
        Args:
            interval_seconds: 刷新间隔（秒），默认 5 秒
            enable_async: 是否支持异步 refresh 方法，默认 True
        
        Returns:
            self（支持链式调用）
        
        示例：
            # default 5 second refresh
            gauge.labels(job='tanka').start_refresh()
            
            # 自定义刷新间隔
            gauge.labels(job='tanka').start_refresh(interval_seconds=10)
            
            # 异步 refresh 方法
            class AsyncGauge(BaseGauge):
                async def refresh(self, labels: dict) -> float:
                    return await self.get_value_async()
            
            gauge.labels(type='A').start_refresh(enable_async=True)
        """
        # 创建包装函数，调用 base_gauge.refresh()
        def refresh_wrapper():
            return self._base_gauge.refresh(self._label_dict)
        
        # 创建刷新任务
        task = RefreshTask(
            refresh_func=refresh_wrapper,
            labeled_gauge=self._labeled_gauge,
            interval_seconds=interval_seconds,
            enable_async=enable_async,
            label_key=self._label_key,
        )
        
        # 存储任务
        self._base_gauge._refresh_tasks[self._label_key] = task
        
        # 启动任务
        task.start()
        
        return self
    
    async def stop_refresh(self) -> None:
        """Stop auto-refresh"""
        task = self._base_gauge._refresh_tasks.get(self._label_key)
        if task:
            await task.stop()
            del self._base_gauge._refresh_tasks[self._label_key]


class RefreshTask:
    """
    刷新任务
    
    每个标签组合一个独立的刷新任务
    """
    
    def __init__(
        self,
        refresh_func: Callable[[], float],
        labeled_gauge: Any,
        interval_seconds: int,
        enable_async: bool,
        label_key: tuple,
    ):
        self.refresh_func = refresh_func
        self.labeled_gauge = labeled_gauge
        self.interval_seconds = interval_seconds
        self.enable_async = enable_async
        self.label_key = label_key
        
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._error_count = 0
    
    def start(self) -> None:
        """Start refresh task"""
        if self._running:
            logger.warning(f"Refresh task already running for {self.label_key}")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._refresh_loop())
        logger.info(
            f"Started refresh task: label_key={self.label_key}, "
            f"interval={self.interval_seconds}s"
        )
    
    async def stop(self) -> None:
        """Stop refresh task"""
        if not self._running:
            return
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        logger.info(f"Stopped refresh task: label_key={self.label_key}")
    
    async def _refresh_loop(self) -> None:
        """Refresh loop"""
        while self._running:
            try:
                # Call refresh function
                if self.enable_async and asyncio.iscoroutinefunction(self.refresh_func):
                    value = await self.refresh_func()
                else:
                    value = self.refresh_func()
                
                # Update Gauge
                self.labeled_gauge.set(value)
                
                # Reset error count
                self._error_count = 0
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._error_count += 1
                logger.error(
                    f"Refresh failed for {self.label_key}: {e} "
                    f"(error_count={self._error_count})",
                    exc_info=True
                )
            
            # Wait for next refresh
            try:
                await asyncio.sleep(self.interval_seconds)
            except asyncio.CancelledError:
                break
```

---

### 4. Unified Export

**File:`src/core/metrics/__init__.py`**

```python
"""
Metrics Library

Business code imports metric classes from here, no direct dependency on prometheus_client needed

使用示例：
    from core.metrics import Counter, Histogram, BaseGauge
    
    # Counter
    requests_total = Counter('http_requests_total', 'Total requests', ['method'])
    requests_total.labels(method='GET').inc()
    
    # Histogram
    request_duration = Histogram('http_request_duration_seconds', 'Request duration', ['method'])
    request_duration.labels(method='GET').observe(0.123)
    
    # Gauge - inheritance pattern
    class QueueSizeGauge(BaseGauge):
        def __init__(self, queue):
            super().__init__('queue_size', 'Queue size', ['queue_name'])
            self.queue = queue
        
        def refresh(self, labels: dict) -> float:
            return self.queue.qsize()
    
    # Using Gauge
    gauge = QueueSizeGauge(queue)
    gauge.labels(queue_name='main').start_refresh()  # default 5 second refresh
    # or manual set
    gauge.labels(queue_name='main').set(42)
"""

from .counter import Counter
from .histogram import Histogram
from .gauge import BaseGauge
from .registry import get_metrics_registry, generate_metrics_response

__all__ = [
    'Counter',
    'Histogram',
    'BaseGauge',
    'get_metrics_registry',
    'generate_metrics_response',
]
```

---

## 💡 Usage Examples

### Example 1: Kafka Metrics

**File:`src/infra_layer/adapters/input/mq/metrics/kafka_metrics.py`**

```python
"""
Kafka metrics definition

Only imports core.metrics, no direct prometheus_client import
"""
from core.metrics import Counter, Histogram, BaseGauge


# ============================================================
# Counter and Histogram - direct usage
# ============================================================

KAFKA_PROCESSED_MESSAGES_TOTAL = Counter(
    name='kafka_processed_messages_total',
    description='Total number of processed Kafka messages',
    labelnames=['job_name', 'status'],
)

KAFKA_MESSAGE_PROCESSING_DURATION = Histogram(
    name='kafka_message_processing_duration_seconds',
    description='Duration of message processing',
    labelnames=['job_name'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
)


# ============================================================
# Gauge - unified inheritance pattern
# ============================================================

class KafkaPendingMessagesGauge(BaseGauge):
    """Kafka pending messages Gauge"""
    
    def __init__(self, kafka_consumer):
        super().__init__(
            name='kafka_prefill_pending_messages',
            description='Number of pending messages in prefill stage',
            labelnames=['job_name'],
        )
        self.kafka_consumer = kafka_consumer
    
    def refresh(self, labels: dict) -> float:
        """Return current pending message count"""
        return len(self.kafka_consumer.prefill_pending_messages)


class KafkaActiveConsumersGauge(BaseGauge):
    """Kafka active consumers Gauge"""
    
    def __init__(self, kafka_consumer):
        super().__init__(
            name='kafka_active_consumers',
            description='Number of active consumer tasks',
            labelnames=['job_name'],
        )
        self.kafka_consumer = kafka_consumer
    
    def refresh(self, labels: dict) -> float:
        """Return current active consumer count"""
        return len(self.kafka_consumer.consumer_tasks)


class KafkaRedisQueueSizeGauge(BaseGauge):
    """Redis queue size Gauge"""
    
    def __init__(self, redis_queue_manager):
        super().__init__(
            name='kafka_redis_queue_size',
            description='Total size of Redis queues',
            labelnames=['job_name'],
        )
        self.redis_queue_manager = redis_queue_manager
    
    def refresh(self, labels: dict) -> float:
        """Return total Redis queue size"""
        if not self.redis_queue_manager:
            return 0
        
        total_size = 0
        for partition in self.redis_queue_manager.partitions.values():
            size = partition.size()
            total_size += size
        
        return total_size


class KafkaMemoryQueueSizeGauge(BaseGauge):
    """Memory queue size Gauge"""
    
    def __init__(self, memory_queue_manager):
        super().__init__(
            name='kafka_memory_queue_size',
            description='Total size of in-memory queues',
            labelnames=['job_name'],
        )
        self.memory_queue_manager = memory_queue_manager
    
    def refresh(self, labels: dict) -> float:
        """Return total memory queue size"""
        if not self.memory_queue_manager:
            return 0
        
        return sum(
            q.qsize() 
            for q in self.memory_queue_manager._queues.values()
        )
```

### Example 2: Business Code Usage

**File:`src/infra_layer/adapters/input/mq/tanka_kafka_consumer.py`**

```python
"""
Kafka consumer - using wrapped metrics library
"""
import time
from .metrics.kafka_metrics import (
    KAFKA_PROCESSED_MESSAGES_TOTAL,
    KAFKA_MESSAGE_PROCESSING_DURATION,
    KafkaPendingMessagesGauge,
    KafkaActiveConsumersGauge,
    KafkaRedisQueueSizeGauge,
    KafkaMemoryQueueSizeGauge,
)


class TankaKafkaConsumer:
    """Kafka consumer"""
    
    def __init__(
        self,
        job_id: str,
        redis_queue_manager=None,
        memory_queue_manager=None,
        **kwargs
    ):
        self.job_id = job_id
        self.redis_queue_manager = redis_queue_manager
        self.memory_queue_manager = memory_queue_manager
        
        # Business properties
        self.prefill_pending_messages = []
        self.consumer_tasks = []
        
        # ... other initialization ...
        
        # ✅ Set up Gauge auto-refresh
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Set up metric auto-refresh"""
        
        # 1. Pending message count（default 5 second refresh）
        pending_gauge = KafkaPendingMessagesGauge(self)
        pending_gauge.labels(
            job_name=self.job_id
        ).start_refresh()  # 默认 5 秒
        
        # 2. Active consumer count（default 5 second refresh）
        active_gauge = KafkaActiveConsumersGauge(self)
        active_gauge.labels(
            job_name=self.job_id
        ).start_refresh()
        
        # 3. Redis 队列大小（自定义 10 秒刷新）
        if self.redis_queue_manager:
            redis_gauge = KafkaRedisQueueSizeGauge(self.redis_queue_manager)
            redis_gauge.labels(
                job_name=self.job_id
            ).start_refresh(interval_seconds=10)
        
        # 4. 内存队列大小（default 5 second refresh）
        if self.memory_queue_manager:
            memory_gauge = KafkaMemoryQueueSizeGauge(self.memory_queue_manager)
            memory_gauge.labels(
                job_name=self.job_id
            ).start_refresh()
    
    async def _process_message(self, message) -> None:
        """处理消息"""
        start_time = time.time()
        
        try:
            # Business logic
            await self._do_process(message)
            
            # ✅ Counter
            KAFKA_PROCESSED_MESSAGES_TOTAL.labels(
                job_name=self.job_id,
                status='success'
            ).inc()
            
        except Exception as e:
            # ✅ Counter（错误）
            KAFKA_PROCESSED_MESSAGES_TOTAL.labels(
                job_name=self.job_id,
                status='error'
            ).inc()
            raise
        
        finally:
            # ✅ Histogram
            duration = time.time() - start_time
            KAFKA_MESSAGE_PROCESSING_DURATION.labels(
                job_name=self.job_id
            ).observe(duration)
```

### 示例 3：Memory Metrics

**File:`src/agentic_layer/metrics/memory_metrics.py`**

```python
"""
Memory 指标定义
"""
from core.metrics import Counter, Histogram, BaseGauge


# Counter 和 Histogram
RETRIEVE_REQUESTS_TOTAL = Counter(
    name='memory_retrieve_requests_total',
    description='Total number of memory retrieve requests',
    labelnames=['retrieve_method', 'status'],
)

RETRIEVE_DURATION_SECONDS = Histogram(
    name='memory_retrieve_duration_seconds',
    description='Duration of memory retrieve operations',
    labelnames=['retrieve_method'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


# Gauge - unified inheritance pattern
class MemoryCacheSizeGauge(BaseGauge):
    """Memory Cache 大小 Gauge"""
    
    def __init__(self, service, cache_type: str):
        super().__init__(
            name='memory_cache_size',
            description='Current size of memory cache',
            labelnames=['cache_type'],
        )
        self.service = service
        self.cache_type = cache_type
    
    def refresh(self, labels: dict) -> float:
        """返回缓存大小"""
        cache = getattr(self.service, 'cache', {})
        return len(cache)


class MemoryActiveRequestsGauge(BaseGauge):
    """Memory 活跃请求数 Gauge"""
    
    def __init__(self, memory_manager, operation: str):
        super().__init__(
            name='memory_active_requests',
            description='Number of active memory requests',
            labelnames=['operation'],
        )
        self.memory_manager = memory_manager
        self.operation = operation
    
    def refresh(self, labels: dict) -> float:
        """返回活跃请求数"""
        if self.operation == 'retrieve':
            return getattr(self.memory_manager, 'active_retrieve_count', 0)
        elif self.operation == 'memorize':
            return getattr(self.memory_manager, 'active_memorize_count', 0)
        return 0
```

**File:`src/agentic_layer/memory_manager.py`**

```python
"""
Memory Manager - 使用 Gauge
"""
from .metrics.memory_metrics import (
    RETRIEVE_REQUESTS_TOTAL,
    RETRIEVE_DURATION_SECONDS,
    MemoryCacheSizeGauge,
    MemoryActiveRequestsGauge,
)


class MemoryManager:
    """Memory Manager"""
    
    def __init__(self, embedding_service, rerank_service):
        self.embedding_service = embedding_service
        self.rerank_service = rerank_service
        self.active_retrieve_count = 0
        self.active_memorize_count = 0
        
        # ✅ Set up Gauge auto-refresh
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Set up metric auto-refresh"""
        
        # Embedding cache 大小（10 秒刷新）
        embedding_cache_gauge = MemoryCacheSizeGauge(
            self.embedding_service, 
            'embedding'
        )
        embedding_cache_gauge.labels(
            cache_type='embedding'
        ).start_refresh(interval_seconds=10)
        
        # Rerank cache 大小（10 秒刷新）
        rerank_cache_gauge = MemoryCacheSizeGauge(
            self.rerank_service,
            'rerank'
        )
        rerank_cache_gauge.labels(
            cache_type='rerank'
        ).start_refresh(interval_seconds=10)
        
        # 活跃的 retrieve 请求（默认 5 秒）
        retrieve_gauge = MemoryActiveRequestsGauge(self, 'retrieve')
        retrieve_gauge.labels(
            operation='retrieve'
        ).start_refresh()
        
        # 活跃的 memorize 请求（默认 5 秒）
        memorize_gauge = MemoryActiveRequestsGauge(self, 'memorize')
        memorize_gauge.labels(
            operation='memorize'
        ).start_refresh()
    
    async def retrieve_mem(self, request):
        """检索记忆"""
        start_time = time.time()
        retrieve_method = request.retrieve_method
        
        try:
            # Business logic
            memories = await self._do_retrieve(request)
            
            # ✅ Counter
            RETRIEVE_REQUESTS_TOTAL.labels(
                retrieve_method=retrieve_method,
                status='success'
            ).inc()
            
            return memories
        
        finally:
            # ✅ Histogram
            duration = time.time() - start_time
            RETRIEVE_DURATION_SECONDS.labels(
                retrieve_method=retrieve_method
            ).observe(duration)
```

---

## 🎯 方案优势

1. **业务代码完全隔离第三方依赖**
   - 业务只引用 `core.metrics`，不直接依赖 `prometheus_client`
   - 方便后续替换底层实现

2. **统一继承方式，简单直观**
   - 所有 Gauge 都继承 BaseGauge 并重写 `refresh()` 方法
   - 接口统一，学习成本低
   - 代码风格一致

3. **轻量级，无全局调度器**
   - 每个 Gauge 实例自己管理刷新任务
   - 每个标签组合一个独立的 `asyncio.Task`
   - 无需全局调度器协调

4. **灵活的刷新间隔**
   - default 5 second refresh
   - 可自定义刷新间隔（3秒、10秒、30秒等）
   - 不同的标签可以有不同的间隔

5. **支持手动 set()**
   - 可以随时手动 `set()` Set value
   - 手动设置和自动刷新互不干扰
   - 灵活应对各种场景

## 📊 对比总结

| 维度 | 直接使用 prometheus_client | 本方案 |
|------|--------------------------|--------|
| **依赖隔离** | 业务代码直接依赖第三方库 | 业务代码只依赖 core.metrics |
| **Gauge 刷新** | 手动调用 .set() | 自动刷新（默认 5 秒） + 支持手动 set() |
| **调度器** | 无（或需要自己实现） | 无需全局调度器（每个 Gauge 独立） |
| **使用方式** | 需要自己实现刷新逻辑 | 继承 BaseGauge 重写 refresh() |
| **代码复杂度** | 简单但功能少 | 封装后同样简单且功能更多 |
| **可扩展性** | 需要自己扩展 | 继承 BaseGauge 即可扩展 |

---

## 🚀 实施步骤

### 阶段 1：创建封装层
1. ✅ 创建 `core/metrics/counter.py`
2. ✅ 创建 `core/metrics/histogram.py`
3. ✅ 创建 `core/metrics/gauge.py`（核心）
4. ✅ 修改 `core/metrics/__init__.py` 导出
5. ✅ 简化 `core/metrics/registry.py`（移除 MetricsSource/Processor）

### 阶段 2：重构业务指标
1. ✅ 重构 `kafka_metrics.py`（改用 core.metrics，Gauge 改为继承）
2. ✅ 重构 `memory_metrics.py`（改用 core.metrics，Gauge 改为继承）

### 阶段 3：业务代码集成
1. ✅ 修改 `TankaKafkaConsumer`（创建 Gauge 实例并 start_refresh）
2. ✅ 修改 `MemoryManager`（创建 Gauge 实例并 start_refresh）

### 阶段 4：清理旧代码
1. ✅ 删除所有 `*_metrics_processor.py` 文件
2. ✅ 删除所有 `*_metrics_source.py` 文件
3. ✅ 删除 `metrics_processor.py` 基类
4. ✅ 删除 `metrics_source.py` 基类

---

## 💡 关键设计要点

1. **统一继承方式**
   - 所有 Gauge 必须继承 BaseGauge
   - 必须重写 `refresh()` 方法
   - 不支持 `set_refresher()` 方式

2. **default 5 second refresh**
   - `start_refresh()` 默认 5 秒间隔
   - 可通过 `interval_seconds` 参数自定义
   - 符合大部分业务场景

3. **支持手动 set()**
   - 可以不启动自动刷新，直接手动 `set()`
   - 手动设置和自动刷新可以混用
   - 灵活应对特殊场景

4. **轻量级实现**
   - RefreshTask 管理单个标签的刷新任务
   - 每个任务独立的 `asyncio.Task`
   - 无全局调度器，无性能开销

5. **异常隔离**
   - 每个 Gauge 的刷新异常不影响其他 Gauge
   - 自动错误日志记录
   - 自动重试机制

---

## 🔧 配置示例

**File:`src/core/config.py`**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Metrics 配置
    PROMETHEUS_METRICS_ENABLED: bool = True
    
    # Gauge 默认刷新间隔（秒）
    # 注意：这是全局默认值，业务代码可以通过 start_refresh(interval_seconds=N) 覆盖
    METRICS_GAUGE_DEFAULT_INTERVAL: int = 5
    
    # ... 其他配置 ...


settings = Settings()
```

需要我开始实施这个方案吗？

