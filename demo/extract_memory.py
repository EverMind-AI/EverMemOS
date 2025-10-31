"""记忆提取脚本 - 从对话数据中提取 MemCell 和个人 Profile

本脚本支持三种运行模式：
1. EXTRACT_ALL: 完整提取（MemCell + Profile）
2. EXTRACT_MEMCELL_ONLY: 仅提取 MemCell
3. EXTRACT_PROFILE_ONLY: 仅提取 Profile（从已有 MemCell）

主要功能：
- 消息归一化和过滤
- 基于 LLM 的对话边界检测
- MemCell 提取和持久化到 MongoDB
- 后台异步聚类
- 个人 Profile 提取（从群组 MemCell 中为每个参与者提取）

使用方法：
    python extract_memory.py
"""

import asyncio
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

# 确保 src 包可被发现
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from memory_layer.memcell_extractor.conv_memcell_extractor_with_cluster import (
    ConvMemCellExtractorWithCluster as ConvMemCellExtractor,
    ConversationMemCellExtractRequest,
)
from src.memory_layer.llm.llm_provider import LLMProvider
from src.common_utils.datetime_utils import from_iso_format, get_now_with_timezone
from src.infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell as DocMemCell,
    DataTypeEnum,
)
from src.memory_layer.memory_extractor.profile_memory_extractor import (
    ProfileMemoryExtractor,
    ProfileMemoryExtractRequest,
)
from demo.profile_extractor import ValueDiscriminatorCompanion, DiscriminatorCompanionConfig
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# 导入共享配置和工具
from demo.memory_config import (
    RunMode,
    ScenarioType,
    ExtractModeConfig,
    LLMConfig,
    MongoDBConfig,
)
from demo.memory_utils import (
    ensure_mongo_beanie_ready,
    serialize_datetime,
    PerformanceMetrics,
    ProgressTracker,
    BatchMongoWriter,
)

load_dotenv()

# ============================================================================
# 全局性能指标（优化版新增）
# ============================================================================

# 全局性能指标对象
PERF_METRICS = PerformanceMetrics()


# ============================================================================
# 全局配置
# ============================================================================

# 💡 快速切换运行模式：
# - RunMode.EXTRACT_ALL: 完整提取（MemCell + Profile）
# - RunMode.EXTRACT_MEMCELL_ONLY: 仅提取 MemCell
# - RunMode.EXTRACT_PROFILE_ONLY: 仅提取 Profile（从已有 MemCell）

# CURRENT_RUN_MODE = RunMode.EXTRACT_PROFILE_ONLY  # ✅ 完整提取：MemCell + Profile
CURRENT_RUN_MODE = RunMode.EXTRACT_ALL  # ✅ 完整提取：MemCell + Profile

# 提取配置
EXTRACT_CONFIG = ExtractModeConfig(
    scenario_type=ScenarioType.GROUP_CHAT,
    # scenario_type=ScenarioType.ASSISTANT,
    enable_profile_extraction=True,
)

# LLM 配置
LLM_CONFIG = LLMConfig()

# MongoDB 配置
MONGO_CONFIG = MongoDBConfig()


# ============================================================================
# 工具函数
# ============================================================================


def estimate_memcell_tokens(memcell) -> int:
    """估算单个 MemCell 的 Token 数量（粗略估算）

    估算逻辑：
    - 英文：约 1 token = 4 字符
    - 中文：约 1 token = 1.5 字符
    - 格式化开销：每条消息额外 20 tokens

    Args:
        memcell: MemCell 对象

    Returns:
        估算的 token 数量
    """
    try:
        total_chars = 0
        message_count = 0

        # 获取 original_data 中的消息列表
        original_data = getattr(memcell, 'original_data', None)
        if original_data and isinstance(original_data, list):
            for msg in original_data:
                if isinstance(msg, dict):
                    # 统计 content 字段的字符数
                    content = msg.get('content', '')
                    if content:
                        total_chars += len(str(content))

                    # 统计 speaker_name 字段的字符数
                    speaker_name = msg.get('speaker_name', '')
                    if speaker_name:
                        total_chars += len(str(speaker_name))

                    message_count += 1

        # 粗略估算：平均 2.5 字符 = 1 token（混合中英文场景）
        # + 每条消息的格式化开销（时间戳、conversation_id 等）
        estimated_tokens = int(total_chars / 2.5) + (message_count * 20)

        return estimated_tokens
    except Exception:
        # 如果估算失败，返回保守值
        return 1000


def estimate_total_tokens(memcell_list: List) -> int:
    """估算 MemCell 列表的总 Token 数量

    Args:
        memcell_list: MemCell 对象列表

    Returns:
        估算的总 token 数量
    """
    if not memcell_list:
        return 2000  # 只返回基础 Prompt 开销

    total_tokens = 0
    for memcell in memcell_list:
        total_tokens += estimate_memcell_tokens(memcell)

    # 添加 Prompt 模板的基础开销（约 2000 tokens）
    total_tokens += 2000

    return total_tokens


def split_memcells_into_batches(
    memcell_list: List, max_batch_size: int, strategy: str = "latest_first"
) -> List[List]:
    """将 MemCell 列表分批，支持多种策略

    Args:
        memcell_list: MemCell 对象列表
        max_batch_size: 每批最大 MemCell 数量（必须 > 0）
        strategy: 分批策略
            - "latest_first": 最新优先（按时间戳降序，最新的在前）
            - "distributed": 均匀分布（尽量让每批包含不同时期的对话）
            - "sequential": 顺序分批（按原始顺序）

    Returns:
        分批后的 MemCell 列表（二维列表）
    """
    # 边界条件检查
    if not memcell_list:
        return []

    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size 必须 > 0，当前值: {max_batch_size}")

    # 如果数量不超过限制，直接返回
    if len(memcell_list) <= max_batch_size:
        return [memcell_list]

    # 根据策略排序
    sorted_memcells = list(memcell_list)  # 复制列表，避免修改原始数据

    if strategy == "latest_first":
        # 按时间戳降序排序（最新的在前）
        sorted_memcells.sort(key=lambda mc: getattr(mc, 'timestamp', ''), reverse=True)
    elif strategy == "distributed":
        # 先按时间排序，然后交错分配
        sorted_memcells.sort(key=lambda mc: getattr(mc, 'timestamp', ''))
        # 使用轮询方式分配到各批次，确保均匀分布
        total_batches = (len(sorted_memcells) + max_batch_size - 1) // max_batch_size
        distributed_batches = [[] for _ in range(total_batches)]
        for idx, memcell in enumerate(sorted_memcells):
            batch_idx = idx % total_batches
            distributed_batches[batch_idx].append(memcell)
        return [batch for batch in distributed_batches if batch]
    # strategy == "sequential" 时不排序，保持原始顺序

    # 顺序分批（latest_first 和 sequential 策略）
    batches = []
    for i in range(0, len(sorted_memcells), max_batch_size):
        batch = sorted_memcells[i : i + max_batch_size]
        batches.append(batch)

    return batches


def ensure_output_dir(output_dir: Path, clean_previous: bool = True) -> None:
    """确保输出目录存在，可选择清理之前的结果

    Args:
        output_dir: 输出目录路径
        clean_previous: 是否清理之前的结果文件
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_previous:
        for existing_file in output_dir.glob("memcell_*.json"):
            existing_file.unlink()


def load_events(path: Path) -> List[Dict[str, Any]]:
    """从 JSON 文件加载对话事件列表并规范化格式

    Args:
        path: JSON 文件路径

    Returns:
        对话事件字典列表
    """
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if isinstance(data, dict):
        conversation_list = data.get("conversation_list")
        if conversation_list is not None:
            if isinstance(conversation_list, list):
                return conversation_list
            raise ValueError("`conversation_list` 字段必须为数组，请检查输入 JSON。")

    if isinstance(data, list):
        return data

    raise ValueError(
        "不支持的群聊数据格式，请提供包含 `conversation_list` 的对象或消息列表。"
    )


def load_memcells_from_files(input_dir: Path) -> List[Any]:
    """从 JSON 文件加载已有的 MemCell 列表

    Args:
        input_dir: MemCell 文件所在目录

    Returns:
        MemCell 对象列表（包装为简单对象以支持属性访问）
    """

    class MemCellWrapper:
        """简单的 MemCell 包装类，支持属性访问"""

        def __init__(self, data: dict):
            self._data = data
            # 将字典的键作为属性
            for key, value in data.items():
                setattr(self, key, value)

        def __getattr__(self, name):
            return self._data.get(name)

    memcells = []
    memcell_files = sorted(input_dir.glob("memcell_*.json"))

    if not memcell_files:
        print(f"[LoadMemCell] ⚠️ 未找到 MemCell 文件: {input_dir}")
        return []

    print(f"[LoadMemCell] 发现 {len(memcell_files)} 个 MemCell 文件")

    for file_path in memcell_files:
        try:
            with file_path.open("r", encoding="utf-8") as fp:
                memcell_data = json.load(fp)
                # 包装为对象以支持属性访问
                memcell_obj = MemCellWrapper(memcell_data)
                memcells.append(memcell_obj)
        except Exception as e:
            print(f"[LoadMemCell] ⚠️ 加载文件失败 {file_path.name}: {e}")
            continue

    print(f"[LoadMemCell] ✅ 成功加载 {len(memcells)} 个 MemCell")
    return memcells


def _is_supported_msg_type(entry: Dict[str, Any], supported_types: set) -> bool:
    """检查消息类型是否支持

    Args:
        entry: 原始消息条目
        supported_types: 支持的消息类型集合

    Returns:
        是否支持该消息类型
    """
    raw_msg_type = entry.get("type")
    if raw_msg_type is None:
        return True

    raw_msg_type = str(raw_msg_type).strip().lower()
    if not raw_msg_type:
        return True

    if supported_types is None:
        return True

    normalized_supported = {str(item).strip().lower() for item in supported_types}
    return raw_msg_type in normalized_supported


def normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any] | None:
    """将原始事件结构转换为 ConvMemCellExtractor 友好的格式

    Args:
        entry: 原始消息条目

    Returns:
        归一化后的消息 payload，如果解析失败返回 None
    """
    # 1. 提取并解析时间戳（兼容多种字段名）
    timestamp = (
        entry.get("create_time")
        or entry.get("createTime")  # 驼峰式
        or entry.get("timestamp")
        or entry.get("created_at")
    )
    if timestamp is None:
        return None

    if isinstance(timestamp, str):
        try:
            timestamp_dt = from_iso_format(timestamp)
        except Exception:
            return None
    else:
        return None

    # 2. 提取发言者名称（兼容多种来源）
    speaker_name = entry.get("sender_name") or entry.get("sender")

    # 如果没有找到，尝试从 origin 字段中获取
    if not speaker_name:
        origin = entry.get("origin")
        if isinstance(origin, dict):
            speaker_name = origin.get("fullName") or origin.get("full_name")

    if not speaker_name:
        return None
    speaker_name = str(speaker_name)

    # 3. 提取发言者 ID（兼容多种字段名，优先级：origin.createBy > sender_id > sender）
    raw_speaker_id = None

    # 优先从 origin 中获取（这是真实的用户ID）
    origin = entry.get("origin")
    if isinstance(origin, dict):
        raw_speaker_id = origin.get("createBy") or origin.get("create_by")

    # 如果没有找到，再从顶层字段获取
    if not raw_speaker_id:
        raw_speaker_id = entry.get("sender_id") or entry.get("sender")

    speaker_id = str(raw_speaker_id) if raw_speaker_id is not None else ""

    # 4. 提取消息内容
    content = entry.get("content", "")
    content = str(content)

    # 5. 提取 @ 提及列表
    refer_entries = (
        entry.get("refer_list")
        or entry.get("referList")
        or entry.get("references")
        or []
    )
    refer_list: List[dict] = []
    for refer in refer_entries:
        if isinstance(refer, dict):
            refer_id = (
                refer.get("message_id")
                or refer.get("id")
                or refer.get("_id")
                or refer.get("refer_id")
            )
            name = (
                refer.get("sender_name")
                or refer.get("name")
                or refer.get("full_name")
                or refer.get("display_name")
            )
            if not refer_id or not name:
                continue
            refer_list.append({"id": str(refer_id), "name": str(name)})
        elif isinstance(refer, str):
            refer_list.append({"id": refer, "name": refer})

    # 6. 提取消息类型并转换为数字格式（兼容多种来源）
    msg_type = entry.get("type") or entry.get("msgType")

    # 如果没有找到，尝试从 origin 字段中获取
    if msg_type is None:
        origin = entry.get("origin")
        if isinstance(origin, dict):
            msg_type = origin.get("msgType") or origin.get("msg_type")

    if msg_type is not None:
        # 消息类型映射表：将字符串类型转换为数字
        msg_type_mapping = {
            "text": 1,
            "image": 2,
            "file": 3,
            "audio": 4,
            "video": 5,
            "link": 6,
            "system": 7,
        }

        # 如果已经是数字，直接使用
        if isinstance(msg_type, int):
            pass  # 保持不变
        else:
            # 先转换为小写字符串
            msg_type_str = str(msg_type).strip().lower()

            # 如果是数字字符串，转换为整数；否则查找映射表
            if msg_type_str.isdigit():
                msg_type = int(msg_type_str)
            else:
                msg_type = msg_type_mapping.get(msg_type_str, 1)  # 默认为 TEXT

    # 7. 构建并返回 payload
    payload: Dict[str, Any] = {
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "content": content,
        "timestamp": timestamp_dt,
        "referList": refer_list,
    }
    if msg_type is not None:
        payload["msgType"] = msg_type

    return payload


def write_memcell_to_file(memcell, index: int, output_dir: Path) -> None:
    """将 MemCell 保存为 JSON 文件

    Args:
        memcell: 要保存的 MemCell 对象
        index: 文件索引编号
        output_dir: 输出目录路径
    """
    payload = memcell.to_dict()

    # 归一化枚举类型为原始值
    memcell_type = payload.get("type")
    if memcell_type is not None and hasattr(memcell_type, "value"):
        payload["type"] = memcell_type.value

    # 将 original_data 中的 datetime 转换为 ISO 字符串
    if "original_data" in payload and payload["original_data"]:
        for item in payload["original_data"]:
            if isinstance(item, dict) and "timestamp" in item:
                ts = item["timestamp"]
                if hasattr(ts, "isoformat"):
                    item["timestamp"] = ts.isoformat()

    # 保存文件
    output_path = output_dir / f"memcell_{index:03d}.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print(f"[Extract] 保存 MemCell #{index} → {output_path.name}")


async def _save_memcell_to_mongodb(memcell) -> None:
    """将 MemCell 保存到 MongoDB

    Args:
        memcell: 要保存的 MemCell 对象
    """
    try:
        # 解析时间戳
        ts = memcell.timestamp
        if isinstance(ts, str):
            ts_dt = from_iso_format(ts)
        elif isinstance(ts, (int, float)):
            tz = get_now_with_timezone().tzinfo
            ts_dt = datetime.fromtimestamp(float(ts), tz=tz)
        else:
            ts_dt = ts or get_now_with_timezone()

        # 获取主用户 ID
        primary_user = (
            memcell.user_id_list[0]
            if getattr(memcell, 'user_id_list', None)
            else "default"
        )

        # 创建文档并插入
        doc = DocMemCell(
            user_id=primary_user,
            timestamp=ts_dt,
            summary=memcell.summary or "",
            group_id=getattr(memcell, 'group_id', None),
            participants=getattr(memcell, 'participants', None),
            type=DataTypeEnum.CONVERSATION,
            original_data=memcell.original_data,
            subject=getattr(memcell, 'subject', None),
            keywords=getattr(memcell, 'keywords', None),
            linked_entities=getattr(memcell, 'linked_entities', None),
            episode=getattr(memcell, 'episode', None),
            semantic_memories=getattr(memcell, 'semantic_memories', None),
            extend=getattr(memcell, 'extend', None),
        )
        await doc.insert()
    except Exception as e:
        print(f"[MongoDB] ⚠️ 保存 MemCell 失败: {e}")


async def _save_clustering_snapshot(
    extractor: ConvMemCellExtractor, output_dir: Path, group_id: str
) -> None:
    """保存聚类快照到文件

    Args:
        extractor: MemCell 提取器（包含 cluster_worker）
        output_dir: 输出目录
        group_id: 群组 ID
    """
    try:
        # 给后台聚类 worker 短暂的处理时间
        await asyncio.sleep(0.1)

        # 获取聚类分配结果
        assignments = extractor.cluster_worker.get_assignments()

        # 保存总聚类分配文件
        out_path = output_dir / "cluster_assignments.json"
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(assignments, fp, ensure_ascii=False, indent=2)

        # 保存每个群组的聚类文件
        try:
            extractor.cluster_worker.dump_to_dir(str(output_dir))
        except Exception:
            pass

        # 额外：保存聚类状态（包含簇质心和计数），便于下次恢复
        try:
            extractor.cluster_worker.dump_state_to_dir(str(output_dir))
        except Exception:
            pass
    except Exception:
        pass


async def save_individual_profile_to_file(
    profile, user_id: str, output_dir: Path
) -> None:
    """保存个人 Profile 到文件

    Args:
        profile: Profile Memory 对象
        user_id: 用户 ID（如 "user_101"）
        output_dir: 输出目录
    """
    try:
        # 转换为字典（处理 timestamp 可能是字符串的情况）
        if hasattr(profile, 'to_dict'):
            try:
                payload = profile.to_dict()
            except (AttributeError, TypeError) as e:
                # 如果 to_dict() 失败（通常是因为 timestamp 已经是字符串）
                # 直接使用 __dict__ 并手动处理
                error_msg = str(e).lower()
                if 'tzinfo' in error_msg or 'isoformat' in error_msg:
                    # timestamp 类型错误，使用备用方案
                    payload = profile.__dict__.copy()
                    # 确保 memory_type 转换为字符串值
                    if hasattr(payload.get('memory_type'), 'value'):
                        payload['memory_type'] = payload['memory_type'].value
                    # timestamp 如果已经是字符串，保持不变
                    # 如果是 datetime，转换为字符串
                    ts = payload.get('timestamp')
                    if ts is not None:
                        if hasattr(ts, 'isoformat'):
                            payload['timestamp'] = ts.isoformat()
                        elif not isinstance(ts, str):
                            # 既不是 datetime 也不是 str，尝试转换
                            payload['timestamp'] = str(ts)
                else:
                    raise  # 其他错误继续抛出
        elif hasattr(profile, '__dict__'):
            payload = profile.__dict__.copy()
            # 处理枚举类型
            if hasattr(payload.get('memory_type'), 'value'):
                payload['memory_type'] = payload['memory_type'].value
            # 处理 timestamp
            ts = payload.get('timestamp')
            if ts is not None:
                if hasattr(ts, 'isoformat'):
                    payload['timestamp'] = ts.isoformat()
                elif not isinstance(ts, str):
                    payload['timestamp'] = str(ts)
        else:
            payload = {}

        # 序列化 datetime（添加更好的错误处理）
        try:
            payload = serialize_datetime(payload)
        except Exception as e:
            # 如果序列化失败，继续使用当前 payload
            # 已经在上面处理了 timestamp，这里通常不会出错
            pass

        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        # 文件名：profile_user_101.json
        output_path = output_dir / f"profile_{user_id}.json"

        # 保存文件，使用 default=str 处理无法序列化的对象
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)

        print(f"[Profile] 保存 {user_id} → {output_path.name}")
    except Exception as e:
        print(f"[Profile] ❌ 保存 {user_id} Profile 失败: {e}")
        import traceback

        traceback.print_exc()


# ============================================================================
# Profile 提取逻辑 - 为每个参与者提取个人 Profile
# ============================================================================


async def extract_individual_profiles_from_group_memcells(
    memcell_list: List,
    profile_extractor: ProfileMemoryExtractor,
    group_id: str,
    group_name: Optional[str],
    output_dir: Path,
    extract_config: Optional[ExtractModeConfig] = None,
) -> int:
    """从群组 MemCell 中为所有参与者提取个人 Profile（智能分批优化版）

    核心逻辑：
    1. 验证输入 MemCell 列表
    2. Token 预估和智能分批决策
    3. 分批调用 LLM 提取 Profile（每批使用增量更新）
    4. 智能合并所有批次的结果
    5. 为每个用户保存独立的 profile_user_{user_id}.json

    重要改进：
    - ✅ 自动检测 MemCell 数量，超过阈值时分批处理
    - ✅ Token 预估，避免超出 LLM 上限
    - ✅ 增量更新机制（前批次结果作为后批次的 old_memory_list）
    - ✅ 支持多种分批策略（最新优先/均匀分布/顺序分批）
    - ✅ 详细的进度报告和性能追踪

    Args:
        memcell_list: 群组的所有 MemCell（包含完整对话上下文）
        profile_extractor: 个人 Profile 提取器
        group_id: 群组 ID
        group_name: 群组名称
        output_dir: 输出目录
        extract_config: 提取配置（包含分批参数）

    Returns:
        成功提取并保存的 Profile 总数
    """

    # 步骤 1：验证输入
    if not memcell_list:
        print("[ProfileExtract] ⚠️ MemCell 列表为空，跳过 Profile 提取")
        return 0

    # 步骤 2：统计所有参与者（用于验证和报告）
    all_users = set()
    for memcell in memcell_list:
        participants = getattr(memcell, 'participants', None)
        if participants and isinstance(participants, list):
            all_users.update(participants)

    if not all_users:
        print("[ProfileExtract] ⚠️ 未找到参与者，跳过 Profile 提取")
        return 0

    print(f"\n[ProfileExtract] 📊 统计信息:")
    print(f"  - MemCell 数量: {len(memcell_list)}")
    print(f"  - 参与者数量: {len(all_users)}")
    print(f"  - 参与者列表: {sorted(all_users)}")

    # 步骤 3：Token 预估和分批决策
    # 使用默认配置（如果未提供）
    if extract_config is None:
        extract_config = EXTRACT_CONFIG

    max_batch_size = extract_config.max_memcells_per_profile_batch
    batch_strategy = extract_config.profile_batch_strategy
    enable_token_estimation = extract_config.enable_token_estimation
    max_estimated_tokens = extract_config.max_estimated_tokens

    # 参数验证
    if max_batch_size <= 0:
        print(f"[ProfileExtract] ⚠️ 无效的批次大小 {max_batch_size}，使用默认值 50")
        max_batch_size = 50

    # Token 预估
    if enable_token_estimation:
        estimated_tokens = estimate_total_tokens(memcell_list)
        print(f"[ProfileExtract] 📏 Token 预估: ~{estimated_tokens:,} tokens")

        if estimated_tokens > max_estimated_tokens:
            print(
                f"[ProfileExtract] ⚠️ 预估 Token 数超过限制 ({max_estimated_tokens:,})，强制分批处理"
            )
            # 根据 token 数动态调整批次大小
            if estimated_tokens > 0:  # 防止除零错误
                adjusted_batch_size = int(
                    max_batch_size * (max_estimated_tokens / estimated_tokens)
                )
                # 确保批次大小在合理范围内（最小 5，最大不超过原始值）
                adjusted_batch_size = max(5, min(max_batch_size, adjusted_batch_size))
                max_batch_size = adjusted_batch_size
                print(f"[ProfileExtract] 🔧 调整批次大小: {max_batch_size}")

    # 步骤 4：分批处理
    batches = split_memcells_into_batches(
        memcell_list, max_batch_size=max_batch_size, strategy=batch_strategy
    )

    total_batches = len(batches)

    if total_batches == 1:
        print(f"[ProfileExtract] 🚀 MemCell 数量在限制内，使用单批次处理")
    else:
        print(
            f"[ProfileExtract] 📦 将 {len(memcell_list)} 个 MemCell 分为 {total_batches} 批处理"
        )
        print(f"[ProfileExtract] 📋 分批策略: {batch_strategy}")

    # 步骤 5：逐批提取 Profile（使用增量更新）
    all_profiles_map: Dict[str, Any] = {}  # user_id -> ProfileMemory
    total_llm_time = 0.0

    try:
        for batch_idx, batch_memcells in enumerate(batches, 1):
            batch_start = time.time()

            print(
                f"\n[ProfileExtract] 🔄 处理批次 {batch_idx}/{total_batches} ({len(batch_memcells)} 个 MemCell)"
            )

            # Token 预估（仅对当前批次）
            if enable_token_estimation:
                batch_tokens = estimate_total_tokens(batch_memcells)
                print(
                    f"[ProfileExtract]   📏 批次 Token 预估: ~{batch_tokens:,} tokens"
                )

            # 构建提取请求（使用增量更新）
            # old_memory_list 包含之前所有批次提取的 Profile
            old_memory_list = list(all_profiles_map.values())

            extract_request = ProfileMemoryExtractRequest(
                memcell_list=batch_memcells,
                user_id_list=[],  # 空列表，让 LLM 自动识别参与者
                group_id=group_id,
                group_name=group_name,
                old_memory_list=old_memory_list,  # 增量更新：传入之前批次的结果
            )

            # 执行提取
            print(f"[ProfileExtract]   📡 调用 LLM 提取 Profile...")
            llm_start = time.time()
            if extract_config.scenario_type == ScenarioType.GROUP_CHAT:
                batch_profile_memories = await profile_extractor.extract_memory(
                    extract_request
                )
            elif extract_config.scenario_type == ScenarioType.ASSISTANT:
                batch_profile_memories = (
                    await profile_extractor.extract_profile_companion(extract_request)
                )
            else:
                raise ValueError(
                    f"Invalid scenario type: {extract_config.scenario_type}"
                )
            llm_elapsed = time.time() - llm_start
            total_llm_time += llm_elapsed

            # 更新性能指标
            PERF_METRICS.llm_calls += 1
            PERF_METRICS.llm_total_time += llm_elapsed

            print(f"[ProfileExtract]   ⏱️  LLM 调用耗时: {llm_elapsed:.2f}s")

            # 验证结果
            if not batch_profile_memories:
                print(f"[ProfileExtract]   ⚠️ 批次 {batch_idx} 未返回 Profile")
                continue

            print(
                f"[ProfileExtract]   ✅ 批次 {batch_idx} 返回 {len(batch_profile_memories)} 个 Profile"
            )

            # 合并结果（按 user_id 去重，后批次覆盖前批次）
            for profile in batch_profile_memories:
                user_id = getattr(profile, 'user_id', None)
                if user_id:
                    all_profiles_map[user_id] = profile
                    print(f"[ProfileExtract]     ✓ 更新 {user_id} 的 Profile")

            batch_elapsed = time.time() - batch_start
            print(
                f"[ProfileExtract]   📊 批次 {batch_idx} 完成，耗时: {batch_elapsed:.2f}s"
            )

        # 步骤 6：保存所有 Profile
        print(f"\n[ProfileExtract] 💾 保存 Profile 到文件...")
        profile_count = 0
        saved_user_ids = []

        for user_id, profile in all_profiles_map.items():
            await save_individual_profile_to_file(
                profile=profile, user_id=user_id, output_dir=output_dir
            )

            profile_count += 1
            saved_user_ids.append(user_id)
            PERF_METRICS.profile_count += 1

            print(f"[ProfileExtract]   ✓ 保存 {user_id} 的 Profile")

        # 步骤 7：生成报告
        print(f"\n[ProfileExtract] 📋 提取结果汇总:")
        print(f"  - 处理批次: {total_batches} 批")
        print(f"  - LLM 总耗时: {total_llm_time:.2f}s")
        print(f"  - 成功提取: {profile_count}/{len(all_users)} 个 Profile")
        print(f"  - 已保存用户: {sorted(saved_user_ids)}")

        # 检查是否有用户未被提取
        missing_users = all_users - set(saved_user_ids)
        if missing_users:
            print(f"  - ⚠️ 未提取到的用户: {sorted(missing_users)}")
            print(f"     提示: 这些用户可能在对话中没有足够的信息")

        return profile_count

    except Exception as e:
        print(f"[ProfileExtract] ❌ Profile 提取失败: {e}")
        import traceback

        traceback.print_exc()
        return 0


# ============================================================================
# 核心业务逻辑 - Extract 模式
# ============================================================================


async def extract_and_dump(
    events: List[Dict[str, Any]],
    extract_config: ExtractModeConfig,
    llm_config: LLMConfig,
    mongo_config: MongoDBConfig,
) -> Dict[str, int]:
    """从对话事件中提取 MemCell 并保存（优化版）

    这是 Extract 模式的核心处理流程：
    1. 初始化 MongoDB 和 Beanie
    2. 创建 LLM Provider 和 MemCell 提取器
    3. 流式处理对话消息
    4. 检测话题边界并提取 MemCell
    5. 自动生成情景记忆
    6. 批量保存到 MongoDB 和本地 JSON（优化版新增）
    7. 提交到后台聚类 worker
    8. 批量提取个人 Profile（从群组 MemCell）
    9. 生成性能报告（优化版新增）

    Args:
        events: 原始对话事件列表
        extract_config: 提取配置
        llm_config: LLM 配置
        mongo_config: MongoDB 配置

    Returns:
        包含统计信息的字典（saved_files、profile_count 等）
    """

    # 记录开始时间
    if extract_config.enable_performance_metrics:
        PERF_METRICS.total_start_time = time.time()

    # 初始化 MongoDB 连接
    await ensure_mongo_beanie_ready(mongo_config)

    # 创建 LLM Provider
    provider = LLMProvider(
        llm_config.provider,
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
    )

    # 创建 MemCell 提取器
    extractor = ConvMemCellExtractor(provider)
    # 尝试从输出目录加载历史聚类状态（可选）
    try:
        extractor.cluster_worker.load_state_from_dir(str(extract_config.output_dir))
        print(f"[Extract] ♻️ 已加载历史聚类状态: {extract_config.output_dir}")
    except Exception:
        pass

    # 创建批量 MongoDB 写入器（优化版新增）
    mongo_writer = BatchMongoWriter(
        batch_size=extract_config.mongo_batch_size,
        perf_metrics=(
            PERF_METRICS if extract_config.enable_performance_metrics else None
        ),
    )

    # 创建 Profile 提取器（如果启用）
    profile_extractor = None
    if extract_config.enable_profile_extraction:
        try:
            # 群聊/陪伴场景共用 Profile 提取器
            profile_extractor = ProfileMemoryExtractor(llm_provider=provider)
            print(f"[Extract] ✅ Profile 提取器已创建")
        except Exception as e:
            print(f"[Extract] ⚠️ Profile 提取器创建失败: {e}")
            print(f"[Extract] 继续执行 MemCell 提取，跳过 Profile 提取")

    history: List[RawData] = []
    saved_files = 0
    profile_count = 0

    # 在线画像：每条 MemCell 完成后触发（替代原来的批量提取）
    # 维护在线画像缓存与簇内 MemCell 缓冲
    # 注意：为最小改动，不引入判别/去抖，后续可加
    memcell_batch: List = []  # 保留但不使用，用于兼容逻辑
    online_profiles_map: Dict[str, Any] = {}
    cluster_to_memcells: Dict[str, List[Any]] = {}
    session_memcells: List[Any] = []

    # Companion 场景价值判别器
    companion_disc: Optional[ValueDiscriminatorCompanion] = None
    if (
        extract_config.scenario_type == ScenarioType.ASSISTANT
        and profile_extractor is not None
    ):
        try:
            companion_disc = ValueDiscriminatorCompanion(
                provider, DiscriminatorCompanionConfig(min_confidence=0.6)
            )
        except Exception:
            companion_disc = None

    # 在线画像输出目录
    profiles_dir_group = extract_config.output_dir / "profiles"
    profiles_dir_group.mkdir(parents=True, exist_ok=True)
    profiles_dir_companion = extract_config.output_dir / "profiles_companion"
    profiles_dir_companion.mkdir(parents=True, exist_ok=True)

    print(f"[Extract] 开始处理 {len(events)} 个对话事件...")
    print(f"[Extract] Profile 提取策略: 所有 MemCell 完成后，为每个参与者提取个人画像")

    # 创建进度追踪器（优化版新增）
    progress_tracker = None
    if extract_config.enable_progress_bar:
        progress_tracker = ProgressTracker(len(events), "MemCell提取")

    for idx, entry in enumerate(events):
        # 1. 过滤不支持的消息类型
        if not _is_supported_msg_type(entry, extract_config.supported_msg_types):
            continue

        # 2. 归一化消息格式
        message_payload = normalize_entry(entry)
        if message_payload is None:
            continue

        message_id = (
            entry.get("message_id")
            or entry.get("id")
            or entry.get("uuid")
            or entry.get("event_id")
        )
        raw_item = RawData(
            content=message_payload,
            data_id=str(message_id or idx),
            data_type=DataTypeEnum.CONVERSATION,
        )

        # 3. 使用第一条有效消息初始化历史记录
        if not history:
            history.append(raw_item)
            continue

        # 4. 构建提取请求（单消息流式处理）
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=list(history),
            new_raw_data_list=[raw_item],
            user_id_list=[],
            group_id=extract_config.group_id,
            smart_mask_flag=True,
        )

        # 5. 执行 MemCell 提取（带性能追踪）
        try:
            llm_start = time.time()
            memcell, status = await extractor.extract_memcell(
                request,
                use_semantic_extraction=extract_config.enable_semantic_extraction,
            )
            llm_elapsed = time.time() - llm_start

            # 更新性能指标
            if extract_config.enable_performance_metrics:
                PERF_METRICS.llm_calls += 1
                PERF_METRICS.llm_total_time += llm_elapsed

            should_end = memcell is not None
            should_wait = bool(status.should_wait) if status is not None else False
        except json.JSONDecodeError as e:
            print(f"\n[Extract] ⚠️ JSON 解析错误，跳过当前消息: {e}")
            history.append(raw_item)
            if len(history) > extract_config.history_window_size:
                history = history[-extract_config.history_window_size :]
            if progress_tracker:
                progress_tracker.update()
            continue
        except Exception as e:
            print(f"\n[Extract] ⚠️ MemCell 提取失败: {e}")
            history.append(raw_item)
            if len(history) > extract_config.history_window_size:
                history = history[-extract_config.history_window_size :]
            if progress_tracker:
                progress_tracker.update()
            continue

        # 6. 根据提取结果处理
        if should_end:
            # 成功提取到 MemCell
            saved_files += 1

            # 更新性能指标
            if extract_config.enable_performance_metrics:
                PERF_METRICS.memcell_count += 1

            write_memcell_to_file(memcell, saved_files, extract_config.output_dir)

            # 批量保存到 MongoDB（优化版）
            # await mongo_writer.add(memcell)
            await _save_memcell_to_mongodb(memcell)
            # 保存聚类快照
            await _save_clustering_snapshot(
                extractor, extract_config.output_dir, extract_config.group_id
            )

            # 在线触发画像（替代原批量提取）
            if extract_config.enable_profile_extraction:
                if profile_extractor is not None:
                    if extract_config.scenario_type == ScenarioType.GROUP_CHAT:
                        # 获取簇ID
                        assignments = extractor.cluster_worker.get_assignments()
                        gid = extract_config.group_id or "__default__"
                        mapping = (
                            assignments.get(gid, {})
                            if isinstance(assignments, dict)
                            else {}
                        )
                        cluster_id = (
                            mapping.get(str(memcell.event_id))
                            if isinstance(mapping, dict)
                            else None
                        )
                        if not cluster_id:
                            cluster_id = f"cluster_{str(memcell.event_id)[:8]}"
                        # 累积该簇的 MemCell
                        bucket = cluster_to_memcells.setdefault(cluster_id, [])
                        bucket.append(memcell)

                        # 发起在线提取（与工作画像一致）
                        extract_request = ProfileMemoryExtractRequest(
                            memcell_list=bucket,
                            user_id_list=[],
                            group_id=extract_config.group_id,
                            group_name=extract_config.group_name,
                            old_memory_list=(
                                list(online_profiles_map.values())
                                if online_profiles_map
                                else None
                            ),
                        )
                        batch_profile_memories = await profile_extractor.extract_memory(
                            extract_request
                        )
                        if batch_profile_memories:
                            for profile in batch_profile_memories:
                                uid = getattr(profile, 'user_id', None)
                                if not uid:
                                    continue
                                online_profiles_map[uid] = profile
                                await save_individual_profile_to_file(
                                    profile=profile,
                                    user_id=uid,
                                    output_dir=profiles_dir_group,
                                )
                    elif extract_config.scenario_type == ScenarioType.ASSISTANT:
                        # 陪伴场景：先做价值判别（最近两条上下文），高价值才触发
                        session_memcells.append(memcell)
                        do_extract = True
                        if companion_disc is not None:
                            recent_ctx = (
                                session_memcells[-3:-1]
                                if len(session_memcells) >= 2
                                else []
                            )
                            ok, conf, reason = await companion_disc.is_high_value(
                                memcell, recent_ctx
                            )
                            do_extract = bool(ok)
                        if do_extract:
                            extract_request = ProfileMemoryExtractRequest(
                                memcell_list=list(session_memcells),
                                user_id_list=[],
                                group_id=extract_config.group_id,
                                group_name=extract_config.group_name,
                                old_memory_list=(
                                    list(online_profiles_map.values())
                                    if online_profiles_map
                                    else None
                                ),
                            )
                            batch_profile_memories = (
                                await profile_extractor.extract_profile_companion(
                                    extract_request
                                )
                            )
                            if batch_profile_memories:
                                for profile in batch_profile_memories:
                                    uid = getattr(profile, 'user_id', None)
                                    if not uid:
                                        continue
                                    online_profiles_map[uid] = profile
                                    await save_individual_profile_to_file(
                                        profile=profile,
                                        user_id=uid,
                                        output_dir=profiles_dir_companion,
                                    )

            # 重置历史记录
            history = [raw_item]

            # 更新进度追踪
            if progress_tracker:
                progress_tracker.update()
            continue

        if should_wait:
            # 等待更多消息来判断话题边界
            history.append(raw_item)
            if len(history) > extract_config.history_window_size:
                history = history[-extract_config.history_window_size :]
            if progress_tracker:
                progress_tracker.update()
            continue

        # if idx == len(events) - 1:

        # 继续当前话题，增长历史记录（带窗口限制）
        history.append(raw_item)
        if len(history) > extract_config.history_window_size:
            history = history[-extract_config.history_window_size :]
        if progress_tracker:
            progress_tracker.update()

    # 7. 刷新剩余的 MongoDB 写入（优化版）
    await mongo_writer.flush()
    print(f"[Extract] ✅ 写入完成")
    # await _save_memcell_to_mongodb(memcell_batch)

    # 8. 允许后台 worker 完成最后的处理
    await asyncio.sleep(0.5)

    # 8. 保存最终聚类结果
    await _save_clustering_snapshot(
        extractor, extract_config.output_dir, extract_config.group_id
    )

    # 9. 旧的批量 Profile 提取已由在线模式替代；此处跳过
    profile_count = len(online_profiles_map)

    print(f"\n[Extract] ✅ 提取完成！")
    print(f"  - MemCell: {saved_files} 个")
    if profile_extractor is not None:
        print(f"  - Profile: {profile_count} 个（个人画像）")
    print(f"  - 输出目录: {extract_config.output_dir}")

    # 生成性能报告（优化版）
    if extract_config.enable_performance_metrics:
        PERF_METRICS.report()

    return {"saved_files": saved_files, "profile_count": profile_count}


async def extract_profiles_only(
    extract_config: ExtractModeConfig, llm_config: LLMConfig
) -> Dict[str, int]:
    """仅提取 Profile 模式 - 从已有 MemCell 中提取个人 Profile

    适用场景：已经生成了 MemCell 文件，现在需要提取 Profile

    Args:
        extract_config: 提取配置
        llm_config: LLM 配置

    Returns:
        包含统计信息的字典
    """
    print(f"\n[ExtractProfileOnly] 开始从已有 MemCell 提取个人 Profile...")
    print(f"  - MemCell 来源: {extract_config.memcell_source}")

    # 1. 加载已有 MemCell
    if extract_config.memcell_source == "file":
        memcells = load_memcells_from_files(extract_config.memcell_input_dir)
    else:
        print(
            f"[ExtractProfileOnly] ❌ 未知的 MemCell 来源: {extract_config.memcell_source}"
        )
        return {"profile_count": 0}

    if not memcells:
        print(f"[ExtractProfileOnly] ❌ 未加载到 MemCell，退出")
        return {"profile_count": 0}

    # 2. 创建 LLM Provider 和 Profile 提取器
    provider = LLMProvider(
        llm_config.provider,
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
    )

    try:
        profile_extractor = ProfileMemoryExtractor(llm_provider=provider)
        print(f"[ExtractProfileOnly] ✅ Profile 提取器已创建")
    except Exception as e:
        print(f"[ExtractProfileOnly] ❌ Profile 提取器创建失败: {e}")
        return {"profile_count": 0}

    # 3. 提取个人 Profile
    profile_count = await extract_individual_profiles_from_group_memcells(
        memcell_list=memcells,
        profile_extractor=profile_extractor,
        group_id=extract_config.group_id,
        group_name=extract_config.group_name,
        output_dir=extract_config.output_dir,
        extract_config=extract_config,  # 传入配置以启用分批优化
    )

    print(f"\n[ExtractProfileOnly] ✅ Profile 提取完成!")
    print(f"  - 输入 MemCell: {len(memcells)} 个")
    print(f"  - 提取 Profile: {profile_count} 个（个人画像）")
    print(f"  - 输出目录: {extract_config.output_dir}")

    return {"profile_count": profile_count}


# ============================================================================
# 主入口函数
# ============================================================================


async def run_extract_mode(
    extract_config: ExtractModeConfig,
    llm_config: LLMConfig,
    mongo_config: MongoDBConfig,
) -> None:
    """运行 Extract 模式

    Args:
        extract_config: 提取配置
        llm_config: LLM 配置
        mongo_config: MongoDB 配置
    """
    # 1. 验证 LLM API Key
    api_key_present = any(
        [
            llm_config.api_key,
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ]
    )
    if not api_key_present:
        print("[Extract] ❌ LLM_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY 未设置")
        print("[Extract] 无法运行 LLM 提取，请配置 API 密钥")
        return

    # 2. 验证输入文件
    if not extract_config.data_file.exists():
        print(f"[Extract] ❌ 输入文件不存在: {extract_config.data_file}")
        return

    # 3. 准备输出目录
    ensure_output_dir(extract_config.output_dir, clean_previous=True)

    # 4. 加载对话事件
    print(f"[Extract] 加载对话数据: {extract_config.data_file}")
    events = load_events(extract_config.data_file)

    # 5. 执行提取
    result = await extract_and_dump(events, extract_config, llm_config, mongo_config)
    saved_memcells = result.get("saved_files", 0)
    saved_profiles = result.get("profile_count", 0)

    print(f"\n[Extract] ✅ 提取完成！")
    print(f"  - 保存了 {saved_memcells} 个 MemCell")
    print(f"  - 保存了 {saved_profiles} 个 Profile（个人画像）")
    print(f"  - 输出目录: {extract_config.output_dir}")


async def run_extract_profile_only_mode(
    extract_config: ExtractModeConfig, llm_config: LLMConfig
) -> None:
    """运行仅提取 Profile 模式

    Args:
        extract_config: 提取配置
        llm_config: LLM 配置
    """
    # 验证 LLM API Key
    api_key_present = any(
        [
            llm_config.api_key,
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ]
    )
    if not api_key_present:
        print("[ExtractProfileOnly] ❌ LLM_API_KEY 未设置")
        return

    # 验证输入目录（file 模式）
    if (
        extract_config.memcell_source == "file"
        and not extract_config.memcell_input_dir.exists()
    ):
        print(
            f"[ExtractProfileOnly] ❌ MemCell 输入目录不存在: {extract_config.memcell_input_dir}"
        )
        return

    # 准备输出目录
    extract_config.output_dir.mkdir(parents=True, exist_ok=True)

    # 执行提取
    result = await extract_profiles_only(extract_config, llm_config)
    profile_count = result.get("profile_count", 0)

    print(f"\n[ExtractProfileOnly] ✅ 完成！")
    print(f"  - 提取了 {profile_count} 个 Profile（个人画像）")


def main() -> None:
    """主入口函数"""
    print("=" * 80)
    print("记忆提取工具 - MemCell & Profile Extractor")
    print("=" * 80)
    print(f"运行模式: {CURRENT_RUN_MODE.value}")
    print(f"场景类型: {EXTRACT_CONFIG.scenario_type.value}")
    print("=" * 80 + "\n")

    try:
        if CURRENT_RUN_MODE in (RunMode.EXTRACT, RunMode.EXTRACT_ALL):
            # 完整提取模式
            asyncio.run(run_extract_mode(EXTRACT_CONFIG, LLM_CONFIG, MONGO_CONFIG))

        elif CURRENT_RUN_MODE == RunMode.EXTRACT_MEMCELL_ONLY:
            # 仅提取 MemCell 模式
            print("[Info] 仅提取 MemCell 模式 - Profile 提取已禁用")
            EXTRACT_CONFIG.enable_profile_extraction = False
            asyncio.run(run_extract_mode(EXTRACT_CONFIG, LLM_CONFIG, MONGO_CONFIG))

        elif CURRENT_RUN_MODE == RunMode.EXTRACT_PROFILE_ONLY:
            # 仅提取 Profile 模式
            asyncio.run(run_extract_profile_only_mode(EXTRACT_CONFIG, LLM_CONFIG))

        else:
            print(f"[Error] 未知的运行模式: {CURRENT_RUN_MODE}")

    except KeyboardInterrupt:
        print("\n\n[Info] 用户中断，退出程序")
    except Exception as e:
        print(f"\n[Error] 程序执行失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
