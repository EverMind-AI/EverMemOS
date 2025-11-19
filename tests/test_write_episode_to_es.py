#!/usr/bin/env python3
"""
处理MongoDB备份的episodic_memories.json文件，将数据写入ES
"""

import json
import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Any
import sys
import os

from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from memory_layer.memory_manager import MemorizeRequest
from memory_layer.types import RawDataType
from agentic_layer.schemas import (
    Request,
    RequestType,
    MemorizeRequest as AgenticMemorizeRequest,
    RequestEntrypointType,
)
from agentic_layer.agentic_layer import AgenticLayer
from agentic_layer.memory_manager import MemoryManager
from agentic_layer.dispatcher import Dispatcher
from common_utils.datetime_utils import get_now_with_timezone


def convert_document_to_raw_data(doc: Dict[str, Any]) -> RawData:
    """
    将MongoDB文档转换为RawData格式

    Args:
        doc: MongoDB文档数据

    Returns:
        RawData对象
    """
    # 提取基本信息
    data_id = str(uuid.uuid4())  # 生成新的UUID作为data_id

    # 构建content字典，包含所有原始数据
    content = {
        "user_id": doc.get("user_id"),
        "group_id": doc.get("group_id"),
        "timestamp": (
            doc.get("timestamp", {}).get("$date") if doc.get("timestamp") else None
        ),
        "participants": doc.get("participants", []),
        "summary": doc.get("summary"),
        "subject": doc.get("subject"),
        "episode": doc.get("episode"),
        "type": doc.get("type"),
        "keywords": doc.get("keywords"),
        "linked_entities": doc.get("linked_entities"),
        "memcell_event_id_list": doc.get("memcell_event_id_list"),
        "extend": doc.get("extend"),
        "created_at": (
            doc.get("created_at", {}).get("$date") if doc.get("created_at") else None
        ),
        "updated_at": (
            doc.get("updated_at", {}).get("$date") if doc.get("updated_at") else None
        ),
    }

    # 构建metadata
    metadata = {
        "original_id": doc.get("_id", {}).get("$oid") if doc.get("_id") else None,
        "source": "mongodb_backup",
        "processed_at": get_now_with_timezone().isoformat(),
    }

    return RawData(
        content=content, data_id=data_id, data_type="Conversation", metadata=metadata
    )


def create_memorize_request_from_documents(
    documents: List[Dict[str, Any]]
) -> MemorizeRequest:
    """
    从文档列表创建MemorizeRequest

    Args:
        documents: MongoDB文档列表

    Returns:
        MemorizeRequest对象
    """
    # 转换所有文档为RawData
    raw_data_list = [convert_document_to_raw_data(doc) for doc in documents]

    # 提取所有参与者
    all_participants = set()
    for doc in documents:
        participants = doc.get("participants", [])
        if participants:
            all_participants.update(participants)

    # 获取第一个文档的group_id（假设所有文档都属于同一个group）
    group_id = documents[0].get("group_id") if documents else None

    return MemorizeRequest(
        history_raw_data_list=raw_data_list[:-1] if len(raw_data_list) > 1 else [],
        new_raw_data_list=raw_data_list[-1:] if raw_data_list else [],
        raw_data_type=RawDataType.CONVERSATION,
        user_id_list=list(all_participants),
        group_id=group_id,
        current_time=get_now_with_timezone(),
    )


async def process_episodic_memories(json_file_path: str, batch_size: int = 10):
    """
    处理episodic_memories.json文件

    Args:
        json_file_path: JSON文件路径
        batch_size: 批处理大小
    """
    print(f"开始处理文件: {json_file_path}")

    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = data.get("documents", [])
    total_docs = len(documents)
    print(f"总共找到 {total_docs} 个文档")

    if total_docs == 0:
        print("没有找到文档，退出")
        return

    # 初始化AgenticLayer
    agentic_layer = AgenticLayer()

    # 分批处理文档
    processed_count = 0
    success_count = 0
    error_count = 0

    for i in range(0, total_docs, batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_docs + batch_size - 1) // batch_size

        print(f"\n处理批次 {batch_num}/{total_batches} ({len(batch_docs)} 个文档)")

        try:
            # 创建MemorizeRequest
            memorize_request = create_memorize_request_from_documents(batch_docs)

            # 创建Request对象
            request = Request(
                request_entrypoint_type=RequestEntrypointType.REST,
                request_type=RequestType.MEMORIZE,
                memorize_request=memorize_request,
            )

            # 调用dispatcher处理
            result = await agentic_layer.handle(request)

            if result.get("status") == "ok":
                saved_memories = result.get("saved_memories", [])
                print(f"批次 {batch_num} 处理成功，保存了 {len(saved_memories)} 个记忆")
                success_count += len(batch_docs)
            else:
                print(f"批次 {batch_num} 处理失败: {result}")
                error_count += len(batch_docs)

        except Exception as e:
            print(f"批次 {batch_num} 处理出错: {str(e)}")
            error_count += len(batch_docs)

        processed_count += len(batch_docs)
        print(f"已处理: {processed_count}/{total_docs}")

    print(f"\n处理完成!")
    print(f"总文档数: {total_docs}")
    print(f"成功处理: {success_count}")
    print(f"失败处理: {error_count}")


async def main():
    """主函数"""
    json_file_path = "/home/gongjie/test/b001-memsys/mongodb_backup/export_20250923_062646/episodic_memories.json"

    if not os.path.exists(json_file_path):
        print(f"文件不存在: {json_file_path}")
        return

    # 处理文件
    await process_episodic_memories(json_file_path, batch_size=5)


if __name__ == "__main__":
    asyncio.run(main())
