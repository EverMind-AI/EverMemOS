"""
恢复 Stage2 Embedding 生成（从断点继续）

如果 stage2_index_building.py 中途卡住，使用此脚本从断点继续。
"""

import json
import os
import sys
import pickle
import asyncio
import time
from pathlib import Path

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from evaluation.locomo_evaluation.config import ExperimentConfig
from src.agentic_layer import vectorize_service


async def resume_build_emb_index(config: ExperimentConfig, data_dir: Path, emb_save_dir: Path, start_from: int = 0):
    """
    从指定的 conversation 开始构建 embedding 索引
    
    Args:
        config: 实验配置
        data_dir: memcells 数据目录
        emb_save_dir: embedding 保存目录
        start_from: 从哪个 conversation 开始（0-based）
    """
    # 🔥 优化后的参数
    BATCH_SIZE = 100  # 更小的批次，更多并发机会
    MAX_CONCURRENT_BATCHES = 10
    
    print(f"\n{'='*60}")
    print(f"Resuming Embedding Generation from Conv {start_from}")
    print(f"{'='*60}\n")

    for i in range(start_from, config.num_conv):
        file_path = data_dir / f"memcell_list_conv_{i}.json"
        if not file_path.exists():
            print(f"Warning: File not found, skipping: {file_path}")
            continue

        # 检查是否已完成
        output_path = emb_save_dir / f"embedding_index_conv_{i}.pkl"
        if output_path.exists():
            print(f"✅ Conv {i} already completed, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {file_path.name} for embedding...")
        print(f"{'='*60}")

        with open(file_path, "r", encoding="utf-8") as f:
            original_docs = json.load(f)

        texts_to_embed = []
        doc_field_map = []
        for doc_idx, doc in enumerate(original_docs):
            # 优先使用event_log（如果存在）
            if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
                atomic_facts = doc["event_log"]["atomic_fact"]
                if isinstance(atomic_facts, list) and atomic_facts:
                    for fact_idx, fact in enumerate(atomic_facts):
                        if fact and isinstance(fact, str) and fact.strip():
                            texts_to_embed.append(fact)
                            doc_field_map.append((doc_idx, f"atomic_fact_{fact_idx}"))
                    continue

            # 回退到原有字段（保持向后兼容）
            for field in ["subject", "summary", "episode"]:
                if text := doc.get(field):
                    texts_to_embed.append(text)
                    doc_field_map.append((doc_idx, field))

        if not texts_to_embed:
            print(f"Warning: No documents found in {file_path.name}. Skipping embedding creation.")
            continue

        total_texts = len(texts_to_embed)
        total_batches = (total_texts + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Total texts to embed: {total_texts}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Total batches: {total_batches}")
        print(f"Max concurrent batches: {MAX_CONCURRENT_BATCHES}")
        print(f"\nStarting parallel embedding generation...")
        
        # 🔥 并发批次处理
        start_time = time.time()
        
        async def process_batch(batch_idx: int, batch_texts: list) -> tuple[int, list]:
            """处理单个批次（异步）"""
            try:
                batch_embeddings = await vectorize_service.get_text_embeddings(batch_texts)
                print(f"  ✓ Batch {batch_idx + 1}/{total_batches} complete ({len(batch_texts)} texts)")
                return (batch_idx, batch_embeddings)
            except Exception as e:
                print(f"  ❌ Batch {batch_idx + 1}/{total_batches} failed: {e}")
                return (batch_idx, [])
        
        # 创建所有批次任务
        tasks = []
        for j in range(0, total_texts, BATCH_SIZE):
            batch_idx = j // BATCH_SIZE
            batch_texts = texts_to_embed[j : j + BATCH_SIZE]
            task = process_batch(batch_idx, batch_texts)
            tasks.append(task)
        
        print(f"Submitting {len(tasks)} batches for concurrent processing...")
        
        # 分批提交任务（避免内存问题）
        batch_results = []
        completed = 0
        chunk_size = MAX_CONCURRENT_BATCHES * 2
        
        for chunk_start in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[chunk_start : chunk_start + chunk_size]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=False)
            batch_results.extend(chunk_results)
            
            completed += len(chunk_tasks)
            progress = (completed / len(tasks)) * 100
            print(f"  Progress: {completed}/{len(tasks)} batches ({progress:.1f}%)")
        
        # 按批次顺序重组结果
        all_embeddings = []
        for batch_idx, batch_embeddings in sorted(batch_results, key=lambda x: x[0]):
            all_embeddings.extend(batch_embeddings)
        
        elapsed_time = time.time() - start_time
        speed = total_texts / elapsed_time if elapsed_time > 0 else 0
        print(f"\n✅ Embedding generation complete!")
        print(f"   - Total texts: {total_texts}")
        print(f"   - Total embeddings: {len(all_embeddings)}")
        print(f"   - Time elapsed: {elapsed_time:.2f}s")
        print(f"   - Speed: {speed:.1f} texts/sec")
        print(f"   - Average batch time: {elapsed_time/total_batches:.2f}s")
        
        # 验证结果完整性
        if len(all_embeddings) != total_texts:
            print(f"   ⚠️  Warning: Expected {total_texts} embeddings, got {len(all_embeddings)}")
        else:
            print(f"   ✓ All embeddings generated successfully")

        # 重组 embeddings
        doc_embeddings = [{"doc": doc, "embeddings": {}} for doc in original_docs]
        
        for (doc_idx, field), emb in zip(doc_field_map, all_embeddings):
            if field.startswith("atomic_fact_"):
                if "atomic_facts" not in doc_embeddings[doc_idx]["embeddings"]:
                    doc_embeddings[doc_idx]["embeddings"]["atomic_facts"] = []
                doc_embeddings[doc_idx]["embeddings"]["atomic_facts"].append(emb)
            else:
                doc_embeddings[doc_idx]["embeddings"][field] = emb

        # 保存结果
        emb_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving embeddings to: {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(doc_embeddings, f)
        
        print(f"✅ Conv {i} completed and saved!")


async def main():
    """主函数"""
    config = ExperimentConfig()
    data_dir = Path(__file__).parent / "results" / config.experiment_name / "memcells"
    emb_save_dir = Path(__file__).parent / "results" / config.experiment_name / "vectors"
    
    # 🔥 检查已完成的 conversation，自动从断点继续
    start_from = 0
    for i in range(config.num_conv):
        output_path = emb_save_dir / f"embedding_index_conv_{i}.pkl"
        if output_path.exists():
            start_from = i + 1
        else:
            break
    
    if start_from >= config.num_conv:
        print(f"✅ All conversations already completed!")
        return
    
    print(f"🔄 Resuming from Conv {start_from}")
    await resume_build_emb_index(config, data_dir, emb_save_dir, start_from=start_from)
    
    print(f"\n{'='*60}")
    print(f"✅ All embedding generation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

