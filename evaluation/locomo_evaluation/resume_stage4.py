"""
从断点恢复 Stage4 Response Generation

如果 stage4_response.py 中途卡住，使用此脚本从最新的 checkpoint 恢复。
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from time import time

import pandas as pd
from openai import AsyncOpenAI

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.locomo_evaluation.config import ExperimentConfig
from evaluation.locomo_evaluation.prompts import (
    ANSWER_PROMPT_NEMORI,
    ANSWER_PROMPT_NEMORI_COT,
)


async def locomo_response(llm_client, llm_config, context: str, question: str, experiment_config: ExperimentConfig) -> str:
    """生成 LLM 响应"""
    if experiment_config.mode == "cot":
        prompt = ANSWER_PROMPT_NEMORI_COT.format(context=context, question=question)
    else:
        prompt = ANSWER_PROMPT_NEMORI.format(context=context, question=question)
    
    for i in range(experiment_config.max_retries):
        try:
            response = await llm_client.chat.completions.create(
                model=llm_config["model"],
                messages=[{"role": "system", "content": prompt}],
                temperature=0,
                max_tokens=4096,
            )
            result = response.choices[0].message.content or ""
            if experiment_config.mode == "cot":
                # 安全解析 FINAL ANSWER
                if "FINAL ANSWER:" in result:
                    parts = result.split("FINAL ANSWER:")
                    if len(parts) > 1:
                        result = parts[1].strip()
                    else:
                        result = result.strip()
                else:
                    result = result.strip()
            
            if result == "":
                continue
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    return result


async def process_qa(qa, search_result, oai_client, llm_config, experiment_config):
    """处理单个 QA 对"""
    start = time()
    query = qa.get("question")
    gold_answer = qa.get("answer")
    qa_category = qa.get("category")

    answer = await locomo_response(
        oai_client, llm_config, search_result.get("context"), query, experiment_config
    )

    response_duration_ms = (time() - start) * 1000

    return {
        "question": query,
        "answer": answer,
        "category": qa_category,
        "golden_answer": gold_answer,
        "search_context": search_result.get("context", ""),
        "response_duration_ms": response_duration_ms,
        "search_duration_ms": search_result.get("duration_ms", 0),
    }


def load_latest_checkpoint(results_dir: Path):
    """加载最新的 checkpoint"""
    checkpoint_files = list(results_dir.glob("responses_checkpoint_*.json"))
    
    if not checkpoint_files:
        return None, 0
    
    # 找到最新的 checkpoint（按完成数量排序）
    def get_checkpoint_number(path):
        name = path.stem  # responses_checkpoint_400
        return int(name.split("_")[-1])
    
    latest_checkpoint = max(checkpoint_files, key=get_checkpoint_number)
    checkpoint_number = get_checkpoint_number(latest_checkpoint)
    
    print(f"Found checkpoint: {latest_checkpoint.name} ({checkpoint_number} questions)")
    
    with open(latest_checkpoint, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)
    
    return checkpoint_data, checkpoint_number


async def main():
    """从断点恢复主函数"""
    config = ExperimentConfig()
    results_dir = Path(__file__).parent / "results" / config.experiment_name
    
    search_result_path = results_dir / "search_results.json"
    save_path = results_dir / "responses.json"
    
    # 🔥 加载最新的 checkpoint
    checkpoint_data, completed_count = load_latest_checkpoint(results_dir)
    
    if checkpoint_data is None:
        print("❌ No checkpoint found. Please run stage4_response.py from the beginning.")
        return
    
    print(f"\n{'='*60}")
    print(f"Resuming Stage4 from Checkpoint")
    print(f"{'='*60}")
    print(f"Completed: {completed_count} questions")
    
    # 初始化 LLM 客户端
    llm_config = config.llm_config["openai"]
    oai_client = AsyncOpenAI(
        api_key=llm_config["api_key"], base_url=llm_config["base_url"]
    )
    
    # 加载数据
    locomo_df = pd.read_json(config.datase_path)
    with open(search_result_path) as file:
        locomo_search_results = json.load(file)
    
    num_users = len(locomo_df)
    
    # 🔥 找出哪些问题已经完成
    completed_questions = set()
    for user_id, responses in checkpoint_data.items():
        for response in responses:
            completed_questions.add(response["question"])
    
    print(f"Already completed questions: {len(completed_questions)}")
    
    # 🔥 收集未完成的 QA 对
    MAX_CONCURRENT = 50
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    async def process_qa_with_semaphore(qa, search_result, group_id):
        async with semaphore:
            result = await process_qa(qa, search_result, oai_client, llm_config, config)
            return (group_id, result)
    
    all_tasks = []
    total_remaining = 0
    
    for group_idx in range(num_users):
        qa_set = locomo_df["qa"].iloc[group_idx]
        qa_set_filtered = [qa for qa in qa_set if qa.get("category") != 5]
        
        group_id = f"locomo_exp_user_{group_idx}"
        search_results = locomo_search_results.get(group_id)
        
        for qa in qa_set_filtered:
            question = qa.get("question")
            
            # 🔥 跳过已完成的问题
            if question in completed_questions:
                continue
            
            matching_result = next(
                (result for result in search_results if result.get("query") == question),
                None,
            )
            
            if matching_result:
                task = process_qa_with_semaphore(qa, matching_result, group_id)
                all_tasks.append(task)
                total_remaining += 1
    
    print(f"Remaining questions: {total_remaining}")
    print(f"Max concurrent requests: {MAX_CONCURRENT}")
    print(f"Estimated time: {total_remaining * 3 / MAX_CONCURRENT / 60:.1f} minutes")
    print(f"\n{'='*60}")
    print(f"Starting parallel processing...")
    print(f"{'='*60}\n")
    
    # 🔥 执行剩余任务
    import time as time_module
    start_time = time_module.time()
    completed_new = 0
    failed = 0
    
    # 从 checkpoint 数据开始
    all_responses = checkpoint_data
    
    CHUNK_SIZE = 200
    SAVE_INTERVAL = 400
    
    for chunk_start in range(0, len(all_tasks), CHUNK_SIZE):
        chunk_tasks = all_tasks[chunk_start : chunk_start + CHUNK_SIZE]
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
        
        for result in chunk_results:
            if isinstance(result, Exception):
                print(f"  ❌ Task failed: {result}")
                failed += 1
                continue
            
            group_id, qa_result = result
            all_responses[group_id].append(qa_result)
        
        completed_new += len(chunk_tasks)
        total_completed = completed_count + completed_new
        elapsed = time_module.time() - start_time
        speed = completed_new / elapsed if elapsed > 0 else 0
        eta = (total_remaining - completed_new) / speed if speed > 0 else 0
        
        print(f"Progress: {completed_new}/{total_remaining} ({completed_new/total_remaining*100:.1f}%) | "
              f"Total: {total_completed}/1540 | Speed: {speed:.1f} qa/s | Failed: {failed} | ETA: {eta/60:.1f} min")
        
        # 增量保存
        if (completed_new % SAVE_INTERVAL == 0) or (completed_new == total_remaining):
            temp_path = results_dir / f"responses_checkpoint_{total_completed}.json"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(all_responses, f, indent=2, ensure_ascii=False)
            print(f"  💾 Checkpoint saved: {temp_path.name}")
    
    elapsed_time = time_module.time() - start_time
    total_completed = completed_count + completed_new
    
    print(f"\n{'='*60}")
    print(f"✅ Resume complete!")
    print(f"   - Previously completed: {completed_count}")
    print(f"   - Newly completed: {completed_new}")
    print(f"   - Total completed: {total_completed}")
    print(f"   - Failed: {failed}")
    print(f"   - Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"   - Average speed: {completed_new/elapsed_time:.1f} qa/s")
    print(f"{'='*60}\n")
    
    # 保存最终结果
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)
        print(f"✅ Final results saved to: {save_path}")
    
    # 清理旧的 checkpoint 文件
    for checkpoint_file in results_dir.glob("responses_checkpoint_*.json"):
        checkpoint_file.unlink()
        print(f"  🗑️  Removed checkpoint: {checkpoint_file.name}")


if __name__ == "__main__":
    asyncio.run(main())

