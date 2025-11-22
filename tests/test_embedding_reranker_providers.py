import asyncio
import os
import numpy as np
from agentic_layer.vectorize_service import get_text_embedding
from agentic_layer.rerank_service import get_rerank_service

# ===== 环境配置 =====
os.environ["VECTORIZE_PROVIDER"] = "vllm"
os.environ["VECTORIZE_BASE_URL"] = "http://localhost:11000/v1"
os.environ["VECTORIZE_MODEL"] = "Qwen3-Embedding-4B"
os.environ["VECTORIZE_DIMENSIONS"] = "1024"
os.environ["VECTORIZE_API_KEY"] = "EMPTY"

os.environ["RERANK_PROVIDER"] = "vllm"
os.environ["RERANK_BASE_URL"] = "http://localhost:12000/score"
os.environ["RERANK_MODEL"] = "Qwen3-Reranker-4B"
os.environ["RERANK_API_KEY"] = "EMPTY"

# os.environ["VECTORIZE_PROVIDER"] = "deepinfra"
# os.environ["VECTORIZE_BASE_URL"] = "https://api.deepinfra.com/v1/openai"
# os.environ["VECTORIZE_MODEL"] = "Qwen/Qwen3-Embedding-4B"
# os.environ["VECTORIZE_DIMENSIONS"] = "1024"

# os.environ["RERANK_PROVIDER"] = "deepinfra"
# os.environ["RERANK_BASE_URL"] = "https://api.deepinfra.com/v1/inference"
# os.environ["RERANK_MODEL"] = "Qwen/Qwen3-Reranker-4B"


async def test_embedding():
    """测试 Embedding 并计算相似度"""
    print("\n=== 测试 Embedding ===")
    
    # 定义 instruction (用于 query)
    query_task = "Given a search query, retrieve relevant passages that answer the query"
    
    # 准备 Query (用户搜索查询)
    query = "水果"
    
    # 准备 Documents (文档内容)
    doc1 = "苹果很好吃"
    doc2 = "香蕉也是水果" 
    doc3 = "汽车速度很快"
    
    print(f"Query Task: {query_task}")
    print(f"Query: {query}")
    print(f"Documents: [{doc1}, {doc2}, {doc3}]")
    
    # Query: 使用 is_query=True
    print("\n--- Query Embedding (is_query=True) ---")
    query_emb = await get_text_embedding(query, instruction=query_task, is_query=True)
    print(f"Query 向量维度: {len(query_emb)}")
    print(f"配置的维度: 1024")
    if len(query_emb) == 1024:
        print("✅ Query 维度正确")
    else:
        print(f"❌ Query 维度不匹配！期望 1024，实际 {len(query_emb)}")
    
    # Documents: 使用 is_query=False (不加 instruction)
    print("\n--- Document Embeddings (is_query=False) ---")
    doc1_emb = await get_text_embedding(doc1, is_query=False)
    doc2_emb = await get_text_embedding(doc2, is_query=False)
    doc3_emb = await get_text_embedding(doc3, is_query=False)
    print(f"Document 向量维度: {len(doc1_emb)}")
    if len(doc1_emb) == 1024:
        print("✅ Document 维度正确")
    else:
        print(f"❌ Document 维度不匹配！期望 1024，实际 {len(doc1_emb)}")
    
    # 验证所有向量维度一致
    if len(query_emb) == len(doc1_emb) == len(doc2_emb) == len(doc3_emb) == 1024:
        print("\n✅ 所有向量维度一致 (1024)")
    else:
        print(f"\n❌ 向量维度不一致！Query:{len(query_emb)}, Doc1:{len(doc1_emb)}, Doc2:{len(doc2_emb)}, Doc3:{len(doc3_emb)}")
        return
    
    # 计算相似度 (Query vs Documents)
    def cos_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    sim_q_doc1 = cos_sim(query_emb, doc1_emb)
    sim_q_doc2 = cos_sim(query_emb, doc2_emb)
    sim_q_doc3 = cos_sim(query_emb, doc3_emb)
    
    print(f"\n相似度结果:")
    print(f"Query '{query}' vs Doc '{doc1}': {sim_q_doc1:.4f}")
    print(f"Query '{query}' vs Doc '{doc2}': {sim_q_doc2:.4f}")
    print(f"Query '{query}' vs Doc '{doc3}': {sim_q_doc3:.4f}")
    
    # 验证：doc2 (香蕉也是水果) 应该和 query (水果) 最相关
    if sim_q_doc2 > sim_q_doc1 and sim_q_doc2 > sim_q_doc3:
        print("✅ 相似度正常（'香蕉也是水果' 与 '水果' 最相关）")
    else:
        print("⚠️  相似度排序与预期不完全一致")


async def test_rerank():
    """测试 Rerank"""
    print("\n=== 测试 Rerank ===")
    
    query = "苹果"
    instruction = "Given a question and a passage, determine if the passage contains information relevant to answering the question."
    
    docs = [
        {"episode": "苹果很好吃"},
        {"episode": "汽车很快"},
        {"episode": "香蕉也是水果"}
    ]
    
    print(f"Query: {query}")
    print(f"Instruction: {instruction}")
    
    # 调用 rerank
    service = get_rerank_service()
    async with service:
        results = await service.rerank_memories(query, docs, instruction)
    
    # 打印结果
    print("Rerank 结果:")
    for r in results:
        score = r.get('_rerank_score', 0)
        text = r['episode']
        print(f"  {score:.4f} - {text}")


async def main():
    await test_embedding()
    await test_rerank()
    print("\n=== 测试完成 ===\n")


if __name__ == "__main__":
    asyncio.run(main())
