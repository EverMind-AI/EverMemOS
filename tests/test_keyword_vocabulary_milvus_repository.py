#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 KeywordVocabularyMilvusRepository 的功能

测试内容包括:
1. 基础CRUD操作（增删改查）
2. 向量相似度搜索
3. 按类型过滤搜索
4. 批量操作
5. 边界情况测试
"""

import asyncio
from typing import List
import numpy as np
from core.di import get_bean_by_type
from infra_layer.adapters.out.search.repository.keyword_vocabulary_milvus_repository import (
    KeywordVocabularyMilvusRepository,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


def generate_random_vector(dim: int = 1024) -> List[float]:
    """生成随机向量用于测试"""
    return np.random.randn(dim).astype(np.float32).tolist()


def generate_similar_vector(
    base_vector: List[float], noise_level: float = 0.1
) -> List[float]:
    """生成与基准向量相似的向量"""
    noise = np.random.normal(0, noise_level, len(base_vector))
    return [float(x + n) for x, n in zip(base_vector, noise)]


async def test_basic_crud_operations():
    """测试基础CRUD操作"""
    logger.info("========== 开始测试基础CRUD操作 ==========")

    repo = get_bean_by_type(KeywordVocabularyMilvusRepository)
    test_keyword = "机器学习"
    test_type = "technology"
    test_vector = generate_random_vector()
    test_model = "bge-m3"

    try:
        # 测试1: 创建关键词（Create）
        logger.info("测试1: 创建关键词")
        doc = await repo.create_and_save_keyword(
            keyword=test_keyword,
            keyword_type=test_type,
            vector=test_vector,
            vector_model=test_model,
        )

        assert doc is not None
        assert doc["keyword"] == test_keyword
        assert doc["type"] == test_type
        assert doc["vector_model"] == test_model
        logger.info("✅ 创建关键词成功: %s", test_keyword)

        # 等待数据刷新
        await repo.flush()
        await asyncio.sleep(1)

        # 测试2: 根据文本精确查询（Read）
        logger.info("测试2: 根据文本精确查询")
        retrieved_doc = await repo.get_keyword_by_text(
            keyword=test_keyword, keyword_type=test_type
        )

        assert retrieved_doc is not None
        assert retrieved_doc["keyword"] == test_keyword
        assert retrieved_doc["type"] == test_type
        assert retrieved_doc["vector_model"] == test_model
        logger.info("✅ 精确查询成功: %s", test_keyword)

        # 测试3: 根据ID删除（Delete）
        logger.info("测试3: 根据ID删除")
        keyword_id = retrieved_doc["id"]
        delete_result = await repo.delete_by_keyword_id(keyword_id)
        assert delete_result is True
        logger.info("✅ 删除关键词成功: id=%s", keyword_id)

        # 验证删除
        await repo.flush()
        await asyncio.sleep(1)
        deleted_check = await repo.get_keyword_by_text(test_keyword, test_type)
        assert deleted_check is None, "关键词应该已被删除"
        logger.info("✅ 验证删除成功")

    except Exception as e:
        logger.error("❌ 测试基础CRUD操作失败: %s", e)
        # 清理可能残留的数据
        try:
            await repo.delete_by_keyword_text(test_keyword, test_type)
            await repo.flush()
        except Exception:
            pass
        raise

    logger.info("✅ 基础CRUD操作测试完成\n")


async def test_vector_similarity_search():
    """测试向量相似度搜索"""
    logger.info("========== 开始测试向量相似度搜索 ==========")

    repo = get_bean_by_type(KeywordVocabularyMilvusRepository)
    test_type = "ai_concept"
    base_vector = generate_random_vector()
    test_model = "bge-m3"

    # 准备测试数据
    test_keywords = [
        {"keyword": "深度学习", "similarity": "high"},
        {"keyword": "神经网络", "similarity": "high"},
        {"keyword": "卷积神经网络", "similarity": "medium"},
        {"keyword": "强化学习", "similarity": "medium"},
        {"keyword": "自然语言处理", "similarity": "low"},
    ]

    try:
        # 创建测试关键词
        logger.info("创建测试关键词...")
        for kw_data in test_keywords:
            # 根据相似度级别生成向量
            if kw_data["similarity"] == "high":
                vector = generate_similar_vector(base_vector, noise_level=0.05)
            elif kw_data["similarity"] == "medium":
                vector = generate_similar_vector(base_vector, noise_level=0.15)
            else:
                vector = generate_random_vector()  # 完全不相似

            await repo.create_and_save_keyword(
                keyword=kw_data["keyword"],
                keyword_type=test_type,
                vector=vector,
                vector_model=test_model,
            )

        await repo.flush()
        await repo.load()
        await asyncio.sleep(2)
        logger.info("✅ 创建了 %d 个测试关键词", len(test_keywords))

        # 测试1: 基础向量搜索
        logger.info("\n测试1: 基础向量搜索（Top 3）")
        results = await repo.search_similar_keywords(
            query_vector=base_vector, keyword_type=test_type, limit=3
        )

        assert len(results) >= 2, f"应该找到至少2个相似关键词，实际找到{len(results)}个"
        logger.info("找到 %d 个相似关键词:", len(results))
        for i, result in enumerate(results, 1):
            logger.info("  %d. %s (score: %.4f)", i, result["keyword"], result["score"])

        # 验证最相似的应该是 high similarity 的关键词
        top_keywords = [r["keyword"] for r in results[:2]]
        high_similarity_keywords = [
            kw["keyword"] for kw in test_keywords if kw["similarity"] == "high"
        ]
        assert any(
            kw in top_keywords for kw in high_similarity_keywords
        ), "Top结果应该包含高相似度的关键词"
        logger.info("✅ 基础向量搜索测试通过")

        # 测试2: 返回所有结果
        logger.info("\n测试2: 返回所有结果")
        all_results = await repo.search_similar_keywords(
            query_vector=base_vector, keyword_type=test_type, limit=100
        )

        assert len(all_results) == len(
            test_keywords
        ), f"应该找到全部 {len(test_keywords)} 个关键词，实际找到 {len(all_results)} 个"
        logger.info("✅ 找到全部 %d 个关键词", len(all_results))

    except Exception as e:
        logger.error("❌ 测试向量相似度搜索失败: %s", e)
        raise
    finally:
        # 清理测试数据
        logger.info("\n清理测试数据...")
        try:
            delete_count = await repo.delete_by_type(test_type)
            await repo.flush()
            logger.info("✅ 清理了 %d 条测试数据", delete_count)
        except Exception as cleanup_error:
            logger.error("清理测试数据时出现错误: %s", cleanup_error)

    logger.info("✅ 向量相似度搜索测试完成\n")


async def test_type_filtering():
    """测试按类型过滤"""
    logger.info("========== 开始测试按类型过滤 ==========")

    repo = get_bean_by_type(KeywordVocabularyMilvusRepository)
    base_vector = generate_random_vector()
    test_model = "bge-m3"

    # 准备不同类型的测试数据
    test_data = [
        {"keyword": "Python", "type": "programming_language"},
        {"keyword": "JavaScript", "type": "programming_language"},
        {"keyword": "React", "type": "framework"},
        {"keyword": "Django", "type": "framework"},
        {"keyword": "MySQL", "type": "database"},
        {"keyword": "PostgreSQL", "type": "database"},
    ]

    try:
        # 创建测试数据
        logger.info("创建不同类型的测试关键词...")
        for data in test_data:
            vector = generate_similar_vector(base_vector, noise_level=0.1)
            await repo.create_and_save_keyword(
                keyword=data["keyword"],
                keyword_type=data["type"],
                vector=vector,
                vector_model=test_model,
            )

        await repo.flush()
        await repo.load()
        await asyncio.sleep(2)
        logger.info("✅ 创建了 %d 个不同类型的关键词", len(test_data))

        # 测试1: 搜索特定类型
        logger.info("\n测试1: 搜索 programming_language 类型")
        pl_results = await repo.search_similar_keywords(
            query_vector=base_vector, keyword_type="programming_language", limit=10
        )

        assert len(pl_results) == 2, f"应该找到2个编程语言，实际找到{len(pl_results)}个"
        for result in pl_results:
            assert result["type"] == "programming_language"
            logger.info("  - %s (type: %s)", result["keyword"], result["type"])
        logger.info("✅ 编程语言类型过滤成功")

        # 测试2: 搜索另一个类型
        logger.info("\n测试2: 搜索 framework 类型")
        fw_results = await repo.search_similar_keywords(
            query_vector=base_vector, keyword_type="framework", limit=10
        )

        assert len(fw_results) == 2, f"应该找到2个框架，实际找到{len(fw_results)}个"
        for result in fw_results:
            assert result["type"] == "framework"
            logger.info("  - %s (type: %s)", result["keyword"], result["type"])
        logger.info("✅ 框架类型过滤成功")

        # 测试3: 不指定类型（搜索全部）
        logger.info("\n测试3: 不指定类型（搜索全部）")
        all_results = await repo.search_similar_keywords(
            query_vector=base_vector, keyword_type=None, limit=10
        )

        assert (
            len(all_results) >= 6
        ), f"应该找到至少6个关键词，实际找到{len(all_results)}个"
        logger.info("找到 %d 个关键词（所有类型）", len(all_results))

        # 统计各类型数量
        type_counts = {}
        for result in all_results:
            result_type = result["type"]
            type_counts[result_type] = type_counts.get(result_type, 0) + 1

        logger.info("类型分布:")
        for kw_type, count in type_counts.items():
            logger.info("  - %s: %d", kw_type, count)
        logger.info("✅ 全类型搜索成功")

        # 测试4: 列出指定类型的所有关键词
        logger.info("\n测试4: 列出 database 类型的所有关键词")
        db_keywords = await repo.list_keywords_by_type("database")

        assert (
            len(db_keywords) == 2
        ), f"应该有2个数据库关键词，实际有{len(db_keywords)}个"
        for kw in db_keywords:
            logger.info("  - %s", kw["keyword"])
        logger.info("✅ 列出类型关键词成功")

    except Exception as e:
        logger.error("❌ 测试按类型过滤失败: %s", e)
        raise
    finally:
        # 清理测试数据
        logger.info("\n清理测试数据...")
        try:
            for kw_type in ["programming_language", "framework", "database"]:
                count = await repo.delete_by_type(kw_type)
                logger.info("  清理 %s: %d 条", kw_type, count)
            await repo.flush()
            logger.info("✅ 测试数据清理完成")
        except Exception as cleanup_error:
            logger.error("清理测试数据时出现错误: %s", cleanup_error)

    logger.info("✅ 按类型过滤测试完成\n")


async def test_batch_operations():
    """测试批量操作"""
    logger.info("========== 开始测试批量操作 ==========")

    repo = get_bean_by_type(KeywordVocabularyMilvusRepository)
    test_type = "batch_test"
    test_model = "bge-m3"

    # 准备批量测试数据
    batch_size = 50
    keywords_data = []

    for i in range(batch_size):
        keywords_data.append(
            {
                "keyword": f"关键词_{i}",
                "type": test_type,
                "vector": generate_random_vector(),
                "vector_model": test_model,
            }
        )

    try:
        # 测试批量创建
        logger.info("测试批量创建 %d 个关键词...", batch_size)
        count = await repo.batch_create_keywords(keywords_data)

        assert (
            count == batch_size
        ), f"应该创建 {batch_size} 个关键词，实际创建 {count} 个"
        logger.info("✅ 批量创建成功: %d 个关键词", count)

        # 刷新并验证
        await repo.flush()
        await asyncio.sleep(1)

        # 验证数据已保存
        logger.info("验证批量数据...")
        all_keywords = await repo.list_keywords_by_type(test_type)
        assert (
            len(all_keywords) >= batch_size
        ), f"应该至少有 {batch_size} 个关键词，实际有 {len(all_keywords)} 个"
        logger.info("✅ 数据验证成功: 找到 %d 个关键词", len(all_keywords))

        # 测试批量删除
        logger.info("测试批量删除...")
        delete_count = await repo.delete_by_type(test_type)
        assert (
            delete_count >= batch_size
        ), f"应该删除至少 {batch_size} 个关键词，实际删除 {delete_count} 个"
        logger.info("✅ 批量删除成功: %d 个关键词", delete_count)

        # 验证删除
        await repo.flush()
        await asyncio.sleep(1)
        remaining = await repo.list_keywords_by_type(test_type)
        assert (
            len(remaining) == 0
        ), f"删除后应该没有剩余数据，实际剩余 {len(remaining)} 个"
        logger.info("✅ 删除验证成功")

    except Exception as e:
        logger.error("❌ 测试批量操作失败: %s", e)
        raise
    finally:
        # 确保清理
        try:
            await repo.delete_by_type(test_type)
            await repo.flush()
        except Exception:
            pass

    logger.info("✅ 批量操作测试完成\n")


async def test_edge_cases():
    """测试边界情况"""
    logger.info("========== 开始测试边界情况 ==========")

    repo = get_bean_by_type(KeywordVocabularyMilvusRepository)
    test_type = "edge_test"
    test_model = "bge-m3"

    try:
        # 测试1: 空关键词
        logger.info("测试1: 空关键词处理")
        try:
            await repo.create_and_save_keyword(
                keyword="",
                keyword_type=test_type,
                vector=generate_random_vector(),
                vector_model=test_model,
            )
            logger.info("⚠️  空关键词被允许创建")
        except Exception as e:
            logger.info("✅ 正确拒绝空关键词: %s", e)

        # 测试2: 查询不存在的关键词
        logger.info("\n测试2: 查询不存在的关键词")
        nonexistent = await repo.get_keyword_by_text(
            keyword="这个关键词绝对不存在_12345", keyword_type=test_type
        )
        assert nonexistent is None, "不存在的关键词应该返回 None"
        logger.info("✅ 正确处理不存在的关键词")

        # 测试3: 删除不存在的关键词
        logger.info("\n测试3: 删除不存在的关键词")
        delete_result = await repo.delete_by_keyword_id("nonexistent_id_99999")
        assert delete_result is True  # Milvus delete 不存在的ID也返回True
        logger.info("✅ 删除不存在的关键词不报错")

        # 测试4: 错误的向量维度
        logger.info("\n测试4: 错误的向量维度")
        try:
            await repo.create_and_save_keyword(
                keyword="错误维度测试",
                keyword_type=test_type,
                vector=[1.0] * 512,  # 错误的维度
                vector_model=test_model,
            )
            assert False, "应该因为向量维度错误而失败"
        except Exception as e:
            assert "512" in str(e) and "1024" in str(e), "错误信息应该包含维度信息"
            logger.info("✅ 正确捕获向量维度错误: %s", str(e)[:100])

        # 测试5: 空类型搜索
        logger.info("\n测试5: 空类型（不存在的类型）搜索")
        empty_results = await repo.search_similar_keywords(
            query_vector=generate_random_vector(),
            keyword_type="这个类型不存在_99999",
            limit=10,
        )
        assert len(empty_results) == 0, "不存在的类型应该返回空结果"
        logger.info("✅ 空类型搜索返回空结果")

        # 测试6: 重复关键词（相同keyword和type）
        logger.info("\n测试6: 重复关键词")
        duplicate_keyword = "重复测试关键词"

        # 创建第一个
        await repo.create_and_save_keyword(
            keyword=duplicate_keyword,
            keyword_type=test_type,
            vector=generate_random_vector(),
            vector_model=test_model,
        )

        # 尝试创建重复的
        try:
            await repo.create_and_save_keyword(
                keyword=duplicate_keyword,
                keyword_type=test_type,
                vector=generate_random_vector(),
                vector_model=test_model,
            )
            logger.info("⚠️  重复关键词被允许创建（相同ID会覆盖）")
        except Exception as e:
            logger.info("✅ 正确拒绝重复关键词: %s", e)

        await repo.flush()

    except Exception as e:
        logger.error("❌ 测试边界情况失败: %s", e)
        raise
    finally:
        # 清理测试数据
        try:
            await repo.delete_by_type(test_type)
            await repo.flush()
            logger.info("✅ 边界测试数据清理完成")
        except Exception:
            pass

    logger.info("✅ 边界情况测试完成\n")


async def test_real_world_scenario():
    """测试真实世界场景：技能词汇表"""
    logger.info("========== 开始测试真实场景：技能词汇表 ==========")

    repo = get_bean_by_type(KeywordVocabularyMilvusRepository)
    test_model = "bge-m3"

    # 模拟真实的技能词汇表
    skills_data = {
        "programming": [
            "Python编程",
            "Java开发",
            "C++编程",
            "JavaScript开发",
            "Go语言",
            "Rust编程",
            "TypeScript开发",
        ],
        "framework": [
            "React框架",
            "Vue.js",
            "Django",
            "Spring Boot",
            "FastAPI",
            "Flask",
            "Express.js",
        ],
        "database": [
            "MySQL数据库",
            "PostgreSQL",
            "MongoDB",
            "Redis缓存",
            "Elasticsearch",
            "Cassandra",
        ],
        "devops": [
            "Docker容器",
            "Kubernetes",
            "Jenkins",
            "GitLab CI",
            "Terraform",
            "Ansible",
        ],
        "ai_ml": [
            "机器学习",
            "深度学习",
            "自然语言处理",
            "计算机视觉",
            "PyTorch",
            "TensorFlow",
            "scikit-learn",
        ],
    }

    # 为每个技能类别创建一个基准向量（同类技能更相似）
    category_base_vectors = {
        category: generate_random_vector() for category in skills_data.keys()
    }

    try:
        # 创建技能词汇表
        logger.info("创建技能词汇表...")
        total_skills = 0

        for category, skills in skills_data.items():
            base_vec = category_base_vectors[category]

            for skill in skills:
                # 同类技能使用相似向量
                vector = generate_similar_vector(base_vec, noise_level=0.08)

                await repo.create_and_save_keyword(
                    keyword=skill,
                    keyword_type=category,
                    vector=vector,
                    vector_model=test_model,
                )
                total_skills += 1

        await repo.flush()
        await repo.load()
        await asyncio.sleep(2)
        logger.info(
            "✅ 创建了 %d 个技能关键词，分为 %d 个类别", total_skills, len(skills_data)
        )

        # 场景1: 根据已知技能查找相似技能
        logger.info("\n场景1: 查找与 'Python编程' 相似的技能")
        python_doc = await repo.get_keyword_by_text("Python编程", "programming")
        assert python_doc is not None, "应该找到 Python编程"

        similar_to_python = await repo.search_similar_keywords(
            query_vector=python_doc["vector"], keyword_type=None, limit=5  # 不限制类型
        )

        logger.info("与 'Python编程' 最相似的5个技能:")
        for i, result in enumerate(similar_to_python, 1):
            logger.info(
                "  %d. %s [%s] (score: %.4f)",
                i,
                result["keyword"],
                result["type"],
                result["score"],
            )

        # 验证：最相似的应该主要是 programming 类别
        top_3_types = [r["type"] for r in similar_to_python[:3]]
        programming_count = sum(1 for t in top_3_types if t == "programming")
        assert programming_count >= 1, "Top 3 应该包含至少1个编程类技能"
        logger.info("✅ 相似技能查找符合预期")

        # 场景2: 按类别查找相似技能
        logger.info("\n场景2: 在 'ai_ml' 类别中查找与某个向量相似的技能")
        ai_base_vec = category_base_vectors["ai_ml"]
        query_vec = generate_similar_vector(ai_base_vec, noise_level=0.05)

        ai_results = await repo.search_similar_keywords(
            query_vector=query_vec, keyword_type="ai_ml", limit=3
        )

        logger.info("AI/ML 类别中最相似的3个技能:")
        for i, result in enumerate(ai_results, 1):
            logger.info("  %d. %s (score: %.4f)", i, result["keyword"], result["score"])

        assert len(ai_results) >= 3, "应该找到至少3个AI/ML技能"
        logger.info("✅ 类别过滤搜索成功")

        # 场景3: 列出某个类别的所有技能
        logger.info("\n场景3: 列出 'database' 类别的所有技能")
        db_skills = await repo.list_keywords_by_type("database")

        logger.info("数据库类技能（共 %d 个）:", len(db_skills))
        for skill in db_skills:
            logger.info("  - %s", skill["keyword"])

        expected_count = len(skills_data["database"])
        assert (
            len(db_skills) == expected_count
        ), f"应该有 {expected_count} 个数据库技能，实际有 {len(db_skills)} 个"
        logger.info("✅ 类别列表查询成功")

        # 场景4: 跨类别搜索
        logger.info("\n场景4: 跨类别搜索（查找所有与开发相关的技能）")
        dev_query_vec = category_base_vectors["programming"]

        all_related = await repo.search_similar_keywords(
            query_vector=dev_query_vec, keyword_type=None, limit=10
        )

        logger.info("与开发最相关的10个技能（跨类别）:")
        for i, result in enumerate(all_related, 1):
            logger.info(
                "  %d. %s [%s] (score: %.4f)",
                i,
                result["keyword"],
                result["type"],
                result["score"],
            )

        assert len(all_related) == 10, "应该返回10个结果"
        logger.info("✅ 跨类别搜索成功")

        # 场景5: 统计各类别数量
        logger.info("\n场景5: 统计各类别技能数量")
        category_counts = {}

        for category in skills_data.keys():
            keywords = await repo.list_keywords_by_type(category)
            category_counts[category] = len(keywords)

        logger.info("各类别技能统计:")
        for category, count in category_counts.items():
            expected = len(skills_data[category])
            logger.info("  - %s: %d (预期: %d)", category, count, expected)
            assert count == expected, f"{category} 类别数量不匹配"

        logger.info("✅ 统计验证成功")

    except Exception as e:
        logger.error("❌ 测试真实场景失败: %s", e)
        raise
    finally:
        # 清理所有测试数据
        logger.info("\n清理所有技能数据...")
        try:
            total_deleted = 0
            for category in skills_data.keys():
                count = await repo.delete_by_type(category)
                total_deleted += count
                logger.info("  清理 %s: %d 条", category, count)
            await repo.flush()
            logger.info("✅ 总共清理了 %d 条数据", total_deleted)
        except Exception as cleanup_error:
            logger.error("清理测试数据时出现错误: %s", cleanup_error)

    logger.info("✅ 真实场景测试完成\n")


async def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("🚀 开始运行 KeywordVocabularyMilvusRepository 所有测试")
    logger.info("=" * 60 + "\n")

    try:
        await test_basic_crud_operations()
        await test_vector_similarity_search()
        await test_type_filtering()
        await test_batch_operations()
        await test_edge_cases()
        await test_real_world_scenario()

        logger.info("=" * 60)
        logger.info("✅ 所有测试完成！")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ 测试过程中出现错误: %s", e)
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
