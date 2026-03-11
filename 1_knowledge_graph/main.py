#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
氢能专利知识图谱构建系统 - 主程序

包含增强功能：
1. 增强实体消解（向量粗筛 + LLM决策）
2. 地点信息提取（中国详细到省市区，外国只到国家）
3. Agentic RAG内嵌式补充检索
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import RAW_DATA_DIR, OUTPUT_DIR, CACHE_DIR
from data_pipeline.loader import DataLoader
from data_pipeline.entity_resolver import EntityResolver
from data_pipeline.enhanced_entity_resolver import UnifiedEntityResolver
from data_pipeline.cypher_generator import CypherGenerator
from vector.searcher import VectorSearcher
from graph_db.importer import Neo4jImporter
from utils.logger import get_logger
from utils.cache import get_cache_manager

logger = get_logger(__name__)


def run_step1_load_data(use_cache: bool = True):
    """Step 1: 加载数据"""
    logger.info("=" * 60)
    logger.info("[Step 1] 数据加载")
    logger.info("=" * 60)

    loader = DataLoader()
    records, stats = loader.load_all(use_cache=use_cache)

    logger.info(f"  总记录数: {stats.get('total_records', len(records))}")
    logger.info(f"  技术领域分布: {dict(stats.get('tech_domains', {}))}")
    logger.info(f"  唯一申请人: {stats.get('unique_applicants', 0)}")

    # 提取所有实体（不区分类型）
    all_entities = loader.extract_all_entities(records)
    logger.info(f"  提取实体: {len(all_entities)} 个（统一处理）")

    # 保存实体列表
    cache = get_cache_manager()
    cache.save_json("step1_all_entities.json", {
        "entities": list(all_entities)
    })

    return records, all_entities


def run_step2_entity_resolution(records, all_entities, use_cache: bool = True):
    """
    Step 2: 统一实体处理（消解 + 类型判断 + 地点提取）

    Args:
        records: 专利记录列表
        all_entities: 所有实体名称集合（不区分类型）
        use_cache: 是否使用缓存
    """
    logger.info("=" * 60)
    logger.info("[Step 2] 统一实体处理（消解 + 类型 + 地点）")
    logger.info("=" * 60)

    cache = get_cache_manager()

    # 检查缓存
    cache_key = "step2_unified_mapping.json"
    if use_cache and cache.exists(cache_key):
        logger.info("从缓存加载统一映射表...")
        unified_mapping = cache.load_json(cache_key, {})

        if unified_mapping:
            logger.info(f"  缓存加载成功: {len(unified_mapping)} 个实体")
            # 应用消解
            resolver = UnifiedEntityResolver()
            resolved_records = resolver.apply_resolution(records, unified_mapping)
            return resolved_records, unified_mapping

    # 统一实体处理
    resolver = UnifiedEntityResolver()
    unified_mapping = resolver.resolve_all(all_entities)

    # 统计
    org_count = sum(1 for v in unified_mapping.values() if v.get("entity_type") == "organization")
    person_count = sum(1 for v in unified_mapping.values() if v.get("entity_type") == "person")
    with_location = sum(1 for v in unified_mapping.values() if v.get("location"))

    logger.info(f"  机构: {org_count}, 人物: {person_count}")
    logger.info(f"  有地点信息: {with_location}")

    # 应用消解
    resolved_records = resolver.apply_resolution(records, unified_mapping)

    # 保存缓存
    cache.save_json(cache_key, unified_mapping)
    # 同时保存包含resolved字段的records
    cache.save_json("step2_resolved_records.json", resolved_records)
    logger.info("  已保存resolved_records缓存")

    return resolved_records, unified_mapping


def run_step3_generate_cypher(records, unified_mapping):
    """
    Step 3: 生成Cypher DSL

    Args:
        records: 专利记录列表（已应用实体消解）
        unified_mapping: 统一实体映射表
    """
    logger.info("=" * 60)
    logger.info("[Step 3] 生成Cypher DSL")
    logger.info("=" * 60)

    generator = CypherGenerator()
    stats = generator.generate_all(
        records,
        unified_mapping
    )

    total = sum(stats.values())
    logger.info(f"  生成 {len(stats)} 个文件, 共 {total} 条语句")

    return stats


def run_step4_build_vector_index(records):
    """Step 4: 构建向量索引"""
    logger.info("=" * 60)
    logger.info("[Step 4] 构建向量索引")
    logger.info("=" * 60)

    searcher = VectorSearcher()
    searcher.build_index(records, save=True)

    logger.info(f"  索引大小: {searcher.size} 向量")


def run_step5_import_neo4j(resume: bool = True, clear: bool = False):
    """Step 5: 导入Neo4j"""
    logger.info("=" * 60)
    logger.info("[Step 5] 导入Neo4j")
    logger.info("=" * 60)

    importer = Neo4jImporter()

    if clear:
        logger.warning("清空数据库...")
        importer.clear_database(confirm=True)

    results = importer.import_all(resume=resume)

    total_success = sum(r.get("success", 0) for r in results.values())
    total_failed = sum(r.get("failed", 0) for r in results.values())

    logger.info(f"  导入完成: 成功 {total_success}, 失败 {total_failed}")


def main():
    parser = argparse.ArgumentParser(description="氢能专利知识图谱构建系统")
    parser.add_argument("--step", type=int, nargs="+",
                        help="执行指定步骤 (1-5), 不指定则执行全部")
    parser.add_argument("--no-cache", action="store_true",
                        help="不使用缓存，从头开始处理")
    parser.add_argument("--clear", action="store_true",
                        help="清空Neo4j数据库后导入")
    parser.add_argument("--data-dir", type=str,
                        help="原始数据目录")
    args = parser.parse_args()

    use_cache = not args.no_cache
    steps = args.step or [1, 2, 3, 4, 5]

    logger.info("=" * 60)
    logger.info("氢能专利知识图谱构建系统 V4 (统一实体处理)")
    logger.info(f"  执行步骤: {steps}")
    logger.info(f"  使用缓存: {use_cache}")
    logger.info("=" * 60)

    records = None
    all_entities = None
    unified_mapping = {}

    # Step 1
    if 1 in steps:
        records, all_entities = run_step1_load_data(use_cache=use_cache)
    else:
        # 从缓存加载
        cache = get_cache_manager()
        records = cache.load_json("step1_records.json", [])
        entities_data = cache.load_json("step1_all_entities.json", {})
        all_entities = set(entities_data.get("entities", []))

    if not records:
        logger.error("没有数据可处理")
        return

    # Step 2
    if 2 in steps:
        records, unified_mapping = run_step2_entity_resolution(
            records, all_entities, use_cache=use_cache
        )
    else:
        cache = get_cache_manager()
        unified_mapping = cache.load_json("step2_unified_mapping.json", {})
        # 尝试加载已解析的records（包含resolved字段）
        if 3 in steps or 4 in steps:
            resolved_records = cache.load_json("step2_resolved_records.json", [])
            if resolved_records:
                records = resolved_records
                logger.info(f"从缓存加载resolved_records: {len(records)} 条")
            elif unified_mapping:
                # 如果没有resolved缓存但有mapping，则重新应用
                logger.info("重新应用实体消解...")
                from data_pipeline.enhanced_entity_resolver import UnifiedEntityResolver
                resolver = UnifiedEntityResolver()
                records = resolver.apply_resolution(records, unified_mapping)

    # Step 3
    if 3 in steps:
        run_step3_generate_cypher(records, unified_mapping)

    # Step 4
    if 4 in steps:
        run_step4_build_vector_index(records)

    # Step 5
    if 5 in steps:
        run_step5_import_neo4j(resume=use_cache, clear=args.clear)

    logger.info("=" * 60)
    logger.info("构建完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
