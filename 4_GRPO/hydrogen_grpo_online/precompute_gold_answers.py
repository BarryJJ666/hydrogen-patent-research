# -*- coding: utf-8 -*-
"""
预执行金标 Cypher 查询，将结果存储为 JSON 文件。

训练时奖励函数只需执行预测 Cypher，然后与预存结果对比，
避免每次都执行金标 Cypher，提升训练效率。

运行方式:
    cd /ssd1/zhangyuzhe/verl-release-v0.7.0
    python hydrogen_grpo_online/precompute_gold_answers.py

输出:
    hydrogen_grpo_online/data/gold_answers.json
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, ClientError
from tqdm import tqdm


# ==============================================================================
# Neo4j 配置
# ==============================================================================

NEO4J_CONFIG = {
    "uri": "bolt://10.223.3.13:7687",
    "user": "neo4j",
    "password": "zhangyuzhe",
    "database": "neo4j",
    "max_connection_lifetime": 3600,
    "max_connection_pool_size": 50,
}

EXECUTION_TIMEOUT = 10  # 秒


# ==============================================================================
# 执行引擎
# ==============================================================================

class CypherExecutor:
    """Neo4j Cypher 执行器，支持连接池和并发执行。"""

    def __init__(self, config: dict):
        self.driver = GraphDatabase.driver(
            config["uri"],
            auth=(config["user"], config["password"]),
            max_connection_lifetime=config.get("max_connection_lifetime", 3600),
            max_connection_pool_size=config.get("max_connection_pool_size", 50),
        )
        self.database = config.get("database", "neo4j")

    def execute(self, cypher: str, timeout: int = EXECUTION_TIMEOUT) -> dict:
        """
        执行 Cypher 查询。

        Returns:
            {
                "success": bool,
                "rows": list[dict] | None,
                "row_count": int,
                "error": str | None,
            }
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, timeout=timeout)
                rows = [dict(record) for record in result]
                return {
                    "success": True,
                    "rows": rows,
                    "row_count": len(rows),
                    "error": None,
                }
        except CypherSyntaxError as e:
            return {
                "success": False,
                "rows": None,
                "row_count": 0,
                "error": f"SyntaxError: {str(e)[:200]}",
            }
        except ClientError as e:
            return {
                "success": False,
                "rows": None,
                "row_count": 0,
                "error": f"ClientError: {str(e)[:200]}",
            }
        except Exception as e:
            return {
                "success": False,
                "rows": None,
                "row_count": 0,
                "error": f"Error: {str(e)[:200]}",
            }

    def close(self):
        self.driver.close()


def _serialize_value(value):
    """将 Neo4j 返回值序列化为 JSON 兼容格式。"""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    # Neo4j 特殊类型转字符串
    return str(value)


def _serialize_rows(rows: list) -> list:
    """序列化查询结果为 JSON 兼容格式。"""
    if rows is None:
        return None
    return [
        {k: _serialize_value(v) for k, v in row.items()}
        for row in rows
    ]


# ==============================================================================
# 主函数
# ==============================================================================

def process_sample(executor: CypherExecutor, sample: dict) -> dict:
    """处理单个样本，执行金标 Cypher 并返回结果。"""
    qid = sample.get("qid", "")
    cypher = sample.get("cypher", "")

    if not cypher:
        return {
            "qid": qid,
            "success": False,
            "rows": None,
            "row_count": 0,
            "error": "Empty cypher",
        }

    result = executor.execute(cypher)
    result["qid"] = qid
    result["rows"] = _serialize_rows(result.get("rows"))
    return result


def main():
    parser = argparse.ArgumentParser(description="预执行金标 Cypher 查询")
    parser.add_argument(
        "--train_file",
        default="/ssd1/zhangyuzhe/hydrogen_benchmark_gen/output/validated/train_raw.json",
        help="训练集 JSON 文件路径",
    )
    parser.add_argument(
        "--test_file",
        default="/ssd1/zhangyuzhe/hydrogen_benchmark_gen/output/validated/test_raw.json",
        help="测试集 JSON 文件路径",
    )
    parser.add_argument(
        "--output_file",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gold_answers.json"),
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发执行线程数",
    )
    args = parser.parse_args()

    # 加载原始数据
    all_samples = []
    for filepath in [args.train_file, args.test_file]:
        if not os.path.exists(filepath):
            print(f"[警告] 文件不存在，跳过: {filepath}")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            samples = json.load(f)
            all_samples.extend(samples)
            print(f"[加载] {filepath}: {len(samples)} 条")

    if not all_samples:
        print("[错误] 未加载任何数据")
        return

    print(f"\n[总计] {len(all_samples)} 条样本待执行")

    # 初始化执行器
    executor = CypherExecutor(NEO4J_CONFIG)

    # 并发执行
    results = {}
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_sample, executor, sample): sample
            for sample in all_samples
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="执行金标 Cypher"):
            result = future.result()
            qid = result["qid"]
            results[qid] = {
                "success": result["success"],
                "rows": result["rows"],
                "row_count": result["row_count"],
                "error": result.get("error"),
            }
            if result["success"]:
                success_count += 1
            else:
                error_count += 1

    executor.close()

    # 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 结果已保存到 {args.output_file}")
    print(f"  - 成功: {success_count}")
    print(f"  - 失败: {error_count}")
    print(f"  - 成功率: {success_count / len(all_samples) * 100:.1f}%")


if __name__ == "__main__":
    main()
