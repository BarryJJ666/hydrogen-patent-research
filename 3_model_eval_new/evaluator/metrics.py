# -*- coding: utf-8 -*-
"""
评估指标计算

核心逻辑参考 CypherBench (EMNLP 2026) 的 execution_accuracy 实现：
- 执行 predicted 和 gold Cypher，比较结果集
- 完全忽略列名(alias)，只比较值
- 尝试所有列排列组合寻找匹配
- 有 ORDER BY 时保持行顺序，否则视为多重集(bag)比较

PSJS (Provenance Subgraph Jaccard Similarity) 指标：
- 提取 MATCH 子句涉及的所有节点 elementId
- 计算预测和金标的节点集合 Jaccard 相似度
"""
import re
import time
import random
from itertools import product
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np

from utils.neo4j_client import get_neo4j_client
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """评估指标"""
    # 核心指标
    execution_accuracy: float = 0.0     # EX准确率（执行结果完全匹配，严格）
    semantic_accuracy: float = 0.0      # 语义准确率（召回率≥80%，宽松）

    # Text-to-Cypher专属指标
    executable_rate: float = 0.0        # 可执行率
    psjs: float = 0.0                   # Provenance Subgraph Jaccard Similarity
    syntax_error_rate: float = 0.0      # 语法错误率

    # 效率指标
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0

    # 错误分析
    timeout_rate: float = 0.0

    # 样本数
    total_samples: int = 0
    successful_samples: int = 0


# =============================================================================
# 以下为 CypherBench 风格的 Execution Accuracy 实现
# =============================================================================

def to_hashable(obj, unorder_list=True):
    """
    递归将 Neo4j 返回值转为可哈希对象。
    - list/set → sorted tuple
    - dict → sorted (k,v) tuple
    - Neo4j Date → ISO string
    """
    if isinstance(obj, (tuple, int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, 'iso_format'):
        # neo4j.time.Date 等
        return obj.iso_format()
    elif isinstance(obj, (list, tuple)):
        if unorder_list:
            return tuple(sorted(str(to_hashable(item)) for item in obj))
        else:
            return tuple(to_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return tuple(sorted(str(to_hashable(item)) for item in obj))
    elif isinstance(obj, dict):
        return tuple(sorted((str(to_hashable(k)), to_hashable(v)) for k, v in obj.items()))
    else:
        # 兜底：转字符串
        return str(obj)


def to_tuples(result: List[Dict]) -> List[Tuple]:
    """将 [{col: val, ...}, ...] 转为 [(val, ...), ...]，丢弃列名"""
    if not result:
        return []
    keys = list(result[0].keys())
    return [tuple(row.get(key) for key in keys) for row in result]


def unorder_row(row: Tuple) -> Tuple:
    """将一行的值排序（用于 quick reject）"""
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    """快速拒绝：对行内值排序后比较"""
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


def multiset_eq(l1: List, l2: List) -> bool:
    """多重集相等：每个元素出现次数都相同"""
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] += 1
    for e in l2:
        d[e] -= 1
        if d[e] < 0:
            return False
    return True


def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    """按排列重排元组"""
    return tuple(element[i] for i in perm)


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    """
    生成列排列候选，通过采样剪枝。
    - 列数 <= 3 时穷举
    - 列数 > 3 时采样 20 行约束排列空间
    """
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    for _ in range(20):
        random_row = random.choice(result2)
        for col1 in range(num_cols):
            for col2 in set(perm_constraints[col1]):
                if random_row[col2] not in tab1_sets_by_columns[col1]:
                    perm_constraints[col1].remove(col2)
    return product(*perm_constraints)


def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    """
    比较两个结果集是否等价（CypherBench 核心算法）。
    尝试所有列排列组合，寻找使两个结果集相等的映射。
    """
    if len(result1) == 0 and len(result2) == 0:
        return True
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])
    if len(result2[0]) != num_cols:
        # 列数不同 → 不等
        return False

    # 快速拒绝
    if not quick_rej(result1, result2, order_matters):
        return False

    # 构建 result1 每列的值集合（用于剪枝）
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


# =============================================================================
# MetricsCalculator
# =============================================================================

class MetricsCalculator:
    """指标计算器"""

    def __init__(self):
        self.client = get_neo4j_client()

    def execute_cypher(self, cypher: str, timeout: int = 120) -> Dict:
        """安全执行 Cypher"""
        if not cypher or not cypher.strip():
            return {"success": False, "data": [], "error": "Empty cypher"}
        return self.client.execute(cypher, timeout=timeout)

    def execution_accuracy(self, pred_cypher: str, gold_cypher: str,
                           timeout: int = 120) -> Tuple[float, Dict]:
        """
        计算单条的执行准确率（CypherBench EX 指标）和语义准确率。

        Returns:
            (score, detail_dict)
            score: 1.0 匹配, 0.0 不匹配
            detail_dict: 包含result_match, semantic_match, recall等详细信息
        """
        detail = {
            "pred_executable": False,
            "gold_executable": False,
            "result_match": False,
            "semantic_match": False,
            "recall": 0.0,
            "error_type": None,
            "pred_rows": 0,
            "gold_rows": 0,
        }

        # 快捷路径：字符串完全一致
        if pred_cypher.strip() == gold_cypher.strip():
            detail["pred_executable"] = True
            detail["gold_executable"] = True
            detail["result_match"] = True
            detail["semantic_match"] = True
            detail["recall"] = 1.0
            return 1.0, detail

        # 执行金标 Cypher
        gold_result = self.execute_cypher(gold_cypher, timeout=timeout)
        if not gold_result["success"]:
            detail["gold_executable"] = False
            detail["error_type"] = "gold_execution_error"
            return 0.0, detail
        detail["gold_executable"] = True
        detail["gold_rows"] = len(gold_result["data"])

        # 执行预测 Cypher
        pred_result = self.execute_cypher(pred_cypher, timeout=timeout)
        if not pred_result["success"]:
            error = pred_result.get("error", "")
            if "syntax" in error.lower():
                detail["error_type"] = "syntax_error"
            elif "timeout" in error.lower():
                detail["error_type"] = "timeout"
            else:
                detail["error_type"] = "execution_error"
            return 0.0, detail

        detail["pred_executable"] = True
        detail["pred_rows"] = len(pred_result["data"])

        # 将结果转为可哈希格式
        try:
            gold_data = [{k: to_hashable(v) for k, v in record.items()}
                         for record in gold_result["data"]]
            pred_data = [{k: to_hashable(v) for k, v in record.items()}
                         for record in pred_result["data"]]
        except Exception as e:
            logger.warning(f"Error converting results to hashable: {e}")
            return 0.0, detail

        # 空结果比较
        if not gold_data and not pred_data:
            detail["result_match"] = True
            detail["semantic_match"] = True
            detail["recall"] = 1.0
            return 1.0, detail
        if not gold_data or not pred_data:
            # 计算召回率用于语义匹配
            if not gold_data:
                detail["recall"] = 1.0  # 没有漏掉任何金标
            else:
                detail["recall"] = 0.0
            detail["semantic_match"] = detail["recall"] >= 0.8
            return 0.0, detail

        # 转为 tuple 列表（丢弃列名）
        gold_tuples = to_tuples(gold_data)
        pred_tuples = to_tuples(pred_data)

        # 判断是否需要保持行顺序
        order_matters = "order by" in gold_cypher.lower()

        # CypherBench 核心比较（严格）
        is_match = result_eq(gold_tuples, pred_tuples, order_matters=order_matters)
        detail["result_match"] = is_match

        # 计算召回率和语义准确率（宽松）
        if is_match:
            # 严格匹配时，召回率和精确率都是100%
            detail["recall"] = 1.0
            detail["semantic_match"] = True
        else:
            # 计算召回率：有多少金标行被覆盖
            matched_count = 0
            for gold_tuple in gold_tuples:
                if gold_tuple in pred_tuples:
                    matched_count += 1

            recall = matched_count / len(gold_tuples) if gold_tuples else 0.0
            detail["recall"] = recall
            detail["semantic_match"] = recall >= 0.8

        return float(is_match), detail

    def calculate_metrics(self, results: List[Dict]) -> EvaluationMetrics:
        """
        从推理结果计算所有指标

        Args:
            results: 推理结果列表，每个包含
                {pred_cypher, gold_cypher, latency_ms, success}
        """
        total = len(results)
        if total == 0:
            return EvaluationMetrics()

        latencies = [r.get("latency_ms", 0) for r in results]
        input_tokens = [r.get("input_tokens", 0) for r in results]
        output_tokens = [r.get("output_tokens", 0) for r in results]
        successful = sum(1 for r in results if r.get("success", False))

        return EvaluationMetrics(
            avg_latency_ms=float(np.mean(latencies)) if latencies else 0,
            p50_latency_ms=float(np.percentile(latencies, 50)) if latencies else 0,
            p99_latency_ms=float(np.percentile(latencies, 99)) if latencies else 0,
            avg_input_tokens=float(np.mean(input_tokens)) if input_tokens else 0,
            avg_output_tokens=float(np.mean(output_tokens)) if output_tokens else 0,
            total_samples=total,
            successful_samples=successful
        )


# =============================================================================
# Provenance Subgraph Jaccard Similarity (PSJS)
# 参考 CypherBench (EMNLP 2026) 实现
# =============================================================================

def split_cypher_into_clauses(cypher_query: str) -> List[str]:
    """将Cypher查询拆分为子句"""
    clause_pattern = r'\b(MATCH|OPTIONAL MATCH|WHERE|RETURN|UNION|WITH|CREATE|SET|DELETE|MERGE|UNWIND|ORDER BY|LIMIT|SKIP|FOREACH|CALL|YIELD)\b'
    matches = list(re.finditer(clause_pattern, cypher_query))
    clauses = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cypher_query)
        clauses.append(cypher_query[start:end].strip())
    return clauses


def extract_match_cypher(cypher: str) -> Optional[str]:
    """提取Cypher的MATCH部分（不含RETURN/ORDER BY/LIMIT等）"""
    if not cypher.strip().upper().startswith('MATCH'):
        return None

    clauses = split_cypher_into_clauses(cypher)
    match_clauses = []
    for clause in clauses:
        if not any(clause.upper().startswith(kw) for kw in ['MATCH', 'OPTIONAL MATCH', 'WITH', 'WHERE']):
            break
        if clause.upper().startswith('WITH'):
            if ' as ' in clause.lower():
                break
            else:
                match_clauses.append('WITH *')
        else:
            match_clauses.append(clause)

    # 移除尾部的WITH子句
    while match_clauses and match_clauses[-1].upper().startswith('WITH'):
        match_clauses.pop()

    return ' '.join(match_clauses) if match_clauses else None


def add_variables_to_match(match_cypher: str) -> str:
    """为匿名节点和关系添加临时变量名"""
    node_counter = [0]
    rel_counter = [0]

    def replace_node(match):
        replacement = f"(ntmp{node_counter[0]}:{match.group(2)}{match.group(3) or ''})"
        node_counter[0] += 1
        return replacement

    def replace_rel(match):
        replacement = f"[rtmp{rel_counter[0]}{match.group(2)}]"
        rel_counter[0] += 1
        return replacement

    clauses = split_cypher_into_clauses(match_cypher)
    for i, clause in enumerate(clauses):
        if clause.upper().startswith('MATCH') or clause.upper().startswith('OPTIONAL MATCH'):
            # 先替换关系
            clause = re.sub(r'(\[)(:.*?)(\])', replace_rel, clause)
            # 再替换匿名节点
            clauses[i] = re.sub(r'(\(:)([A-Za-z]+)(\s*\{.*?\})?\)', replace_node, clause)

    return ' '.join(clauses)


def extract_node_variables(match_cypher: str) -> List[str]:
    """提取MATCH子句中的所有节点变量"""
    # 替换属性块
    clean = re.sub(r'\{[^}]*\}', '{dummy}', match_cypher)
    pattern = r'\((\w+)(?::[^\)]*|\))'
    variables = []
    clauses = split_cypher_into_clauses(clean)
    for clause in clauses:
        if clause.upper().startswith('MATCH') or clause.upper().startswith('OPTIONAL MATCH'):
            variables += re.findall(pattern, clause)
    return sorted(list(set(variables)))


def split_by_union(cypher: str) -> List[str]:
    """按UNION拆分Cypher查询"""
    pattern = r'\bUNION\b'
    if cypher.strip().upper().startswith("CALL"):
        inner_match = re.search(r'CALL\s*\{(.*?)\}\s*(WITH|RETURN|WHERE|UNWIND)', cypher, re.DOTALL)
        if inner_match:
            inner = inner_match.group(1)
            return [q.strip() for q in re.split(pattern, inner)]
        return [cypher.strip()]
    return [q.strip() for q in re.split(pattern, cypher)]


def get_provenance_cypher(cypher: str, return_var: str = 'elemId') -> str:
    """
    生成用于获取查询溯源子图的Cypher
    返回所有MATCH子句涉及的节点的elementId
    """
    sub_cyphers = split_by_union(cypher)
    ps_cyphers = []

    for sub_cypher in sub_cyphers:
        match_cypher = extract_match_cypher(sub_cypher)
        if match_cypher:
            match_cypher = add_variables_to_match(match_cypher)
            node_vars = extract_node_variables(match_cypher)
            if node_vars:
                node_expr = ' + '.join(f'collect(distinct elementId({var}))' for var in node_vars)
                ps_cyphers.append(
                    f'{match_cypher} WITH {node_expr} AS elemIds '
                    f'UNWIND elemIds AS elemId RETURN elemId AS {return_var}'
                )

    if not ps_cyphers:
        return f'UNWIND [] AS elemId RETURN elemId AS {return_var}'

    return ' UNION '.join(ps_cyphers)


def provenance_subgraph_jaccard_similarity(
    pred_cypher: str,
    gold_cypher: str,
    client,
    timeout: int = 120
) -> float:
    """
    计算 Provenance Subgraph Jaccard Similarity (PSJS)

    提取预测和金标Cypher的MATCH子句涉及的所有节点，
    计算节点集合的Jaccard相似度: |交集| / |并集|

    Args:
        pred_cypher: 预测的Cypher
        gold_cypher: 金标Cypher
        client: Neo4j客户端
        timeout: 超时时间(秒)

    Returns:
        PSJS分数 [0.0, 1.0]
    """
    if not pred_cypher or not pred_cypher.strip():
        return 0.0

    # 快捷路径：完全一致
    if pred_cypher.strip() == gold_cypher.strip():
        return 1.0

    gold_ps_cypher = get_provenance_cypher(gold_cypher, return_var='elemId1')
    pred_ps_cypher = get_provenance_cypher(pred_cypher, return_var='elemId2')

    try:
        # 执行金标溯源查询（不设超时）
        gold_result = client.execute(gold_ps_cypher, timeout=300)
        if not gold_result["success"]:
            return 0.0
        gold_ps = set(record['elemId1'] for record in gold_result["data"])

        # 执行预测溯源查询（设超时）
        pred_result = client.execute(pred_ps_cypher, timeout=timeout)
        if not pred_result["success"]:
            return 0.0
        pred_ps = set(record['elemId2'] for record in pred_result["data"])

        # 计算Jaccard相似度
        intersection = len(gold_ps.intersection(pred_ps))
        union = len(gold_ps.union(pred_ps))
        return intersection / union if union > 0 else 0.0

    except Exception as e:
        logger.warning(f"PSJS computation error: {e}")
        return 0.0
