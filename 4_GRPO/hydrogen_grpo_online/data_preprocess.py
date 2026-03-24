# -*- coding: utf-8 -*-
"""
将氢能专利知识图谱 QA 数据集转换为 VERL GRPO 训练所需的 Parquet 格式。
结合预执行的金标答案，用于在线奖励函数。

运行方式:
    cd /ssd1/zhangyuzhe/verl-release-v0.7.0
    # 先执行预计算
    python hydrogen_grpo_online/precompute_gold_answers.py
    # 再生成 parquet
    python hydrogen_grpo_online/data_preprocess.py
"""

import argparse
import json
import os

import pandas as pd


# ==============================================================================
# 系统提示（与 SFT 训练时完全一致）
# ==============================================================================

SCHEMA_DESCRIPTION = """氢能专利知识图谱Schema:

节点类型:
- Patent: 专利（属性：application_no, title_cn, abstract_cn, application_date, patent_type, transfer_count, license_count, pledge_count, litigation_count）
- Organization: 机构（属性：name, entity_type）
- Person: 人物（属性：uid, name）
- TechDomain: 技术领域（属性：name, level）
- IPCCode: IPC分类号（属性：code, section, class_code）
- Country: 国家（属性：name）
- LegalStatus: 法律状态（属性：name）
- Location: 地点（属性：location_id, name, level, country, province, city）
- LitigationType: 诉讼类型（属性：name）

关系类型:
- APPLIED_BY: Patent -> Organization/Person（申请人）
- OWNED_BY: Patent -> Organization/Person（权利人）
- TRANSFERRED_FROM/TO: Patent -> Organization/Person（转让）
- LICENSED_FROM/TO: Patent -> Organization/Person（许可）
- PLEDGED_FROM/TO: Patent -> Organization/Person（质押）
- LITIGATED_WITH: Patent -> Organization/Person（诉讼，有role属性）
- BELONGS_TO: Patent -> TechDomain（所属领域）
- CLASSIFIED_AS: Patent -> IPCCode（IPC分类）
- LOCATED_IN: Organization -> Location（所在地）
- PUBLISHED_IN: Patent -> Country（公开国家）
- HAS_STATUS: Patent -> LegalStatus（法律状态）

技术领域: 制氢技术, 储氢技术, 物理储氢, 合金储氢, 无机储氢, 有机储氢, 氢燃料电池, 氢制冷

注意: application_date是字符串格式'YYYY-MM-DD'，提取年份使用substring(p.application_date, 0, 4)"""

SYSTEM_PROMPT = f"""你是氢能专利知识图谱查询助手。根据用户的问题，生成对应的Neo4j Cypher查询语句。

{SCHEMA_DESCRIPTION}

请直接输出Cypher查询语句，不要添加任何解释。"""

DATA_SOURCE = "hydrogen_patent_cypher_online"


def process_sample(sample: dict, gold_answers: dict, split: str, idx: int) -> dict:
    """将单条原始 QA 样本转换为 VERL parquet 行格式。"""
    qid = sample.get("qid", f"{split}_{idx}")

    # 获取预执行的金标答案
    gold_answer = gold_answers.get(qid, {})

    ground_truth_payload = json.dumps(
        {
            "cypher": sample["cypher"],
            "gold_rows": gold_answer.get("rows"),
            "gold_row_count": gold_answer.get("row_count", 0),
            "gold_success": gold_answer.get("success", False),
        },
        ensure_ascii=False,
    )

    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]},
        ],
        "ability": "text2cypher",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth_payload,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "qid": qid,
            "question": sample["question"],
            "category": sample.get("category", ""),
            "complexity": sample.get("complexity", 1),
            "gold_cypher": sample["cypher"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="生成 VERL parquet 格式数据")
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
        "--gold_answers_file",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gold_answers.json"),
        help="预执行金标答案 JSON 文件",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="parquet 输出目录",
    )
    args = parser.parse_args()

    # 加载预执行结果
    if os.path.exists(args.gold_answers_file):
        with open(args.gold_answers_file, "r", encoding="utf-8") as f:
            gold_answers = json.load(f)
        print(f"[加载] 预执行金标答案: {len(gold_answers)} 条")
    else:
        print(f"[警告] 未找到预执行结果文件: {args.gold_answers_file}")
        print("[提示] 请先运行 python hydrogen_grpo_online/precompute_gold_answers.py")
        gold_answers = {}

    os.makedirs(args.output_dir, exist_ok=True)

    for split, filepath in [("train", args.train_file), ("test", args.test_file)]:
        if not os.path.exists(filepath):
            print(f"[警告] 文件不存在，跳过: {filepath}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        rows = [process_sample(s, gold_answers, split, i) for i, s in enumerate(raw_data)]
        df = pd.DataFrame(rows)

        out_path = os.path.join(args.output_dir, f"{split}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[{split}] 已保存 {len(rows)} 条样本 -> {out_path}")

        # 统计金标匹配情况
        matched = sum(1 for s in raw_data if s.get("qid") in gold_answers)
        print(f"  - 匹配预执行结果: {matched}/{len(raw_data)}")

    print("\n数据预处理完成。")


if __name__ == "__main__":
    main()
