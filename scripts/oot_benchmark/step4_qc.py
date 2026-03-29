#!/usr/bin/env python
"""Step 4: Quality Control for out-of-template benchmark.
   - Template overlap detection (checks against training templates)
   - Diversity analysis (question similarity clustering)
   - Coverage analysis (dimension, family, result size distribution)
"""
import json
import os
import re
from collections import Counter

SCRIPT_DIR = os.path.dirname(__file__)

# Training template relationships (from match_patterns.json)
TEMPLATE_RELATIONSHIPS = {'BELONGS_TO', 'APPLIED_BY', 'LOCATED_IN', 'TRANSFERRED_TO'}

# Training template operators (simple only)
TEMPLATE_OPERATORS = {'MATCH', 'WHERE', 'WITH', 'RETURN', 'ORDER BY', 'LIMIT',
                      'count', 'DISTINCT', 'substring', 'toInteger', 'DESC', 'ASC'}

# Non-template markers (if any of these appear, the query is out-of-template)
NON_TEMPLATE_MARKERS = [
    'NOT EXISTS',
    'NOT (',
    'OPTIONAL MATCH',
    'CASE WHEN',
    'CASE ',
    'toFloat',
    'collect(',
    'any(',
    'all(',
    'size(',
    'UNION',
    'UNWIND',
    'coalesce(',
    'EXISTS {',
    'STARTS WITH',
]

# Non-template relationships
NON_TEMPLATE_RELATIONSHIPS = [
    'HAS_STATUS', 'IN_FAMILY', 'PUBLISHED_IN', 'OWNED_BY',
    'CLASSIFIED_AS', 'LICENSED_TO', 'LICENSED_FROM',
    'PLEDGED_TO', 'PLEDGED_FROM', 'LITIGATED_WITH',
    'HAS_LITIGATION_TYPE', 'PARENT_DOMAIN', 'PARENT_LOCATION',
    'TRANSFERRED_FROM', 'AFFILIATED_WITH',
]


def extract_relationships(cypher):
    """Extract relationship types used in a Cypher query."""
    pattern = r'\[:([A-Z_]+)\]'
    return set(re.findall(pattern, cypher))


def extract_operators(cypher):
    """Extract non-template operators used in a Cypher query."""
    found = []
    for marker in NON_TEMPLATE_MARKERS:
        if marker in cypher:
            found.append(marker)
    return found


def check_out_of_template(question):
    """Check if a question is genuinely out-of-template.
    Returns (is_oot, reasons).
    """
    cypher = question['cypher']
    reasons = []

    # Check 1: Uses non-template relationships
    rels = extract_relationships(cypher)
    non_template_rels = rels - TEMPLATE_RELATIONSHIPS
    if non_template_rels:
        reasons.append(f"non-template relations: {non_template_rels}")

    # Check 2: Uses non-template operators
    ops = extract_operators(cypher)
    if ops:
        reasons.append(f"non-template operators: {ops}")

    is_oot = len(reasons) > 0
    return is_oot, reasons


def analyze_diversity(questions):
    """Analyze question diversity using simple surface-level metrics."""
    # Question length distribution
    lengths = [len(q['question']) for q in questions]

    # Starting patterns
    starts = Counter()
    for q in questions:
        text = q['question']
        # Get first 4 chars
        start = text[:4] if len(text) >= 4 else text
        starts[start] += 1

    # Question word patterns
    q_words = Counter()
    patterns = ['哪些', '统计', '列出', '有没有', '是否', '能否', '对比', '给我',
                '请', '帮我', '找出', '有多少', '什么', '按', '在']
    for q in questions:
        for p in patterns:
            if p in q['question']:
                q_words[p] += 1

    return {
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'top_starts': starts.most_common(10),
        'question_words': q_words.most_common(15),
        'unique_starts_ratio': len(starts) / len(questions),
    }


def run_qc(questions):
    """Run full QC and return report."""
    lines = []
    lines.append("# Out-of-Template Benchmark QC Report\n")

    # 1. Template overlap detection
    lines.append("## 1. Template Overlap Detection\n")
    oot_count = 0
    not_oot = []
    for q in questions:
        is_oot, reasons = check_out_of_template(q)
        if is_oot:
            oot_count += 1
        else:
            not_oot.append(q['id'])

    lines.append(f"Out-of-template: {oot_count}/{len(questions)} ({oot_count/len(questions)*100:.1f}%)")
    if not_oot:
        lines.append(f"\n**WARNING: {len(not_oot)} questions may overlap with training templates:**")
        for qid in not_oot:
            q = next(x for x in questions if x['id'] == qid)
            rels = extract_relationships(q['cypher'])
            lines.append(f"  - {qid}: rels={rels}, question={q['question'][:60]}")
    lines.append("")

    # 2. Relationship usage
    lines.append("## 2. Relationship Coverage\n")
    rel_usage = Counter()
    for q in questions:
        for r in extract_relationships(q['cypher']):
            rel_usage[r] += 1
    for r, c in sorted(rel_usage.items(), key=lambda x: -x[1]):
        marker = " (template)" if r in TEMPLATE_RELATIONSHIPS else " *NEW*"
        lines.append(f"  {r}: {c}{marker}")
    lines.append("")

    # 3. Operator usage
    lines.append("## 3. Non-Template Operator Usage\n")
    op_usage = Counter()
    for q in questions:
        for op in extract_operators(q['cypher']):
            op_usage[op] += 1
    for op, c in sorted(op_usage.items(), key=lambda x: -x[1]):
        lines.append(f"  {op}: {c}")
    lines.append("")

    # 4. Diversity
    lines.append("## 4. Question Diversity\n")
    div = analyze_diversity(questions)
    lines.append(f"Avg question length: {div['avg_length']:.1f} chars")
    lines.append(f"Min/Max: {div['min_length']}/{div['max_length']}")
    lines.append(f"Unique starting patterns: {div['unique_starts_ratio']*100:.1f}%")
    lines.append("\nQuestion word distribution:")
    for word, count in div['question_words']:
        lines.append(f"  {word}: {count} ({count/len(questions)*100:.1f}%)")
    lines.append("")

    # 5. Coverage matrix
    lines.append("## 5. Coverage Matrix\n")

    # By family
    lines.append("### By Family")
    fam_dist = Counter(q['family'] for q in questions)
    for f, c in sorted(fam_dist.items(), key=lambda x: -x[1]):
        lines.append(f"  {f}: {c}")

    # By n_dims
    lines.append("\n### By Dimension Count")
    dim_dist = Counter(q['n_dims'] for q in questions)
    for d, c in sorted(dim_dist.items()):
        lines.append(f"  {d}-dim: {c} ({c/len(questions)*100:.1f}%)")

    # By dimension
    lines.append("\n### By Dimension")
    all_dims = Counter()
    for q in questions:
        for d in q['dims']:
            all_dims[d] += 1
    for d, c in sorted(all_dims.items(), key=lambda x: -x[1]):
        lines.append(f"  {d}: {c}")

    return '\n'.join(lines)


def main():
    # Load questions (use validated if available, otherwise all)
    validated_path = os.path.join(SCRIPT_DIR, 'oot_questions_validated.json')
    all_path = os.path.join(SCRIPT_DIR, 'oot_questions_all.json')

    if os.path.exists(validated_path):
        input_path = validated_path
    else:
        input_path = all_path

    print(f"Loading questions from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions\n")

    report = run_qc(questions)

    report_path = os.path.join(SCRIPT_DIR, 'qc_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
