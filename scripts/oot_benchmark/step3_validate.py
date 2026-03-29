#!/usr/bin/env python
"""Step 3: Validate out-of-template questions against Neo4j.
   - Executes each Cypher query
   - Records success/failure, result size, empty/non-empty
   - Outputs validated dataset + validation report
"""
import json
import os
import sys
import time
from neo4j import GraphDatabase

NEO4J_URI = 'bolt://localhost:7687'
NEO4J_USER = 'neo4j'
NEO4J_PWD = 'hydrogen2026'

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))

INPUT_FILE = os.path.join(os.path.dirname(__file__), 'oot_questions_all.json')
OUTPUT_DIR = os.path.dirname(__file__)


def execute_cypher(cypher, timeout=30):
    """Execute Cypher and return (success, result_rows, error_msg)."""
    try:
        with driver.session() as session:
            result = session.run(cypher)
            records = []
            for record in result:
                records.append(dict(record))
            return True, records, None
    except Exception as e:
        return False, [], str(e)


def validate_all(questions):
    """Validate all questions against Neo4j."""
    results = []
    success_count = 0
    fail_count = 0
    empty_count = 0

    for i, q in enumerate(questions):
        qid = q['id']
        cypher = q['cypher']

        success, rows, error = execute_cypher(cypher)

        result_entry = {
            **q,
            'valid': success,
            'result_count': len(rows) if success else 0,
            'is_empty': len(rows) == 0 if success else None,
            'error': error,
            'sample_result': rows[:3] if success and rows else None,
        }

        # Convert non-serializable types
        if result_entry['sample_result']:
            cleaned = []
            for row in result_entry['sample_result']:
                cleaned_row = {}
                for k, v in row.items():
                    if isinstance(v, list):
                        cleaned_row[k] = [str(x) for x in v]
                    else:
                        cleaned_row[k] = str(v) if v is not None else None
                cleaned.append(cleaned_row)
            result_entry['sample_result'] = cleaned

        results.append(result_entry)

        if success:
            success_count += 1
            if len(rows) == 0:
                empty_count += 1
            status = f"OK ({len(rows)} rows)"
        else:
            fail_count += 1
            status = f"FAIL: {error[:80]}"

        print(f"[{i+1:3d}/{len(questions)}] {qid}: {status}")

    return results, success_count, fail_count, empty_count


def generate_report(results, success_count, fail_count, empty_count, total):
    """Generate validation report."""
    from collections import Counter

    lines = []
    lines.append("# Out-of-Template Benchmark Validation Report\n")
    lines.append(f"Total questions: {total}")
    lines.append(f"Executable: {success_count} ({success_count/total*100:.1f}%)")
    lines.append(f"Failed: {fail_count} ({fail_count/total*100:.1f}%)")
    lines.append(f"Empty results: {empty_count} ({empty_count/total*100:.1f}% of executable)")
    lines.append("")

    # Failed queries
    if fail_count > 0:
        lines.append("## Failed Queries\n")
        for r in results:
            if not r['valid']:
                lines.append(f"### {r['id']}: {r['question'][:60]}")
                lines.append(f"Error: {r['error'][:200]}")
                lines.append(f"```cypher\n{r['cypher']}\n```\n")

    # Distribution of result sizes
    sizes = [r['result_count'] for r in results if r['valid']]
    if sizes:
        lines.append("## Result Size Distribution\n")
        lines.append(f"  Empty (0): {sizes.count(0)}")
        lines.append(f"  Small (1-5): {sum(1 for s in sizes if 1 <= s <= 5)}")
        lines.append(f"  Medium (6-20): {sum(1 for s in sizes if 6 <= s <= 20)}")
        lines.append(f"  Large (>20): {sum(1 for s in sizes if s > 20)}")

    # By family
    lines.append("\n## By Family\n")
    family_stats = Counter()
    family_fails = Counter()
    for r in results:
        family_stats[r['family']] += 1
        if not r['valid']:
            family_fails[r['family']] += 1
    for fam, total_f in sorted(family_stats.items()):
        fails = family_fails.get(fam, 0)
        lines.append(f"  {fam}: {total_f} total, {fails} failed")

    return '\n'.join(lines)


def main():
    print("Loading questions...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions\n")

    print("Validating against Neo4j...")
    results, success, fail, empty = validate_all(questions)

    # Save full results
    results_path = os.path.join(OUTPUT_DIR, 'validation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save validated (executable) questions only
    valid_questions = [
        {k: v for k, v in r.items() if k not in ('valid', 'result_count', 'is_empty', 'error', 'sample_result')}
        for r in results if r['valid']
    ]
    valid_path = os.path.join(OUTPUT_DIR, 'oot_questions_validated.json')
    with open(valid_path, 'w', encoding='utf-8') as f:
        json.dump(valid_questions, f, ensure_ascii=False, indent=2)
    print(f"Validated questions saved to {valid_path} ({len(valid_questions)} questions)")

    # Generate report
    report = generate_report(results, success, fail, empty, len(questions))
    report_path = os.path.join(OUTPUT_DIR, 'validation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    print(f"\n=== Summary ===")
    print(f"  Executable: {success}/{len(questions)} ({success/len(questions)*100:.1f}%)")
    print(f"  Failed: {fail}/{len(questions)}")
    print(f"  Empty results: {empty}/{success} ({empty/success*100:.1f}% of executable)")


if __name__ == '__main__':
    main()
    driver.close()
