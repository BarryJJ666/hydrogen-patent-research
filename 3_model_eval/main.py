# -*- coding: utf-8 -*-
"""
氢能模型评估系统 - 主入口（简化版）

结果保存结构：
- results/raw/{model_name}_{timestamp}.jsonl
- results/reports/{model_name}_{timestamp}.md
"""
import argparse
import json
import sys
import fnmatch
from pathlib import Path
from datetime import datetime

from config.settings import MODEL_CONFIGS, RESULTS_DIR, RAW_DIR, REPORTS_DIR
from models import ModelFactory
from evaluator import CypherEvaluator, MetricsCalculator
from runner.batch_runner import BatchRunner
from reporter import ReportGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


def load_test_data(test_path):
    """加载测试数据"""
    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info("Loaded %d test samples from %s", len(data), test_path)
    return data


def filter_models_by_pattern(model_names, pattern):
    """使用通配符过滤模型"""
    if not pattern:
        return model_names
    filtered = [m for m in model_names if fnmatch.fnmatch(m, pattern)]
    return filtered


def print_available_models():
    """打印所有可用模型"""
    print("")
    print("=" * 60)
    print("可用模型列表:")
    print("=" * 60)

    # 分类显示
    local_models = []
    direct_models = []
    tool_single_models = []
    tool_multi_models = []

    for name, config in MODEL_CONFIGS.items():
        model_type = config.get("type", "")
        mode = config.get("mode", "direct")

        if model_type == "local_vllm":
            local_models.append(name)
        elif mode == "direct":
            direct_models.append(name)
        elif config.get("max_steps") == 1:
            tool_single_models.append(name)
        else:
            tool_multi_models.append(name)

    print("")
    print("[本地模型] (需要GPU)")
    for m in local_models:
        print("  - %s" % m)

    print("")
    print("[大模型 Text-to-Cypher]")
    for m in direct_models:
        print("  - %s" % m)

    print("")
    print("[大模型 一轮工具调用]")
    for m in tool_single_models:
        print("  - %s" % m)

    print("")
    print("[大模型 多轮工具调用]")
    for m in tool_multi_models:
        print("  - %s" % m)

    print("")
    print("=" * 60)
    print("总计: %d 个模型配置" % len(MODEL_CONFIGS))
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='氢能模型评估系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 评估单个模型
  python main.py --models deepseek_v32_direct --max-samples 50

  # 使用通配符过滤
  python main.py --model-filter "*_direct"
  python main.py --model-filter "deepseek*"

  # 评估所有模型
  python main.py --models all

  # 查看可用模型
  python main.py --list-models

结果保存:
  results/raw/{model_name}_{timestamp}.jsonl
  results/reports/{model_name}_{timestamp}.md
        '''
    )

    parser.add_argument(
        '--test-data',
        type=str,
        default='../hydrogen_benchmark_gen/output/sft_format/test.json',
        help='测试数据路径'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help='要评估的模型列表，或 "all" 评估所有模型'
    )
    parser.add_argument(
        '--model-filter',
        type=str,
        default=None,
        help='模型过滤器（支持通配符），如 "deepseek*" 或 "*_direct"'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='批量推理大小'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大评估样本数（用于测试）'
    )
    parser.add_argument(
        '--no-psjs',
        action='store_true',
        help='跳过PSJS指标计算（可加快评估速度）'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='列出所有可用模型'
    )

    args = parser.parse_args()

    # 如果只是列出模型，打印后退出
    if args.list_models:
        print_available_models()
        sys.exit(0)

    # 确定要评估的模型
    if 'all' in args.models:
        model_names = list(MODEL_CONFIGS.keys())
    else:
        # 验证模型名称
        valid_models = []
        for m in args.models:
            if m in MODEL_CONFIGS:
                valid_models.append(m)
            else:
                logger.warning("Unknown model: %s, skipping...", m)
        model_names = valid_models

    # 应用模型过滤器
    if args.model_filter:
        model_names = filter_models_by_pattern(model_names, args.model_filter)
        logger.info("Filtered models with pattern '%s': %d models", args.model_filter, len(model_names))

    if not model_names:
        logger.error("No models to evaluate!")
        print("")
        print("可用模型列表:")
        for m in sorted(MODEL_CONFIGS.keys()):
            print("  - %s" % m)
        sys.exit(1)

    logger.info("Models to evaluate: %s", model_names)

    # 加载测试数据
    test_data = load_test_data(args.test_data)

    # 限制样本数（用于测试）
    if args.max_samples and args.max_samples < len(test_data):
        test_data = test_data[:args.max_samples]
        logger.info("Limited to %d samples for evaluation", args.max_samples)

    # 提取问题和金标Cypher
    questions = [item['input'] for item in test_data]
    gold_cyphers = [item['output'] for item in test_data]

    # 创建批量运行器（简化版）
    runner = BatchRunner()

    # 运行评估
    all_results = runner.run_all_models(
        model_names=model_names,
        questions=questions,
        gold_cyphers=gold_cyphers,
        batch_size=args.batch_size
    )

    if not all_results:
        logger.warning("No evaluation results!")
        sys.exit(0)

    # 如果评估了多个模型，生成对比报告
    if len(all_results) > 1:
        reporter = ReportGenerator()
        comparison_path = reporter.generate_comparison_report(all_results, len(test_data))
        csv_path = reporter.generate_comparison_csv(all_results)
        logger.info("Comparison report: %s", comparison_path)
        logger.info("Comparison CSV: %s", csv_path)

    # 打印摘要
    print("")
    print("=" * 60)
    print("评估完成！")
    print("=" * 60)

    for model_name, result in all_results.items():
        metrics = result.get('metrics', {})
        mode = result.get('mode', 'direct')

        print("")
        print("%s (%s):" % (model_name, mode))
        if mode == 'direct':
            print("  执行准确率(EX): %.2f%%" % (metrics.get('execution_accuracy', 0) * 100))
            print("  PSJS: %.2f%%" % (metrics.get('psjs', 0) * 100))
            print("  可执行率: %.2f%%" % (metrics.get('executable_rate', 0) * 100))
            print("  语法错误率: %.2f%%" % (metrics.get('syntax_error_rate', 0) * 100))
        else:
            print("  答案准确率(EX): %.2f%%" % (metrics.get('answer_accuracy', 0) * 100))
            print("  工具选择准确率: %.2f%%" % (metrics.get('tool_selection_accuracy', 0) * 100))
            print("  平均调用轮数: %.2f" % metrics.get('avg_turns', 0))
        print("  平均延迟: %.1fms" % metrics.get('avg_latency_ms', 0))

    print("")
    print("=" * 60)
    print("结果目录:")
    print("  Raw: %s" % RAW_DIR)
    print("  Reports: %s" % REPORTS_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
