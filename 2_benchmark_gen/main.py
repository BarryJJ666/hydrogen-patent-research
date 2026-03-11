# -*- coding: utf-8 -*-
"""
氢能专利知识图谱 Benchmark 生成系统

使用方法:
    python main.py --target 5000 --use-llm
    python main.py --target 100 --no-llm  # 快速测试
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.full_pipeline import BenchmarkPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="氢能Benchmark生成系统")
    parser.add_argument("--target", "--total-samples", type=int, default=5000,
                        dest="target",
                        help="目标生成数量 (default: 5000)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="训练集比例 (default: 0.8)")
    parser.add_argument("--use-llm", action="store_true", default=True,
                        help="使用LLM生成问题 (default: True)")
    parser.add_argument("--no-llm", action="store_true",
                        help="不使用LLM，只用模板生成")
    parser.add_argument("--no-intermediate", action="store_true",
                        help="不保存中间结果")

    args = parser.parse_args()

    use_llm = not args.no_llm
    save_intermediate = not args.no_intermediate
    train_ratio = args.train_ratio

    logger.info("=" * 60)
    logger.info("氢能专利知识图谱 Benchmark 生成系统")
    logger.info("=" * 60)
    logger.info(f"目标数量: {args.target}")
    logger.info(f"训练集比例: {train_ratio}")
    logger.info(f"使用LLM: {use_llm}")
    logger.info(f"保存中间结果: {save_intermediate}")
    logger.info("=" * 60)

    try:
        pipeline = BenchmarkPipeline()
        stats = pipeline.run(
            target_count=args.target,
            use_llm=use_llm,
            save_intermediate=save_intermediate,
            train_ratio=train_ratio
        )

        logger.info("\n" + "=" * 60)
        logger.info("生成完成！统计信息:")
        logger.info("=" * 60)
        logger.info(f"生成Cypher: {stats['generated']}")
        logger.info(f"语法验证通过: {stats['syntax_valid']}")
        logger.info(f"执行验证通过: {stats['execution_valid']}")
        logger.info(f"最终QA对: {stats['final_count']}")
        logger.info(f"训练集: {stats['train_size']}")
        logger.info(f"测试集: {stats['test_size']}")
        logger.info(f"耗时: {stats['elapsed_seconds']}秒")
        logger.info("=" * 60)

        logger.info("\n类别分布:")
        for cat, count in stats.get("by_category", {}).items():
            logger.info(f"  {cat}: {count}")

        logger.info("\n输出文件:")
        logger.info("  - output/sft_format/train.json")
        logger.info("  - output/sft_format/test.json")
        logger.info("  - output/sft_format/dataset_info.json")

    except Exception as e:
        logger.error(f"生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
