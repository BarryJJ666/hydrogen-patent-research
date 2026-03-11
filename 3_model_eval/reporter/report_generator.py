# -*- coding: utf-8 -*-
"""
评估报告生成器（简化版）

单模型报告已由 BatchRunner 直接生成，
此类仅用于生成汇总对比报告（可选）
"""
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from config.settings import REPORTS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """评估报告生成器"""

    def __init__(self):
        """初始化报告生成器"""
        self.reports_dir = REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison_report(self, all_results: Dict, test_size: int) -> str:
        """
        生成多模型对比报告（可选）

        Args:
            all_results: 所有模型的评估结果
            test_size: 测试集大小

        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        lines = [
            "# 模型对比评估报告",
            "",
            "**生成时间**: %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "**测试集大小**: %d 条" % test_size,
            "",
            "---",
            "",
            "## 评估结果对比",
            "",
            "| 模型 | 模式 | 准确率(EX) | 工具选择 | 平均轮数 | 平均延迟(ms) | 平均输出Token |",
            "|------|------|------------|----------|----------|-------------|---------------|",
        ]

        for model_name, result in all_results.items():
            metrics = result.get("metrics", {})
            mode = result.get("mode", "direct")

            if mode == "direct":
                accuracy = "%.2f%%" % (metrics.get("execution_accuracy", 0) * 100)
                tool_sel = "-"
                avg_turns = "-"
            else:
                accuracy = "%.2f%%" % (metrics.get("answer_accuracy", 0) * 100)
                tool_sel = "%.2f%%" % (metrics.get("tool_selection_accuracy", 0) * 100)
                avg_turns = "%.2f" % metrics.get("avg_turns", 0)

            latency = "%.1f" % metrics.get("avg_latency_ms", 0)
            tokens = "%.1f" % metrics.get("avg_output_tokens", 0)

            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                model_name, mode, accuracy, tool_sel, avg_turns, latency, tokens
            ))

        lines.extend([
            "",
            "*注: 准确率(EX)表示执行结果与金标一致的比例，Direct模式和Tool Calling模式均基于同一标准*",
            "",
            "---",
            "",
            "*报告由氢能模型评估系统自动生成*",
        ])

        report_path = self.reports_dir / ("comparison_%s.md" % timestamp)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info("Generated comparison report: %s", report_path)
        return str(report_path)

    def generate_comparison_csv(self, all_results: Dict) -> str:
        """生成对比表格（CSV格式）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_lines = [
            "model,mode,execution_accuracy,answer_accuracy,tool_selection_accuracy,avg_turns,psjs,executable_rate,avg_latency_ms,avg_output_tokens"
        ]

        for model_name, result in all_results.items():
            metrics = result.get("metrics", {})
            mode = result.get("mode", "direct")

            csv_lines.append(
                "%s,%s,%.4f,%.4f,%.4f,%.2f,%.4f,%.4f,%.2f,%.1f" % (
                    model_name,
                    mode,
                    metrics.get('execution_accuracy', 0),
                    metrics.get('answer_accuracy', 0),
                    metrics.get('tool_selection_accuracy', 0),
                    metrics.get('avg_turns', 0),
                    metrics.get('psjs', 0),
                    metrics.get('executable_rate', 0),
                    metrics.get('avg_latency_ms', 0),
                    metrics.get('avg_output_tokens', 0)
                )
            )

        csv_path = self.reports_dir / ("comparison_%s.csv" % timestamp)
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(csv_lines))

        logger.info("Generated comparison CSV: %s", csv_path)
        return str(csv_path)
