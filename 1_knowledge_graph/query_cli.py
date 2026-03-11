#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
氢能专利知识图谱 - 智能问答CLI V3
基于完全自主Agent架构

核心特点：
1. LLM自主分析问题、规划策略、选择工具
2. 无固定模板、无正则匹配
3. 支持任意复杂度的问题
4. 详细日志记录到文件便于调试
"""
import sys
import os
import glob
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from langgraph_agent import run_query
from utils.logger import get_logger

logger = get_logger(__name__)


class QueryCLI:
    """交互式问答CLI"""

    def __init__(self):
        self.debug_mode = False
        self.log_dir = PROJECT_ROOT / "logs"

    def query(self, question: str) -> str:
        """处理查询"""
        return run_query(question, debug_mode=self.debug_mode)

    def run(self):
        """运行交互式CLI"""
        print("=" * 60)
        print("  氢能专利知识图谱 - 智能问答系统 V3")
        print("  基于完全自主Agent架构")
        print("=" * 60)
        print("  特点:")
        print("    - LLM自主分析、规划、执行")
        print("    - 支持任意复杂度问题")
        print("    - 自动多步查询和结果整合")
        print("    - 详细日志记录到文件")
        print("=" * 60)
        print("  命令:")
        print("    输入问题开始查询")
        print("    输入 'debug' 切换调试模式")
        print("    输入 'stats' 查看图谱统计")
        print("    输入 'logs'  查看最近日志文件")
        print("    输入 'log'   查看最新日志内容")
        print("    输入 'quit'  退出")
        print("=" * 60)

        while True:
            try:
                question = input("\n请输入问题: ").strip()

                if not question:
                    continue

                if question.lower() in ("quit", "exit", "q"):
                    break

                if question.lower() == "debug":
                    self.debug_mode = not self.debug_mode
                    print("调试模式: {}".format('开启' if self.debug_mode else '关闭'))
                    continue

                if question.lower() == "stats":
                    self._show_stats()
                    continue

                if question.lower() == "logs":
                    self._show_log_files()
                    continue

                if question.lower() == "log":
                    self._show_latest_log()
                    continue

                # 处理查询
                print("\n正在分析问题...")

                # 获取答案
                answer = self.query(question)

                # 显示结果
                print("\n" + "=" * 60)
                print("回答")
                print("=" * 60)
                print("\n{}".format(answer))

            except KeyboardInterrupt:
                print("\n\n检测到中断，退出...")
                break
            except Exception as e:
                logger.error("处理失败: {}".format(e))
                print("\n处理出错: {}".format(e))

        print("\n再见!")

    def _show_stats(self):
        """显示图谱统计信息"""
        try:
            from graph_db.statistics import get_grounding_context
            print("\n" + "=" * 60)
            print("图谱统计信息")
            print("=" * 60)
            print(get_grounding_context())
        except Exception as e:
            print("获取统计信息失败: {}".format(e))

    def _show_log_files(self):
        """显示最近的日志文件列表"""
        try:
            log_files = sorted(
                glob.glob(str(self.log_dir / "agent_*.log")),
                key=os.path.getmtime,
                reverse=True
            )

            print("\n" + "=" * 60)
            print("最近的日志文件 (最新在前)")
            print("=" * 60)

            if not log_files:
                print("暂无日志文件")
                return

            for i, f in enumerate(log_files[:10], 1):
                mtime = os.path.getmtime(f)
                from datetime import datetime
                time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                size = os.path.getsize(f)
                print("  {}. {} ({}, {:.1f}KB)".format(
                    i, os.path.basename(f), time_str, size/1024
                ))

            print("\n日志目录: {}".format(self.log_dir))

        except Exception as e:
            print("获取日志文件列表失败: {}".format(e))

    def _show_latest_log(self):
        """显示最新日志文件的内容"""
        try:
            log_files = sorted(
                glob.glob(str(self.log_dir / "agent_*.log")),
                key=os.path.getmtime,
                reverse=True
            )

            if not log_files:
                print("暂无日志文件")
                return

            latest_log = log_files[0]
            print("\n" + "=" * 60)
            print("最新日志: {}".format(os.path.basename(latest_log)))
            print("=" * 60)

            with open(latest_log, 'r', encoding='utf-8') as f:
                content = f.read()

            # 显示最后2000个字符
            if len(content) > 2000:
                print("... (前面省略 {} 字符)\n".format(len(content) - 2000))
                print(content[-2000:])
            else:
                print(content)

            print("\n完整日志路径: {}".format(latest_log))

        except Exception as e:
            print("读取日志失败: {}".format(e))


def main():
    cli = QueryCLI()
    cli.run()


if __name__ == "__main__":
    main()
