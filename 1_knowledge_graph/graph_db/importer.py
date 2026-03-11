# -*- coding: utf-8 -*-
"""
Neo4j并行导入器
支持批量导入和断点续传
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import NEO4J_IMPORT, OUTPUT_DIR, CACHE_DIR
from utils.logger import get_logger
from .connection import Neo4jConnection, get_connection

logger = get_logger(__name__)


class Neo4jImporter:
    """
    Neo4j并行导入器
    - 读取Cypher DSL文件
    - 并行执行
    - 断点续传
    - 进度跟踪
    """

    def __init__(self, output_dir: Path = None,
                 connection: Neo4jConnection = None):
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.connection = connection or get_connection()
        self.config = NEO4J_IMPORT

        # 进度跟踪
        self.progress_file = Path(CACHE_DIR) / "import_progress.json"
        self.progress = self._load_progress()
        self._lock = Lock()

    def _load_progress(self) -> Dict:
        """加载导入进度"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed_files": [], "failed_files": {}}

    def _save_progress(self):
        """保存导入进度"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)

    def import_all(self, resume: bool = True) -> Dict[str, Dict]:
        """
        导入所有Cypher文件

        Args:
            resume: 是否从断点恢复

        Returns:
            {filename: {success: int, failed: int}}
        """
        # 加载清单
        manifest_path = self.output_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error(f"清单文件不存在: {manifest_path}")
            return {}

        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        results = {}

        # 按阶段执行
        for phase in manifest.get("phases", []):
            phase_name = phase["name"]
            files = phase["files"]
            parallel = phase.get("parallel", True)

            logger.info(f"=== Phase: {phase_name} ===")

            # 过滤已完成的文件
            if resume:
                files = [f for f in files if f not in self.progress["completed_files"]]

            if not files:
                logger.info(f"  跳过（已完成）")
                continue

            if parallel:
                phase_results = self._import_parallel(files)
            else:
                phase_results = self._import_serial(files)

            results.update(phase_results)

        logger.info("导入完成")
        return results

    def _import_serial(self, files: List[str]) -> Dict[str, Dict]:
        """串行导入"""
        results = {}

        for filename in files:
            result = self._import_file(filename)
            results[filename] = result

            # 更新进度
            with self._lock:
                if result["failed"] == 0:
                    self.progress["completed_files"].append(filename)
                else:
                    self.progress["failed_files"][filename] = result["failed"]
                self._save_progress()

        return results

    def _import_parallel(self, files: List[str]) -> Dict[str, Dict]:
        """并行导入"""
        results = {}
        workers = self.config.get("workers", 4)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._import_file, filename): filename
                for filename in files
            }

            for future in as_completed(futures):
                filename = futures[future]
                try:
                    result = future.result()
                    results[filename] = result

                    # 更新进度
                    with self._lock:
                        if result["failed"] == 0:
                            self.progress["completed_files"].append(filename)
                        else:
                            self.progress["failed_files"][filename] = result["failed"]
                        self._save_progress()

                except Exception as e:
                    logger.error(f"导入失败 {filename}: {e}")
                    results[filename] = {"success": 0, "failed": 1, "error": str(e)}

        return results

    def _import_file(self, filename: str) -> Dict:
        """导入单个Cypher文件"""
        filepath = self.output_dir / filename

        if not filepath.exists():
            return {"success": 0, "failed": 1, "error": "文件不存在"}

        # 读取语句
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        statements = self._parse_statements(content)

        success_count = 0
        failed_count = 0
        batch_size = self.config.get("batch_size", 5000)
        retry_times = self.config.get("retry_times", 3)

        # 批量执行
        for i in range(0, len(statements), batch_size):
            batch = statements[i:i + batch_size]

            for stmt in batch:
                if not stmt.strip():
                    continue

                # 带重试的执行
                success = self._execute_with_retry(stmt, retry_times)

                if success:
                    success_count += 1
                else:
                    failed_count += 1

        logger.info(f"  {filename}: 成功={success_count}, 失败={failed_count}")
        return {"success": success_count, "failed": failed_count}

    def _execute_with_retry(self, statement: str, max_retries: int) -> bool:
        """带重试的执行"""
        for attempt in range(max_retries):
            try:
                with self.connection.session() as session:
                    session.run(statement)
                    return True

            except Exception as e:
                error_msg = str(e)

                # 死锁可以重试
                if "deadlock" in error_msg.lower() or "ForsetiClient" in error_msg:
                    delay = min(10, self.config.get("retry_delay", 2) * (attempt + 1))
                    time.sleep(delay)
                    continue

                # 语法错误不重试
                if "syntax" in error_msg.lower():
                    logger.debug(f"语法错误: {statement[:100]}...")
                    return False

                # 其他错误重试
                if attempt < max_retries - 1:
                    time.sleep(self.config.get("retry_delay", 2))

        return False

    @staticmethod
    def _parse_statements(content: str) -> List[str]:
        """解析Cypher语句"""
        statements = []

        for line in content.split('\n'):
            line = line.strip()

            # 跳过注释和空行
            if not line or line.startswith('//'):
                continue

            # 去除结尾分号
            if line.endswith(';'):
                line = line[:-1]

            if line:
                statements.append(line)

        return statements

    def clear_database(self, confirm: bool = False):
        """清空数据库（危险操作）"""
        if not confirm:
            logger.warning("清空数据库需要确认: clear_database(confirm=True)")
            return

        logger.warning("正在清空数据库...")

        with self.connection.session() as session:
            # 删除所有关系
            session.run("MATCH ()-[r]->() DELETE r")
            # 删除所有节点
            session.run("MATCH (n) DELETE n")

        # 重置进度
        self.progress = {"completed_files": [], "failed_files": {}}
        self._save_progress()

        logger.info("数据库已清空")

    def reset_progress(self):
        """重置导入进度"""
        self.progress = {"completed_files": [], "failed_files": {}}
        self._save_progress()
        logger.info("导入进度已重置")
