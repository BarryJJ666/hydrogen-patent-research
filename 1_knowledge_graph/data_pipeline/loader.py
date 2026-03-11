# -*- coding: utf-8 -*-
"""
数据加载器
从Excel文件加载专利数据，进行预处理和清洗
"""
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import (
    RAW_DATA_DIR, COLUMN_MAPPING, TECH_DOMAIN_MAPPING,
    CACHE_DIR, DATA_DIR
)
from utils.logger import get_logger
from utils.cache import get_cache_manager

logger = get_logger(__name__)


class DataLoader:
    """
    数据加载器
    - 从Excel加载数据
    - 列名标准化
    - 数据清洗
    - 多值字段拆分
    - 按申请号去重
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = Path(data_dir or RAW_DATA_DIR)
        self.cache = get_cache_manager()

    def load_all(self, use_cache: bool = True) -> Tuple[List[Dict], Dict]:
        """
        加载所有Excel文件

        Args:
            use_cache: 是否使用缓存

        Returns:
            (records, stats) - 专利记录列表和统计信息
        """
        cache_file = "step1_records.json"

        if use_cache and self.cache.exists(cache_file):
            logger.info("从缓存加载数据...")
            records = self.cache.load_json(cache_file)
            if records:
                stats = self._compute_stats(records)
                logger.info(f"缓存加载完成: {len(records)} 条记录")
                return records, stats

        # 获取所有Excel文件
        excel_files = list(self.data_dir.glob("*.xlsx"))
        if not excel_files:
            logger.error(f"未找到Excel文件: {self.data_dir}")
            return [], {}

        logger.info(f"找到 {len(excel_files)} 个Excel文件")

        all_records = []
        file_stats = {}

        for file_path in excel_files:
            logger.info(f"加载: {file_path.name}")
            records, stats = self._load_file(file_path)
            all_records.extend(records)
            file_stats[file_path.stem] = stats

        # 按申请号去重
        unique_records, dup_count = self._deduplicate(all_records)

        # 统计
        total_stats = self._compute_stats(unique_records)
        total_stats["file_stats"] = file_stats
        total_stats["duplicates_removed"] = dup_count

        logger.info(f"加载完成: {len(unique_records)} 条唯一记录 (去重 {dup_count} 条)")

        # 保存缓存
        self.cache.save_json(cache_file, unique_records)

        return unique_records, total_stats

    def _load_file(self, file_path: Path) -> Tuple[List[Dict], Dict]:
        """加载单个Excel文件"""
        try:
            df = pd.read_excel(file_path, dtype=str)
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return [], {}

        # 确定技术领域
        tech_domain = TECH_DOMAIN_MAPPING.get(
            file_path.stem,
            self._infer_tech_domain(file_path.stem)
        )

        records = []
        for _, row in df.iterrows():
            record = self._process_row(row, tech_domain)
            if record and record.get("application_no"):
                records.append(record)

        stats = {
            "total_rows": len(df),
            "valid_records": len(records),
            "tech_domain": tech_domain,
        }

        return records, stats

    def _process_row(self, row: pd.Series, tech_domain: str) -> Optional[Dict]:
        """处理单行数据"""
        record = {"tech_domain": tech_domain}

        for cn_col, en_key in COLUMN_MAPPING.items():
            # 列名可能有变体
            val = None
            for col in row.index:
                if self._normalize_column_name(col) == cn_col:
                    val = row[col]
                    break

            if val is None:
                # 尝试直接匹配
                if cn_col in row.index:
                    val = row[cn_col]

            # 清洗值
            cleaned = self._clean_value(val)
            record[en_key] = cleaned

        # 处理多值字段
        multi_value_fields = [
            "applicants", "current_rights_holders", "transferors", "transferees",
            "licensors", "licensees", "current_licensees", "pledgors", "pledgees",
            "current_pledgees", "plaintiffs", "defendants", "litigation_types"
        ]

        for field in multi_value_fields:
            if record.get(field):
                record[field] = self._split_multi_value(record[field])
            else:
                record[field] = []

        # 处理IPC分类
        if record.get("ipc_main"):
            record["ipc_parsed"] = self._parse_ipc(record["ipc_main"])

        # 构建全文字段
        record["full_text"] = self._build_full_text(record)

        return record

    @staticmethod
    def _normalize_column_name(col: str) -> str:
        """标准化列名"""
        if not isinstance(col, str):
            return str(col)

        # 去除多余空格、换行
        col_clean = re.sub(r'\s+', ' ', col.strip())

        # 处理括号中的备注
        col_clean = re.sub(r'\s*（[^）]*）\s*', '', col_clean)
        col_clean = re.sub(r'\s*\([^)]*\)\s*', '', col_clean)

        # 去除引号
        col_clean = col_clean.strip('"\'')

        return col_clean.strip()

    @staticmethod
    def _clean_value(val: Any) -> Optional[str]:
        """清洗值"""
        if val is None or pd.isna(val):
            return None

        s = str(val).strip()

        # 空值标记
        if s.lower() in ("", "nan", "none", "null", "/", "-", "—", "n/a", "无", "暂无"):
            return None

        return s

    @staticmethod
    def _split_multi_value(val: Optional[str], sep_pattern: str = r'[;；|\n]+') -> List[str]:
        """拆分多值字段"""
        if not val:
            return []

        parts = re.split(sep_pattern, val)
        result = []
        seen = set()

        for p in parts:
            p = p.strip()
            if p and p.lower() not in seen:
                result.append(p)
                seen.add(p.lower())

        return result

    @staticmethod
    def _parse_ipc(ipc_code: str) -> Dict:
        """解析IPC分类号"""
        # IPC格式: A01B1/02
        result = {
            "section": None,
            "class_code": None,
            "subclass": None,
            "group": None,
            "subgroup": None,
        }

        if not ipc_code:
            return result

        ipc_code = ipc_code.strip().upper()

        # 提取section (第一个字母)
        if len(ipc_code) >= 1:
            result["section"] = ipc_code[0]

        # 提取class (section + 2位数字)
        match = re.match(r'^([A-H]\d{2})', ipc_code)
        if match:
            result["class_code"] = match.group(1)

        # 提取subclass (class + 1个字母)
        match = re.match(r'^([A-H]\d{2}[A-Z])', ipc_code)
        if match:
            result["subclass"] = match.group(1)

        return result

    @staticmethod
    def _build_full_text(record: Dict) -> str:
        """构建全文字段"""
        parts = []

        # 标题
        if record.get("title_cn"):
            parts.append(record["title_cn"])
        if record.get("title_en"):
            parts.append(record["title_en"])

        # 摘要
        if record.get("abstract_cn"):
            parts.append(record["abstract_cn"])
        if record.get("abstract_en"):
            parts.append(record["abstract_en"])

        return " ".join(parts)

    @staticmethod
    def _deduplicate(records: List[Dict]) -> Tuple[List[Dict], int]:
        """按申请号去重"""
        seen = set()
        unique = []
        dup_count = 0

        for record in records:
            app_no = record.get("application_no")
            if not app_no:
                continue

            if app_no in seen:
                dup_count += 1
                continue

            seen.add(app_no)
            unique.append(record)

        return unique, dup_count

    @staticmethod
    def _infer_tech_domain(filename: str) -> str:
        """从文件名推断技术领域"""
        if "制氢" in filename:
            return "制氢技术"
        elif "物理储氢" in filename:
            return "物理储氢"
        elif "合金储氢" in filename:
            return "合金储氢"
        elif "无机储氢" in filename:
            return "无机储氢"
        elif "有机储氢" in filename:
            return "有机储氢"
        elif "储氢" in filename:
            return "储氢技术"
        elif "燃料电池" in filename:
            return "氢燃料电池"
        elif "氢制冷" in filename:
            return "氢制冷"
        else:
            return "氢能技术"

    @staticmethod
    def _compute_stats(records: List[Dict]) -> Dict:
        """计算统计信息"""
        stats = {
            "total_records": len(records),
            "tech_domains": defaultdict(int),
            "patent_types": defaultdict(int),
            "countries": defaultdict(int),
            "legal_statuses": defaultdict(int),
        }

        all_applicants = set()
        all_ipc = set()

        for r in records:
            # 技术领域
            td = r.get("tech_domain")
            if td:
                stats["tech_domains"][td] += 1

            # 专利类型
            pt = r.get("patent_type")
            if pt:
                stats["patent_types"][pt] += 1

            # 国家
            country = r.get("publication_country")
            if country:
                stats["countries"][country] += 1

            # 法律状态
            ls = r.get("legal_status")
            if ls:
                stats["legal_statuses"][ls] += 1

            # 申请人
            for app in r.get("applicants", []):
                all_applicants.add(app)

            # IPC
            ipc = r.get("ipc_main")
            if ipc:
                all_ipc.add(ipc)

        stats["unique_applicants"] = len(all_applicants)
        stats["unique_ipc_codes"] = len(all_ipc)

        return stats

    def extract_all_entities(self, records: List[Dict]) -> Set[str]:
        """
        提取所有实体名称（统一收集，不区分person/organization）

        类型判断将由LLM在实体消解阶段完成

        Returns:
            所有实体名称集合
        """
        all_entities = set()

        entity_fields = [
            "applicants", "current_rights_holders", "transferors", "transferees",
            "licensors", "licensees", "current_licensees", "pledgors", "pledgees",
            "current_pledgees", "plaintiffs", "defendants"
        ]

        for record in records:
            for field in entity_fields:
                for name in record.get(field, []):
                    if name and name.strip():
                        all_entities.add(name.strip())

        return all_entities


def load_data(use_cache: bool = True) -> Tuple[List[Dict], Dict]:
    """快捷加载函数"""
    loader = DataLoader()
    return loader.load_all(use_cache=use_cache)
