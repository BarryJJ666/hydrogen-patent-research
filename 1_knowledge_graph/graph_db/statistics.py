# -*- coding: utf-8 -*-
"""
图谱统计信息模块

为LLM提供实际图谱内容的"接地"信息，避免LLM虚构不存在的节点。
统计信息会被注入到Agent的Prompt中，帮助LLM了解图谱中实际有什么数据。
"""
import json
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from utils.logger import get_logger

logger = get_logger(__name__)

# 缓存文件路径
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
STATS_CACHE_FILE = os.path.join(CACHE_DIR, 'graph_statistics.json')


class GraphStatistics:
    """
    图谱统计信息管理

    提供图谱实际内容的统计信息，用于：
    1. 为LLM提供接地信息，防止虚构不存在的节点
    2. 帮助LLM选择正确的查询策略
    3. 提供数据范围参考
    """

    CACHE_TTL_HOURS = 24  # 缓存有效期

    def __init__(self):
        self._cache: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._executor = None

    def _get_executor(self):
        """延迟加载查询执行器"""
        if self._executor is None:
            from graph_db.query_executor import QueryExecutor
            self._executor = QueryExecutor()
        return self._executor

    def get_statistics(self, force_refresh: bool = False) -> Dict:
        """
        获取图谱统计信息

        Args:
            force_refresh: 是否强制刷新缓存

        Returns:
            统计信息字典
        """
        # 检查内存缓存
        if not force_refresh and self._cache:
            if self._cache_time and datetime.now() - self._cache_time < timedelta(hours=self.CACHE_TTL_HOURS):
                return self._cache

        # 检查文件缓存
        if not force_refresh and os.path.exists(STATS_CACHE_FILE):
            try:
                with open(STATS_CACHE_FILE, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                cache_time_str = cached.get('_cache_time')
                if cache_time_str:
                    cache_time = datetime.fromisoformat(cache_time_str)
                    if datetime.now() - cache_time < timedelta(hours=self.CACHE_TTL_HOURS):
                        self._cache = cached
                        self._cache_time = cache_time
                        logger.info("从文件缓存加载图谱统计信息")
                        return self._cache
            except Exception as e:
                logger.warning(f"读取统计缓存失败: {e}")

        # 重新计算统计信息
        logger.info("正在计算图谱统计信息...")
        stats = self._compute_statistics()

        # 保存缓存
        self._cache = stats
        self._cache_time = datetime.now()
        stats['_cache_time'] = self._cache_time.isoformat()

        # 保存到文件
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(STATS_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            logger.info(f"统计信息已缓存到 {STATS_CACHE_FILE}")
        except Exception as e:
            logger.warning(f"保存统计缓存失败: {e}")

        return stats

    def _compute_statistics(self) -> Dict:
        """计算图谱统计信息"""
        executor = self._get_executor()

        stats = {
            'tech_domains': self._query_tech_domains(executor),
            'top_organizations': self._query_top_organizations(executor, limit=100),
            'top_universities': self._query_top_universities(executor, limit=50),
            'date_range': self._query_date_range(executor),
            'total_patents': self._query_total_count(executor),
            'patent_by_year': self._query_patent_by_year(executor),
            'legal_statuses': self._query_legal_statuses(executor),
            'countries': self._query_countries(executor, limit=30),
            'node_counts': self._query_node_counts(executor),
        }

        return stats

    def _query_tech_domains(self, executor) -> List[Dict]:
        """查询技术领域及其专利数量"""
        cypher = """
        MATCH (td:TechDomain)
        OPTIONAL MATCH (p:Patent)-[:BELONGS_TO]->(td)
        OPTIONAL MATCH (td)-[:PARENT_DOMAIN]->(parent:TechDomain)
        RETURN td.name AS name,
               td.level AS level,
               parent.name AS parent_name,
               count(p) AS patent_count
        ORDER BY td.level, patent_count DESC
        """
        result = executor.execute(cypher)
        return result.data if result.success else []

    def _query_top_organizations(self, executor, limit: int) -> List[Dict]:
        """查询专利数量最多的机构"""
        cypher = f"""
        MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
        RETURN o.name AS name,
               o.entity_type AS entity_type,
               count(p) AS patent_count
        ORDER BY patent_count DESC
        LIMIT {limit}
        """
        result = executor.execute(cypher)
        return result.data if result.success else []

    def _query_top_universities(self, executor, limit: int) -> List[Dict]:
        """查询专利数量最多的高校"""
        cypher = f"""
        MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
        WHERE o.entity_type = '高校'
        RETURN o.name AS name, count(p) AS patent_count
        ORDER BY patent_count DESC
        LIMIT {limit}
        """
        result = executor.execute(cypher)
        return result.data if result.success else []

    def _query_date_range(self, executor) -> Dict:
        """查询专利日期范围"""
        cypher = """
        MATCH (p:Patent)
        WHERE p.application_date IS NOT NULL AND p.application_date <> ''
        RETURN min(p.application_date) AS earliest,
               max(p.application_date) AS latest
        """
        result = executor.execute(cypher)
        if result.success and result.data:
            return result.data[0]
        return {'earliest': 'N/A', 'latest': 'N/A'}

    def _query_total_count(self, executor) -> int:
        """查询专利总数"""
        cypher = "MATCH (p:Patent) RETURN count(p) AS total"
        result = executor.execute(cypher)
        if result.success and result.data:
            return result.data[0].get('total', 0)
        return 0

    def _query_patent_by_year(self, executor) -> List[Dict]:
        """查询各年份专利数量"""
        cypher = """
        MATCH (p:Patent)
        WHERE p.application_date IS NOT NULL
        WITH substring(p.application_date, 0, 4) AS year
        WHERE year >= '2000' AND year <= '2030'
        RETURN year, count(*) AS patent_count
        ORDER BY year
        """
        result = executor.execute(cypher)
        return result.data if result.success else []

    def _query_legal_statuses(self, executor) -> List[Dict]:
        """查询法律状态分布"""
        cypher = """
        MATCH (ls:LegalStatus)
        OPTIONAL MATCH (p:Patent)-[:HAS_STATUS]->(ls)
        RETURN ls.name AS name, count(p) AS count
        ORDER BY count DESC
        """
        result = executor.execute(cypher)
        return result.data if result.success else []

    def _query_countries(self, executor, limit: int) -> List[Dict]:
        """查询公开国家分布"""
        cypher = f"""
        MATCH (p:Patent)-[:PUBLISHED_IN]->(c:Country)
        RETURN c.name AS name, count(p) AS patent_count
        ORDER BY patent_count DESC
        LIMIT {limit}
        """
        result = executor.execute(cypher)
        return result.data if result.success else []

    def _query_node_counts(self, executor) -> Dict:
        """查询各类节点数量"""
        counts = {}
        labels = ['Patent', 'Organization', 'Person', 'TechDomain', 'IPCCode', 'Country', 'LegalStatus', 'PatentFamily']

        for label in labels:
            cypher = f"MATCH (n:{label}) RETURN count(n) AS count"
            result = executor.execute(cypher)
            if result.success and result.data:
                counts[label] = result.data[0].get('count', 0)
            else:
                counts[label] = 0

        return counts

    def get_grounding_context(self) -> str:
        """
        生成LLM接地上下文

        返回一个格式化的字符串，包含图谱的关键统计信息，
        帮助LLM了解图谱中实际有什么数据，避免虚构。

        Returns:
            格式化的接地上下文字符串
        """
        stats = self.get_statistics()

        lines = [
            "## 图谱实际内容 (请基于此信息回答，不要虚构不存在的数据)",
            "",
        ]

        # 基本统计
        total = stats.get('total_patents', 0)
        date_range = stats.get('date_range', {})
        lines.append(f"### 数据概览")
        lines.append(f"- 专利总数: {total:,}件")
        lines.append(f"- 日期范围: {date_range.get('earliest', 'N/A')} 至 {date_range.get('latest', 'N/A')}")
        lines.append("")

        # 节点统计
        node_counts = stats.get('node_counts', {})
        if node_counts:
            lines.append(f"### 节点数量")
            for label, count in node_counts.items():
                if count > 0:
                    lines.append(f"- {label}: {count:,}")
            lines.append("")

        # 技术领域
        tech_domains = stats.get('tech_domains', [])
        if tech_domains:
            lines.append("### 技术领域节点 (只有这些是有效的TechDomain)")
            for td in tech_domains:
                name = td.get('name', '')
                level = td.get('level', 0)
                count = td.get('patent_count', 0)
                indent = "  " * (level - 1) if level else ""
                lines.append(f"{indent}- {name} ({count:,}件)")
            lines.append("")
            lines.append("**注意**: '氢能技术'作为顶层概念节点可能不存在，查询所有氢能专利请直接查Patent节点")
            lines.append("")

        # Top机构
        top_orgs = stats.get('top_organizations', [])[:30]
        if top_orgs:
            lines.append("### Top 30 机构 (专利数量)")
            for i, org in enumerate(top_orgs, 1):
                name = org.get('name', '')
                count = org.get('patent_count', 0)
                entity_type = org.get('entity_type', '')
                type_str = f"[{entity_type}]" if entity_type else ""
                lines.append(f"{i}. {name} {type_str} ({count}件)")
            lines.append("")

        # 年度分布
        by_year = stats.get('patent_by_year', [])
        if by_year:
            recent_years = [y for y in by_year if y.get('year', '') >= '2018']
            if recent_years:
                lines.append("### 近年专利数量")
                for y in recent_years:
                    lines.append(f"- {y.get('year')}: {y.get('patent_count', 0):,}件")
                lines.append("")

        # 法律状态
        legal_statuses = stats.get('legal_statuses', [])
        if legal_statuses:
            lines.append("### 法律状态分布")
            for ls in legal_statuses[:10]:
                lines.append(f"- {ls.get('name')}: {ls.get('count', 0):,}件")
            lines.append("")

        return "\n".join(lines)

    def get_schema_summary(self) -> str:
        """
        获取Schema摘要

        Returns:
            Schema摘要字符串
        """
        executor = self._get_executor()
        schema = executor.get_schema_info()

        lines = [
            "## 图谱Schema",
            "",
            "### 节点类型",
        ]

        for label in schema.get('node_labels', []):
            lines.append(f"- {label}")

        lines.append("")
        lines.append("### 关系类型")

        for rel in schema.get('relationship_types', []):
            lines.append(f"- {rel}")

        return "\n".join(lines)


# 全局单例
_graph_stats_instance: Optional[GraphStatistics] = None


def get_graph_statistics() -> GraphStatistics:
    """获取图谱统计单例"""
    global _graph_stats_instance
    if _graph_stats_instance is None:
        _graph_stats_instance = GraphStatistics()
    return _graph_stats_instance


def get_grounding_context() -> str:
    """快捷函数：获取接地上下文"""
    return get_graph_statistics().get_grounding_context()


if __name__ == '__main__':
    # 测试
    stats = GraphStatistics()
    print(stats.get_grounding_context())
