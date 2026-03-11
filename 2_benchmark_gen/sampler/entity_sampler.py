# -*- coding: utf-8 -*-
"""
实体采样器 - 为模式参数提供采样值
"""
import random
from typing import Dict, List, Any, Optional
from .neo4j_sampler import get_sampler
from config.settings import TECH_DOMAINS, ORG_TYPES
from utils.logger import get_logger

logger = get_logger(__name__)


class EntitySampler:
    """实体采样器"""

    def __init__(self):
        self.sampler = get_sampler()
        self._cached_values = {}

    def _ensure_cached(self, key: str, loader_func, *args, **kwargs):
        """确保缓存已加载"""
        if key not in self._cached_values:
            self._cached_values[key] = loader_func(*args, **kwargs)
        return self._cached_values[key]

    def sample_param(self, param_type: str, count: int = 1) -> List[Any]:
        """
        根据参数类型采样值

        Args:
            param_type: 参数类型
            count: 采样数量

        Returns:
            采样值列表
        """
        if param_type == "domain_enum":
            return random.choices(TECH_DOMAINS, k=count)

        elif param_type == "org_type_enum":
            return random.choices(ORG_TYPES, k=count)

        elif param_type == "org_sample":
            orgs = self._ensure_cached("orgs", self.sampler.sample_organizations, 200)
            if orgs:
                return random.choices(orgs, k=min(count, len(orgs)))
            return []

        elif param_type == "province_sample":
            provinces = self._ensure_cached("provinces", self.sampler.sample_provinces, 50)
            if provinces:
                return random.choices(provinces, k=min(count, len(provinces)))
            return []

        elif param_type == "city_sample":
            cities = self._ensure_cached("cities", self.sampler.sample_cities, 100)
            if cities:
                return random.choices(cities, k=min(count, len(cities)))
            return []

        elif param_type == "country_sample":
            countries = self._ensure_cached("countries", self.sampler.sample_countries, 30)
            if countries:
                return random.choices(countries, k=min(count, len(countries)))
            return []

        elif param_type == "year_sample":
            years = self._ensure_cached("years", self.sampler.sample_years, 2005, 2025)
            if years:
                return random.choices(years, k=min(count, len(years)))
            return [str(random.randint(2010, 2024)) for _ in range(count)]

        elif param_type == "transferee_sample":
            transferees = self._ensure_cached("transferees", self.sampler.sample_transferees, 100)
            if transferees:
                return random.choices(transferees, k=min(count, len(transferees)))
            return []

        else:
            logger.warning(f"Unknown param type: {param_type}")
            return []

    def sample_params_for_pattern(self, pattern: Dict) -> Dict[str, Any]:
        """
        为模式的所有参数采样值

        Args:
            pattern: 模式定义

        Returns:
            参数名到采样值的映射
        """
        params = pattern.get("params", {})
        sampled = {}

        for param_name, param_def in params.items():
            param_type = param_def.get("type")
            if param_type:
                values = self.sample_param(param_type, 1)
                if values:
                    sampled[param_name] = values[0]
                else:
                    logger.warning(f"Failed to sample param {param_name} of type {param_type}")
                    return None  # 返回None表示采样失败

        return sampled

    def sample_year_range(self) -> tuple:
        """采样年份范围"""
        years = self._ensure_cached("years", self.sampler.sample_years, 2005, 2025)
        if len(years) >= 2:
            # 确保start <= end
            year1, year2 = random.sample(years, 2)
            start = min(year1, year2)
            end = max(year1, year2)
            return start, end
        return "2015", "2023"


# 单例
_entity_sampler_instance = None

def get_entity_sampler() -> EntitySampler:
    """获取实体采样器单例"""
    global _entity_sampler_instance
    if _entity_sampler_instance is None:
        _entity_sampler_instance = EntitySampler()
    return _entity_sampler_instance
