# -*- coding: utf-8 -*-
"""
搜索引擎 Provider 集合。

职责：聚合各搜索引擎 Provider，统一从 ``src.search_providers`` 导出。
本包从 ``src.search_service`` 拆出（v1.1.1 重构），公开导入路径保持不变。
"""

from src.search_providers.anspire import AnspireSearchProvider
from src.search_providers.base import BaseSearchProvider
from src.search_providers.bocha import BochaSearchProvider
from src.search_providers.brave import BraveSearchProvider
from src.search_providers.intelligence import IntelligenceSearchProvider
from src.search_providers.minimax import MiniMaxSearchProvider
from src.search_providers.serpapi import SerpAPISearchProvider
from src.search_providers.searxng import SearXNGSearchProvider
from src.search_providers.tavily import TavilySearchProvider

__all__ = [
    "AnspireSearchProvider",
    "BaseSearchProvider",
    "BochaSearchProvider",
    "BraveSearchProvider",
    "IntelligenceSearchProvider",
    "MiniMaxSearchProvider",
    "SerpAPISearchProvider",
    "SearXNGSearchProvider",
    "TavilySearchProvider",
]
