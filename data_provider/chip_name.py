# -*- coding: utf-8 -*-
"""筹码分布、股票名称批量获取与板块归属。

本模块从 ``data_provider.base`` 拆出（v1.1.1 重构），
``DataFetcherManager`` 通过多继承聚合，方法经 MRO 解析，
行为与拆分前完全一致。
"""

from __future__ import annotations

from typing import Callable, Optional, List, Tuple, Dict, Any
from src.services.stock_list_parser import AnalysisTarget, ParseStatus, parse_analysis_target
from src.data.stock_mapping import STOCK_NAME_MAP, is_meaningful_stock_name
from src.data.stock_index_loader import get_index_stock_name
from src.services.run_diagnostics import record_provider_run, record_provider_run_started
import time
import json
from typing import Set
import logging
logger = logging.getLogger("data_provider.base")
from data_provider.base_shared import (
    summarize_exception,
    normalize_stock_code,
    _is_meaningful_chip_distribution,
    _market_tag,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_provider.base import BaseFetcher, DataFetcherManager


class DataFetcherChipNameMixin:
    """Mixin：筹码分布、股票名称批量获取与板块归属。"""

    def get_chip_distribution(self, stock_code: str):
        """
        获取筹码分布数据（带熔断和多数据源降级）

        策略：
        1. 检查配置开关
        2. 检查熔断器状态
        3. 依次尝试多个数据源：数据源优先级与获取daily的数据优先级一致
        4. 所有数据源失败则返回 None（降级兜底）

        Args:
            stock_code: 股票代码

        Returns:
            ChipDistribution 对象，失败则返回 None
        """
        # Normalize code (strip SH/SZ prefix etc.)
        stock_code = normalize_stock_code(stock_code)

        from .realtime_types import get_chip_circuit_breaker
        from src.config import get_config

        config = get_config()

        # 如果筹码分布功能被禁用，直接返回 None
        if not config.enable_chip_distribution:
            logger.debug(f"[筹码分布] 功能已禁用，跳过 {stock_code}")
            return None

        circuit_breaker = get_chip_circuit_breaker()

        candidate_fetchers = []
        # 直接遍历管理器已经按 priority 排好序的数据源列表
        for fetcher in self._get_fetchers_snapshot():
            # 只处理实现了筹码分布逻辑的数据源
            if not hasattr(fetcher, 'get_chip_distribution'):
                continue

            fetcher_name = fetcher.name
            # 动态生成熔断器的 key，例如 "TushareFetcher" -> "tushare_chip"
            source_key = f"{fetcher_name.replace('Fetcher', '').lower()}_chip"

            # 检查熔断器状态
            if not circuit_breaker.is_available(source_key):
                logger.debug(f"[熔断] {fetcher_name} 筹码接口处于熔断状态，尝试下一个")
                continue

            candidate_fetchers.append((fetcher, fetcher_name, source_key))

        for index, (fetcher, fetcher_name, source_key) in enumerate(candidate_fetchers):
            fallback_to = (
                candidate_fetchers[index + 1][1]
                if index + 1 < len(candidate_fetchers)
                else None
            )
            attempt_start = time.time()
            try:
                record_provider_run_started(
                    data_type="chip",
                    provider=fetcher_name,
                    operation="get_chip_distribution",
                )
                chip = self._call_fetcher_method(fetcher, 'get_chip_distribution', stock_code)
                latency_ms = int((time.time() - attempt_start) * 1000)
                if _is_meaningful_chip_distribution(chip):
                    record_provider_run(
                        data_type="chip",
                        provider=fetcher_name,
                        operation="get_chip_distribution",
                        success=True,
                        latency_ms=latency_ms,
                        record_count=1,
                    )
                    circuit_breaker.record_success(source_key)
                    logger.info(f"[筹码分布] {stock_code} 成功获取 (来源: {fetcher_name})")
                    return chip
                else:
                    record_provider_run(
                        data_type="chip",
                        provider=fetcher_name,
                        operation="get_chip_distribution",
                        success=False,
                        latency_ms=latency_ms,
                        error_type="empty",
                        error_message="empty or incomplete chip distribution",
                        fallback_to=fallback_to,
                        record_count=0,
                    )
                    if chip is not None:
                        logger.warning(
                            "[筹码分布] %s 返回字段不完整或占位值，继续尝试下一个数据源",
                            fetcher_name,
                        )
                    # 空结果或占位结果：释放 HALF_OPEN 探测名额，避免卡死
                    circuit_breaker.record_inconclusive(source_key)
            except Exception as e:
                error_type, error_reason = summarize_exception(e)
                record_provider_run(
                    data_type="chip",
                    provider=fetcher_name,
                    operation="get_chip_distribution",
                    success=False,
                    latency_ms=int((time.time() - attempt_start) * 1000),
                    error_type=error_type,
                    error_message=error_reason,
                    fallback_to=fallback_to,
                )
                logger.warning(f"[筹码分布] {fetcher_name} 获取 {stock_code} 失败: {e}")
                circuit_breaker.record_failure(source_key, str(e))
                continue

        logger.warning(f"[筹码分布] {stock_code} 所有数据源均失败")
        return None


    def get_stock_name(self, stock_code: str, allow_realtime: bool = True) -> Optional[str]:
        """
        获取股票中文名称（自动切换数据源）
        
        尝试从多个数据源获取股票名称：
        1. 先从内存缓存中获取（如果有）
        2. 再尝试本地维护映射与 stocks.index.json 索引
        3. 然后按需查询实时行情
        4. 依次尝试各个数据源的 get_stock_name 方法
        
        Args:
            stock_code: 股票代码
            allow_realtime: Whether to query realtime quote first. Set False when
                caller only wants lightweight prefetch without triggering heavy
                realtime source calls.
            
        Returns:
            股票中文名称，所有数据源都失败则返回 None
        """
        raw_stock_code = (stock_code or "").strip()
        target = parse_analysis_target(raw_stock_code)
        self._warn_bare_index_conflict(target)
        if target.asset_type == ParseStatus.UNSUPPORTED:
            logger.warning(
                "[股票名称] 跳过不支持的输入 %s: %s",
                raw_stock_code,
                target.unsupported_reason or "unsupported analysis target",
            )
            return ""
        if target.asset_type == ParseStatus.INDEX:
            return self._get_cn_index_name(target)

        # Normalize code (strip SH/SZ prefix etc.)
        stock_code = normalize_stock_code(stock_code)
        static_name = STOCK_NAME_MAP.get(stock_code)

        # 1. 先检查缓存
        cached_name = self._get_cached_stock_name(stock_code)
        if cached_name is not None:
            return cached_name
        
        if is_meaningful_stock_name(static_name, stock_code):
            return self._cache_stock_name(stock_code, static_name) or static_name

        index_name = get_index_stock_name(stock_code)
        if is_meaningful_stock_name(index_name, stock_code):
            return self._cache_stock_name(stock_code, index_name) or index_name

        # 2. 尝试从实时行情中获取（最快，可按需禁用）
        if allow_realtime:
            quote = self.get_realtime_quote(raw_stock_code or stock_code, log_final_failure=False)
            if quote and hasattr(quote, 'name') and is_meaningful_stock_name(getattr(quote, 'name', ''), stock_code):
                name = quote.name
                self._cache_stock_name(stock_code, name)
                logger.info(f"[股票名称] 从实时行情获取: {stock_code} -> {name}")
                return name

        # 3. 依次尝试各个数据源
        from .akshare_fetcher import _is_us_code
        is_us = _is_us_code(stock_code)
        _US_CAPABLE_FETCHERS = {"YfinanceFetcher", "LongbridgeFetcher", "FinnhubFetcher", "AlphaVantageFetcher"}
        for fetcher in self._get_fetchers_snapshot():
            if not hasattr(fetcher, 'get_stock_name'):
                continue
            if is_us and fetcher.name not in _US_CAPABLE_FETCHERS:
                continue
            if not self._is_fetcher_available(fetcher, capability="stock_name"):
                continue
            try:
                name = self._call_fetcher_method(fetcher, 'get_stock_name', stock_code)
                if is_meaningful_stock_name(name, stock_code):
                    self._cache_stock_name(stock_code, name)
                    logger.info(f"[股票名称] 从 {fetcher.name} 获取: {stock_code} -> {name}")
                    return name
            except Exception as e:
                logger.debug(f"[股票名称] {fetcher.name} 获取失败: {e}")
                continue

        # 4. 所有数据源都失败
        logger.warning(f"[股票名称] 所有数据源都无法获取 {stock_code} 的名称")
        return ""


    def get_belong_boards(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        Get stock membership boards through capability probing.

        Keep this at manager layer to avoid changing BaseFetcher abstraction.
        """
        stock_code = normalize_stock_code(stock_code)
        if _market_tag(stock_code) != "cn":
            return []
        candidate_fetchers = [
            fetcher
            for fetcher in self._fetchers
            if hasattr(fetcher, "get_belong_board")
        ]
        for index, fetcher in enumerate(candidate_fetchers):
            fallback_to = (
                candidate_fetchers[index + 1].name
                if index + 1 < len(candidate_fetchers)
                else None
            )
            start = time.time()
            try:
                record_provider_run_started(
                    data_type="belong_boards",
                    provider=fetcher.name,
                    operation="get_belong_board",
                )
                raw_data = fetcher.get_belong_board(stock_code)
                boards = self._normalize_belong_boards(raw_data)
                if boards:
                    record_provider_run(
                        data_type="belong_boards",
                        provider=fetcher.name,
                        operation="get_belong_board",
                        success=True,
                        latency_ms=int((time.time() - start) * 1000),
                        record_count=len(boards),
                    )
                    logger.info(f"[{fetcher.name}] 获取所属板块成功: {stock_code}, count={len(boards)}")
                    return boards
                record_provider_run(
                    data_type="belong_boards",
                    provider=fetcher.name,
                    operation="get_belong_board",
                    success=False,
                    latency_ms=int((time.time() - start) * 1000),
                    error_type="empty",
                    error_message="empty belong boards",
                    fallback_to=fallback_to,
                    record_count=0,
                )
            except Exception as e:
                error_type, error_reason = summarize_exception(e)
                record_provider_run(
                    data_type="belong_boards",
                    provider=fetcher.name,
                    operation="get_belong_board",
                    success=False,
                    latency_ms=int((time.time() - start) * 1000),
                    error_type=error_type,
                    error_message=error_reason,
                    fallback_to=fallback_to,
                )
                logger.debug(f"[{fetcher.name}] 获取所属板块失败: {e}")
                continue
        return []


    def prefetch_stock_names(self, stock_codes: List[str], use_bulk: bool = False) -> None:
        """
        Pre-fetch stock names into cache before parallel analysis (Issue #455).

        When use_bulk=False, only calls get_stock_name per code (no get_stock_list),
        avoiding full-market fetch. Sequential execution to avoid rate limits.

        Args:
            stock_codes: Stock codes to prefetch.
            use_bulk: If True, may use get_stock_list (full fetch). Default False.
        """
        if not stock_codes:
            return
        normalized_codes: List[str] = []
        for code in stock_codes:
            target = parse_analysis_target(code)
            if target.asset_type == ParseStatus.UNSUPPORTED:
                logger.warning(
                    "[股票名称] 预取跳过不支持的输入 %s: %s",
                    code,
                    target.unsupported_reason or "unsupported analysis target",
                )
                continue
            normalized_codes.append(
                target.canonical_id
                if target.asset_type == ParseStatus.INDEX
                else normalize_stock_code(code)
            )
        stock_codes = normalized_codes
        if use_bulk:
            self.batch_get_stock_names(stock_codes)
            return
        for code in stock_codes:
            # Skip realtime lookup to avoid triggering expensive full-market quote
            # requests during the prefetch phase.
            self.get_stock_name(code, allow_realtime=False)


    def batch_get_stock_names(self, stock_codes: List[str]) -> Dict[str, str]:
        """
        批量获取股票中文名称
        
        先尝试从支持批量查询的数据源获取股票列表，
        然后再逐个查询缺失的股票名称。
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            {股票代码: 股票名称} 字典
        """
        result = {}
        missing_codes = set(stock_codes)
        
        # 1. 先检查缓存
        self._ensure_concurrency_guards()
        with self._stock_name_cache_lock:
            for code in stock_codes:
                cached_name = self._stock_name_cache.get(code)
                if cached_name is not None:
                    result[code] = cached_name
                    missing_codes.discard(code)
        
        if not missing_codes:
            return result
        
        # 2. 尝试批量获取股票列表
        for fetcher in self._get_fetchers_snapshot():
            if not hasattr(fetcher, 'get_stock_list') or not missing_codes:
                continue
            if not self._is_fetcher_available(fetcher, capability="stock_list"):
                continue
            try:
                stock_list = self._call_fetcher_method(fetcher, 'get_stock_list')
                if stock_list is not None and not stock_list.empty:
                    cache_updates: Dict[str, str] = {}
                    for _, row in stock_list.iterrows():
                        code = row.get('code')
                        name = row.get('name')
                        if code and name:
                            cache_updates[code] = name
                            if code in missing_codes:
                                result[code] = name
                                missing_codes.discard(code)

                    if cache_updates:
                        with self._stock_name_cache_lock:
                            self._stock_name_cache.update(cache_updates)
                    
                    if not missing_codes:
                        break
                    
                    logger.info(f"[股票名称] 从 {fetcher.name} 批量获取完成，剩余 {len(missing_codes)} 个待查")
            except Exception as e:
                logger.debug(f"[股票名称] {fetcher.name} 批量获取失败: {e}")
                continue
        
        # 3. 逐个获取剩余的
        for code in list(missing_codes):
            name = self.get_stock_name(code)
            if name:
                result[code] = name
                missing_codes.discard(code)
        
        logger.info(f"[股票名称] 批量获取完成，成功 {len(result)}/{len(stock_codes)}")
        return result
