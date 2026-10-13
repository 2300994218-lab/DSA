# -*- coding: utf-8 -*-
"""指数/市场统计/板块与概念排名/热点/涨停池。

本模块从 ``data_provider.base`` 拆出（v1.1.1 重构），
``DataFetcherManager`` 通过多继承聚合，方法经 MRO 解析，
行为与拆分前完全一致。
"""

from __future__ import annotations

from typing import Callable, Optional, List, Tuple, Dict, Any
from threading import BoundedSemaphore, RLock, Thread
import time
import logging
logger = logging.getLogger("data_provider.base")
from data_provider.base_shared import (
    summarize_exception,
)


class DataFetcherRankingsMixin:
    """Mixin：指数/市场统计/板块与概念排名/热点/涨停池。"""

    _CONCEPT_RANKINGS_CACHE_TTL_SECONDS = 300.0


    _CONCEPT_RANKINGS_EMPTY_CACHE_TTL_SECONDS = 30.0


    _concept_rankings_cache_lock = RLock()


    _concept_rankings_cache: Dict[int, Tuple[float, List[Dict], List[Dict]]] = {}


    def get_main_indices(self, region: str = "cn") -> List[Dict[str, Any]]:
        """获取主要指数实时行情（自动切换数据源）"""
        if region == "cn":
            tickflow_fetcher = self._get_tickflow_fetcher()
            if tickflow_fetcher is not None:
                try:
                    data = tickflow_fetcher.get_main_indices(region=region)
                    if data:
                        logger.info("[TickFlowFetcher] 获取指数行情成功")
                        return data
                except Exception as e:
                    logger.warning(f"[TickFlowFetcher] 获取指数行情失败: {e}")

        for fetcher in self._fetchers:
            if region == "cn" and fetcher.name == "TickFlowFetcher":
                continue
            try:
                data = fetcher.get_main_indices(region=region)
                if data:
                    logger.info(f"[{fetcher.name}] 获取指数行情成功")
                    return data
            except Exception as e:
                logger.warning(f"[{fetcher.name}] 获取指数行情失败: {e}")
                continue
        return []


    def get_market_stats(self, *, purpose: str = "unspecified") -> Dict[str, Any]:
        """获取市场涨跌统计（自动切换数据源）"""
        logger.info("[MarketStats] component=market_stats action=start purpose=%s", purpose)
        tickflow_fetcher = self._get_tickflow_fetcher()
        if tickflow_fetcher is not None:
            started_at = time.monotonic()
            try:
                data = tickflow_fetcher.get_market_stats()
                elapsed = time.monotonic() - started_at
                if data:
                    logger.info(
                        "[MarketStats] component=market_stats action=provider_success "
                        "purpose=%s provider=TickFlowFetcher elapsed=%.2fs",
                        purpose,
                        elapsed,
                    )
                    return data
                logger.info(
                    "[MarketStats] component=market_stats action=provider_empty "
                    "purpose=%s provider=TickFlowFetcher elapsed=%.2fs",
                    purpose,
                    elapsed,
                )
            except Exception as e:
                elapsed = time.monotonic() - started_at
                logger.warning(
                    "[MarketStats] component=market_stats action=provider_failed "
                    "purpose=%s provider=TickFlowFetcher elapsed=%.2fs error=%s",
                    purpose,
                    elapsed,
                    e,
                )

        for fetcher in self._fetchers:
            if fetcher.name == "TickFlowFetcher":
                continue
            started_at = time.monotonic()
            try:
                data = fetcher.get_market_stats()
                elapsed = time.monotonic() - started_at
                if data:
                    logger.info(
                        "[MarketStats] component=market_stats action=provider_success "
                        "purpose=%s provider=%s elapsed=%.2fs",
                        purpose,
                        fetcher.name,
                        elapsed,
                    )
                    return data
                logger.info(
                    "[MarketStats] component=market_stats action=provider_empty "
                    "purpose=%s provider=%s elapsed=%.2fs",
                    purpose,
                    fetcher.name,
                    elapsed,
                )
            except Exception as e:
                elapsed = time.monotonic() - started_at
                logger.warning(
                    "[MarketStats] component=market_stats action=provider_failed "
                    "purpose=%s provider=%s elapsed=%.2fs error=%s",
                    purpose,
                    fetcher.name,
                    elapsed,
                    e,
                )
                continue
        logger.warning("[MarketStats] component=market_stats action=complete status=empty purpose=%s", purpose)
        return {}


    def _run_with_timeout(
        self,
        task: Callable[[], Any],
        timeout_seconds: float,
        task_name: str,
    ) -> Tuple[Optional[Any], Optional[str], int]:
        """
        Execute a task in a short-lived thread and enforce a timeout.

        Returns:
            (result, error, duration_ms)
        """
        start = time.time()
        timeout_value = max(0.0, timeout_seconds)
        if timeout_value <= 0:
            return None, f"{task_name} timeout", 0
        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, Exception] = {}

        if not self._fundamental_timeout_slots.acquire(blocking=False):
            return None, f"{task_name} timeout worker pool exhausted", int(timeout_value * 1000)

        def runner() -> None:
            try:
                result_holder["value"] = task()
            except Exception as exc:
                error_holder["value"] = exc
            finally:
                try:
                    self._fundamental_timeout_slots.release()
                except ValueError:
                    pass

        worker = Thread(target=runner, daemon=True, name=f"fundamental-{task_name}")
        try:
            worker.start()
        except Exception as exc:
            try:
                self._fundamental_timeout_slots.release()
            except ValueError:
                pass
            return None, str(exc), int((time.time() - start) * 1000)
        worker.join(timeout=timeout_value)
        if worker.is_alive():
            return None, f"{task_name} timeout", int(timeout_value * 1000)
        if "value" in error_holder:
            return None, str(error_holder["value"]), int((time.time() - start) * 1000)
        return result_holder.get("value"), None, int((time.time() - start) * 1000)


    def _run_with_retry(
        self,
        task: Callable[[], Any],
        timeout_seconds: float,
        task_name: str,
    ) -> Tuple[Optional[Any], Optional[str], int]:
        """
        Execute a task with bounded budget and best-effort retries.

        Returns:
            (result, error, total_duration_ms)
        """
        config = self._get_fundamental_config()
        attempts = max(1, int(config.fundamental_retry_max))
        remaining_seconds = max(0.0, float(timeout_seconds))
        total_cost_ms = 0
        last_error: Optional[str] = None

        for _ in range(attempts):
            if remaining_seconds <= 0:
                break
            result, err, cost_ms = self._run_with_timeout(task, remaining_seconds, task_name)
            total_cost_ms += cost_ms
            remaining_seconds = max(0.0, remaining_seconds - cost_ms / 1000)
            if err is None:
                return result, None, total_cost_ms
            last_error = err
            if remaining_seconds <= 0:
                break

        return None, last_error, total_cost_ms


    def _get_sector_rankings_with_meta(
            self,
            n: int = 5,
        ) -> Tuple[List[Dict], List[Dict], List[Dict[str, Any]], str]:
            """Get sector rankings with ordered fallback chain metadata."""
            source_chain: List[Dict[str, Any]] = []
            last_error = ""

            # 直接遍历管理器已经按 priority 排好序的数据源列表
            for fetcher in self._fetchers:
                if not hasattr(fetcher, 'get_sector_rankings'):
                    continue

                start = time.time()
                try:
                    data = fetcher.get_sector_rankings(n)
                    duration_ms = int((time.time() - start) * 1000)
                    if data and data[0] is not None and data[1] is not None:
                        source_chain.append(
                            {
                                "provider": fetcher.name,
                                "result": "ok",
                                "duration_ms": duration_ms,
                            }
                        )
                        logger.info(f"[{fetcher.name}] 获取板块排行成功")
                        return data[0], data[1], source_chain, ""

                    last_error = f"{fetcher.name}返回空结果"
                    source_chain.append(
                        {
                            "provider": fetcher.name,
                            "result": "empty",
                            "duration_ms": duration_ms,
                            "error": last_error,
                        }
                    )
                except Exception as e:
                    error_type, error_reason = summarize_exception(e)
                    last_error = f"{fetcher.name} ({error_type}) {error_reason}"
                    duration_ms = int((time.time() - start) * 1000)
                    source_chain.append(
                        {
                            "provider": fetcher.name,
                            "result": "failed",
                            "duration_ms": duration_ms,
                            "error": error_reason,
                        }
                    )
                    logger.warning(f"[{fetcher.name}] 获取板块排行失败: {error_reason}")

            return [], [], source_chain, last_error


    def get_sector_rankings(self, n: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """获取板块涨跌榜（自动切换数据源）"""
        # 按需求固定回退顺序：Akshare(EM) -> Akshare(Sina) -> Tushare -> Efinance
        top, bottom, _, last_error = self._get_sector_rankings_with_meta(n)
        if top or bottom:
            return top, bottom
        logger.warning(f"[板块排行] 所有数据源均失败，最终错误: {last_error}")
        return [], []


    @staticmethod
    def _copy_ranking_rows(rows: List[Dict]) -> List[Dict]:
        return [dict(row) if isinstance(row, dict) else row for row in rows or []]


    @classmethod
    def clear_concept_rankings_cache_for_tests(cls) -> None:
        with cls._concept_rankings_cache_lock:
            cls._concept_rankings_cache.clear()


    def get_concept_rankings(self, n: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """获取概念/题材涨跌榜（自动切换数据源）。"""
        try:
            normalized_n = int(n)
        except (TypeError, ValueError):
            normalized_n = 5
        if normalized_n <= 0:
            normalized_n = 5

        last_error = ""
        now = time.monotonic()

        with self.__class__._concept_rankings_cache_lock:
            cached = self.__class__._concept_rankings_cache.get(normalized_n)
            if cached and cached[0] > now:
                logger.debug("[概念排行] 命中共享缓存 n=%s", normalized_n)
                return self._copy_ranking_rows(cached[1]), self._copy_ranking_rows(cached[2])

            top: List[Dict] = []
            bottom: List[Dict] = []
            for fetcher in self._get_fetchers_snapshot():
                try:
                    data = fetcher.get_concept_rankings(normalized_n)
                    if data and (data[0] or data[1]):
                        top = data[0] or []
                        bottom = data[1] or []
                        logger.info(f"[{fetcher.name}] 获取概念排行成功")
                        break
                    last_error = f"{fetcher.name}返回空结果"
                except Exception as e:
                    error_type, error_reason = summarize_exception(e)
                    last_error = f"{fetcher.name} ({error_type}) {error_reason}"
                    logger.warning(f"[{fetcher.name}] 获取概念排行失败: {error_reason}")

            if not top and not bottom and last_error:
                logger.warning(f"[概念排行] 所有数据源均失败，最终错误: {last_error}")

            ttl = (
                self.__class__._CONCEPT_RANKINGS_CACHE_TTL_SECONDS
                if top or bottom
                else self.__class__._CONCEPT_RANKINGS_EMPTY_CACHE_TTL_SECONDS
            )
            cached_top = self._copy_ranking_rows(top)
            cached_bottom = self._copy_ranking_rows(bottom)
            self.__class__._concept_rankings_cache[normalized_n] = (
                time.monotonic() + ttl,
                cached_top,
                cached_bottom,
            )
            return self._copy_ranking_rows(cached_top), self._copy_ranking_rows(cached_bottom)


    def get_hot_stocks(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取市场人气股榜（自动切换数据源）。"""
        last_error = ""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_hot_stocks(n)
                if data:
                    logger.info(f"[{fetcher.name}] 获取人气股成功")
                    return data[:n]
                last_error = f"{fetcher.name}返回空结果"
            except Exception as e:
                error_type, error_reason = summarize_exception(e)
                last_error = f"{fetcher.name} ({error_type}) {error_reason}"
                logger.warning(f"[{fetcher.name}] 获取人气股失败: {error_reason}")
        if last_error:
            logger.warning(f"[人气股] 所有数据源均失败，最终错误: {last_error}")
        return []


    def get_limit_up_pool(
        self,
        date: Optional[str] = None,
        n: int = 20,
    ) -> List[Dict[str, Any]]:
        """获取涨停池与连板梯队（自动切换数据源）。"""
        last_error = ""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_limit_up_pool(date=date, n=n)
                if data:
                    logger.info(f"[{fetcher.name}] 获取涨停池成功")
                    return data[:n]
                last_error = f"{fetcher.name}返回空结果"
            except Exception as e:
                error_type, error_reason = summarize_exception(e)
                last_error = f"{fetcher.name} ({error_type}) {error_reason}"
                logger.warning(f"[{fetcher.name}] 获取涨停池失败: {error_reason}")
        if last_error:
            logger.warning(f"[涨停池] 所有数据源均失败，最终错误: {last_error}")
        return []
