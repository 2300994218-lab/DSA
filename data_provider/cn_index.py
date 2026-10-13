# -*- coding: utf-8 -*-
"""A 股指数（daily/realtime/name）数据获取与来源路由。

本模块从 ``data_provider.base`` 拆出（v1.1.1 重构），
``DataFetcherManager`` 通过多继承聚合，方法经 MRO 解析，
行为与拆分前完全一致。
"""

from __future__ import annotations

from src.services.stock_list_parser import AnalysisTarget, ParseStatus, parse_analysis_target
from typing import Callable, Optional, List, Tuple, Dict, Any
from src.data.stock_mapping import STOCK_NAME_MAP, is_meaningful_stock_name
import pandas as pd
from src.services.run_diagnostics import record_provider_run, record_provider_run_started
import time
import logging
logger = logging.getLogger("data_provider.base")
from data_provider.base_shared import (
    STANDARD_COLUMNS,
    summarize_exception,
    normalize_stock_code,
)


class DataFetcherCnIndexMixin:
    """Mixin：A 股指数（daily/realtime/name）数据获取与来源路由。"""

    _CN_INDEX_DAILY_SOURCE_ORDER = (
        "TencentFetcher",
        "AkshareFetcher",
        "TickFlowFetcher",
        "YfinanceFetcher",
    )


    _CN_INDEX_REALTIME_SOURCE_ORDER = (
        ("AkshareFetcher", "tencent"),
        ("AkshareFetcher", "sina"),
        ("EfinanceFetcher", "index"),
        ("TickFlowFetcher", "tickflow"),
    )


    _CN_INDEX_NAME_SOURCE_ORDER = (
        "TencentFetcher",
        "AkshareFetcher",
        "TickFlowFetcher",
    )


    _CN_INDEX_BARE_CODE_CONFLICTS = frozenset({"000001", "000016", "000688"})


    @classmethod
    def _warn_bare_index_conflict(cls, target: AnalysisTarget) -> None:
        if target.asset_type != ParseStatus.STOCK or target.normalized_prefix is not None:
            return
        bare_code = target.normalized_code or (target.raw_input or "").strip()
        if (
            target.matched_index is None
            and bare_code not in cls._CN_INDEX_BARE_CODE_CONFLICTS
        ):
            return
        logger.warning(
            "[指数路由] 裸代码 %s 存在股票/指数歧义，按股票路由处理",
            bare_code,
        )


    @staticmethod
    def _cn_index_provider_symbol(target: AnalysisTarget, fetcher_name: str) -> str:
        entry = target.matched_index
        if entry is None:
            return ""
        exchange = entry.exchange.upper()
        if exchange == "CSI":
            # CSI indices are only supported by AkShare (``csi{code}``); the
            # other providers in the fixed daily chain return an empty symbol
            # so the caller records an ``unsupported`` provider-run and skips.
            if fetcher_name == "AkshareFetcher":
                return f"csi{entry.bare_code}"
            return ""
        if exchange not in {"SH", "SZ"}:
            return ""
        if fetcher_name in {"TencentFetcher", "AkshareFetcher"}:
            return f"{exchange.lower()}{entry.bare_code}"
        if fetcher_name == "TickFlowFetcher":
            return f"{entry.bare_code}.{exchange}"
        if fetcher_name == "YfinanceFetcher":
            suffix = "SS" if exchange == "SH" else "SZ"
            return f"{entry.bare_code}.{suffix}"
        return ""


    @classmethod
    def _is_meaningful_cn_index_name(
        cls, name: Optional[str], target: AnalysisTarget
    ) -> bool:
        if not is_meaningful_stock_name(name, target.canonical_id):
            return False

        aliases = {target.canonical_id}
        entry = target.matched_index
        if entry is not None:
            if normalize_stock_code(str(name).strip()) == entry.bare_code:
                return False
            aliases.add(entry.bare_code)
            aliases.update(entry.aliases)
            display_code = (target.display_code or "").strip()
            registry_name = (entry.display_name or "").strip()
            # The parser currently uses display_code for the human label too.
            if display_code and (
                display_code != registry_name
                or normalize_stock_code(display_code) == entry.bare_code
            ):
                aliases.add(display_code)
            for source_name in cls._CN_INDEX_DAILY_SOURCE_ORDER:
                aliases.add(cls._cn_index_provider_symbol(target, source_name))

        candidate = str(name).strip().upper()
        code_aliases = {
            str(alias).strip().upper()
            for alias in aliases
            if alias is not None and str(alias).strip()
        }
        return candidate not in code_aliases


    def _get_cn_index_daily_data(
        self,
        target: AnalysisTarget,
        start_date: Optional[str],
        end_date: Optional[str],
        days: int,
    ) -> Tuple[pd.DataFrame, str]:
        source_order = self._CN_INDEX_DAILY_SOURCE_ORDER
        fetchers_by_name = {
            fetcher.name: fetcher for fetcher in self._get_fetchers_snapshot()
        }
        errors: List[str] = []
        request_start = time.time()

        for index, source_name in enumerate(source_order):
            fallback_to = source_order[index + 1] if index + 1 < len(source_order) else None
            fetcher = fetchers_by_name.get(source_name)
            provider_symbol = self._cn_index_provider_symbol(target, source_name)

            if not provider_symbol:
                reason = (
                    "unsupported index provider symbol: "
                    f"{target.canonical_id} -> {source_name}"
                )
                record_provider_run(
                    data_type="daily_data",
                    provider=source_name,
                    operation="get_daily_data",
                    success=False,
                    latency_ms=0,
                    error_type="unsupported",
                    error_message=reason,
                    fallback_to=fallback_to,
                    record_count=0,
                )
                logger.warning(
                    "[指数数据源不支持 %d/%d] [%s] %s: %s",
                    index + 1,
                    len(source_order),
                    source_name,
                    target.canonical_id,
                    reason,
                )
                errors.append(f"[{source_name}] {reason}")
                continue

            if fetcher is None or not self._is_fetcher_available(
                fetcher, capability="daily_data"
            ):
                reason = "数据源未配置或暂不可用"
                record_provider_run(
                    data_type="daily_data",
                    provider=source_name,
                    operation="get_daily_data",
                    success=False,
                    latency_ms=0,
                    error_type="unavailable",
                    error_message=reason,
                    fallback_to=fallback_to,
                    record_count=0,
                )
                logger.warning(
                    "[指数数据源失败 %d/%d] [%s] %s: %s",
                    index + 1,
                    len(source_order),
                    source_name,
                    target.canonical_id,
                    reason,
                )
                errors.append(f"[{source_name}] {reason}")
                continue

            if not self._is_daily_source_available(fetcher, "cn_index"):
                reason = self._daily_source_unavailable_error(fetcher)
                record_provider_run(
                    data_type="daily_data",
                    provider=source_name,
                    operation="get_daily_data",
                    success=False,
                    latency_ms=0,
                    error_type="CircuitOpen",
                    error_message=reason,
                    fallback_to=fallback_to,
                    record_count=0,
                )
                logger.warning(
                    "[指数数据源失败 %d/%d] [%s] %s: %s",
                    index + 1,
                    len(source_order),
                    source_name,
                    target.canonical_id,
                    reason,
                )
                errors.append(reason)
                continue

            attempt_start = time.time()
            try:
                logger.info(
                    "[指数数据源尝试 %d/%d] [%s] %s -> %s",
                    index + 1,
                    len(source_order),
                    source_name,
                    target.canonical_id,
                    provider_symbol,
                )
                record_provider_run_started(
                    data_type="daily_data",
                    provider=source_name,
                    operation="get_daily_data",
                )
                df = self._call_fetcher_method(
                    fetcher,
                    "get_daily_data",
                    stock_code=provider_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    days=days,
                )
                duration_ms = int((time.time() - attempt_start) * 1000)
                if df is not None and not df.empty:
                    record_provider_run(
                        data_type="daily_data",
                        provider=source_name,
                        operation="get_daily_data",
                        success=True,
                        latency_ms=duration_ms,
                        record_count=len(df),
                    )
                    self._record_daily_source_success(fetcher, "cn_index")
                    logger.info(
                        "[指数数据源完成] %s 使用 [%s] 获取成功: rows=%d, elapsed=%.2fs",
                        target.canonical_id,
                        source_name,
                        len(df),
                        time.time() - request_start,
                    )
                    return df, source_name

                reason = "empty result"
                record_provider_run(
                    data_type="daily_data",
                    provider=source_name,
                    operation="get_daily_data",
                    success=False,
                    latency_ms=duration_ms,
                    error_type="empty",
                    error_message=reason,
                    fallback_to=fallback_to,
                    record_count=0,
                )
                if df is not None and df.empty:
                    self._record_daily_source_success(fetcher, "cn_index")
                else:
                    self._record_daily_source_failure(fetcher, "cn_index", reason)
                logger.warning(
                    "[指数数据源失败 %d/%d] [%s] %s: %s",
                    index + 1,
                    len(source_order),
                    source_name,
                    target.canonical_id,
                    reason,
                )
                errors.append(f"[{source_name}] {reason}")
            except Exception as exc:
                error_type, error_reason = summarize_exception(exc)
                duration_ms = int((time.time() - attempt_start) * 1000)
                record_provider_run(
                    data_type="daily_data",
                    provider=source_name,
                    operation="get_daily_data",
                    success=False,
                    latency_ms=duration_ms,
                    error_type=error_type,
                    error_message=error_reason,
                    fallback_to=fallback_to,
                    record_count=0,
                )
                self._record_daily_source_failure(
                    fetcher, "cn_index", error_reason
                )
                logger.warning(
                    "[指数数据源失败 %d/%d] [%s] %s: error_type=%s, reason=%s",
                    index + 1,
                    len(source_order),
                    source_name,
                    target.canonical_id,
                    error_type,
                    error_reason,
                )
                errors.append(f"[{source_name}] ({error_type}) {error_reason}")

        logger.warning(
            "[指数数据源终止] %s 所有指数日线数据源均失败: elapsed=%.2fs; %s",
            target.canonical_id,
            time.time() - request_start,
            "; ".join(errors) or "暂无可用数据源",
        )
        return pd.DataFrame(columns=STANDARD_COLUMNS), ""


    def _get_cn_index_realtime_quote(
        self,
        target: AnalysisTarget,
        *,
        log_final_failure: bool = True,
    ):
        """Fetch a realtime quote for a registered CN index via a fixed chain.

        Chain (Story 1.5):
        1. Tencent via AkshareFetcher (``sh000016``/``sz399001`` prefixed symbol)
        2. Sina via AkshareFetcher (same prefixed symbol)
        3. Eastmoney single-stock secid via EfinanceFetcher (SH/SZ/CSI)
        4. TickFlow (``000016.SH`` symbol; SH/SZ only)

        The explicit index identity is preserved end-to-end: provider symbols
        are derived from the registry entry, never from ``normalize_stock_code``,
        so ``sh000016`` can never degrade into the stock ``000016`` path.
        """
        fetchers_by_name = {
            fetcher.name: fetcher for fetcher in self._get_fetchers_snapshot()
        }
        errors: List[str] = []
        request_start = time.time()
        source_order = self._CN_INDEX_REALTIME_SOURCE_ORDER

        for index, (source_name, source_kind) in enumerate(source_order):
            fallback_to = (
                source_order[index + 1][0] if index + 1 < len(source_order) else None
            )
            fetcher = fetchers_by_name.get(source_name)
            provider_symbol = self._cn_index_realtime_provider_symbol(
                target, source_name, source_kind
            )

            if not provider_symbol:
                reason = (
                    "unsupported index provider symbol: "
                    f"{target.canonical_id} -> {source_name}"
                )
                record_provider_run(
                    data_type="realtime_quote",
                    provider=source_name,
                    operation="get_realtime_quote",
                    success=False,
                    latency_ms=0,
                    error_type="unsupported",
                    error_message=reason,
                    fallback_to=fallback_to,
                    record_count=0,
                )
                logger.warning(
                    "[指数实时行情不支持 %d/%d] [%s] %s: %s",
                    index + 1,
                    len(source_order),
                    source_name,
                    target.canonical_id,
                    reason,
                )
                errors.append(f"[{source_name}] {reason}")
                continue

            if fetcher is None or not self._is_fetcher_available(
                fetcher, capability="realtime_quote"
            ):
                reason = "数据源未配置或暂不可用"
                record_provider_run(
                    data_type="realtime_quote",
                    provider=source_name,
                    operation="get_realtime_quote",
                    success=False,
                    latency_ms=0,
                    error_type="unavailable",
                    error_message=reason,
                    fallback_to=fallback_to,
                    record_count=0,
                )
                logger.warning(
                    "[指数实时行情失败 %d/%d] [%s] %s: %s",
                    index + 1,
                    len(source_order),
                    source_name,
                    target.canonical_id,
                    reason,
                )
                errors.append(f"[{source_name}] {reason}")
                continue

            attempt_start = time.time()
            try:
                logger.info(
                    "[指数实时行情尝试 %d/%d] [%s] %s -> %s",
                    index + 1,
                    len(source_order),
                    source_name,
                    target.canonical_id,
                    provider_symbol,
                )
                record_provider_run_started(
                    data_type="realtime_quote",
                    provider=source_name,
                    operation="get_realtime_quote",
                )
                if source_kind == "index":
                    quote = self._call_fetcher_method(
                        fetcher,
                        "get_index_realtime_quote",
                        target.canonical_id,
                    )
                elif source_kind == "tickflow":
                    quote = self._call_fetcher_method(
                        fetcher,
                        "get_realtime_quote",
                        provider_symbol,
                    )
                else:
                    quote = self._call_fetcher_method(
                        fetcher,
                        "get_realtime_quote",
                        provider_symbol,
                        source=source_kind,
                    )
                duration_ms = int((time.time() - attempt_start) * 1000)
                if quote is not None and quote.has_basic_data():
                    record_provider_run(
                        data_type="realtime_quote",
                        provider=source_name,
                        operation="get_realtime_quote",
                        success=True,
                        latency_ms=duration_ms,
                        record_count=1,
                    )
                    logger.info(
                        "[指数实时行情完成] %s 使用 [%s] 获取成功: elapsed=%.2fs",
                        target.canonical_id,
                        source_name,
                        time.time() - request_start,
                    )
                    from src.config import get_config as _get_config_safe

                    return self._enrich_realtime_quote(
                        quote,
                        realtime_cache_ttl=getattr(
                            _get_config_safe(), "realtime_cache_ttl", None
                        ),
                    )

                reason = "empty or incomplete quote"
                record_provider_run(
                    data_type="realtime_quote",
                    provider=source_name,
                    operation="get_realtime_quote",
                    success=False,
                    latency_ms=duration_ms,
                    error_type="empty",
                    error_message=reason,
                    fallback_to=fallback_to,
                    record_count=0,
                )
                logger.warning(
                    "[指数实时行情失败 %d/%d] [%s] %s: %s",
                    index + 1,
                    len(source_order),
                    source_name,
                    target.canonical_id,
                    reason,
                )
                errors.append(f"[{source_name}] {reason}")
            except Exception as exc:
                error_type, error_reason = summarize_exception(exc)
                duration_ms = int((time.time() - attempt_start) * 1000)
                record_provider_run(
                    data_type="realtime_quote",
                    provider=source_name,
                    operation="get_realtime_quote",
                    success=False,
                    latency_ms=duration_ms,
                    error_type=error_type,
                    error_message=error_reason,
                    fallback_to=fallback_to,
                    record_count=0,
                )
                logger.warning(
                    "[指数实时行情失败 %d/%d] [%s] %s: error_type=%s, reason=%s",
                    index + 1,
                    len(source_order),
                    source_name,
                    target.canonical_id,
                    error_type,
                    error_reason,
                )
                errors.append(f"[{source_name}] ({error_type}) {error_reason}")

        if log_final_failure:
            logger.warning(
                "[指数实时行情终止] %s 所有指数实时行情数据源均失败: elapsed=%.2fs; %s",
                target.canonical_id,
                time.time() - request_start,
                "; ".join(errors) or "暂无可用数据源",
            )
        return None


    @classmethod
    def _cn_index_realtime_provider_symbol(
        cls,
        target: AnalysisTarget,
        fetcher_name: str,
        source_kind: str,
    ) -> str:
        """Derive the provider symbol for the CN index realtime chain."""
        entry = target.matched_index
        if entry is None:
            return ""
        exchange = entry.exchange.upper()
        if exchange == "CSI":
            # CSI indices are only supported by the Eastmoney single-stock
            # secid endpoint (EfinanceFetcher); other providers return an
            # empty symbol so the caller records ``unsupported`` and skips.
            if fetcher_name == "EfinanceFetcher" and source_kind == "index":
                return target.canonical_id
            return ""
        if exchange not in {"SH", "SZ"}:
            return ""
        if fetcher_name == "AkshareFetcher" and source_kind in ("tencent", "sina"):
            return f"{exchange.lower()}{entry.bare_code}"
        if fetcher_name == "EfinanceFetcher" and source_kind == "index":
            return target.canonical_id
        if fetcher_name == "TickFlowFetcher" and source_kind == "tickflow":
            return f"{entry.bare_code}.{exchange}"
        return ""


    def _get_cn_index_name(self, target: AnalysisTarget) -> str:
        cache_key = target.canonical_id
        entry = target.matched_index
        display_name = entry.display_name if entry is not None else ""
        if self._is_meaningful_cn_index_name(display_name, target):
            return self._cache_stock_name(cache_key, display_name) or display_name

        cached_name = self._get_cached_stock_name(cache_key)
        if cached_name is not None:
            if self._is_meaningful_cn_index_name(cached_name, target):
                return cached_name
            self._discard_cached_stock_name(cache_key)

        fetchers_by_name = {
            fetcher.name: fetcher for fetcher in self._get_fetchers_snapshot()
        }
        for source_name in self._CN_INDEX_NAME_SOURCE_ORDER:
            provider_symbol = self._cn_index_provider_symbol(target, source_name)
            if not provider_symbol:
                logger.warning(
                    "[指数名称] [%s] 不支持 provider symbol: %s",
                    source_name,
                    cache_key,
                )
                continue
            fetcher = fetchers_by_name.get(source_name)
            if (
                fetcher is None
                or not hasattr(fetcher, "get_stock_name")
                or not self._is_fetcher_available(fetcher, capability="stock_name")
            ):
                logger.warning(
                    "[指数名称] [%s] 未配置或暂不可用: %s",
                    source_name,
                    cache_key,
                )
                continue
            try:
                name = self._call_fetcher_method(
                    fetcher, "get_stock_name", provider_symbol
                )
            except Exception as exc:
                logger.warning(
                    "[指数名称] [%s] 获取 %s 失败: %s",
                    source_name,
                    cache_key,
                    exc,
                )
                continue
            if self._is_meaningful_cn_index_name(name, target):
                self._cache_stock_name(cache_key, name)
                logger.info(
                    "[指数名称] 从 %s 获取: %s -> %s",
                    source_name,
                    cache_key,
                    name,
                )
                return name
            logger.warning(
                "[指数名称] [%s] 获取 %s 返回空结果",
                source_name,
                cache_key,
            )

        static_name = STOCK_NAME_MAP.get(cache_key)
        if static_name and self._is_meaningful_cn_index_name(static_name, target):
            self._cache_stock_name(cache_key, static_name)
            logger.info(
                "[指数名称] 从静态映射获取: %s -> %s",
                cache_key,
                static_name,
            )
            return static_name

        logger.warning("[指数名称] 所有数据源都无法获取 %s 的名称", cache_key)
        return cache_key
