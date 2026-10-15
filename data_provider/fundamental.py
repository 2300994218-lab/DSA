# -*- coding: utf-8 -*-
"""基本面/资金流/龙虎榜/板块上下文、离岸数据捆绑与缓存。

本模块从 ``data_provider.base`` 拆出（v1.1.1 重构），
``DataFetcherManager`` 通过多继承聚合，方法经 MRO 解析，
行为与拆分前完全一致。
"""

from __future__ import annotations

from typing import Callable, Optional, List, Tuple, Dict, Any
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import time
import re
import logging
logger = logging.getLogger("data_provider.base")
from data_provider.base_shared import (
    normalize_stock_code,
    _is_etf_code,
    _market_tag,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_provider.base import BaseFetcher, DataFetcherManager


class DataFetcherFundamentalMixin:
    """Mixin：基本面/资金流/龙虎榜/板块上下文、离岸数据捆绑与缓存。"""

    def _get_fundamental_cache_key(self, stock_code: str, budget_seconds: Optional[float] = None) -> str:
        """生成基本面缓存 key（包含预算分桶以避免低预算结果污染高预算请求）。"""
        normalized_code = normalize_stock_code(stock_code)
        if budget_seconds is None:
            return f"{normalized_code}|budget=default"
        try:
            budget = max(0.0, float(budget_seconds))
        except (TypeError, ValueError):
            budget = 0.0
        # 100ms bucket to balance cache reuse and scenario isolation.
        budget_bucket = int(round(budget * 10))
        return f"{normalized_code}|budget={budget_bucket}"


    def _prune_fundamental_cache(self, ttl_seconds: int, max_entries: int) -> None:
        """Prune expired and overflow fundamental cache items."""
        with self._fundamental_cache_lock:
            if not self._fundamental_cache:
                return

            now_ts = time.time()
            if ttl_seconds > 0:
                cache_items = list(self._fundamental_cache.items())
                expired_keys = [
                    key
                    for key, value in cache_items
                    if now_ts - float(value.get("ts", 0)) > ttl_seconds
                ]
                for key in expired_keys:
                    self._fundamental_cache.pop(key, None)

            if max_entries > 0 and len(self._fundamental_cache) > max_entries:
                overflow = len(self._fundamental_cache) - max_entries
                sorted_items = sorted(
                    list(self._fundamental_cache.items()),
                    key=lambda item: float(item[1].get("ts", 0)),
                )
                for key, _ in sorted_items[:overflow]:
                    self._fundamental_cache.pop(key, None)


    @staticmethod
    def _try_scalar_isna(value: Any, context: str) -> Optional[bool]:
        """Return scalar ``pd.isna`` result, or ``None`` when callers should use fallback logic."""
        if isinstance(value, (dict, list, tuple, set, pd.DataFrame, pd.Series, pd.Index)):
            return None

        if isinstance(value, np.ndarray):
            if value.ndim != 0:
                return None
            value = value.item()

        try:
            isna_result = pd.isna(value)
        except (TypeError, ValueError) as exc:
            if hasattr(value, "__array__"):
                logger.debug(
                    "[%s] pd.isna failed for array-like object; re-raise: value_type=%s error_type=%s",
                    context,
                    type(value).__name__,
                    type(exc).__name__,
                )
                raise
            logger.debug(
                "[%s] pd.isna fallback: value_type=%s error_type=%s",
                context,
                type(value).__name__,
                type(exc).__name__,
            )
            return None

        if isinstance(isna_result, (bool, np.bool_)):
            return bool(isna_result)

        if isinstance(isna_result, np.ndarray):
            if isna_result.ndim == 0:
                return bool(isna_result.item())
            logger.debug(
                "[%s] pd.isna returned non-scalar result: value_type=%s result_type=%s",
                context,
                type(value).__name__,
                type(isna_result).__name__,
            )
            return None

        logger.debug(
            "[%s] pd.isna returned unexpected result type: value_type=%s result_type=%s",
            context,
            type(value).__name__,
            type(isna_result).__name__,
        )
        return None


    @staticmethod
    def _is_missing_board_value(value: Any) -> bool:
        """Return True when a board field value should be treated as missing."""
        if value is None:
            return True
        is_missing = DataFetcherFundamentalMixin._try_scalar_isna(value, "board_value")
        if is_missing is True:
            return True
        text = str(value).strip()
        return text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}


    @staticmethod
    def _normalize_belong_boards(raw_data: Any) -> List[Dict[str, Any]]:
        """Normalize belong-board results from heterogeneous providers."""
        if DataFetcherFundamentalMixin._is_missing_board_value(raw_data):
            return []

        normalized: List[Dict[str, Any]] = []
        dedupe = set()

        if isinstance(raw_data, pd.DataFrame):
            if raw_data.empty:
                return []
            name_col = next(
                (
                    col
                    for col in raw_data.columns
                    if str(col) in {"板块名称", "板块", "所属板块", "板块名", "name", "industry"}
                ),
                None,
            )
            code_col = next(
                (
                    col
                    for col in raw_data.columns
                    if str(col) in {"板块代码", "代码", "code"}
                ),
                None,
            )
            type_col = next(
                (
                    col
                    for col in raw_data.columns
                    if str(col) in {"板块类型", "类别", "type"}
                ),
                None,
            )
            if name_col is None:
                return []
            for _, row in raw_data.iterrows():
                board_name_raw = row.get(name_col, "")
                if DataFetcherFundamentalMixin._is_missing_board_value(board_name_raw):
                    continue
                board_name = str(board_name_raw).strip()
                if board_name in dedupe:
                    continue
                dedupe.add(board_name)
                item = {"name": board_name}
                if code_col is not None:
                    board_code_raw = row.get(code_col, "")
                    if not DataFetcherFundamentalMixin._is_missing_board_value(board_code_raw):
                        item["code"] = str(board_code_raw).strip()
                if type_col is not None:
                    board_type_raw = row.get(type_col, "")
                    if not DataFetcherFundamentalMixin._is_missing_board_value(board_type_raw):
                        item["type"] = str(board_type_raw).strip()
                normalized.append(item)
            return normalized

        if isinstance(raw_data, dict):
            raw_data = [raw_data]

        if isinstance(raw_data, (list, tuple, set)):
            for item in raw_data:
                if isinstance(item, dict):
                    board_name_raw = (
                        item.get("name")
                        or item.get("board_name")
                        or item.get("板块名称")
                        or item.get("板块")
                        or item.get("所属板块")
                        or item.get("板块名")
                        or item.get("industry")
                        or item.get("行业")
                    )
                    if DataFetcherFundamentalMixin._is_missing_board_value(board_name_raw):
                        continue
                    board_name = str(board_name_raw).strip()
                    if board_name in dedupe:
                        continue
                    dedupe.add(board_name)
                    normalized_item: Dict[str, Any] = {"name": board_name}
                    code_raw = (
                        item.get("code")
                        or item.get("板块代码")
                        or item.get("代码")
                    )
                    if not DataFetcherFundamentalMixin._is_missing_board_value(code_raw):
                        normalized_item["code"] = str(code_raw).strip()
                    type_raw = (
                        item.get("type")
                        or item.get("板块类型")
                        or item.get("类别")
                    )
                    if not DataFetcherFundamentalMixin._is_missing_board_value(type_raw):
                        normalized_item["type"] = str(type_raw).strip()
                    normalized.append(normalized_item)
                    continue
                if DataFetcherFundamentalMixin._is_missing_board_value(item):
                    continue
                board_name = str(item).strip()
                if board_name in dedupe:
                    continue
                dedupe.add(board_name)
                normalized.append({"name": board_name})
            return normalized

        if not DataFetcherFundamentalMixin._is_missing_board_value(raw_data):
            board_name = str(raw_data).strip()
            return [{"name": board_name}]
        return []


    def _get_fundamental_config(self):
        from src.config import get_config
        return get_config()


    @staticmethod
    def _normalize_source_chain(
        entries: Any,
        provider: str,
        result: str,
        duration_ms: int,
    ) -> List[Dict[str, Any]]:
        """Normalize free-form source chain entries to structured dict list."""
        if entries is None:
            return [{"provider": provider, "result": result, "duration_ms": duration_ms}]

        normalized: List[Dict[str, Any]] = []
        if not isinstance(entries, (list, tuple)):
            entries = [entries]

        for item in entries:
            if isinstance(item, dict):
                normalized.append({
                    "provider": str(item.get("provider") or provider),
                    "result": str(item.get("result") or result),
                    "duration_ms": int(item.get("duration_ms", duration_ms)),
                })
                continue

            if item is None:
                continue

            provider_name = str(item)
            normalized.append({
                "provider": provider_name,
                "result": result,
                "duration_ms": duration_ms,
            })

        if not normalized:
            return [{"provider": provider, "result": result, "duration_ms": duration_ms}]

        return normalized


    @staticmethod
    def _block_status(payload: Dict[str, Any], available: bool = True) -> str:
        if not available:
            return "not_supported"
        if not payload:
            return "partial"
        return "ok"


    @staticmethod
    def _build_fundamental_block(
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        source_chain: Optional[List[Dict[str, Any]]] = None,
        errors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "coverage": {"status": status},
            "source_chain": source_chain or [],
            "errors": errors or [],
            "data": payload or {},
        }


    @staticmethod
    def _has_meaningful_payload(payload: Any) -> bool:
        if payload is None:
            return False
        if isinstance(payload, str):
            normalized = payload.strip().lower()
            return normalized not in ("", "-", "nan", "none", "null", "n/a", "na")
        if isinstance(payload, dict):
            return any(DataFetcherFundamentalMixin._has_meaningful_payload(v) for v in payload.values())
        if isinstance(payload, pd.DataFrame):
            if payload.empty:
                return False
            return any(
                DataFetcherFundamentalMixin._has_meaningful_payload(v)
                for v in payload.to_numpy().flat
            )
        if isinstance(payload, (pd.Series, pd.Index)):
            return any(DataFetcherFundamentalMixin._has_meaningful_payload(v) for v in payload.tolist())
        if isinstance(payload, np.ndarray):
            if payload.ndim == 0:
                payload = payload.item()
            else:
                return any(
                    DataFetcherFundamentalMixin._has_meaningful_payload(v)
                    for v in payload.flat
                )
        if isinstance(payload, (list, tuple, set)):
            return any(DataFetcherFundamentalMixin._has_meaningful_payload(v) for v in payload)
        if DataFetcherFundamentalMixin._try_scalar_isna(payload, "fundamental_payload") is True:
            return False
        return True


    @staticmethod
    def _dividend_contract_has_values(payload: Any) -> bool:
        """Check whether the dividend block satisfies the repo contract.

        Downstream consumers read ttm_cash_dividend_per_share /
        ttm_dividend_yield_pct and events[].cash_dividend_per_share /
        ex_dividend_date / event_date. Raw provider events that only carry
        provider-native keys (statement/ex_date/record_date) do not satisfy
        the contract, so they must be treated as missing.
        """
        if not isinstance(payload, dict):
            return DataFetcherFundamentalMixin._has_meaningful_payload(payload)
        # The repo contract consumes ttm_cash_dividend_per_share and
        # ttm_dividend_yield_pct as a pair. A block with TTM cash but no
        # yield (e.g. the extra realtime price snapshot failed or returned
        # no price) is still missing a consumed field, so it must count as
        # a gap and be supplemented instead of being treated as complete.
        if DataFetcherFundamentalMixin._has_meaningful_payload(payload.get("ttm_cash_dividend_per_share")):
            return DataFetcherFundamentalMixin._has_meaningful_payload(payload.get("ttm_dividend_yield_pct"))
        for key in ("ttm_cash_dividend_per_share", "ttm_dividend_yield_pct"):
            if DataFetcherFundamentalMixin._has_meaningful_payload(payload.get(key)):
                return True
        events = payload.get("events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                if DataFetcherFundamentalMixin._has_meaningful_payload(
                    event.get("cash_dividend_per_share")
                ) or DataFetcherFundamentalMixin._has_meaningful_payload(
                    event.get("ex_dividend_date") or event.get("event_date")
                ):
                    return True
        return False


    @staticmethod
    def _earnings_block_has_values(payload: Any) -> bool:
        """Field-level check for the earnings block.

        A truthy dict with only metadata (report_date/period/currency) is a
        shell, not usable earnings. Require a core numeric field (revenue /
        net_profit_parent / basic_eps / gross_profit) or a populated dividend
        section before treating the block as usable.
        """
        if not isinstance(payload, dict):
            return DataFetcherFundamentalMixin._has_meaningful_payload(payload)
        report = payload.get("financial_report")
        if isinstance(report, dict):
            for key in ("revenue", "net_profit_parent", "basic_eps", "gross_profit"):
                if DataFetcherFundamentalMixin._has_meaningful_payload(report.get(key)):
                    return True
        dividend = payload.get("dividend")
        if DataFetcherFundamentalMixin._has_meaningful_payload(dividend):
            return True
        # Fall back to the generic check for other earnings sub-blocks.
        for key in ("financial_reports", "indicators"):
            if key in payload and DataFetcherFundamentalMixin._has_meaningful_payload(payload.get(key)):
                return True
        return False


    @staticmethod
    def _infer_block_status(payload: Any, fallback_status: str) -> str:
        if DataFetcherFundamentalMixin._has_meaningful_payload(payload):
            return "ok"
        if fallback_status in ("failed", "partial", "not_supported"):
            return fallback_status
        return "partial"


    @staticmethod
    def _should_cache_fundamental_context(context: Any) -> bool:
        if not isinstance(context, dict):
            return False
        status = str(context.get("status", "")).strip().lower()
        if status == "ok":
            return True
        if status == "failed":
            return False
        for block in (
            "valuation",
            "growth",
            "earnings",
            "institution",
            "capital_flow",
            "dragon_tiger",
            "boards",
        ):
            payload = context.get(block, {})
            if isinstance(payload, dict) and DataFetcherFundamentalMixin._has_meaningful_payload(payload.get("data")):
                return True
        return False


    def _build_market_not_supported(self, market: str, reason: str) -> Dict[str, Any]:
        blocks = {
            "valuation": self._build_fundamental_block(
                "partial" if market == "etf" else "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "growth": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "earnings": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "institution": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "capital_flow": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "dragon_tiger": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "boards": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
        }
        return {
            "market": market,
            "status": "partial" if market == "etf" else "not_supported",
            "coverage": {
                block: blocks[block]["status"] for block in blocks
            },
            "source_chain": [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
            "errors": [reason],
            **blocks,
        }


    def _fetch_offshore_fundamental_bundle(
        self,
        stock_code: str,
        market: str,
        bundle_timeout: float,
    ) -> Tuple[Dict[str, Any], Optional[str], int, str]:
        """Fetch the fundamental bundle for offshore markets.

        For HK with a configured Futu OpenD endpoint, try the Futu fundamental
        adapter first (company profile, statements, dividends/splits, capital
        flow, boards). When Futu succeeds only partially (e.g. statements
        failed but static info worked), fetch the yfinance bundle as well and
        merge the missing blocks so existing HK growth/earnings capability is
        never silently downgraded. Returns (payload, error, duration_ms,
        provider_name).
        """

        def _use_yfinance() -> Tuple[Dict[str, Any], Optional[str], int, str]:
            payload, err, ms = self._run_with_retry(
                lambda: self._yfinance_fundamental_adapter.get_fundamental_bundle(stock_code),
                bundle_timeout,
                "fundamental_bundle_yfinance",
            )
            return payload or {}, err, ms, "fundamental_bundle_yfinance"

        def _field_gaps(payload: Dict[str, Any]) -> List[str]:
            """List core growth/earnings fields that are missing or value-less.

            A field is a gap when it has no usable value, so the yfinance
            bundle should be consulted to fill it regardless of block-level
            truthiness.
            """
            gaps: List[str] = []
            growth = payload.get("growth")
            if isinstance(growth, dict):
                for field in ("revenue_yoy", "net_profit_yoy", "gross_margin"):
                    if not self._has_meaningful_payload(growth.get(field)):
                        gaps.append(f"growth.{field}")
            earnings = payload.get("earnings")
            report = earnings.get("financial_report") if isinstance(earnings, dict) else None
            if isinstance(report, dict):
                for field in ("revenue", "net_profit_parent", "basic_eps", "gross_profit"):
                    if not self._has_meaningful_payload(report.get(field)):
                        gaps.append(f"earnings.financial_report.{field}")
            # Dividend: the repo contract consumes ttm_* fields and/or
            # events[].cash_dividend_per_share / ex_dividend_date. Raw OpenD
            # events (statement/ex_date/record_date) without normalization do
            # not satisfy it, so treat the block as a gap unless the contract
            # fields carry usable values.
            dividend = earnings.get("dividend") if isinstance(earnings, dict) else None
            if not DataFetcherFundamentalMixin._dividend_contract_has_values(dividend):
                gaps.append("earnings.dividend")
            return gaps

        def _merge_bundles(
            futu_payload: Dict[str, Any],
            yfinance_payload: Dict[str, Any],
            futu_ms: int,
            yfinance_ms: int,
        ) -> Tuple[Dict[str, Any], Optional[str], int, str]:
            """Field-level merge: keep Futu values, fill gaps from yfinance."""
            merged: Dict[str, Any] = dict(futu_payload)

            # growth: field-level fill.
            futu_growth = futu_payload.get("growth")
            yf_growth = yfinance_payload.get("growth")
            if isinstance(yf_growth, dict):
                growth = dict(futu_growth) if isinstance(futu_growth, dict) else {}
                for field, value in yf_growth.items():
                    if not self._has_meaningful_payload(growth.get(field)):
                        growth[field] = value
                if any(self._has_meaningful_payload(v) for v in growth.values()):
                    merged["growth"] = growth

            # earnings: financial_report field-level fill, dividend block-level.
            futu_earnings = futu_payload.get("earnings")
            yf_earnings = yfinance_payload.get("earnings")
            if isinstance(yf_earnings, dict):
                earnings = dict(futu_earnings) if isinstance(futu_earnings, dict) else {}
                futu_report = earnings.get("financial_report")
                yf_report = yf_earnings.get("financial_report")
                if isinstance(yf_report, dict):
                    report = dict(futu_report) if isinstance(futu_report, dict) else {}
                    for field, value in yf_report.items():
                        if not self._has_meaningful_payload(report.get(field)):
                            report[field] = value
                    if any(self._has_meaningful_payload(v) for v in report.values()):
                        earnings["financial_report"] = report
                if not DataFetcherFundamentalMixin._dividend_contract_has_values(
                    earnings.get("dividend")
                ) and DataFetcherFundamentalMixin._dividend_contract_has_values(
                    yf_earnings.get("dividend")
                ):
                    earnings["dividend"] = yf_earnings.get("dividend")
                if any(
                    DataFetcherFundamentalMixin._earnings_block_has_values(earnings.get(key))
                    or self._has_meaningful_payload(earnings.get(key))
                    for key in ("financial_report", "financial_reports", "dividend", "indicators")
                ):
                    merged["earnings"] = earnings

            # Other blocks stay block-level (Futu wins, yfinance fills absent blocks).
            for key in ("institution", "capital_flow", "belong_boards"):
                if not self._has_meaningful_payload(merged.get(key)) and self._has_meaningful_payload(
                    yfinance_payload.get(key)
                ):
                    merged[key] = yfinance_payload.get(key)

            merged["source_chain"] = list(
                futu_payload.get("source_chain", [])
            ) + list(yfinance_payload.get("source_chain", []))
            merged["errors"] = list(futu_payload.get("errors", [])) + list(
                yfinance_payload.get("errors", [])
            )
            has_content = any(
                merged.get(key)
                for key in ("growth", "earnings", "institution", "capital_flow", "belong_boards")
            )
            merged["status"] = "partial" if has_content else "not_supported"
            return merged, None, futu_ms + yfinance_ms, "fundamental_bundle_futu"

        try:
            from data_provider.futu_fetcher import FutuFetcher
            from data_provider.futu_fundamental_adapter import FutuFundamentalAdapter
        except Exception as exc:  # noqa: BLE001 - fail open to yfinance
            logger.warning("[futu-fundamental] import failed, using yfinance: %s", exc)
            return _use_yfinance()

        if market != "hk" or not FutuFetcher.has_configured_endpoint():
            return _use_yfinance()

        futu_fetcher = getattr(self, "_futu_fundamental_fetcher", None)
        if futu_fetcher is None:
            try:
                futu_fetcher = FutuFetcher()
                self._futu_fundamental_fetcher = futu_fetcher
            except Exception as exc:  # noqa: BLE001 - fail open to yfinance
                logger.warning("[futu-fundamental] fetcher init failed, using yfinance: %s", exc)
                return _use_yfinance()

        adapter = FutuFundamentalAdapter(futu_fetcher)
        futu_payload, futu_err, futu_ms = self._run_with_retry(
            lambda: adapter.get_fundamental_bundle(stock_code),
            bundle_timeout,
            "fundamental_bundle_futu",
        )
        if futu_err is None and isinstance(futu_payload, dict):
            has_content = any(
                futu_payload.get(key)
                for key in ("growth", "earnings", "institution", "capital_flow", "belong_boards")
            )
            if has_content:
                # Futu partial success: keep the blocks it returned but do not
                # silently drop any core growth/earnings field that yfinance
                # could still provide. Decide by field gaps (not block-level
                # truthiness) so partial Futu results are supplemented.
                gaps = _field_gaps(futu_payload)
                remaining_timeout = max(bundle_timeout - futu_ms / 1000.0, 0.0)
                if gaps and remaining_timeout > 0:
                    yfinance_payload, yfinance_err, yfinance_ms = self._run_with_retry(
                        lambda: self._yfinance_fundamental_adapter.get_fundamental_bundle(stock_code),
                        remaining_timeout,
                        "fundamental_bundle_yfinance",
                    )
                    if yfinance_err is None and isinstance(yfinance_payload, dict):
                        return _merge_bundles(futu_payload, yfinance_payload, futu_ms, yfinance_ms)
                    logger.warning(
                        "[futu-fundamental] %s yfinance supplement failed (%s); keeping partial Futu bundle",
                        stock_code,
                        yfinance_err,
                    )
                return futu_payload, None, futu_ms, "fundamental_bundle_futu"
            logger.info(
                "[futu-fundamental] %s bundle empty (status=%s), falling back to yfinance",
                stock_code,
                futu_payload.get("status"),
            )
        return _use_yfinance()


    def _build_offshore_fundamental_context(
        self,
        stock_code: str,
        market: str,
        budget_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """HK/US fundamental aggregation via yfinance.

        Mirrors :meth:`get_fundamental_context` but skips A-share-specific
        blocks (capital_flow, dragon_tiger, sector rankings). belong_boards is
        sourced from yfinance ``info.sector`` / ``info.industry``.

        Cache, retry and fail-open semantics intentionally match the CN path so
        upstream callers see the same shape regardless of market.
        """
        from src.config import get_config

        config = get_config()
        stage_timeout = float(
            budget_seconds if budget_seconds is not None else config.fundamental_stage_timeout_seconds
        )
        stage_timeout = max(0.0, stage_timeout)
        fetch_timeout = float(config.fundamental_fetch_timeout_seconds)
        fetch_timeout = max(0.0, fetch_timeout)

        cache_ttl = int(config.fundamental_cache_ttl_seconds)
        cache_max_entries = max(0, int(getattr(config, "fundamental_cache_max_entries", 256)))
        cache_key = self._get_fundamental_cache_key(stock_code, stage_timeout)
        if cache_ttl > 0:
            self._prune_fundamental_cache(cache_ttl, cache_max_entries)
            with self._fundamental_cache_lock:
                cache_item = self._fundamental_cache.get(cache_key)
                if cache_item:
                    age = time.time() - float(cache_item.get("ts", 0))
                    if age <= cache_ttl:
                        return cache_item.get("context", {})

        result_ctx: Dict[str, Any] = {
            "market": market,
            "provider": "yfinance",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "data_quality": "unavailable",
            "missing_fields": [],
            "valuation": {},
            "growth": {},
            "earnings": {},
            "institution": {},
            "capital_flow": {},
            "dragon_tiger": {},
            "boards": {},
            "belong_boards": [],
            "coverage": {},
            "source_chain": [],
            "errors": [],
        }
        start_ts = time.time()

        # Valuation: reuse realtime quote payload — yfinance returns pe/pb in the
        # same shape as AkShare, so the existing block formatter still works.
        valuation_timeout = min(fetch_timeout, stage_timeout) if stage_timeout > 0 else 0
        if valuation_timeout > 0:
            quote_payload, valuation_err, valuation_ms = self._run_with_retry(
                lambda: self.get_realtime_quote(stock_code),
                valuation_timeout,
                "fundamental_valuation",
            )
        else:
            quote_payload, valuation_err, valuation_ms = None, "fundamental stage timeout", 0
        valuation_payload = {
            "pe_ratio": getattr(quote_payload, "pe_ratio", None) if quote_payload else None,
            "pb_ratio": getattr(quote_payload, "pb_ratio", None) if quote_payload else None,
            "total_mv": getattr(quote_payload, "total_mv", None) if quote_payload else None,
            "circ_mv": getattr(quote_payload, "circ_mv", None) if quote_payload else None,
        }
        valuation_status = self._infer_block_status(
            valuation_payload,
            "partial" if quote_payload is not None else "not_supported",
        )
        if valuation_status == "partial" and valuation_err and not self._has_meaningful_payload(valuation_payload):
            valuation_status = "failed"
        result_ctx["valuation"] = self._build_fundamental_block(
            valuation_status,
            valuation_payload,
            self._normalize_source_chain(
                [{"provider": "realtime_quote", "result": valuation_status, "duration_ms": valuation_ms}],
                "realtime_quote",
                valuation_status,
                valuation_ms,
            ),
            [valuation_err] if valuation_err else [],
        )

        # Fundamental bundle via Futu (HK only, when OpenD is configured), then
        # fall back to yfinance for the same payload shape.
        bundle_timeout = min(fetch_timeout, max(stage_timeout - (time.time() - start_ts), 0.0))
        if bundle_timeout <= 0:
            bundle_payload, bundle_err, bundle_ms = {}, "fundamental stage timeout", 0
            bundle_provider = "fundamental_bundle_yfinance"
        else:
            bundle_payload, bundle_err, bundle_ms, bundle_provider = (
                self._fetch_offshore_fundamental_bundle(
                    stock_code,
                    market,
                    bundle_timeout,
                )
            )
        if not isinstance(bundle_payload, dict):
            bundle_payload = {}

        bundle_chain = self._normalize_source_chain(
            bundle_payload.get("source_chain", []),
            bundle_provider,
            str(bundle_payload.get("status", "not_supported")),
            bundle_ms,
        )
        adapter_errors = list(bundle_payload.get("errors", []))
        if bundle_err:
            adapter_errors.append(bundle_err)

        growth_payload = bundle_payload.get("growth", {}) if isinstance(bundle_payload.get("growth"), dict) else {}
        earnings_payload = bundle_payload.get("earnings", {}) if isinstance(bundle_payload.get("earnings"), dict) else {}
        belong_boards = bundle_payload.get("belong_boards") if isinstance(bundle_payload.get("belong_boards"), list) else []

        growth_status = self._infer_block_status(growth_payload, str(bundle_payload.get("status", "not_supported")))
        earnings_status = self._infer_block_status(earnings_payload, str(bundle_payload.get("status", "not_supported")))

        result_ctx["growth"] = self._build_fundamental_block(
            growth_status,
            growth_payload,
            bundle_chain,
            list(adapter_errors),
        )
        result_ctx["earnings"] = self._build_fundamental_block(
            earnings_status,
            earnings_payload,
            bundle_chain,
            list(adapter_errors),
        )

        # capital_flow / dragon_tiger / boards: Futu fills capital_flow and
        # belong_boards for HK; everything else keeps not_supported (fail-open).
        futu_capital_flow = (
            bundle_payload.get("capital_flow")
            if isinstance(bundle_payload.get("capital_flow"), dict) and bundle_payload.get("capital_flow")
            else {}
        )
        if futu_capital_flow:
            result_ctx["capital_flow"] = self._build_fundamental_block(
                "ok" if futu_capital_flow.get("latest") or futu_capital_flow.get("rows") else "partial",
                futu_capital_flow,
                bundle_chain,
                [],
            )
        else:
            result_ctx["capital_flow"] = self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["not supported for this source"],
            )
        result_ctx["dragon_tiger"] = self._build_fundamental_block(
            "not_supported",
            {},
            [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
            ["not supported for offshore market"],
        )
        futu_boards = (
            bundle_payload.get("belong_boards")
            if isinstance(bundle_payload.get("belong_boards"), list) and bundle_payload.get("belong_boards")
            else []
        )
        if futu_boards:
            result_ctx["boards"] = self._build_fundamental_block(
                "ok",
                {"boards": futu_boards},
                bundle_chain,
                [],
            )
        else:
            result_ctx["boards"] = self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["not supported for offshore market"],
            )

        # institution: tw (台股) has a free official 三大法人 (institutional net buy/sell)
        # feed (TWSE T86 / TPEx OpenAPI); every other offshore market keeps not_supported.
        # tw-only + strictly additive + fail-open: any error or no-data -> not_supported,
        # which never interrupts the main analysis. Raw net figures only — no derived
        # signal / score / schema (per the v2 scope confirmed on issue #1777).
        tw_record = None
        if market == "tw":
            fetcher = getattr(self, "_tw_institutional_fetcher", None)
            if fetcher is None:
                # Wiring (import + construct) is a one-time op; a failure here is a
                # programming / deploy bug, so log it LOUD (error). Still fail-open
                # (never interrupt the main analysis — a hard requirement of #1777).
                try:
                    from data_provider.tw_institutional_fetcher import TwInstitutionalFetcher

                    fetcher = TwInstitutionalFetcher()
                    self._tw_institutional_fetcher = fetcher
                except Exception as exc:  # noqa: BLE001 - wiring failure: loud but fail-open
                    logger.error("[tw-inst] fetcher init failed (wiring bug?) code=%s: %s", stock_code, exc)
                    fetcher = None
            # fetch_timeout == 0 disables per-fetch fundamental fetches (same as valuation /
            # bundle above, which gate on fetch_timeout); honour that for institution too so
            # the FUNDAMENTAL_FETCH_TIMEOUT_SECONDS=0 config semantic is not bypassed.
            if fetcher is not None and fetch_timeout > 0:
                # The tw institution block is a WHOLE-MARKET download (~4-5s), far slower
                # than the per-symbol quote/bundle fetches, and it is the LAST offshore
                # block. When enabled, give it the full REMAINING stage budget rather than
                # the ~3s per-fetch cap that starves it and makes the first/only stock of a
                # run coin-flip between ok and not_supported. Bounded by the stage deadline
                # via _run_with_retry, so it fails open (never blocks).
                inst_timeout = max(stage_timeout - (time.time() - start_ts), 0.0)
                if inst_timeout > 0:
                    tw_record, inst_err, _inst_ms = self._run_with_retry(
                        lambda: fetcher.get_institutional_net(stock_code),
                        inst_timeout,
                        "fundamental_tw_institution",
                    )
                    if inst_err:
                        logger.warning("[tw-inst] fetch failed/timeout code=%s: %s", stock_code, inst_err)
                else:
                    tw_record = None
        # status 'ok' only when the record carries all core net figures (a genuine 0 is
        # kept — 0 is not None); None / missing core field / fetch failure -> not_supported.
        _tw_core = ("foreign_net", "trust_net", "dealer_net", "total_net")
        if tw_record is not None and all(tw_record.get(key) is not None for key in _tw_core):
            institution_status = "ok"
            result_ctx["institution"] = self._build_fundamental_block(
                "ok",
                {
                    "foreign_net": tw_record.get("foreign_net"),
                    "trust_net": tw_record.get("trust_net"),
                    "dealer_net": tw_record.get("dealer_net"),
                    "total_net": tw_record.get("total_net"),
                    "unit": tw_record.get("unit"),
                    "date": tw_record.get("date"),
                    "source": tw_record.get("source"),
                },
                [{"provider": tw_record.get("source", "tw-institutional"), "result": "ok", "duration_ms": 0}],
                [],
            )
        else:
            institution_status = "not_supported"
            result_ctx["institution"] = self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["not supported for offshore market"],
            )

        result_ctx["belong_boards"] = belong_boards or futu_boards

        capital_flow_status = result_ctx["capital_flow"].get("status", "not_supported")
        boards_status = result_ctx["boards"].get("status", "not_supported")
        block_statuses = {
            "valuation": result_ctx["valuation"].get("status", "not_supported"),
            "growth": growth_status,
            "earnings": earnings_status,
            "institution": institution_status,
            "capital_flow": capital_flow_status,
            "dragon_tiger": "not_supported",
            "boards": boards_status,
        }
        result_ctx["coverage"] = block_statuses
        for block in ("valuation", "growth", "earnings", "institution", "capital_flow", "dragon_tiger", "boards"):
            result_ctx["errors"].extend(result_ctx[block].get("errors", []))
            result_ctx["source_chain"].extend(result_ctx[block].get("source_chain", []))

        active_statuses = {
            "valuation": valuation_status,
            "growth": growth_status,
            "earnings": earnings_status,
            "capital_flow": capital_flow_status,
            "boards": boards_status,
        }
        # tw institution (when present) counts toward the OVERALL status so a report that
        # only has 三大法人 data still surfaces fundamentals (consumers key off the top-level
        # status). Futu capital_flow / boards count the same way when they are available.
        status_values = list(active_statuses.values())
        if institution_status == "ok":
            status_values.append("ok")
        if all(value == "not_supported" for value in status_values):
            result_ctx["status"] = "not_supported"
            result_ctx["data_quality"] = "unavailable"
        elif "failed" in status_values or "partial" in status_values:
            result_ctx["status"] = "partial"
            result_ctx["data_quality"] = "partial"
        else:
            result_ctx["status"] = "ok"
            result_ctx["data_quality"] = "ok"
        result_ctx["missing_fields"] = [
            block for block, status in active_statuses.items() if status != "ok"
        ]

        result_ctx["elapsed_ms"] = int((time.time() - start_ts) * 1000)
        if cache_ttl > 0 and self._should_cache_fundamental_context(result_ctx):
            with self._fundamental_cache_lock:
                self._fundamental_cache[cache_key] = {
                    "ts": time.time(),
                    "context": result_ctx,
                }
            self._prune_fundamental_cache(cache_ttl, cache_max_entries)
        return result_ctx


    def build_failed_fundamental_context(self, stock_code: str, reason: str) -> Dict[str, Any]:
        """Build a consistent failed-context payload for caller-side fallback."""
        market = _market_tag(stock_code)
        block_names = (
            "valuation",
            "growth",
            "earnings",
            "institution",
            "capital_flow",
            "dragon_tiger",
            "boards",
        )
        blocks = {
            block: self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": 0}],
                [reason],
            )
            for block in block_names
        }
        return {
            "market": market,
            "status": "failed",
            "coverage": {block: "failed" for block in block_names},
            "source_chain": [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": 0}],
            "errors": [reason],
            **blocks,
        }


    def build_not_supported_fundamental_context(
        self, stock_code: str, reason: str
    ) -> Dict[str, Any]:
        """Build a consistent not-supported payload without calling providers."""
        return self._build_market_not_supported(_market_tag(stock_code), reason)


    def get_fundamental_context(
        self,
        stock_code: str,
        budget_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Aggregate fundamental blocks with fail-open semantics.
        """
        from src.config import get_config

        config = get_config()
        if not config.enable_fundamental_pipeline:
            return self._build_market_not_supported(
                market=_market_tag(stock_code),
                reason="fundamental pipeline disabled",
            )

        stock_code = normalize_stock_code(stock_code)
        market = _market_tag(stock_code)
        is_etf = _is_etf_code(stock_code)
        if market in {"us", "hk", "jp", "kr", "tw"}:
            return self._build_offshore_fundamental_context(
                stock_code,
                market=market,
                budget_seconds=budget_seconds,
            )

        stage_timeout = float(
            budget_seconds if budget_seconds is not None else config.fundamental_stage_timeout_seconds
        )
        stage_timeout = max(0.0, stage_timeout)
        fetch_timeout = float(config.fundamental_fetch_timeout_seconds)
        fetch_timeout = max(0.0, fetch_timeout)

        cache_ttl = int(config.fundamental_cache_ttl_seconds)
        cache_max_entries = max(0, int(getattr(config, "fundamental_cache_max_entries", 256)))
        cache_key = self._get_fundamental_cache_key(stock_code, stage_timeout)
        if cache_ttl > 0:
            self._prune_fundamental_cache(cache_ttl, cache_max_entries)
            with self._fundamental_cache_lock:
                cache_item = self._fundamental_cache.get(cache_key)
                if cache_item:
                    age = time.time() - float(cache_item.get("ts", 0))
                    if age <= cache_ttl:
                        return cache_item.get("context", {})

        remaining_seconds = stage_timeout
        result_ctx: Dict[str, Any] = {
            "market": market,
            "valuation": {},
            "growth": {},
            "earnings": {},
            "institution": {},
            "capital_flow": {},
            "dragon_tiger": {},
            "boards": {},
            "coverage": {},
            "source_chain": [],
            "errors": [],
        }

        start_ts = time.time()

        def _consume_budget(consumed_ms: int) -> None:
            nonlocal remaining_seconds
            remaining_seconds = max(0.0, remaining_seconds - consumed_ms / 1000.0)

        valuation_timeout = min(fetch_timeout, remaining_seconds)
        if valuation_timeout > 0:
            quote_payload, valuation_err, valuation_ms = self._run_with_retry(
                lambda: self.get_realtime_quote(stock_code),
                valuation_timeout,
                "fundamental_valuation",
            )
            _consume_budget(valuation_ms)
        else:
            quote_payload, valuation_err, valuation_ms = None, "fundamental stage timeout", 0

        valuation_payload = {
            "pe_ratio": getattr(quote_payload, "pe_ratio", None) if quote_payload else None,
            "pb_ratio": getattr(quote_payload, "pb_ratio", None) if quote_payload else None,
            "total_mv": getattr(quote_payload, "total_mv", None) if quote_payload else None,
            "circ_mv": getattr(quote_payload, "circ_mv", None) if quote_payload else None,
        }
        valuation_status = self._infer_block_status(
            valuation_payload,
            "partial" if quote_payload is not None else "not_supported",
        )
        if valuation_status == "partial" and valuation_err and not self._has_meaningful_payload(valuation_payload):
            valuation_status = "failed"
        result_ctx["valuation"] = self._build_fundamental_block(
            valuation_status,
            valuation_payload,
            self._normalize_source_chain(
                [{"provider": "realtime_quote", "result": valuation_status, "duration_ms": valuation_ms}],
                "realtime_quote",
                valuation_status,
                valuation_ms,
            ),
            [valuation_err] if valuation_err else [],
        )

        # growth / earnings / institution (one AkShare call)
        if remaining_seconds <= 0:
            bundle_status = "failed"
            bundle_payload: Dict[str, Any] = {}
            bundle_errors = ["fundamental stage timeout"]
            bundle_ms = 0
        else:
            bundle_timeout = min(fetch_timeout, remaining_seconds)
            bundle_payload, bundle_err_msg, bundle_ms = self._run_with_retry(
                lambda: self._fundamental_adapter.get_fundamental_bundle(stock_code),
                bundle_timeout,
                "fundamental_bundle",
            )
            _consume_budget(bundle_ms)
            if not isinstance(bundle_payload, dict):
                bundle_status = "failed"
                bundle_payload = {}
                bundle_errors = ["fundamental_bundle failed"]
                if bundle_err_msg:
                    bundle_errors.append(bundle_err_msg)
            else:
                bundle_status = str(bundle_payload.get("status", "not_supported"))
                bundle_errors = [bundle_err_msg] if bundle_err_msg else []

        bundle_chain = self._normalize_source_chain(
            bundle_payload.get("source_chain", []),
            "fundamental_bundle",
            bundle_status,
            bundle_ms,
        ) if isinstance(bundle_payload, dict) else self._normalize_source_chain(
            None,
            "fundamental_bundle",
            bundle_status,
            bundle_ms,
        )
        growth_payload = bundle_payload.get("growth", {}) if isinstance(bundle_payload, dict) else {}
        earnings_payload = bundle_payload.get("earnings", {}) if isinstance(bundle_payload, dict) else {}
        institution_payload = bundle_payload.get("institution", {}) if isinstance(bundle_payload, dict) else {}
        if not isinstance(growth_payload, dict):
            growth_payload = {}
        else:
            growth_payload = dict(growth_payload)
        if not isinstance(earnings_payload, dict):
            earnings_payload = {}
        else:
            earnings_payload = dict(earnings_payload)
        if not isinstance(institution_payload, dict):
            institution_payload = {}
        else:
            institution_payload = dict(institution_payload)

        # Derive TTM dividend yield from already-fetched quote price; avoid extra quote calls.
        earnings_extra_errors: List[str] = []
        dividend_payload = earnings_payload.get("dividend")
        if isinstance(dividend_payload, dict):
            dividend_payload = dict(dividend_payload)
            ttm_cash_raw = dividend_payload.get("ttm_cash_dividend_per_share")
            ttm_cash = None
            if ttm_cash_raw is not None:
                try:
                    ttm_cash = float(ttm_cash_raw)
                except (TypeError, ValueError):
                    earnings_extra_errors.append("invalid_ttm_cash_dividend_per_share")
            if isinstance(quote_payload, dict):
                latest_price_raw = quote_payload.get("price")
            else:
                latest_price_raw = getattr(quote_payload, "price", None) if quote_payload else None
            latest_price = None
            if latest_price_raw is not None:
                try:
                    latest_price = float(latest_price_raw)
                except (TypeError, ValueError):
                    latest_price = None
            ttm_yield = None
            if ttm_cash is not None:
                if latest_price is not None and latest_price > 0:
                    ttm_yield = round(ttm_cash / latest_price * 100.0, 4)
                else:
                    earnings_extra_errors.append("invalid_price_for_ttm_dividend_yield")

            dividend_payload["ttm_dividend_yield_pct"] = ttm_yield
            if ttm_yield is not None:
                dividend_payload["yield_formula"] = "ttm_cash_dividend_per_share / latest_price * 100"
            earnings_payload["dividend"] = dividend_payload

        adapter_errors = list(bundle_payload.get("errors", [])) if isinstance(bundle_payload, dict) else []
        adapter_errors.extend(bundle_errors)
        growth_errors = list(adapter_errors)
        earnings_errors = list(adapter_errors)
        earnings_errors.extend(earnings_extra_errors)
        institution_errors = list(adapter_errors)

        growth_status = self._infer_block_status(growth_payload, bundle_status)
        earnings_status = self._infer_block_status(earnings_payload, bundle_status)
        institution_status = self._infer_block_status(institution_payload, bundle_status)

        result_ctx["growth"] = self._build_fundamental_block(
            growth_status,
            growth_payload,
            bundle_chain,
            growth_errors,
        )
        result_ctx["earnings"] = self._build_fundamental_block(
            earnings_status,
            earnings_payload,
            bundle_chain,
            earnings_errors,
        )
        result_ctx["institution"] = self._build_fundamental_block(
            institution_status,
            institution_payload,
            bundle_chain,
            institution_errors,
        )

        # capital flow
        if is_etf:
            result_ctx["capital_flow"] = self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["etf not fully supported"],
            )
            result_ctx["dragon_tiger"] = self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["etf not fully supported"],
            )
            result_ctx["boards"] = self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["etf not fully supported"],
            )
            result_ctx["status"] = "partial"
        else:
            capital_flow_budget = min(fetch_timeout, remaining_seconds)
            capital_flow_start = time.time()
            result_ctx["capital_flow"] = self.get_capital_flow_context(
                stock_code,
                budget_seconds=capital_flow_budget,
            )
            _consume_budget(int((time.time() - capital_flow_start) * 1000))

            dragon_tiger_budget = min(fetch_timeout, remaining_seconds)
            dragon_tiger_start = time.time()
            result_ctx["dragon_tiger"] = self.get_dragon_tiger_context(
                stock_code,
                budget_seconds=dragon_tiger_budget,
            )
            _consume_budget(int((time.time() - dragon_tiger_start) * 1000))

            result_ctx["boards"] = self.get_board_context(
                stock_code,
                budget_seconds=min(fetch_timeout, remaining_seconds),
            )

        block_statuses = {
            "valuation": result_ctx["valuation"].get("status", "not_supported"),
            "growth": result_ctx["growth"].get("status", "not_supported"),
            "earnings": result_ctx["earnings"].get("status", "not_supported"),
            "institution": result_ctx["institution"].get("status", "not_supported"),
            "capital_flow": result_ctx["capital_flow"].get("status", "not_supported"),
            "dragon_tiger": result_ctx["dragon_tiger"].get("status", "not_supported"),
            "boards": result_ctx["boards"].get("status", "not_supported"),
        }
        result_ctx["coverage"] = block_statuses
        for block in (
            "valuation",
            "growth",
            "earnings",
            "institution",
            "capital_flow",
            "dragon_tiger",
            "boards",
        ):
            result_ctx["errors"].extend(result_ctx[block].get("errors", []))
            result_ctx["source_chain"].extend(result_ctx[block].get("source_chain", []))

        if is_etf:
            # Keep ETF downgrade semantics for overall status even when valuation is available.
            result_ctx["status"] = (
                "not_supported" if all(value == "not_supported" for value in block_statuses.values()) else "partial"
            )
        elif all(value == "not_supported" for value in block_statuses.values()):
            result_ctx["status"] = "not_supported"
        elif "failed" in block_statuses.values() or "partial" in block_statuses.values():
            result_ctx["status"] = "partial"
        else:
            result_ctx["status"] = "ok"

        result_ctx["elapsed_ms"] = int((time.time() - start_ts) * 1000)
        if cache_ttl > 0 and self._should_cache_fundamental_context(result_ctx):
            with self._fundamental_cache_lock:
                self._fundamental_cache[cache_key] = {
                    "ts": time.time(),
                    "context": result_ctx,
                }
            self._prune_fundamental_cache(cache_ttl, cache_max_entries)
        return result_ctx


    def get_capital_flow_context(self, stock_code: str, budget_seconds: Optional[float] = None) -> Dict[str, Any]:
        """资金流向块（fail-open）。"""
        from src.config import get_config

        config = get_config()
        stock_code = normalize_stock_code(stock_code)
        timeout = float(budget_seconds if budget_seconds is not None else config.fundamental_fetch_timeout_seconds)
        if _market_tag(stock_code) != "cn" or _is_etf_code(stock_code):
            return self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["not supported"],
            )

        if timeout <= 0:
            return self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": 0}],
                ["fundamental stage timeout"],
            )
        payload, err, cost_ms = self._run_with_retry(
            lambda: self._fundamental_adapter.get_capital_flow(stock_code),
            timeout,
            "capital_flow",
        )
        if not isinstance(payload, dict):
            return self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": cost_ms}],
                [err or "capital_flow failed"],
            )

        stock_flow = payload.get("stock_flow") or {}
        sector_rankings = payload.get("sector_rankings") or {}
        has_stock_flow = False
        if isinstance(stock_flow, dict):
            has_stock_flow = any(v is not None for v in stock_flow.values())
        has_sector_rankings = bool(sector_rankings.get("top")) or bool(sector_rankings.get("bottom"))
        adapter_status = str(payload.get("status", "not_supported"))
        if has_stock_flow or has_sector_rankings:
            capital_flow_status = "ok"
        elif adapter_status == "not_supported":
            capital_flow_status = "not_supported"
        else:
            capital_flow_status = "partial"

        return self._build_fundamental_block(
            capital_flow_status,
            {
                "stock_flow": payload.get("stock_flow", {}),
                "sector_rankings": payload.get("sector_rankings", {}),
            },
            self._normalize_source_chain(
                payload.get("source_chain", []),
                "capital_flow",
                capital_flow_status,
                cost_ms,
            ),
            list(payload.get("errors", [])) + ([err] if err else []),
        )


    def get_dragon_tiger_context(self, stock_code: str, budget_seconds: Optional[float] = None) -> Dict[str, Any]:
        """龙虎榜块（fail-open）。"""
        from src.config import get_config

        config = get_config()
        stock_code = normalize_stock_code(stock_code)
        timeout = float(budget_seconds if budget_seconds is not None else config.fundamental_fetch_timeout_seconds)
        if _market_tag(stock_code) != "cn" or _is_etf_code(stock_code):
            return self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["not supported"],
            )

        if timeout <= 0:
            return self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": 0}],
                ["fundamental stage timeout"],
            )
        payload, err, cost_ms = self._run_with_retry(
            lambda: self._fundamental_adapter.get_dragon_tiger_flag(stock_code),
            timeout,
            "dragon_tiger",
        )
        if not isinstance(payload, dict):
            return self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": cost_ms}],
                [err or "dragon_tiger failed"],
            )
        return self._build_fundamental_block(
            (payload.get("status") if isinstance(payload.get("status"), str) else "partial"),
            {
                "is_on_list": bool(payload.get("is_on_list", False)),
                "recent_count": int(payload.get("recent_count", 0)),
                "latest_date": payload.get("latest_date"),
            },
            self._normalize_source_chain(
                payload.get("source_chain", []),
                "dragon_tiger",
                str(payload.get("status", "ok")),
                cost_ms,
            ),
            list(payload.get("errors", [])) + ([err] if err else []),
        )


    def get_board_context(self, stock_code: str, budget_seconds: Optional[float] = None) -> Dict[str, Any]:
        """板块榜单块（fail-open）。"""
        from src.config import get_config

        config = get_config()
        stock_code = normalize_stock_code(stock_code)
        timeout = float(budget_seconds if budget_seconds is not None else config.fundamental_fetch_timeout_seconds)
        if _market_tag(stock_code) != "cn" or _is_etf_code(stock_code):
            return self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["not supported"],
            )

        if timeout <= 0:
            return self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": 0}],
                ["fundamental stage timeout"],
            )

        def task() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
            return self._get_sector_rankings_with_meta(5)

        rankings, err, cost_ms = self._run_with_retry(task, timeout, "boards")
        if isinstance(rankings, tuple) and len(rankings) == 4:
            top, bottom, chain, chain_error = rankings
            if chain_error and not err:
                err = chain_error
            if not top and not bottom:
                return self._build_fundamental_block(
                    "failed",
                    {},
                    chain if chain else [{"provider": "sector_rankings", "result": "failed", "duration_ms": cost_ms}],
                    [err or "boards empty from all sources"],
                )
            board_status = "ok" if top and bottom else "partial"
            return self._build_fundamental_block(
                board_status,
                {"top": top or [], "bottom": bottom or []},
                chain if chain else self._normalize_source_chain(
                    ["sector_rankings"],
                    "boards",
                    board_status,
                    cost_ms,
                ),
                [err] if err else [],
            )

        return self._build_fundamental_block(
            "failed",
            {},
            [{"provider": "sector_rankings", "result": "failed", "duration_ms": cost_ms}],
            [err or "boards failed"],
        )
