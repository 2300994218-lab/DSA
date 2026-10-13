# -*- coding: utf-8 -*-
"""实时行情获取、预取、多源补充与美股利序优先级。

本模块从 ``data_provider.base`` 拆出（v1.1.1 重构），
``DataFetcherManager`` 通过多继承聚合，方法经 MRO 解析，
行为与拆分前完全一致。
"""

from __future__ import annotations

from typing import Callable, Optional, List, Tuple, Dict, Any
from src.services.stock_list_parser import AnalysisTarget, ParseStatus, parse_analysis_target
from datetime import datetime, timezone
from src.services.run_diagnostics import record_provider_run, record_provider_run_started
import time
import logging
logger = logging.getLogger("data_provider.base")
from data_provider.base_shared import (
    summarize_exception,
    normalize_stock_code,
    _is_hk_market,
    _is_jp_market,
    _is_kr_market,
    _is_tw_market,
)


class DataFetcherRealtimeMixin:
    """Mixin：实时行情获取、预取、多源补充与美股利序优先级。"""

    def prefetch_realtime_quotes(self, stock_codes: List[str]) -> int:
        """
        批量预取实时行情数据（在分析开始前调用）
        
        策略：
        1. 检查优先级中是否包含适合预取的数据源（efinance/akshare_em/tushare/tickflow）
        2. 如果不包含，跳过预取（新浪/腾讯是单股票查询，无需预取）
        3. 如果自选股数量 >= 5 且使用可预取数据源，则预取填充缓存
        
        这样做的好处：
        - 使用新浪/腾讯时：每只股票独立查询，无全量拉取问题
        - 使用 efinance/东财/Tushare 时：预取一次，后续缓存命中
        - 使用 TickFlow 时：按当前自选股批量预取，避免逐股重复请求
        
        Args:
            stock_codes: 待分析的股票代码列表
            
        Returns:
            预取的股票数量（0 表示跳过预取）
        """
        # Normalize all codes, preserving explicit index identities
        # (``sh000016`` / ``csi930955``) so the prefetch never degrades an
        # index into the colliding stock bucket (Story 1.5).
        normalized_codes: List[str] = []
        for code in stock_codes:
            target = parse_analysis_target(code)
            if target.asset_type == ParseStatus.INDEX:
                normalized_codes.append(target.canonical_id)
            else:
                normalized_codes.append(normalize_stock_code(code))
        stock_codes = normalized_codes

        from src.config import get_config

        config = get_config()

        # Issue #455: PREFETCH_REALTIME_QUOTES=false 可禁用预取，避免全市场拉取
        if not getattr(config, "prefetch_realtime_quotes", True):
            logger.debug("[预取] component=realtime_prefetch action=skip reason=disabled")
            return 0

        # 如果实时行情被禁用，跳过预取
        if not config.enable_realtime_quote:
            logger.debug("[预取] component=realtime_prefetch action=skip reason=realtime_quote_disabled")
            return 0
        
        # 检查优先级中是否包含适合批量预取的数据源
        # efinance/akshare_em/tushare 通过一次调用填充全市场缓存；
        # tickflow 通过 symbols 批量接口预取当前自选股缓存。
        priority = config.realtime_source_priority.lower()
        prefetch_sources = ['efinance', 'akshare_em', 'tushare', 'tickflow']
        
        # 如果优先级中前两个都不是可预取数据源，跳过预取
        # 因为新浪/腾讯是单股票查询，不需要预取
        priority_list = [s.strip() for s in priority.split(',')]
        first_prefetch_source_index = None
        for i, source in enumerate(priority_list):
            if source in prefetch_sources:
                first_prefetch_source_index = i
                break
        
        # 如果没有可预取数据源，或者它排在第 3 位之后，跳过预取
        if first_prefetch_source_index is None or first_prefetch_source_index >= 2:
            logger.info(
                "[预取] component=realtime_prefetch action=skip reason=no_early_prefetch_source priority=%s",
                priority,
            )
            return 0
        
        # 如果股票数量少于 5 个，不进行批量预取（逐个查询更高效）
        if len(stock_codes) < 5:
            logger.info(
                "[预取] component=realtime_prefetch action=skip reason=small_batch "
                "stock_count=%d threshold=5 prefetch_source=%s",
                len(stock_codes),
                priority_list[first_prefetch_source_index],
            )
            return 0
        
        prefetch_source = priority_list[first_prefetch_source_index]
        logger.info(
            "[预取] component=realtime_prefetch action=start stock_count=%d prefetch_source=%s first_code=%s",
            len(stock_codes),
            prefetch_source,
            stock_codes[0],
        )
        
        # TickFlow 使用 symbols 批量接口；其他可预取源通过首次查询触发自身缓存。
        if prefetch_source == "tickflow":
            fetcher = self._get_fetcher_by_name("TickFlowFetcher", capability="realtime_quote")
            if fetcher is None or not hasattr(fetcher, "prefetch_realtime_quotes"):
                logger.info(
                    "[prefetch] component=realtime_prefetch action=skip reason=tickflow_unavailable"
                )
                return 0
            try:
                return int(
                    self._call_fetcher_method(
                        fetcher,
                        "prefetch_realtime_quotes",
                        stock_codes,
                        batch_size=getattr(config, "tickflow_batch_size", 100),
                    )
                    or 0
                )
            except Exception as exc:
                logger.warning("[TickFlowFetcher] realtime prefetch failed: %s", exc)
                return 0

        try:
            # 用第一只股票触发全量拉取
            first_code = stock_codes[0]
            quote = self.get_realtime_quote(first_code)
            
            if quote:
                logger.info(
                    "[预取] component=realtime_prefetch action=complete status=success "
                    "stock_count=%d prefetch_source=%s",
                    len(stock_codes),
                    prefetch_source,
                )
                return len(stock_codes)
            else:
                logger.warning(
                    "[预取] component=realtime_prefetch action=complete status=failed "
                    "stock_count=%d prefetch_source=%s fallback=per_stock",
                    len(stock_codes),
                    prefetch_source,
                )
                return 0
                
        except Exception as e:
            logger.error(
                "[预取] component=realtime_prefetch action=complete status=error "
                "stock_count=%d prefetch_source=%s error=%s",
                len(stock_codes),
                prefetch_source,
                e,
            )
            return 0


    def prefetch_daily_klines(self, stock_codes: List[str], days: int = 30) -> int:
        """Batch-prefetch TickFlow daily K-lines without changing per-stock callers."""
        fetcher = self._get_fetcher_by_name("TickFlowFetcher", capability="daily_data")
        if fetcher is None or not hasattr(fetcher, "prefetch_daily_klines"):
            return 0

        try:
            return int(
                self._call_fetcher_method(
                    fetcher,
                    "prefetch_daily_klines",
                    stock_codes,
                    days=days,
                )
                or 0
            )
        except Exception as exc:
            logger.warning("[TickFlowFetcher] daily K-line prefetch failed: %s", exc)
            return 0


    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


    @staticmethod
    def _parse_realtime_timestamp(value: Any) -> Optional[datetime]:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            text = str(value).strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)


    @staticmethod
    def _realtime_fetcher_token(fetcher_name: str, **kw) -> str:
        if fetcher_name == "AkshareFetcher" and kw.get("source") == "hk":
            return "akshare_hk"
        mapping = {
            "LongbridgeFetcher": "longbridge",
            "FutuFetcher": "futu",
            "YfinanceFetcher": "yfinance",
            "AkshareFetcher": "akshare",
            "FinnhubFetcher": "finnhub",
            "AlphaVantageFetcher": "alphavantage",
            "EfinanceFetcher": "efinance",
            "TushareFetcher": "tushare",
        }
        return mapping.get(fetcher_name, fetcher_name.replace("Fetcher", "").lower())


    def _enrich_realtime_quote(
        self,
        quote,
        *,
        fallback_from: Optional[str] = None,
        realtime_cache_ttl: Optional[int] = None,
    ):
        """Attach runtime metadata without inventing provider-side timestamps."""
        if quote is None:
            return None

        fetched_at = self._utc_now_iso()
        setattr(quote, "fetched_at", fetched_at)
        if fallback_from:
            setattr(quote, "fallback_from", str(fallback_from))

        provider_dt = self._parse_realtime_timestamp(
            getattr(quote, "provider_timestamp", None)
        )
        if provider_dt is None:
            setattr(quote, "provider_timestamp", None)
            setattr(quote, "stale_seconds", None)
            setattr(quote, "is_stale", None)
            return quote

        setattr(quote, "provider_timestamp", provider_dt.isoformat())
        fetched_dt = self._parse_realtime_timestamp(fetched_at) or datetime.now(timezone.utc)
        stale_seconds = max(0, int((fetched_dt - provider_dt).total_seconds()))
        ttl = realtime_cache_ttl if realtime_cache_ttl is not None else 600
        setattr(quote, "stale_seconds", stale_seconds)
        setattr(quote, "is_stale", stale_seconds > int(ttl))
        return quote


    def get_realtime_quote(self, stock_code: str, *, log_final_failure: bool = True):
        """
        获取实时行情数据（自动故障切换）
        
        故障切换策略（按配置的优先级）：
        1. 美股：使用 YfinanceFetcher.get_realtime_quote()
        2. EfinanceFetcher.get_realtime_quote()
        3. AkshareFetcher.get_realtime_quote(source="em")  - 东财
        4. AkshareFetcher.get_realtime_quote(source="sina") - 新浪
        5. AkshareFetcher.get_realtime_quote(source="tencent") - 腾讯
        6. 返回 None（降级兜底）
        
        Args:
            stock_code: 股票代码
            log_final_failure: Whether to emit the final "all sources failed"
                summary log when no realtime quote is available.
            
        Returns:
            UnifiedRealtimeQuote 对象，所有数据源都失败则返回 None
        """
        raw_stock_code = (stock_code or "").strip()
        # Normalize code (strip SH/SZ prefix etc.)
        stock_code = normalize_stock_code(stock_code)

        from .akshare_fetcher import _is_us_code
        from .us_index_mapping import is_us_index_code
        from src.config import get_config

        config = get_config()

        # 如果实时行情功能被禁用，直接返回 None
        if not config.enable_realtime_quote:
            logger.debug(f"[实时行情] 功能已禁用，跳过 {stock_code}")
            return None

        # ----------------------------------------------------------
        # 已登记 A 股指数 — 固定实时行情链（Story 1.5）
        #   显式指数身份在 normalize 前解析，避免 sh000016 被剥成
        #   股票 000016 后误取同名股票行情。
        # ----------------------------------------------------------
        index_target = parse_analysis_target(raw_stock_code)
        if index_target.asset_type == ParseStatus.INDEX:
            return self._get_cn_index_realtime_quote(
                index_target, log_final_failure=log_final_failure
            )

        # ----------------------------------------------------------
        # 美股 (指数 + 个股) / 港股 — 专用双源路由
        #   配置长桥后: Longbridge 首选, YFinance/AkShare 补充
        #   未配置长桥: YFinance/AkShare 首选, Longbridge 补充
        #   美股指数:   始终 YFinance 首选（Longbridge 不提供指数行情）
        # ----------------------------------------------------------
        is_us_index = is_us_index_code(stock_code)
        is_us = is_us_index or _is_us_code(stock_code)
        is_hk = (not is_us) and _is_hk_market(stock_code)
        is_jp = (not is_us) and (not is_hk) and _is_jp_market(stock_code)
        is_kr = (not is_us) and (not is_hk) and _is_kr_market(stock_code)
        is_tw = (not is_us) and (not is_hk) and _is_tw_market(stock_code)

        if is_jp or is_kr or is_tw:
            market_label = "日股" if is_jp else "韩股" if is_kr else "台股"
            quote = self._try_fetcher_quote(stock_code, "YfinanceFetcher")
            if quote is not None:
                logger.info(f"[实时行情] {market_label} {stock_code} 成功获取 (来源: YfinanceFetcher)")
                return self._enrich_realtime_quote(
                    quote,
                    realtime_cache_ttl=getattr(config, "realtime_cache_ttl", None),
                )
            if log_final_failure:
                logger.info(f"[实时行情] {market_label} {stock_code} 无可用数据源")
            return None

        if is_us or is_hk:
            prefer_lb = self._longbridge_preferred() and not is_us_index
            if is_us:
                primary_src = "LongbridgeFetcher" if prefer_lb else "YfinanceFetcher"
                secondary_src = "YfinanceFetcher" if prefer_lb else "LongbridgeFetcher"
                market_label = "美股指数" if is_us_index else "美股"
                primary_kw: dict = {}
                secondary_kw: dict = {}
            else:
                hk_priority = [
                    source.strip().lower()
                    for source in getattr(
                        config,
                        "futu_hk_realtime_source_priority",
                        "futu,longbridge,akshare,yfinance",
                    ).split(",")
                    if source.strip()
                ]
                source_map = {
                    "futu": ("FutuFetcher", {}),
                    "longbridge": ("LongbridgeFetcher", {}),
                    "akshare": ("AkshareFetcher", {"source": "hk"}),
                    "yfinance": ("YfinanceFetcher", {}),
                }
                primary_quote = None
                primary_token = None
                primary_src_index = -1
                fallback_from = None
                # Futu only participates when an OpenD endpoint is configured.
                # Skipping an unconfigured source here (instead of letting
                # _try_fetcher_quote fail on it) avoids recording a never-enabled
                # source as the failed primary, which would wrongly mark a
                # successful quote from the next enabled source as fallback.
                futu_enabled = False
                try:
                    from data_provider.futu_fetcher import FutuFetcher
                    futu_enabled = FutuFetcher.has_configured_endpoint()
                except Exception:  # noqa: BLE001 - fail closed: treat futu as disabled
                    futu_enabled = False
                for index, source in enumerate(hk_priority):
                    mapped = source_map.get(source)
                    if mapped is None:
                        logger.warning("[实时行情] 忽略未知港股数据源: %s", source)
                        continue
                    fetcher_name, fetcher_kw = mapped
                    if fetcher_name == "FutuFetcher" and not futu_enabled:
                        logger.info(
                            "[实时行情] 港股 %s 未配置 FUTU_OPEND_HOST，跳过 futu 源", stock_code
                        )
                        continue
                    quote = self._try_fetcher_quote(stock_code, fetcher_name, **fetcher_kw)
                    if quote is not None:
                        primary_quote = quote
                        primary_token = self._realtime_fetcher_token(fetcher_name, **fetcher_kw)
                        primary_src_index = index
                        logger.info("[实时行情] 港股 %s 成功获取 (来源: %s)", stock_code, fetcher_name)
                        break
                    # 该源失败：记住它的 token，供后续成功源作为 fallback_from 使用。
                    if fallback_from is None:
                        fallback_from = self._realtime_fetcher_token(fetcher_name, **fetcher_kw)
                if primary_quote is not None:
                    # 用后续数据源补充缺失字段（volume_ratio / turnover_rate / 估值 / 市值），
                    # 保持与美股路径一致的 _supplement_quote 补字段能力。
                    for source in hk_priority[primary_src_index + 1:]:
                        mapped = source_map.get(source)
                        if mapped is None:
                            continue
                        if not self._quote_needs_supplement(primary_quote):
                            break
                        fetcher_name, fetcher_kw = mapped
                        self._supplement_quote(stock_code, primary_quote, fetcher_name, **fetcher_kw)
                    return self._enrich_realtime_quote(
                        primary_quote,
                        fallback_from=fallback_from,
                        realtime_cache_ttl=getattr(config, "realtime_cache_ttl", None),
                    )
                if log_final_failure:
                    logger.info("[实时行情] 港股 %s 无可用数据源", stock_code)
                return None
            primary_token = self._realtime_fetcher_token(primary_src, **primary_kw)
            primary_quote = self._try_fetcher_quote(stock_code, primary_src, **primary_kw)
            fallback_from = primary_token if primary_quote is None else None
            if primary_quote is not None:
                logger.info(f"[实时行情] {market_label} {stock_code} 成功获取 (来源: {primary_src})")
            # US index quotes are YFinance-only. Longbridge accepts index-like
            # symbols syntactically, but does not provide the index quote
            # contract used by this manager, so it must not be used as either
            # fallback or field supplement here.
            if not is_us_index:
                primary_quote = self._supplement_quote(
                    stock_code, primary_quote, secondary_src, **secondary_kw,
                )
            if is_us and not is_us_index and primary_quote is not None:
                for extra_src in ["FinnhubFetcher", "AlphaVantageFetcher"]:
                    primary_quote = self._supplement_quote(
                        stock_code, primary_quote, extra_src,
                    )
            if primary_quote is not None:
                return self._enrich_realtime_quote(
                    primary_quote,
                    fallback_from=fallback_from,
                    realtime_cache_ttl=getattr(config, "realtime_cache_ttl", None),
                )
            if log_final_failure:
                logger.info(f"[实时行情] {market_label} {stock_code} 无可用数据源")
            return None
        
        # 获取配置的数据源优先级
        source_priority = [
            source.strip().lower()
            for source in config.realtime_source_priority.split(',')
            if source.strip()
        ]
        
        errors = []
        failed_sources: List[str] = []
        # primary_quote holds the first successful result; we may supplement
        # missing fields (volume_ratio, turnover_rate, etc.) from later sources.
        primary_quote = None
        primary_fallback_from: Optional[str] = None
        
        for source_index, source in enumerate(source_priority):
            attempt_start = time.time()
            fallback_to = source_priority[source_index + 1] if source_index + 1 < len(source_priority) else None
            fetcher = None
            try:
                quote = None
                
                if source == "efinance":
                    fetcher = self._get_fetcher_by_name("EfinanceFetcher", capability="realtime_quote")
                    if fetcher is not None and hasattr(fetcher, 'get_realtime_quote'):
                        record_provider_run_started(
                            data_type="realtime_quote",
                            provider=fetcher.name,
                            operation="get_realtime_quote",
                        )
                        quote = self._call_fetcher_method(fetcher, 'get_realtime_quote', stock_code)
                
                elif source == "akshare_em":
                    fetcher = self._get_fetcher_by_name("AkshareFetcher", capability="realtime_quote")
                    if fetcher is not None and hasattr(fetcher, 'get_realtime_quote'):
                        record_provider_run_started(
                            data_type="realtime_quote",
                            provider=fetcher.name,
                            operation="get_realtime_quote",
                        )
                        quote = self._call_fetcher_method(fetcher, 'get_realtime_quote', stock_code, source="em")
                
                elif source == "akshare_sina":
                    fetcher = self._get_fetcher_by_name("AkshareFetcher", capability="realtime_quote")
                    if fetcher is not None and hasattr(fetcher, 'get_realtime_quote'):
                        record_provider_run_started(
                            data_type="realtime_quote",
                            provider=fetcher.name,
                            operation="get_realtime_quote",
                        )
                        quote = self._call_fetcher_method(fetcher, 'get_realtime_quote', stock_code, source="sina")
                
                elif source in ("tencent", "akshare_qq"):
                    fetcher = self._get_fetcher_by_name("AkshareFetcher", capability="realtime_quote")
                    if fetcher is not None and hasattr(fetcher, 'get_realtime_quote'):
                        record_provider_run_started(
                            data_type="realtime_quote",
                            provider=fetcher.name,
                            operation="get_realtime_quote",
                        )
                        quote = self._call_fetcher_method(fetcher, 'get_realtime_quote', stock_code, source="tencent")
                
                elif source == "tushare":
                    fetcher = self._get_fetcher_by_name("TushareFetcher", capability="realtime_quote")
                    if fetcher is not None and hasattr(fetcher, 'get_realtime_quote'):
                        record_provider_run_started(
                            data_type="realtime_quote",
                            provider=fetcher.name,
                            operation="get_realtime_quote",
                        )
                        quote = self._call_fetcher_method(fetcher, 'get_realtime_quote', raw_stock_code or stock_code)

                elif source == "tickflow":
                    fetcher = self._get_fetcher_by_name("TickFlowFetcher", capability="realtime_quote")
                    if fetcher is not None and hasattr(fetcher, 'get_realtime_quote'):
                        record_provider_run_started(
                            data_type="realtime_quote",
                            provider=fetcher.name,
                            operation="get_realtime_quote",
                        )
                        quote = self._call_fetcher_method(fetcher, 'get_realtime_quote', raw_stock_code or stock_code)

                provider_name = fetcher.name if fetcher is not None else source
                
                if quote is not None and quote.has_basic_data():
                    record_provider_run(
                        data_type="realtime_quote",
                        provider=provider_name,
                        operation="get_realtime_quote",
                        success=True,
                        latency_ms=int((time.time() - attempt_start) * 1000),
                        fallback_to=fallback_to if primary_quote is None and self._quote_needs_supplement(quote) else None,
                        record_count=1,
                    )
                    if primary_quote is None:
                        # First successful source becomes primary
                        primary_quote = quote
                        primary_fallback_from = failed_sources[0] if failed_sources else None
                        logger.info(f"[实时行情] {stock_code} 成功获取 (来源: {source})")
                        # If all key supplementary fields are present, return early
                        if not self._quote_needs_supplement(primary_quote):
                            return self._enrich_realtime_quote(
                                primary_quote,
                                fallback_from=primary_fallback_from,
                                realtime_cache_ttl=getattr(config, "realtime_cache_ttl", None),
                            )
                        # Otherwise, continue to try later sources for missing fields
                        logger.debug(f"[实时行情] {stock_code} 部分字段缺失，尝试从后续数据源补充")
                        supplement_attempts = 0
                    else:
                        # Supplement missing fields from this source (limit attempts)
                        supplement_attempts += 1
                        if supplement_attempts > 1:
                            logger.debug(f"[实时行情] {stock_code} 补充尝试已达上限，停止继续")
                            break
                        merged = self._merge_quote_fields(primary_quote, quote)
                        if merged:
                            logger.info(f"[实时行情] {stock_code} 从 {source} 补充了缺失字段: {merged}")
                        # Stop supplementing once all key fields are filled
                        if not self._quote_needs_supplement(primary_quote):
                            break
                else:
                    record_provider_run(
                        data_type="realtime_quote",
                        provider=provider_name,
                        operation="get_realtime_quote",
                        success=False,
                        latency_ms=int((time.time() - attempt_start) * 1000),
                        error_type="empty",
                        error_message="empty or incomplete quote",
                        fallback_to=fallback_to,
                        record_count=0,
                    )
                    if primary_quote is None:
                        failed_sources.append(source)
                    
            except Exception as e:
                error_msg = f"[{source}] 失败: {str(e)}"
                error_type, error_reason = summarize_exception(e)
                record_provider_run(
                    data_type="realtime_quote",
                    provider=getattr(fetcher, "name", source),
                    operation="get_realtime_quote",
                    success=False,
                    latency_ms=int((time.time() - attempt_start) * 1000),
                    error_type=error_type,
                    error_message=error_reason,
                    fallback_to=fallback_to,
                )
                logger.info(f"[实时行情] {stock_code} {error_msg}，继续尝试下一个数据源")
                errors.append(error_msg)
                if primary_quote is None:
                    failed_sources.append(source)
                continue
        
        # Return primary even if some fields are still missing
        if primary_quote is not None:
            return self._enrich_realtime_quote(
                primary_quote,
                fallback_from=primary_fallback_from,
                realtime_cache_ttl=getattr(config, "realtime_cache_ttl", None),
            )

        # 所有数据源都失败，返回 None（降级兜底）
        if log_final_failure:
            if errors:
                logger.info(f"[实时行情] {stock_code} 所有数据源均失败: {'; '.join(errors)}")
            else:
                logger.info(f"[实时行情] {stock_code} 无可用数据源")

        return None


    @classmethod
    def _quote_needs_supplement(cls, quote) -> bool:
        """Check if any key supplementary field is still None."""
        for f in cls._SUPPLEMENT_FIELDS:
            if getattr(quote, f, None) is None:
                return True
        return False


    @classmethod
    def _merge_quote_fields(cls, primary, secondary) -> list:
        """
        Copy non-None fields from *secondary* into *primary* where
        *primary* has None. Returns list of field names that were filled.
        """
        filled = []
        for f in cls._SUPPLEMENT_FIELDS:
            if getattr(primary, f, None) is None:
                val = getattr(secondary, f, None)
                if val is not None:
                    setattr(primary, f, val)
                    filled.append(f)
        return filled


    def _order_us_sources_by_priority(self, source_order: List[str], *, pin_first: bool) -> List[str]:
        """按各数据源当前优先级重排美股日线路由(消费既有 *_PRIORITY 配置)。

        稳定排序:各源默认优先级(Finnhub=2/AlphaVantage=3/Yfinance=4/Longbridge=5)
        与内置链路一致,默认行为不变;单项 *_PRIORITY 调整即时生效。
        pin_first=True 时保持链路首位(指数固定 Yfinance、Longbridge preferred),
        其余成员按优先级排序。
        """
        self._ensure_concurrency_guards()
        if not source_order:
            return source_order
        priority_by_name = {f.name: f.priority for f in self._get_fetchers_snapshot()}
        if pin_first:
            pinned, rest = source_order[0], source_order[1:]
            return [pinned] + sorted(rest, key=lambda name: priority_by_name.get(name, 10 ** 9))
        return sorted(source_order, key=lambda name: priority_by_name.get(name, 10 ** 9))


    def _longbridge_preferred(self, capability: str = "realtime_quote") -> bool:
        """Return True when Longbridge keys are configured and available.

        When True, non-A-share routing (US & HK) uses Longbridge as the
        primary data source with Yfinance/AkShare as fallback.
        """
        return self._get_fetcher_by_name(
            "LongbridgeFetcher",
            capability=capability,
        ) is not None


    def _try_fetcher_quote(self, stock_code: str, fetcher_name: str, **kw):
        """Try to get a realtime quote from a named fetcher; returns quote or None."""
        fetcher = self._get_fetcher_by_name(fetcher_name, capability="realtime_quote")
        if fetcher is None or not hasattr(fetcher, 'get_realtime_quote'):
            record_provider_run(
                data_type="realtime_quote",
                provider=fetcher_name,
                operation="get_realtime_quote",
                success=False,
                error_type="unavailable",
                error_message="fetcher unavailable",
            )
            return None
        attempt_start = time.time()
        try:
            record_provider_run_started(
                data_type="realtime_quote",
                provider=fetcher.name,
                operation="get_realtime_quote",
            )
            q = self._call_fetcher_method(fetcher, 'get_realtime_quote', stock_code, **kw)
            if q is not None and q.has_basic_data():
                record_provider_run(
                    data_type="realtime_quote",
                    provider=fetcher.name,
                    operation="get_realtime_quote",
                    success=True,
                    latency_ms=int((time.time() - attempt_start) * 1000),
                    record_count=1,
                )
                return q
            record_provider_run(
                data_type="realtime_quote",
                provider=fetcher.name,
                operation="get_realtime_quote",
                success=False,
                latency_ms=int((time.time() - attempt_start) * 1000),
                error_type="empty",
                error_message="empty or incomplete quote",
                record_count=0,
            )
        except Exception as e:
            error_type, error_reason = summarize_exception(e)
            record_provider_run(
                data_type="realtime_quote",
                provider=fetcher.name,
                operation="get_realtime_quote",
                success=False,
                latency_ms=int((time.time() - attempt_start) * 1000),
                error_type=error_type,
                error_message=error_reason,
            )
            logger.debug(f"[实时行情] {stock_code} {fetcher_name} 获取失败: {e}")
        return None


    def _supplement_quote(self, stock_code: str, primary_quote, fetcher_name: str, **kw):
        """Supplement *primary_quote* with data from *fetcher_name*.

        If *primary_quote* is None, try *fetcher_name* as the sole source.
        Returns the (potentially enriched) quote, or None.
        """
        if primary_quote is not None:
            if not self._quote_needs_supplement(primary_quote):
                return primary_quote
            try:
                secondary = self._try_fetcher_quote(stock_code, fetcher_name, **kw)
                if secondary is not None:
                    filled = self._merge_quote_fields(primary_quote, secondary)
                    if filled:
                        logger.info(f"[实时行情] {stock_code} 从 {fetcher_name} 补充了: {filled}")
            except Exception as e:
                logger.debug(f"[实时行情] {stock_code} {fetcher_name} 补充失败: {e}")
            return primary_quote

        q = self._try_fetcher_quote(stock_code, fetcher_name, **kw)
        if q is not None:
            logger.info(f"[实时行情] {stock_code} 从 {fetcher_name} 获取成功 (独立数据源)")
        return q


    def _supplement_from_longbridge(self, stock_code: str, primary_quote):
        """Shortcut kept for backward-compat with A-share general loop."""
        return self._supplement_quote(stock_code, primary_quote, "LongbridgeFetcher")
