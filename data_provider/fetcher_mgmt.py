# -*- coding: utf-8 -*-
"""fetcher 注册、并发保护、可用性探测、健康熔断与日线数据源路由。

本模块从 ``data_provider.base`` 拆出（v1.1.1 重构），
``DataFetcherManager`` 通过多继承聚合，方法经 MRO 解析，
行为与拆分前完全一致。
"""

from __future__ import annotations

from .fundamental_adapter import AkshareFundamentalAdapter
from typing import Callable, Optional, List, Tuple, Dict, Any
from threading import BoundedSemaphore, RLock, Thread
from .realtime_types import CircuitBreaker
from .yfinance_fundamental_adapter import YfinanceFundamentalAdapter
import time
import logging
logger = logging.getLogger("data_provider.base")
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_provider.base import BaseFetcher, DataFetcherManager


class DataFetcherManagementMixin:
    """Mixin：fetcher 注册、并发保护、可用性探测、健康熔断与日线数据源路由。"""

    _DAILY_MARKET_FETCHER_SUPPORT = {
        "EfinanceFetcher": {"cn"},
        "TencentFetcher": {"cn"},
        "AkshareFetcher": {"cn", "hk"},
        "TushareFetcher": {"cn", "hk"},
        "TickFlowFetcher": {"cn"},
        "PytdxFetcher": {"cn"},
        "BaostockFetcher": {"cn"},
        "YfinanceFetcher": {"cn", "hk", "us", "jp", "kr", "tw"},
        "LongbridgeFetcher": {"hk", "us"},
        "FutuFetcher": {"hk"},
        "FinnhubFetcher": {"us"},
        "AlphaVantageFetcher": {"us"},
    }


    _daily_source_health = CircuitBreaker(failure_threshold=3, cooldown_seconds=300.0)


    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        """
        初始化管理器
        
        Args:
            fetchers: 数据源列表（可选，默认按优先级自动创建）
        """
        self._fetchers: List[BaseFetcher] = []
        self._fetchers_lock = RLock()
        self._fetchers_by_name: Dict[str, BaseFetcher] = {}
        self._fetcher_call_locks: Dict[int, RLock] = {}
        self._fetcher_call_locks_lock = RLock()
        self._stock_name_cache: Dict[str, str] = {}
        self._stock_name_cache_lock = RLock()
        
        if fetchers:
            # 按优先级排序
            self._fetchers = sorted(fetchers, key=lambda f: f.priority)
            self._refresh_fetcher_indexes_locked()
        else:
            # 默认数据源将在首次使用时延迟加载
            self._init_default_fetchers()
        self._fundamental_adapter = AkshareFundamentalAdapter()
        self._yfinance_fundamental_adapter = YfinanceFundamentalAdapter()
        self._futu_fundamental_fetcher = None
        self._tickflow_fetcher = None
        self._tickflow_api_key: Optional[str] = None
        self._tickflow_lock = RLock()
        self._fundamental_cache: Dict[str, Dict[str, Any]] = {}
        self._fundamental_cache_lock = RLock()
        self._fundamental_timeout_worker_limit = 8
        self._fundamental_timeout_slots = BoundedSemaphore(self._fundamental_timeout_worker_limit)


    def _ensure_concurrency_guards(self) -> None:
        """Lazily initialize thread-safety primitives for test scaffolds using __new__."""
        if not hasattr(self, "_fetchers_lock") or self._fetchers_lock is None:
            self._fetchers_lock = RLock()
        if not hasattr(self, "_fetchers_by_name") or self._fetchers_by_name is None:
            self._fetchers_by_name = {}
        if not hasattr(self, "_fetcher_call_locks") or self._fetcher_call_locks is None:
            self._fetcher_call_locks = {}
        if not hasattr(self, "_fetcher_call_locks_lock") or self._fetcher_call_locks_lock is None:
            self._fetcher_call_locks_lock = RLock()
        if not hasattr(self, "_stock_name_cache") or self._stock_name_cache is None:
            self._stock_name_cache = {}
        if not hasattr(self, "_stock_name_cache_lock") or self._stock_name_cache_lock is None:
            self._stock_name_cache_lock = RLock()


    def _get_fetchers_snapshot(self) -> List[BaseFetcher]:
        self._ensure_concurrency_guards()
        with self._fetchers_lock:
            return list(getattr(self, "_fetchers", []))


    def _refresh_fetcher_indexes_locked(self) -> None:
        self._fetchers_by_name = {fetcher.name: fetcher for fetcher in self._fetchers}


    def _get_fetcher_by_name(self, fetcher_name: str, capability: str = "") -> Optional[BaseFetcher]:
        self._ensure_concurrency_guards()
        with self._fetchers_lock:
            fetcher = self._fetchers_by_name.get(fetcher_name)
            if fetcher is None and self._fetchers:
                self._refresh_fetcher_indexes_locked()
                fetcher = self._fetchers_by_name.get(fetcher_name)
        if fetcher is None:
            return None
        if not self._is_fetcher_available(fetcher, capability=capability):
            return None
        return fetcher


    @staticmethod
    def _call_availability_probe(fetcher: BaseFetcher, probe_name: str, capability: str) -> Optional[bool]:
        probe = getattr(fetcher, probe_name, None)
        if not callable(probe):
            return None
        try:
            if probe_name == "is_available_for_request":
                return bool(probe(capability))
            return bool(probe())
        except TypeError:
            return bool(probe())
        except Exception as exc:
            logger.debug(
                "[数据源可用性] %s.%s 检查失败(capability=%s): %s",
                fetcher.name,
                probe_name,
                capability or "default",
                exc,
            )
            return False


    @classmethod
    def _is_fetcher_available(cls, fetcher: BaseFetcher, capability: str = "") -> bool:
        for probe_name in ("is_available_for_request", "is_available", "_is_available"):
            result = cls._call_availability_probe(fetcher, probe_name, capability)
            if result is not None:
                return result
        return True


    def _get_fetcher_call_lock(self, fetcher: BaseFetcher) -> RLock:
        self._ensure_concurrency_guards()
        fetcher_id = id(fetcher)
        with self._fetcher_call_locks_lock:
            lock = self._fetcher_call_locks.get(fetcher_id)
            if lock is None:
                lock = RLock()
                self._fetcher_call_locks[fetcher_id] = lock
            return lock


    def _call_fetcher_method(self, fetcher: BaseFetcher, method_name: str, *args, **kwargs):
        """Serialize shared fetcher state access through manager-owned per-instance locks."""
        method = getattr(fetcher, method_name)
        with self._get_fetcher_call_lock(fetcher):
            return method(*args, **kwargs)


    @classmethod
    def _filter_daily_fetchers_for_market(
        cls,
        fetchers: List[BaseFetcher],
        market: str,
    ) -> List[BaseFetcher]:
        """Skip built-in daily fetchers that are known not to support a market."""

        kept: List[BaseFetcher] = []
        skipped: List[str] = []
        for fetcher in fetchers:
            supported = cls._DAILY_MARKET_FETCHER_SUPPORT.get(fetcher.name)
            if supported is not None and market not in supported:
                skipped.append(fetcher.name)
            else:
                kept.append(fetcher)

        if skipped:
            logger.info(
                "[数据源路由] %s 日线跳过不支持的数据源: %s",
                market,
                ", ".join(skipped),
            )
        return kept


    @classmethod
    def _filter_fetchers_by_capability(
        cls,
        fetchers: List[BaseFetcher],
        capability: str,
    ) -> List[BaseFetcher]:
        """Skip request-time unavailable fetchers before entering route-specific loops."""
        kept: List[BaseFetcher] = []
        skipped: List[str] = []

        for fetcher in fetchers:
            if cls._is_fetcher_available(fetcher, capability=capability):
                kept.append(fetcher)
            else:
                skipped.append(fetcher.name)

        if skipped:
            logger.info(
                "[数据源路由] %s 跳过暂不可用的数据源: %s",
                capability or "request",
                ", ".join(skipped),
            )

        return kept


    @classmethod
    def _daily_health_key(cls, fetcher: BaseFetcher, market: str) -> str:
        return f"daily_data:{market}:{fetcher.name}"


    @classmethod
    def _is_daily_source_available(
        cls,
        fetcher: BaseFetcher,
        market: str,
    ) -> bool:
        key = cls._daily_health_key(fetcher, market)
        if cls._daily_source_health.is_available(key):
            return True
        logger.info(
            "[数据源健康度] %s 日线跳过短期熔断的数据源: %s",
            market,
            fetcher.name,
        )
        return False


    @staticmethod
    def _daily_source_unavailable_error(fetcher: BaseFetcher) -> str:
        return f"[{fetcher.name}] (CircuitOpen) 数据源短期熔断"


    @classmethod
    def _record_daily_source_success(cls, fetcher: BaseFetcher, market: str) -> None:
        cls._daily_source_health.record_success(cls._daily_health_key(fetcher, market))


    @classmethod
    def _record_daily_source_failure(cls, fetcher: BaseFetcher, market: str, error: str) -> None:
        cls._daily_source_health.record_failure(cls._daily_health_key(fetcher, market), error=error)


    @classmethod
    def reset_daily_source_health(cls) -> None:
        """Reset daily source health state for tests/admin diagnostics."""
        cls._daily_source_health.reset()


    def _get_cached_stock_name(self, stock_code: str) -> Optional[str]:
        self._ensure_concurrency_guards()
        with self._stock_name_cache_lock:
            return self._stock_name_cache.get(stock_code)


    def _cache_stock_name(self, stock_code: str, name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        self._ensure_concurrency_guards()
        with self._stock_name_cache_lock:
            self._stock_name_cache[stock_code] = name
        return name


    def _discard_cached_stock_name(self, stock_code: str) -> None:
        self._ensure_concurrency_guards()
        with self._stock_name_cache_lock:
            self._stock_name_cache.pop(stock_code, None)


    def _get_tickflow_fetcher(self):
        """Lazily create a TickFlow fetcher for market-review-only calls."""
        from src.config import get_config

        config = get_config()
        api_key = (getattr(config, "tickflow_api_key", None) or "").strip()

        if not hasattr(self, "_tickflow_lock") or self._tickflow_lock is None:
            self._tickflow_lock = RLock()

        with self._tickflow_lock:
            current_fetcher = getattr(self, "_tickflow_fetcher", None)
            current_key = getattr(self, "_tickflow_api_key", None)

            if not api_key:
                if current_fetcher is not None and hasattr(current_fetcher, "close"):
                    try:
                        current_fetcher.close()
                    except Exception as exc:
                        logger.debug("[TickFlowFetcher] 关闭旧实例失败: %s", exc)
                self._tickflow_fetcher = None
                self._tickflow_api_key = None
                return None

            configured_fetcher = self._get_fetcher_by_name("TickFlowFetcher")
            if configured_fetcher is not None:
                return configured_fetcher

            if current_fetcher is not None and current_key == api_key:
                return current_fetcher

            if current_fetcher is not None and hasattr(current_fetcher, "close"):
                try:
                    current_fetcher.close()
                except Exception as exc:
                    logger.debug("[TickFlowFetcher] 切换实例时关闭失败: %s", exc)

            try:
                from .tickflow_fetcher import TickFlowFetcher

                fetcher = TickFlowFetcher(
                    api_key=api_key,
                    kline_adjust=getattr(config, "tickflow_kline_adjust", "none"),
                    batch_daily_enabled=getattr(config, "tickflow_batch_daily_enabled", True),
                    batch_size=getattr(config, "tickflow_batch_size", 100),
                    priority=getattr(config, "tickflow_priority", 2),
                )
                self._tickflow_fetcher = fetcher
                self._tickflow_api_key = api_key
                return fetcher
            except Exception as exc:
                logger.warning("[TickFlowFetcher] 初始化失败: %s", exc)
                self._tickflow_fetcher = None
                self._tickflow_api_key = None
                return None


    def close(self) -> None:
        """Best-effort release of manager-owned resources."""
        if not hasattr(self, "_tickflow_lock") or self._tickflow_lock is None:
            self._tickflow_lock = RLock()

        with self._tickflow_lock:
            current_fetcher = getattr(self, "_tickflow_fetcher", None)
            self._tickflow_fetcher = None
            self._tickflow_api_key = None

        if current_fetcher is not None and hasattr(current_fetcher, "close"):
            try:
                current_fetcher.close()
            except Exception as exc:
                logger.debug("[TickFlowFetcher] 关闭管理器资源失败: %s", exc)

        # The HK Futu fundamental path lazily creates and caches its own
        # FutuFetcher (an OpenQuoteContext-backed connection) on
        # _futu_fundamental_fetcher; release it here so explicit close /
        # reload paths do not leak the OpenD connection.
        futu_fundamental_fetcher = getattr(self, "_futu_fundamental_fetcher", None)
        if futu_fundamental_fetcher is not None:
            self._futu_fundamental_fetcher = None
            close_futu = getattr(futu_fundamental_fetcher, "close", None)
            if callable(close_futu):
                try:
                    close_futu()
                except Exception as exc:
                    logger.debug("[FutuFetcher] 关闭管理器资源失败: %s", exc)

        for fetcher in self._get_fetchers_snapshot():
            close = getattr(fetcher, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    logger.debug("[%s] close failed: %s", fetcher.name, exc)


    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Best-effort cleanup during interpreter shutdown.
            pass


    def _init_default_fetchers(self) -> None:
        """
        初始化默认数据源列表

        优先级动态调整逻辑：
        - 如果配置了 TUSHARE_TOKEN：实例化 TushareFetcher，并按其内部逻辑提升优先级
        - 如果配置了 Longbridge OAuth 或 Legacy 凭据：实例化 LongbridgeFetcher 作为美股/港股兜底
        - 未配置的可选数据源不实例化，避免在批量拉取时反复探测无效源
        - 默认优先级：
          0. EfinanceFetcher (Priority 0) - 最高优先级
          1. AkshareFetcher (Priority 1)
          2. PytdxFetcher (Priority 2) - 通达信
          3. BaostockFetcher (Priority 3)
          4. YfinanceFetcher (Priority 4)
          5. TencentFetcher (Priority 5) - A 股最终兜底
        """
        from src.config import get_config
        from .efinance_fetcher import EfinanceFetcher
        from .tencent_fetcher import TencentFetcher
        from .akshare_fetcher import AkshareFetcher
        from .tushare_fetcher import TushareFetcher
        from .tickflow_fetcher import TickFlowFetcher
        from .pytdx_fetcher import PytdxFetcher
        from .baostock_fetcher import BaostockFetcher
        from .yfinance_fetcher import YfinanceFetcher
        from .longbridge_fetcher import LongbridgeFetcher
        from .futu_fetcher import FutuFetcher
        config = get_config()
        # 创建所有数据源实例（优先级在各 Fetcher 的 __init__ 中确定）
        efinance = EfinanceFetcher()
        tencent = TencentFetcher()
        akshare = AkshareFetcher()
        pytdx = PytdxFetcher()      # 通达信数据源（可配 PYTDX_HOST/PYTDX_PORT）
        baostock = BaostockFetcher()
        yfinance = YfinanceFetcher()
        optional_fetchers: List[BaseFetcher] = []

        tushare_token = (getattr(config, "tushare_token", None) or "").strip()
        if tushare_token:
            optional_fetchers.append(TushareFetcher())  # 会根据 Token 配置自动调整优先级
        else:
            logger.debug("[数据源初始化] 跳过未配置的 TushareFetcher")

        tickflow_api_key = (getattr(config, "tickflow_api_key", None) or "").strip()
        if tickflow_api_key:
            optional_fetchers.append(
                TickFlowFetcher(
                    api_key=tickflow_api_key,
                    kline_adjust=getattr(config, "tickflow_kline_adjust", "none"),
                    batch_daily_enabled=getattr(config, "tickflow_batch_daily_enabled", True),
                    batch_size=getattr(config, "tickflow_batch_size", 100),
                    priority=getattr(config, "tickflow_priority", 2),
                )
            )
        else:
            logger.debug("[data source init] skip TickFlowFetcher because TICKFLOW_API_KEY is not configured")

        if LongbridgeFetcher.has_configured_credentials(config):
            optional_fetchers.append(LongbridgeFetcher())  # 长桥（美股/港股兜底，懒加载）
        else:
            logger.debug("[数据源初始化] 跳过未配置的 LongbridgeFetcher")

        if FutuFetcher.has_configured_endpoint():
            optional_fetchers.append(FutuFetcher())  # 富途（港股，依赖 OpenD）
        else:
            logger.debug("[数据源初始化] 跳过未配置的 FutuFetcher")

        finnhub_api_key = (getattr(config, "finnhub_api_key", None) or "").strip()
        if finnhub_api_key:
            from .finnhub_fetcher import FinnhubFetcher
            optional_fetchers.append(FinnhubFetcher())
        else:
            logger.debug("[数据源初始化] 跳过未配置的 FinnhubFetcher")

        alphavantage_api_key = (getattr(config, "alphavantage_api_key", None) or "").strip()
        if alphavantage_api_key:
            from .alphavantage_fetcher import AlphaVantageFetcher
            optional_fetchers.append(AlphaVantageFetcher())
        else:
            logger.debug("[数据源初始化] 跳过未配置的 AlphaVantageFetcher")

        # 初始化数据源列表
        self._ensure_concurrency_guards()
        with self._fetchers_lock:
            self._fetchers = [
                efinance,
                akshare,
                pytdx,
                baostock,
                yfinance,
                tencent,
                *optional_fetchers,
            ]

            # 按优先级排序（Tushare 如果配置了 Token 且初始化成功，优先级为 0）
            self._fetchers.sort(key=lambda f: f.priority)
            self._refresh_fetcher_indexes_locked()

        # 构建优先级说明
        priority_info = ", ".join([f"{f.name}(P{f.priority})" for f in self._get_fetchers_snapshot()])
        logger.info(f"已初始化 {len(self._fetchers)} 个数据源（按优先级）: {priority_info}")


    def add_fetcher(self, fetcher: BaseFetcher) -> None:
        """添加数据源并重新排序"""
        self._ensure_concurrency_guards()
        with self._fetchers_lock:
            self._fetchers.append(fetcher)
            self._fetchers.sort(key=lambda f: f.priority)
            self._refresh_fetcher_indexes_locked()
