# -*- coding: utf-8 -*-
"""情报搜索：RSS / NewsNow / AKShare 新闻池。

从 ``src.search_service`` 拆出（v1.1.1 重构），公开导入路径保持不变。
"""

import logging

import re
import threading
import time
from typing import Any, Dict

from src.search_models import SearchResponse, SearchResult
from src.search_providers.base import BaseSearchProvider

logger = logging.getLogger("src.search_service")


class IntelligenceSearchProvider(BaseSearchProvider):
    """Search the user-configured RSS, NewsNow and AKShare news pool.

    Sources are managed by the Settings > News Sources page.  The provider
    deliberately sits after remote search providers, so Tavily/Brave keep
    their existing priority while configured local sources remain available as
    a first-party fallback for every AI news search.
    """

    _refresh_lock = threading.Lock()
    _last_refresh_at = 0.0
    _refresh_interval_seconds = 60.0

    def __init__(self):
        super().__init__([], "Configured News Sources")

    @classmethod
    def has_enabled_sources(cls) -> bool:
        try:
            from src.repositories.intelligence_repo import IntelligenceRepository
            _, total = IntelligenceRepository().list_sources(enabled=True, page=1, page_size=1)
            return bool(total)
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self.has_enabled_sources()

    def search(self, query: str, max_results: int = 5, days: int = 7) -> SearchResponse:
        start = time.time()
        try:
            from src.services.intelligence_service import IntelligenceService

            now = time.time()
            if now - type(self)._last_refresh_at >= type(self)._refresh_interval_seconds:
                with type(self)._refresh_lock:
                    now = time.time()
                    if now - type(self)._last_refresh_at >= type(self)._refresh_interval_seconds:
                        # A failed upstream source is isolated by the service;
                        # already persisted items remain searchable.
                        IntelligenceService().fetch_enabled_sources()
                        type(self)._last_refresh_at = time.time()

            payload = IntelligenceService().list_items(
                days=max(1, int(days)),
                page=1,
                page_size=100,
            )
            items = payload.get("items") or []
            normalized_query = re.sub(r"\s+", " ", str(query or "").strip().lower())
            terms = [term for term in normalized_query.split(" ") if len(term) >= 2]

            def score(item: Dict[str, Any]) -> int:
                haystack = " ".join(
                    str(item.get(key) or "") for key in ("title", "summary", "source_name", "source")
                ).lower()
                if not terms:
                    return 0
                return sum(1 for term in terms if term in haystack)

            ranked = sorted(items, key=score, reverse=True)
            matched = [item for item in ranked if score(item) > 0]
            selected = (matched or ranked)[:max_results]
            results = [
                SearchResult(
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("summary") or "")[:1000],
                    url=str(item.get("url") or ""),
                    source=str(item.get("source_name") or item.get("source") or self.name),
                    published_date=str(item.get("published_at") or "") or None,
                )
                for item in selected
                if str(item.get("title") or "").strip()
            ]
            return SearchResponse(
                query=query,
                results=results,
                provider=self.name,
                success=True,
                search_time=time.time() - start,
            )
        except Exception as exc:
            logger.warning("[%s] configured source search failed: %s", self.name, exc)
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                success=False,
                error_message=str(exc),
                search_time=time.time() - start,
            )

    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        # BaseSearchProvider requires this abstract hook; search() owns the
        # keyless implementation above.
        return self.search(query, max_results=max_results, days=days)
