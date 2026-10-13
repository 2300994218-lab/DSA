# -*- coding: utf-8 -*-
"""Configuration field metadata registry.

This module is the single source of truth for configuration UI metadata,
validation hints, and category grouping.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


from src.core.config_fields import (
    FIELDS_BASE,
    FIELDS_AI_MODEL_CORE,
    FIELDS_DATA_SOURCE_TOKENS,
    FIELDS_BASE_SCREENING,
    FIELDS_DATA_SOURCE_RUNTIME,
    FIELDS_AI_MODEL_PROVIDERS,
    FIELDS_NOTIFICATION_WECHAT,
    FIELDS_NOTIFICATION_FEISHU,
    FIELDS_NOTIFICATION_TELEGRAM,
    FIELDS_NOTIFICATION_EMAIL,
    FIELDS_NOTIFICATION_DISCORD,
    FIELDS_NOTIFICATION_SLACK,
    FIELDS_NOTIFICATION_PUSHOVER,
    FIELDS_NOTIFICATION_MISC,
    FIELDS_SYSTEM,
    FIELDS_BACKTEST,
    FIELDS_AGENT,
)
from src.core.config_fields_help import _FIELD_HELP_METADATA


SCHEMA_VERSION = "2026-06-29-claude-code-cli-backend"

_CATEGORY_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "category": "base",
        "title": "Base Settings",
        "description": "Watchlist and foundational application settings.",
        "display_order": 10,
    },
    {
        "category": "ai_model",
        "title": "AI Model",
        "description": "Model providers, model names, and inference parameters.",
        "display_order": 20,
    },
    {
        "category": "data_source",
        "title": "Data Source",
        "description": "Market data provider credentials and priority settings.",
        "display_order": 30,
    },
    {
        "category": "news_source",
        "title": "News Sources",
        "description": "Configure and test RSS, NewsNow and AKShare news sources used by AI search.",
        "display_order": 35,
    },
    {
        "category": "notification",
        "title": "Notification",
        "description": "Bot, webhook, and push channel related settings.",
        "display_order": 40,
    },
    {
        "category": "system",
        "title": "System",
        "description": "Runtime and scheduling controls.",
        "display_order": 50,
    },
    {
        "category": "agent",
        "title": "Agent",
        "description": "Agent mode and strategy-skill settings.",
        "display_order": 55,
    },
    {
        "category": "backtest",
        "title": "Backtest",
        "description": "Backtest engine behavior and evaluation parameters.",
        "display_order": 60,
    },
    {
        "category": "uncategorized",
        "title": "Uncategorized",
        "description": "Keys not mapped in the field registry.",
        "display_order": 99,
    },
]

WEB_SETTINGS_HIDDEN_FROM_UI = {
    "DATABASE_PATH",
    "SQLITE_WAL_ENABLED",
    "SQLITE_BUSY_TIMEOUT_MS",
    "SQLITE_WRITE_RETRY_MAX",
    "SQLITE_WRITE_RETRY_BASE_DELAY",
    "USE_PROXY",
    "PROXY_HOST",
    "PROXY_PORT",
    "SEARXNG_TIMEOUT_SECONDS",
}


_FIELD_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    **FIELDS_BASE,
    **FIELDS_AI_MODEL_CORE,
    **FIELDS_DATA_SOURCE_TOKENS,
    **FIELDS_BASE_SCREENING,
    **FIELDS_DATA_SOURCE_RUNTIME,
    **FIELDS_AI_MODEL_PROVIDERS,
    **FIELDS_NOTIFICATION_WECHAT,
    **FIELDS_NOTIFICATION_FEISHU,
    **FIELDS_NOTIFICATION_TELEGRAM,
    **FIELDS_NOTIFICATION_EMAIL,
    **FIELDS_NOTIFICATION_DISCORD,
    **FIELDS_NOTIFICATION_SLACK,
    **FIELDS_NOTIFICATION_PUSHOVER,
    **FIELDS_NOTIFICATION_MISC,
    **FIELDS_SYSTEM,
    **FIELDS_BACKTEST,
    **FIELDS_AGENT,
}



def get_category_definitions() -> List[Dict[str, Any]]:
    """Return deep-copied category metadata."""
    return deepcopy(_CATEGORY_DEFINITIONS)


def get_registered_field_keys() -> List[str]:
    """Return all explicitly registered keys."""
    return list(_FIELD_DEFINITIONS.keys())


def _extract_option_values(options: List[Any]) -> List[str]:
    """Extract canonical option values from string/object style select options."""
    values: List[str] = []
    for option in options:
        if isinstance(option, str):
            values.append(option)
            continue
        if isinstance(option, dict):
            value = option.get("value")
            if isinstance(value, str) and value:
                values.append(value)
    return values


def get_field_definition(key: str, value_hint: Optional[str] = None) -> Dict[str, Any]:
    """Return field definition for key, including inferred fallback metadata."""
    key_upper = key.upper()
    if key_upper in _FIELD_DEFINITIONS:
        field = deepcopy(_FIELD_DEFINITIONS[key_upper])
        if key_upper in _FIELD_HELP_METADATA:
            field.update(deepcopy(_FIELD_HELP_METADATA[key_upper]))
        field["key"] = key_upper
        validation = deepcopy(field.get("validation") or {})
        option_values = _extract_option_values(field.get("options", []))
        if field.get("ui_control") == "select" and option_values and "enum" not in validation:
            validation["enum"] = option_values
        field["validation"] = validation
        return field

    category = _infer_category(key_upper)
    data_type = _infer_data_type(key_upper, value_hint)
    field = {
        "key": key_upper,
        "title": key_upper.replace("_", " ").title(),
        "description": "Auto-inferred field metadata.",
        "category": category,
        "data_type": data_type,
        "ui_control": _infer_ui_control(data_type, key_upper),
        "is_sensitive": _is_sensitive_key(key_upper),
        "is_required": False,
        "is_editable": True,
        "default_value": None,
        "options": [],
        "validation": {},
        "display_order": 9000,
    }
    return field


def build_schema_response() -> Dict[str, Any]:
    """Build schema payload grouped by category."""
    category_map: Dict[str, Dict[str, Any]] = {}
    for category in get_category_definitions():
        category_map[category["category"]] = {**category, "fields": []}

    for key in sorted(_FIELD_DEFINITIONS.keys()):
        field = get_field_definition(key)
        category_map[field["category"]]["fields"].append(field)

    categories = sorted(category_map.values(), key=lambda item: item["display_order"])
    for category in categories:
        category["fields"] = sorted(
            category["fields"],
            key=lambda item: (item.get("display_order", 9999), item["key"]),
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "categories": categories,
    }


def _is_sensitive_key(key: str) -> bool:
    markers = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    return any(marker in key for marker in markers)


def _infer_category(key: str) -> str:
    if key == "STOCK_LIST":
        return "base"
    if key.startswith("BACKTEST_"):
        return "backtest"
    if key.startswith(("GEMINI_", "OPENAI_", "ANTHROPIC_", "LITELLM_", "AIHUBMIX_", "DEEPSEEK_", "LLM_")):
        return "ai_model"
    if key.endswith("_PRIORITY") or key.startswith(
        (
            "TUSHARE",
            "TICKFLOW",
            "AKSHARE",
            "EFINANCE",
            "PYTDX",
            "BAOSTOCK",
            "YFINANCE",
            "TAVILY",
            "SERPAPI",
            "BRAVE",
            "BOCHA",
            "ANSPIRE",
            "SEARXNG",
            "NEWS_",
            "BIAS_",
        )
    ) or key in ("ENABLE_REALTIME_QUOTE", "ENABLE_CHIP_DISTRIBUTION"):
        return "data_source"
    if key.startswith((
        "WECHAT",
        "FEISHU",
        "TELEGRAM",
        "EMAIL",
        "PUSHOVER",
        "NTFY",
        "GOTIFY",
        "PUSHPLUS",
        "SERVERCHAN",
        "DINGTALK",
        "DISCORD",
        "SLACK",
        "CUSTOM_WEBHOOK",
        "WECOM",
        "ASTRBOT",
    )) or "WEBHOOK" in key:
        return "notification"
    if key.startswith(("LOG_", "SCHEDULE_", "WEBUI_", "HTTP_", "HTTPS_", "MAX_", "DEBUG", "MARKET_REVIEW_", "TRADING_DAY_", "ANALYSIS_DELAY")):
        return "system"
    return "uncategorized"


def _infer_data_type(key: str, value_hint: Optional[str]) -> str:
    if key.endswith("_TIME"):
        return "time"
    if value_hint is None:
        return "string"

    lowered = value_hint.strip().lower()
    if lowered in {"true", "false"}:
        return "boolean"

    try:
        int(value_hint)
        return "integer"
    except (TypeError, ValueError):
        pass

    try:
        float(value_hint)
        return "number"
    except (TypeError, ValueError):
        pass

    if key in {"STOCK_LIST", "EMAIL_RECEIVERS", "CUSTOM_WEBHOOK_URLS"}:
        return "array"
    return "string"


def _infer_ui_control(data_type: str, key: str) -> str:
    if _is_sensitive_key(key):
        return "password"
    if data_type == "boolean":
        return "switch"
    if data_type in {"integer", "number"}:
        return "number"
    if data_type == "time":
        return "time"
    if data_type == "array":
        return "textarea"
    return "text"
