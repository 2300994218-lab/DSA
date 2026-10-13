# -*- coding: utf-8 -*-
"""基础设置 – 选股开关（自 config_registry 拆出）。

本模块从 ``src.core.config_registry`` 拆出（v1.1.1 重构），
由 config_registry 按原文件顺序组装为 ``_FIELD_DEFINITIONS``。
"""

from typing import Any, Dict


FIELDS_BASE_SCREENING: Dict[str, Dict[str, Any]] = {
"SCREENING_ENABLED": {
    "title": "Built-in Stock Screening",
    "description": "Enable DSA's built-in stock screening tab. The implementation is based on AlphaSift and maintained as part of DSA. Disabled by default.",
    "category": "base",
    "data_type": "boolean",
    "ui_control": "switch",
    "is_sensitive": False,
    "is_required": False,
    "is_editable": True,
    "default_value": "false",
    "options": [],
    "validation": {},
    "display_order": 17,
    "help_key": "settings.base.SCREENING_ENABLED",
    "examples": [
        "SCREENING_ENABLED=false",
        "SCREENING_ENABLED=true",
    ],
    "docs": [
        {
            "label": "LiteLLM Providers（官方）",
            "href": "https://docs.litellm.ai/docs/providers",
        },
        {
            "label": "LiteLLM OpenAI-compatible（官方）",
            "href": "https://docs.litellm.ai/docs/providers/openai_compatible",
        },
        {
            "label": "OpenAI 请求与鉴权（官方）",
            "href": "https://platform.openai.com/docs/api-reference/authentication",
        },
        {
            "label": "选股说明",
            "href": "https://github.com/ZhuLinsen/daily_stock_analysis/blob/main/docs/screening-engine.md",
        },
    ],
},
}
