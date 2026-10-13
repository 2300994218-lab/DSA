# -*- coding: utf-8 -*-
"""通知 – Notification – Telegram（自 config_registry 拆出）。

本模块从 ``src.core.config_registry`` 拆出（v1.1.1 重构），
由 config_registry 按原文件顺序组装为 ``_FIELD_DEFINITIONS``。
"""

from typing import Any, Dict


FIELDS_NOTIFICATION_TELEGRAM: Dict[str, Dict[str, Any]] = {
"TELEGRAM_BOT_TOKEN": {
    "title": "Telegram Bot Token",
    "description": "Telegram bot token (from @BotFather).",
    "category": "notification",
    "data_type": "string",
    "ui_control": "password",
    "is_sensitive": True,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 17,
    "help_key": "settings.notification.telegram",
    "examples": [
        "TELEGRAM_BOT_TOKEN=123456:ABC-DEF",
        "TELEGRAM_CHAT_ID=-1001234567890",
    ],
    "docs": [
        {
            "label": "完整指南：Telegram",
            "href": "https://github.com/ZhuLinsen/daily_stock_analysis/blob/main/docs/full-guide.md#telegram",
        },
    ],
    "warning_codes": ["secret_value"],
},
"TELEGRAM_CHAT_ID": {
    "title": "Telegram Chat ID",
    "description": "Telegram chat/group ID to send messages to.",
    "category": "notification",
    "data_type": "string",
    "ui_control": "text",
    "is_sensitive": False,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 18,
    "help_key": "settings.notification.telegram",
    "examples": [
        "TELEGRAM_CHAT_ID=-1001234567890",
        "TELEGRAM_MESSAGE_THREAD_ID=123",
    ],
    "docs": [
        {
            "label": "完整指南：Telegram",
            "href": "https://github.com/ZhuLinsen/daily_stock_analysis/blob/main/docs/full-guide.md#telegram",
        },
    ],
    "warning_codes": [],
},
"TELEGRAM_MESSAGE_THREAD_ID": {
    "title": "Telegram Thread ID",
    "description": "Telegram topic/thread ID for group messages (optional).",
    "category": "notification",
    "data_type": "string",
    "ui_control": "text",
    "is_sensitive": False,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 19,
},
}
