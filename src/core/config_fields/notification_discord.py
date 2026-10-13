# -*- coding: utf-8 -*-
"""通知 – Notification – Discord（自 config_registry 拆出）。

本模块从 ``src.core.config_registry`` 拆出（v1.1.1 重构），
由 config_registry 按原文件顺序组装为 ``_FIELD_DEFINITIONS``。
"""

from typing import Any, Dict


FIELDS_NOTIFICATION_DISCORD: Dict[str, Dict[str, Any]] = {
"DISCORD_WEBHOOK_URL": {
    "title": "Discord Webhook URL",
    "description": "Discord webhook URL for channel notifications.",
    "category": "notification",
    "data_type": "string",
    "ui_control": "password",
    "is_sensitive": True,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 33,
    "help_key": "settings.notification.chat_bots",
    "examples": [
        "DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx/yyy",
    ],
    "docs": [
        {
            "label": "完整指南：Discord",
            "href": "https://github.com/ZhuLinsen/daily_stock_analysis/blob/main/docs/full-guide.md#discord",
        },
    ],
    "warning_codes": ["webhook_secret_value"],
},
"DISCORD_BOT_TOKEN": {
    "title": "Discord Bot Token",
    "description": "Discord bot token for interactive bot mode.",
    "category": "notification",
    "data_type": "string",
    "ui_control": "password",
    "is_sensitive": True,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 34,
    "help_key": "settings.notification.chat_bots",
    "examples": [
        "DISCORD_BOT_TOKEN=your_discord_bot_token",
        "DISCORD_MAIN_CHANNEL_ID=123456789012345678",
    ],
    "docs": [
        {
            "label": "完整指南：Discord",
            "href": "https://github.com/ZhuLinsen/daily_stock_analysis/blob/main/docs/full-guide.md#discord",
        },
    ],
    "warning_codes": ["secret_value"],
},
"DISCORD_MAIN_CHANNEL_ID": {
    "title": "Discord Channel ID",
    "description": "Discord main channel ID for sending messages.",
    "category": "notification",
    "data_type": "string",
    "ui_control": "text",
    "is_sensitive": False,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 35,
},
"DISCORD_INTERACTIONS_PUBLIC_KEY": {
    "title": "Discord Interactions Public Key",
    "description": "Discord public key used to verify inbound interaction/webhook signatures.",
    "category": "notification",
    "data_type": "string",
    "ui_control": "text",
    "is_sensitive": False,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 36,
},
}
