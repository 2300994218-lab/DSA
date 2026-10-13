# -*- coding: utf-8 -*-
"""通知 – Notification – Slack  (Bot > Webhook when both configured)（自 config_registry 拆出）。

本模块从 ``src.core.config_registry`` 拆出（v1.1.1 重构），
由 config_registry 按原文件顺序组装为 ``_FIELD_DEFINITIONS``。
"""

from typing import Any, Dict


FIELDS_NOTIFICATION_SLACK: Dict[str, Dict[str, Any]] = {
"SLACK_BOT_TOKEN": {
    "title": "Slack Bot Token",
    "description": "Slack Bot Token (xoxb-...). Recommended; supports image upload. Takes priority over Webhook when both are configured.",
    "category": "notification",
    "data_type": "string",
    "ui_control": "password",
    "is_sensitive": True,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 37,
    "help_key": "settings.notification.chat_bots",
    "examples": [
        "SLACK_BOT_TOKEN=xoxb-xxxx",
        "SLACK_CHANNEL_ID=C0123456789",
    ],
    "docs": [
        {
            "label": "完整指南：通知渠道配置",
            "href": "https://github.com/ZhuLinsen/daily_stock_analysis/blob/main/docs/full-guide.md#通知渠道详细配置",
        },
    ],
    "warning_codes": ["secret_value"],
},
"SLACK_CHANNEL_ID": {
    "title": "Slack Channel ID",
    "description": "Slack channel ID (required when using Bot Token).",
    "category": "notification",
    "data_type": "string",
    "ui_control": "text",
    "is_sensitive": False,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 38,
},
"SLACK_WEBHOOK_URL": {
    "title": "Slack Incoming Webhook URL",
    "description": "Slack Incoming Webhook URL (text only, no image support).",
    "category": "notification",
    "data_type": "string",
    "ui_control": "password",
    "is_sensitive": True,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 39,
},
}
