# -*- coding: utf-8 -*-
"""通知 – Notification – Email（自 config_registry 拆出）。

本模块从 ``src.core.config_registry`` 拆出（v1.1.1 重构），
由 config_registry 按原文件顺序组装为 ``_FIELD_DEFINITIONS``。
"""

from typing import Any, Dict


FIELDS_NOTIFICATION_EMAIL: Dict[str, Dict[str, Any]] = {
"EMAIL_SENDER": {
    "title": "Email Sender",
    "description": "Sender email address (SMTP host auto-detected).",
    "category": "notification",
    "data_type": "string",
    "ui_control": "text",
    "is_sensitive": False,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 25,
    "help_key": "settings.notification.email",
    "examples": [
        "EMAIL_SENDER=sender@example.com",
        "EMAIL_PASSWORD=your_smtp_authorization_code",
        "EMAIL_RECEIVERS=user1@example.com,user2@example.com",
    ],
    "docs": [
        {
            "label": "完整指南：邮件",
            "href": "https://github.com/ZhuLinsen/daily_stock_analysis/blob/main/docs/full-guide.md#邮件",
        },
    ],
    "warning_codes": [],
},
"EMAIL_PASSWORD": {
    "title": "Email Password",
    "description": "Email password or app-specific authorization code.",
    "category": "notification",
    "data_type": "string",
    "ui_control": "password",
    "is_sensitive": True,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {},
    "display_order": 26,
    "help_key": "settings.notification.email",
    "examples": [
        "EMAIL_PASSWORD=your_smtp_authorization_code",
    ],
    "docs": [
        {
            "label": "完整指南：邮件",
            "href": "https://github.com/ZhuLinsen/daily_stock_analysis/blob/main/docs/full-guide.md#邮件",
        },
    ],
    "warning_codes": ["secret_value", "smtp_authorization_code"],
},
"EMAIL_RECEIVERS": {
    "title": "Email Receivers",
    "description": "Comma-separated recipient email addresses. Leave empty to send to yourself.",
    "category": "notification",
    "data_type": "array",
    "ui_control": "textarea",
    "is_sensitive": False,
    "is_required": False,
    "is_editable": True,
    "default_value": None,
    "options": [],
    "validation": {"multi_value": True, "delimiter": ","},
    "display_order": 27,
    "help_key": "settings.notification.email",
    "examples": [
        "EMAIL_RECEIVERS=user1@example.com,user2@example.com",
        "STOCK_GROUP_1=600519,300750",
        "EMAIL_GROUP_1=user1@example.com",
    ],
    "docs": [
        {
            "label": "完整指南：邮件",
            "href": "https://github.com/ZhuLinsen/daily_stock_analysis/blob/main/docs/full-guide.md#邮件",
        },
    ],
    "warning_codes": ["comma_separated_values"],
},
}
