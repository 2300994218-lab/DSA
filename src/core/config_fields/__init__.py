# -*- coding: utf-8 -*-
"""配置字段元数据（按 section 拆分，v1.1.1 重构）。

由 ``src.core.config_registry`` 按原文件顺序组装为 ``_FIELD_DEFINITIONS``，
保证字段注册顺序与拆分前完全一致。
"""

from src.core.config_fields.base import FIELDS_BASE
from src.core.config_fields.ai_model_core import FIELDS_AI_MODEL_CORE
from src.core.config_fields.data_source_tokens import FIELDS_DATA_SOURCE_TOKENS
from src.core.config_fields.base_screening import FIELDS_BASE_SCREENING
from src.core.config_fields.data_source_runtime import FIELDS_DATA_SOURCE_RUNTIME
from src.core.config_fields.ai_model_providers import FIELDS_AI_MODEL_PROVIDERS
from src.core.config_fields.notification_wechat import FIELDS_NOTIFICATION_WECHAT
from src.core.config_fields.notification_feishu import FIELDS_NOTIFICATION_FEISHU
from src.core.config_fields.notification_telegram import FIELDS_NOTIFICATION_TELEGRAM
from src.core.config_fields.notification_email import FIELDS_NOTIFICATION_EMAIL
from src.core.config_fields.notification_discord import FIELDS_NOTIFICATION_DISCORD
from src.core.config_fields.notification_slack import FIELDS_NOTIFICATION_SLACK
from src.core.config_fields.notification_pushover import FIELDS_NOTIFICATION_PUSHOVER
from src.core.config_fields.notification_misc import FIELDS_NOTIFICATION_MISC
from src.core.config_fields.system import FIELDS_SYSTEM
from src.core.config_fields.backtest import FIELDS_BACKTEST
from src.core.config_fields.agent import FIELDS_AGENT

__all__ = [
    "FIELDS_BASE",
    "FIELDS_AI_MODEL_CORE",
    "FIELDS_DATA_SOURCE_TOKENS",
    "FIELDS_BASE_SCREENING",
    "FIELDS_DATA_SOURCE_RUNTIME",
    "FIELDS_AI_MODEL_PROVIDERS",
    "FIELDS_NOTIFICATION_WECHAT",
    "FIELDS_NOTIFICATION_FEISHU",
    "FIELDS_NOTIFICATION_TELEGRAM",
    "FIELDS_NOTIFICATION_EMAIL",
    "FIELDS_NOTIFICATION_DISCORD",
    "FIELDS_NOTIFICATION_SLACK",
    "FIELDS_NOTIFICATION_PUSHOVER",
    "FIELDS_NOTIFICATION_MISC",
    "FIELDS_SYSTEM",
    "FIELDS_BACKTEST",
    "FIELDS_AGENT",
]
