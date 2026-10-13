# -*- coding: utf-8 -*-
"""系统配置服务异常类型（自 system_config_service 拆出，v1.1.1 重构）。

由 ``src.services.system_config_service`` 再导出，历史导入路径保持不变。
"""

from __future__ import annotations

from typing import Any, Dict, List


class ConfigValidationError(Exception):
    """Raised when one or more submitted fields fail validation."""

    def __init__(self, issues: List[Dict[str, Any]]):
        super().__init__("Configuration validation failed")
        self.issues = issues


class ConfigConflictError(Exception):
    """Raised when submitted config_version is stale."""

    def __init__(self, current_version: str):
        super().__init__("Configuration version conflict")
        self.current_version = current_version


class ConfigImportError(Exception):
    """Raised when an imported `.env` payload is invalid."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
