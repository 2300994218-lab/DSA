# -*- coding: utf-8 -*-
"""配置导入导出（自 system_config_service 拆出，v1.1.1 重构）。

公开入口 ``src.services.system_config_service.SystemConfigService`` 通过多继承
聚合各 Mixin，方法经 MRO 解析，行为与拆分前完全一致。
"""

from __future__ import annotations

from src.services.system_config_errors import (
    ConfigConflictError,
    ConfigImportError,
)
from typing import (
    Any,
    Dict,
    List,
)
import io


class SystemConfigImportExportMixin:
    """Mixin：见模块 docstring。"""

    def export_env(self) -> Dict[str, Any]:
        """Return the raw active `.env` content for backup."""
        if self._manager.env_path.exists():
            content = self._manager.env_path.read_text(encoding="utf-8")
        else:
            content = ""

        return {
            "content": content,
            "config_version": self._manager.get_config_version(),
            "updated_at": self._manager.get_updated_at(),
        }


    def export_desktop_env(self) -> Dict[str, Any]:
        """Return the raw active `.env` content for desktop backup compatibility."""
        return self.export_env()


    def import_env(
        self,
        *,
        config_version: str,
        content: str,
        reload_now: bool = True,
    ) -> Dict[str, Any]:
        """Merge imported `.env` assignments into the active config."""
        current_version = self._manager.get_config_version()
        if current_version != config_version:
            raise ConfigConflictError(current_version=current_version)

        updates = self._parse_imported_env_content(content)
        return self.update(
            config_version=config_version,
            items=updates,
            mask_token="__DSA_IMPORT_LITERAL_MASK__",
            reload_now=reload_now,
        )


    def import_desktop_env(
        self,
        *,
        config_version: str,
        content: str,
        reload_now: bool = True,
    ) -> Dict[str, Any]:
        """Merge imported `.env` assignments for desktop backup compatibility."""
        return self.import_env(
            config_version=config_version,
            content=content,
            reload_now=reload_now,
        )


    @staticmethod
    def _parse_imported_env_content(content: str) -> List[Dict[str, str]]:
        """Parse raw `.env` text into update items without expanding app templates."""
        normalized_content = content.replace("\ufeff", "")
        if not normalized_content.strip():
            raise ConfigImportError("未识别到有效 .env 配置")

        from dotenv import dotenv_values

        parsed = dotenv_values(stream=io.StringIO(normalized_content), interpolate=False)
        updates: List[Dict[str, str]] = []
        for key, value in parsed.items():
            if key is None:
                continue
            updates.append(
                {
                    "key": str(key).upper(),
                    "value": "" if value is None else str(value),
                }
            )

        if not updates:
            raise ConfigImportError("未识别到有效 .env 配置")

        return updates
