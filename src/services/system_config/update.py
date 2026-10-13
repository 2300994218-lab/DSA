# -*- coding: utf-8 -*-
"""配置更新与告警构建（自 system_config_service 拆出，v1.1.1 重构）。

公开入口 ``src.services.system_config_service.SystemConfigService`` 通过多继承
聚合各 Mixin，方法经 MRO 解析，行为与拆分前完全一致。
"""

from __future__ import annotations

import logging
from src.config import (
    Config,
    normalize_news_strategy_profile,
    resolve_news_window_days,
    setup_env,
)
from src.core.config_registry import get_field_definition
from src.services.system_config_errors import (
    ConfigConflictError,
    ConfigValidationError,
)
from typing import (
    Any,
    Dict,
    List,
    Sequence,
    Set,
    Tuple,
)

logger = logging.getLogger("src.services.system_config_service")


class SystemConfigUpdateMixin:
    """Mixin：见模块 docstring。"""

    def update(
        self,
        config_version: str,
        items: Sequence[Dict[str, str]],
        mask_token: str = "******",
        reload_now: bool = True,
    ) -> Dict[str, Any]:
        """Validate and persist updates into `.env`, then reload runtime config."""
        current_version = self._manager.get_config_version()
        if current_version != config_version:
            raise ConfigConflictError(current_version=current_version)

        issues = self._collect_issues(items=items, mask_token=mask_token)
        errors = [issue for issue in issues if issue["severity"] == "error"]
        if errors:
            raise ConfigValidationError(issues=errors)

        previous_map = self._manager.read_config_map()
        submitted_keys: Set[str] = set()
        updates: List[Tuple[str, str]] = []
        sensitive_keys: Set[str] = set()
        for item in items:
            key = item["key"].upper()
            value = item["value"]
            field_schema = get_field_definition(key, value)
            normalized_value = self._normalize_value_for_storage(value, field_schema)
            submitted_keys.add(key)
            updates.append((key, normalized_value))
            if bool(field_schema.get("is_sensitive", False)):
                sensitive_keys.add(key)

        updated_keys, skipped_masked_keys, new_version = self._manager.apply_updates(
            updates=updates,
            sensitive_keys=sensitive_keys,
            mask_token=mask_token,
        )

        warnings: List[str] = []
        reload_triggered = False
        if reload_now:
            try:
                Config.reset_instance()
                self._reload_runtime_singletons()
                setup_env(override=True)
                config = Config.get_instance()
                warnings.extend(config.validate())
                reload_triggered = True
            except Exception as exc:  # pragma: no cover - defensive branch
                logger.error("Configuration reload failed: %s", exc, exc_info=True)
                warnings.append("Configuration updated but reload failed")

        warnings.extend(
            self._build_explainability_warnings(
                submitted_keys=submitted_keys,
                reload_now=reload_now,
            )
        )
        update_map = dict(updates)
        warnings.extend(
            self._build_runtime_model_cleanup_warnings(
                previous_map=previous_map,
                updates=update_map,
            )
        )
        warnings.extend(
            self._build_hermes_unsupported_key_cleanup_warnings(
                previous_map=previous_map,
                updates=update_map,
            )
        )
        if self._runtime_scheduler is not None and submitted_keys & {
            "SCHEDULE_ENABLED",
            "SCHEDULE_TIME",
            "SCHEDULE_TIMES",
        }:
            try:
                self._runtime_scheduler.reconcile_from_config(
                    clear_enabled_override="SCHEDULE_ENABLED" in submitted_keys,
                )
            except Exception as exc:  # pragma: no cover - defensive branch
                logger.error("Runtime scheduler reconcile failed: %s", exc, exc_info=True)
                warnings.append("Configuration updated but runtime scheduler reconcile failed")

        return {
            "success": True,
            "config_version": new_version,
            "applied_count": len(updated_keys),
            "skipped_masked_count": len(skipped_masked_keys),
            "reload_triggered": reload_triggered,
            "updated_keys": updated_keys,
            "warnings": warnings,
        }


    def _build_explainability_warnings(
        self,
        *,
        submitted_keys: Set[str],
        reload_now: bool,
    ) -> List[str]:
        """Append user-facing runtime explainability warnings for key settings."""
        warnings: List[str] = []
        if not submitted_keys:
            return warnings

        current_map = self._manager.read_config_map()

        if submitted_keys & {"NEWS_MAX_AGE_DAYS", "NEWS_STRATEGY_PROFILE"}:
            raw_profile = current_map.get("NEWS_STRATEGY_PROFILE", "short")
            profile = normalize_news_strategy_profile(raw_profile)
            try:
                max_age = max(1, int(current_map.get("NEWS_MAX_AGE_DAYS", "3") or "3"))
            except (TypeError, ValueError):
                max_age = 3
            effective_days = resolve_news_window_days(
                news_max_age_days=max_age,
                news_strategy_profile=profile,
            )
            warnings.append(
                (
                    "新闻窗口已按策略计算："
                    f"NEWS_STRATEGY_PROFILE={profile}, "
                    f"NEWS_MAX_AGE_DAYS={max_age}, "
                    f"effective_days={effective_days} "
                    "(effective_days=min(profile_days, NEWS_MAX_AGE_DAYS))."
                )
            )

        if "MAX_WORKERS" in submitted_keys:
            try:
                max_workers = max(1, int(current_map.get("MAX_WORKERS", "3") or "3"))
            except (TypeError, ValueError):
                max_workers = 3
            if reload_now:
                warnings.append(
                    (
                        f"MAX_WORKERS={max_workers} 已保存。任务队列空闲时会自动应用；"
                        "若当前存在运行中任务，将在队列空闲后生效。"
                    )
                )
            else:
                warnings.append(
                    (
                        f"MAX_WORKERS={max_workers} 已写入 .env，但本次未触发运行时重载"
                        "（reload_now=false）；重载后才会应用。"
                    )
                )

        startup_only_run_keys = submitted_keys & {
            "RUN_IMMEDIATELY",
        }
        if startup_only_run_keys:
            warnings.append(
                (
                    f"{', '.join(sorted(startup_only_run_keys))} 已写入 .env。"
                    "它属于启动期单次运行配置：当前已运行的 WebUI/API 进程不会因为本次保存立即触发分析；"
                    "请重启当前进程后，在非 schedule 模式下按新值生效。"
                )
            )

        startup_only_schedule_keys = submitted_keys & {
            "SCHEDULE_RUN_IMMEDIATELY",
        }
        if startup_only_schedule_keys:
            warnings.append(
                (
                    f"{', '.join(sorted(startup_only_schedule_keys))} 已写入 .env。"
                    "这些属于启动期调度模式配置：当前已运行的 WebUI/API 进程不会因为本次保存启动、"
                    "停止或重建 scheduler；请重启当前进程，并以 schedule 模式重新启动后生效。"
                )
            )

        if "SCHEDULE_ENABLED" in submitted_keys:
            schedule_enabled = (current_map.get("SCHEDULE_ENABLED", "false") or "false").strip().lower()
            warnings.append(
                (
                    f"SCHEDULE_ENABLED={schedule_enabled} 已写入 .env。"
                    "如果当前进程是 WebUI/API/Desktop 长运行进程，runtime scheduler 会按新配置启停；"
                    "CLI schedule 模式仍按启动参数和配置运行。"
                )
            )

        if "SCHEDULE_TIMES" in submitted_keys:
            schedule_times = (current_map.get("SCHEDULE_TIMES", "") or "").strip()
            schedule_time = (current_map.get("SCHEDULE_TIME", "") or "").strip() or "18:00"
            effective = schedule_times or schedule_time
            warnings.append(
                (
                    f"SCHEDULE_TIMES={effective} 已写入 .env。"
                    "有效时间点会去重、排序；为空时继续使用 SCHEDULE_TIME。"
                    "如果当前进程存在 runtime scheduler，会按新时间重建 daily jobs。"
                )
            )

        if "SCHEDULE_TIME" in submitted_keys:
            schedule_time = (current_map.get("SCHEDULE_TIME", "") or "").strip() or "18:00"
            warnings.append(
                (
                    f"SCHEDULE_TIME={schedule_time} 已写入 .env。"
                    "如果当前进程已经以 schedule 模式运行，scheduler 会在下一轮检查中自动重建 daily job；"
                    "如果当前进程未以 schedule 模式运行，本次保存不会启动 scheduler。"
                )
            )

        startup_only_bind_keys = submitted_keys & {
            "WEBUI_HOST",
            "WEBUI_PORT",
        }
        if startup_only_bind_keys:
            warnings.append(
                (
                    f"{', '.join(sorted(startup_only_bind_keys))} 已写入 .env。"
                    "这些属于启动期监听配置：当前已运行的 WebUI/API 进程不会因为本次保存重新绑定监听地址或端口；"
                    "请重启当前进程、Docker 容器或服务管理器后生效。"
                )
            )

        return warnings


    @staticmethod
    def _build_runtime_model_cleanup_warnings(
        *,
        previous_map: Dict[str, str],
        updates: Dict[str, str],
    ) -> List[str]:
        """Explain when save payload clears stale runtime model references."""
        runtime_labels = {
            "LITELLM_MODEL": "主模型",
            "AGENT_LITELLM_MODEL": "Agent 主模型",
            "VISION_MODEL": "Vision 模型",
        }
        cleared_labels: List[str] = []
        for key, label in runtime_labels.items():
            if previous_map.get(key, "").strip() and key in updates and not updates[key].strip():
                cleared_labels.append(label)

        removed_fallbacks: List[str] = []
        if "LITELLM_FALLBACK_MODELS" in updates:
            previous_fallbacks = [
                item.strip()
                for item in previous_map.get("LITELLM_FALLBACK_MODELS", "").split(",")
                if item.strip()
            ]
            next_fallbacks = {
                item.strip()
                for item in updates["LITELLM_FALLBACK_MODELS"].split(",")
                if item.strip()
            }
            removed_fallbacks = [item for item in previous_fallbacks if item not in next_fallbacks]

        if not cleared_labels and not removed_fallbacks:
            return []

        cleaned_targets = list(cleared_labels)
        if removed_fallbacks:
            cleaned_targets.append("备选模型中的失效项")

        cleaned_text = " / ".join(cleaned_targets)
        warning = (
            f"检测到已同步清理失效的运行时模型引用：{cleaned_text}。"
            "如需恢复，请先补回对应渠道模型列表后重新选择；"
            "也可用桌面端导出备份或手动 .env 还原之前的 LLM_* / "
            "LITELLM_MODEL / AGENT_LITELLM_MODEL / VISION_MODEL / LLM_TEMPERATURE。"
        )
        return [warning]


    @staticmethod
    def _build_hermes_unsupported_key_cleanup_warnings(
        *,
        previous_map: Dict[str, str],
        updates: Dict[str, str],
    ) -> List[str]:
        """Explain when Hermes save clears unsupported Phase 3 key/header fields."""
        unsupported_labels = {
            "LLM_HERMES_API_KEYS": "LLM_HERMES_API_KEYS",
            "LLM_HERMES_EXTRA_HEADERS": "LLM_HERMES_EXTRA_HEADERS",
        }
        cleared = [
            label
            for key, label in unsupported_labels.items()
            if previous_map.get(key, "").strip() and key in updates and not updates[key].strip()
        ]
        if not cleared:
            return []

        return [
            (
                "检测到已清理 Hermes Phase 3 不支持的配置项："
                f"{', '.join(cleared)}。"
                "Hermes reserved channel 只支持单个 LLM_HERMES_API_KEY，不支持多 Key 或额外 Header；"
                "如需恢复旧值，请从 .env 备份、Git 历史或桌面端导出备份手动还原，"
                "但非空 LLM_HERMES_API_KEYS / LLM_HERMES_EXTRA_HEADERS 仍会被后端校验拒绝。"
            )
        ]


    def apply_simple_updates(
        self,
        updates: Sequence[Tuple[str, str]],
        mask_token: str = "******",
    ) -> None:
        """Apply raw key updates without validation (internal service use only)."""
        self._manager.apply_updates(
            updates=updates,
            sensitive_keys=set(),
            mask_token=mask_token,
        )
