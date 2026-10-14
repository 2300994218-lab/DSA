# -*- coding: utf-8 -*-
"""Tests for FastAPI app CORS configuration."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.middleware.cors import CORSMiddleware

from api.app import create_app
from src.config import Config


class AppCorsConfigTestCase(unittest.TestCase):
    """CORS configuration should stay browser-compatible."""

    def _build_app(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return create_app(static_dir=Path(temp_dir.name))

    def test_allow_all_disables_credentials(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true"}, clear=False):
            app = self._build_app()

        cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
        self.assertEqual(cors.kwargs["allow_origins"], ["*"])
        self.assertFalse(cors.kwargs["allow_credentials"])

    def test_allow_all_warns_when_auth_is_disabled(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.app.logger.warning") as warning:
            self._build_app()

        warning.assert_called_once()
        self.assertIn("CORS_ALLOW_ALL=true", warning.call_args.args[0])
        self.assertIn("ADMIN_AUTH_ENABLED is false", warning.call_args.args[0])

    def test_allow_all_does_not_warn_when_auth_is_enabled(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=True), \
             patch("api.app.logger.warning") as warning:
            self._build_app()

        warning.assert_not_called()

    def test_explicit_origin_list_keeps_credentials_enabled(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "false"}, clear=False):
            app = self._build_app()

        cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
        self.assertIn("http://localhost:5173", cors.kwargs["allow_origins"])
        self.assertTrue(cors.kwargs["allow_credentials"])

    def _origins_with_env(self, extra_env: dict):
        """Build the app from a fresh Config instance loaded under ``extra_env``."""
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        env_file = Path(temp_dir.name) / ".env"
        env_file.write_text("STOCK_LIST=600519\nADMIN_AUTH_ENABLED=false\n", encoding="utf-8")
        previous_env_file = os.environ.get("ENV_FILE")
        os.environ["ENV_FILE"] = str(env_file)
        self.addCleanup(self._restore_env_file, previous_env_file)

        env = {"CORS_ALLOW_ALL": "false", **extra_env}
        with patch.dict(os.environ, env, clear=False):
            Config.reset_instance()
            self.addCleanup(Config.reset_instance)
            app = create_app(static_dir=Path(temp_dir.name) / "static")
        return next(m for m in app.user_middleware if m.cls is CORSMiddleware).kwargs["allow_origins"]

    @staticmethod
    def _restore_env_file(previous_env_file):
        if previous_env_file is None:
            os.environ.pop("ENV_FILE", None)
        else:
            os.environ["ENV_FILE"] = previous_env_file

    def test_default_allow_list_keeps_builtin_localhost_origins(self):
        origins = self._origins_with_env({})

        for origin in (
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ):
            self.assertIn(origin, origins)
        self.assertEqual(origins.count("http://localhost:5173"), 1)

    def test_cors_origins_setting_appends_extra_allow_list_entries(self):
        origins = self._origins_with_env(
            {"CORS_ORIGINS": "https://dsa.example.com, http://192.168.1.10:5173 ,http://localhost:5173"}
        )

        self.assertIn("https://dsa.example.com", origins)
        self.assertIn("http://192.168.1.10:5173", origins)
        self.assertNotIn(" http://192.168.1.10:5173 ", origins)
        # extras append to the built-in base and never duplicate or replace it
        self.assertIn("http://localhost:5173", origins)
        self.assertIn("http://127.0.0.1:3000", origins)
        self.assertEqual(origins.count("http://localhost:5173"), 1)
        self.assertLess(origins.index("http://localhost:5173"), origins.index("https://dsa.example.com"))


if __name__ == "__main__":
    unittest.main()
