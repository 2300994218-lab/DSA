# -*- coding: utf-8 -*-
"""Pytest compatibility hooks."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
import threading
from collections.abc import Awaitable, Callable
from contextvars import copy_context
from functools import wraps
from typing import Any, TypeVar
from warnings import warn

import anyio.to_thread
import fastapi.testclient
import httpx
import starlette.testclient
from anyio._backends import _asyncio

import os as _os

import pytest as _pytest

T = TypeVar("T")

_original_call_soon_threadsafe = asyncio.BaseEventLoop.call_soon_threadsafe

# Environment captured *before* collection imports any test module.
_PRISTINE_ENV: dict = {}
# Keys injected by ``.env`` loading (``dotenv.load_dotenv``) that were absent
# from the pristine environment, mapped to the value dotenv wrote. Real
# ``litellm`` calls ``load_dotenv()`` at import time (litellm/__init__.py), so
# importing any module that pulls litellm in (``api.app``, ``test_agent_executor``)
# loads the developer's real ``.env`` into the process *during collection* —
# before the first per-test snapshot exists — and the per-test restore alone
# cannot see it. Wrapping the dotenv entry point itself (instead of only
# ``src.config.setup_env``) catches every injector.
_DOTENV_INJECTED: dict = {}


def pytest_configure(config) -> None:
    _PRISTINE_ENV.clear()
    _PRISTINE_ENV.update(_os.environ)
    _install_dotenv_recorder()


def _record_env_delta(before: dict) -> None:
    for key in _os.environ:
        if key not in before and key not in _PRISTINE_ENV:
            _DOTENV_INJECTED[key] = _os.environ[key]


def _install_dotenv_recorder() -> None:
    """Wrap ``load_dotenv`` so every key dotenv writes is recorded.

    Both bindings must be patched: the ``dotenv`` package re-exports
    ``load_dotenv`` from ``dotenv.main`` as a separate attribute, and callers
    use either (``litellm`` calls ``dotenv.load_dotenv``, ``src.config`` binds
    ``from dotenv import load_dotenv`` at import time — rebound here too).
    """
    import dotenv
    import dotenv.main as _dotenv_main
    import src.config as _config_module

    original_load_dotenv = dotenv.load_dotenv

    def recording_load_dotenv(*args, **kwargs):
        before = dict(_os.environ)
        try:
            return original_load_dotenv(*args, **kwargs)
        finally:
            _record_env_delta(before)

    dotenv.load_dotenv = recording_load_dotenv  # type: ignore[attr-defined]
    _dotenv_main.load_dotenv = recording_load_dotenv  # type: ignore[attr-defined]
    _config_module.load_dotenv = recording_load_dotenv  # type: ignore[attr-defined]


@_pytest.fixture(autouse=True)
def _isolate_process_env():
    """Snapshot and restore ``os.environ`` around every test.

    Several modules call ``load_dotenv()`` at import time, which loads the
    developer's real ``.env`` into the process environment. Whatever one test
    module leaves behind (or whatever the local .env contributed) then leaks
    into later test modules that merge ``os.environ`` into their effective
    config maps — e.g. config-validation tests suddenly seeing a configured
    FEISHU_WEBHOOK_URL. Restoring the snapshot makes every test start from the
    same environment regardless of execution order.

    Import-time pollution needs one extra step: because it happens during
    collection, it is already part of every per-test snapshot. Keys recorded as
    dotenv-injected are therefore scrubbed after the snapshot restore — by key
    identity, not by value, because a later in-test ``.env`` load may legitimately
    rewrite the same key with a different value (the snapshot restore already
    reverted that). Keys written directly by test modules at import time
    (e.g. ``ENV_FILE``) are never touched — dotenv only writes keys that appear
    inside ``.env`` files, and ``ENV_FILE`` is not one of them.
    """
    snapshot = dict(_os.environ)
    yield
    _os.environ.clear()
    _os.environ.update(snapshot)
    for key in _DOTENV_INJECTED:
        if key not in _PRISTINE_ENV:
            _os.environ.pop(key, None)


async def _shutdown_default_executor_inline(
    self: asyncio.BaseEventLoop,
    timeout: float | None = None,
) -> None:
    """Avoid lost wakeups while asyncio.run() tears down test-only executors."""
    del timeout
    executor = getattr(self, "_default_executor", None)
    if executor is None:
        return
    self._executor_shutdown_called = True
    self._default_executor = None
    executor.shutdown(wait=True)


def _call_soon_threadsafe_with_extra_wakeup(
    self: asyncio.BaseEventLoop,
    callback,
    *args,
    context=None,
):
    """Wake the selector again for sandboxed test runs where the first wake is lost."""
    handle = _original_call_soon_threadsafe(self, callback, *args, context=context)
    write_to_self = getattr(self, "_write_to_self", None)
    if callable(write_to_self):
        write_to_self()
        threading.Timer(0.001, write_to_self).start()
    return handle


asyncio.BaseEventLoop.call_soon_threadsafe = _call_soon_threadsafe_with_extra_wakeup
asyncio.BaseEventLoop.shutdown_default_executor = _shutdown_default_executor_inline


async def _run_sync_via_asyncio_to_thread(
    func: Callable[..., T],
    *args: Any,
    abandon_on_cancel: bool = False,
    cancellable: bool | None = None,
    limiter: Any = None,
) -> T:
    """Use asyncio's executor path when AnyIO worker queues miss wakeups."""
    del abandon_on_cancel, limiter
    if cancellable is not None:
        warn(
            "The `cancellable=` keyword argument to `anyio.to_thread.run_sync` is "
            "deprecated since AnyIO 4.1.0; use `abandon_on_cancel=` instead",
            DeprecationWarning,
            stacklevel=2,
        )
    future: concurrent.futures.Future[T] = concurrent.futures.Future()
    context = copy_context()

    def runner() -> None:
        try:
            future.set_result(context.run(func, *args))
        except BaseException as exc:
            future.set_exception(exc)

    threading.Thread(target=runner, name="pytest-anyio-worker", daemon=True).start()
    while not future.done():
        await asyncio.sleep(0.001)
    return future.result()


def _wait_for_cross_thread_result(loop: asyncio.AbstractEventLoop, future: concurrent.futures.Future[T]) -> T:
    write_to_self = getattr(loop, "_write_to_self", None)
    while not future.done():
        if callable(write_to_self):
            write_to_self()
        time.sleep(0.001)
    return future.result()


def _run_sync_from_thread_with_wakeup(
    cls,
    func: Callable[..., T],
    args: tuple[Any, ...],
    token: object,
) -> T:
    @wraps(func)
    def wrapper() -> None:
        try:
            _asyncio.set_current_async_library("asyncio")
            future.set_result(func(*args))
        except BaseException as exc:
            future.set_exception(exc)
            if not isinstance(exc, Exception):
                raise

    loop = token or _asyncio.threadlocals.current_token.native_token
    if loop.is_closed():
        raise _asyncio.RunFinishedError
    future: concurrent.futures.Future[T] = concurrent.futures.Future()
    loop.call_soon_threadsafe(wrapper)
    return _wait_for_cross_thread_result(loop, future)


def _run_async_from_thread_with_wakeup(
    cls,
    func: Callable[..., Awaitable[T]],
    args: tuple[Any, ...],
    token: object,
) -> T:
    loop = token or _asyncio.threadlocals.current_token.native_token
    if loop.is_closed():
        raise _asyncio.RunFinishedError
    context = copy_context()
    context.run(_asyncio.set_current_async_library, "asyncio")
    future = context.run(asyncio.run_coroutine_threadsafe, func(*args), loop=loop)
    return _wait_for_cross_thread_result(loop, future)


anyio.to_thread.run_sync = _run_sync_via_asyncio_to_thread
_asyncio.AsyncIOBackend.run_sync_from_thread = classmethod(_run_sync_from_thread_with_wakeup)
_asyncio.AsyncIOBackend.run_async_from_thread = classmethod(_run_async_from_thread_with_wakeup)


class _ThreadlessTestClient:
    """Small TestClient replacement that avoids AnyIO's cross-thread portal."""

    def __init__(
        self,
        app,
        base_url: str = "http://testserver",
        raise_server_exceptions: bool = True,
        follow_redirects: bool = True,
        **_: Any,
    ) -> None:
        self.app = app
        self.base_url = base_url
        self.raise_server_exceptions = raise_server_exceptions
        self.follow_redirects = follow_redirects
        self.cookies = httpx.Cookies()
        self._lifespan_ctx = None
        self._lifespan_enter_count = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _get_lifespan_context(self):
        return getattr(getattr(self.app, "router", None), "lifespan_context", None)

    def _build_async_client(self, follow_redirects: bool) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(
            app=self.app,
            raise_app_exceptions=self.raise_server_exceptions,
        )
        return httpx.AsyncClient(
            transport=transport,
            base_url=self.base_url,
            follow_redirects=follow_redirects,
            cookies=self.cookies,
        )

    def __enter__(self):
        if self._lifespan_enter_count == 0:
            self._loop = asyncio.new_event_loop()
            lifespan_context = self._get_lifespan_context()
            if callable(lifespan_context):
                self._lifespan_ctx = lifespan_context(self.app)
                self._loop.run_until_complete(self._lifespan_ctx.__aenter__())
            self._async_client = self._build_async_client(self.follow_redirects)
        self._lifespan_enter_count += 1
        return self

    def __exit__(self, *args: Any) -> None:
        if self._lifespan_enter_count == 0:
            return None

        self._lifespan_enter_count -= 1
        if self._lifespan_enter_count == 0 and self._loop is not None:
            try:
                async def _close() -> None:
                    if self._async_client is not None:
                        await self._async_client.aclose()
                    if self._lifespan_ctx is not None:
                        await self._lifespan_ctx.__aexit__(*args)

                self._loop.run_until_complete(_close())
            finally:
                self._lifespan_ctx = None
                self._async_client = None
                self._loop.close()
                self._loop = None
        return None

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        follow_redirects = kwargs.pop("follow_redirects", self.follow_redirects)
        kwargs.pop("allow_redirects", None)

        if self._lifespan_enter_count > 0 and self._loop is not None and self._async_client is not None:
            response = self._loop.run_until_complete(
                self._async_client.request(method, url, follow_redirects=follow_redirects, **kwargs)
            )
            self.cookies = httpx.Cookies(self._async_client.cookies)
            return response

        async def _send() -> httpx.Response:
            async with self._build_async_client(follow_redirects) as client:
                response = await client.request(method, url, **kwargs)
                self.cookies = httpx.Cookies(client.cookies)
                return response

        return asyncio.run(_send())

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def head(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("HEAD", url, **kwargs)


fastapi.testclient.TestClient = _ThreadlessTestClient
starlette.testclient.TestClient = _ThreadlessTestClient
