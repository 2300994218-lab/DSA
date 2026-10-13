# -*- coding: utf-8 -*-
"""data_provider.base 模块级共享定义（自 base.py 拆出，v1.1.1 重构）。

``data_provider.base`` 与各 Mixin 模块共同引用；``data_provider.base``
原样再导出这些名字，``from data_provider.base import ...`` 行为不变。
"""

from __future__ import annotations

from typing import Callable, Optional, List, Tuple, Dict, Any
from src.services.market_symbol_utils import is_suffix_market_symbol
import numpy as np

SHARED = [
    "STANDARD_COLUMNS",
    "unwrap_exception",
    "summarize_exception",
    "normalize_stock_code",
    "ETF_PREFIXES",
    "_is_us_market",
    "_is_hk_market",
    "_is_jp_market",
    "_is_kr_market",
    "_is_tw_market",
    "_is_etf_code",
    "_coerce_chip_metric",
    "_is_meaningful_chip_distribution",
    "_market_tag",
]

STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']

def unwrap_exception(exc: Exception) -> Exception:
    """
    Follow chained exceptions and return the deepest non-cyclic cause.
    """
    current = exc
    visited = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        next_exc = current.__cause__ or current.__context__
        if next_exc is None:
            break
        current = next_exc

    return current

def summarize_exception(exc: Exception) -> Tuple[str, str]:
    """
    Build a stable summary for logs while preserving the application-layer message.
    """
    root = unwrap_exception(exc)
    error_type = type(root).__name__
    message = str(exc).strip() or str(root).strip() or error_type
    return error_type, " ".join(message.split())

def normalize_stock_code(stock_code: str) -> str:
    """
    Normalize stock code by stripping exchange prefixes/suffixes.

    Accepted formats and their normalized results:
    - '600519'      -> '600519'   (already clean)
    - 'SH600519'    -> '600519'   (strip SH prefix)
    - 'SH.600519'   -> '600519'   (strip SH. prefix)
    - 'SZ000001'    -> '000001'   (strip SZ prefix)
    - 'SS600519'    -> '600519'   (strip legacy Yahoo Shanghai prefix)
    - 'SZ.000001'   -> '000001'   (strip SZ. prefix)
    - 'BJ920748'    -> '920748'   (strip BJ prefix, BSE)
    - 'BJ.920748'   -> '920748'   (strip BJ. prefix, BSE)
    - 'sh600519'    -> '600519'   (case-insensitive)
    - '600519.SH'   -> '600519'   (strip .SH suffix)
    - '000001.SZ'   -> '000001'   (strip .SZ suffix)
    - '920748.BJ'   -> '920748'   (strip .BJ suffix, BSE)
    - 'HK00700'     -> 'HK00700'  (keep HK prefix for HK stocks)
    - '1810.HK'     -> 'HK01810'  (normalize HK suffix to canonical prefix form)
    - '7203.T'      -> '7203.T'   (keep Japan Yahoo suffix form)
    - '005930.KS'   -> '005930.KS' (keep Korea Yahoo suffix form)
    - '2330.TW'     -> '2330.TW'  (keep Taiwan TWSE Yahoo suffix form)
    - '6505.TWO'    -> '6505.TWO' (keep Taiwan TPEx Yahoo suffix form)
    - 'AAPL'        -> 'AAPL'     (keep US stock ticker as-is)

    This function is applied at the DataProviderManager layer so that
    all individual fetchers receive a clean 6-digit code (for A-shares/ETFs).
    """
    code = stock_code.strip()
    upper = code.upper()

    # Normalize HK prefix to a canonical 5-digit form (e.g. hk1810 -> HK01810)
    if upper.startswith('HK') and not upper.startswith('HK.'):
        candidate = upper[2:]
        if candidate.isdigit() and 1 <= len(candidate) <= 5:
            return f"HK{candidate.zfill(5)}"

    # Strip SH/SZ/SS prefix (e.g. SH600519 -> 600519, SS600519 -> 600519)
    if upper.startswith(('SH', 'SZ', 'SS')) and not upper.startswith(('SH.', 'SZ.', 'SS.')):
        candidate = code[2:]
        # Only strip if the remainder looks like a valid numeric code
        if candidate.isdigit() and len(candidate) in (5, 6):
            return candidate

    # Strip dotted SH/SZ/SS prefix (e.g. SH.600519 -> 600519)
    if upper.startswith(('SH.', 'SZ.', 'SS.')):
        candidate = code[3:]
        if candidate.isdigit() and len(candidate) in (5, 6):
            return candidate

    # Strip BJ prefix (e.g. BJ920748 -> 920748)
    if upper.startswith('BJ') and not upper.startswith('BJ.'):
        candidate = code[2:]
        if candidate.isdigit() and len(candidate) == 6:
            return candidate

    # Strip dotted BJ prefix (e.g. BJ.920748 -> 920748)
    if upper.startswith('BJ.'):
        candidate = code[3:]
        if candidate.isdigit() and len(candidate) == 6:
            return candidate

    # Strip .SH/.SZ/.BJ suffix (e.g. 600519.SH -> 600519, 920748.BJ -> 920748)
    # while preserving explicit Yahoo suffix forms for JP/KR/TW.
    if '.' in code:
        base, suffix = code.rsplit('.', 1)
        if suffix.upper() == 'T' and base.isdigit() and len(base) in (4, 5):
            return f"{base}.{suffix.upper()}"
        if suffix.upper() in ('KS', 'KQ') and base.isdigit() and len(base) == 6:
            return f"{base}.{suffix.upper()}"
        if suffix.upper() in ('TW', 'TWO') and base.isdigit() and 4 <= len(base) <= 6:
            return f"{base}.{suffix.upper()}"
        if suffix.upper() == 'HK' and base.isdigit() and 1 <= len(base) <= 5:
            return f"HK{base.zfill(5)}"
        if base.upper() in ('SH', 'SS', 'SZ', 'BJ') and suffix.isdigit():
            return suffix
        if suffix.upper() in ('SH', 'SZ', 'SS', 'BJ') and base.isdigit():
            return base

    return code

ETF_PREFIXES = ("51", "52", "56", "58", "15", "16", "18")

def _is_us_market(code: str) -> bool:
    """判断是否为美股/美股指数代码（不含中文前后缀）。"""
    from .us_index_mapping import is_us_stock_code, is_us_index_code

    normalized = (code or "").strip().upper()
    return is_us_index_code(normalized) or is_us_stock_code(normalized)

def _is_hk_market(code: str) -> bool:
    """
    判定是否为港股代码。

    支持 ``.HK`` 后缀、``HK00700`` 前缀形式，以及 4-5 位纯数字裸码
    （A 股 ETF/股票为 6 位，与港股 4-5 位裸数字不冲突）。``YfinanceFetcher``
    与 ``AkshareFetcher`` / ``LongbridgeFetcher`` 的 ``_is_hk_code`` 与本
    函数对裸港股码的位数范围保持一致。
    """
    normalized = (code or "").strip().upper()
    if normalized.endswith(".HK"):
        base = normalized[:-3]
        return base.isdigit() and 1 <= len(base) <= 5
    if normalized.startswith("HK"):
        digits = normalized[2:]
        return digits.isdigit() and 1 <= len(digits) <= 5
    if normalized.isdigit() and 4 <= len(normalized) <= 5:
        return True
    return False

def _is_jp_market(code: str) -> bool:
    """判定是否为日本 Yahoo Finance suffix 代码（如 7203.T）。"""
    return is_suffix_market_symbol(code, "jp")

def _is_kr_market(code: str) -> bool:
    """判定是否为韩国 Yahoo Finance suffix 代码（如 005930.KS / 035720.KQ）。"""
    return is_suffix_market_symbol(code, "kr")

def _is_tw_market(code: str) -> bool:
    """判定是否为台湾 Yahoo Finance suffix 代码（TWSE 上市 2330.TW / TPEx 上柜 6505.TWO）。

    台股 base 为 4-6 位（普通股 4 位，ETF/其他至 6 位，如 00878 / 006208）。
    仅带 .TW/.TWO 后缀的代码才识别为台股，裸 6 位代码仍按 A 股语义处理。
    """
    return is_suffix_market_symbol(code, "tw")

def _is_etf_code(code: str) -> bool:
    """判定 A 股 ETF 基金代码（保守规则）。"""
    normalized = normalize_stock_code(code)
    return (
        normalized.isdigit()
        and len(normalized) == 6
        and normalized.startswith(ETF_PREFIXES)
    )

def _coerce_chip_metric(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        numeric = float(value)
        if np.isnan(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None

def _is_meaningful_chip_distribution(chip: Any) -> bool:
    """Validate that a provider returned usable core chip metrics."""
    if chip is None:
        return False
    avg_cost = _coerce_chip_metric(getattr(chip, "avg_cost", None))
    concentration_90 = _coerce_chip_metric(getattr(chip, "concentration_90", None))
    concentration_70 = _coerce_chip_metric(getattr(chip, "concentration_70", None))
    return (
        avg_cost is not None
        and avg_cost > 0
        and (
            (concentration_90 is not None and concentration_90 >= 0)
            or (concentration_70 is not None and concentration_70 >= 0)
        )
    )

def _market_tag(code: str) -> str:
    """返回市场标签: cn/us/hk/jp/kr/tw."""
    if _is_us_market(code):
        return "us"
    if _is_hk_market(code):
        return "hk"
    if _is_jp_market(code):
        return "jp"
    if _is_kr_market(code):
        return "kr"
    if _is_tw_market(code):
        return "tw"
    return "cn"
