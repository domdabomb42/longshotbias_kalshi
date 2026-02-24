"""Utility helpers for data normalization and IO."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

logger = logging.getLogger(__name__)


class _NullProgress:
    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def progress_bar(
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
    initial: int = 0,
) -> Any:
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc=desc, unit=unit, initial=initial)
    except Exception:
        return _NullProgress()


def progress_iter(iterable: Iterable[Any], total: int | None = None, desc: str | None = None, unit: str = "it") -> Iterable[Any]:
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc, unit=unit)
    except Exception:
        return iterable


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]], append: bool = True) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    def _iter():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    return _iter()


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # assume seconds if too small, ms if too large
        if value > 1e12:
            return pd.to_datetime(value, unit="ms", utc=True)
        return pd.to_datetime(value, unit="s", utc=True)
    if isinstance(value, str):
        try:
            return pd.to_datetime(value, utc=True)
        except Exception:
            return None
    if isinstance(value, datetime):
        return pd.Timestamp(value, tz=timezone.utc)
    return None


def to_dollars(price: Any) -> float | None:
    if price is None:
        return None
    try:
        val = float(price)
    except (TypeError, ValueError):
        return None
    if val < 0:
        return None
    if val > 1.5:
        return val / 100.0
    return val


def ceil_to_cent(value: float) -> float:
    return (int(value * 100 + 0.9999)) / 100.0


def outcome_to_bool(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return float(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in ("yes", "y", "true", "won", "1"):
            return 1.0
        if val in ("no", "n", "false", "lost", "0"):
            return 0.0
    return None


def pick_first(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def best_bid(orderbook: Dict[str, Any]) -> float | None:
    # Kalshi orderbook lists bids as [price, size] or dicts
    if not orderbook:
        return None
    bids = orderbook.get("bids") or orderbook.get("orders") or orderbook.get("yes_bids") or orderbook.get("no_bids")
    if not bids:
        return None
    first = bids[0]
    if isinstance(first, dict):
        price = first.get("price")
    else:
        price = first[0] if isinstance(first, (list, tuple)) and first else None
    return to_dollars(price)
