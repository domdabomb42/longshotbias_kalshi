"""Data ingestion and caching for Kalshi markets."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .kalshi_client import KalshiClient
from .config import SETTINGS
from .utils import (
    ensure_dir,
    parse_timestamp,
    pick_first,
    read_jsonl,
    write_jsonl,
    progress_bar,
    progress_iter,
    iter_jsonl,
)

logger = logging.getLogger(__name__)


def _iter_candle_windows(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    chunk_hours: int | None,
) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    if not chunk_hours or chunk_hours <= 0:
        yield start_ts, end_ts
        return
    delta = timedelta(hours=chunk_hours)
    current = start_ts
    while current < end_ts:
        window_end = min(current + delta, end_ts)
        yield current, window_end
        current = window_end


def fetch_cutoff(client: KalshiClient) -> pd.Timestamp | None:
    payload = client.get("/historical/cutoff")
    cutoff_val = pick_first(payload, ["cutoff_ts", "cutoff_time", "cutoff"])
    return parse_timestamp(cutoff_val)


def _market_close_ts(market: Dict[str, Any]) -> pd.Timestamp | None:
    return parse_timestamp(
        pick_first(
            market,
            [
                "close_time",
                "close_ts",
                "close_time_utc",
                "settlement_time",
                "settled_time",
                "end_time",
                "end_ts",
            ],
        )
    )


def download_historical(
    client: KalshiClient,
    start: datetime,
    end: datetime,
    cache_dir: str,
    include_candles: bool = True,
    candle_interval_sec: int = 60,
    candle_window_days: int | None = None,
    candle_chunk_hours: int | None = None,
    horizons_days: Iterable[float] | None = None,
    max_markets: int | None = None,
    all_history: bool = False,
    resume: bool = False,
    newest_first: bool = True,
    force_candles: bool = False,
) -> List[Dict[str, Any]]:
    """Download settled markets between start and end and cache raw responses."""
    raw_dir = ensure_dir(cache_dir)
    markets_path = raw_dir / "historical_markets.jsonl"
    details_path = raw_dir / "historical_market_details.jsonl"
    candles_path = raw_dir / "historical_candlesticks.jsonl"

    if all_history:
        return _download_historical_all(
            client=client,
            start=start,
            end=end,
            markets_path=markets_path,
            details_path=details_path,
            candles_path=candles_path,
            include_candles=include_candles,
            candle_interval_sec=candle_interval_sec,
            candle_window_days=candle_window_days,
            candle_chunk_hours=candle_chunk_hours,
            horizons_days=horizons_days,
            max_markets=max_markets,
            resume=resume,
            newest_first=newest_first,
            force_candles=force_candles,
        )

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    logger.info("Downloading historical markets from %s to %s", start_ts, end_ts)
    collected: List[Dict[str, Any]] = []
    checkpoint_path = markets_path.with_suffix(".cursor")
    existing_tickers = set()
    if resume and markets_path.exists():
        for rec in iter_jsonl(markets_path):
            ticker = rec.get("ticker") or rec.get("market_ticker")
            if ticker:
                existing_tickers.add(ticker)

    bar = progress_bar(desc="historical markets", unit="mkts")
    last_date = None
    order_hint = None
    last_close_ts = None
    seen_in_range = False
    stop_early = False
    cursor = None
    if resume and checkpoint_path.exists():
        cursor = checkpoint_path.read_text(encoding="utf-8").strip() or None
    completed = False
    while True:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        payload = client.get("/historical/markets", params=params)
        batch = client._extract_items(payload, "markets")
        bar.update(len(batch))
        if not batch:
            completed = True
            break
        for market in batch:
            close_ts = _market_close_ts(market)
            if close_ts is None:
                continue
            ticker = market.get("ticker") or market.get("market_ticker")
            if ticker and ticker in existing_tickers:
                continue
            if last_close_ts is not None and order_hint is None:
                order_hint = "desc" if close_ts <= last_close_ts else "asc"
            last_close_ts = close_ts
            if last_date != close_ts.date():
                last_date = close_ts.date()
                if hasattr(bar, "set_postfix_str"):
                    bar.set_postfix_str(str(last_date))
                else:
                    logger.info("Processing close date: %s", last_date)
            in_range = start_ts <= close_ts <= end_ts
            if not in_range:
                if newest_first and order_hint == "desc" and seen_in_range and close_ts < start_ts:
                    stop_early = True
                    break
                if newest_first and order_hint == "asc" and seen_in_range and close_ts > end_ts:
                    stop_early = True
                    break
                continue
            collected.append(market)
            if ticker:
                existing_tickers.add(ticker)
            seen_in_range = True
            if max_markets and len(collected) >= max_markets:
                break
        next_cursor = (
            payload.get("next_cursor")
            or payload.get("cursor")
            or payload.get("next")
            or payload.get("pagination", {}).get("next_cursor")
        )
        if max_markets and len(collected) >= max_markets:
            if next_cursor:
                checkpoint_path.write_text(str(next_cursor), encoding="utf-8")
            break
        if stop_early:
            completed = True
            break
        if next_cursor:
            checkpoint_path.write_text(str(next_cursor), encoding="utf-8")
            cursor = next_cursor
        else:
            completed = True
            break
    bar.close()

    if completed and resume:
        try:
            checkpoint_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass

    if newest_first:
        collected.sort(key=_market_close_ts, reverse=True)

    write_jsonl(markets_path, collected, append=resume)

    details_records: List[Dict[str, Any]] = []
    candles_records: List[Dict[str, Any]] = []

    details_existing = set()
    if resume and details_path.exists():
        for rec in iter_jsonl(details_path):
            ticker = rec.get("ticker") or rec.get("market_ticker")
            if ticker:
                details_existing.add(ticker)
    candles_existing = set()
    if resume and candles_path.exists() and not force_candles:
        for rec in iter_jsonl(candles_path):
            ticker = rec.get("ticker") or rec.get("market_ticker")
            if ticker:
                candles_existing.add(ticker)

    for idx, market in enumerate(progress_iter(collected, total=len(collected), desc="details+candles", unit="mkt"), 1):
        ticker = market.get("ticker") or market.get("market_ticker")
        if not ticker:
            continue
        detail = None
        if ticker not in details_existing:
            try:
                detail = client.get(f"/historical/markets/{ticker}")
            except Exception as exc:
                logger.warning("Failed to fetch details for %s: %s", ticker, exc)
                continue
            detail["ticker"] = ticker
            details_records.append(detail)
            details_existing.add(ticker)

        if include_candles:
            close_ts = _market_close_ts(detail) if detail else _market_close_ts(market)
            if close_ts is None:
                continue
            if ticker in candles_existing:
                continue
            horizon_list = list(horizons_days or [0])
            max_horizon = max(horizon_list) if horizon_list else 0
            window_days = candle_window_days if candle_window_days is not None else (max_horizon + 1)
            start_candle = close_ts - timedelta(days=max(window_days, max_horizon + 1))
            for window_start, window_end in _iter_candle_windows(start_candle, close_ts, candle_chunk_hours):
                params = {
                    "period_interval": candle_interval_sec,
                    "start_ts": int(window_start.timestamp()),
                    "end_ts": int(window_end.timestamp()),
                }
                try:
                    candles = client.get(f"/historical/markets/{ticker}/candlesticks", params=params)
                    candles["ticker"] = ticker
                    candles["close_ts"] = int(close_ts.timestamp())
                    candles["window_start_ts"] = int(window_start.timestamp())
                    candles["window_end_ts"] = int(window_end.timestamp())
                    candles_records.append(candles)
                    candles_existing.add(ticker)
                except Exception as exc:
                    logger.warning("Failed to fetch candlesticks for %s: %s", ticker, exc)

    write_jsonl(details_path, details_records, append=resume)
    write_jsonl(candles_path, candles_records, append=resume and not force_candles)

    return collected


def _download_historical_all(
    client: KalshiClient,
    start: datetime,
    end: datetime,
    markets_path: Path,
    details_path: Path,
    candles_path: Path,
    include_candles: bool,
    candle_interval_sec: int,
    candle_window_days: int | None,
    candle_chunk_hours: int | None,
    horizons_days: Iterable[float] | None,
    max_markets: int | None,
    resume: bool,
    newest_first: bool,
    force_candles: bool,
) -> List[Dict[str, Any]]:
    """Download all historical markets with streaming writes and optional resume."""
    checkpoint_path = markets_path.with_suffix(".cursor")
    existing_tickers = set()
    if resume and markets_path.exists():
        for rec in iter_jsonl(markets_path):
            ticker = rec.get("ticker") or rec.get("market_ticker")
            if ticker:
                existing_tickers.add(ticker)

    mode = "a" if resume else "w"
    if not resume:
        checkpoint_path.unlink(missing_ok=True)  # type: ignore[attr-defined]

    candle_mode = "w" if force_candles else mode
    with markets_path.open(mode, encoding="utf-8") as m_f, details_path.open(mode, encoding="utf-8") as d_f, candles_path.open(candle_mode, encoding="utf-8") as c_f:
        cursor = None
        if resume and checkpoint_path.exists():
            cursor = checkpoint_path.read_text(encoding="utf-8").strip() or None

        bar = progress_bar(desc="historical markets (all)", unit="mkts")
        last_date = None
        processed = 0
        order_hint = None
        last_close_ts = None
        seen_in_range = False
        start_ts = pd.Timestamp(start, tz="UTC") if start else None
        end_ts = pd.Timestamp(end, tz="UTC") if end else None
        stop_early = False
        while True:
            params = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            payload = client.get("/historical/markets", params=params)
            batch = client._extract_items(payload, "markets")
            bar.update(len(batch))
            if not batch:
                break
            for market in batch:
                ticker = market.get("ticker") or market.get("market_ticker")
                if not ticker or ticker in existing_tickers:
                    continue
                close_ts = _market_close_ts(market)
                if close_ts is not None and last_close_ts is not None and order_hint is None:
                    order_hint = "desc" if close_ts <= last_close_ts else "asc"
                if close_ts is not None:
                    last_close_ts = close_ts
                if close_ts is not None and last_date != close_ts.date():
                    last_date = close_ts.date()
                    if hasattr(bar, "set_postfix_str"):
                        bar.set_postfix_str(str(last_date))
                    else:
                        logger.info("Processing close date: %s", last_date)

                in_range = True
                if close_ts is not None and start_ts is not None and close_ts < start_ts:
                    in_range = False
                if close_ts is not None and end_ts is not None and close_ts > end_ts:
                    in_range = False
                if not in_range:
                    if newest_first and order_hint == "desc" and seen_in_range and close_ts is not None and close_ts < start_ts:
                        stop_early = True
                        break
                    if newest_first and order_hint == "asc" and seen_in_range and close_ts is not None and close_ts > end_ts:
                        stop_early = True
                        break
                    continue

                m_f.write(json.dumps(market))
                m_f.write("\n")
                if resume:
                    existing_tickers.add(ticker)
                processed += 1
                seen_in_range = True
                if max_markets and processed >= max_markets:
                    break

                # details
                detail = None
                try:
                    detail = client.get(f"/historical/markets/{ticker}")
                    detail["ticker"] = ticker
                    d_f.write(json.dumps(detail))
                    d_f.write("\n")
                except Exception as exc:
                    logger.warning("Failed to fetch details for %s: %s", ticker, exc)

                # candles
                if include_candles:
                    close_ts = _market_close_ts(detail) if detail else _market_close_ts(market)
                    if close_ts is not None:
                        horizon_list = list(horizons_days or [0])
                        max_horizon = max(horizon_list) if horizon_list else 0
                        window_days = candle_window_days if candle_window_days is not None else (max_horizon + 1)
                        start_candle = close_ts - timedelta(days=max(window_days, max_horizon + 1))
                        for window_start, window_end in _iter_candle_windows(start_candle, close_ts, candle_chunk_hours):
                            params = {
                                "period_interval": candle_interval_sec,
                                "start_ts": int(window_start.timestamp()),
                                "end_ts": int(window_end.timestamp()),
                            }
                            try:
                                candles = client.get(f"/historical/markets/{ticker}/candlesticks", params=params)
                                candles["ticker"] = ticker
                                candles["close_ts"] = int(close_ts.timestamp())
                                candles["window_start_ts"] = int(window_start.timestamp())
                                candles["window_end_ts"] = int(window_end.timestamp())
                                c_f.write(json.dumps(candles))
                                c_f.write("\n")
                            except Exception as exc:
                                logger.warning("Failed to fetch candlesticks for %s: %s", ticker, exc)

            if max_markets and processed >= max_markets:
                break
            if stop_early:
                break

            cursor = (
                payload.get("next_cursor")
                or payload.get("cursor")
                or payload.get("next")
                or payload.get("pagination", {}).get("next_cursor")
            )
            if cursor:
                checkpoint_path.write_text(str(cursor), encoding="utf-8")
            else:
                checkpoint_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
                break
        bar.close()

    return []


def download_live_markets(
    client: KalshiClient,
    cache_dir: str,
    status_filter: Iterable[str] | None = None,
    max_markets: int | None = None,
) -> List[Dict[str, Any]]:
    raw_dir = ensure_dir(cache_dir)
    markets_path = raw_dir / "live_markets.jsonl"

    status_filter = {s.lower() for s in (status_filter or ["open", "active"])}
    api_status = None
    if len(status_filter) == 1:
        candidate = next(iter(status_filter))
        # Kalshi API rejects some status values (e.g., "active"); only pass known-safe ones.
        if candidate in {"open", "closed", "resolved"}:
            api_status = candidate
    collected: List[Dict[str, Any]] = []
    bar = progress_bar(desc="live markets", unit="mkts")
    def _paginate_with_params(params: Dict[str, Any]) -> None:
        for batch in client.paginate("/markets", params=params, data_key="markets"):
            bar.update(len(batch))
            for market in batch:
                status = str(market.get("status") or market.get("state") or "").lower()
                if not status:
                    continue
                if status_filter and status not in status_filter:
                    continue
                collected.append(market)
                if max_markets and len(collected) >= max_markets:
                    break
            if max_markets and len(collected) >= max_markets:
                break

    params = {"limit": SETTINGS.live_page_limit}
    if api_status:
        params["status"] = api_status
    try:
        _paginate_with_params(params)
    except Exception as exc:
        msg = str(exc)
        if "invalid status filter" in msg and "status" in params:
            logger.warning("Status filter rejected by API; retrying without status param.")
            params.pop("status", None)
            try:
                _paginate_with_params(params)
            except Exception as exc2:
                logger.warning("Live market fetch interrupted: %s", exc2)
        else:
            logger.warning("Live market fetch interrupted: %s", exc)
    bar.close()

    logger.info("Live markets collected: %s", len(collected))

    write_jsonl(markets_path, collected, append=False)
    return collected


def download_orderbooks(
    client: KalshiClient,
    tickers: Iterable[str],
    cache_dir: str,
    max_markets: int | None = None,
) -> List[Dict[str, Any]]:
    raw_dir = ensure_dir(cache_dir)
    orderbooks_path = raw_dir / "live_orderbooks.jsonl"

    tickers_list = list(tickers)
    if max_markets:
        tickers_list = tickers_list[:max_markets]

    records: List[Dict[str, Any]] = []
    for idx, ticker in enumerate(progress_iter(tickers_list, total=len(tickers_list), desc="orderbooks", unit="mkt"), 1):
        try:
            ob = client.get(f"/markets/{ticker}/orderbook")
            ob["ticker"] = ticker
            records.append(ob)
        except Exception as exc:
            logger.warning("Failed to fetch orderbook for %s: %s", ticker, exc)
            continue

    write_jsonl(orderbooks_path, records, append=False)
    return records


def load_raw(path: str | Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def backfill_candles_from_cache(
    client: KalshiClient,
    cache_dir: str,
    start: datetime,
    end: datetime,
    candle_interval_sec: int,
    candle_window_days: int | None,
    candle_chunk_hours: int | None,
    horizons_days: Iterable[float] | None,
    force_candles: bool,
    resume: bool,
    max_markets: int | None = None,
) -> None:
    raw_dir = ensure_dir(cache_dir)
    markets_path = raw_dir / "historical_markets.jsonl"
    details_path = raw_dir / "historical_market_details.jsonl"
    candles_path = raw_dir / "historical_candlesticks.jsonl"

    if not markets_path.exists():
        raise FileNotFoundError("historical_markets.jsonl not found for candle backfill.")

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    details_existing = set()
    if resume and details_path.exists():
        for rec in iter_jsonl(details_path):
            ticker = rec.get("ticker") or rec.get("market_ticker")
            if ticker:
                details_existing.add(ticker)

    candles_existing = set()
    if resume and candles_path.exists() and not force_candles:
        for rec in iter_jsonl(candles_path):
            ticker = rec.get("ticker") or rec.get("market_ticker")
            if ticker:
                candles_existing.add(ticker)

    candle_mode = "w" if force_candles else ("a" if resume else "w")
    processed = 0
    bar = progress_bar(desc="candles backfill", unit="mkts")

    # Load and sort markets by close date (most recent first)
    markets_in_range: List[Dict[str, Any]] = []
    load_bar = progress_bar(desc="loading markets", unit="mkts")
    for market in iter_jsonl(markets_path):
        load_bar.update(1)
        close_ts = _market_close_ts(market)
        if close_ts is None:
            continue
        if not (start_ts <= close_ts <= end_ts):
            continue
        market["_close_ts"] = close_ts
        markets_in_range.append(market)
    load_bar.close()
    if hasattr(bar, "set_postfix_str"):
        bar.set_postfix_str("sorting")
    else:
        logger.info("Sorting markets by close date (most recent first).")
    markets_in_range.sort(key=lambda m: m.get("_close_ts"), reverse=True)
    if markets_in_range:
        first_date = markets_in_range[0].get("_close_ts")
        if first_date is not None:
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(f"first={first_date.date()}")
            else:
                logger.info("First market close date in backfill: %s", first_date.date())

    with details_path.open("a", encoding="utf-8") as d_f, candles_path.open(candle_mode, encoding="utf-8") as c_f:
        for market in markets_in_range:
            if max_markets and processed >= max_markets:
                break
            close_ts = market.get("_close_ts") or _market_close_ts(market)
            if close_ts is None:
                continue
            ticker = market.get("ticker") or market.get("market_ticker")
            if not ticker:
                continue
            bar.update(1)
            processed += 1
            if processed % 1000 == 0:
                if hasattr(bar, "set_postfix_str"):
                    bar.set_postfix_str(str(close_ts.date()))
                else:
                    logger.info("Backfill progress at %s markets; current close date: %s", processed, close_ts.date())

            detail = None
            if ticker not in details_existing:
                try:
                    detail = client.get(f"/historical/markets/{ticker}")
                    detail["ticker"] = ticker
                    d_f.write(json.dumps(detail))
                    d_f.write("\n")
                    details_existing.add(ticker)
                except Exception as exc:
                    logger.warning("Failed to fetch details for %s: %s", ticker, exc)

            if ticker in candles_existing and not force_candles:
                continue

            close_ts = _market_close_ts(detail) if detail else close_ts
            horizon_list = list(horizons_days or [0])
            max_horizon = max(horizon_list) if horizon_list else 0
            window_days = candle_window_days if candle_window_days is not None else (max_horizon + 1)
            start_candle = close_ts - timedelta(days=max(window_days, max_horizon + 1))

            for window_start, window_end in _iter_candle_windows(start_candle, close_ts, candle_chunk_hours):
                params = {
                    "period_interval": candle_interval_sec,
                    "start_ts": int(window_start.timestamp()),
                    "end_ts": int(window_end.timestamp()),
                }
                try:
                    candles = client.get(f"/historical/markets/{ticker}/candlesticks", params=params)
                    candles["ticker"] = ticker
                    candles["close_ts"] = int(close_ts.timestamp())
                    candles["window_start_ts"] = int(window_start.timestamp())
                    candles["window_end_ts"] = int(window_end.timestamp())
                    c_f.write(json.dumps(candles))
                    c_f.write("\n")
                except Exception as exc:
                    logger.warning("Failed to fetch candlesticks for %s: %s", ticker, exc)

            if resume:
                candles_existing.add(ticker)
    bar.close()


def backfill_candles_from_api(
    client: KalshiClient,
    cache_dir: str,
    start: datetime,
    end: datetime,
    candle_interval_sec: int,
    candle_window_days: int | None,
    candle_chunk_hours: int | None,
    horizons_days: Iterable[float] | None,
    force_candles: bool,
    resume: bool,
    max_markets: int | None = None,
) -> None:
    """Backfill candles by streaming markets directly from the API (no preload)."""
    raw_dir = ensure_dir(cache_dir)
    markets_path = raw_dir / "historical_markets.jsonl"
    details_path = raw_dir / "historical_market_details.jsonl"
    candles_path = raw_dir / "historical_candlesticks.jsonl"
    cursor_path = raw_dir / "historical_candles_backfill.cursor"

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    cutoff_ts = fetch_cutoff(client)
    if cutoff_ts is not None and end_ts > cutoff_ts:
        logger.warning(
            "Historical cutoff is %s; requested end %s is newer and not yet available in /historical.",
            cutoff_ts.date(),
            end_ts.date(),
        )

    markets_existing = set()
    if resume and markets_path.exists():
        for rec in iter_jsonl(markets_path):
            ticker = rec.get("ticker") or rec.get("market_ticker")
            if ticker:
                markets_existing.add(ticker)

    details_existing = set()
    if resume and details_path.exists():
        for rec in iter_jsonl(details_path):
            ticker = rec.get("ticker") or rec.get("market_ticker")
            if ticker:
                details_existing.add(ticker)

    candles_existing = set()
    if resume and candles_path.exists():
        for rec in iter_jsonl(candles_path):
            ticker = rec.get("ticker") or rec.get("market_ticker")
            if ticker:
                candles_existing.add(ticker)

    if force_candles and not resume:
        candle_mode = "w"
    else:
        candle_mode = "a" if resume else "w"
    market_mode = "a" if resume else "w"

    cursor = None
    if resume and cursor_path.exists():
        cursor = cursor_path.read_text(encoding="utf-8").strip() or None

    try:
        from .config import SETTINGS

        total_est = max(SETTINGS.backfill_total_estimate, len(candles_existing))
    except Exception:
        total_est = max(129000, len(candles_existing))

    initial_done = len(candles_existing) if resume else 0
    processed = initial_done
    bar = progress_bar(desc="candles backfill (api)", unit="mkts", total=total_est, initial=initial_done)

    with markets_path.open(market_mode, encoding="utf-8") as m_f, details_path.open("a", encoding="utf-8") as d_f, candles_path.open(candle_mode, encoding="utf-8") as c_f:
        order_hint = None
        last_close_ts = None
        seen_in_range = False
        stop_early = False
        while True:
            params = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            payload = client.get("/historical/markets", params=params)
            batch = client._extract_items(payload, "markets")
            if not batch:
                break

            for market in batch:
                close_ts = _market_close_ts(market)
                if close_ts is None:
                    continue
                if last_close_ts is not None and order_hint is None:
                    order_hint = "desc" if close_ts <= last_close_ts else "asc"
                last_close_ts = close_ts

                in_range = start_ts <= close_ts <= end_ts
                if not in_range:
                    if order_hint == "desc" and seen_in_range and close_ts < start_ts:
                        stop_early = True
                        break
                    if order_hint == "asc" and seen_in_range and close_ts > end_ts:
                        stop_early = True
                        break
                    continue

                seen_in_range = True
                ticker = market.get("ticker") or market.get("market_ticker")
                if not ticker:
                    continue

                if resume and ticker in candles_existing:
                    continue

                if resume and ticker in markets_existing:
                    pass
                else:
                    m_f.write(json.dumps(market))
                    m_f.write("\n")
                    markets_existing.add(ticker)

                detail = None
                if ticker not in details_existing:
                    try:
                        detail = client.get(f"/historical/markets/{ticker}")
                        detail["ticker"] = ticker
                        d_f.write(json.dumps(detail))
                        d_f.write("\n")
                        details_existing.add(ticker)
                    except Exception as exc:
                        logger.warning("Failed to fetch details for %s: %s", ticker, exc)

                detail_close = _market_close_ts(detail) if detail else None
                if detail_close is not None:
                    close_ts = detail_close

                horizon_list = list(horizons_days or [0])
                max_horizon = max(horizon_list) if horizon_list else 0
                window_days = candle_window_days if candle_window_days is not None else (max_horizon + 1)
                start_candle = close_ts - timedelta(days=max(window_days, max_horizon + 1))

                for window_start, window_end in _iter_candle_windows(start_candle, close_ts, candle_chunk_hours):
                    params = {
                        "period_interval": candle_interval_sec,
                        "start_ts": int(window_start.timestamp()),
                        "end_ts": int(window_end.timestamp()),
                    }
                    try:
                        candles = client.get(f"/historical/markets/{ticker}/candlesticks", params=params)
                        candles["ticker"] = ticker
                        candles["close_ts"] = int(close_ts.timestamp())
                        candles["window_start_ts"] = int(window_start.timestamp())
                        candles["window_end_ts"] = int(window_end.timestamp())
                        c_f.write(json.dumps(candles))
                        c_f.write("\n")
                    except Exception as exc:
                        logger.warning("Failed to fetch candlesticks for %s: %s", ticker, exc)

                processed += 1
                bar.update(1)
                if processed % 1000 == 0:
                    if hasattr(bar, "set_postfix_str"):
                        bar.set_postfix_str(str(close_ts.date()))
                    else:
                        logger.info("Backfill progress at %s markets; current close date: %s", processed, close_ts.date())

                if resume:
                    candles_existing.add(ticker)

            next_cursor = (
                payload.get("next_cursor")
                or payload.get("cursor")
                or payload.get("next")
                or payload.get("pagination", {}).get("next_cursor")
            )
            if max_markets and (processed - initial_done) >= max_markets:
                if next_cursor:
                    cursor_path.write_text(str(next_cursor), encoding="utf-8")
                break
            if stop_early:
                break
            if not next_cursor:
                break
            cursor_path.write_text(str(next_cursor), encoding="utf-8")
            cursor = next_cursor

    bar.close()
    try:
        cursor_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
    except Exception:
        pass
