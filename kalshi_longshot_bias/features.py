"""Feature engineering and market classification."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

import pandas as pd

from .utils import outcome_to_bool, parse_timestamp, pick_first, progress_iter, to_dollars

logger = logging.getLogger(__name__)


PRICE_BIN_LABELS = [
    "01-10",
    "11-20",
    "21-30",
    "31-40",
    "41-50",
    "51-60",
    "61-70",
    "71-80",
    "81-90",
    "91-99",
]


def normalize_market_row(market: Dict[str, Any]) -> Dict[str, Any]:
    ticker = market.get("ticker") or market.get("market_ticker")
    title = market.get("title") or market.get("market_title") or ""
    subtitle = market.get("subtitle") or market.get("yes_subtitle") or market.get("market_subtitle") or ""
    event_ticker = market.get("event_ticker") or market.get("event_id") or market.get("event")
    series_ticker = market.get("series_ticker") or market.get("series_id") or market.get("series")
    category = market.get("category") or market.get("event_category") or market.get("series_category")

    close_ts = parse_timestamp(
        pick_first(
            market,
            ["close_time", "close_ts", "close_time_utc", "settlement_time", "end_time", "end_ts"],
        )
    )
    volume = pick_first(market, ["volume", "volume_total", "volume_24h", "volume24h"])
    open_interest = pick_first(market, ["open_interest", "open_int", "oi"])

    yes_bid = to_dollars(pick_first(market, ["yes_bid", "best_yes_bid", "best_bid"]))
    yes_ask = to_dollars(pick_first(market, ["yes_ask", "best_yes_ask", "best_ask"]))
    no_bid = to_dollars(pick_first(market, ["no_bid", "best_no_bid"]))
    no_ask = to_dollars(pick_first(market, ["no_ask", "best_no_ask"]))
    last_price = to_dollars(pick_first(market, ["last_price", "last_yes", "last_yes_price"]))

    outcome = outcome_to_bool(pick_first(market, ["outcome", "result", "winning_outcome", "settlement"]))

    return {
        "ticker": ticker,
        "title": title,
        "subtitle": subtitle,
        "event_ticker": event_ticker,
        "series_ticker": series_ticker,
        "category": category,
        "close_ts": close_ts,
        "volume": volume,
        "open_interest": open_interest,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "last_price": last_price,
        "outcome": outcome,
    }


def _extract_candle_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("candlesticks", "candles", "data", "results"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]
    for value in payload.values():
        if isinstance(value, list):
            return value
    return []


def _extract_candle_price(candle: Dict[str, Any]) -> float | None:
    if "price" in candle and isinstance(candle["price"], dict):
        close_val = candle["price"].get("close") or candle["price"].get("mean")
        if close_val is not None:
            return to_dollars(close_val)
    if "yes_bid" in candle and isinstance(candle["yes_bid"], dict):
        bid_close = to_dollars(candle["yes_bid"].get("close"))
    else:
        bid_close = None
    if "yes_ask" in candle and isinstance(candle["yes_ask"], dict):
        ask_close = to_dollars(candle["yes_ask"].get("close"))
    else:
        ask_close = None
    if bid_close is not None and ask_close is not None:
        return (bid_close + ask_close) / 2
    if bid_close is not None:
        return bid_close
    if ask_close is not None:
        return ask_close
    for key in ("close", "price", "yes_price", "yes", "yes_close", "c"):
        if key in candle:
            return to_dollars(candle[key])
    return None


def build_candles_df(candles_payloads: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for payload in candles_payloads:
        ticker = payload.get("ticker")
        candles = _extract_candle_list(payload)
        for candle in candles:
            ts = parse_timestamp(
                candle.get("end_period_ts")
                or candle.get("ts")
                or candle.get("timestamp")
                or candle.get("time")
            )
            price = _extract_candle_price(candle)
            if ticker and ts is not None and price is not None:
                records.append({"ticker": ticker, "ts": ts, "price": price})
    if not records:
        return pd.DataFrame(columns=["ticker", "ts", "price"])
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["ticker", "ts"], keep="last")
    return df.sort_values(["ticker", "ts"]).reset_index(drop=True)


def price_bin(prob: float | None) -> str | None:
    if prob is None or pd.isna(prob):
        return None
    cents = int(round(prob * 100))
    if cents <= 0:
        return "00"
    if cents >= 100:
        return "100"
    idx = min((cents - 1) // 10, 9)
    return PRICE_BIN_LABELS[idx]


def classify_structure(markets: pd.DataFrame) -> pd.Series:
    labels: Dict[str, str] = {}
    by_event = markets.groupby("event_ticker", dropna=False)
    for _, group in by_event:
        tickers = group["ticker"].tolist()
        if len(group) <= 1:
            for ticker in tickers:
                labels[ticker] = "single"
            continue

        titles = (group["title"].fillna("") + " " + group["subtitle"].fillna("")).str.lower()
        threshold_hits = titles.str.contains(r"(?:at least|more than|less than|under|over)")
        threshold_hits = threshold_hits | titles.str.contains(r"[<>]=?")
        range_hits = titles.str.contains(r"\bbetween\b|\d+\s*(?:-|to)\s*\d+")

        threshold_ratio = threshold_hits.mean() if len(titles) else 0.0
        range_ratio = range_hits.mean() if len(titles) else 0.0

        if threshold_ratio >= 0.6:
            label = "ladder"
        elif range_ratio >= 0.6:
            label = "numeric_buckets"
        else:
            label = "mutual_other"

        for ticker in tickers:
            labels[ticker] = label

    return markets["ticker"].map(labels).fillna("unknown")


def _normalize_category_value(value: str) -> str | None:
    val = value.strip().lower()
    if not val:
        return None
    aliases = {
        "politics": ["politic", "election", "government"],
        "economics": ["econom", "macro", "cpi", "gdp", "unemployment", "jobs"],
        "weather_climate": ["weather", "climate", "hurricane", "storm"],
        "sports": ["sport", "nfl", "nba", "mlb", "nhl", "soccer", "match"],
        "financials": ["finance", "financial", "stocks", "rates", "bonds", "markets"],
        "culture": ["culture", "entertainment", "awards", "movie", "music", "celebrity"],
        "crypto": ["crypto", "bitcoin", "btc", "eth", "ethereum"],
        "technology": ["tech", "ai", "software", "hardware", "chip", "semiconductor"],
        "mentions": ["mentions", "mention", "trending", "trend", "google trends", "twitter", "x.com"],
    }
    for key, keys in aliases.items():
        if any(k in val for k in keys):
            return key
    return None


def map_category(row: pd.Series) -> str:
    category = row.get("category")
    if isinstance(category, str) and category.strip():
        normalized = _normalize_category_value(category)
        if normalized:
            return normalized
        return category.strip().lower()

    series = str(row.get("series_ticker") or "").lower()
    title = str(row.get("title") or "").lower()
    subtitle = str(row.get("subtitle") or "").lower()
    combo = f"{series} {title} {subtitle}"

    mapping = {
        "politics": ["pres", "election", "senate", "house", "governor", "poll", "primary", "ballot"],
        "economics": ["cpi", "gdp", "jobs", "unemployment", "fed", "rate", "inflation", "ppi"],
        "weather_climate": ["hurricane", "storm", "climate", "temperature", "rainfall", "snow", "tornado"],
        "sports": ["nfl", "nba", "mlb", "nhl", "soccer", "match", "tournament", "olympics"],
        "financials": ["spx", "nasdaq", "dow", "rates", "yield", "treasury", "stock", "earnings"],
        "culture": ["oscar", "grammy", "emmy", "box office", "album", "movie", "celebrity", "tv show"],
        "crypto": ["btc", "bitcoin", "eth", "ethereum", "crypto", "solana"],
        "technology": ["ai", "artificial intelligence", "chip", "semiconductor", "software", "hardware", "tech"],
        "mentions": ["mentions", "mention", "trending", "trend", "google trends", "twitter", "x.com", "tiktok"],
    }
    for key, keywords in mapping.items():
        if any(k in combo for k in keywords):
            return key
    return "other"


def add_liquidity_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    df["spread"] = (df["yes_ask"] - df["yes_bid"]).where(df["yes_ask"].notna() & df["yes_bid"].notna())

    for col, label in [("volume", "volume_bin"), ("open_interest", "oi_bin"), ("spread", "spread_bin")]:
        try:
            df[label] = pd.qcut(df[col], q=5, labels=False, duplicates="drop")
        except ValueError:
            df[label] = pd.Series([None] * len(df), index=df.index)
    return df


def build_observations(
    markets: List[Dict[str, Any]],
    details: List[Dict[str, Any]],
    candlesticks: List[Dict[str, Any]],
    horizons_days: Iterable[float],
) -> pd.DataFrame:
    if not markets:
        return pd.DataFrame()

    markets_df = pd.DataFrame([normalize_market_row(m) for m in markets])
    if details:
        details_df = pd.DataFrame([normalize_market_row(m) for m in details])
        markets_df = markets_df.set_index("ticker").combine_first(details_df.set_index("ticker")).reset_index()
    markets_df = markets_df[markets_df["ticker"].notna()].reset_index(drop=True)

    markets_df["structure"] = classify_structure(markets_df)
    markets_df["category_mapped"] = markets_df.apply(map_category, axis=1)
    markets_df = add_liquidity_bins(markets_df)

    candles_df = build_candles_df(candlesticks)
    if candles_df.empty:
        logger.warning("No candlesticks found; using last_price as fallback")
        markets_df["price"] = markets_df["last_price"]
        markets_df["implied_prob"] = markets_df["price"]
        markets_df["price_bin"] = markets_df["implied_prob"].apply(price_bin)
        markets_df["horizon_days"] = 0
        return markets_df

    targets = []
    for _, row in progress_iter(markets_df.iterrows(), total=len(markets_df), desc="build targets", unit="mkt"):
        close_ts = row["close_ts"]
        if pd.isna(close_ts):
            continue
        for horizon in horizons_days:
            target_ts = close_ts - pd.Timedelta(days=horizon)
            targets.append(
                {
                    "ticker": row["ticker"],
                    "horizon_days": horizon,
                    "target_ts": target_ts,
                    "close_ts": close_ts,
                }
            )

    targets_df = pd.DataFrame(targets)
    if targets_df.empty:
        return pd.DataFrame()

    targets_df = targets_df.sort_values(["ticker", "target_ts"]).reset_index(drop=True)
    candles_df = candles_df.sort_values(["ticker", "ts"]).reset_index(drop=True)

    merged_chunks: List[pd.DataFrame] = []
    total_tickers = targets_df["ticker"].nunique()
    for ticker, tdf in progress_iter(targets_df.groupby("ticker"), total=total_tickers, desc="merge candles", unit="ticker"):
        cdf = candles_df[candles_df["ticker"] == ticker].drop(columns=["ticker"], errors="ignore")
        tdf = tdf.sort_values("target_ts")
        if cdf.empty:
            tdf = tdf.copy()
            tdf["ts"] = pd.NaT
            tdf["price"] = pd.NA
            merged_chunks.append(tdf)
            continue
        cdf = cdf.sort_values("ts")
        merged_chunks.append(
            pd.merge_asof(
                tdf,
                cdf,
                left_on="target_ts",
                right_on="ts",
                direction="backward",
            )
        )

    merged = pd.concat(merged_chunks, ignore_index=True) if merged_chunks else pd.DataFrame()

    markets_df = markets_df.drop(columns=["close_ts"], errors="ignore")
    merged = merged.merge(markets_df, on="ticker", how="left")
    merged.rename(columns={"price": "price_at_time"}, inplace=True)
    merged["implied_prob"] = merged["price_at_time"]
    merged["price_bin"] = merged["implied_prob"].apply(price_bin)
    if "ts" in merged.columns:
        merged["ts"] = pd.to_datetime(merged["ts"], errors="coerce", utc=True)
    if "close_ts" in merged.columns:
        merged["close_ts"] = pd.to_datetime(merged["close_ts"], errors="coerce", utc=True)
    merged["time_to_close_hours"] = (
        (merged["close_ts"] - merged["ts"]).dt.total_seconds() / 3600
        if "ts" in merged.columns and "close_ts" in merged.columns
        else pd.NA
    )
    return merged


def build_candle_observations(
    markets: List[Dict[str, Any]],
    details: List[Dict[str, Any]],
    candlesticks: List[Dict[str, Any]],
    resample_minutes: int = 60,
    max_time_to_close_days: int | None = None,
) -> pd.DataFrame:
    if not markets:
        return pd.DataFrame()

    markets_df = pd.DataFrame([normalize_market_row(m) for m in markets])
    if details:
        details_df = pd.DataFrame([normalize_market_row(m) for m in details])
        markets_df = markets_df.set_index("ticker").combine_first(details_df.set_index("ticker")).reset_index()
    markets_df = markets_df[markets_df["ticker"].notna()].reset_index(drop=True)

    markets_df["structure"] = classify_structure(markets_df)
    markets_df["category_mapped"] = markets_df.apply(map_category, axis=1)
    markets_df = add_liquidity_bins(markets_df)

    candles_df = build_candles_df(candlesticks)
    if candles_df.empty:
        logger.warning("No candlesticks found; candle observations empty")
        return pd.DataFrame()

    candles_df["ts"] = pd.to_datetime(candles_df["ts"], errors="coerce", utc=True)
    candles_df = candles_df.dropna(subset=["ts"])

    if resample_minutes and resample_minutes > 1:
        resampled = []
        grouped = candles_df.groupby("ticker")
        for ticker, g in progress_iter(grouped, total=grouped.ngroups, desc="resample candles", unit="ticker"):
            g = g.set_index("ts")["price"].resample(f"{int(resample_minutes)}min").last().dropna()
            if g.empty:
                continue
            out = g.reset_index()
            out["ticker"] = ticker
            resampled.append(out)
        if resampled:
            candles_df = pd.concat(resampled, ignore_index=True)
        else:
            candles_df = pd.DataFrame(columns=["ticker", "ts", "price"])

    merged = candles_df.merge(markets_df, on="ticker", how="left")
    merged = merged.dropna(subset=["close_ts"])
    merged["implied_prob"] = merged["price"]
    merged["price_bin"] = merged["implied_prob"].apply(price_bin)

    merged["time_to_close_hours"] = (merged["close_ts"] - merged["ts"]).dt.total_seconds() / 3600
    merged = merged[merged["time_to_close_hours"].notna()]
    merged = merged[merged["time_to_close_hours"] >= 0]

    if max_time_to_close_days is not None:
        merged = merged[merged["time_to_close_hours"] <= max_time_to_close_days * 24]

    bins = [0, 1, 3, 6, 12, 24, 72, 168, 720, 2000]
    labels = [
        "0-1h",
        "1-3h",
        "3-6h",
        "6-12h",
        "12-24h",
        "1-3d",
        "3-7d",
        "7-30d",
        "30d+",
    ]
    merged["time_to_close_bucket"] = pd.cut(
        merged["time_to_close_hours"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    merged["horizon_days"] = merged["time_to_close_hours"] / 24.0
    return merged.reset_index(drop=True)
