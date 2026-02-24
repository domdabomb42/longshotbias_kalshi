"""Live EV scanner for Kalshi orderbooks."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

import pandas as pd

from .bias_metrics import fee_dollars
from .model import BiasModel
from .utils import to_dollars

logger = logging.getLogger(__name__)

EV_COLUMNS = [
    "ticker",
    "title",
    "side",
    "liquidity",
    "price",
    "implied_prob",
    "q_hat",
    "fee",
    "ev",
    "roi",
    "spread",
    "depth",
    "volume_24h",
    "open_interest",
    "category",
    "structure",
]


def fetch_fee_multipliers(client) -> Dict[str, float]:
    try:
        payload = client.get("/series/fee_changes")
    except Exception as exc:
        logger.warning("Failed to fetch fee changes: %s", exc)
        return {}

    candidates = []
    for key in ("fee_changes", "series", "data", "results"):
        if key in payload and isinstance(payload[key], list):
            candidates = payload[key]
            break
    if not candidates and isinstance(payload, list):
        candidates = payload

    mapping: Dict[str, float] = {}
    for item in candidates:
        ticker = item.get("series_ticker") or item.get("ticker") or item.get("series")
        fee = item.get("fee_multiplier") or item.get("fee") or item.get("multiplier")
        if ticker and fee is not None:
            try:
                mapping[ticker] = float(fee)
            except ValueError:
                continue
    return mapping


def _extract_orderbook_sides(orderbook: Dict[str, Any]) -> Dict[str, List[Any]]:
    if "yes_bids" in orderbook or "no_bids" in orderbook:
        return {"yes": orderbook.get("yes_bids", []), "no": orderbook.get("no_bids", [])}
    if "orderbook" in orderbook:
        ob = orderbook["orderbook"]
        if isinstance(ob, dict):
            return {"yes": ob.get("yes", []), "no": ob.get("no", [])}
    if "bids" in orderbook and "side" in orderbook:
        # not expected but handle
        return {"yes": orderbook.get("bids", []), "no": []}
    return {"yes": [], "no": []}


def _best_bid_and_depth(bids: List[Any], depth_window: float = 0.02) -> tuple[float | None, float]:
    if not bids:
        return None, 0.0
    best = bids[0]
    if isinstance(best, dict):
        best_price = to_dollars(best.get("price"))
        best_size = float(best.get("size") or best.get("quantity") or 0)
    else:
        best_price = to_dollars(best[0])
        best_size = float(best[1]) if len(best) > 1 else 0.0

    depth = 0.0
    if best_price is None:
        return None, 0.0
    for level in bids:
        if isinstance(level, dict):
            price = to_dollars(level.get("price"))
            size = float(level.get("size") or level.get("quantity") or 0)
        else:
            price = to_dollars(level[0])
            size = float(level[1]) if len(level) > 1 else 0.0
        if price is None:
            continue
        if best_price - price <= depth_window:
            depth += size
    return best_price, depth + best_size


def evaluate_market_ev(
    market: Dict[str, Any],
    orderbook: Dict[str, Any],
    model: BiasModel,
    default_fee_multiplier: float,
    fee_multipliers: Dict[str, float],
    tick_size: float,
    max_spread: float,
    min_depth: float,
    min_ev: float,
    maker_fill_prob: float,
    allow_illiquid: bool = False,
    stats: Dict[str, int] | None = None,
    slippage_ticks: int = 1,
    adverse_selection_penalty: float = 0.02,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ticker = market.get("ticker") or market.get("market_ticker")
    if not ticker:
        if stats is not None:
            stats["no_ticker"] = stats.get("no_ticker", 0) + 1
        return rows

    sides = _extract_orderbook_sides(orderbook)
    yes_best, yes_depth = _best_bid_and_depth(sides["yes"])
    no_best, no_depth = _best_bid_and_depth(sides["no"])

    market_yes_bid = to_dollars(market.get("yes_bid") or market.get("best_yes_bid"))
    market_no_bid = to_dollars(market.get("no_bid") or market.get("best_no_bid"))
    market_yes_ask = to_dollars(market.get("yes_ask") or market.get("best_yes_ask"))
    market_no_ask = to_dollars(market.get("no_ask") or market.get("best_no_ask"))
    market_last = to_dollars(market.get("last_price") or market.get("last_yes") or market.get("last_yes_price"))

    if yes_best is None:
        yes_best = market_yes_bid
    if no_best is None:
        no_best = market_no_bid

    if yes_best is None and no_best is None and not allow_illiquid:
        if stats is not None:
            stats["no_bids"] = stats.get("no_bids", 0) + 1
        return rows

    def _first_with_source(*values: tuple[str, float | None]) -> tuple[float | None, str | None]:
        for label, val in values:
            if val is not None:
                return val, label
        return None, None

    # Base price estimates even when one side is missing.
    base_yes, base_yes_src = _first_with_source(
        ("yes_best", yes_best),
        ("yes_bid", market_yes_bid),
        ("yes_ask", market_yes_ask),
        ("last", market_last),
        ("inv_no_best", (1 - no_best) if no_best is not None else None),
        ("inv_no_bid", (1 - market_no_bid) if market_no_bid is not None else None),
        ("inv_no_ask", (1 - market_no_ask) if market_no_ask is not None else None),
    )
    base_no, base_no_src = _first_with_source(
        ("no_best", no_best),
        ("no_bid", market_no_bid),
        ("no_ask", market_no_ask),
        ("inv_yes_best", (1 - yes_best) if yes_best is not None else None),
        ("inv_yes_bid", (1 - market_yes_bid) if market_yes_bid is not None else None),
        ("inv_yes_ask", (1 - market_yes_ask) if market_yes_ask is not None else None),
        ("inv_last", (1 - market_last) if market_last is not None else None),
    )

    ask_yes = base_yes
    ask_no = base_no

    series = market.get("series_ticker") or market.get("series_id") or market.get("series")
    fee_multiplier = fee_multipliers.get(series, default_fee_multiplier)

    category = market.get("category") or market.get("event_category") or "other"
    structure = market.get("structure") or "unknown"
    volume_24h = market.get("volume_24h") or market.get("volume") or None
    open_interest = market.get("open_interest") or None

    def add_candidate(side: str, price: float | None, depth: float, liquidity: str) -> None:
        if stats is not None:
            stats["candidate_attempts"] = stats.get("candidate_attempts", 0) + 1
        if price is None:
            if stats is not None:
                stats["price_missing"] = stats.get("price_missing", 0) + 1
            return
        if price <= 0 or price >= 1:
            if stats is not None:
                stats["price_out_of_bounds"] = stats.get("price_out_of_bounds", 0) + 1
            return
        spread = None
        if side == "YES" and yes_best is not None and ask_yes is not None:
            spread = ask_yes - yes_best
        if side == "NO" and no_best is not None and ask_no is not None:
            spread = ask_no - no_best
        if (not allow_illiquid) and depth < min_depth:
            if stats is not None:
                stats["depth_too_low"] = stats.get("depth_too_low", 0) + 1
            return

        slip = max(slippage_ticks, 0) * tick_size
        price_eff = price + slip
        if price_eff <= 0 or price_eff >= 1:
            if stats is not None:
                stats["price_out_of_bounds"] = stats.get("price_out_of_bounds", 0) + 1
            return

        implied_yes = price_eff if side == "YES" else 1 - price_eff
        q_hat = model.predict_one(implied_yes, str(category).lower(), str(structure).lower())
        fee = fee_dollars(price_eff, fee_multiplier)

        penalty = adverse_selection_penalty
        if spread is not None:
            penalty += max(spread, 0.0) * 0.5
        inferred_yes = base_yes_src is not None and base_yes_src.startswith("inv_")
        inferred_no = base_no_src is not None and base_no_src.startswith("inv_")
        if (side == "YES" and inferred_yes) or (side == "NO" and inferred_no):
            penalty += adverse_selection_penalty
        if liquidity == "maker":
            penalty += adverse_selection_penalty
        if side == "YES":
            q_adj = max(0.0, q_hat - penalty)
            ev = q_adj - price_eff - fee
        else:
            q_no = max(0.0, (1 - q_hat) - penalty)
            ev = q_no - price_eff - fee
        if liquidity == "maker":
            ev = ev * maker_fill_prob
        roi = ev / (price_eff + fee) if price_eff + fee > 0 else 0
        if ev <= min_ev:
            if stats is not None:
                stats["ev_too_low"] = stats.get("ev_too_low", 0) + 1
            return
        if stats is not None:
            stats["positive_ev"] = stats.get("positive_ev", 0) + 1

        rows.append(
            {
                "ticker": ticker,
                "title": market.get("title"),
                "side": side,
                "liquidity": liquidity,
                "price": price_eff,
                "implied_prob": implied_yes,
                "q_hat": q_hat,
                "fee": fee,
                "ev": ev,
                "roi": roi,
                "spread": spread,
                "depth": depth,
                "volume_24h": volume_24h,
                "open_interest": open_interest,
                "category": category,
                "structure": structure,
            }
        )

    # Taker candidates
    add_candidate("YES", ask_yes, yes_depth, "taker")
    add_candidate("NO", ask_no, no_depth, "taker")

    # Maker candidates
    if yes_best is not None and ask_yes is not None:
        maker_yes = min(ask_yes - tick_size, yes_best + tick_size)
        add_candidate("YES", maker_yes, yes_depth, "maker")
    if no_best is not None and ask_no is not None:
        maker_no = min(ask_no - tick_size, no_best + tick_size)
        add_candidate("NO", maker_no, no_depth, "maker")
    # Allow illiquid maker bids using inferred base prices if no bids exist
    if allow_illiquid and yes_best is None and no_best is None:
        if base_yes is not None:
            maker_yes = max(base_yes - tick_size, 0.01)
            add_candidate("YES", maker_yes, 0.0, "maker")
        if base_no is not None:
            maker_no = max(base_no - tick_size, 0.01)
            add_candidate("NO", maker_no, 0.0, "maker")

    return rows


def scan_positive_ev(
    markets: List[Dict[str, Any]],
    orderbooks: List[Dict[str, Any]],
    model: BiasModel,
    default_fee_multiplier: float,
    fee_multipliers: Dict[str, float],
    tick_size: float,
    max_spread: float,
    min_depth: float,
    min_ev: float,
    maker_fill_prob: float,
    allow_illiquid: bool = False,
    stats: Dict[str, int] | None = None,
    slippage_ticks: int = 1,
    adverse_selection_penalty: float = 0.02,
) -> pd.DataFrame:
    ob_map = {ob.get("ticker"): ob for ob in orderbooks}
    rows: List[Dict[str, Any]] = []

    from .utils import progress_iter

    for market in progress_iter(markets, total=len(markets), desc="scan EV", unit="mkt"):
        ticker = market.get("ticker") or market.get("market_ticker")
        if not ticker:
            continue
        ob = ob_map.get(ticker)
        if not ob:
            continue
        rows.extend(
            evaluate_market_ev(
                market=market,
                orderbook=ob,
                model=model,
                default_fee_multiplier=default_fee_multiplier,
                fee_multipliers=fee_multipliers,
                tick_size=tick_size,
                max_spread=max_spread,
                min_depth=min_depth,
                min_ev=min_ev,
                maker_fill_prob=maker_fill_prob,
                allow_illiquid=allow_illiquid,
                stats=stats,
                slippage_ticks=slippage_ticks,
                adverse_selection_penalty=adverse_selection_penalty,
            )
        )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("ev", ascending=False).reset_index(drop=True)
