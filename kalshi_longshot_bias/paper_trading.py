"""Paper trading loop for live Kalshi EV signals.

This module does not place real orders. It simulates entries, fills, and settlements,
and writes incremental artifacts so runs can be paused/resumed across machines.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

from .config import SETTINGS
from .ev_scanner import EV_COLUMNS, evaluate_market_ev, fetch_fee_multipliers
from .kalshi_client import KalshiClient
from .model import BiasModel
from .utils import ensure_dir, outcome_to_bool, parse_timestamp, pick_first, progress_iter, to_dollars

logger = logging.getLogger(__name__)

STATE_VERSION = 1

ORDER_COLUMNS = [
    "order_id",
    "ts",
    "ticker",
    "title",
    "side",
    "liquidity",
    "status",
    "price",
    "fee",
    "contracts",
    "cost",
    "q_hat",
    "implied_prob",
    "ev",
    "roi",
    "category",
    "structure",
    "reason",
]

FILL_COLUMNS = [
    "order_id",
    "position_id",
    "ts",
    "ticker",
    "side",
    "liquidity",
    "price",
    "fee",
    "contracts",
    "cost",
    "q_hat",
    "implied_prob",
    "ev",
    "roi",
]

CLOSED_COLUMNS = [
    "position_id",
    "order_id",
    "entry_ts",
    "exit_ts",
    "ticker",
    "title",
    "side",
    "contracts",
    "entry_price",
    "entry_fee",
    "cost",
    "payout",
    "pnl",
    "roi_pct",
    "won",
    "outcome_yes",
    "category",
    "structure",
]

EQUITY_COLUMNS = [
    "ts",
    "cash",
    "reserved_cash",
    "position_mark_value",
    "cost_basis_open",
    "unrealized_pnl",
    "realized_pnl",
    "total_equity",
    "total_pnl",
    "open_orders",
    "open_positions",
    "markets_scanned",
    "candidates",
    "new_orders",
    "new_fills",
    "settled_positions",
]

SCAN_SUMMARY_COLUMNS = [
    "ts",
    "markets_total",
    "markets_after_filters",
    "markets_scanned",
    "candidate_attempts",
    "positive_ev_candidates",
    "new_orders",
    "new_fills",
    "settled_positions",
    "open_orders",
    "open_positions",
]


@dataclass(frozen=True)
class PaperPaths:
    root: Path
    state: Path
    events_jsonl: Path
    candidates_csv: Path
    orders_csv: Path
    fills_csv: Path
    closed_csv: Path
    equity_csv: Path
    open_orders_csv: Path
    open_positions_csv: Path
    scan_summary_csv: Path


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utcnow().isoformat()


def _ensure_paths(root: Path) -> PaperPaths:
    root = ensure_dir(root)
    return PaperPaths(
        root=root,
        state=root / "state.json",
        events_jsonl=root / "events.jsonl",
        candidates_csv=root / "candidates_latest.csv",
        orders_csv=root / "orders.csv",
        fills_csv=root / "fills.csv",
        closed_csv=root / "closed_trades.csv",
        equity_csv=root / "equity_curve.csv",
        open_orders_csv=root / "open_orders.csv",
        open_positions_csv=root / "open_positions.csv",
        scan_summary_csv=root / "scan_summary.csv",
    )


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def _append_csv_row(path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    ensure_dir(path.parent)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _initial_state(initial_cash: float) -> Dict[str, Any]:
    now = _iso_now()
    return {
        "version": STATE_VERSION,
        "created_at": now,
        "updated_at": now,
        "last_loop_ts": None,
        "loops_completed": 0,
        "initial_cash": float(initial_cash),
        "cash": float(initial_cash),
        "realized_pnl": 0.0,
        "next_order_id": 1,
        "open_orders": [],
        "open_positions": [],
    }


def load_or_init_state(path: Path, initial_cash: float) -> Dict[str, Any]:
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if "version" not in state:
                state["version"] = STATE_VERSION
            state.setdefault("open_orders", [])
            state.setdefault("open_positions", [])
            state.setdefault("cash", float(initial_cash))
            state.setdefault("initial_cash", float(initial_cash))
            state.setdefault("realized_pnl", 0.0)
            state.setdefault("next_order_id", 1)
            state.setdefault("loops_completed", 0)
            state.setdefault("last_loop_ts", None)
            return state
        except Exception as exc:
            logger.warning("Failed to load paper state from %s: %s. Starting fresh.", path, exc)
    return _initial_state(initial_cash)


def save_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = _iso_now()
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _reserved_cash(state: Dict[str, Any]) -> float:
    total = 0.0
    for order in state.get("open_orders", []):
        try:
            total += float(order.get("cost", 0.0))
        except (TypeError, ValueError):
            continue
    return total


def _best_bid_and_depth(bids: List[Any], depth_window: float = 0.02) -> Tuple[float | None, float]:
    if not bids:
        return None, 0.0
    best = bids[0]
    if isinstance(best, dict):
        best_price = to_dollars(best.get("price"))
        best_size = float(best.get("size") or best.get("quantity") or 0.0)
    else:
        best_price = to_dollars(best[0])
        best_size = float(best[1]) if len(best) > 1 else 0.0
    if best_price is None:
        return None, 0.0
    depth = 0.0
    for level in bids:
        if isinstance(level, dict):
            price = to_dollars(level.get("price"))
            size = float(level.get("size") or level.get("quantity") or 0.0)
        else:
            price = to_dollars(level[0])
            size = float(level[1]) if len(level) > 1 else 0.0
        if price is None:
            continue
        if best_price - price <= depth_window:
            depth += size
    return best_price, depth + best_size


def _extract_orderbook_sides(orderbook: Dict[str, Any]) -> Dict[str, List[Any]]:
    if "yes_bids" in orderbook or "no_bids" in orderbook:
        return {"yes": orderbook.get("yes_bids", []), "no": orderbook.get("no_bids", [])}
    ob = orderbook.get("orderbook")
    if isinstance(ob, dict):
        return {"yes": ob.get("yes", []), "no": ob.get("no", [])}
    return {"yes": [], "no": []}


def _effective_bids(market: Dict[str, Any], orderbook: Dict[str, Any]) -> Tuple[float | None, float | None]:
    sides = _extract_orderbook_sides(orderbook)
    yes_best, _ = _best_bid_and_depth(sides["yes"])
    no_best, _ = _best_bid_and_depth(sides["no"])

    market_yes_bid = to_dollars(market.get("yes_bid") or market.get("best_yes_bid"))
    market_no_bid = to_dollars(market.get("no_bid") or market.get("best_no_bid"))
    if yes_best is None:
        yes_best = market_yes_bid
    if no_best is None:
        no_best = market_no_bid
    return yes_best, no_best


def _effective_asks(market: Dict[str, Any], orderbook: Dict[str, Any]) -> Tuple[float | None, float | None]:
    yes_best, no_best = _effective_bids(market, orderbook)
    market_yes_ask = to_dollars(market.get("yes_ask") or market.get("best_yes_ask"))
    market_no_ask = to_dollars(market.get("no_ask") or market.get("best_no_ask"))
    ask_yes = market_yes_ask if market_yes_ask is not None else (1.0 - no_best if no_best is not None else None)
    ask_no = market_no_ask if market_no_ask is not None else (1.0 - yes_best if yes_best is not None else None)
    return ask_yes, ask_no


def _extract_market_outcome(market: Dict[str, Any]) -> Tuple[bool, float | None]:
    status = str(market.get("status") or market.get("state") or "").lower()
    outcome = outcome_to_bool(
        pick_first(
            market,
            [
                "outcome",
                "result",
                "winning_outcome",
                "settlement",
                "yes_settlement",
                "resolved_outcome",
            ],
        )
    )
    resolved = outcome is not None or status in {"resolved", "settled", "finalized"}
    return resolved, outcome


def _normalize_market(market: Dict[str, Any]) -> Dict[str, Any]:
    ticker = market.get("ticker") or market.get("market_ticker")
    if ticker:
        market["ticker"] = ticker
    return market


def _close_positions_if_resolved(
    client: KalshiClient,
    state: Dict[str, Any],
    paths: PaperPaths,
) -> int:
    open_positions = state.get("open_positions", [])
    if not open_positions:
        return 0

    by_ticker: Dict[str, Dict[str, Any]] = {}
    for pos in open_positions:
        ticker = str(pos.get("ticker") or "")
        if not ticker or ticker in by_ticker:
            continue
        try:
            by_ticker[ticker] = _normalize_market(client.get(f"/markets/{ticker}"))
        except Exception as exc:
            logger.warning("Failed to fetch market for open position %s: %s", ticker, exc)

    remaining: List[Dict[str, Any]] = []
    settled = 0
    now = _iso_now()
    for pos in open_positions:
        ticker = str(pos.get("ticker") or "")
        market = by_ticker.get(ticker)
        if not market:
            remaining.append(pos)
            continue
        resolved, outcome_yes = _extract_market_outcome(market)
        if not resolved or outcome_yes is None:
            remaining.append(pos)
            continue

        side = str(pos.get("side") or "YES").upper()
        contracts = int(pos.get("contracts") or 0)
        cost = float(pos.get("cost") or 0.0)
        won = 1 if (side == "YES" and outcome_yes == 1.0) or (side == "NO" and outcome_yes == 0.0) else 0
        payout = float(contracts) if won else 0.0
        pnl = payout - cost
        roi_pct = (pnl / cost) * 100.0 if cost > 0 else 0.0

        state["cash"] = float(state.get("cash", 0.0)) + payout
        state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
        settled += 1

        closed_row = {
            "position_id": pos.get("position_id"),
            "order_id": pos.get("order_id"),
            "entry_ts": pos.get("entry_ts"),
            "exit_ts": now,
            "ticker": ticker,
            "title": pos.get("title"),
            "side": side,
            "contracts": contracts,
            "entry_price": pos.get("price"),
            "entry_fee": pos.get("fee"),
            "cost": cost,
            "payout": payout,
            "pnl": pnl,
            "roi_pct": roi_pct,
            "won": won,
            "outcome_yes": outcome_yes,
            "category": pos.get("category"),
            "structure": pos.get("structure"),
        }
        _append_csv_row(paths.closed_csv, closed_row, CLOSED_COLUMNS)
        _append_jsonl(
            paths.events_jsonl,
            {
                "ts": now,
                "event": "position_settled",
                "ticker": ticker,
                "position_id": pos.get("position_id"),
                "order_id": pos.get("order_id"),
                "won": won,
                "pnl": pnl,
                "cash": state.get("cash"),
            },
        )
    state["open_positions"] = remaining
    return settled


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fill_pending_orders(
    client: KalshiClient,
    state: Dict[str, Any],
    markets_by_ticker: Dict[str, Dict[str, Any]],
    orderbooks_by_ticker: Dict[str, Dict[str, Any]],
    paths: PaperPaths,
) -> int:
    open_orders = state.get("open_orders", [])
    if not open_orders:
        return 0

    now = _utcnow()
    now_iso = now.isoformat()
    remaining_orders: List[Dict[str, Any]] = []
    fills = 0
    for order in open_orders:
        ticker = str(order.get("ticker") or "")
        if not ticker:
            continue
        market = markets_by_ticker.get(ticker)
        if market is None:
            try:
                market = _normalize_market(client.get(f"/markets/{ticker}"))
                markets_by_ticker[ticker] = market
            except Exception as exc:
                logger.warning("Pending order market fetch failed for %s: %s", ticker, exc)
                remaining_orders.append(order)
                continue
        resolved, _ = _extract_market_outcome(market)
        if resolved:
            state["cash"] = float(state.get("cash", 0.0)) + float(order.get("cost") or 0.0)
            _append_csv_row(
                paths.orders_csv,
                {
                    **order,
                    "ts": now_iso,
                    "status": "cancelled_resolved",
                    "reason": "market_resolved_before_fill",
                },
                ORDER_COLUMNS,
            )
            _append_jsonl(
                paths.events_jsonl,
                {
                    "ts": now_iso,
                    "event": "order_cancelled",
                    "ticker": ticker,
                    "order_id": order.get("order_id"),
                    "reason": "market_resolved_before_fill",
                },
            )
            continue

        expires_ts = parse_timestamp(order.get("expires_ts"))
        if expires_ts is not None and expires_ts <= pd_timestamp(now):
            state["cash"] = float(state.get("cash", 0.0)) + float(order.get("cost") or 0.0)
            _append_csv_row(
                paths.orders_csv,
                {
                    **order,
                    "ts": now_iso,
                    "status": "cancelled_expired",
                    "reason": "ttl_expired",
                },
                ORDER_COLUMNS,
            )
            _append_jsonl(
                paths.events_jsonl,
                {
                    "ts": now_iso,
                    "event": "order_cancelled",
                    "ticker": ticker,
                    "order_id": order.get("order_id"),
                    "reason": "ttl_expired",
                },
            )
            continue

        ob = orderbooks_by_ticker.get(ticker)
        if ob is None:
            try:
                ob = client.get(f"/markets/{ticker}/orderbook")
                ob["ticker"] = ticker
                orderbooks_by_ticker[ticker] = ob
            except Exception as exc:
                logger.warning("Pending order orderbook fetch failed for %s: %s", ticker, exc)
                remaining_orders.append(order)
                continue

        ask_yes, ask_no = _effective_asks(market, ob)
        side = str(order.get("side") or "YES").upper()
        limit_price = to_dollars(order.get("price"))
        if limit_price is None:
            remaining_orders.append(order)
            continue

        ask = ask_yes if side == "YES" else ask_no
        if ask is None or ask > limit_price:
            remaining_orders.append(order)
            continue

        fills += 1
        position_id = f"pos-{order.get('order_id')}"
        pos = {
            "position_id": position_id,
            "order_id": order.get("order_id"),
            "ticker": ticker,
            "title": order.get("title"),
            "side": side,
            "entry_ts": now_iso,
            "price": float(order.get("price") or 0.0),
            "fee": float(order.get("fee") or 0.0),
            "contracts": int(order.get("contracts") or 0),
            "cost": float(order.get("cost") or 0.0),
            "q_hat": float(order.get("q_hat") or 0.0),
            "implied_prob": float(order.get("implied_prob") or 0.0),
            "ev": float(order.get("ev") or 0.0),
            "roi": float(order.get("roi") or 0.0),
            "category": order.get("category"),
            "structure": order.get("structure"),
        }
        state.setdefault("open_positions", []).append(pos)

        _append_csv_row(
            paths.orders_csv,
            {
                **order,
                "ts": now_iso,
                "status": "filled",
                "reason": "crossed_limit",
            },
            ORDER_COLUMNS,
        )
        _append_csv_row(
            paths.fills_csv,
            {
                "order_id": order.get("order_id"),
                "position_id": position_id,
                "ts": now_iso,
                "ticker": ticker,
                "side": side,
                "liquidity": order.get("liquidity"),
                "price": order.get("price"),
                "fee": order.get("fee"),
                "contracts": order.get("contracts"),
                "cost": order.get("cost"),
                "q_hat": order.get("q_hat"),
                "implied_prob": order.get("implied_prob"),
                "ev": order.get("ev"),
                "roi": order.get("roi"),
            },
            FILL_COLUMNS,
        )
        _append_jsonl(
            paths.events_jsonl,
            {
                "ts": now_iso,
                "event": "order_filled",
                "ticker": ticker,
                "order_id": order.get("order_id"),
                "position_id": position_id,
                "price": order.get("price"),
                "contracts": order.get("contracts"),
            },
        )
    state["open_orders"] = remaining_orders
    return fills


def _mark_positions(
    positions: List[Dict[str, Any]],
    markets_by_ticker: Dict[str, Dict[str, Any]],
    orderbooks_by_ticker: Dict[str, Dict[str, Any]],
) -> Tuple[float, float]:
    mark_value = 0.0
    cost_basis = 0.0
    for pos in positions:
        ticker = str(pos.get("ticker") or "")
        side = str(pos.get("side") or "YES").upper()
        contracts = int(pos.get("contracts") or 0)
        if contracts <= 0 or not ticker:
            continue
        market = markets_by_ticker.get(ticker, {})
        ob = orderbooks_by_ticker.get(ticker, {})
        yes_bid, no_bid = _effective_bids(market, ob)
        mark = yes_bid if side == "YES" else no_bid
        if mark is None:
            mark = _safe_float(pos.get("price"), 0.0)
        mark_value += float(contracts) * max(mark, 0.0)
        cost_basis += _safe_float(pos.get("cost"), 0.0)
    return mark_value, cost_basis


def pd_timestamp(dt: datetime) -> Any:
    try:
        import pandas as pd  # type: ignore

        return pd.Timestamp(dt)
    except Exception:
        return dt


def _coerce_volume(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _market_volume(market: Dict[str, Any]) -> float:
    return _coerce_volume(market.get("volume_24h") or market.get("volume24h") or market.get("volume_total") or market.get("volume"))


def _market_oi(market: Dict[str, Any]) -> float:
    return _coerce_volume(market.get("open_interest") or market.get("open_int") or market.get("oi"))


def _allowed_liquidity(rows: List[Dict[str, Any]], liquidity_mode: str) -> List[Dict[str, Any]]:
    mode = (liquidity_mode or "both").lower()
    if mode == "both":
        return rows
    if mode == "taker":
        return [r for r in rows if str(r.get("liquidity") or "").lower() == "taker"]
    if mode == "maker":
        return [r for r in rows if str(r.get("liquidity") or "").lower() == "maker"]
    return rows


def _write_snapshots(
    paths: PaperPaths,
    state: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> None:
    _write_csv(paths.open_orders_csv, list(state.get("open_orders", [])), ORDER_COLUMNS)
    _write_csv(
        paths.open_positions_csv,
        list(state.get("open_positions", [])),
        [
            "position_id",
            "order_id",
            "ticker",
            "title",
            "side",
            "entry_ts",
            "price",
            "fee",
            "contracts",
            "cost",
            "q_hat",
            "implied_prob",
            "ev",
            "roi",
            "category",
            "structure",
        ],
    )
    top_candidates = sorted(candidates, key=lambda r: _safe_float(r.get("ev"), -1e9), reverse=True)[:200]
    _write_csv(paths.candidates_csv, top_candidates, EV_COLUMNS)


def _ensure_csv_headers(paths: PaperPaths) -> None:
    if not paths.orders_csv.exists():
        _write_csv(paths.orders_csv, [], ORDER_COLUMNS)
    if not paths.fills_csv.exists():
        _write_csv(paths.fills_csv, [], FILL_COLUMNS)
    if not paths.closed_csv.exists():
        _write_csv(paths.closed_csv, [], CLOSED_COLUMNS)
    if not paths.equity_csv.exists():
        _write_csv(paths.equity_csv, [], EQUITY_COLUMNS)
    if not paths.scan_summary_csv.exists():
        _write_csv(paths.scan_summary_csv, [], SCAN_SUMMARY_COLUMNS)
    if not paths.open_orders_csv.exists():
        _write_csv(paths.open_orders_csv, [], ORDER_COLUMNS)
    if not paths.open_positions_csv.exists():
        _write_csv(
            paths.open_positions_csv,
            [],
            [
                "position_id",
                "order_id",
                "ticker",
                "title",
                "side",
                "entry_ts",
                "price",
                "fee",
                "contracts",
                "cost",
                "q_hat",
                "implied_prob",
                "ev",
                "roi",
                "category",
                "structure",
            ],
        )
    if not paths.candidates_csv.exists():
        _write_csv(paths.candidates_csv, [], EV_COLUMNS)


def run_paper_trading(
    client: KalshiClient,
    model: BiasModel,
    *,
    output_dir: str | Path,
    initial_cash: float = 1000.0,
    stake: float = 10.0,
    min_ev: float = 0.0,
    max_open_positions: int = 200,
    max_new_orders_per_loop: int = 25,
    poll_seconds: float = 60.0,
    live_status: str = "open",
    min_volume_24h: float = 0.0,
    min_open_interest: float = 0.0,
    allow_illiquid: bool = False,
    max_markets: int | None = None,
    once: bool = False,
    liquidity_mode: str = "both",
    maker_order_ttl_minutes: int = 120,
) -> None:
    """Run continuous paper-trading loop and persist all artifacts incrementally."""
    paths = _ensure_paths(Path(output_dir))
    _ensure_csv_headers(paths)
    state = load_or_init_state(paths.state, initial_cash=initial_cash)
    fee_multipliers = fetch_fee_multipliers(client)
    _append_jsonl(
        paths.events_jsonl,
        {
            "ts": _iso_now(),
            "event": "paper_trader_started",
            "output_dir": str(paths.root),
            "cash": state.get("cash"),
            "open_orders": len(state.get("open_orders", [])),
            "open_positions": len(state.get("open_positions", [])),
        },
    )
    logger.info(
        "Paper trader initialized: cash=%.2f open_orders=%s open_positions=%s output_dir=%s",
        float(state.get("cash", 0.0)),
        len(state.get("open_orders", [])),
        len(state.get("open_positions", [])),
        paths.root,
    )

    while True:
        loop_started = _utcnow()
        loop_iso = loop_started.isoformat()
        loop_counter = int(state.get("loops_completed", 0)) + 1
        rejection_stats: Counter[str] = Counter()
        markets_by_ticker: Dict[str, Dict[str, Any]] = {}
        orderbooks_by_ticker: Dict[str, Dict[str, Any]] = {}

        settled = _close_positions_if_resolved(client, state, paths)
        if settled:
            save_state(paths.state, state)

        status_filter = [live_status] if live_status else ["open"]
        from .ingest import download_live_markets  # local import avoids circular import

        fetch_limit = max_markets if max_markets else 2000
        markets = download_live_markets(
            client=client,
            cache_dir=SETTINGS.cache_dir,
            status_filter=status_filter,
            max_markets=fetch_limit,
        )
        total_markets = len(markets)
        if min_volume_24h or min_open_interest:
            markets = [m for m in markets if _market_volume(m) >= min_volume_24h and _market_oi(m) >= min_open_interest]
        markets.sort(key=_market_volume, reverse=True)
        if max_markets:
            markets = markets[:max_markets]
        filtered_markets = len(markets)

        candidates: List[Dict[str, Any]] = []
        blocked_tickers = {str(p.get("ticker")) for p in state.get("open_positions", []) if p.get("ticker")}
        blocked_tickers |= {str(o.get("ticker")) for o in state.get("open_orders", []) if o.get("ticker")}

        bar = progress_iter(markets, total=len(markets), desc="paper scan", unit="mkt")
        for market in bar:
            ticker = market.get("ticker") or market.get("market_ticker")
            if not ticker:
                continue
            market = _normalize_market(market)
            markets_by_ticker[str(ticker)] = market
            if ticker in blocked_tickers:
                continue

            try:
                ob = client.get(f"/markets/{ticker}/orderbook")
            except Exception as exc:
                logger.warning("Orderbook fetch failed for %s: %s", ticker, exc)
                rejection_stats["orderbook_error"] += 1
                continue
            ob["ticker"] = ticker
            orderbooks_by_ticker[str(ticker)] = ob

            rows = evaluate_market_ev(
                market=market,
                orderbook=ob,
                model=model,
                default_fee_multiplier=SETTINGS.default_fee_multiplier,
                fee_multipliers=fee_multipliers,
                tick_size=SETTINGS.tick_size,
                max_spread=SETTINGS.max_spread,
                min_depth=SETTINGS.min_depth,
                min_ev=min_ev,
                maker_fill_prob=SETTINGS.maker_fill_prob,
                allow_illiquid=allow_illiquid,
                stats=rejection_stats,
                slippage_ticks=SETTINGS.slippage_ticks,
                adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
            )
            if not rows:
                continue
            rows = _allowed_liquidity(rows, liquidity_mode=liquidity_mode)
            if not rows:
                continue
            best = max(rows, key=lambda r: _safe_float(r.get("ev"), -1e9))
            candidates.append(best)
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(str(ticker))

        new_fills = _fill_pending_orders(client, state, markets_by_ticker, orderbooks_by_ticker, paths)
        if new_fills:
            save_state(paths.state, state)

        candidates.sort(key=lambda r: _safe_float(r.get("ev"), -1e9), reverse=True)
        new_orders = 0
        for row in candidates:
            if new_orders >= max_new_orders_per_loop:
                break
            if len(state.get("open_positions", [])) >= max_open_positions:
                break

            ticker = str(row.get("ticker") or "")
            if not ticker:
                continue
            if ticker in blocked_tickers:
                continue

            price = to_dollars(row.get("price"))
            fee = to_dollars(row.get("fee"))
            if price is None:
                continue
            if fee is None:
                fee = 0.0
            cost_per = price + fee
            if cost_per <= 0.0:
                continue

            contracts = int(stake // cost_per) if stake > 0 else 1
            if contracts < 1:
                contracts = 1
            cost = float(contracts) * cost_per
            if float(state.get("cash", 0.0)) < cost:
                continue

            order_id = int(state.get("next_order_id", 1))
            state["next_order_id"] = order_id + 1
            state["cash"] = float(state.get("cash", 0.0)) - cost

            liquidity = str(row.get("liquidity") or "taker").lower()
            order_rec: Dict[str, Any] = {
                "order_id": order_id,
                "ts": loop_iso,
                "ticker": ticker,
                "title": row.get("title"),
                "side": str(row.get("side") or "YES").upper(),
                "liquidity": liquidity,
                "status": "submitted",
                "price": price,
                "fee": fee,
                "contracts": contracts,
                "cost": cost,
                "q_hat": _safe_float(row.get("q_hat")),
                "implied_prob": _safe_float(row.get("implied_prob")),
                "ev": _safe_float(row.get("ev")),
                "roi": _safe_float(row.get("roi")),
                "category": row.get("category"),
                "structure": row.get("structure"),
                "reason": "positive_ev",
            }
            _append_csv_row(paths.orders_csv, order_rec, ORDER_COLUMNS)
            _append_jsonl(
                paths.events_jsonl,
                {
                    "ts": loop_iso,
                    "event": "order_submitted",
                    "order_id": order_id,
                    "ticker": ticker,
                    "liquidity": liquidity,
                    "side": order_rec["side"],
                    "price": price,
                    "contracts": contracts,
                    "ev": order_rec["ev"],
                },
            )

            if liquidity == "taker":
                position_id = f"pos-{order_id}"
                pos = {
                    "position_id": position_id,
                    "order_id": order_id,
                    "ticker": ticker,
                    "title": row.get("title"),
                    "side": order_rec["side"],
                    "entry_ts": loop_iso,
                    "price": price,
                    "fee": fee,
                    "contracts": contracts,
                    "cost": cost,
                    "q_hat": order_rec["q_hat"],
                    "implied_prob": order_rec["implied_prob"],
                    "ev": order_rec["ev"],
                    "roi": order_rec["roi"],
                    "category": row.get("category"),
                    "structure": row.get("structure"),
                }
                state.setdefault("open_positions", []).append(pos)
                _append_csv_row(
                    paths.orders_csv,
                    {**order_rec, "ts": loop_iso, "status": "filled", "reason": "taker_immediate"},
                    ORDER_COLUMNS,
                )
                _append_csv_row(
                    paths.fills_csv,
                    {
                        "order_id": order_id,
                        "position_id": position_id,
                        "ts": loop_iso,
                        "ticker": ticker,
                        "side": order_rec["side"],
                        "liquidity": liquidity,
                        "price": price,
                        "fee": fee,
                        "contracts": contracts,
                        "cost": cost,
                        "q_hat": order_rec["q_hat"],
                        "implied_prob": order_rec["implied_prob"],
                        "ev": order_rec["ev"],
                        "roi": order_rec["roi"],
                    },
                    FILL_COLUMNS,
                )
                _append_jsonl(
                    paths.events_jsonl,
                    {
                        "ts": loop_iso,
                        "event": "order_filled",
                        "order_id": order_id,
                        "position_id": position_id,
                        "ticker": ticker,
                        "liquidity": liquidity,
                    },
                )
                new_fills += 1
            else:
                expires_ts = (loop_started + timedelta(minutes=max(maker_order_ttl_minutes, 1))).isoformat()
                pending = {**order_rec, "expires_ts": expires_ts}
                state.setdefault("open_orders", []).append(pending)

            new_orders += 1
            blocked_tickers.add(ticker)
            save_state(paths.state, state)

        mark_value, cost_basis_open = _mark_positions(
            state.get("open_positions", []),
            markets_by_ticker=markets_by_ticker,
            orderbooks_by_ticker=orderbooks_by_ticker,
        )
        reserved_cash = _reserved_cash(state)
        cash = float(state.get("cash", 0.0))
        realized_pnl = float(state.get("realized_pnl", 0.0))
        unrealized_pnl = mark_value - cost_basis_open
        total_equity = cash + reserved_cash + mark_value
        total_pnl = total_equity - float(state.get("initial_cash", initial_cash))

        equity_row = {
            "ts": loop_iso,
            "cash": cash,
            "reserved_cash": reserved_cash,
            "position_mark_value": mark_value,
            "cost_basis_open": cost_basis_open,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "total_equity": total_equity,
            "total_pnl": total_pnl,
            "open_orders": len(state.get("open_orders", [])),
            "open_positions": len(state.get("open_positions", [])),
            "markets_scanned": filtered_markets,
            "candidates": len(candidates),
            "new_orders": new_orders,
            "new_fills": new_fills,
            "settled_positions": settled,
        }
        _append_csv_row(paths.equity_csv, equity_row, EQUITY_COLUMNS)

        scan_summary = {
            "ts": loop_iso,
            "markets_total": total_markets,
            "markets_after_filters": filtered_markets,
            "markets_scanned": filtered_markets,
            "candidate_attempts": int(rejection_stats.get("candidate_attempts", 0)),
            "positive_ev_candidates": len(candidates),
            "new_orders": new_orders,
            "new_fills": new_fills,
            "settled_positions": settled,
            "open_orders": len(state.get("open_orders", [])),
            "open_positions": len(state.get("open_positions", [])),
        }
        _append_csv_row(paths.scan_summary_csv, scan_summary, SCAN_SUMMARY_COLUMNS)

        _write_snapshots(paths, state, candidates)

        state["last_loop_ts"] = loop_iso
        state["loops_completed"] = loop_counter
        save_state(paths.state, state)

        logger.info(
            "Paper loop %s complete: scanned=%s candidates=%s new_orders=%s new_fills=%s settled=%s total_equity=%.2f",
            loop_counter,
            filtered_markets,
            len(candidates),
            new_orders,
            new_fills,
            settled,
            total_equity,
        )

        _append_jsonl(
            paths.events_jsonl,
            {
                "ts": loop_iso,
                "event": "loop_complete",
                "loop": loop_counter,
                "scanned": filtered_markets,
                "candidates": len(candidates),
                "new_orders": new_orders,
                "new_fills": new_fills,
                "settled_positions": settled,
                "total_equity": total_equity,
                "total_pnl": total_pnl,
            },
        )

        if once:
            return
        sleep_for = max(float(poll_seconds), 1.0)
        time.sleep(sleep_for)
