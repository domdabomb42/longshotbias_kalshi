"""Crypto-focused Kalshi strategy: buy low / sell high with backtest + live API loop.

Usage examples:
  python cryptoconclave.py backtest
  python cryptoconclave.py backtest --lookback-bars 12 --low-quantile 0.2 --high-quantile 0.85
  python cryptoconclave.py live --once
  python cryptoconclave.py live --place-live-orders
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Deque, Dict, List
from uuid import uuid4

import numpy as np
import pandas as pd

from kalshi_longshot_bias.bias_metrics import fee_dollars
from kalshi_longshot_bias.config import SETTINGS
from kalshi_longshot_bias.ingest import download_live_markets
from kalshi_longshot_bias.kalshi_client import KalshiClient, KalshiClientConfig
from kalshi_longshot_bias.utils import ensure_dir, to_dollars

logger = logging.getLogger("cryptoconclave")

CRYPTO_KEYWORDS = ("crypto", "bitcoin", "btc", "ethereum", "eth", "sol", "doge", "xrp")


@dataclass
class Position:
    ticker: str
    title: str
    entry_ts: pd.Timestamp
    close_ts: pd.Timestamp
    entry_price: float
    entry_fee: float
    contracts: int
    cost: float


@dataclass
class BacktestResult:
    summary: Dict[str, Any]
    trades: pd.DataFrame
    equity: pd.DataFrame


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _market_is_crypto(market: Dict[str, Any]) -> bool:
    category = str(market.get("category") or market.get("event_category") or "").lower()
    if "crypto" in category:
        return True
    hay = " ".join(
        [
            str(market.get("ticker") or ""),
            str(market.get("event_ticker") or ""),
            str(market.get("title") or ""),
            str(market.get("subtitle") or ""),
            str(market.get("series_ticker") or ""),
        ]
    ).lower()
    return any(k in hay for k in CRYPTO_KEYWORDS)


def _row_is_crypto(df: pd.DataFrame) -> pd.Series:
    category = df.get("category_mapped", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    ticker = df.get("ticker", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    title = df.get("title", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    mask = category.eq("crypto")
    text_mask = pd.Series(False, index=df.index)
    for key in CRYPTO_KEYWORDS:
        text_mask = text_mask | ticker.str.contains(key, na=False) | title.str.contains(key, na=False)
    return mask | text_mask


def _load_candle_observations(processed_dir: Path) -> pd.DataFrame:
    parquet_path = processed_dir / "candle_observations.parquet"
    csv_path = processed_dir / "candle_observations.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing candle observations in {processed_dir}")


def _extract_price(df: pd.DataFrame) -> pd.Series:
    if "price" in df.columns:
        price = pd.to_numeric(df["price"], errors="coerce")
    elif "implied_prob" in df.columns:
        price = pd.to_numeric(df["implied_prob"], errors="coerce")
    else:
        yes_bid = pd.to_numeric(df.get("yes_bid"), errors="coerce")
        yes_ask = pd.to_numeric(df.get("yes_ask"), errors="coerce")
        price = (yes_bid + yes_ask) / 2.0
        price = price.fillna(yes_bid).fillna(yes_ask)
    return price.where(price.between(0.001, 0.999))


def _mark_to_market_cash(cash: float, states: Dict[str, Dict[str, Any]], latest_prices: Dict[str, float]) -> float:
    value = cash
    for state in states.values():
        pos = state.get("position")
        if not pos:
            continue
        mark_price = latest_prices.get(pos.ticker, pos.entry_price)
        value += float(pos.contracts) * mark_price
    return value


def run_backtest(
    *,
    processed_dir: Path,
    initial_cash: float,
    stake: float,
    lookback_bars: int,
    low_quantile: float,
    high_quantile: float,
    take_profit: float,
    stop_loss: float,
    force_exit_minutes: float,
    fee_multiplier: float,
    tick_size: float,
    slippage_ticks: int,
    max_open_positions: int,
    max_rows: int | None,
    latency_bars: int,
    base_fill_prob: float,
    min_fill_prob: float,
    spread_fill_penalty: float,
    liquidity_fill_boost: float,
    partial_fills: bool,
    impact_coeff: float,
    intrabar_noise: float,
    order_reject_prob: float,
    outage_prob: float,
    outage_min_bars: int,
    outage_max_bars: int,
    random_seed: int,
) -> BacktestResult:
    df = _load_candle_observations(processed_dir).copy()
    total_rows = len(df)
    df = df[_row_is_crypto(df)].copy()
    crypto_rows = len(df)
    if df.empty:
        return BacktestResult(
            summary={
                "initial_cash": initial_cash,
                "final_cash": initial_cash,
                "total_pnl": 0.0,
                "roi": 0.0,
                "trades": 0,
                "wins": 0,
                "win_rate": 0.0,
                "rows_total": total_rows,
                "rows_crypto": crypto_rows,
                "note": "No crypto observations found.",
            },
            trades=pd.DataFrame(),
            equity=pd.DataFrame(),
        )

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df["close_ts"] = pd.to_datetime(df["close_ts"], errors="coerce", utc=True)
    df["price"] = _extract_price(df)
    df["outcome"] = pd.to_numeric(df.get("outcome"), errors="coerce")
    yes_ask = pd.to_numeric(df.get("yes_ask"), errors="coerce")
    yes_bid = pd.to_numeric(df.get("yes_bid"), errors="coerce")
    spread_proxy = (yes_ask - yes_bid).where(yes_ask.notna() & yes_bid.notna())
    df["spread_proxy"] = pd.to_numeric(spread_proxy, errors="coerce").clip(lower=0.0).fillna(0.03)
    vol_col = pd.to_numeric(df.get("volume"), errors="coerce")
    oi_col = pd.to_numeric(df.get("open_interest"), errors="coerce")
    df["liquidity_proxy"] = pd.concat([vol_col, oi_col], axis=1).mean(axis=1, skipna=True).fillna(0.0).clip(lower=0.0)
    df = df.dropna(subset=["ticker", "ts", "close_ts", "price"])
    df = df[df["ts"] <= df["close_ts"]]
    df = df.sort_values(["ts", "ticker"]).drop_duplicates(["ticker", "ts"], keep="last").reset_index(drop=True)
    if max_rows:
        df = df.head(max_rows).copy()

    lookback_bars = max(int(lookback_bars), 2)
    low_quantile = float(np.clip(low_quantile, 0.01, 0.49))
    high_quantile = float(np.clip(high_quantile, 0.51, 0.99))
    slippage = max(int(slippage_ticks), 0) * float(tick_size)
    max_open_positions = max(int(max_open_positions), 1)
    latency_bars = max(int(latency_bars), 0)
    base_fill_prob = float(np.clip(base_fill_prob, 0.01, 1.0))
    min_fill_prob = float(np.clip(min_fill_prob, 0.0, base_fill_prob))
    spread_fill_penalty = max(float(spread_fill_penalty), 0.0)
    liquidity_fill_boost = max(float(liquidity_fill_boost), 1e-6)
    impact_coeff = max(float(impact_coeff), 0.0)
    intrabar_noise = max(float(intrabar_noise), 0.0)
    order_reject_prob = float(np.clip(order_reject_prob, 0.0, 1.0))
    outage_prob = float(np.clip(outage_prob, 0.0, 1.0))
    outage_min_bars = max(int(outage_min_bars), 1)
    outage_max_bars = max(int(outage_max_bars), outage_min_bars)
    rng = np.random.default_rng(int(random_seed))

    if latency_bars > 0:
        grp = df.groupby("ticker", sort=False)
        df["exec_price"] = grp["price"].shift(-latency_bars)
        df["exec_ts"] = grp["ts"].shift(-latency_bars)
    else:
        df["exec_price"] = df["price"]
        df["exec_ts"] = df["ts"]
    df["exec_ts"] = pd.to_datetime(df["exec_ts"], errors="coerce", utc=True)

    states: Dict[str, Dict[str, Any]] = {}
    latest_prices: Dict[str, float] = {}
    trades: List[Dict[str, Any]] = []
    equity_events: List[Dict[str, Any]] = []
    exit_reasons: Counter[str] = Counter()
    execution_stats: Counter[str] = Counter()
    cash = float(initial_cash)
    skipped_cash = 0
    outage_rows_remaining = 0
    outage_active = False

    def _vol_est(history_vals: np.ndarray) -> float:
        if history_vals.size < 3:
            return max(float(tick_size) * 0.25, 0.0025)
        recent = history_vals[-min(history_vals.size, lookback_bars) :]
        vol = float(np.nanstd(np.diff(recent)))
        return max(vol, float(tick_size) * 0.25, 0.0025)

    def _fill_probability(spread: float, liquidity: float) -> float:
        spread_adj = float(np.exp(-spread_fill_penalty * max(spread, 0.0)))
        liq_strength = 1.0 - float(np.exp(-max(liquidity, 0.0) / liquidity_fill_boost))
        liq_adj = 0.35 + 0.65 * liq_strength
        prob = base_fill_prob * spread_adj * liq_adj
        return float(np.clip(prob, min_fill_prob, 1.0))

    def _attempt_execution(
        *,
        side: str,
        stage: str,
        desired_contracts: int,
        base_price: float,
        spread: float,
        liquidity: float,
        volatility: float,
    ) -> Dict[str, Any] | None:
        execution_stats[f"{stage}_attempts"] += 1
        if outage_active:
            execution_stats["blocked_by_outage"] += 1
            return None
        if rng.random() < order_reject_prob:
            execution_stats[f"{stage}_rejected"] += 1
            return None

        fill_prob = _fill_probability(spread=spread, liquidity=liquidity)
        if rng.random() > fill_prob:
            execution_stats[f"{stage}_unfilled"] += 1
            return None

        contracts = max(int(desired_contracts), 1)
        if partial_fills and contracts > 1:
            lo = max(0.20, fill_prob * 0.40)
            hi = min(1.0, fill_prob + 0.25)
            frac = float(rng.uniform(lo, hi))
            partial_contracts = max(1, int(np.floor(contracts * frac)))
            if partial_contracts < contracts:
                execution_stats["partial_fills"] += 1
            contracts = partial_contracts

        impact = impact_coeff * np.sqrt(contracts / max(liquidity, 1.0))
        micro_noise = abs(float(rng.normal(0.0, volatility * intrabar_noise)))
        if side == "buy":
            exec_price = base_price + slippage + impact + micro_noise
        else:
            exec_price = base_price - slippage - impact - micro_noise
        exec_price = float(np.clip(exec_price, 0.001, 0.999))
        fee = fee_dollars(exec_price, fee_multiplier)
        execution_stats[f"{stage}_fills"] += 1
        return {
            "contracts": contracts,
            "price": exec_price,
            "fee": fee,
            "fill_prob": fill_prob,
            "impact": impact,
            "noise": micro_noise,
        }

    def close_position(
        *,
        state: Dict[str, Any],
        exit_ts: pd.Timestamp,
        reason: str,
        exit_price: float | None = None,
        settlement_outcome: float | None = None,
        contracts_to_close: int | None = None,
        fill_prob: float | None = None,
    ) -> None:
        nonlocal cash
        pos: Position | None = state.get("position")
        if pos is None:
            return
        original_contracts = int(pos.contracts)
        close_contracts = original_contracts if contracts_to_close is None else min(max(int(contracts_to_close), 1), original_contracts)
        if close_contracts <= 0:
            return

        cost_portion = float(pos.cost) * (close_contracts / max(original_contracts, 1))

        exit_fee = 0.0
        if settlement_outcome is not None and settlement_outcome in (0.0, 1.0):
            payout = float(close_contracts) * (1.0 if settlement_outcome == 1.0 else 0.0)
            realized_exit_price = 1.0 if settlement_outcome == 1.0 else 0.0
        else:
            realized_exit_price = float(np.clip(exit_price if exit_price is not None else pos.entry_price, 0.001, 0.999))
            exit_fee = fee_dollars(realized_exit_price, fee_multiplier)
            payout = float(close_contracts) * max(realized_exit_price - exit_fee, 0.0)

        cash += payout
        pnl = payout - cost_portion
        roi = (pnl / cost_portion) if cost_portion > 0 else 0.0

        trades.append(
            {
                "ticker": pos.ticker,
                "title": pos.title,
                "entry_ts": pos.entry_ts,
                "exit_ts": exit_ts,
                "close_ts": pos.close_ts,
                "entry_price": pos.entry_price,
                "entry_fee": pos.entry_fee,
                "exit_price": realized_exit_price,
                "exit_fee": exit_fee,
                "contracts": close_contracts,
                "cost": cost_portion,
                "payout": payout,
                "pnl": pnl,
                "roi": roi,
                "reason": reason,
                "won": int(pnl > 0),
                "fill_prob": fill_prob,
                "remaining_contracts": max(original_contracts - close_contracts, 0),
            }
        )
        if close_contracts >= original_contracts:
            state["position"] = None
        else:
            pos.contracts = original_contracts - close_contracts
            pos.cost = max(float(pos.cost) - cost_portion, 0.0)
        exit_reasons[reason] += 1

    for row in df.itertuples(index=False):
        if outage_rows_remaining > 0:
            outage_active = True
            outage_rows_remaining -= 1
            execution_stats["rows_during_outage"] += 1
        else:
            outage_active = False
            if outage_prob > 0 and rng.random() < outage_prob:
                outage_active = True
                duration = int(rng.integers(outage_min_bars, outage_max_bars + 1))
                outage_rows_remaining = max(duration - 1, 0)
                execution_stats["outage_events"] += 1
                execution_stats["rows_during_outage"] += 1

        ticker = str(row.ticker)
        ts = pd.Timestamp(row.ts)
        close_ts = pd.Timestamp(row.close_ts)
        price = float(row.price)
        exec_price_base = float(row.exec_price) if pd.notna(getattr(row, "exec_price", np.nan)) else np.nan
        exec_ts_raw = getattr(row, "exec_ts", pd.NaT)
        exec_ts = pd.Timestamp(exec_ts_raw) if pd.notna(exec_ts_raw) else pd.NaT
        spread = float(getattr(row, "spread_proxy", 0.03) or 0.03)
        liquidity = float(getattr(row, "liquidity_proxy", 0.0) or 0.0)
        title = str(getattr(row, "title", "") or "")
        outcome_raw = getattr(row, "outcome", np.nan)
        outcome = float(outcome_raw) if pd.notna(outcome_raw) else None

        state = states.setdefault(
            ticker,
            {"window": deque(maxlen=lookback_bars), "position": None, "title": title, "close_ts": close_ts, "outcome": outcome},
        )
        state["title"] = title or state["title"]
        state["close_ts"] = close_ts
        if outcome in (0.0, 1.0):
            state["outcome"] = outcome

        pos: Position | None = state.get("position")
        if pos is not None and ts >= pos.close_ts:
            close_position(state=state, exit_ts=ts, reason="settlement", settlement_outcome=state.get("outcome"))
            pos = None

        window: Deque[float] = state["window"]
        history = np.array(window, dtype=float) if window else np.array([], dtype=float)
        volatility = _vol_est(history)
        minutes_to_close = (close_ts - ts).total_seconds() / 60.0

        if pos is not None and history.size >= 2:
            local_high = float(np.quantile(history, high_quantile))
            hit_high = price >= local_high
            hit_take_profit = price >= pos.entry_price + take_profit
            hit_stop = price <= max(pos.entry_price - stop_loss, 0.001)
            near_close = minutes_to_close <= force_exit_minutes
            if hit_high or hit_take_profit or hit_stop or near_close:
                reason = "high_quantile"
                if hit_take_profit:
                    reason = "take_profit"
                elif hit_stop:
                    reason = "stop_loss"
                elif near_close:
                    reason = "near_close"
                execution_stats["exit_signals"] += 1
                latency_ok = pd.notna(exec_ts) and pd.notna(exec_price_base) and exec_ts <= close_ts
                if not latency_ok:
                    execution_stats["latency_missed"] += 1
                else:
                    attempt = _attempt_execution(
                        side="sell",
                        stage="exit",
                        desired_contracts=int(pos.contracts),
                        base_price=exec_price_base,
                        spread=spread,
                        liquidity=liquidity,
                        volatility=volatility,
                    )
                    if attempt is not None:
                        close_position(
                            state=state,
                            exit_ts=exec_ts,
                            reason=reason,
                            exit_price=float(attempt["price"]),
                            settlement_outcome=None,
                            contracts_to_close=int(attempt["contracts"]),
                            fill_prob=float(attempt["fill_prob"]),
                        )
                        pos = state.get("position")

        if pos is None and history.size >= lookback_bars and minutes_to_close > force_exit_minutes:
            local_low = float(np.quantile(history, low_quantile))
            if price <= local_low:
                execution_stats["entry_signals"] += 1
                latency_ok = pd.notna(exec_ts) and pd.notna(exec_price_base) and exec_ts <= close_ts
                if not latency_ok:
                    execution_stats["latency_missed"] += 1
                else:
                    preview_price = float(np.clip(exec_price_base + slippage, 0.001, 0.999))
                    preview_fee = fee_dollars(preview_price, fee_multiplier)
                    preview_cost_per = preview_price + preview_fee
                    if preview_cost_per > 0:
                        desired_contracts = int(stake // preview_cost_per) if stake > 0 else 1
                        desired_contracts = max(desired_contracts, 1)
                        open_count = sum(1 for s in states.values() if s.get("position") is not None)
                        if open_count >= max_open_positions:
                            skipped_cash += 1
                            execution_stats["skipped_capacity"] += 1
                        else:
                            attempt = _attempt_execution(
                                side="buy",
                                stage="entry",
                                desired_contracts=desired_contracts,
                                base_price=exec_price_base,
                                spread=spread,
                                liquidity=liquidity,
                                volatility=volatility,
                            )
                            if attempt is not None:
                                actual_cost_per = float(attempt["price"]) + float(attempt["fee"])
                                affordable = int(cash // actual_cost_per) if actual_cost_per > 0 else 0
                                if affordable < 1:
                                    skipped_cash += 1
                                    execution_stats["cash_insufficient"] += 1
                                else:
                                    filled_contracts = min(int(attempt["contracts"]), affordable)
                                    if filled_contracts < int(attempt["contracts"]):
                                        execution_stats["downsized_for_cash"] += 1
                                    cost = filled_contracts * actual_cost_per
                                    cash -= cost
                                    state["position"] = Position(
                                        ticker=ticker,
                                        title=state["title"],
                                        entry_ts=exec_ts,
                                        close_ts=close_ts,
                                        entry_price=float(attempt["price"]),
                                        entry_fee=float(attempt["fee"]),
                                        contracts=filled_contracts,
                                        cost=cost,
                                    )

        window.append(price)
        latest_prices[ticker] = price
        open_positions = sum(1 for s in states.values() if s.get("position") is not None)
        equity_events.append(
            {
                "ts": ts,
                "cash": cash,
                "mark_to_market": _mark_to_market_cash(cash, states, latest_prices),
                "open_positions": open_positions,
            }
        )

    last_ts = pd.Timestamp(df["ts"].max()) if not df.empty else pd.Timestamp.now(tz="UTC")
    for state in states.values():
        pos: Position | None = state.get("position")
        if pos is None:
            continue
        if state.get("outcome") in (0.0, 1.0):
            close_position(
                state=state,
                exit_ts=max(last_ts, pos.close_ts),
                reason="end_settlement",
                settlement_outcome=float(state["outcome"]),
            )
        else:
            px = latest_prices.get(pos.ticker, pos.entry_price)
            close_position(
                state=state,
                exit_ts=last_ts,
                reason="end_mark",
                exit_price=px - slippage,
                settlement_outcome=None,
            )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_events).sort_values("ts") if equity_events else pd.DataFrame()
    final_cash = float(cash)
    total_pnl = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
    wins = int((trades_df["pnl"] > 0).sum()) if not trades_df.empty else 0
    trade_count = int(len(trades_df))

    summary = {
        "initial_cash": float(initial_cash),
        "final_cash": final_cash,
        "total_pnl": total_pnl,
        "roi": (final_cash - initial_cash) / initial_cash if initial_cash else 0.0,
        "trades": trade_count,
        "wins": wins,
        "win_rate": (wins / trade_count) if trade_count else 0.0,
        "avg_trade_pnl": float(trades_df["pnl"].mean()) if not trades_df.empty else 0.0,
        "median_trade_pnl": float(trades_df["pnl"].median()) if not trades_df.empty else 0.0,
        "skipped_for_cash_or_capacity": int(skipped_cash),
        "rows_total": int(total_rows),
        "rows_crypto": int(crypto_rows),
        "rows_used": int(len(df)),
        "unique_crypto_tickers": int(df["ticker"].nunique()),
        "start_ts": str(df["ts"].min()) if not df.empty else None,
        "end_ts": str(df["ts"].max()) if not df.empty else None,
        "params": {
            "stake": float(stake),
            "lookback_bars": int(lookback_bars),
            "low_quantile": float(low_quantile),
            "high_quantile": float(high_quantile),
            "take_profit": float(take_profit),
            "stop_loss": float(stop_loss),
            "force_exit_minutes": float(force_exit_minutes),
            "fee_multiplier": float(fee_multiplier),
            "tick_size": float(tick_size),
            "slippage_ticks": int(slippage_ticks),
            "max_open_positions": int(max_open_positions),
            "latency_bars": int(latency_bars),
            "base_fill_prob": float(base_fill_prob),
            "min_fill_prob": float(min_fill_prob),
            "spread_fill_penalty": float(spread_fill_penalty),
            "liquidity_fill_boost": float(liquidity_fill_boost),
            "partial_fills": bool(partial_fills),
            "impact_coeff": float(impact_coeff),
            "intrabar_noise": float(intrabar_noise),
            "order_reject_prob": float(order_reject_prob),
            "outage_prob": float(outage_prob),
            "outage_min_bars": int(outage_min_bars),
            "outage_max_bars": int(outage_max_bars),
            "random_seed": int(random_seed),
        },
        "execution_stats": {k: int(v) for k, v in sorted(execution_stats.items())},
        "exit_reasons": dict(exit_reasons),
        "generated_at": _utcnow_iso(),
    }
    return BacktestResult(summary=summary, trades=trades_df, equity=equity_df)


def save_backtest_outputs(result: BacktestResult, output_dir: Path, label: str) -> None:
    ensure_dir(output_dir)
    prefix = label.strip() if label.strip() else "cryptoconclave"
    result.trades.to_csv(output_dir / f"{prefix}_trades.csv", index=False)
    result.equity.to_csv(output_dir / f"{prefix}_equity.csv", index=False)
    (output_dir / f"{prefix}_summary.json").write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    try:
        pdf_path = generate_backtest_report(result, output_dir=output_dir, label=prefix)
        logger.info("Backtest report saved to %s", pdf_path)
    except Exception as exc:
        logger.warning("Failed to generate backtest PDF report: %s", exc)


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    running_max = values.cummax()
    drawdown = (values / running_max) - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _longest_streak(flags: List[bool], target: bool) -> int:
    best = 0
    cur = 0
    for flag in flags:
        if bool(flag) == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _prepare_report_tables(result: BacktestResult) -> Dict[str, Any]:
    summary = dict(result.summary or {})
    trades = result.trades.copy()
    equity = result.equity.copy()

    if not trades.empty:
        for col in ("entry_ts", "exit_ts", "close_ts"):
            if col in trades.columns:
                trades[col] = pd.to_datetime(trades[col], errors="coerce", utc=True)
        trades = trades.sort_values("exit_ts" if "exit_ts" in trades.columns else trades.columns[0]).reset_index(drop=True)
        if "entry_ts" in trades.columns and "exit_ts" in trades.columns:
            hold_min = (trades["exit_ts"] - trades["entry_ts"]).dt.total_seconds() / 60.0
            trades["holding_minutes"] = hold_min
    if not equity.empty and "ts" in equity.columns:
        equity["ts"] = pd.to_datetime(equity["ts"], errors="coerce", utc=True)
        equity = equity.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    initial_cash = float(summary.get("initial_cash", 0.0))
    total_pnl = float(trades["pnl"].sum()) if not trades.empty and "pnl" in trades.columns else 0.0
    trades_count = int(len(trades))
    wins = int((trades["pnl"] > 0).sum()) if not trades.empty and "pnl" in trades.columns else 0
    losses = int((trades["pnl"] <= 0).sum()) if not trades.empty and "pnl" in trades.columns else 0
    win_rate = (wins / trades_count) if trades_count else 0.0

    avg_pnl = float(trades["pnl"].mean()) if trades_count else 0.0
    median_pnl = float(trades["pnl"].median()) if trades_count else 0.0
    pnl_std = float(trades["pnl"].std(ddof=1)) if trades_count > 1 else 0.0
    avg_roi = float(trades["roi"].mean()) if trades_count and "roi" in trades.columns else 0.0
    median_roi = float(trades["roi"].median()) if trades_count and "roi" in trades.columns else 0.0

    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum()) if trades_count else 0.0
    gross_loss = float(trades.loc[trades["pnl"] < 0, "pnl"].sum()) if trades_count else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else np.inf
    avg_win = float(trades.loc[trades["pnl"] > 0, "pnl"].mean()) if wins else 0.0
    avg_loss = float(trades.loc[trades["pnl"] < 0, "pnl"].mean()) if losses else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    sharpe_like = 0.0
    if pnl_std > 0 and trades_count > 1:
        sharpe_like = (avg_pnl / pnl_std) * np.sqrt(trades_count)

    win_flags = trades["pnl"].gt(0).tolist() if trades_count and "pnl" in trades.columns else []
    longest_win = _longest_streak(win_flags, True)
    longest_loss = _longest_streak(win_flags, False)

    if not equity.empty and "mark_to_market" in equity.columns:
        mtm = pd.to_numeric(equity["mark_to_market"], errors="coerce").dropna()
        max_dd_mtm = _max_drawdown(mtm) if not mtm.empty else 0.0
    else:
        max_dd_mtm = 0.0

    if trades_count and "pnl" in trades.columns:
        cum = trades["pnl"].cumsum() + initial_cash
        max_dd_realized = _max_drawdown(cum)
    else:
        max_dd_realized = 0.0

    best_trade = float(trades["pnl"].max()) if trades_count and "pnl" in trades.columns else 0.0
    worst_trade = float(trades["pnl"].min()) if trades_count and "pnl" in trades.columns else 0.0

    hold_minutes = pd.to_numeric(trades.get("holding_minutes"), errors="coerce").dropna() if trades_count else pd.Series(dtype=float)
    avg_hold_h = float((hold_minutes.mean() / 60.0)) if not hold_minutes.empty else 0.0
    med_hold_h = float((hold_minutes.median() / 60.0)) if not hold_minutes.empty else 0.0

    out = {
        "summary": summary,
        "trades": trades,
        "equity": equity,
        "metrics": {
            "initial_cash": initial_cash,
            "final_cash": float(summary.get("final_cash", initial_cash + total_pnl)),
            "total_pnl": total_pnl,
            "roi": float(summary.get("roi", 0.0)),
            "trades": trades_count,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "median_pnl": median_pnl,
            "avg_roi": avg_roi,
            "median_roi": median_roi,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "pnl_std": pnl_std,
            "sharpe_like": sharpe_like,
            "max_drawdown_mark_to_market": max_dd_mtm,
            "max_drawdown_realized": max_dd_realized,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "longest_win_streak": longest_win,
            "longest_loss_streak": longest_loss,
            "avg_hold_h": avg_hold_h,
            "median_hold_h": med_hold_h,
        },
    }
    return out


def generate_backtest_report(result: BacktestResult, output_dir: Path, label: str) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = ensure_dir(output_dir)
    pdf_path = output_dir / f"{label}_report.pdf"

    prepared = _prepare_report_tables(result)
    summary = prepared["summary"]
    trades = prepared["trades"]
    equity = prepared["equity"]
    m = prepared["metrics"]

    def money(x: float) -> str:
        return f"${x:,.2f}"

    def pct(x: float) -> str:
        return f"{x * 100.0:.2f}%"

    def num(x: float) -> str:
        return f"{x:,.4f}"

    def short_label(value: Any, max_len: int = 28) -> str:
        text = str(value) if value is not None else ""
        return text if len(text) <= max_len else (text[: max_len - 1] + "...")

    def no_data(ax, message: str) -> None:
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    def style_date_axis(ax) -> None:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    params = dict(summary.get("params") or {})
    exit_reasons = dict(summary.get("exit_reasons") or {})
    execution_stats = dict(summary.get("execution_stats") or {})

    modeled_items = [
        "Entry and exit slippage: slippage_ticks * tick_size",
        "Kalshi-style fees: fee_multiplier * p * (1-p), rounded up",
        "Latency bars before execution attempts (latency_bars)",
        "Queue/liquidity-adjusted probabilistic fills",
        "Partial fills on entries and exits (configurable)",
        "Order rejection probability and exchange outage windows",
        "Market impact and intrabar adverse execution noise",
        "Position sizing by stake budget and available cash",
        "Concurrent position cap (max_open_positions)",
        "Forced pre-close exits and settlement resolution",
    ]
    not_modeled_items = [
        "Exact L2 queue position, hidden liquidity, and cancellations by others",
        "True intrabar OHLC/microsecond tape path (hourly proxy only)",
        "Broker/exchange-specific throttling and auth edge-cases",
        "Cross-market portfolio constraints/margin and borrow effects",
        "Self-impact feedback loops from repeated strategy activity",
    ]

    with PdfPages(pdf_path) as pdf:
        # Page 1: executive summary, assumptions, equity + drawdown.
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("Cryptoconclave Backtest Report", fontsize=18, fontweight="bold", y=0.985)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.25, 1.0], hspace=0.36, wspace=0.28)

        ax_summary = fig.add_subplot(gs[0, 0])
        ax_summary.axis("off")
        summary_rows = [
            ("Initial cash", money(m["initial_cash"])),
            ("Final cash", money(m["final_cash"])),
            ("Total PnL", money(m["total_pnl"])),
            ("ROI", pct(m["roi"])),
            ("Trades", str(int(m["trades"]))),
            ("Win rate", pct(m["win_rate"])),
            ("Avg trade PnL", money(m["avg_pnl"])),
            ("Median trade PnL", money(m["median_pnl"])),
            ("Profit factor", num(m["profit_factor"]) if np.isfinite(m["profit_factor"]) else "inf"),
            ("Expectancy", money(m["expectancy"])),
            ("Sharpe-like (trade)", num(m["sharpe_like"])),
            ("Max DD (MTM)", pct(m["max_drawdown_mark_to_market"])),
            ("Max DD (realized)", pct(m["max_drawdown_realized"])),
            ("Avg hold (hours)", f"{m['avg_hold_h']:.2f}"),
        ]
        tbl = ax_summary.table(
            cellText=[[k, v] for k, v in summary_rows],
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.18)
        ax_summary.set_title("Performance Summary", fontsize=12, pad=8)

        ax_assump = fig.add_subplot(gs[0, 1])
        ax_assump.axis("off")
        params_rows = [
            ("stake", params.get("stake", "n/a")),
            ("lookback_bars", params.get("lookback_bars", "n/a")),
            ("low_quantile", params.get("low_quantile", "n/a")),
            ("high_quantile", params.get("high_quantile", "n/a")),
            ("take_profit", params.get("take_profit", "n/a")),
            ("stop_loss", params.get("stop_loss", "n/a")),
            ("force_exit_minutes", params.get("force_exit_minutes", "n/a")),
            ("fee_multiplier", params.get("fee_multiplier", "n/a")),
            ("tick_size", params.get("tick_size", "n/a")),
            ("slippage_ticks", params.get("slippage_ticks", "n/a")),
            ("latency_bars", params.get("latency_bars", "n/a")),
            ("base_fill_prob", params.get("base_fill_prob", "n/a")),
            ("min_fill_prob", params.get("min_fill_prob", "n/a")),
            ("impact_coeff", params.get("impact_coeff", "n/a")),
            ("intrabar_noise", params.get("intrabar_noise", "n/a")),
            ("order_reject_prob", params.get("order_reject_prob", "n/a")),
            ("outage_prob", params.get("outage_prob", "n/a")),
            ("random_seed", params.get("random_seed", "n/a")),
        ]
        params_tbl = ax_assump.table(
            cellText=[[k, str(v)] for k, v in params_rows],
            colLabels=["Parameter", "Value"],
            cellLoc="left",
            bbox=[0.0, 0.48, 1.0, 0.52],
        )
        params_tbl.auto_set_font_size(False)
        params_tbl.set_fontsize(8.1)
        params_tbl.scale(1, 1.0)

        reason_rows = sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True)
        reason_rows = reason_rows[:8] if reason_rows else [("none", 0)]
        reasons_tbl = ax_assump.table(
            cellText=[[short_label(k, 22), str(v)] for k, v in reason_rows],
            colLabels=["Exit reason", "Count"],
            cellLoc="left",
            bbox=[0.0, 0.30, 0.58, 0.16],
        )
        reasons_tbl.auto_set_font_size(False)
        reasons_tbl.set_fontsize(8.1)
        reasons_tbl.scale(1, 1.0)

        exec_rows = sorted(execution_stats.items(), key=lambda x: x[0])
        exec_rows = exec_rows[:10] if exec_rows else [("none", 0)]
        exec_tbl = ax_assump.table(
            cellText=[[short_label(k, 22), str(v)] for k, v in exec_rows],
            colLabels=["Exec stat", "Count"],
            cellLoc="left",
            bbox=[0.60, 0.30, 0.40, 0.16],
        )
        exec_tbl.auto_set_font_size(False)
        exec_tbl.set_fontsize(8.1)
        exec_tbl.scale(1, 1.0)

        modeled_text = "Modeled:\n- " + "\n- ".join(modeled_items)
        not_modeled_text = "Not Modeled:\n- " + "\n- ".join(not_modeled_items)
        ax_assump.text(0.0, 0.28, modeled_text, va="top", fontsize=7.9)
        ax_assump.text(0.0, 0.11, not_modeled_text, va="top", fontsize=7.9)
        ax_assump.text(
            0.0,
            0.01,
            f"Window: {summary.get('start_ts', 'n/a')} -> {summary.get('end_ts', 'n/a')}\n"
            f"Rows used: {summary.get('rows_used', 0)} | Crypto rows: {summary.get('rows_crypto', 0)} | "
            f"Tickers: {summary.get('unique_crypto_tickers', 0)}",
            fontsize=8.2,
            va="bottom",
            color="#444444",
        )
        ax_assump.set_title("Assumptions and Backtest Context", fontsize=12, pad=8)

        ax_curve = fig.add_subplot(gs[1, :])
        if not equity.empty and "ts" in equity.columns:
            if "cash" in equity.columns:
                ax_curve.plot(equity["ts"], equity["cash"], label="Cash", linewidth=1.5, color="#2C7FB8")
            if "mark_to_market" in equity.columns:
                ax_curve.plot(equity["ts"], equity["mark_to_market"], label="Mark-to-market", linewidth=1.3, color="#F58518")
            ax_curve.set_title("Equity Curve")
            ax_curve.set_ylabel("Value ($)")
            style_date_axis(ax_curve)
            ax_curve.grid(True, alpha=0.25)
            ax_curve.legend(loc="upper left")
        else:
            no_data(ax_curve, "No equity data available")

        ax_draw = fig.add_subplot(gs[2, :])
        if not equity.empty and "mark_to_market" in equity.columns:
            mtm = pd.to_numeric(equity["mark_to_market"], errors="coerce")
            runmax = mtm.cummax()
            dd = ((mtm / runmax) - 1.0).fillna(0.0)
            ax_draw.fill_between(equity["ts"], dd.values, 0.0, color="#E45756", alpha=0.35, step=None)
            ax_draw.plot(equity["ts"], dd.values, color="#E45756", linewidth=1.1)
            ax_draw.set_title("Drawdown (Mark-to-market)")
            ax_draw.set_ylabel("Drawdown")
            style_date_axis(ax_draw)
            ax_draw.set_ylim(min(float(dd.min()) * 1.1, -0.01), 0.02)
            ax_draw.grid(True, alpha=0.25)
        else:
            no_data(ax_draw, "No drawdown data available")

        fig.subplots_adjust(top=0.95, bottom=0.04, left=0.07, right=0.98)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: trade distributions and progression.
        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.suptitle("Trade-Level Diagnostics", fontsize=18, fontweight="bold", y=0.985)
        gs2 = fig2.add_gridspec(2, 2, hspace=0.34, wspace=0.25)

        ax_pnl_hist = fig2.add_subplot(gs2[0, 0])
        if not trades.empty and "pnl" in trades.columns:
            pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna()
            ax_pnl_hist.hist(pnl.values, bins=40, color="#4C78A8", alpha=0.85)
            ax_pnl_hist.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
            ax_pnl_hist.set_title("PnL per Trade Distribution")
            ax_pnl_hist.set_xlabel("PnL ($)")
            ax_pnl_hist.grid(True, alpha=0.2)
        else:
            no_data(ax_pnl_hist, "No trade PnL data")

        ax_roi_hist = fig2.add_subplot(gs2[0, 1])
        if not trades.empty and "roi" in trades.columns:
            roi = pd.to_numeric(trades["roi"], errors="coerce").dropna() * 100.0
            ax_roi_hist.hist(roi.values, bins=40, color="#72B7B2", alpha=0.85)
            ax_roi_hist.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
            ax_roi_hist.set_title("ROI % per Trade Distribution")
            ax_roi_hist.set_xlabel("ROI (%)")
            ax_roi_hist.grid(True, alpha=0.2)
        else:
            no_data(ax_roi_hist, "No trade ROI data")

        ax_cum = fig2.add_subplot(gs2[1, 0])
        if not trades.empty and "pnl" in trades.columns:
            cum_pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0).cumsum()
            ax_cum.plot(np.arange(1, len(cum_pnl) + 1), cum_pnl.values, color="#54A24B", linewidth=1.4)
            ax_cum.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
            ax_cum.set_title("Cumulative PnL by Trade")
            ax_cum.set_xlabel("Trade Number")
            ax_cum.set_ylabel("Cumulative PnL ($)")
            ax_cum.grid(True, alpha=0.2)
        else:
            no_data(ax_cum, "No cumulative trade data")

        ax_hold = fig2.add_subplot(gs2[1, 1])
        if not trades.empty and "holding_minutes" in trades.columns:
            hold_hours = pd.to_numeric(trades["holding_minutes"], errors="coerce").dropna() / 60.0
            if not hold_hours.empty:
                ax_hold.hist(hold_hours.values, bins=40, color="#F58518", alpha=0.9)
                ax_hold.set_title("Holding Time Distribution")
                ax_hold.set_xlabel("Hours")
                ax_hold.grid(True, alpha=0.2)
            else:
                no_data(ax_hold, "No holding-time data")
        else:
            no_data(ax_hold, "No holding-time data")

        fig2.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.98)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # Page 3: ticker and temporal breakdown.
        fig3 = plt.figure(figsize=(8.5, 11))
        fig3.suptitle("Cross-Sectional and Temporal Breakdown", fontsize=18, fontweight="bold", y=0.985)
        gs3 = fig3.add_gridspec(2, 2, hspace=0.34, wspace=0.32)

        ax_top = fig3.add_subplot(gs3[0, 0])
        if not trades.empty and "ticker" in trades.columns and "pnl" in trades.columns:
            by_ticker = trades.groupby("ticker", as_index=False).agg(
                trades=("ticker", "count"),
                pnl=("pnl", "sum"),
                win_rate=("won", "mean"),
            )
            top = by_ticker.sort_values("pnl", ascending=False).head(10).sort_values("pnl", ascending=True)
            labels = [short_label(x, 24) for x in top["ticker"].tolist()]
            ax_top.barh(labels, top["pnl"], color="#4C78A8", alpha=0.9)
            ax_top.set_title("Top 10 Tickers by Total PnL")
            ax_top.grid(True, axis="x", alpha=0.2)
        else:
            no_data(ax_top, "No ticker breakdown data")

        ax_worst = fig3.add_subplot(gs3[0, 1])
        if not trades.empty and "ticker" in trades.columns and "pnl" in trades.columns:
            by_ticker = trades.groupby("ticker", as_index=False)["pnl"].sum()
            worst = by_ticker.sort_values("pnl", ascending=True).head(10)
            labels = [short_label(x, 24) for x in worst["ticker"].tolist()]
            ax_worst.barh(labels, worst["pnl"], color="#E45756", alpha=0.9)
            ax_worst.set_title("Worst 10 Tickers by Total PnL")
            ax_worst.grid(True, axis="x", alpha=0.2)
        else:
            no_data(ax_worst, "No ticker loss data")

        ax_month = fig3.add_subplot(gs3[1, 0])
        if not trades.empty and "exit_ts" in trades.columns and "pnl" in trades.columns:
            work = trades.dropna(subset=["exit_ts"]).copy()
            if not work.empty:
                work["month_ts"] = work["exit_ts"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
                by_month = work.groupby("month_ts", as_index=False)["pnl"].sum().sort_values("month_ts")
                colors = ["#54A24B" if v >= 0 else "#E45756" for v in by_month["pnl"].values]
                ax_month.bar(by_month["month_ts"], by_month["pnl"], color=colors, alpha=0.9, width=20)
                ax_month.set_title("Monthly Realized PnL")
                style_date_axis(ax_month)
                ax_month.grid(True, axis="y", alpha=0.2)
            else:
                no_data(ax_month, "No monthly data")
        else:
            no_data(ax_month, "No monthly data")

        ax_reasons = fig3.add_subplot(gs3[1, 1])
        if not trades.empty and "reason" in trades.columns:
            reason_counts = trades["reason"].fillna("unknown").astype(str).value_counts().head(10)
            labels = [short_label(x, 18) for x in reason_counts.index.tolist()]
            ax_reasons.bar(labels, reason_counts.values, color="#F58518", alpha=0.9)
            ax_reasons.set_title("Exit Reason Frequency")
            ax_reasons.tick_params(axis="x", rotation=25)
            ax_reasons.grid(True, axis="y", alpha=0.2)
        else:
            no_data(ax_reasons, "No reason data")

        fig3.subplots_adjust(top=0.95, bottom=0.05, left=0.09, right=0.98)
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        # Page 4: advanced diagnostics.
        fig4 = plt.figure(figsize=(8.5, 11))
        fig4.suptitle("Advanced Diagnostics", fontsize=18, fontweight="bold", y=0.985)
        gs4 = fig4.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

        ax_week = fig4.add_subplot(gs4[0, 0])
        if not trades.empty and "exit_ts" in trades.columns and "pnl" in trades.columns:
            work = trades.dropna(subset=["exit_ts"]).copy()
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if not work.empty:
                work["weekday"] = work["exit_ts"].dt.day_name()
                by_week = work.groupby("weekday", as_index=False)["pnl"].mean().set_index("weekday").reindex(order)
                yvals = by_week["pnl"].values.astype(float)
                xvals = np.arange(len(order))
                colors = ["#54A24B" if (not np.isnan(v) and v >= 0) else "#E45756" for v in yvals]
                ax_week.bar(xvals, np.nan_to_num(yvals, nan=0.0), color=colors, alpha=0.9)
                ax_week.set_xticks(xvals, [x[:3] for x in order])
                ax_week.set_title("Average PnL by Exit Weekday")
                ax_week.grid(True, axis="y", alpha=0.2)
            else:
                no_data(ax_week, "No weekday data")
        else:
            no_data(ax_week, "No weekday data")

        ax_hour = fig4.add_subplot(gs4[0, 1])
        if not trades.empty and "entry_ts" in trades.columns and "pnl" in trades.columns:
            work = trades.dropna(subset=["entry_ts"]).copy()
            if not work.empty:
                work["entry_hour"] = work["entry_ts"].dt.hour
                by_hour = work.groupby("entry_hour", as_index=False).agg(
                    avg_pnl=("pnl", "mean"),
                    trades=("pnl", "count"),
                )
                ax_hour.bar(by_hour["entry_hour"], by_hour["avg_pnl"], color="#4C78A8", alpha=0.9)
                ax_hour.set_title("Average PnL by Entry Hour (UTC)")
                ax_hour.set_xlabel("Hour")
                ax_hour.set_ylabel("Avg PnL ($)")
                ax_hour.grid(True, axis="y", alpha=0.2)
            else:
                no_data(ax_hour, "No hour data")
        else:
            no_data(ax_hour, "No hour data")

        ax_roll = fig4.add_subplot(gs4[1, 0])
        if not trades.empty and "won" in trades.columns:
            won_series = pd.to_numeric(trades["won"], errors="coerce").fillna(0.0)
            window = min(50, max(5, len(won_series) // 5 if len(won_series) > 10 else 5))
            rolling = won_series.rolling(window=window, min_periods=max(2, window // 2)).mean()
            ax_roll.plot(np.arange(1, len(rolling) + 1), rolling.values * 100.0, color="#72B7B2", linewidth=1.4)
            ax_roll.axhline(50.0, color="#666666", linestyle="--", linewidth=1.0)
            ax_roll.set_title(f"Rolling Win Rate ({window}-trade window)")
            ax_roll.set_xlabel("Trade Number")
            ax_roll.set_ylabel("Win Rate (%)")
            ax_roll.set_ylim(0, 100)
            ax_roll.grid(True, alpha=0.2)
        else:
            no_data(ax_roll, "No rolling win-rate data")

        ax_scatter = fig4.add_subplot(gs4[1, 1])
        if not trades.empty and "entry_price" in trades.columns and "pnl" in trades.columns:
            x = pd.to_numeric(trades["entry_price"], errors="coerce")
            y = pd.to_numeric(trades["pnl"], errors="coerce")
            ok = x.notna() & y.notna()
            if int(ok.sum()) > 0:
                ax_scatter.scatter(x[ok], y[ok], s=18, alpha=0.5, color="#F58518")
                if int(ok.sum()) >= 3:
                    slope, intercept = np.polyfit(x[ok], y[ok], 1)
                    xs = np.linspace(float(x[ok].min()), float(x[ok].max()), 100)
                    ys = (slope * xs) + intercept
                    ax_scatter.plot(xs, ys, color="#333333", linewidth=1.2, linestyle="--", label="Trend")
                    corr = float(np.corrcoef(x[ok], y[ok])[0, 1]) if int(ok.sum()) > 2 else 0.0
                    ax_scatter.legend(title=f"corr={corr:.2f}", loc="upper right")
                ax_scatter.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
                ax_scatter.set_title("Entry Price vs Trade PnL")
                ax_scatter.set_xlabel("Entry Price (YES probability)")
                ax_scatter.set_ylabel("PnL ($)")
                ax_scatter.grid(True, alpha=0.2)
            else:
                no_data(ax_scatter, "No scatter data")
        else:
            no_data(ax_scatter, "No scatter data")

        fig4.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.98)
        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)

    return pdf_path


def _extract_orderbook_sides(orderbook: Dict[str, Any]) -> Dict[str, List[Any]]:
    if "yes_bids" in orderbook or "no_bids" in orderbook:
        return {"yes": orderbook.get("yes_bids", []), "no": orderbook.get("no_bids", [])}
    ob = orderbook.get("orderbook")
    if isinstance(ob, dict):
        return {"yes": ob.get("yes", []), "no": ob.get("no", [])}
    return {"yes": [], "no": []}


def _best_bid(levels: List[Any]) -> float | None:
    if not levels:
        return None
    first = levels[0]
    if isinstance(first, dict):
        return to_dollars(first.get("price"))
    if isinstance(first, (list, tuple)) and first:
        return to_dollars(first[0])
    return None


def _yes_market_prices(market: Dict[str, Any], orderbook: Dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    sides = _extract_orderbook_sides(orderbook)
    yes_bid = _best_bid(sides["yes"])
    no_bid = _best_bid(sides["no"])
    market_yes_bid = to_dollars(market.get("yes_bid") or market.get("best_yes_bid"))
    market_no_bid = to_dollars(market.get("no_bid") or market.get("best_no_bid"))
    market_yes_ask = to_dollars(market.get("yes_ask") or market.get("best_yes_ask"))

    if yes_bid is None:
        yes_bid = market_yes_bid
    if no_bid is None:
        no_bid = market_no_bid
    ask_yes = market_yes_ask if market_yes_ask is not None else ((1.0 - no_bid) if no_bid is not None else None)
    if yes_bid is None and ask_yes is None:
        return None, None, None
    if yes_bid is None:
        mid = ask_yes
    elif ask_yes is None:
        mid = yes_bid
    else:
        mid = (yes_bid + ask_yes) / 2.0
    return yes_bid, ask_yes, mid


def _load_live_state(path: Path, initial_cash: float) -> Dict[str, Any]:
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            state.setdefault("history", {})
            state.setdefault("open_positions", {})
            state.setdefault("cash", float(initial_cash))
            state.setdefault("loops_completed", 0)
            return state
        except Exception as exc:
            logger.warning("Could not parse %s (%s). Starting fresh live state.", path, exc)
    return {"history": {}, "open_positions": {}, "cash": float(initial_cash), "loops_completed": 0}


def _save_live_state(path: Path, state: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def _post_json(client: KalshiClient, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    headers = None
    if getattr(client, "_signer", None):
        full_path = client._full_path_for_sign(path)  # noqa: SLF001
        headers = client._signer.headers_for("POST", full_path)  # noqa: SLF001
    response = client._client.request("POST", path, headers=headers, json=payload)  # noqa: SLF001
    if response.status_code >= 400:
        raise RuntimeError(f"Kalshi order POST failed ({response.status_code}): {response.text[:300]}")
    if not response.content:
        return {}
    return response.json()


def _build_order_payload(
    *,
    ticker: str,
    side: str,
    action: str,
    price: float,
    contracts: int,
) -> Dict[str, Any]:
    cents = int(round(float(np.clip(price, 0.001, 0.999)) * 100))
    payload: Dict[str, Any] = {
        "ticker": ticker,
        "type": "limit",
        "action": action.lower(),
        "side": side.lower(),
        "count": int(contracts),
        "client_order_id": f"cryptoconclave-{uuid4().hex[:20]}",
    }
    if side.upper() == "YES":
        payload["yes_price"] = cents
    else:
        payload["no_price"] = cents
    return payload


def run_live(
    *,
    output_dir: Path,
    initial_cash: float,
    stake: float,
    lookback_bars: int,
    low_quantile: float,
    high_quantile: float,
    take_profit: float,
    stop_loss: float,
    fee_multiplier: float,
    tick_size: float,
    slippage_ticks: int,
    max_markets: int,
    poll_seconds: float,
    once: bool,
    place_live_orders: bool,
    order_endpoint: str,
) -> None:
    ensure_dir(output_dir)
    state_path = output_dir / "live_state.json"
    actions_path = output_dir / "live_actions.jsonl"
    state = _load_live_state(state_path, initial_cash=initial_cash)

    lookback_bars = max(int(lookback_bars), 2)
    low_quantile = float(np.clip(low_quantile, 0.01, 0.49))
    high_quantile = float(np.clip(high_quantile, 0.51, 0.99))
    slippage = max(int(slippage_ticks), 0) * float(tick_size)
    dry_run = not bool(place_live_orders)

    client = KalshiClient(
        KalshiClientConfig(
            base_url=SETTINGS.base_url,
            access_key_id=SETTINGS.access_key_id,
            private_key_path=SETTINGS.private_key_path,
            private_key_pem=SETTINGS.private_key_pem,
            verify=SETTINGS.ssl_verify,
            request_delay=SETTINGS.request_delay,
            timeout=SETTINGS.timeout,
        )
    )
    try:
        while True:
            loop_ts = _utcnow_iso()
            loops_completed = int(state.get("loops_completed", 0)) + 1
            state["loops_completed"] = loops_completed

            markets = download_live_markets(
                client=client,
                cache_dir=SETTINGS.cache_dir,
                status_filter=["open"],
                max_markets=max_markets,
            )
            crypto_markets = [m for m in markets if _market_is_crypto(m)]

            actions_this_loop = 0
            for market in crypto_markets:
                ticker = str(market.get("ticker") or market.get("market_ticker") or "")
                if not ticker:
                    continue
                try:
                    orderbook = client.get(f"/markets/{ticker}/orderbook")
                except Exception as exc:
                    logger.warning("Orderbook fetch failed for %s: %s", ticker, exc)
                    continue

                _, _, mid = _yes_market_prices(market, orderbook)
                if mid is None:
                    continue
                mid = float(np.clip(mid, 0.001, 0.999))

                hist_map: Dict[str, List[float]] = state.setdefault("history", {})
                hist = hist_map.setdefault(ticker, [])
                hist_arr = np.array(hist, dtype=float) if hist else np.array([], dtype=float)
                open_positions: Dict[str, Dict[str, Any]] = state.setdefault("open_positions", {})
                pos = open_positions.get(ticker)

                if hist_arr.size >= lookback_bars:
                    low = float(np.quantile(hist_arr, low_quantile))
                    high = float(np.quantile(hist_arr, high_quantile))

                    if pos is None and mid <= low:
                        entry_price = float(np.clip(mid + slippage, 0.001, 0.999))
                        entry_fee = fee_dollars(entry_price, fee_multiplier)
                        cost_per = entry_price + entry_fee
                        contracts = int(stake // cost_per) if stake > 0 else 1
                        contracts = max(contracts, 1)
                        cost = contracts * cost_per
                        if float(state.get("cash", 0.0)) >= cost:
                            payload = _build_order_payload(
                                ticker=ticker,
                                side="YES",
                                action="buy",
                                price=entry_price,
                                contracts=contracts,
                            )
                            api_response = {"dry_run": True}
                            if not dry_run:
                                api_response = _post_json(client, order_endpoint, payload)
                            state["cash"] = float(state.get("cash", 0.0)) - cost
                            open_positions[ticker] = {
                                "entry_ts": loop_ts,
                                "entry_price": entry_price,
                                "entry_fee": entry_fee,
                                "contracts": contracts,
                                "cost": cost,
                                "title": market.get("title"),
                            }
                            actions_this_loop += 1
                            _append_jsonl(
                                actions_path,
                                {
                                    "ts": loop_ts,
                                    "event": "buy",
                                    "ticker": ticker,
                                    "price": entry_price,
                                    "contracts": contracts,
                                    "cost": cost,
                                    "dry_run": dry_run,
                                    "response": api_response,
                                },
                            )

                    elif pos is not None:
                        entry_price = float(pos.get("entry_price") or mid)
                        hit_high = mid >= high
                        hit_take_profit = mid >= entry_price + take_profit
                        hit_stop = mid <= max(entry_price - stop_loss, 0.001)
                        if hit_high or hit_take_profit or hit_stop:
                            exit_price = float(np.clip(mid - slippage, 0.001, 0.999))
                            exit_fee = fee_dollars(exit_price, fee_multiplier)
                            proceeds = float(pos.get("contracts", 0)) * max(exit_price - exit_fee, 0.0)
                            payload = _build_order_payload(
                                ticker=ticker,
                                side="YES",
                                action="sell",
                                price=exit_price,
                                contracts=int(pos.get("contracts", 0)),
                            )
                            api_response = {"dry_run": True}
                            if not dry_run:
                                api_response = _post_json(client, order_endpoint, payload)
                            state["cash"] = float(state.get("cash", 0.0)) + proceeds
                            pnl = proceeds - float(pos.get("cost", 0.0))
                            reason = "high_quantile"
                            if hit_take_profit:
                                reason = "take_profit"
                            elif hit_stop:
                                reason = "stop_loss"
                            actions_this_loop += 1
                            _append_jsonl(
                                actions_path,
                                {
                                    "ts": loop_ts,
                                    "event": "sell",
                                    "ticker": ticker,
                                    "price": exit_price,
                                    "contracts": int(pos.get("contracts", 0)),
                                    "proceeds": proceeds,
                                    "pnl": pnl,
                                    "reason": reason,
                                    "dry_run": dry_run,
                                    "response": api_response,
                                },
                            )
                            open_positions.pop(ticker, None)

                hist.append(mid)
                if len(hist) > lookback_bars * 4:
                    del hist[: len(hist) - lookback_bars * 4]

            state["updated_at"] = loop_ts
            _save_live_state(state_path, state)
            logger.info(
                "Live loop %s complete: markets=%s crypto=%s actions=%s cash=%.2f dry_run=%s",
                loops_completed,
                len(markets),
                len(crypto_markets),
                actions_this_loop,
                float(state.get("cash", 0.0)),
                dry_run,
            )
            if once:
                break
            time.sleep(max(float(poll_seconds), 1.0))
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cryptoconclave: Kalshi crypto buy-low/sell-high backtest + live loop.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")

    sub = parser.add_subparsers(dest="command", required=True)

    p_backtest = sub.add_parser("backtest", help="Backtest buy-low/sell-high strategy on longshotbias crypto candles")
    p_backtest.add_argument("--processed-dir", type=str, default=SETTINGS.processed_dir, help="Path containing candle_observations")
    p_backtest.add_argument("--output-dir", type=str, default=str(Path(SETTINGS.output_dir) / "cryptoconclave"), help="Output directory")
    p_backtest.add_argument("--label", type=str, default="cryptoconclave_backtest", help="Output filename prefix")
    p_backtest.add_argument("--initial-cash", type=float, default=1000.0, help="Starting bankroll")
    p_backtest.add_argument("--stake", type=float, default=20.0, help="Max dollars per entry")
    p_backtest.add_argument("--lookback-bars", type=int, default=12, help="Rolling bars used to define local highs/lows")
    p_backtest.add_argument("--low-quantile", type=float, default=0.2, help="Buy threshold quantile")
    p_backtest.add_argument("--high-quantile", type=float, default=0.8, help="Sell threshold quantile")
    p_backtest.add_argument("--take-profit", type=float, default=0.08, help="Take-profit in price points")
    p_backtest.add_argument("--stop-loss", type=float, default=0.08, help="Stop-loss in price points")
    p_backtest.add_argument("--force-exit-minutes", type=float, default=60.0, help="Force exit before close")
    p_backtest.add_argument("--fee-multiplier", type=float, default=SETTINGS.default_fee_multiplier, help="Kalshi fee multiplier")
    p_backtest.add_argument("--tick-size", type=float, default=SETTINGS.tick_size, help="Tick size in dollars")
    p_backtest.add_argument("--slippage-ticks", type=int, default=1, help="Entry/exit slippage ticks")
    p_backtest.add_argument("--max-open-positions", type=int, default=100, help="Cap simultaneous positions")
    p_backtest.add_argument("--max-rows", type=int, default=None, help="Optional row cap for smoke tests")
    p_backtest.add_argument("--latency-bars", type=int, default=1, help="Order latency in bars before execution attempt")
    p_backtest.add_argument("--base-fill-prob", type=float, default=0.90, help="Base probability an order gets filled")
    p_backtest.add_argument("--min-fill-prob", type=float, default=0.10, help="Minimum fill probability floor")
    p_backtest.add_argument("--spread-fill-penalty", type=float, default=8.0, help="Higher spreads reduce fill probability")
    p_backtest.add_argument("--liquidity-fill-boost", type=float, default=1200.0, help="Liquidity scale that boosts fill probability")
    p_backtest.add_argument("--partial-fills", action=argparse.BooleanOptionalAction, default=True, help="Allow partial fills")
    p_backtest.add_argument("--impact-coeff", type=float, default=0.03, help="Market impact coefficient on execution price")
    p_backtest.add_argument("--intrabar-noise", type=float, default=0.75, help="Within-bar adverse price noise multiplier")
    p_backtest.add_argument("--order-reject-prob", type=float, default=0.01, help="Probability an order gets rejected")
    p_backtest.add_argument("--outage-prob", type=float, default=0.0008, help="Per-row probability of exchange outage")
    p_backtest.add_argument("--outage-min-bars", type=int, default=3, help="Minimum outage duration in rows")
    p_backtest.add_argument("--outage-max-bars", type=int, default=20, help="Maximum outage duration in rows")
    p_backtest.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible fills/outages")

    p_live = sub.add_parser("live", help="Run live API loop with the same buy-low/sell-high logic")
    p_live.add_argument("--output-dir", type=str, default=str(Path(SETTINGS.output_dir) / "cryptoconclave"), help="Output directory")
    p_live.add_argument("--initial-cash", type=float, default=1000.0, help="Paper cash tracking for live loop")
    p_live.add_argument("--stake", type=float, default=20.0, help="Max dollars per entry")
    p_live.add_argument("--lookback-bars", type=int, default=12, help="Rolling bars used to define local highs/lows")
    p_live.add_argument("--low-quantile", type=float, default=0.2, help="Buy threshold quantile")
    p_live.add_argument("--high-quantile", type=float, default=0.8, help="Sell threshold quantile")
    p_live.add_argument("--take-profit", type=float, default=0.08, help="Take-profit in price points")
    p_live.add_argument("--stop-loss", type=float, default=0.08, help="Stop-loss in price points")
    p_live.add_argument("--fee-multiplier", type=float, default=SETTINGS.default_fee_multiplier, help="Kalshi fee multiplier")
    p_live.add_argument("--tick-size", type=float, default=SETTINGS.tick_size, help="Tick size in dollars")
    p_live.add_argument("--slippage-ticks", type=int, default=1, help="Entry/exit slippage ticks")
    p_live.add_argument("--max-markets", type=int, default=500, help="Max open markets to fetch each loop")
    p_live.add_argument("--poll-seconds", type=float, default=60.0, help="Sleep between loops")
    p_live.add_argument("--once", action="store_true", help="Run a single loop")
    p_live.add_argument("--place-live-orders", action="store_true", help="POST real orders (default is dry-run)")
    p_live.add_argument("--order-endpoint", type=str, default="/portfolio/orders", help="Kalshi order endpoint path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if args.command == "backtest":
        result = run_backtest(
            processed_dir=Path(args.processed_dir),
            initial_cash=args.initial_cash,
            stake=args.stake,
            lookback_bars=args.lookback_bars,
            low_quantile=args.low_quantile,
            high_quantile=args.high_quantile,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            force_exit_minutes=args.force_exit_minutes,
            fee_multiplier=args.fee_multiplier,
            tick_size=args.tick_size,
            slippage_ticks=args.slippage_ticks,
            max_open_positions=args.max_open_positions,
            max_rows=args.max_rows,
            latency_bars=args.latency_bars,
            base_fill_prob=args.base_fill_prob,
            min_fill_prob=args.min_fill_prob,
            spread_fill_penalty=args.spread_fill_penalty,
            liquidity_fill_boost=args.liquidity_fill_boost,
            partial_fills=args.partial_fills,
            impact_coeff=args.impact_coeff,
            intrabar_noise=args.intrabar_noise,
            order_reject_prob=args.order_reject_prob,
            outage_prob=args.outage_prob,
            outage_min_bars=args.outage_min_bars,
            outage_max_bars=args.outage_max_bars,
            random_seed=args.random_seed,
        )
        save_backtest_outputs(result, Path(args.output_dir), label=args.label)
        logger.info("Backtest complete: %s", json.dumps(result.summary, indent=2))
        return

    if args.command == "live":
        run_live(
            output_dir=Path(args.output_dir),
            initial_cash=args.initial_cash,
            stake=args.stake,
            lookback_bars=args.lookback_bars,
            low_quantile=args.low_quantile,
            high_quantile=args.high_quantile,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            fee_multiplier=args.fee_multiplier,
            tick_size=args.tick_size,
            slippage_ticks=args.slippage_ticks,
            max_markets=args.max_markets,
            poll_seconds=args.poll_seconds,
            once=args.once,
            place_live_orders=args.place_live_orders,
            order_endpoint=args.order_endpoint,
        )
        return


if __name__ == "__main__":
    main()
