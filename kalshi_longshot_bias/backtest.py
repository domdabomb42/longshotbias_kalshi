"""Backtest the bias model on historical candle data."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .bias_metrics import fee_dollars
from .config import SETTINGS
from .model import BiasModel


@dataclass
class BacktestResult:
    summary: Dict[str, Any]
    trades: pd.DataFrame
    equity: pd.DataFrame


def build_price_index(candle_obs: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    df = candle_obs[["ticker", "ts", "price"]].copy()
    df = df.dropna(subset=["ticker", "ts", "price"])
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"]).sort_values(["ticker", "ts"])
    index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for ticker, group in df.groupby("ticker"):
        ts = group["ts"].astype("int64").to_numpy()
        price = group["price"].astype(float).to_numpy()
        index[str(ticker)] = (ts, price)
    return index


def _filter_bid_through(
    entries: pd.DataFrame,
    price_index: Dict[str, Tuple[np.ndarray, np.ndarray]],
    min_touches: int = 2,
    max_hours: float = 12.0,
    latency_candles: int = 1,
) -> pd.DataFrame:
    if entries.empty:
        return entries

    entry_ts = pd.to_datetime(entries["ts"], errors="coerce", utc=True)
    close_ts = pd.to_datetime(entries["close_ts"], errors="coerce", utc=True)
    entry_ts_int = entry_ts.astype("int64")
    close_ts_int = close_ts.astype("int64")
    entry_ts_na = entry_ts.isna()
    close_ts_na = close_ts.isna()

    filled_mask = []
    for idx, row in entries.iterrows():
        ticker = str(row.get("ticker"))
        series = price_index.get(ticker)
        if series is None:
            filled_mask.append(False)
            continue
        ts_arr, price_arr = series
        if entry_ts_na.loc[idx] or close_ts_na.loc[idx]:
            filled_mask.append(False)
            continue
        start = entry_ts_int.loc[idx]
        end = close_ts_int.loc[idx]
        if end <= start:
            filled_mask.append(False)
            continue
        left = np.searchsorted(ts_arr, start, side="right") + max(latency_candles, 0)
        max_end = end
        if max_hours and max_hours > 0:
            max_end = min(max_end, start + int(max_hours * 3600 * 1e9))
        right = np.searchsorted(ts_arr, max_end, side="right")
        if left >= right:
            filled_mask.append(False)
            continue
        window = price_arr[left:right]
        side = str(row.get("side"))
        price = float(row.get("price_raw", row.get("price", np.nan)))
        if np.isnan(price):
            filled_mask.append(False)
            continue
        if side == "YES":
            touches = int(np.sum(window <= price))
            filled_mask.append(touches >= max(min_touches, 1))
        else:
            threshold = 1.0 - price
            touches = int(np.sum(window >= threshold))
            filled_mask.append(touches >= max(min_touches, 1))

    return entries.loc[pd.Series(filled_mask, index=entries.index)]


def load_candle_observations(processed_dir: Path) -> pd.DataFrame:
    parquet_path = processed_dir / "candle_observations.parquet"
    csv_path = processed_dir / "candle_observations.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Candle observations not found. Run build step first.")


def _select_entry_rows(
    df: pd.DataFrame,
    target_days: float,
) -> pd.DataFrame:
    target_hours = target_days * 24.0
    df = df.copy()
    df["diff_hours"] = (df["time_to_close_hours"] - target_hours).abs()
    df = df.sort_values(["ticker", "diff_hours", "time_to_close_hours"], ascending=[True, True, False])
    return df.groupby("ticker", as_index=False).first()


def run_backtest(
    candle_obs: pd.DataFrame,
    model: BiasModel,
    initial_cash: float,
    stake: float,
    min_ev: float,
    fee_multiplier: float,
    target_days: float = 7.0,
    max_trades: int | None = None,
    strategy: str = "model_ev",
    favorite_threshold: float = 0.9,
    underdog_threshold: float = 0.1,
    price_mode: str = "ask",
    price_index: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None,
    slippage_ticks: int = 1,
    adverse_selection_penalty: float = 0.02,
    bid_through_min_touches: int = 2,
    bid_through_max_hours: float = 12.0,
    bid_through_latency_candles: int = 1,
) -> BacktestResult:
    df = candle_obs.copy()
    df = df.dropna(subset=["implied_prob", "outcome", "close_ts"])
    if "ts" not in df.columns:
        raise ValueError("Candle observations missing timestamp column 'ts'.")

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df["close_ts"] = pd.to_datetime(df["close_ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts", "close_ts"])

    df["time_to_close_hours"] = (df["close_ts"] - df["ts"]).dt.total_seconds() / 3600.0
    df = df[(df["time_to_close_hours"] >= 0) & df["implied_prob"].between(0.001, 0.999)]

    entries = _select_entry_rows(df, target_days=target_days)

    entries["q_hat"] = entries.apply(
        lambda r: model.predict_one(
            float(r["implied_prob"]),
            str(r.get("category_mapped") or "").lower() or None,
            str(r.get("structure") or "").lower() or None,
        ),
        axis=1,
    )
    def _select_price(col: str, fallback: pd.Series) -> pd.Series:
        if col in entries.columns:
            values = pd.to_numeric(entries[col], errors="coerce")
            values = values.where(values.between(0.0, 1.0))
            return values.fillna(fallback)
        return fallback

    implied = entries["implied_prob"].astype(float)
    price_mode_key = (price_mode or "ask").lower()
    if price_mode_key in {"bid", "bid_through"}:
        entries["price_yes_raw"] = _select_price("yes_bid", implied)
        entries["price_no_raw"] = _select_price("no_bid", 1.0 - implied)
    else:
        entries["price_yes_raw"] = _select_price("yes_ask", implied)
        entries["price_no_raw"] = _select_price("no_ask", 1.0 - implied)

    slip = max(slippage_ticks, 0) * SETTINGS.tick_size
    entries["price_yes"] = entries["price_yes_raw"] + slip
    entries["price_no"] = entries["price_no_raw"] + slip

    entries.loc[~entries["price_yes"].between(0.001, 0.999), "price_yes"] = np.nan
    entries.loc[~entries["price_no"].between(0.001, 0.999), "price_no"] = np.nan

    entries["fee_yes"] = entries["price_yes"].apply(lambda p: fee_dollars(p, fee_multiplier) if pd.notna(p) else np.nan)
    entries["fee_no"] = entries["price_no"].apply(lambda p: fee_dollars(p, fee_multiplier) if pd.notna(p) else np.nan)
    spread_est = None
    if "yes_ask" in entries.columns and "yes_bid" in entries.columns:
        spread_est = pd.to_numeric(entries["yes_ask"], errors="coerce") - pd.to_numeric(entries["yes_bid"], errors="coerce")
        spread_est = spread_est.where(spread_est > 0, 0.0)
    else:
        spread_est = 0.0

    penalty = adverse_selection_penalty + 0.5 * spread_est
    if price_mode_key in {"bid", "bid_through"}:
        penalty = penalty + adverse_selection_penalty
    penalty = penalty.fillna(adverse_selection_penalty) if isinstance(penalty, pd.Series) else penalty

    q_adj_yes = (entries["q_hat"] - penalty).clip(lower=0.0, upper=1.0)
    q_adj_no = ((1.0 - entries["q_hat"]) - penalty).clip(lower=0.0, upper=1.0)

    entries["ev_yes"] = q_adj_yes - entries["price_yes"] - entries["fee_yes"]
    entries["ev_no"] = q_adj_no - entries["price_no"] - entries["fee_no"]

    strategy_key = (strategy or "model_ev").lower()
    if strategy_key == "model_ev":
        choose_yes = entries["ev_yes"] >= entries["ev_no"]
        entries["side"] = np.where(choose_yes, "YES", "NO")
        entries["ev"] = np.where(choose_yes, entries["ev_yes"], entries["ev_no"])
        entries["price_raw"] = np.where(choose_yes, entries["price_yes_raw"], entries["price_no_raw"])
        entries["price"] = np.where(choose_yes, entries["price_yes"], entries["price_no"])
        entries["fee"] = np.where(choose_yes, entries["fee_yes"], entries["fee_no"])
    elif strategy_key == "favorites_yes":
        entries = entries[entries["price_yes"] >= favorite_threshold].copy()
        entries["side"] = "YES"
        entries["ev"] = entries["ev_yes"]
        entries["price_raw"] = entries["price_yes_raw"]
        entries["price"] = entries["price_yes"]
        entries["fee"] = entries["fee_yes"]
    elif strategy_key == "underdogs_no":
        entries = entries[entries["price_yes"] <= underdog_threshold].copy()
        entries["side"] = "NO"
        entries["ev"] = entries["ev_no"]
        entries["price_raw"] = entries["price_no_raw"]
        entries["price"] = entries["price_no"]
        entries["fee"] = entries["fee_no"]
    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")

    if strategy_key in {"favorites_yes", "underdogs_no"}:
        # Enforce positive model-EV for these strategies regardless of --min-ev.
        entries = entries[entries["ev"] > 0].copy()
    entries = entries[entries["ev"] > min_ev].copy()

    if price_mode_key == "bid_through":
        if price_index is None:
            price_index = build_price_index(candle_obs)
        entries = _filter_bid_through(
            entries,
            price_index,
            min_touches=bid_through_min_touches,
            max_hours=bid_through_max_hours,
            latency_candles=bid_through_latency_candles,
        )

    entries = entries.sort_values("ts").reset_index(drop=True)

    cash = float(initial_cash)
    trades: list[Dict[str, Any]] = []
    equity_events: list[Dict[str, Any]] = []
    open_positions: list[Dict[str, Any]] = []
    total_trades = 0
    wins = 0
    skipped = 0

    def settle_positions(current_ts: pd.Timestamp) -> None:
        nonlocal cash, wins
        remaining = []
        for pos in open_positions:
            if pos["close_ts"] <= current_ts:
                cash += pos["payout"]
                wins += pos["win"]
                equity_events.append(
                    {
                        "ts": pos["close_ts"],
                        "event": "settle",
                        "cash": cash,
                        "open_positions": len(open_positions) - 1,
                    }
                )
            else:
                remaining.append(pos)
        open_positions[:] = remaining

    from .utils import progress_iter
    bar = progress_iter(entries.iterrows(), total=len(entries), desc="backtest", unit="trade")
    for _, row in bar:
        if hasattr(bar, "set_postfix_str") and "entry_ts" in row:
            try:
                bar.set_postfix_str(str(pd.to_datetime(row["entry_ts"]).date()))
            except Exception:
                pass
        if max_trades and total_trades >= max_trades:
            break
        entry_ts = row["ts"]
        settle_positions(entry_ts)

        price = float(row["price"])
        fee = float(row["fee"])
        cost_per = price + fee
        if cost_per <= 0:
            skipped += 1
            continue

        if stake > 0 and cost_per > stake:
            skipped += 1
            continue
        contracts = int(stake // cost_per) if stake > 0 else 1
        if contracts < 1:
            contracts = 1
        cost = contracts * cost_per
        if cash < cost:
            skipped += 1
            continue

        cash -= cost
        outcome = float(row["outcome"])
        side = str(row["side"])
        win = 1 if (side == "YES" and outcome == 1) or (side == "NO" and outcome == 0) else 0
        payout = contracts * (1.0 if win else 0.0)
        pnl = payout - cost

        open_positions.append(
            {
                "close_ts": row["close_ts"],
                "payout": payout,
                "win": win,
            }
        )

        trades.append(
            {
                "ticker": row.get("ticker"),
                "title": row.get("title"),
                "entry_ts": entry_ts,
                "close_ts": row.get("close_ts"),
                "side": side,
                "strategy": strategy_key,
                "price": price,
                "fee": fee,
                "q_hat": float(row["q_hat"]),
                "ev": float(row["ev"]),
                "contracts": contracts,
                "cost": cost,
                "payout": payout,
                "pnl": pnl,
                "outcome": outcome,
                "win": win,
                "category": row.get("category_mapped"),
                "structure": row.get("structure"),
                "time_to_close_hours": float(row.get("time_to_close_hours") or 0.0),
            }
        )
        total_trades += 1
        equity_events.append(
            {
                "ts": entry_ts,
                "event": "entry",
                "cash": cash,
                "open_positions": len(open_positions),
            }
        )

    if open_positions:
        last_close = max(pos["close_ts"] for pos in open_positions)
        settle_positions(last_close)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_events).sort_values("ts") if equity_events else pd.DataFrame()

    final_cash = cash
    total_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0.0
    win_rate = (wins / total_trades) if total_trades else 0.0
    avg_pct_return = 0.0
    if not trades_df.empty and "cost" in trades_df.columns:
        valid = trades_df[trades_df["cost"] > 0].copy()
        if not valid.empty:
            avg_pct_return = float((valid["pnl"] / valid["cost"]).mean())

    summary = {
        "strategy": strategy_key,
        "price_mode": price_mode_key,
        "initial_cash": initial_cash,
        "final_cash": final_cash,
        "total_pnl": total_pnl,
        "roi": (final_cash - initial_cash) / initial_cash if initial_cash else 0.0,
        "trades": total_trades,
        "wins": wins,
        "win_rate": win_rate,
        "avg_pct_return_per_bet": avg_pct_return * 100.0,
        "skipped": skipped,
        "min_ev": min_ev,
        "stake": stake,
        "target_days": target_days,
        "favorite_threshold": favorite_threshold,
        "underdog_threshold": underdog_threshold,
    }
    return BacktestResult(summary=summary, trades=trades_df, equity=equity_df)


def save_backtest_result(result: BacktestResult, output_dir: Path, label: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "backtest" if not label else f"backtest_{label}"
    result.trades.to_csv(output_dir / f"{prefix}_trades.csv", index=False)
    result.equity.to_csv(output_dir / f"{prefix}_equity.csv", index=False)
    (output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(result.summary, indent=2), encoding="utf-8"
    )


def _load_backtest_artifacts(output_dir: Path, label: str | None = None) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    prefix = "backtest" if not label else f"backtest_{label}"
    summary_path = output_dir / f"{prefix}_summary.json"
    trades_path = output_dir / f"{prefix}_trades.csv"
    equity_path = output_dir / f"{prefix}_equity.csv"

    if not summary_path.exists() or not trades_path.exists() or not equity_path.exists():
        raise FileNotFoundError(f"Backtest outputs not found for '{label or 'model'}'.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trades = pd.read_csv(trades_path)
    equity = pd.read_csv(equity_path)
    return summary, trades, equity


def _prep_equity(equity: pd.DataFrame) -> pd.DataFrame:
    equity = equity.copy()
    if not equity.empty and "ts" in equity.columns:
        equity["ts"] = pd.to_datetime(equity["ts"], errors="coerce", utc=True)
        equity = equity.dropna(subset=["ts"]).copy()
        try:
            equity["ts"] = equity["ts"].dt.tz_convert(None)
        except Exception:
            pass
    return equity


def _value_series(trades: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    if trades.empty or "close_ts" not in trades.columns or "pnl" not in trades.columns:
        return pd.DataFrame()
    value_series = trades.copy()
    value_series["close_ts"] = pd.to_datetime(value_series["close_ts"], errors="coerce", utc=True)
    value_series = value_series.dropna(subset=["close_ts"]).copy()
    try:
        value_series["close_ts"] = value_series["close_ts"].dt.tz_convert(None)
    except Exception:
        pass
    if value_series.empty:
        return pd.DataFrame()
    value_series = (
        value_series.sort_values("close_ts")
        .groupby("close_ts", as_index=False)["pnl"]
        .sum()
    )
    value_series["value"] = initial_cash + value_series["pnl"].cumsum()
    return value_series


def _render_backtest_pages_compare(
    summary_ask: Dict[str, Any],
    trades_ask: pd.DataFrame,
    equity_ask: pd.DataFrame,
    summary_bid: Dict[str, Any],
    trades_bid: pd.DataFrame,
    equity_bid: pd.DataFrame,
    summary_through: Dict[str, Any],
    trades_through: pd.DataFrame,
    equity_through: pd.DataFrame,
    title: str,
    pdf: "PdfPages",
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    equity_ask = _prep_equity(equity_ask)
    equity_bid = _prep_equity(equity_bid)
    equity_through = _prep_equity(equity_through)

    def fmt_money(value: float) -> str:
        return f"${value:,.2f}"

    def fmt_pct(value: float) -> str:
        return f"{value:.2f}%"

    def col(value: str, width: int = 20) -> str:
        return f"{value:>{width}}"

    min_ev_ask = f"{float(summary_ask.get('min_ev', 0.0)):.4f}"
    min_ev_bid = f"{float(summary_bid.get('min_ev', 0.0)):.4f}"
    min_ev_through = f"{float(summary_through.get('min_ev', 0.0)):.4f}"
    target_ask = f"{float(summary_ask.get('target_days', 0.0)):.2f}"
    target_bid = f"{float(summary_bid.get('target_days', 0.0)):.2f}"
    target_through = f"{float(summary_through.get('target_days', 0.0)):.2f}"

    summary_lines = [
        title,
        "Metric                              | Ask (taker)           | Bid (slippage)       | Bid (through)",
        "-" * 98,
        f"Initial cash                        | {col(fmt_money(float(summary_ask.get('initial_cash', 0.0))))} | {col(fmt_money(float(summary_bid.get('initial_cash', 0.0))))} | {col(fmt_money(float(summary_through.get('initial_cash', 0.0))))}",
        f"Final cash                          | {col(fmt_money(float(summary_ask.get('final_cash', 0.0))))} | {col(fmt_money(float(summary_bid.get('final_cash', 0.0))))} | {col(fmt_money(float(summary_through.get('final_cash', 0.0))))}",
        f"Total PnL                           | {col(fmt_money(float(summary_ask.get('total_pnl', 0.0))))} | {col(fmt_money(float(summary_bid.get('total_pnl', 0.0))))} | {col(fmt_money(float(summary_through.get('total_pnl', 0.0))))}",
        f"ROI                                 | {col(fmt_pct(float(summary_ask.get('roi', 0.0)) * 100.0))} | {col(fmt_pct(float(summary_bid.get('roi', 0.0)) * 100.0))} | {col(fmt_pct(float(summary_through.get('roi', 0.0)) * 100.0))}",
        f"Trades                              | {col(str(int(summary_ask.get('trades', 0))))} | {col(str(int(summary_bid.get('trades', 0))))} | {col(str(int(summary_through.get('trades', 0))))}",
        f"Wins                                | {col(str(int(summary_ask.get('wins', 0))))} | {col(str(int(summary_bid.get('wins', 0))))} | {col(str(int(summary_through.get('wins', 0))))}",
        f"Win rate                            | {col(fmt_pct(float(summary_ask.get('win_rate', 0.0)) * 100.0))} | {col(fmt_pct(float(summary_bid.get('win_rate', 0.0)) * 100.0))} | {col(fmt_pct(float(summary_through.get('win_rate', 0.0)) * 100.0))}",
        f"Avg % return per bet                | {col(fmt_pct(float(summary_ask.get('avg_pct_return_per_bet', 0.0))))} | {col(fmt_pct(float(summary_bid.get('avg_pct_return_per_bet', 0.0))))} | {col(fmt_pct(float(summary_through.get('avg_pct_return_per_bet', 0.0))))}",
        f"Skipped                             | {col(str(int(summary_ask.get('skipped', 0))))} | {col(str(int(summary_bid.get('skipped', 0))))} | {col(str(int(summary_through.get('skipped', 0))))}",
        f"Min EV                              | {col(min_ev_ask)} | {col(min_ev_bid)} | {col(min_ev_through)}",
        f"Stake                               | {col(fmt_money(float(summary_ask.get('stake', 0.0))))} | {col(fmt_money(float(summary_bid.get('stake', 0.0))))} | {col(fmt_money(float(summary_through.get('stake', 0.0))))}",
        f"Target days to close                | {col(target_ask)} | {col(target_bid)} | {col(target_through)}",
    ]

    fig1 = plt.figure(figsize=(8.5, 11))
    fig1.suptitle("Backtest Report", fontsize=18, fontweight="bold")
    gs1 = fig1.add_gridspec(4, 1, height_ratios=[1.2, 2.4, 2.4, 0.6])

    ax_summary = fig1.add_subplot(gs1[0, 0])
    ax_summary.axis("off")
    ax_summary.text(
        0.01,
        0.98,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )

    ax_equity = fig1.add_subplot(gs1[1, 0])
    if not equity_ask.empty and "cash" in equity_ask.columns:
        ax_equity.plot(equity_ask["ts"], equity_ask["cash"], label="Ask (taker)", color="#2C7FB8")
    if not equity_bid.empty and "cash" in equity_bid.columns:
        ax_equity.plot(equity_bid["ts"], equity_bid["cash"], label="Bid (slippage)", color="#F58518")
    if not equity_through.empty and "cash" in equity_through.columns:
        ax_equity.plot(equity_through["ts"], equity_through["cash"], label="Bid (through)", color="#54A24B")
    if equity_ask.empty and equity_bid.empty and equity_through.empty:
        ax_equity.text(0.5, 0.5, "Equity curve unavailable", ha="center", va="center")
    ax_equity.set_title("Equity Curve (Cash)")
    ax_equity.set_ylabel("Cash")
    ax_equity.legend(loc="upper left")
    ax_equity.grid(True, alpha=0.2)

    ax_value = fig1.add_subplot(gs1[2, 0])
    value_series_ask = _value_series(trades_ask, float(summary_ask.get("initial_cash", 0.0)))
    value_series_bid = _value_series(trades_bid, float(summary_bid.get("initial_cash", 0.0)))
    value_series_through = _value_series(trades_through, float(summary_through.get("initial_cash", 0.0)))

    if not value_series_ask.empty:
        ax_value.plot(value_series_ask["close_ts"], value_series_ask["value"], color="#2C7FB8", label="Ask (taker)")
    if not value_series_bid.empty:
        ax_value.plot(value_series_bid["close_ts"], value_series_bid["value"], color="#F58518", label="Bid (slippage)")
    if not value_series_through.empty:
        ax_value.plot(value_series_through["close_ts"], value_series_through["value"], color="#54A24B", label="Bid (through)")
    if value_series_ask.empty and value_series_bid.empty and value_series_through.empty:
        ax_value.text(0.5, 0.5, "Portfolio value unavailable", ha="center", va="center")
    ax_value.set_title("Portfolio Value (Realized at Settlement)")
    ax_value.set_ylabel("Value")
    ax_value.legend(loc="upper left")
    ax_value.grid(True, alpha=0.2)

    ax_note = fig1.add_subplot(gs1[3, 0])
    ax_note.axis("off")
    ax_note.text(
        0.01,
        0.5,
        "Note: Portfolio value only updates on settlement; entries do not reduce value until a loss is realized.",
        fontsize=9,
        color="#666666",
    )

    fig1.tight_layout(rect=[0, 0.02, 1, 0.97])
    pdf.savefig(fig1, bbox_inches="tight")
    plt.close(fig1)

    # Category breakdown page
    fig2 = plt.figure(figsize=(8.5, 11))
    fig2.suptitle(f"Category Breakdown - {title}", fontsize=18, fontweight="bold")
    gs2 = fig2.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 1.2])

    def _cat_stats(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        work = df.copy()
        if "category" in work.columns:
            work["category"] = work["category"].fillna("Unknown").astype(str)
        else:
            work["category"] = "Unknown"
        stats = work.groupby("category").agg(
            trades=("ticker", "count"),
            avg_pnl=("pnl", "mean"),
            win_rate=("win", "mean"),
            avg_cost=("cost", "mean"),
        )
        stats["avg_roi"] = stats.apply(
            lambda r: (r["avg_pnl"] / r["avg_cost"]) if r["avg_cost"] > 0 else 0.0,
            axis=1,
        )
        return stats

    cat_ask = _cat_stats(trades_ask)
    cat_bid = _cat_stats(trades_bid)
    cat_through = _cat_stats(trades_through)
    if not cat_ask.empty:
        cat_top = cat_ask.sort_values("trades", ascending=False).head(15)
    elif not cat_bid.empty:
        cat_top = cat_bid.sort_values("trades", ascending=False).head(15)
    elif not cat_through.empty:
        cat_top = cat_through.sort_values("trades", ascending=False).head(15)
    else:
        cat_top = pd.DataFrame()

    categories = cat_top.index.tolist() if not cat_top.empty else []
    if categories:
        cat_ask = cat_ask.reindex(categories)
        cat_bid = cat_bid.reindex(categories)
        cat_through = cat_through.reindex(categories)

    ax_cat_pnl = fig2.add_subplot(gs2[0, 0])
    if categories:
        y = np.arange(len(categories))
        ax_cat_pnl.barh(y - 0.25, cat_ask["avg_pnl"].fillna(0).values, height=0.25, label="Ask", color="#4C78A8")
        ax_cat_pnl.barh(y, cat_bid["avg_pnl"].fillna(0).values, height=0.25, label="Bid", color="#F58518")
        ax_cat_pnl.barh(y + 0.25, cat_through["avg_pnl"].fillna(0).values, height=0.25, label="Through", color="#54A24B")
        ax_cat_pnl.set_yticks(y, labels=[str(c) for c in categories])
        ax_cat_pnl.set_title("Avg PnL per Trade by Category (Top 15)")
        ax_cat_pnl.set_xlabel("Avg PnL")
        ax_cat_pnl.legend(loc="lower right")
    else:
        ax_cat_pnl.text(0.5, 0.5, "Category PnL unavailable", ha="center", va="center")
    ax_cat_pnl.grid(True, alpha=0.2)

    ax_cat_wr = fig2.add_subplot(gs2[1, 0])
    if categories:
        y = np.arange(len(categories))
        ax_cat_wr.barh(y - 0.25, cat_ask["win_rate"].fillna(0).values, height=0.25, label="Ask", color="#72B7B2")
        ax_cat_wr.barh(y, cat_bid["win_rate"].fillna(0).values, height=0.25, label="Bid", color="#F58518")
        ax_cat_wr.barh(y + 0.25, cat_through["win_rate"].fillna(0).values, height=0.25, label="Through", color="#54A24B")
        ax_cat_wr.set_yticks(y, labels=[str(c) for c in categories])
        ax_cat_wr.set_title("Win Rate by Category (Top 15)")
        ax_cat_wr.set_xlabel("Win Rate")
        ax_cat_wr.set_xlim(0, 1)
        ax_cat_wr.legend(loc="lower right")
    else:
        ax_cat_wr.text(0.5, 0.5, "Category win rate unavailable", ha="center", va="center")
    ax_cat_wr.grid(True, alpha=0.2)

    ax_cat_roi = fig2.add_subplot(gs2[2, 0])
    if categories:
        y = np.arange(len(categories))
        ax_cat_roi.barh(y - 0.25, cat_ask["avg_roi"].fillna(0).values, height=0.25, label="Ask", color="#4C78A8")
        ax_cat_roi.barh(y, cat_bid["avg_roi"].fillna(0).values, height=0.25, label="Bid", color="#F58518")
        ax_cat_roi.barh(y + 0.25, cat_through["avg_roi"].fillna(0).values, height=0.25, label="Through", color="#54A24B")
        ax_cat_roi.set_yticks(y, labels=[str(c) for c in categories])
        ax_cat_roi.set_title("Avg ROI per Trade by Category (Top 15)")
        ax_cat_roi.set_xlabel("Avg ROI")
        ax_cat_roi.legend(loc="lower right")
    else:
        ax_cat_roi.text(0.5, 0.5, "Category ROI unavailable", ha="center", va="center")
    ax_cat_roi.grid(True, alpha=0.2)

    fig2.tight_layout(rect=[0, 0.02, 1, 0.97])
    pdf.savefig(fig2, bbox_inches="tight")
    plt.close(fig2)


def generate_backtest_report(output_dir: Path, labels: Tuple[Tuple[str | None, str, str, str], ...] | None = None) -> Path:
    """Generate a multi-page PDF report for backtest outputs."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = output_dir / "backtest_report.pdf"
    with PdfPages(pdf_path) as pdf:
        default_labels: Tuple[Tuple[str | None, str, str, str], ...] = (
            (None, "bid", "through", "Model EV Strategy"),
            ("favorites", "favorites_bid", "favorites_through", "Heavy Favorites (YES)"),
            ("underdogs", "underdogs_bid", "underdogs_through", "Heavy Underdogs (NO)"),
        )
        for ask_label, bid_label, through_label, title in labels or default_labels:
            summary_ask, trades_ask, equity_ask = _load_backtest_artifacts(output_dir, label=ask_label)
            summary_bid, trades_bid, equity_bid = _load_backtest_artifacts(output_dir, label=bid_label)
            summary_through, trades_through, equity_through = _load_backtest_artifacts(output_dir, label=through_label)
            _render_backtest_pages_compare(
                summary_ask,
                trades_ask,
                equity_ask,
                summary_bid,
                trades_bid,
                equity_bid,
                summary_through,
                trades_through,
                equity_through,
                title=title,
                pdf=pdf,
            )

    return pdf_path
