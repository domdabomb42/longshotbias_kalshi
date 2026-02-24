"""Command line interface for the Kalshi longshot bias pipeline."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
import sys
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .backtest import (
    build_price_index,
    generate_backtest_report,
    load_candle_observations,
    run_backtest,
    save_backtest_result,
)
from .bias_metrics import build_bias_stats, build_bias_stats_candles, plot_calibration, plot_roi_by_bin, save_report
from .config import SETTINGS
from .ev_scanner import EV_COLUMNS, evaluate_market_ev, fetch_fee_multipliers, scan_positive_ev
from .features import build_observations, build_candle_observations
from .ingest import backfill_candles_from_api, backfill_candles_from_cache, download_historical, download_live_markets, download_orderbooks, load_raw
from .kalshi_client import KalshiClient, KalshiClientConfig
from .model import build_correction_curves, load_model, save_model, train_models, walk_forward_validation
from .paper_trading import run_paper_trading
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def _write_rejection_report(stats: Counter, output_dir: Path, scanned: int) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "rejections_report.pdf"

    total_attempts = stats.get("candidate_attempts", 0)
    positive = stats.get("positive_ev", 0)
    markets_scanned = stats.get("markets_scanned", scanned)

    # Filter out meta keys
    exclude = {"candidate_attempts", "positive_ev", "markets_scanned"}
    reasons = {k: v for k, v in stats.items() if k not in exclude}

    # Sort reasons by count
    reason_items = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in reason_items]
    values = [v for _, v in reason_items]

    try:
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle("Live Scan Rejection Report", fontsize=18, fontweight="bold")
            ax_text = fig.add_subplot(2, 1, 1)
            ax_text.axis("off")
            lines = [
                f"Markets scanned: {markets_scanned}",
                f"Candidate attempts: {total_attempts}",
                f"Positive EV: {positive}",
            ]
            if total_attempts:
                lines.append(f"Positive EV rate: {positive / total_attempts:.2%}")
            ax_text.text(0.01, 0.95, "\n".join(lines), va="top", fontsize=11)

            ax = fig.add_subplot(2, 1, 2)
            if values:
                ax.barh(labels[::-1], values[::-1], color="#4C78A8")
                ax.set_title("Rejection Reasons (count)")
                ax.set_xlabel("Count")
            else:
                ax.text(0.5, 0.5, "No rejection data", ha="center", va="center")
            ax.grid(True, axis="x", alpha=0.2)

            fig.tight_layout(rect=[0, 0.02, 1, 0.95])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        return pdf_path
    except Exception as exc:
        logger.warning("Failed to write rejection report: %s", exc)
        return None

def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Keep progress bars readable by muting chatty HTTP client logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _client() -> KalshiClient:
    return KalshiClient(
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


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)


def _require_dates(args: argparse.Namespace) -> None:
    if args.start is None:
        args.start = datetime(2024, 4, 1)
        logger.info("No --start provided; defaulting to 2024-04-01.")
    if args.end is None:
        args.end = datetime.utcnow()
        logger.info("No --end provided; defaulting to now (UTC).")


def cmd_ingest(args: argparse.Namespace) -> None:
    _require_dates(args)
    if getattr(args, "use_cache", False):
        raw_dir = Path(SETTINGS.cache_dir)
        required = [
            raw_dir / "historical_markets.jsonl",
            raw_dir / "historical_market_details.jsonl",
            raw_dir / "historical_candlesticks.jsonl",
        ]
        if all(path.exists() and path.stat().st_size > 0 for path in required):
            logger.info("Using cached historical data; skipping download.")
            return
        logger.warning("Cached historical data missing or empty; cannot use --use-cache.")
        return

    client = _client()
    try:
        if getattr(args, "backfill_candles", False):
            backfill_candles_from_api(
                client=client,
                cache_dir=SETTINGS.cache_dir,
                start=args.start,
                end=args.end,
                candle_interval_sec=SETTINGS.candle_interval_sec,
                candle_window_days=SETTINGS.candle_window_days,
                candle_chunk_hours=SETTINGS.candle_chunk_hours,
                horizons_days=SETTINGS.horizons_days,
                force_candles=getattr(args, "force_candles", False),
                resume=getattr(args, "resume", False),
                max_markets=args.max_markets,
            )
            return
        newest_first = True
        if getattr(args, "oldest_first", False):
            newest_first = False
        download_historical(
            client,
            start=args.start,
            end=args.end,
            cache_dir=SETTINGS.cache_dir,
            include_candles=True,
            candle_interval_sec=SETTINGS.candle_interval_sec,
            candle_window_days=SETTINGS.candle_window_days,
            candle_chunk_hours=SETTINGS.candle_chunk_hours,
            horizons_days=SETTINGS.horizons_days,
            max_markets=args.max_markets,
            all_history=getattr(args, "all_history", False),
            resume=getattr(args, "resume", False),
            newest_first=newest_first,
            force_candles=getattr(args, "force_candles", False),
        )
    finally:
        client.close()


def cmd_build(args: argparse.Namespace) -> pd.DataFrame:
    raw_dir = Path(SETTINGS.cache_dir)
    from .utils import progress_bar

    bar = progress_bar(total=3, desc="loading raw data", unit="files")
    markets = load_raw(raw_dir / "historical_markets.jsonl")
    bar.update(1)
    details = load_raw(raw_dir / "historical_market_details.jsonl")
    bar.update(1)
    candles = load_raw(raw_dir / "historical_candlesticks.jsonl")
    bar.update(1)
    bar.close()

    observations = build_observations(markets, details, candles, SETTINGS.horizons_days)
    if observations.empty:
        logger.warning("No observations to build stats.")
        return observations

    processed_dir = Path(SETTINGS.processed_dir)
    _save_dataframe(observations, processed_dir / "observations")

    bias_stats = build_bias_stats(
        observations,
        fee_multiplier=SETTINGS.default_fee_multiplier,
        maker_fill_prob=SETTINGS.maker_fill_prob,
        tick_size=SETTINGS.tick_size,
    )
    output_dir = Path(SETTINGS.output_dir)
    ensure_dir(output_dir)
    bias_stats.to_csv(output_dir / "bias_stats.csv", index=False)

    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)
    calibration_path = plots_dir / "calibration.png"
    roi_path = plots_dir / "roi_by_bin.png"
    plot_calibration(observations, calibration_path)
    plot_roi_by_bin(bias_stats, roi_path)

    save_report(
        output_dir=str(output_dir),
        bias_stats=bias_stats,
        calibration_path=str(calibration_path),
        roi_path=str(roi_path),
    )

    candle_obs = build_candle_observations(
        markets,
        details,
        candles,
        resample_minutes=SETTINGS.candle_resample_minutes,
        max_time_to_close_days=SETTINGS.candle_window_days,
    )
    if not candle_obs.empty:
        _save_dataframe(candle_obs, processed_dir / "candle_observations")
        candle_bias = build_bias_stats_candles(
            candle_obs,
            fee_multiplier=SETTINGS.default_fee_multiplier,
            maker_fill_prob=SETTINGS.maker_fill_prob,
            tick_size=SETTINGS.tick_size,
        )
        candle_bias.to_csv(output_dir / "bias_stats_candles.csv", index=False)
    return observations


def cmd_train(args: argparse.Namespace) -> None:
    processed_dir = Path(SETTINGS.processed_dir)
    obs_path_parquet = processed_dir / "observations.parquet"
    obs_path_csv = processed_dir / "observations.csv"
    if obs_path_parquet.exists():
        observations = pd.read_parquet(obs_path_parquet)
    elif obs_path_csv.exists():
        observations = pd.read_csv(obs_path_csv)
    else:
        logger.warning("Observations not found. Run build step first.")
        return

    model = train_models(observations)
    save_model(model, processed_dir / "bias_model.joblib")

    validation = walk_forward_validation(observations)
    validation.to_csv(processed_dir / "walk_forward_validation.csv", index=False)

    curves = build_correction_curves(
        model,
        categories=observations["category_mapped"].dropna().unique().tolist(),
        structures=observations["structure"].dropna().unique().tolist(),
    )
    curves.to_csv(processed_dir / "correction_curves.csv", index=False)


def cmd_scan(args: argparse.Namespace) -> None:
    processed_dir = Path(SETTINGS.processed_dir)
    model_path = processed_dir / "bias_model.joblib"
    if not model_path.exists():
        logger.warning("Model not found. Run train step first.")
        return
    model = load_model(model_path)

    min_volume = getattr(args, "min_volume_24h", SETTINGS.min_volume_24h)
    min_oi = getattr(args, "min_open_interest", SETTINGS.min_open_interest)
    allow_illiquid = getattr(args, "allow_illiquid", SETTINGS.allow_illiquid_live)

    if getattr(args, "use_cache", False):
        raw_dir = Path(SETTINGS.cache_dir)
        markets = load_raw(raw_dir / "live_markets.jsonl")
        orderbooks = load_raw(raw_dir / "live_orderbooks.jsonl")
        if not markets or not orderbooks:
            logger.warning("Cached live data missing; cannot use --use-cache for scan.")
            return
        fee_multipliers = {}
    else:
        client = _client()
        try:
            fetch_limit = args.max_markets or 1000
            markets = download_live_markets(
                client,
                cache_dir=SETTINGS.cache_dir,
                status_filter=[args.live_status] if args.live_status else None,
                max_markets=fetch_limit,
            )
            def _volume_val(m: dict) -> float:
                val = m.get("volume_24h") or m.get("volume24h") or m.get("volume_total") or m.get("volume")
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return 0.0

            def _oi_val(m: dict) -> float:
                val = m.get("open_interest") or m.get("open_int") or m.get("oi")
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return 0.0

            total_markets = len(markets)
            if min_volume or min_oi:
                markets = [m for m in markets if _volume_val(m) >= min_volume and _oi_val(m) >= min_oi]
            markets.sort(key=_volume_val, reverse=True)
            if args.max_markets:
                markets = markets[: args.max_markets]
            elif not args.max_markets:
                # Default to 1,000 markets to keep scan + report cadence manageable.
                markets = markets[:1000]
            logger.info(
                "Scan markets: total=%s, after filters=%s (min_volume_24h=%s, min_open_interest=%s)",
                total_markets,
                len(markets),
                min_volume,
                min_oi,
            )

            if not markets:
                rejection_stats = Counter()
                rejection_stats["markets_scanned"] = 0
                _write_rejection_report(rejection_stats, Path(SETTINGS.output_dir), 0)
                logger.info("No markets to scan; rejection report written.")
                return
            fee_multipliers = fetch_fee_multipliers(client)

            tickers = [
                m.get("ticker") or m.get("market_ticker")
                for m in markets
                if m.get("ticker") or m.get("market_ticker")
            ]
            orderbooks = []
            rejection_stats = Counter()
            scanned = 0
            best_by_ticker: Dict[str, Dict[str, Any]] = {}
            top_rows: List[Dict[str, Any]] = []
            orderbooks_path = Path(SETTINGS.cache_dir) / "live_orderbooks.jsonl"
            ensure_dir(orderbooks_path.parent)
            from .utils import progress_iter
            output_dir = Path(SETTINGS.output_dir)
            ensure_dir(output_dir)
            out_path = output_dir / "positive_ev_bets.csv"
            if out_path.exists():
                out_path.unlink()
            # Ensure file exists with headers even if no trades qualify.
            pd.DataFrame(columns=EV_COLUMNS).to_csv(out_path, index=False)

            def _refresh_top() -> None:
                nonlocal top_rows
                if not best_by_ticker:
                    top_rows = []
                    return
                top_rows = sorted(
                    best_by_ticker.values(),
                    key=lambda r: float(r.get("ev") or 0.0),
                    reverse=True,
                )[:100]

            def _write_top_csv() -> None:
                if top_rows:
                    pd.DataFrame(top_rows, columns=EV_COLUMNS).to_csv(out_path, index=False)
                else:
                    pd.DataFrame(columns=EV_COLUMNS).to_csv(out_path, index=False)
            chunk_size = 1000
            with orderbooks_path.open("w", encoding="utf-8") as f:
                for offset in range(0, len(markets), chunk_size):
                    chunk = markets[offset : offset + chunk_size]
                    bar = progress_iter(chunk, total=len(chunk), desc="orderbooks+scan", unit="mkt")
                    for market in bar:
                        ticker = market.get("ticker") or market.get("market_ticker")
                        if not ticker:
                            rejection_stats["no_ticker"] += 1
                            continue
                        try:
                            ob = client.get(f"/markets/{ticker}/orderbook")
                        except Exception as exc:
                            logger.warning("Failed to fetch orderbook for %s: %s", ticker, exc)
                            rejection_stats["orderbook_error"] += 1
                            continue
                        ob["ticker"] = ticker
                        f.write(json.dumps(ob, ensure_ascii=False) + "\n")
                        orderbooks.append(ob)
                        # evaluate immediately
                        rows = evaluate_market_ev(
                            market=market,
                            orderbook=ob,
                            model=model,
                            default_fee_multiplier=SETTINGS.default_fee_multiplier,
                            fee_multipliers=fee_multipliers,
                            tick_size=SETTINGS.tick_size,
                            max_spread=SETTINGS.max_spread,
                            min_depth=SETTINGS.min_depth,
                            min_ev=SETTINGS.min_ev,
                            maker_fill_prob=SETTINGS.maker_fill_prob,
                            allow_illiquid=allow_illiquid,
                            stats=rejection_stats,
                            slippage_ticks=SETTINGS.slippage_ticks,
                            adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
                        )
                        if rows:
                            best_row = max(rows, key=lambda r: float(r.get("ev") or -1e9))
                            tkr = best_row.get("ticker") or ticker
                            existing = best_by_ticker.get(tkr)
                            if existing is None or float(best_row.get("ev") or 0.0) > float(existing.get("ev") or 0.0):
                                best_by_ticker[tkr] = best_row
                                _refresh_top()
                        scanned += 1
                        rejection_stats["markets_scanned"] = scanned
                        if scanned % 2000 == 0:
                            _write_top_csv()
                            logger.info(
                                "Top 100 EV list written: %s (after %s markets)",
                                out_path,
                                scanned,
                            )
                    # End chunk: generate report and pause
                    report_path = _write_rejection_report(rejection_stats, output_dir, scanned)
                    if report_path:
                        logger.info(
                            "Rejections report generated: %s (after %s markets)",
                            report_path,
                            scanned,
                        )
                    else:
                        logger.warning("Rejections report not generated (after %s markets)", scanned)
                    time.sleep(10)
            # Final write for top list
            _write_top_csv()
        finally:
            client.close()

    if getattr(args, "use_cache", False):
        rejection_stats = Counter()
        df = scan_positive_ev(
            markets=markets,
            orderbooks=orderbooks,
            model=model,
            default_fee_multiplier=SETTINGS.default_fee_multiplier,
            fee_multipliers=fee_multipliers,
            tick_size=SETTINGS.tick_size,
            max_spread=SETTINGS.max_spread,
            min_depth=SETTINGS.min_depth,
            min_ev=SETTINGS.min_ev,
            maker_fill_prob=SETTINGS.maker_fill_prob,
            allow_illiquid=allow_illiquid,
            stats=rejection_stats,
            slippage_ticks=SETTINGS.slippage_ticks,
            adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
        )
        output_dir = Path(SETTINGS.output_dir)
        ensure_dir(output_dir)
        df.to_csv(output_dir / "positive_ev_bets.csv", index=False)
        _write_rejection_report(rejection_stats, output_dir, len(markets))
    else:
        output_dir = Path(SETTINGS.output_dir)
        if (output_dir / "positive_ev_bets.csv").exists():
            df = pd.read_csv(output_dir / "positive_ev_bets.csv")
        else:
            df = pd.DataFrame()

    # Update report with latest EV opportunities if present
    bias_stats_path = output_dir / "bias_stats.csv"
    if bias_stats_path.exists():
        bias_stats = pd.read_csv(bias_stats_path)
        save_report(output_dir=str(output_dir), bias_stats=bias_stats, top_ev=df)


def cmd_paper_trade(args: argparse.Namespace) -> None:
    processed_dir = Path(SETTINGS.processed_dir)
    model_path = processed_dir / "bias_model.joblib"
    if not model_path.exists():
        logger.warning("Model not found. Run train step first.")
        return

    model = load_model(model_path)
    output_dir = Path(args.output_dir or (Path(SETTINGS.output_dir) / "paper_trading"))
    ensure_dir(output_dir)

    client = _client()
    try:
        run_paper_trading(
            client=client,
            model=model,
            output_dir=output_dir,
            initial_cash=args.initial_cash,
            stake=args.stake,
            min_ev=args.min_ev,
            max_open_positions=args.max_open_positions,
            max_new_orders_per_loop=args.max_new_orders_per_loop,
            poll_seconds=args.poll_seconds,
            live_status=args.live_status,
            min_volume_24h=args.min_volume_24h,
            min_open_interest=args.min_open_interest,
            allow_illiquid=args.allow_illiquid,
            max_markets=args.max_markets,
            once=args.once,
            liquidity_mode=args.liquidity,
            maker_order_ttl_minutes=args.maker_ttl_minutes,
        )
    except KeyboardInterrupt:
        logger.info("Paper trading interrupted by user. State has been saved.")
    finally:
        client.close()


def cmd_backtest(args: argparse.Namespace) -> None:
    processed_dir = Path(SETTINGS.processed_dir)
    model_path = processed_dir / "bias_model.joblib"
    if not model_path.exists():
        logger.warning("Model not found. Run train step first.")
        return
    model = load_model(model_path)

    try:
        candle_obs = load_candle_observations(processed_dir)
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        return

    output_dir = Path(SETTINGS.output_dir)
    price_index = build_price_index(candle_obs)

    result_model = run_backtest(
        candle_obs=candle_obs,
        model=model,
        initial_cash=args.initial_cash,
        stake=args.stake,
        min_ev=args.min_ev,
        fee_multiplier=SETTINGS.default_fee_multiplier,
        target_days=args.target_days,
        max_trades=args.max_trades,
        strategy="model_ev",
        favorite_threshold=args.favorite_threshold,
        underdog_threshold=args.underdog_threshold,
        price_mode="ask",
        slippage_ticks=SETTINGS.slippage_ticks,
        adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
        bid_through_min_touches=SETTINGS.bid_through_min_touches,
        bid_through_max_hours=SETTINGS.bid_through_max_hours,
        bid_through_latency_candles=SETTINGS.bid_through_latency_candles,
    )
    save_backtest_result(result_model, output_dir)

    result_model_bid = run_backtest(
        candle_obs=candle_obs,
        model=model,
        initial_cash=args.initial_cash,
        stake=args.stake,
        min_ev=args.min_ev,
        fee_multiplier=SETTINGS.default_fee_multiplier,
        target_days=args.target_days,
        max_trades=args.max_trades,
        strategy="model_ev",
        favorite_threshold=args.favorite_threshold,
        underdog_threshold=args.underdog_threshold,
        price_mode="bid",
        slippage_ticks=SETTINGS.slippage_ticks,
        adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
        bid_through_min_touches=SETTINGS.bid_through_min_touches,
        bid_through_max_hours=SETTINGS.bid_through_max_hours,
        bid_through_latency_candles=SETTINGS.bid_through_latency_candles,
    )
    save_backtest_result(result_model_bid, output_dir, label="bid")

    result_model_through = run_backtest(
        candle_obs=candle_obs,
        model=model,
        initial_cash=args.initial_cash,
        stake=args.stake,
        min_ev=args.min_ev,
        fee_multiplier=SETTINGS.default_fee_multiplier,
        target_days=args.target_days,
        max_trades=args.max_trades,
        strategy="model_ev",
        favorite_threshold=args.favorite_threshold,
        underdog_threshold=args.underdog_threshold,
        price_mode="bid_through",
        price_index=price_index,
        slippage_ticks=SETTINGS.slippage_ticks,
        adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
        bid_through_min_touches=SETTINGS.bid_through_min_touches,
        bid_through_max_hours=SETTINGS.bid_through_max_hours,
        bid_through_latency_candles=SETTINGS.bid_through_latency_candles,
    )
    save_backtest_result(result_model_through, output_dir, label="through")

    result_fav = run_backtest(
        candle_obs=candle_obs,
        model=model,
        initial_cash=args.initial_cash,
        stake=args.stake,
        min_ev=args.min_ev,
        fee_multiplier=SETTINGS.default_fee_multiplier,
        target_days=args.target_days,
        max_trades=args.max_trades,
        strategy="favorites_yes",
        favorite_threshold=args.favorite_threshold,
        underdog_threshold=args.underdog_threshold,
        price_mode="ask",
        slippage_ticks=SETTINGS.slippage_ticks,
        adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
        bid_through_min_touches=SETTINGS.bid_through_min_touches,
        bid_through_max_hours=SETTINGS.bid_through_max_hours,
        bid_through_latency_candles=SETTINGS.bid_through_latency_candles,
    )
    save_backtest_result(result_fav, output_dir, label="favorites")

    result_fav_bid = run_backtest(
        candle_obs=candle_obs,
        model=model,
        initial_cash=args.initial_cash,
        stake=args.stake,
        min_ev=args.min_ev,
        fee_multiplier=SETTINGS.default_fee_multiplier,
        target_days=args.target_days,
        max_trades=args.max_trades,
        strategy="favorites_yes",
        favorite_threshold=args.favorite_threshold,
        underdog_threshold=args.underdog_threshold,
        price_mode="bid",
        slippage_ticks=SETTINGS.slippage_ticks,
        adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
        bid_through_min_touches=SETTINGS.bid_through_min_touches,
        bid_through_max_hours=SETTINGS.bid_through_max_hours,
        bid_through_latency_candles=SETTINGS.bid_through_latency_candles,
    )
    save_backtest_result(result_fav_bid, output_dir, label="favorites_bid")

    result_fav_through = run_backtest(
        candle_obs=candle_obs,
        model=model,
        initial_cash=args.initial_cash,
        stake=args.stake,
        min_ev=args.min_ev,
        fee_multiplier=SETTINGS.default_fee_multiplier,
        target_days=args.target_days,
        max_trades=args.max_trades,
        strategy="favorites_yes",
        favorite_threshold=args.favorite_threshold,
        underdog_threshold=args.underdog_threshold,
        price_mode="bid_through",
        price_index=price_index,
        slippage_ticks=SETTINGS.slippage_ticks,
        adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
        bid_through_min_touches=SETTINGS.bid_through_min_touches,
        bid_through_max_hours=SETTINGS.bid_through_max_hours,
        bid_through_latency_candles=SETTINGS.bid_through_latency_candles,
    )
    save_backtest_result(result_fav_through, output_dir, label="favorites_through")

    result_underdog = run_backtest(
        candle_obs=candle_obs,
        model=model,
        initial_cash=args.initial_cash,
        stake=args.stake,
        min_ev=args.min_ev,
        fee_multiplier=SETTINGS.default_fee_multiplier,
        target_days=args.target_days,
        max_trades=args.max_trades,
        strategy="underdogs_no",
        favorite_threshold=args.favorite_threshold,
        underdog_threshold=args.underdog_threshold,
        price_mode="ask",
        slippage_ticks=SETTINGS.slippage_ticks,
        adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
        bid_through_min_touches=SETTINGS.bid_through_min_touches,
        bid_through_max_hours=SETTINGS.bid_through_max_hours,
        bid_through_latency_candles=SETTINGS.bid_through_latency_candles,
    )
    save_backtest_result(result_underdog, output_dir, label="underdogs")

    result_underdog_bid = run_backtest(
        candle_obs=candle_obs,
        model=model,
        initial_cash=args.initial_cash,
        stake=args.stake,
        min_ev=args.min_ev,
        fee_multiplier=SETTINGS.default_fee_multiplier,
        target_days=args.target_days,
        max_trades=args.max_trades,
        strategy="underdogs_no",
        favorite_threshold=args.favorite_threshold,
        underdog_threshold=args.underdog_threshold,
        price_mode="bid",
        slippage_ticks=SETTINGS.slippage_ticks,
        adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
        bid_through_min_touches=SETTINGS.bid_through_min_touches,
        bid_through_max_hours=SETTINGS.bid_through_max_hours,
        bid_through_latency_candles=SETTINGS.bid_through_latency_candles,
    )
    save_backtest_result(result_underdog_bid, output_dir, label="underdogs_bid")

    result_underdog_through = run_backtest(
        candle_obs=candle_obs,
        model=model,
        initial_cash=args.initial_cash,
        stake=args.stake,
        min_ev=args.min_ev,
        fee_multiplier=SETTINGS.default_fee_multiplier,
        target_days=args.target_days,
        max_trades=args.max_trades,
        strategy="underdogs_no",
        favorite_threshold=args.favorite_threshold,
        underdog_threshold=args.underdog_threshold,
        price_mode="bid_through",
        price_index=price_index,
        slippage_ticks=SETTINGS.slippage_ticks,
        adverse_selection_penalty=SETTINGS.adverse_selection_penalty,
        bid_through_min_touches=SETTINGS.bid_through_min_touches,
        bid_through_max_hours=SETTINGS.bid_through_max_hours,
        bid_through_latency_candles=SETTINGS.bid_through_latency_candles,
    )
    save_backtest_result(result_underdog_through, output_dir, label="underdogs_through")
    try:
        pdf_path = generate_backtest_report(output_dir)
        logger.info("Backtest report saved to %s", pdf_path)
    except Exception as exc:
        logger.warning("Failed to generate backtest report: %s", exc)
    logger.info("Backtest complete: %s", result_model.summary)


def cmd_backtest_report(_: argparse.Namespace) -> None:
    output_dir = Path(SETTINGS.output_dir)
    try:
        pdf_path = generate_backtest_report(output_dir)
        logger.info("Backtest report saved to %s", pdf_path)
    except Exception as exc:
        logger.warning("Failed to generate backtest report: %s", exc)


def cmd_run_all(args: argparse.Namespace) -> None:
    _require_dates(args)
    cmd_ingest(args)
    observations = cmd_build(args)
    if observations is not None and not observations.empty:
        cmd_train(args)
    cmd_scan(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalshi longshot bias pipeline")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    def add_dates(p: argparse.ArgumentParser, required: bool = True) -> None:
        p.add_argument("--start", required=required, type=lambda s: datetime.fromisoformat(s), help="Start date YYYY-MM-DD")
        p.add_argument("--end", required=required, type=lambda s: datetime.fromisoformat(s), help="End date YYYY-MM-DD")
        p.add_argument("--max-markets", type=int, default=None, help="Limit markets for testing")

    p_ingest = sub.add_parser("ingest", help="Download historical data")
    add_dates(p_ingest, required=False)
    order_group = p_ingest.add_mutually_exclusive_group()
    order_group.add_argument("--newest-first", action="store_true", help="Process newest markets first (default)")
    order_group.add_argument("--oldest-first", action="store_true", help="Process oldest markets first")
    p_ingest.set_defaults(newest_first=True)
    p_ingest.add_argument("--use-cache", action="store_true", help="Use cached raw data only")
    p_ingest.add_argument("--all-history", action="store_true", help="Download all available historical markets")
    p_ingest.add_argument("--resume", action="store_true", help="Resume from last cursor and append to cache")
    p_ingest.add_argument("--backfill-candles", action="store_true", help="Fetch candlesticks for cached markets")
    p_ingest.add_argument("--force-candles", action="store_true", help="Overwrite existing candlesticks data")

    p_build = sub.add_parser("build", help="Build features and bias stats")

    p_train = sub.add_parser("train", help="Train bias-correction model")

    p_scan = sub.add_parser("scan", help="Scan live markets for EV")
    p_scan.add_argument("--max-markets", type=int, default=None, help="Limit markets for testing")
    p_scan.add_argument("--use-cache", action="store_true", help="Use cached raw data only")
    p_scan.add_argument("--live-status", type=str, default="open", help="Live market status filter (open or active)")
    p_scan.add_argument("--min-volume-24h", type=float, default=SETTINGS.min_volume_24h, help="Minimum 24h volume filter")
    p_scan.add_argument("--min-open-interest", type=float, default=SETTINGS.min_open_interest, help="Minimum open interest filter")
    p_scan.add_argument("--allow-illiquid", action="store_true", help="Do not filter illiquid markets by depth")

    p_paper = sub.add_parser("paper-trade", help="Run continuous paper trading simulation on live markets")
    p_paper.add_argument("--output-dir", type=str, default=str(Path(SETTINGS.output_dir) / "paper_trading"), help="Directory for paper-trading artifacts")
    p_paper.add_argument("--initial-cash", type=float, default=1000.0, help="Starting paper cash")
    p_paper.add_argument("--stake", type=float, default=10.0, help="Max dollars per new trade")
    p_paper.add_argument("--min-ev", type=float, default=SETTINGS.min_ev, help="Minimum EV to submit a paper order")
    p_paper.add_argument("--poll-seconds", type=float, default=60.0, help="Seconds between loops")
    p_paper.add_argument("--live-status", type=str, default="open", help="Market status filter (open/active)")
    p_paper.add_argument("--max-markets", type=int, default=2000, help="Max live markets to evaluate per loop")
    p_paper.add_argument("--max-open-positions", type=int, default=200, help="Cap open paper positions")
    p_paper.add_argument("--max-new-orders-per-loop", type=int, default=25, help="Cap newly submitted paper orders per loop")
    p_paper.add_argument("--maker-ttl-minutes", type=int, default=120, help="Maker order expiration in minutes")
    p_paper.add_argument("--liquidity", choices=["taker", "maker", "both"], default="both", help="Which candidate liquidity type to trade")
    p_paper.add_argument("--min-volume-24h", type=float, default=SETTINGS.min_volume_24h, help="Minimum 24h volume filter")
    p_paper.add_argument("--min-open-interest", type=float, default=SETTINGS.min_open_interest, help="Minimum open interest filter")
    p_paper.add_argument("--allow-illiquid", action="store_true", help="Allow zero-depth candidate pricing")
    p_paper.add_argument("--once", action="store_true", help="Run one loop then exit")

    p_run = sub.add_parser("run-all", help="Run full pipeline")
    add_dates(p_run, required=False)
    order_group_run = p_run.add_mutually_exclusive_group()
    order_group_run.add_argument("--newest-first", action="store_true", help="Process newest markets first (default)")
    order_group_run.add_argument("--oldest-first", action="store_true", help="Process oldest markets first")
    p_run.set_defaults(newest_first=True)
    p_run.add_argument("--all-history", action="store_true", help="Use all available historical markets")
    p_run.add_argument("--resume", action="store_true", help="Resume from last cursor and append to cache")
    p_run.add_argument("--use-cache", action="store_true", help="Use cached raw data only")

    p_backtest = sub.add_parser("backtest", help="Backtest the bias model on historical candles")
    p_backtest.add_argument("--initial-cash", type=float, default=1000.0, help="Starting cash balance")
    p_backtest.add_argument("--stake", type=float, default=10.0, help="Max dollars per trade")
    p_backtest.add_argument("--min-ev", type=float, default=0.0, help="Minimum EV per contract")
    p_backtest.add_argument("--target-days", type=float, default=7.0, help="Target days before close for entry")
    p_backtest.add_argument("--max-trades", type=int, default=None, help="Cap number of trades")
    p_backtest.add_argument("--favorite-threshold", type=float, default=0.9, help="Min implied prob for heavy favorites")
    p_backtest.add_argument("--underdog-threshold", type=float, default=0.1, help="Max implied prob for heavy underdogs")

    p_backtest_report = sub.add_parser("backtest-report", help="Generate backtest PDF from existing outputs")

    p_key = sub.add_parser("set-key", help="Save Kalshi private key from stdin")
    p_key.add_argument("--path", type=str, default="secrets/kalshi_private_key.pem", help="Path to write key file")

    p_pub = sub.add_parser("set-public-key", help="Save Kalshi access key from stdin")
    p_pub.add_argument("--path", type=str, default="secrets/kalshi_access_key.txt", help="Path to write access key")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "paper-trade":
        cmd_paper_trade(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "backtest-report":
        cmd_backtest_report(args)
    elif args.command == "set-key":
        key_data = sys.stdin.read()
        if not key_data.strip():
            raise SystemExit("No key data provided on stdin.")
        path = Path(args.path)
        ensure_dir(path.parent)
        path.write_text(key_data.strip() + "\n", encoding="utf-8")
        logger.info("Saved private key to %s", path)
    elif args.command == "set-public-key":
        key_data = sys.stdin.read()
        if not key_data.strip():
            raise SystemExit("No key data provided on stdin.")
        path = Path(args.path)
        ensure_dir(path.parent)
        path.write_text(key_data.strip() + "\n", encoding="utf-8")
        logger.info("Saved access key to %s", path)


if __name__ == "__main__":
    main()
