"""Bias metrics, calibration, and reporting utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .utils import ceil_to_cent

logger = logging.getLogger(__name__)


def fee_dollars(price: float, fee_multiplier: float) -> float:
    return ceil_to_cent(fee_multiplier * price * (1 - price))


def mincer_zarnowitz(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["outcome", "implied_prob"])
    if len(df) < 30:
        return {"alpha": np.nan, "psi": np.nan, "alpha_se": np.nan, "psi_se": np.nan, "alpha_p": np.nan, "psi_p": np.nan}
    y = df["outcome"] - df["implied_prob"]
    X = sm.add_constant(df["implied_prob"])
    model = sm.OLS(y, X).fit(cov_type="HC3")
    return {
        "alpha": model.params.get("const", np.nan),
        "psi": model.params.get("implied_prob", np.nan),
        "alpha_se": model.bse.get("const", np.nan),
        "psi_se": model.bse.get("implied_prob", np.nan),
        "alpha_p": model.pvalues.get("const", np.nan),
        "psi_p": model.pvalues.get("implied_prob", np.nan),
    }


def compute_group_stats(
    df: pd.DataFrame,
    group_cols: List[str],
    fee_multiplier: float,
    maker_fill_prob: float,
    tick_size: float,
) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["implied_prob", "outcome"])
    df["fee"] = df["implied_prob"].apply(lambda p: fee_dollars(p, fee_multiplier))
    df["roi_realized"] = (df["outcome"] - df["implied_prob"] - df["fee"]) / (df["implied_prob"] + df["fee"])

    grouped = df.groupby(group_cols, dropna=False)
    stats = grouped.agg(
        markets=("ticker", "nunique"),
        observations=("implied_prob", "count"),
        mean_prob=("implied_prob", "mean"),
        win_rate=("outcome", "mean"),
        volume_median=("volume", "median"),
        volume_mean=("volume", "mean"),
        oi_median=("open_interest", "median"),
        oi_mean=("open_interest", "mean"),
        roi_realized=("roi_realized", "mean"),
        brier=("outcome", lambda s: np.mean((s - df.loc[s.index, "implied_prob"]) ** 2)),
        mae=("outcome", lambda s: np.mean(np.abs(s - df.loc[s.index, "implied_prob"]))),
    ).reset_index()

    stats["bias"] = stats["win_rate"] - stats["mean_prob"]
    stats["ratio"] = stats["win_rate"] / stats["mean_prob"]

    maker_price = (stats["mean_prob"] - tick_size).clip(lower=0.01)
    maker_fee = maker_price.apply(lambda p: fee_dollars(p, fee_multiplier))

    stats["expected_roi_taker"] = (stats["win_rate"] - stats["mean_prob"] - stats["mean_prob"].apply(lambda p: fee_dollars(p, fee_multiplier))) / (
        stats["mean_prob"] + stats["mean_prob"].apply(lambda p: fee_dollars(p, fee_multiplier))
    )
    stats["expected_roi_maker"] = maker_fill_prob * (
        (stats["win_rate"] - maker_price - maker_fee) / (maker_price + maker_fee)
    )

    mz_records = []
    for _, group in grouped:
        mz_records.append(mincer_zarnowitz(group))
    mz_df = pd.DataFrame(mz_records)
    stats = pd.concat([stats, mz_df], axis=1)
    stats["share_obs"] = stats["observations"] / stats["observations"].sum()
    return stats


def plot_calibration(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    grouped = df.groupby("price_bin", dropna=False)
    calib = grouped.agg(mean_prob=("implied_prob", "mean"), win_rate=("outcome", "mean")).reset_index()
    plt.figure(figsize=(6, 4))
    plt.plot(calib["mean_prob"], calib["win_rate"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Implied probability")
    plt.ylabel("Empirical win rate")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roi_by_bin(stats: pd.DataFrame, output_path: Path) -> None:
    if stats.empty or "price_bin" not in stats.columns:
        return
    plt.figure(figsize=(7, 4))
    subset = stats.sort_values("price_bin")
    plt.bar(subset["price_bin"], subset["roi_realized"], alpha=0.7, label="Realized")
    plt.plot(subset["price_bin"], subset["expected_roi_taker"], marker="o", label="Expected (taker)")
    plt.xticks(rotation=45)
    plt.ylabel("ROI")
    plt.title("ROI by Price Bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_bias_stats(
    observations: pd.DataFrame,
    fee_multiplier: float,
    maker_fill_prob: float,
    tick_size: float,
) -> pd.DataFrame:
    group_cols = [
        "price_bin",
        "structure",
        "category_mapped",
        "volume_bin",
        "oi_bin",
        "spread_bin",
        "horizon_days",
    ]
    return compute_group_stats(observations, group_cols, fee_multiplier, maker_fill_prob, tick_size)


def build_bias_stats_candles(
    observations: pd.DataFrame,
    fee_multiplier: float,
    maker_fill_prob: float,
    tick_size: float,
) -> pd.DataFrame:
    group_cols = [
        "price_bin",
        "structure",
        "category_mapped",
        "volume_bin",
        "oi_bin",
        "spread_bin",
        "time_to_close_bucket",
    ]
    return compute_group_stats(observations, group_cols, fee_multiplier, maker_fill_prob, tick_size)


def save_report(
    output_dir: str,
    bias_stats: pd.DataFrame,
    top_ev: pd.DataFrame | None = None,
    calibration_path: str | None = None,
    roi_path: str | None = None,
) -> None:
    output_dir = Path(output_dir)
    report_path = output_dir / "report.md"

    lines: List[str] = []
    lines.append("# Kalshi Longshot Bias Report")
    lines.append("")

    if bias_stats.empty:
        lines.append("No bias statistics available.")
    else:
        lines.append("## Headline Prevalence")
        total_obs = bias_stats["observations"].sum()
        total_markets = bias_stats["markets"].sum()
        lines.append(f"- Observations: {int(total_obs)}")
        lines.append(f"- Markets: {int(total_markets)}")

        lines.append("")
        lines.append("## Biggest Bias Categories")
        top_bias = bias_stats.sort_values("bias").head(10)
        for _, row in top_bias.iterrows():
            lines.append(
                f"- {row.get('category_mapped','')} / {row.get('structure','')} / {row.get('price_bin','')}: bias {row.get('bias', float('nan')):.3f}"
            )

        if calibration_path:
            lines.append("")
            lines.append(f"![Calibration]({Path(calibration_path).as_posix()})")
        if roi_path:
            lines.append("")
            lines.append(f"![ROI]({Path(roi_path).as_posix()})")

    if top_ev is not None and not top_ev.empty:
        lines.append("")
        lines.append("## What To Do (Positive-EV Types)")
        lines.append("Look for opportunities where corrected probability exceeds price after fees.")
        lines.append("Examples from latest scan:")
        for _, row in top_ev.head(5).iterrows():
            lines.append(f"- {row.get('ticker')} {row.get('side')} EV {row.get('ev', 0):.3f} ROI {row.get('roi', 0):.2%}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
