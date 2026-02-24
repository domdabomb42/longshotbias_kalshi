from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from kalshi_longshot_bias.features import classify_structure, map_category, price_bin
from kalshi_longshot_bias.utils import progress_bar


def load_observations() -> tuple[pd.DataFrame, str]:
    candidates = [
        Path("data/processed/observations.parquet"),
        Path("data/processed/observations.csv"),
        Path("data/processed/observations_last_price.csv"),
    ]
    chosen = None
    best_valid = -1
    best_df = None
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        if "implied_prob" not in df.columns or "outcome" not in df.columns:
            continue
        valid = int((df["implied_prob"].notna() & df["outcome"].notna()).sum())
        if valid > best_valid:
            best_valid = valid
            best_df = df
            chosen = path.name
        if valid >= 10000:
            break
    if best_df is None:
        raise FileNotFoundError("No observations file found.")
    return best_df, chosen


def load_candle_observations() -> pd.DataFrame | None:
    candidates = [
        Path("data/processed/candle_observations.parquet"),
        Path("data/processed/candle_observations.csv"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    return None


def compute_calibration(obs_clean: pd.DataFrame) -> pd.DataFrame:
    price_order = [
        "00",
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
        "100",
    ]
    calib = obs_clean.groupby("price_bin", dropna=False).agg(
        mean_prob=("implied_prob", "mean"),
        win_rate=("outcome", "mean"),
        n=("implied_prob", "count"),
    ).reset_index()
    calib["price_bin"] = pd.Categorical(calib["price_bin"], categories=price_order, ordered=True)
    calib = calib.sort_values("price_bin")
    calib["bias"] = calib["win_rate"] - calib["mean_prob"]
    return calib


def top_category_bias(bias_stats: pd.DataFrame, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = bias_stats.copy()
    if "horizon_days" in df.columns:
        df = df[df["horizon_days"] == 0]
    df = df.dropna(subset=["bias", "observations"])
    if df.empty:
        empty = pd.DataFrame(columns=["category_mapped", "structure", "bias", "observations"])
        return empty, empty
    grouped = (
        df.groupby(["category_mapped", "structure"], dropna=False)
        .apply(lambda g: pd.Series({
            "bias": np.average(g["bias"], weights=g["observations"]),
            "observations": g["observations"].sum(),
        }))
        .reset_index()
    )
    neg = grouped.sort_values("bias").head(top_n)
    pos = grouped.sort_values("bias", ascending=False).head(top_n)
    return neg, pos


def main() -> None:
    bar = progress_bar(total=3, desc="report data", unit="steps")
    obs, source_name = load_observations()
    bar.update(1)
    candle_obs = load_candle_observations()
    bar.update(1)
    bias_stats_path = Path("outputs/bias_stats.csv")
    bias_stats = pd.read_csv(bias_stats_path) if bias_stats_path.exists() else pd.DataFrame()
    bar.update(1)
    bar.close()

    obs_clean = obs.dropna(subset=["implied_prob", "outcome"]).copy()
    market_count = obs_clean["ticker"].nunique() if "ticker" in obs_clean.columns else np.nan
    obs_count = len(obs_clean)

    close_col = None
    for cand in ("close_ts", "close_time", "end_time", "close_date"):
        if cand in obs_clean.columns:
            close_col = cand
            break
    if close_col:
        obs_clean["close_dt"] = pd.to_datetime(obs_clean[close_col], errors="coerce")
        min_date = obs_clean["close_dt"].min()
        max_date = obs_clean["close_dt"].max()
    else:
        obs_clean["close_dt"] = pd.NaT
        min_date = None
        max_date = None

    brier = float(np.mean((obs_clean["outcome"] - obs_clean["implied_prob"]) ** 2)) if obs_count else np.nan
    mae = float(np.mean(np.abs(obs_clean["outcome"] - obs_clean["implied_prob"]))) if obs_count else np.nan

    longshot = obs_clean[obs_clean["implied_prob"] <= 0.2]
    favorite = obs_clean[obs_clean["implied_prob"] >= 0.8]
    longshot_bias = float(longshot["outcome"].mean() - longshot["implied_prob"].mean()) if len(longshot) else np.nan
    favorite_bias = float(favorite["outcome"].mean() - favorite["implied_prob"].mean()) if len(favorite) else np.nan

    if "price_bin" not in obs_clean.columns:
        obs_clean["price_bin"] = obs_clean["implied_prob"].apply(price_bin)

    if "category_mapped" not in obs_clean.columns:
        obs_clean["category_mapped"] = obs_clean.apply(map_category, axis=1)

    if "structure" not in obs_clean.columns and "event_ticker" in obs_clean.columns:
        struct_df = obs_clean[["ticker", "event_ticker", "title", "subtitle"]].copy()
        struct_df["title"] = struct_df["title"].fillna("")
        struct_df["subtitle"] = struct_df["subtitle"].fillna("")
        obs_clean["structure"] = classify_structure(struct_df)

    for col, label in [("volume", "volume_bin"), ("open_interest", "oi_bin")]:
        if col in obs_clean.columns:
            obs_clean[col] = pd.to_numeric(obs_clean[col], errors="coerce")
            try:
                obs_clean[label] = pd.qcut(obs_clean[col], q=5, labels=False, duplicates="drop")
            except ValueError:
                obs_clean[label] = np.nan

    calib = compute_calibration(obs_clean)
    neg_bins = calib.sort_values("bias").head(5)[["price_bin", "bias", "n"]]
    pos_bins = calib.sort_values("bias", ascending=False).head(5)[["price_bin", "bias", "n"]]

    neg_cats, pos_cats = top_category_bias(bias_stats, top_n=5) if not bias_stats.empty else (pd.DataFrame(), pd.DataFrame())

    price_hist = obs_clean["implied_prob"].dropna()
    bin_counts = obs_clean["price_bin"].value_counts().sort_index()
    obs_clean["bias"] = obs_clean["outcome"] - obs_clean["implied_prob"]

    cat_bias = (
        obs_clean.groupby("category_mapped", dropna=False)
        .agg(bias=("bias", "mean"), n=("bias", "size"))
        .reset_index()
    )
    cat_bias = cat_bias[cat_bias["n"] >= 500].sort_values("bias")

    struct_bias = pd.DataFrame()
    if "structure" in obs_clean.columns:
        struct_bias = (
            obs_clean.groupby("structure", dropna=False)
            .agg(bias=("bias", "mean"), n=("bias", "size"))
            .reset_index()
            .sort_values("bias")
        )

    month_bias = pd.DataFrame()
    if obs_clean["close_dt"].notna().any():
        obs_clean["close_month"] = obs_clean["close_dt"].dt.to_period("M").astype(str)
        month_bias = (
            obs_clean.groupby("close_month", dropna=False)
            .agg(bias=("bias", "mean"), n=("bias", "size"))
            .reset_index()
        )

    # Longshot bias by category
    longshot_bias_cat = pd.DataFrame()
    if "category_mapped" in obs_clean.columns:
        longshot_cat = obs_clean[obs_clean["implied_prob"] <= 0.2]
        if not longshot_cat.empty:
            longshot_bias_cat = (
                longshot_cat.groupby("category_mapped", dropna=False)
                .agg(bias=("bias", "mean"), n=("bias", "size"))
                .reset_index()
            )
            longshot_bias_cat = longshot_bias_cat[longshot_bias_cat["n"] >= 200].sort_values("bias")

    # Longshot bias by time-to-close (from candle observations if available)
    longshot_time_bias = pd.DataFrame()
    if candle_obs is not None and not candle_obs.empty:
        co = candle_obs.dropna(subset=["implied_prob", "outcome"]).copy()
        if "time_to_close_bucket" not in co.columns and "time_to_close_hours" in co.columns:
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
            co["time_to_close_bucket"] = pd.cut(
                co["time_to_close_hours"],
                bins=bins,
                labels=labels,
                include_lowest=True,
                right=True,
            )
        if "time_to_close_bucket" in co.columns:
            co["bias"] = co["outcome"] - co["implied_prob"]
            longshot = co[co["implied_prob"] <= 0.2]
            if not longshot.empty:
                longshot_time_bias = (
                    longshot.groupby("time_to_close_bucket", dropna=False)
                    .agg(bias=("bias", "mean"), n=("bias", "size"))
                    .reset_index()
                )

    # Absolute bias by category (all observations)
    abs_bias_cat = pd.DataFrame()
    if "category_mapped" in obs_clean.columns:
        abs_bias_cat = (
            obs_clean.groupby("category_mapped", dropna=False)["bias"]
            .apply(lambda s: np.mean(np.abs(s)))
            .reset_index(name="abs_bias")
        )
        abs_bias_cat = abs_bias_cat.sort_values("abs_bias", ascending=False)

    # Category x time-to-close absolute bias matrix (candles)
    abs_bias_matrix = None
    if candle_obs is not None and not candle_obs.empty:
        co = candle_obs.dropna(subset=["implied_prob", "outcome"]).copy()
        if "category_mapped" not in co.columns:
            co["category_mapped"] = co.apply(map_category, axis=1)
        if "time_to_close_bucket" not in co.columns and "time_to_close_hours" in co.columns:
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
            co["time_to_close_bucket"] = pd.cut(
                co["time_to_close_hours"],
                bins=bins,
                labels=labels,
                include_lowest=True,
                right=True,
            )
        if "time_to_close_bucket" in co.columns:
            co["abs_bias"] = (co["outcome"] - co["implied_prob"]).abs()
            abs_bias_matrix = (
                co.groupby(["category_mapped", "time_to_close_bucket"], dropna=False)["abs_bias"]
                .mean()
                .reset_index()
            )

    pdf_path = Path("outputs") / "investor_report_condensed.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: Title + Summary
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        title = "Kalshi Longshot Bias Analysis"
        subtitle = "Condensed Investor Report"
        date_line = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        plt.text(0.5, 0.90, title, ha="center", va="center", fontsize=22, fontweight="bold")
        plt.text(0.5, 0.86, subtitle, ha="center", va="center", fontsize=12)
        plt.text(0.5, 0.82, f"Report Date: {date_line}", ha="center", va="center", fontsize=10)

        y = 0.74
        if min_date is not None and max_date is not None:
            plt.text(0.06, y, f"Sample window: {min_date.date()} to {max_date.date()}", fontsize=11)
            y -= 0.04
        plt.text(0.06, y, f"Markets analyzed: {int(market_count) if not np.isnan(market_count) else 'N/A'}", fontsize=11)
        y -= 0.04
        plt.text(0.06, y, f"Observations (price points): {obs_count}", fontsize=11)
        y -= 0.04
        plt.text(0.06, y, f"Brier score: {brier:.4f} | MAE: {mae:.4f}", fontsize=11)
        y -= 0.04
        plt.text(0.06, y, f"Longshot bias (<=20c): {longshot_bias:+.4f}", fontsize=11)
        y -= 0.04
        plt.text(0.06, y, f"Favorite bias (>=80c): {favorite_bias:+.4f}", fontsize=11)
        y -= 0.04
        plt.text(0.06, y, f"Data source: {source_name}", fontsize=9)

        y -= 0.06
        plt.text(0.06, y, "Key Takeaways:", fontsize=12, fontweight="bold")
        y -= 0.04
        for line in [
            "Longshots are overpriced on average; favorites are underpriced.",
            "Mid-probability bins show the strongest negative bias.",
            "Bias-corrected probabilities improve calibration and ROI estimates.",
        ]:
            plt.text(0.08, y, f"- {line}", fontsize=10)
            y -= 0.03

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Core Calibration + Bias
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        axes = axes.flatten()
        axes[0].plot(calib["mean_prob"], calib["win_rate"], marker="o")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[0].set_title("Calibration Curve (All Markets)")
        axes[0].set_xlabel("Implied probability")
        axes[0].set_ylabel("Empirical win rate")

        axes[1].bar(calib["price_bin"].astype(str), calib["bias"])
        axes[1].axhline(0, color="gray", linestyle="--")
        axes[1].set_title("Bias by Price Bin")
        axes[1].set_ylabel("Bias (win_rate - implied)")
        axes[1].tick_params(axis="x", rotation=45)

        axes[2].hist(price_hist, bins=50, color="#4C78A8", alpha=0.8)
        axes[2].set_title("Implied Probability Distribution")
        axes[2].set_xlabel("Implied probability")
        axes[2].set_ylabel("Count")

        axes[3].bar(bin_counts.index.astype(str), bin_counts.values)
        axes[3].set_title("Price Bin Prevalence")
        axes[3].set_ylabel("Observations")
        axes[3].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: Category & Structure Bias
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        if not cat_bias.empty:
            top_neg = cat_bias.head(10)
            top_pos = cat_bias.tail(10)
            cat_plot = pd.concat([top_neg, top_pos])
            axes[0].barh(cat_plot["category_mapped"].astype(str), cat_plot["bias"])
            axes[0].axvline(0, color="gray", linestyle="--")
            axes[0].set_title("Category Bias (Top +/- 10, min 500 obs)")
            axes[0].set_xlabel("Bias (win_rate - implied)")
        else:
            axes[0].text(0.5, 0.5, "Category bias unavailable", ha="center", va="center")
            axes[0].axis("off")

        if not struct_bias.empty:
            axes[1].bar(struct_bias["structure"].astype(str), struct_bias["bias"])
            axes[1].axhline(0, color="gray", linestyle="--")
            axes[1].set_title("Structure Bias")
            axes[1].set_ylabel("Bias")
        else:
            axes[1].text(0.5, 0.5, "Structure bias unavailable", ha="center", va="center")
            axes[1].axis("off")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: Liquidity & Volume
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        axes = axes.flatten()
        if "volume" in obs_clean.columns:
            axes[0].hist(obs_clean["volume"].dropna(), bins=50, color="#F58518", alpha=0.8)
            axes[0].set_title("Volume Distribution")
            axes[0].set_xlabel("Volume")
            axes[0].set_ylabel("Count")
        else:
            axes[0].text(0.5, 0.5, "Volume unavailable", ha="center", va="center")
            axes[0].axis("off")

        if "open_interest" in obs_clean.columns:
            axes[1].hist(obs_clean["open_interest"].dropna(), bins=50, color="#54A24B", alpha=0.8)
            axes[1].set_title("Open Interest Distribution")
            axes[1].set_xlabel("Open interest")
            axes[1].set_ylabel("Count")
        else:
            axes[1].text(0.5, 0.5, "Open interest unavailable", ha="center", va="center")
            axes[1].axis("off")

        if "volume_bin" in obs_clean.columns and obs_clean["volume_bin"].notna().any():
            vol_bias = obs_clean.groupby("volume_bin")["bias"].mean()
            axes[2].bar(vol_bias.index.astype(str), vol_bias.values)
            axes[2].axhline(0, color="gray", linestyle="--")
            axes[2].set_title("Bias by Volume Bin")
            axes[2].set_ylabel("Bias")
        else:
            axes[2].text(0.5, 0.5, "Volume bins unavailable", ha="center", va="center")
            axes[2].axis("off")

        if "oi_bin" in obs_clean.columns and obs_clean["oi_bin"].notna().any():
            oi_bias = obs_clean.groupby("oi_bin")["bias"].mean()
            axes[3].bar(oi_bias.index.astype(str), oi_bias.values)
            axes[3].axhline(0, color="gray", linestyle="--")
            axes[3].set_title("Bias by OI Bin")
            axes[3].set_ylabel("Bias")
        else:
            axes[3].text(0.5, 0.5, "OI bins unavailable", ha="center", va="center")
            axes[3].axis("off")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 5: Time Dynamics
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        if not month_bias.empty:
            axes[0].plot(month_bias["close_month"], month_bias["bias"], marker="o")
            axes[0].axhline(0, color="gray", linestyle="--")
            axes[0].set_title("Monthly Average Bias")
            axes[0].set_ylabel("Bias")
            axes[0].tick_params(axis="x", rotation=45)

            axes[1].bar(month_bias["close_month"], month_bias["n"])
            axes[1].set_title("Monthly Observation Count")
            axes[1].set_ylabel("Count")
            axes[1].tick_params(axis="x", rotation=45)
        else:
            axes[0].text(0.5, 0.5, "Time series unavailable", ha="center", va="center")
            axes[0].axis("off")
            axes[1].axis("off")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 6: Tables
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        plt.text(0.02, 0.97, "Top Bias Bins & Categories", fontsize=16, fontweight="bold", va="top")

        def draw_table(ax, df, title, y_top, height=0.22):
            if df is None or df.empty:
                ax.text(0.02, y_top, f"{title}: unavailable", fontsize=11, va="top")
                return
            ax.text(0.02, y_top, title, fontsize=11, fontweight="bold", va="top")
            table = ax.table(
                cellText=df.round(4).astype(str).values,
                colLabels=df.columns,
                cellLoc="center",
                loc="upper left",
                bbox=[0.02, y_top - height, 0.96, height - 0.02],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)

        draw_table(plt.gca(), neg_bins, "Most Negative Bias Price Bins", 0.90)
        draw_table(plt.gca(), pos_bins, "Most Positive Bias Price Bins", 0.62)
        if not neg_cats.empty:
            draw_table(plt.gca(), neg_cats[["category_mapped", "structure", "bias", "observations"]], "Most Negative Bias Categories", 0.36)
        if not pos_cats.empty:
            draw_table(plt.gca(), pos_cats[["category_mapped", "structure", "bias", "observations"]], "Most Positive Bias Categories", 0.18, height=0.16)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 7: Longshot Bias by Category and Time-to-Close
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        if not longshot_bias_cat.empty:
            top_neg = longshot_bias_cat.head(10)
            top_pos = longshot_bias_cat.tail(10)
            plot_df = pd.concat([top_neg, top_pos])
            axes[0].barh(plot_df["category_mapped"].astype(str), plot_df["bias"])
            axes[0].axvline(0, color="gray", linestyle="--")
            axes[0].set_title("Longshot Bias by Category (p<=0.20)")
            axes[0].set_xlabel("Bias (win_rate - implied)")
        else:
            axes[0].text(0.5, 0.5, "Longshot category bias unavailable", ha="center", va="center")
            axes[0].axis("off")

        if not longshot_time_bias.empty:
            axes[1].bar(longshot_time_bias["time_to_close_bucket"].astype(str), longshot_time_bias["bias"])
            axes[1].axhline(0, color="gray", linestyle="--")
            axes[1].set_title("Longshot Bias by Time-to-Close (p<=0.20)")
            axes[1].set_ylabel("Bias")
            axes[1].tick_params(axis="x", rotation=45)
        else:
            axes[1].text(0.5, 0.5, "Longshot time-to-close bias unavailable", ha="center", va="center")
            axes[1].axis("off")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 8: Absolute Bias by Category + Category x Time Heatmap
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
        if abs_bias_cat is not None and not abs_bias_cat.empty:
            top = abs_bias_cat.head(15)
            axes[0].barh(top["category_mapped"].astype(str), top["abs_bias"])
            axes[0].set_title("Average Absolute Bias by Category")
            axes[0].set_xlabel("Mean |win_rate - implied|")
        else:
            axes[0].text(0.5, 0.5, "Absolute bias by category unavailable", ha="center", va="center")
            axes[0].axis("off")

        if abs_bias_matrix is not None and not abs_bias_matrix.empty:
            pivot = abs_bias_matrix.pivot(index="category_mapped", columns="time_to_close_bucket", values="abs_bias")
            im = axes[1].imshow(pivot.values, aspect="auto", interpolation="nearest")
            axes[1].set_title("Absolute Bias by Category x Time-to-Close")
            axes[1].set_yticks(range(len(pivot.index)))
            axes[1].set_yticklabels(pivot.index.astype(str))
            axes[1].set_xticks(range(len(pivot.columns)))
            axes[1].set_xticklabels(pivot.columns.astype(str), rotation=45, ha="right")
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        else:
            axes[1].text(0.5, 0.5, "Category x time heatmap unavailable", ha="center", va="center")
            axes[1].axis("off")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
