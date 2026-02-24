import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def load_observations() -> pd.DataFrame:
    obs_path = Path("data/processed/observations.parquet")
    if obs_path.exists():
        return pd.read_parquet(obs_path)
    return pd.read_csv("data/processed/observations.csv")


def ensure_plots(obs: pd.DataFrame, bias_stats: pd.DataFrame) -> dict:
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    obs_clean = obs.dropna(subset=["implied_prob", "outcome"])
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

    # Calibration plot
    plt.figure(figsize=(6, 4))
    plt.plot(calib["mean_prob"], calib["win_rate"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Implied probability")
    plt.ylabel("Empirical win rate")
    plt.title("Calibration Curve (All Markets)")
    plt.tight_layout()
    calibration_path = plots_dir / "calibration_curve.png"
    plt.savefig(calibration_path)
    plt.close()

    # Bias by bin
    plt.figure(figsize=(7, 4))
    plt.bar(calib["price_bin"].astype(str), calib["bias"])
    plt.axhline(0, color="gray", linestyle="--")
    plt.xticks(rotation=45)
    plt.ylabel("Bias (win_rate - implied)")
    plt.title("Bias by Price Bin")
    plt.tight_layout()
    bias_path = plots_dir / "bias_by_bin.png"
    plt.savefig(bias_path)
    plt.close()

    # ROI by bin
    roi = bias_stats.groupby("price_bin", dropna=False).agg(
        expected_roi_taker=("expected_roi_taker", "mean"),
        roi_realized=("roi_realized", "mean"),
    ).reset_index()
    roi["price_bin"] = pd.Categorical(roi["price_bin"], categories=price_order, ordered=True)
    roi = roi.sort_values("price_bin")

    plt.figure(figsize=(7, 4))
    plt.bar(roi["price_bin"].astype(str), roi["roi_realized"], alpha=0.7, label="Realized")
    plt.plot(roi["price_bin"].astype(str), roi["expected_roi_taker"], marker="o", label="Expected (taker)")
    plt.xticks(rotation=45)
    plt.ylabel("ROI")
    plt.title("ROI by Price Bin")
    plt.legend()
    plt.tight_layout()
    roi_path = plots_dir / "roi_by_bin.png"
    plt.savefig(roi_path)
    plt.close()

    return {
        "calibration": calibration_path,
        "bias_by_bin": bias_path,
        "roi_by_bin": roi_path,
        "calib_table": calib,
    }


def render_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    plt.text(0.02, 0.97, title, fontsize=18, fontweight="bold", va="top")
    y = 0.9
    for line in lines:
        plt.text(0.04, y, line, fontsize=11, va="top")
        y -= 0.04
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    plt.text(0.02, 0.97, title, fontsize=18, fontweight="bold", va="top")
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    obs = load_observations()
    bias_stats = pd.read_csv("outputs/bias_stats.csv")
    positive_ev_path = Path("outputs/positive_ev_bets.csv")
    if positive_ev_path.exists() and positive_ev_path.stat().st_size > 0:
        try:
            positive_ev = pd.read_csv(positive_ev_path)
        except pd.errors.EmptyDataError:
            positive_ev = pd.DataFrame()
    else:
        positive_ev = pd.DataFrame()

    obs_clean = obs.dropna(subset=["implied_prob", "outcome"])
    market_count = obs_clean["ticker"].nunique() if "ticker" in obs_clean.columns else np.nan
    obs_count = len(obs_clean)

    if "close_ts" in obs_clean.columns:
        min_date = pd.to_datetime(obs_clean["close_ts"]).min()
        max_date = pd.to_datetime(obs_clean["close_ts"]).max()
    else:
        min_date = None
        max_date = None

    brier = float(np.mean((obs_clean["outcome"] - obs_clean["implied_prob"]) ** 2)) if obs_count else np.nan
    mae = float(np.mean(np.abs(obs_clean["outcome"] - obs_clean["implied_prob"]))) if obs_count else np.nan

    longshot = obs_clean[obs_clean["implied_prob"] <= 0.2]
    favorite = obs_clean[obs_clean["implied_prob"] >= 0.8]
    longshot_bias = float(longshot["outcome"].mean() - longshot["implied_prob"].mean()) if len(longshot) else np.nan
    favorite_bias = float(favorite["outcome"].mean() - favorite["implied_prob"].mean()) if len(favorite) else np.nan

    plots = ensure_plots(obs, bias_stats)
    calib = plots["calib_table"]

    neg_bias = calib.sort_values("bias").head(5)[["price_bin", "mean_prob", "win_rate", "bias", "n"]]
    pos_bias = calib.sort_values("bias", ascending=False).head(5)[["price_bin", "mean_prob", "win_rate", "bias", "n"]]

    pdf_path = Path("outputs") / "investor_report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Title
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        title = "Kalshi Longshot Bias Analysis"
        subtitle = "Investor Summary Report"
        date_line = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        plt.text(0.5, 0.78, title, ha="center", va="center", fontsize=24, fontweight="bold")
        plt.text(0.5, 0.72, subtitle, ha="center", va="center", fontsize=14)
        plt.text(0.5, 0.66, f"Report Date: {date_line}", ha="center", va="center", fontsize=11)
        plt.text(0.5, 0.58, "Prepared by: Longshot Bias Research Pipeline", ha="center", va="center", fontsize=10, color="gray")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Executive summary
        summary_lines = []
        if min_date is not None and max_date is not None:
            summary_lines.append(f"- Sample window: {min_date.date()} to {max_date.date()}")
        summary_lines.append(f"- Markets analyzed: {int(market_count) if not np.isnan(market_count) else 'N/A'}")
        summary_lines.append(f"- Observations (price points): {obs_count}")
        summary_lines.append(f"- Brier score: {brier:.4f} | MAE: {mae:.4f}")
        summary_lines.append(f"- Longshot bias (<=20c): {longshot_bias:+.4f}")
        summary_lines.append(f"- Favorite bias (>=80c): {favorite_bias:+.4f}")
        if positive_ev is not None and not positive_ev.empty:
            summary_lines.append(f"- Live positive-EV opportunities: {len(positive_ev)} (see Appendix)")
        else:
            summary_lines.append("- Live positive-EV opportunities: None in latest scan")
        render_text_page(pdf, "Executive Summary", summary_lines)

        render_image_page(pdf, "Calibration", plots["calibration"])
        render_image_page(pdf, "Bias by Price Bin", plots["bias_by_bin"])
        render_image_page(pdf, "ROI by Price Bin", plots["roi_by_bin"])

        # Top bias tables
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        plt.text(0.02, 0.97, "Top Bias Bins", fontsize=18, fontweight="bold", va="top")

        def draw_table(ax, df, title, y_top):
            ax.text(0.02, y_top, title, fontsize=12, fontweight="bold", va="top")
            table = ax.table(
                cellText=df.round(4).astype(str).values,
                colLabels=df.columns,
                cellLoc="center",
                loc="upper left",
                bbox=[0.02, y_top - 0.32, 0.96, 0.28],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)

        draw_table(plt.gca(), neg_bias, "Most Negative Bias (Overpriced)", 0.90)
        draw_table(plt.gca(), pos_bias, "Most Positive Bias (Underpriced)", 0.52)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Methodology
        methodology_lines = [
            "- Data source: Kalshi API v2 (settled historical markets)",
            "- Prices mapped to empirical win probabilities via isotonic regression",
            "- Bias metrics: calibration curves, Mincer-Zarnowitz regression, ROI by price bin",
            "- Fees modeled with Kalshi quadratic fee approximation",
            "- Look-ahead avoided using time-sliced (walk-forward) validation",
            "- Live scan uses best executable prices with fee-adjusted EV",
        ]
        render_text_page(pdf, "Methodology (Summary)", methodology_lines)

        # Appendix: Live positive-EV
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        plt.text(0.02, 0.97, "Appendix: Live Positive-EV Candidates", fontsize=18, fontweight="bold", va="top")
        if positive_ev is not None and not positive_ev.empty:
            keep_cols = [
                "ticker", "side", "price", "implied_prob", "q_hat", "fee", "ev", "roi",
                "spread", "volume_24h", "open_interest", "category", "structure",
            ]
            available = [c for c in keep_cols if c in positive_ev.columns]
            subset = positive_ev[available].head(15)
            table = plt.table(
                cellText=subset.values,
                colLabels=subset.columns,
                cellLoc="center",
                loc="upper left",
                bbox=[0.02, 0.05, 0.96, 0.88],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(6)
        else:
            plt.text(0.04, 0.9, "No positive-EV opportunities in the latest scan.", fontsize=11, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
