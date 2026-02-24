"""Configuration for the Kalshi longshot bias pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List


def _default_private_key_path() -> str | None:
    env_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if env_path:
        return env_path
    default_path = Path("secrets") / "kalshi_private_key.pem"
    return str(default_path) if default_path.exists() else None


def _default_public_key() -> str | None:
    env_key = os.getenv("KALSHI_ACCESS_KEY")
    if env_key:
        return env_key
    key_path = Path("secrets") / "kalshi_access_key.txt"
    if key_path.exists():
        try:
            return key_path.read_text(encoding="utf-8").strip() or None
        except Exception:
            return None
    return None


@dataclass(frozen=True)
class Settings:
    base_url: str = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
    # Kalshi RSA auth
    access_key_id: str | None = _default_public_key()
    private_key_path: str | None = _default_private_key_path()
    private_key_pem: str | None = os.getenv("KALSHI_PRIVATE_KEY_PEM")
    ssl_verify: bool = os.getenv("KALSHI_SSL_VERIFY", "true").lower() != "false"
    request_delay: float = float(os.getenv("KALSHI_REQUEST_DELAY", "0.2"))
    timeout: float = float(os.getenv("KALSHI_TIMEOUT", "20"))
    backfill_total_estimate: int = int(os.getenv("KALSHI_BACKFILL_TOTAL_ESTIMATE", "129000"))

    cache_dir: str = os.getenv("KALSHI_CACHE_DIR", "data/raw")
    processed_dir: str = os.getenv("KALSHI_PROCESSED_DIR", "data/processed")
    output_dir: str = os.getenv("KALSHI_OUTPUT_DIR", "outputs")

    dry_run: bool = os.getenv("DRY_RUN", "True").lower() != "false"

    default_fee_multiplier: float = float(os.getenv("KALSHI_DEFAULT_FEE_MULTIPLIER", "0.07"))
    tick_size: float = float(os.getenv("KALSHI_TICK_SIZE", "0.01"))

    candle_interval_sec: int = int(os.getenv("KALSHI_CANDLE_INTERVAL_SEC", "60"))
    candle_window_days: int = int(os.getenv("KALSHI_CANDLE_WINDOW_DAYS", "7"))
    candle_chunk_hours: int = int(os.getenv("KALSHI_CANDLE_CHUNK_HOURS", "0"))
    candle_resample_minutes: int = int(os.getenv("KALSHI_CANDLE_RESAMPLE_MINUTES", "60"))
    horizons_days: List[float] = field(
        default_factory=lambda: [0, 1 / 24, 3 / 24, 6 / 24, 12 / 24, 1, 2, 3, 5, 7, 14, 30]
    )

    max_spread: float = float(os.getenv("KALSHI_MAX_SPREAD", "0.08"))
    min_depth: float = float(os.getenv("KALSHI_MIN_DEPTH", "10"))
    min_ev: float = float(os.getenv("KALSHI_MIN_EV", "0.0"))
    top_n: int = int(os.getenv("KALSHI_TOP_N", "50"))

    maker_fill_prob: float = float(os.getenv("KALSHI_MAKER_FILL_PROB", "0.2"))
    slippage_ticks: int = int(os.getenv("KALSHI_SLIPPAGE_TICKS", "1"))
    adverse_selection_penalty: float = float(os.getenv("KALSHI_ADVERSE_SELECTION_PENALTY", "0.02"))
    bid_through_min_touches: int = int(os.getenv("KALSHI_BID_THROUGH_MIN_TOUCHES", "2"))
    bid_through_max_hours: float = float(os.getenv("KALSHI_BID_THROUGH_MAX_HOURS", "12"))
    bid_through_latency_candles: int = int(os.getenv("KALSHI_BID_THROUGH_LATENCY_CANDLES", "1"))

    fee_multiplier_overrides: Dict[str, float] = field(default_factory=dict)

    min_volume_24h: float = float(os.getenv("KALSHI_MIN_VOLUME_24H", "0"))
    min_open_interest: float = float(os.getenv("KALSHI_MIN_OPEN_INTEREST", "0"))
    allow_illiquid_live: bool = os.getenv("KALSHI_ALLOW_ILLIQUID", "false").lower() == "true"
    live_page_limit: int = int(os.getenv("KALSHI_LIVE_PAGE_LIMIT", "200"))


SETTINGS = Settings()
