"""
config.py — Centralized configuration for the AntiGrav data pipeline.

All thresholds, paths, and flags live here. No magic numbers in pipeline code.
"""
from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


# ── Root paths ────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent.resolve()
DATA_DIR    = BASE_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"

# Ensure directories exist on import
for _d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Dataset URLs ──────────────────────────────────────────────────────────────

UCI_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
)
UCI_ZIP_PATH  = RAW_DIR / "uci_retail_ii.zip"
UCI_XLSX_PATH = RAW_DIR / "online_retail_II.xlsx"  # file name inside zip


# ── Expected schema ───────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

SHEET_NAMES = [
    "Year 2009-2010",
    "Year 2010-2011",
]

# Canonical column rename map (raw → internal)
# Removes space in "Customer ID" for easier attribute access
COLUMN_RENAME: dict[str, str] = {
    "Customer ID": "CustomerID",
}

EXPECTED_DATE_MIN = "2009-12-01"
EXPECTED_DATE_MAX = "2011-12-31"


# ── Cleaning thresholds ───────────────────────────────────────────────────────

class CleaningConfig(BaseModel):
    """All cleaning thresholds in one validated place."""

    # Pass 3 — Quantity
    min_quantity: int = Field(1, description="Minimum valid quantity (inclusive)")

    # Pass 4 — Price
    min_price: float = Field(0.0, description="Minimum valid price (exclusive → must be > 0)")

    # Pass 5 — StockCode normalization
    # StockCodes shorter than this after stripping are considered junk
    min_stockcode_length: int = Field(2)
    # StockCodes matching these patterns are system/test entries (case-insensitive)
    junk_stockcode_patterns: list[str] = Field(default=[
        r"^POST$",          # Postage
        r"^D$",             # Discount
        r"^M$",             # Manual
        r"^PADS$",          # Pads
        r"^DOT$",           # Dots
        r"^BANK CHARGES$",  # Bank charges
        r"^C2$",            # Carriage
        r"^S$",             # Sample
        r"^gift_0001_\d+$", # Gift vouchers
        r"^AMAZONFEE$",     # Amazon fee
        r"^[a-zA-Z]{1,2}$",# Single/double letter codes (non-product)
    ])

    # Pass 7 — Outlier capping
    quantity_outlier_percentile: float = Field(99.5, ge=90.0, le=100.0)
    price_outlier_percentile: float    = Field(99.5, ge=90.0, le=100.0)
    # If True, cap outliers; if False, drop them
    cap_outliers: bool = Field(True)

    # Pass 8 — Duplicate detection
    # Two rows with same CustomerID+StockCode are considered near-duplicates
    # if their InvoiceDate differs by less than this many seconds
    duplicate_time_window_seconds: int = Field(60)

    @field_validator("min_price")
    @classmethod
    def price_must_allow_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("min_price cannot be negative")
        return v


class SequenceConfig(BaseModel):
    """Configuration for sequence building (Stage 4)."""

    # Minimum number of transactions for a customer to be included
    min_transactions_per_customer: int = Field(3)

    # Maximum gap (days) between orders before we split into a new "session"
    session_gap_days: int = Field(30)

    # Include quantity and price in sequence events? (richer but larger)
    include_financials: bool = Field(True)


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    cleaning:  CleaningConfig  = Field(default_factory=CleaningConfig)
    sequences: SequenceConfig  = Field(default_factory=SequenceConfig)

    # Download settings
    download_timeout_seconds: int = Field(120)
    download_max_retries: int = Field(5)
    download_chunk_size: int = Field(65536)  # 64 KB

    # Logging
    log_level: str = Field("INFO")

    # If True, save intermediate DataFrames after each cleaning pass (debugging)
    save_intermediates: bool = Field(False)


# ── Default singleton ─────────────────────────────────────────────────────────

def get_config() -> PipelineConfig:
    """Return the default pipeline configuration.

    Override via environment variables or pass kwargs:
        cfg = PipelineConfig(cleaning=CleaningConfig(cap_outliers=False))
    """
    return PipelineConfig()


# Allow overrides from environment for CI/CD or Docker deployments
def get_config_from_env() -> PipelineConfig:
    """Build config from environment variables where available."""
    cleaning_overrides: dict = {}
    seq_overrides: dict = {}
    pipeline_overrides: dict = {}

    env_map = {
        "ANTIGRAV_MIN_QUANTITY":           ("cleaning", "min_quantity", int),
        "ANTIGRAV_MIN_PRICE":              ("cleaning", "min_price", float),
        "ANTIGRAV_CAP_OUTLIERS":           ("cleaning", "cap_outliers", lambda x: x.lower() == "true"),
        "ANTIGRAV_DUP_WINDOW_SECS":        ("cleaning", "duplicate_time_window_seconds", int),
        "ANTIGRAV_MIN_TX_PER_CUSTOMER":    ("sequences", "min_transactions_per_customer", int),
        "ANTIGRAV_SESSION_GAP_DAYS":       ("sequences", "session_gap_days", int),
        "ANTIGRAV_DOWNLOAD_RETRIES":       ("pipeline", "download_max_retries", int),
        "ANTIGRAV_SAVE_INTERMEDIATES":     ("pipeline", "save_intermediates", lambda x: x.lower() == "true"),
        "ANTIGRAV_LOG_LEVEL":              ("pipeline", "log_level", str),
    }

    for env_var, (section, key, cast) in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            if section == "cleaning":
                cleaning_overrides[key] = cast(val)  # type: ignore[operator]
            elif section == "sequences":
                seq_overrides[key] = cast(val)  # type: ignore[operator]
            else:
                pipeline_overrides[key] = cast(val)  # type: ignore[operator]

    return PipelineConfig(
        cleaning=CleaningConfig(**cleaning_overrides),
        sequences=SequenceConfig(**seq_overrides),
        **pipeline_overrides,
    )
