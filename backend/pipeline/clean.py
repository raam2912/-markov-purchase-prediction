"""
pipeline/clean.py — Stage 3: Production-grade multi-pass data cleaning.

Eight ordered cleaning passes, each fully logged with before/after counts
and a per-pass reason summary. The DataFrame is immutable between passes
(each pass returns a new filtered/transformed copy so intermediates can
optionally be saved).

Cleaning passes (in order):
  P1  — Drop rows with critical null fields (CustomerID, StockCode, InvoiceDate, Quantity, Price)
  P2  — Remove cancellation invoices (Invoice starts with 'C')
  P3  — Remove invalid quantities (Quantity <= 0)
  P4  — Remove invalid prices (Price <= 0)
  P5  — Normalize and filter StockCodes (junk entries, too-short codes)
  P6  — Normalize Description text (uppercase, strip, remove numeric-only)
  P7  — Detect and handle outliers (Quantity, Price > 99.5th percentile)
  P8  — Remove exact and near-duplicate transactions

Final output: a clean DataFrame + a CleaningReport with per-pass metrics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from config import CleaningConfig, PipelineConfig, PROCESSED_DIR


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class PassResult:
    """Metrics captured after each cleaning pass."""
    pass_id: str
    description: str
    rows_before: int
    rows_removed: int
    rows_after: int
    removal_pct: float
    details: dict = field(default_factory=dict)

    def log(self) -> None:
        logger.info(
            f"  [{self.pass_id}] {self.description}: "
            f"removed {self.rows_removed:,} ({self.removal_pct:.2f}%) → "
            f"{self.rows_after:,} rows remain"
        )
        for k, v in self.details.items():
            logger.debug(f"         {k}: {v}")


@dataclass
class CleaningReport:
    passes: list[PassResult] = field(default_factory=list)
    rows_raw: int = 0
    rows_final: int = 0
    total_removed: int = 0
    total_removed_pct: float = 0.0
    outlier_cap_summary: dict = field(default_factory=dict)

    def add(self, result: PassResult) -> None:
        self.passes.append(result)
        result.log()

    def finalize(self, rows_raw: int, rows_final: int) -> None:
        self.rows_raw = rows_raw
        self.rows_final = rows_final
        self.total_removed = rows_raw - rows_final
        self.total_removed_pct = round(self.total_removed / rows_raw * 100, 2) if rows_raw else 0.0
        logger.info("-" * 50)
        logger.info(f"CLEANING COMPLETE:")
        logger.info(f"  Raw rows    : {rows_raw:,}")
        logger.info(f"  Final rows  : {rows_final:,}")
        logger.info(f"  Removed     : {self.total_removed:,} ({self.total_removed_pct:.2f}%)")
        logger.info("-" * 50)


# ── Helper: pass wrapper ──────────────────────────────────────────────────────

def _pass(
    df: pd.DataFrame,
    pass_id: str,
    description: str,
    filter_fn: Callable[[pd.DataFrame], pd.Series],
    details: dict | None = None,
) -> tuple[pd.DataFrame, PassResult]:
    """Apply a boolean filter function, log results, and return cleaned df + metrics."""
    before = len(df)
    keep_mask = filter_fn(df)
    df_out = df[keep_mask].copy()
    removed = before - len(df_out)
    pct = round(removed / before * 100, 2) if before else 0.0
    result = PassResult(
        pass_id=pass_id,
        description=description,
        rows_before=before,
        rows_removed=removed,
        rows_after=len(df_out),
        removal_pct=pct,
        details=details or {},
    )
    return df_out, result


# ── Individual cleaning passes ────────────────────────────────────────────────

def _p1_drop_critical_nulls(
    df: pd.DataFrame, report: CleaningReport
) -> pd.DataFrame:
    """P1 — Drop rows missing any field required to build sequences."""
    critical_cols = ["CustomerID", "StockCode", "InvoiceDate", "Quantity", "Price"]
    null_counts = {c: int(df[c].isna().sum()) for c in critical_cols if c in df.columns}

    df_out, result = _pass(
        df, "P1", "Drop null critical fields",
        lambda d: d[critical_cols].notna().all(axis=1),
        details={"null_counts_per_col": null_counts},
    )
    report.add(result)
    return df_out


def _p2_remove_cancellations(
    df: pd.DataFrame, report: CleaningReport
) -> pd.DataFrame:
    """P2 — Remove invoices that are cancellations (Invoice starts with 'C')."""
    sample_cancelled = df[df["Invoice"].str.startswith("C", na=False)]["Invoice"].head(5).tolist()

    df_out, result = _pass(
        df, "P2", "Remove cancellation invoices",
        lambda d: ~d["Invoice"].str.startswith("C", na=False),
        details={"sample_cancelled_invoices": sample_cancelled},
    )
    report.add(result)
    return df_out


def _p3_remove_invalid_quantities(
    df: pd.DataFrame, cfg: CleaningConfig, report: CleaningReport
) -> pd.DataFrame:
    """P3 — Remove rows where Quantity <= 0 (returns, errors)."""
    neg_sample = df[df["Quantity"] <= 0]["Quantity"].describe().to_dict()

    df_out, result = _pass(
        df, "P3", f"Remove Quantity <= 0",
        lambda d: d["Quantity"] >= cfg.min_quantity,
        details={"invalid_quantity_distribution": {k: round(float(v or 0), 2) for k, v in neg_sample.items()}},
    )
    report.add(result)
    return df_out


def _p4_remove_invalid_prices(
    df: pd.DataFrame, cfg: CleaningConfig, report: CleaningReport
) -> pd.DataFrame:
    """P4 — Remove rows where Price <= 0 (free items or data errors)."""
    zero_price_skus = (
        df[df["Price"] <= cfg.min_price]["StockCode"]
        .value_counts()
        .head(10)
        .to_dict()
    )

    df_out, result = _pass(
        df, "P4", "Remove Price <= 0",
        lambda d: d["Price"] > cfg.min_price,
        details={"top_zero_price_skus": zero_price_skus},
    )
    report.add(result)
    return df_out


def _p5_normalize_stockcodes(
    df: pd.DataFrame, cfg: CleaningConfig, report: CleaningReport
) -> pd.DataFrame:
    """P5 — Normalize StockCode and remove junk system entries.

    Transformations (order matters):
      1. Uppercase + strip whitespace (already done in ingest, belt-and-braces)
      2. Filter out StockCodes matching junk patterns (POST, D, M, etc.)
      3. Filter out StockCodes shorter than min_stockcode_length after stripping
    """
    # 5a: normalize
    df = df.copy()
    df["StockCode"] = df["StockCode"].str.upper().str.strip()

    # 5b: build combined junk regex
    junk_re = re.compile(
        "|".join(f"(?:{p})" for p in cfg.junk_stockcode_patterns),
        re.IGNORECASE,
    )
    junk_mask = df["StockCode"].str.match(junk_re, na=False)
    junk_sample = df[junk_mask]["StockCode"].value_counts().head(15).to_dict()

    # 5c: too-short codes
    short_mask = df["StockCode"].str.len() < cfg.min_stockcode_length

    remove_mask = junk_mask | short_mask
    before = len(df)
    df_out = df[~remove_mask].copy()
    removed = before - len(df_out)
    pct = round(removed / before * 100, 2) if before else 0.0

    result = PassResult(
        pass_id="P5",
        description="Normalize StockCode + remove junk entries",
        rows_before=before,
        rows_removed=removed,
        rows_after=len(df_out),
        removal_pct=pct,
        details={
            "junk_pattern_removals": int(junk_mask.sum()),
            "short_code_removals": int(short_mask.sum()),
            "top_junk_codes": junk_sample,
        },
    )
    report.add(result)
    return df_out


def _p6_normalize_descriptions(
    df: pd.DataFrame, report: CleaningReport
) -> pd.DataFrame:
    """P6 — Normalize Description text and remove numeric-only descriptions.

    Transformations:
      1. Uppercase + strip
      2. Remove rows where Description is numeric-only (data coding artifacts)
      3. Remove rows where Description is a single character (likely meaningless)
    """
    df = df.copy()
    df["Description"] = df["Description"].str.upper().str.strip()

    numeric_mask = df["Description"].str.match(r"^\d+$", na=False)
    single_char_mask = df["Description"].str.len() <= 1

    remove_mask = numeric_mask | single_char_mask
    before = len(df)
    df_out = df[~remove_mask].copy()
    removed = before - len(df_out)
    pct = round(removed / before * 100, 2) if before else 0.0

    result = PassResult(
        pass_id="P6",
        description="Normalize Description + remove numeric/empty",
        rows_before=before,
        rows_removed=removed,
        rows_after=len(df_out),
        removal_pct=pct,
        details={
            "numeric_description_removals": int(numeric_mask.sum()),
            "single_char_description_removals": int(single_char_mask.sum()),
        },
    )
    report.add(result)
    return df_out


def _p7_handle_outliers(
    df: pd.DataFrame, cfg: CleaningConfig, report: CleaningReport
) -> pd.DataFrame:
    """P7 — Detect and cap (or drop) Quantity and Price outliers.

    Uses configured percentile thresholds. Capping is preferred over dropping
    since high-quantity B2B orders are still valid signal — we just bound them.
    """
    df = df.copy()
    cap_summary: dict = {}

    for col in ("Quantity", "Price"):
        threshold = df[col].quantile(cfg.quantity_outlier_percentile / 100
                                     if col == "Quantity"
                                     else cfg.price_outlier_percentile / 100)
        outlier_mask = df[col] > threshold
        outlier_count = int(outlier_mask.sum())

        cap_summary[col] = {
            f"p{cfg.quantity_outlier_percentile}_threshold": round(float(threshold), 4),
            "outlier_count": outlier_count,
            "outlier_pct": round(outlier_count / len(df) * 100, 3),
            "action": "capped" if cfg.cap_outliers else "dropped",
        }

        if cfg.cap_outliers:
            df.loc[outlier_mask, col] = threshold
        else:
            df = df[~outlier_mask].copy()

    before = len(df)  # after potential drops

    result = PassResult(
        pass_id="P7",
        description=f"Outlier {'capping' if cfg.cap_outliers else 'removal'} (Qty + Price)",
        rows_before=before,
        rows_removed=0 if cfg.cap_outliers else sum(v.get("outlier_count", 0) for v in cap_summary.values()),
        rows_after=len(df),
        removal_pct=0.0 if cfg.cap_outliers else round(
            sum(v.get("outlier_count", 0) for v in cap_summary.values()) / before * 100, 2
        ),
        details=cap_summary,
    )
    report.outlier_cap_summary = cap_summary
    report.add(result)
    return df


def _p8_remove_duplicates(
    df: pd.DataFrame, cfg: CleaningConfig, report: CleaningReport
) -> pd.DataFrame:
    """P8 — Remove exact duplicates and near-duplicates.

    Near-duplicate definition: same CustomerID + StockCode + InvoiceDate
    within a configurable time window (default 60s). These arise from data
    entry double-submissions.
    """
    df = df.copy().reset_index(drop=True)
    before = len(df)

    # 8a: exact duplicates (all fields identical)
    exact_dup_mask = df.duplicated(
        subset=["Invoice", "StockCode", "CustomerID", "InvoiceDate", "Quantity", "Price"],
        keep="first",
    )
    exact_dup_count = int(exact_dup_mask.sum())
    df = df[~exact_dup_mask].copy()

    # 8b: near-duplicates (same customer + SKU within time window)
    # Sort by CustomerID, StockCode, InvoiceDate to make time-window calc efficient
    df = df.sort_values(["CustomerID", "StockCode", "InvoiceDate"]).reset_index(drop=True)

    # Compute time delta within each (CustomerID, StockCode) group
    df["_time_delta"] = df.groupby(["CustomerID", "StockCode"])["InvoiceDate"].diff()
    window = pd.Timedelta(seconds=cfg.duplicate_time_window_seconds)
    near_dup_mask = df["_time_delta"] < window
    near_dup_count = int(near_dup_mask.sum())
    df = df[~near_dup_mask].drop(columns=["_time_delta"]).copy()

    removed = before - len(df)
    pct = round(removed / before * 100, 2) if before else 0.0

    result = PassResult(
        pass_id="P8",
        description="Remove exact + near-duplicates",
        rows_before=before,
        rows_removed=removed,
        rows_after=len(df),
        removal_pct=pct,
        details={
            "exact_duplicates_removed": exact_dup_count,
            "near_duplicates_removed": near_dup_count,
            "time_window_seconds": cfg.duplicate_time_window_seconds,
        },
    )
    report.add(result)
    return df


# ── Public entry point ────────────────────────────────────────────────────────

def run_clean(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, CleaningReport]:
    """Execute all 8 cleaning passes in order.

    Args:
        df:  Raw DataFrame from Stage 1 (post-ingest).
        cfg: Pipeline configuration.

    Returns:
        Tuple of (cleaned DataFrame, CleaningReport with per-pass metrics).
    """
    logger.info("=" * 60)
    logger.info("STAGE 3 — DATA CLEANING (8 passes)")
    logger.info("=" * 60)

    cc = cfg.cleaning
    report = CleaningReport()
    rows_raw = len(df)

    df = _p1_drop_critical_nulls(df, report)
    df = _p2_remove_cancellations(df, report)
    df = _p3_remove_invalid_quantities(df, cc, report)
    df = _p4_remove_invalid_prices(df, cc, report)
    df = _p5_normalize_stockcodes(df, cc, report)
    df = _p6_normalize_descriptions(df, report)
    df = _p7_handle_outliers(df, cc, report)
    df = _p8_remove_duplicates(df, cc, report)

    # Reset index for clean sequential IDs
    df = df.reset_index(drop=True)

    # Optionally save intermediate final state
    if cfg.save_intermediates:
        out = PROCESSED_DIR / "transactions_clean.parquet"
        df.to_parquet(out, index=False, engine="pyarrow")
        logger.debug(f"Intermediate clean saved: {out}")

    report.finalize(rows_raw=rows_raw, rows_final=len(df))

    # ── Post-cleaning assertions (fail fast) ─────────────────────────────────
    _assert_clean(df)

    return df, report


def _assert_clean(df: pd.DataFrame) -> None:
    """Hard assertions on the cleaned DataFrame. Raises if any fail."""
    assertions = [
        (df["CustomerID"].isna().sum() == 0,        "CustomerID has nulls after cleaning"),
        (df["StockCode"].isna().sum() == 0,          "StockCode has nulls after cleaning"),
        (df["InvoiceDate"].isna().sum() == 0,        "InvoiceDate has nulls after cleaning"),
        ((df["Quantity"] > 0).all(),                 "Non-positive Quantity found after cleaning"),
        ((df["Price"] > 0).all(),                    "Non-positive Price found after cleaning"),
        (~df["Invoice"].str.startswith("C").any(),   "Cancellation invoices remain after cleaning"),
    ]
    failures = [msg for ok, msg in assertions if not ok]
    if failures:
        for msg in failures:
            logger.critical(f"POST-CLEAN ASSERTION FAILED: {msg}")
        raise AssertionError(
            f"Cleaned DataFrame failed {len(failures)} assertion(s). "
            "Check logs above for details."
        )
    logger.success("All post-cleaning assertions passed ✓")
