"""
pipeline/validate.py — Stage 2: Schema & statistical validation.

Runs BEFORE any cleaning. Produces a ValidationReport that describes
the raw data's health. Does NOT mutate the DataFrame — validation is
purely observational so problems can be logged and surfaced in the
quality report.

Checks performed:
  1. Required columns present
  2. Unexpected columns detected (warning, not failure)
  3. Column dtype consistency
  4. Date range sanity
  5. Null rate per column
  6. Negative quantity / zero price prevalence
  7. Cancellation rate (Invoice starts with 'C')
  8. CustomerID null rate
  9. Duplicate exact-row count
  10. Value distribution summary (for report)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from loguru import logger

from config import (
    EXPECTED_DATE_MAX,
    EXPECTED_DATE_MIN,
    REQUIRED_COLUMNS,
    PipelineConfig,
)


# ── Results ────────────────────────────────────────────────────────────────────

@dataclass
class ColumnCheck:
    name: str
    present: bool
    dtype: str
    null_count: int
    null_pct: float
    sample_values: list[Any]


@dataclass
class ValidationReport:
    total_rows: int
    total_columns: int

    missing_required_columns: list[str]
    unexpected_columns: list[str]
    column_checks: list[ColumnCheck]

    date_min: str | None
    date_max: str | None
    date_range_valid: bool

    negative_quantity_count: int
    negative_quantity_pct: float

    zero_price_count: int
    zero_price_pct: float

    cancellation_count: int
    cancellation_pct: float

    null_customer_count: int
    null_customer_pct: float

    exact_duplicate_count: int
    exact_duplicate_pct: float

    unique_customers: int
    unique_skus: int
    unique_countries: int

    distribution_summary: dict[str, Any]

    passed: bool  # True if no critical issues
    warnings: list[str]
    errors: list[str]

    def log_summary(self) -> None:
        """Print a concise summary to the logger."""
        logger.info("-" * 50)
        logger.info(f"Validation: {self.total_rows:,} rows, {self.total_columns} columns")
        logger.info(f"  Date range: {self.date_min} → {self.date_max} (valid={self.date_range_valid})")
        logger.info(f"  Null CustomerIDs : {self.null_customer_count:,}  ({self.null_customer_pct:.1f}%)")
        logger.info(f"  Cancellations    : {self.cancellation_count:,}  ({self.cancellation_pct:.1f}%)")
        logger.info(f"  Negative qty     : {self.negative_quantity_count:,}  ({self.negative_quantity_pct:.1f}%)")
        logger.info(f"  Zero/null price  : {self.zero_price_count:,}  ({self.zero_price_pct:.1f}%)")
        logger.info(f"  Exact duplicates : {self.exact_duplicate_count:,}  ({self.exact_duplicate_pct:.1f}%)")
        logger.info(f"  Unique customers : {self.unique_customers:,}")
        logger.info(f"  Unique SKUs      : {self.unique_skus:,}")
        logger.info(f"  Unique countries : {self.unique_countries}")
        if self.errors:
            for e in self.errors:
                logger.error(f"  [ERROR] {e}")
        if self.warnings:
            for w in self.warnings:
                logger.warning(f"  [WARN]  {w}")
        status = "PASSED" if self.passed else "FAILED"
        logger.info(f"Validation result: {status}")
        logger.info("-" * 50)


# ── Validation logic ──────────────────────────────────────────────────────────

def run_validate(df: pd.DataFrame, cfg: PipelineConfig) -> ValidationReport:
    """Run all validation checks against the raw DataFrame.

    Args:
        df:  Raw DataFrame from Stage 1 (already renamed columns).
        cfg: Pipeline configuration.

    Returns:
        ValidationReport with all findings.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2 — VALIDATION")
    logger.info("=" * 60)

    n = len(df)
    errors: list[str] = []
    warnings: list[str] = []

    # ── 1. Column presence ────────────────────────────────────────────────────
    # Note: 'Customer ID' was renamed to 'CustomerID' in ingest.py
    required = [c if c != "Customer ID" else "CustomerID" for c in REQUIRED_COLUMNS]
    present_cols = set(df.columns.tolist())
    missing_required = [c for c in required if c not in present_cols]
    unexpected = [
        c for c in present_cols
        if c not in required and c not in ("source_sheet",)
    ]

    if missing_required:
        errors.append(f"Missing required columns: {missing_required}")
    if unexpected:
        warnings.append(f"Unexpected columns (will be kept): {unexpected}")

    # ── 2. Column-level checks ────────────────────────────────────────────────
    column_checks: list[ColumnCheck] = []
    for col in required:
        if col not in df.columns:
            column_checks.append(ColumnCheck(
                name=col, present=False, dtype="N/A",
                null_count=n, null_pct=100.0, sample_values=[],
            ))
            continue
        series = df[col]
        null_count = int(series.isna().sum())
        column_checks.append(ColumnCheck(
            name=col,
            present=True,
            dtype=str(series.dtype),
            null_count=null_count,
            null_pct=round(null_count / n * 100, 2) if n > 0 else 0.0,
            sample_values=series.dropna().head(5).tolist(),
        ))

    # ── 3. Date range ─────────────────────────────────────────────────────────
    date_min = date_max = None
    date_range_valid = False
    if "InvoiceDate" in df.columns:
        valid_dates = df["InvoiceDate"].dropna()
        if len(valid_dates) > 0:
            date_min = str(valid_dates.min().date())
            date_max = str(valid_dates.max().date())
            date_range_valid = (
                date_min >= EXPECTED_DATE_MIN and date_max <= EXPECTED_DATE_MAX
            )
            if not date_range_valid:
                warnings.append(
                    f"Date range {date_min}→{date_max} outside expected "
                    f"{EXPECTED_DATE_MIN}→{EXPECTED_DATE_MAX}"
                )

    # ── 4. Quantity checks ────────────────────────────────────────────────────
    neg_qty_count = 0
    if "Quantity" in df.columns:
        neg_qty_count = int((df["Quantity"].fillna(0) <= 0).sum())

    # ── 5. Price checks ───────────────────────────────────────────────────────
    zero_price_count = 0
    if "Price" in df.columns:
        zero_price_count = int((df["Price"].fillna(0) <= 0).sum())

    # ── 6. Cancellations ──────────────────────────────────────────────────────
    cancellation_count = 0
    if "Invoice" in df.columns:
        cancellation_count = int(df["Invoice"].astype(str).str.startswith("C").sum())

    # ── 7. Null customers ─────────────────────────────────────────────────────
    null_customer_count = 0
    if "CustomerID" in df.columns:
        null_customer_count = int(df["CustomerID"].isna().sum())

    # ── 8. Exact duplicates ───────────────────────────────────────────────────
    dup_cols = [c for c in ["Invoice", "StockCode", "CustomerID", "InvoiceDate", "Quantity"] if c in df.columns]
    exact_dup_count = int(df.duplicated(subset=dup_cols).sum()) if dup_cols else 0

    # ── 9. Cardinality ────────────────────────────────────────────────────────
    unique_customers = int(df["CustomerID"].nunique(dropna=True)) if "CustomerID" in df.columns else 0
    unique_skus      = int(df["StockCode"].nunique(dropna=True))  if "StockCode"   in df.columns else 0
    unique_countries = int(df["Country"].nunique(dropna=True))    if "Country"     in df.columns else 0

    # ── 10. Distribution summary ──────────────────────────────────────────────
    distribution_summary: dict = {}
    for col, agg in [("Quantity", ["min", "max", "mean", "median"]),
                     ("Price",    ["min", "max", "mean", "median"])]:
        if col in df.columns:
            s = df[col].dropna()
            distribution_summary[col] = {
                "min":    round(float(s.min()), 4),
                "max":    round(float(s.max()), 4),
                "mean":   round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std":    round(float(s.std()), 4),
                "p95":    round(float(s.quantile(0.95)), 4),
                "p99":    round(float(s.quantile(0.99)), 4),
                "p99_5":  round(float(s.quantile(0.995)), 4),
            }

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def pct(count: int) -> float:
        return round(count / n * 100, 2) if n > 0 else 0.0

    passed = len(errors) == 0

    report = ValidationReport(
        total_rows=n,
        total_columns=len(df.columns),
        missing_required_columns=missing_required,
        unexpected_columns=unexpected,
        column_checks=column_checks,
        date_min=date_min,
        date_max=date_max,
        date_range_valid=date_range_valid,
        negative_quantity_count=neg_qty_count,
        negative_quantity_pct=pct(neg_qty_count),
        zero_price_count=zero_price_count,
        zero_price_pct=pct(zero_price_count),
        cancellation_count=cancellation_count,
        cancellation_pct=pct(cancellation_count),
        null_customer_count=null_customer_count,
        null_customer_pct=pct(null_customer_count),
        exact_duplicate_count=exact_dup_count,
        exact_duplicate_pct=pct(exact_dup_count),
        unique_customers=unique_customers,
        unique_skus=unique_skus,
        unique_countries=unique_countries,
        distribution_summary=distribution_summary,
        passed=passed,
        warnings=warnings,
        errors=errors,
    )

    report.log_summary()

    if not passed:
        raise RuntimeError(
            f"Validation FAILED with {len(errors)} critical error(s). "
            "Fix the raw data or config before proceeding."
        )

    return report
