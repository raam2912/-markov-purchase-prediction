"""
pipeline/sequence_builder.py — Stage 4: Build per-customer purchase sequences.

Takes the cleaned flat transaction DataFrame and produces:
  1. A long-format Parquet file (one row per event, ordered by customer + time)
  2. A JSON file of per-customer sequence dictionaries (for easy inspection)
  3. Derived features on each event:
     - days_since_prior_order      (float, NaN for first order)
     - session_id                  (int, resets when gap > session_gap_days)
     - purchase_count              (cumulative purchase count per customer)
     - is_repeat_sku               (bool, has this customer bought this SKU before?)
     - total_spend                 (Quantity * Price for this row)
     - customer_total_skus         (unique SKUs bought by this customer to date)

Customers with fewer transactions than min_transactions_per_customer are excluded.
This prevents noisy single-purchase customers from polluting the transition matrix.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from config import PipelineConfig, PROCESSED_DIR


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class SequenceEvent:
    """One purchase event in a customer's sequence."""
    event_idx: int
    invoice: str
    sku: str
    description: str
    quantity: float
    price: float
    total_spend: float
    country: str
    timestamp: str           # ISO-8601 string
    days_since_prior: float | None
    session_id: int
    purchase_count: int
    is_repeat_sku: bool
    customer_total_skus: int


@dataclass
class CustomerSequence:
    customer_id: str
    country: str             # most frequent country of purchase
    total_transactions: int
    unique_skus: int
    total_spend: float
    date_first: str
    date_last: str
    num_sessions: int
    events: list[SequenceEvent] = field(default_factory=list)


@dataclass
class SequenceReport:
    total_customers_raw: int
    total_customers_kept: int
    customers_excluded_short: int
    total_events: int
    avg_sequence_length: float
    median_sequence_length: float
    avg_unique_skus_per_customer: float
    avg_sessions_per_customer: float
    total_unique_skus: int
    date_range_min: str
    date_range_max: str

    def log(self) -> None:
        logger.info("-" * 50)
        logger.info("SEQUENCE BUILD SUMMARY:")
        logger.info(f"  Customers kept       : {self.total_customers_kept:,}")
        logger.info(f"  Customers excluded   : {self.customers_excluded_short:,} (too few transactions)")
        logger.info(f"  Total events         : {self.total_events:,}")
        logger.info(f"  Avg sequence length  : {self.avg_sequence_length:.1f}")
        logger.info(f"  Median seq length    : {self.median_sequence_length:.1f}")
        logger.info(f"  Avg unique SKUs/cust : {self.avg_unique_skus_per_customer:.1f}")
        logger.info(f"  Avg sessions/cust    : {self.avg_sessions_per_customer:.1f}")
        logger.info(f"  Total unique SKUs    : {self.total_unique_skus:,}")
        logger.info(f"  Date range           : {self.date_range_min} → {self.date_range_max}")
        logger.info("-" * 50)


# ── Core build logic ──────────────────────────────────────────────────────────

def _build_customer_sequence(
    customer_id: str,
    grp: pd.DataFrame,
    session_gap_days: int,
    include_financials: bool,
) -> CustomerSequence:
    """Build a CustomerSequence from one customer's sorted transaction group."""
    grp = grp.sort_values("InvoiceDate").reset_index(drop=True)

    events: list[SequenceEvent] = []
    seen_skus: set[str] = set()
    session_id = 0
    prev_date: pd.Timestamp | None = None

    for idx, row in grp.iterrows():
        ts = row["InvoiceDate"]
        sku = row["StockCode"]

        # Days since prior order
        if prev_date is None:
            days_since_prior = None
        else:
            delta_days = (ts - prev_date).total_seconds() / 86400
            days_since_prior = round(delta_days, 4)
            if delta_days > session_gap_days:
                session_id += 1

        is_repeat = sku in seen_skus
        seen_skus.add(sku)

        total_spend = round(float(row["Quantity"]) * float(row["Price"]), 4)

        events.append(SequenceEvent(
            event_idx=idx,
            invoice=str(row["Invoice"]),
            sku=sku,
            description=str(row["Description"]),
            quantity=float(row["Quantity"]),
            price=float(row["Price"]) if include_financials else 0.0,
            total_spend=total_spend if include_financials else 0.0,
            country=str(row["Country"]),
            timestamp=ts.isoformat(),
            days_since_prior=days_since_prior,
            session_id=session_id,
            purchase_count=int(idx) + 1,
            is_repeat_sku=is_repeat,
            customer_total_skus=len(seen_skus),
        ))

        prev_date = ts

    # Customer-level summary
    most_common_country = grp["Country"].mode().iloc[0] if len(grp) > 0 else "UNKNOWN"
    total_spend = sum(e.total_spend for e in events)

    return CustomerSequence(
        customer_id=customer_id,
        country=str(most_common_country),
        total_transactions=len(events),
        unique_skus=len(seen_skus),
        total_spend=round(total_spend, 4),
        date_first=grp["InvoiceDate"].min().isoformat(),
        date_last=grp["InvoiceDate"].max().isoformat(),
        num_sessions=session_id + 1,
        events=events,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run_sequence_builder(
    df: pd.DataFrame,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, list[CustomerSequence], SequenceReport]:
    """Build per-customer sequences from the cleaned flat DataFrame.

    Args:
        df:  Cleaned DataFrame from Stage 3.
        cfg: Pipeline configuration.

    Returns:
        - Long-format DataFrame (each row = one event with derived features)
        - List of CustomerSequence objects (for JSON export)
        - SequenceReport with statistics
    """
    logger.info("=" * 60)
    logger.info("STAGE 4 — SEQUENCE BUILDING")
    logger.info("=" * 60)

    sc = cfg.sequences
    total_customers_raw = df["CustomerID"].nunique()
    logger.info(f"Building sequences for {total_customers_raw:,} unique customers …")

    # Sort globally before grouping
    df = df.sort_values(["CustomerID", "InvoiceDate"]).reset_index(drop=True)

    # Filter customers with too few transactions
    tx_counts = df.groupby("CustomerID").size()
    eligible_customers = tx_counts[tx_counts >= sc.min_transactions_per_customer].index
    customers_excluded = total_customers_raw - len(eligible_customers)

    if customers_excluded > 0:
        logger.info(
            f"Excluded {customers_excluded:,} customers with < "
            f"{sc.min_transactions_per_customer} transactions"
        )
    df_eligible = df[df["CustomerID"].isin(eligible_customers)]

    # Build sequences customer by customer
    sequences: list[CustomerSequence] = []
    groups = list(df_eligible.groupby("CustomerID"))

    for cid, grp in tqdm(groups, desc="Building sequences", unit="customers"):
        seq = _build_customer_sequence(
            customer_id=str(cid),
            grp=grp,
            session_gap_days=sc.session_gap_days,
            include_financials=sc.include_financials,
        )
        sequences.append(seq)

    # ── Build long-format DataFrame ───────────────────────────────────────────
    records: list[dict] = []
    for seq in sequences:
        for ev in seq.events:
            records.append({
                "customer_id":         seq.customer_id,
                "customer_country":    seq.country,
                "invoice":             ev.invoice,
                "sku":                 ev.sku,
                "description":         ev.description,
                "quantity":            ev.quantity,
                "price":               ev.price,
                "total_spend":         ev.total_spend,
                "timestamp":           ev.timestamp,
                "days_since_prior":    ev.days_since_prior,
                "session_id":          ev.session_id,
                "purchase_count":      ev.purchase_count,
                "is_repeat_sku":       ev.is_repeat_sku,
                "customer_total_skus": ev.customer_total_skus,
            })

    df_seq = pd.DataFrame(records)
    if not df_seq.empty:
        df_seq["timestamp"] = pd.to_datetime(df_seq["timestamp"])

    # ── Persist outputs ───────────────────────────────────────────────────────
    seq_parquet = PROCESSED_DIR / "sequences.parquet"
    df_seq.to_parquet(seq_parquet, index=False, engine="pyarrow")
    logger.success(f"Sequences parquet saved: {seq_parquet}")

    # Save transactions_clean.parquet (flat cleaned, for reference)
    clean_parquet = PROCESSED_DIR / "transactions_clean.parquet"
    df_eligible.reset_index(drop=True).to_parquet(clean_parquet, index=False, engine="pyarrow")
    logger.success(f"Clean transactions saved: {clean_parquet}")

    # Save JSON (top 5000 customers only if dataset is large, else all)
    json_limit = 5000
    seq_json_path = PROCESSED_DIR / "sequences.json"
    json_data = [
        {
            "customer_id": s.customer_id,
            "country": s.country,
            "total_transactions": s.total_transactions,
            "unique_skus": s.unique_skus,
            "total_spend": s.total_spend,
            "date_first": s.date_first,
            "date_last": s.date_last,
            "num_sessions": s.num_sessions,
            "events": [asdict(e) for e in s.events],
        }
        for s in sequences[:json_limit]
    ]
    with open(seq_json_path, "w", encoding="utf-8") as fh:
        json.dump(json_data, fh, indent=2, default=str)
    if len(sequences) > json_limit:
        logger.warning(f"JSON export limited to {json_limit} customers (full data in Parquet)")
    logger.success(f"Sequences JSON saved: {seq_json_path}")

    # ── Build report ──────────────────────────────────────────────────────────
    seq_lengths = [s.total_transactions for s in sequences]
    unique_skus_per = [s.unique_skus for s in sequences]
    sessions_per = [s.num_sessions for s in sequences]

    report = SequenceReport(
        total_customers_raw=total_customers_raw,
        total_customers_kept=len(sequences),
        customers_excluded_short=customers_excluded,
        total_events=len(df_seq),
        avg_sequence_length=round(float(np.mean(seq_lengths)), 2) if seq_lengths else 0.0,
        median_sequence_length=round(float(np.median(seq_lengths)), 2) if seq_lengths else 0.0,
        avg_unique_skus_per_customer=round(float(np.mean(unique_skus_per)), 2) if unique_skus_per else 0.0,
        avg_sessions_per_customer=round(float(np.mean(sessions_per)), 2) if sessions_per else 0.0,
        total_unique_skus=int(df_seq["sku"].nunique()) if not df_seq.empty else 0,
        date_range_min=str(df_seq["timestamp"].min().date()) if not df_seq.empty else "N/A",
        date_range_max=str(df_seq["timestamp"].max().date()) if not df_seq.empty else "N/A",
    )

    report.log()
    logger.success("Stage 4 complete.")
    return df_seq, sequences, report
