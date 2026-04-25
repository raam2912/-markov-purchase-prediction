"""
run_baseline.py — Week 1 deliverable: run the full pipeline and output baseline results.

Run from the backend directory (with venv activated):
    python run_baseline.py

This script:
1. Loads the cleaned sequences.parquet
2. Assigns product categories (keyword rules)
3. Applies temporal train/test split (10 months train, 2 months test)
4. Trains the Markov chain model on training data
5. Trains the most-popular baseline model
6. Evaluates both at K=1, 3, 5 with 95% bootstrap confidence intervals
7. Prints a clear comparison table
8. Saves results to data/reports/week1_results.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd
import numpy as np

from models.category_encoder import encode_dataframe
from models.markov            import build_markov, df_to_sequences
from models.baseline          import BaselineModel
from models.evaluator         import (
    temporal_split,
    build_test_transitions,
    evaluate_model,
    print_comparison_table,
)

PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
REPORTS_DIR   = BACKEND_DIR / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    width = 62
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def run() -> None:
    # ── 1. Load data ──────────────────────────────────────────────────────────
    section("1. LOADING DATA")
    seq_path = PROCESSED_DIR / "sequences.parquet"
    if not seq_path.exists():
        print(f"[ERROR] File not found: {seq_path}")
        print("Run the main pipeline first: python main.py")
        sys.exit(1)

    df = pd.read_parquet(seq_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"  Rows loaded:         {len(df):>10,}")
    print(f"  Unique customers:    {df['customer_id'].nunique():>10,}")
    print(f"  Unique SKUs:         {df['sku'].nunique():>10,}")
    print(f"  Date range:          {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")

    # ── 2. Category assignment ────────────────────────────────────────────────
    section("2. CATEGORY ASSIGNMENT (Keyword Rules)")
    df = encode_dataframe(df, desc_col="description")

    cat_dist = df["category"].value_counts()
    print(f"  Total categories: {len(cat_dist)}")
    print(f"  {'Category':<22} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*22} {'-'*8}  {'-'*6}")
    for cat, cnt in cat_dist.items():
        print(f"  {cat:<22} {cnt:>8,}  {cnt/len(df)*100:>5.1f}%")

    other_pct = cat_dist.get("OTHER", 0) / len(df) * 100
    if other_pct > 40:
        print(f"\n  [WARN] OTHER is {other_pct:.1f}% — consider refining keyword rules.")

    # ── 3. Temporal split ─────────────────────────────────────────────────────
    section("3. TEMPORAL TRAIN/TEST SPLIT")
    df_train, df_test = temporal_split(df, timestamp_col="timestamp", test_months=2)
    print(f"\n  Train fraction: {len(df_train)/len(df):>6.1%}")
    print(f"  Test  fraction: {len(df_test)/len(df):>6.1%}")

    # ── 4. Build sequences ────────────────────────────────────────────────────
    section("4. BUILDING TRAINING SEQUENCES")
    train_seqs = df_to_sequences(
        df_train,
        customer_col  = "customer_id",
        category_col  = "category",
        timestamp_col = "timestamp",
    )

    # ── 5. Train Markov model ──────────────────────────────────────────────────
    section("5. TRAINING MARKOV CHAIN (order=1)")
    markov_model = build_markov(train_seqs, order=1)

    # ── 6. Train baseline model ────────────────────────────────────────────────
    section("6. TRAINING BASELINE MODEL")
    baseline = BaselineModel()
    baseline.fit(df_train, category_col="category")
    print(f"\n  Baseline top-3: {baseline.predict_categories(k=3)}")
    print(f"  Baseline top-5: {baseline.predict_categories(k=5)}")

    # ── 7. Build test transitions ─────────────────────────────────────────────
    section("7. BUILDING TEST TRANSITIONS")
    test_pairs = build_test_transitions(
        df_train      = df_train,
        df_test       = df_test,
        customer_col  = "customer_id",
        category_col  = "category",
        timestamp_col = "timestamp",
    )

    if len(test_pairs) < 50:
        print(f"\n[WARN] Only {len(test_pairs)} test transitions. "
              "Results may be unreliable. Check split parameters.")

    # ── 8. Evaluate ────────────────────────────────────────────────────────────
    section("8. EVALUATION — Hit Rate @ K=1,3,5 (1000 bootstrap samples)")
    print("\n  >> Markov Chain (order=1):")
    results_markov = evaluate_model(
        markov_model,
        test_pairs,
        model_name  = "Markov (order=1)",
        k_values    = [1, 3, 5],
        n_bootstrap = 1000,
    )

    print("\n  >> Most-Popular Baseline:")
    results_baseline = evaluate_model(
        baseline,
        test_pairs,
        model_name  = "Most-Popular",
        k_values    = [1, 3, 5],
        n_bootstrap = 1000,
    )

    # ── 9. Comparison table ───────────────────────────────────────────────────
    section("9. FINAL RESULTS COMPARISON")
    print_comparison_table(results_markov, results_baseline)

    # ── 10. Save results ──────────────────────────────────────────────────────
    output = {
        "dataset": {
            "total_rows": len(df),
            "unique_customers": int(df["customer_id"].nunique()),
            "date_range": [
                str(df["timestamp"].min().date()),
                str(df["timestamp"].max().date()),
            ],
            "train_rows": len(df_train),
            "test_rows":  len(df_test),
            "test_transitions": len(test_pairs),
        },
        "markov": [r.to_dict() for r in results_markov],
        "baseline": [r.to_dict() for r in results_baseline],
        "category_distribution": {
            cat: int(cnt) for cat, cnt in cat_dist.items()
        },
    }

    out_path = REPORTS_DIR / "week1_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n  Results saved -> {out_path}")
    print()
    print("  Week 1 complete. Ready for GitHub push.")


if __name__ == "__main__":
    run()
