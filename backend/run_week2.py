"""
run_week2.py — Week 2 deliverable: full evaluation with 2nd-order Markov + all visualisations.

Run from the backend directory (with venv activated):
    python run_week2.py

What this does:
  1. Loads cleaned sequences.parquet
  2. Assigns 15 product categories
  3. Temporal split (last 2 months as test)
  4. Trains Markov order=1, Markov order=2, Baseline
  5. Evaluates all three at K=1,3,5 with bootstrap 95% CI
  6. Generates 7 publication-quality dark-theme charts
  7. Prints final comparison table
  8. Saves week2_results.json + chart paths manifest
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BACKEND_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import pandas as pd

from models.category_encoder import encode_dataframe
from models.markov            import build_markov, df_to_sequences, Order2Adapter
from models.baseline          import BaselineModel
from models.evaluator         import (
    temporal_split,
    build_test_transitions,
    evaluate_model,
    print_comparison_table,
    EvalResult,
)
from models.visualiser        import generate_all_charts

PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
REPORTS_DIR   = BACKEND_DIR / "data" / "reports"
CHARTS_DIR    = BACKEND_DIR / "data" / "charts"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


# ── Order-2 evaluation helper ─────────────────────────────────────────────────

def evaluate_order2(
    markov2_raw:      object,          # raw order=2 MarkovModel
    df_train:         pd.DataFrame,
    df_test:          pd.DataFrame,
    k_values:         list[int] = None,
    n_bootstrap:      int = 1000,
) -> list[EvalResult]:
    """Evaluate order-2 Markov using 2-step transitions from the test set.

    For order-2 we need (prev_cat, current_cat) as the state.
    We build these from consecutive test transactions per customer.
    """
    from models.evaluator import bootstrap_ci, EvalResult

    if k_values is None:
        k_values = [1, 3, 5]

    # Build (prev_cat, current_cat, true_next_cat) triples from test data
    df_test_sorted = df_test.sort_values(["customer_id", "timestamp"])
    triples = []
    for _, grp in df_test_sorted.groupby("customer_id"):
        cats = grp["category"].dropna().tolist()
        if len(cats) >= 3:
            for i in range(len(cats) - 2):
                triples.append((cats[i], cats[i+1], cats[i+2]))

    # Also try train-last + test-first-two for customers with only 1-2 test rows
    df_train_sorted = df_train.sort_values(["customer_id", "timestamp"])
    last2_train = (
        df_train_sorted.groupby("customer_id")
        .apply(lambda g: g.tail(2)["category"].tolist())
    )
    first_test = (
        df_test_sorted.groupby("customer_id")
        .apply(lambda g: g.iloc[0]["category"] if len(g) > 0 else None)
    )
    for cid in last2_train.index:
        if cid in first_test.index and first_test[cid]:
            prev_cats = last2_train[cid]
            if len(prev_cats) == 2:
                triples.append((prev_cats[0], prev_cats[1], first_test[cid]))

    print(f"[Order-2 Eval] {len(triples):,} (prev, current, next) test triples")

    results = []
    adapter = Order2Adapter(markov2_raw)

    for k in k_values:
        hits_arr = []
        for prev_cat, cur_cat, true_next in triples:
            adapter.set_context(prev_cat)
            try:
                preds = adapter.predict_categories(cur_cat, k=k)
            except Exception:
                preds = []
            hits_arr.append(1 if true_next in preds else 0)

        hr       = float(np.mean(hits_arr)) if hits_arr else 0.0
        ci_lo, ci_hi = bootstrap_ci(hits_arr, n_bootstrap=n_bootstrap)
        result = EvalResult(
            model_name  = "Markov (order=2)",
            k           = k,
            hit_rate    = round(hr, 4),
            hits        = int(sum(hits_arr)),
            total       = len(hits_arr),
            ci_lower    = round(ci_lo, 4),
            ci_upper    = round(ci_hi, 4),
            n_bootstrap = n_bootstrap,
        )
        results.append(result)
        print(str(result))

    return results


def run() -> None:

    # ── 1. Load ────────────────────────────────────────────────────────────────
    section("1. LOADING DATA")
    seq_path = PROCESSED_DIR / "sequences.parquet"
    if not seq_path.exists():
        print(f"[ERROR] Not found: {seq_path}\nRun: python main.py")
        sys.exit(1)

    df = pd.read_parquet(seq_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  Rows: {len(df):,}  |  Customers: {df['customer_id'].nunique():,}  "
          f"|  Date range: {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")

    # ── 2. Categories ──────────────────────────────────────────────────────────
    section("2. CATEGORY ASSIGNMENT")
    df = encode_dataframe(df, desc_col="description")
    cat_dist = df["category"].value_counts()
    print(f"  {'Category':<22} {'Count':>8}  {'%':>5}")
    print(f"  {'-'*22} {'-'*8}  {'-'*5}")
    for cat, cnt in cat_dist.items():
        print(f"  {cat:<22} {cnt:>8,}  {cnt/len(df)*100:>4.1f}%")

    # ── 3. Temporal split ──────────────────────────────────────────────────────
    section("3. TEMPORAL TRAIN/TEST SPLIT  (last 2 months held out)")
    df_train, df_test = temporal_split(df, timestamp_col="timestamp", test_months=2)
    print(f"\n  Train: {len(df_train):,} rows ({len(df_train)/len(df):.1%})")
    print(f"  Test:  {len(df_test):,} rows ({len(df_test)/len(df):.1%})")

    # ── 4. Build sequences ─────────────────────────────────────────────────────
    section("4. BUILDING TRAINING SEQUENCES")
    train_seqs = df_to_sequences(df_train, customer_col="customer_id",
                                  category_col="category", timestamp_col="timestamp")

    # ── 5. Train models ────────────────────────────────────────────────────────
    section("5. TRAINING MODELS")
    print("\n  [5a] Markov order=1 ...")
    markov1 = build_markov(train_seqs, order=1)

    print("\n  [5b] Markov order=2 ...")
    markov2_raw = build_markov(train_seqs, order=2)

    print("\n  [5c] Baseline ...")
    baseline = BaselineModel()
    baseline.fit(df_train, category_col="category")
    print(f"  Top-3: {baseline.predict_categories(k=3)}")

    # ── 6. Build test transitions ──────────────────────────────────────────────
    section("6. BUILDING TEST TRANSITIONS")
    test_pairs = build_test_transitions(
        df_train, df_test,
        customer_col="customer_id", category_col="category", timestamp_col="timestamp",
    )

    # ── 7. Evaluate ────────────────────────────────────────────────────────────
    section("7. EVALUATION  (1000 bootstrap samples, seed=42)")

    print("\n  >> Markov (order=1):")
    res_m1 = evaluate_model(markov1, test_pairs, model_name="Markov (order=1)",
                             k_values=[1, 3, 5], n_bootstrap=1000)

    print("\n  >> Markov (order=2):")
    res_m2 = evaluate_order2(markov2_raw, df_train, df_test,
                              k_values=[1, 3, 5], n_bootstrap=1000)

    print("\n  >> Most-Popular Baseline:")
    res_b = evaluate_model(baseline, test_pairs, model_name="Most-Popular",
                            k_values=[1, 3, 5], n_bootstrap=1000)

    # ── 8. Comparison table ────────────────────────────────────────────────────
    section("8. FINAL RESULTS — Markov (order=1) vs Baseline")
    print_comparison_table(res_m1, res_b)

    section("9. FINAL RESULTS — Markov (order=2) vs Baseline")
    print_comparison_table(res_m2, res_b)

    # ── 9. Generate charts ─────────────────────────────────────────────────────
    section("10. GENERATING CHARTS")
    chart_paths = generate_all_charts(
        df               = df,
        markov_model     = markov1,
        baseline_model   = baseline,
        results_markov   = res_m1,
        results_baseline = res_b,
        test_transitions = test_pairs,
        results_markov2  = res_m2,
    )

    # ── 10. Save JSON results ─────────────────────────────────────────────────
    output = {
        "dataset": {
            "total_rows":       len(df),
            "unique_customers": int(df["customer_id"].nunique()),
            "train_rows":       len(df_train),
            "test_rows":        len(df_test),
            "test_transitions": len(test_pairs),
        },
        "markov_order1":  [r.to_dict() for r in res_m1],
        "markov_order2":  [r.to_dict() for r in res_m2],
        "baseline":       [r.to_dict() for r in res_b],
        "charts":         [str(p) for p in chart_paths],
        "category_distribution": {cat: int(cnt) for cat, cnt in cat_dist.items()},
    }

    out_path = REPORTS_DIR / "week2_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(f"\n  Results saved -> {out_path}")
    print(f"  Charts saved  -> {CHARTS_DIR}")
    print()
    print("  WEEK 2 COMPLETE.")
    print("  Charts: ", [p.name for p in chart_paths])


if __name__ == "__main__":
    run()
