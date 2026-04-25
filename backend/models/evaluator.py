"""
models/evaluator.py — Hit Rate @ K evaluation with temporal train/test split.

Temporal split rule:
  Train : months 1–10  (Dec 2009 → Sep 2011 for the full 2-year dataset)
  Test  : months 11–12 (Oct 2011 → Nov 2011)
  
  CRITICAL: We NEVER shuffle. Time order must be respected.
  Data leakage would inflate results by ~15-20 percentage points.

Hit Rate @ K:
  For each (customer, sequence position) in the test set:
    - Given: all purchases up to position i (the "history")
    - Target: the purchase at position i+1 (the "true next")
    - Predict: top-K categories using the model's predict_top_k()
    - Hit: 1 if true_next in top-K, 0 otherwise
  HR@K = mean(hits) over all test transitions

Bootstrap Confidence Intervals:
  - 1000 bootstrap samples
  - 95% CI: [2.5th percentile, 97.5th percentile]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


# ── Protocol for any model that can predict ───────────────────────────────────

class PredictableModel(Protocol):
    """Any object with a predict_categories(state, k) method."""
    def predict_categories(self, current_state: str, k: int = 3) -> list[str]:
        ...


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Results from one evaluation run at one K value."""
    model_name:  str
    k:           int
    hit_rate:    float
    hits:        int
    total:       int
    ci_lower:    float
    ci_upper:    float
    n_bootstrap: int

    def __str__(self) -> str:
        return (
            f"{self.model_name:<18} HR@{self.k}: {self.hit_rate:.4f} "
            f"({self.hits}/{self.total}) "
            f"95%CI [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model":    self.model_name,
            "k":        self.k,
            "hit_rate": round(self.hit_rate, 4),
            "hits":     self.hits,
            "total":    self.total,
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
        }


# ── Temporal split ────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_months: int   = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted DataFrame into train and test sets.

    Holds out the LAST `test_months` calendar months as the test set.
    Everything before that is training data.

    Args:
        df:            Long-format DataFrame with a timestamp column.
        timestamp_col: Name of the datetime column.
        test_months:   Number of months from the END to hold out as test.
                       Default: 2 (last 2 months = Oct–Nov 2011 for this dataset).

    Returns:
        (df_train, df_test) — non-overlapping, time-ordered DataFrames.

    Notes:
        - Split point = max_date - test_months.  This gives a stable ratio
          regardless of total dataset length (never trains on <50% of data).
        - Customers can appear in both sets — this is the realistic deployment
          scenario (we use train history to predict first test purchase).
        - NEVER shuffle — time order is sacred to avoid data leakage.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)

    max_date   = df[timestamp_col].max()
    split_date = max_date - pd.DateOffset(months=test_months)
    # Snap to start of that month for clean calendar boundaries
    split_date = split_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    df_train = df[df[timestamp_col] <  split_date].copy()
    df_test  = df[df[timestamp_col] >= split_date].copy()

    print("[Evaluator] Temporal split (last 2 months held out as test):")
    print(f"  Train: {df_train[timestamp_col].min().date()} -> "
          f"{df_train[timestamp_col].max().date()} ({len(df_train):,} rows)")
    print(f"  Test:  {df_test[timestamp_col].min().date()} -> "
          f"{df_test[timestamp_col].max().date()} ({len(df_test):,} rows)")
    print(f"  Split date: {split_date.date()}")

    return df_train, df_test


# ── Test sequence builder ─────────────────────────────────────────────────────

def build_test_transitions(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    customer_col: str  = "customer_id",
    category_col: str  = "category",
    timestamp_col: str = "timestamp",
) -> list[tuple[str, str]]:
    """Build (current_category, true_next_category) pairs for evaluation.

    For each customer in the test set:
      - Take their last purchase from TRAINING data as the "current state"
      - Take their FIRST purchase from TEST data as the "true next"
      - This avoids any data leakage — the model never sees the true next

    Customers not present in training data are excluded (cold-start problem
    is out of scope for Phase 1).

    Returns:
        List of (current_cat, true_next_cat) tuples.
    """
    # Last category in TRAIN per customer
    df_train = df_train.sort_values([customer_col, timestamp_col])
    last_train = (
        df_train.groupby(customer_col)
        .apply(lambda g: g.iloc[-1][category_col])
        .rename("last_train_cat")
    )

    # First category in TEST per customer
    df_test = df_test.sort_values([customer_col, timestamp_col])
    first_test = (
        df_test.groupby(customer_col)
        .apply(lambda g: g.iloc[0][category_col])
        .rename("first_test_cat")
    )

    # Inner join: only customers present in BOTH sets
    merged = pd.DataFrame({
        "current_cat":  last_train,
        "true_next_cat": first_test,
    }).dropna()

    transitions = list(zip(merged["current_cat"], merged["true_next_cat"]))

    print(f"[Evaluator] Built {len(transitions):,} test transitions "
          f"from {len(merged):,} overlapping customers.")

    # Remove pairs where either category is NaN / missing
    transitions = [(c, n) for c, n in transitions if c and n]
    return transitions


# ── Hit Rate @ K ─────────────────────────────────────────────────────────────

def hit_rate_at_k(
    predictions: list[list[str]],
    true_nexts:  list[str],
    k: int = 3,
) -> float:
    """Compute Hit Rate @ K.

    Args:
        predictions: List of top-K prediction lists (one per test case).
        true_nexts:  List of true next categories (one per test case).
        k:           Truncation threshold.

    Returns:
        Hit rate as a float in [0, 1].
    """
    assert len(predictions) == len(true_nexts), "Length mismatch."
    hits = sum(
        1 for preds, true in zip(predictions, true_nexts)
        if true in preds[:k]
    )
    return hits / len(true_nexts) if true_nexts else 0.0


def bootstrap_ci(
    hit_indicators: list[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for hit rate.

    Args:
        hit_indicators: List of 1/0 hit indicators (one per test case).
        n_bootstrap:    Number of bootstrap resamples.
        confidence:     Confidence level (default 0.95 → 95% CI).
        rng_seed:       Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) of the CI.
    """
    rng  = np.random.default_rng(rng_seed)
    arr  = np.array(hit_indicators, dtype=float)
    means = [
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    alpha = 1 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


# ── Full evaluation run ───────────────────────────────────────────────────────

def evaluate_model(
    model: Any,                          # any object with predict_categories(state, k)
    test_transitions: list[tuple[str, str]],
    model_name: str = "Model",
    k_values: list[int] = None,
    n_bootstrap: int = 1000,
) -> list[EvalResult]:
    """Run a full evaluation for multiple K values with confidence intervals.

    Args:
        model:            Fitted model with predict_categories(state, k) method.
        test_transitions: List of (current_cat, true_next_cat) pairs.
        model_name:       Label for display.
        k_values:         List of K values to evaluate (default: [1, 3, 5]).
        n_bootstrap:      Bootstrap samples for CI.

    Returns:
        List of EvalResult objects, one per K value.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    results = []

    for k in k_values:
        hits_arr = []
        for current_cat, true_next in test_transitions:
            try:
                preds = model.predict_categories(current_cat, k=k)
            except Exception:
                preds = []
            hits_arr.append(1 if true_next in preds else 0)

        hr        = float(np.mean(hits_arr)) if hits_arr else 0.0
        ci_lo, ci_hi = bootstrap_ci(hits_arr, n_bootstrap=n_bootstrap)

        result = EvalResult(
            model_name  = model_name,
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


def print_comparison_table(
    results_markov: list[EvalResult],
    results_baseline: list[EvalResult],
) -> None:
    """Print a side-by-side comparison table of Markov vs Baseline results."""
    print("\n" + "=" * 72)
    print(f"{'K':<4} {'Markov HR':>10} {'95% CI':>18} {'Baseline HR':>12} {'Lift':>8}")
    print("=" * 72)

    markov_dict   = {r.k: r for r in results_markov}
    baseline_dict = {r.k: r for r in results_baseline}
    all_k         = sorted(set(markov_dict) | set(baseline_dict))

    for k in all_k:
        m = markov_dict.get(k)
        b = baseline_dict.get(k)
        if m and b:
            lift = m.hit_rate - b.hit_rate
            ci   = f"[{m.ci_lower:.4f}, {m.ci_upper:.4f}]"
            marker = " *" if m.ci_lower > b.ci_upper else ""  # statistically significant
            print(f"{k:<4} {m.hit_rate:>10.4f} {ci:>18} {b.hit_rate:>12.4f} {lift:>+8.4f}{marker}")

    print("=" * 72)
    print("* = statistically significant improvement (CI lower > baseline upper)")
    print()
