"""
models/baseline.py — Most-popular-globally baseline model.

This is the floor every real model must beat.

Strategy:
  "Always predict the same top-K categories — the ones bought most often
   in the training data." No personalisation, no sequence awareness.

This is not a bad model for e-commerce — popular products are popular.
But it has a critical flaw: it never adapts per customer. The Markov chain
exploits the conditional distribution P(next | current), which this model ignores.

Usage:
    from models.baseline import BaselineModel
    model = BaselineModel()
    model.fit(df_train)
    preds = model.predict_top_k(k=3)   # always the same list
    h3    = model.hit_rate_at_k(df_test, k=3)
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
import numpy as np


class BaselineModel:
    """Predict the globally top-K most purchased categories."""

    def __init__(self) -> None:
        self._category_counts: Counter = Counter()
        self._top_k_cache: dict[int, list[str]] = {}
        self._is_fitted: bool = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame, category_col: str = "category") -> "BaselineModel":
        """Compute global category frequencies from training data.

        Args:
            df_train:     Training DataFrame with a category column.
            category_col: Name of the category column.

        Returns:
            self (for chaining)
        """
        if category_col not in df_train.columns:
            raise ValueError(f"Column '{category_col}' not found in DataFrame. "
                             f"Available: {list(df_train.columns)}")

        self._category_counts = Counter(df_train[category_col].dropna().tolist())
        self._top_k_cache.clear()
        self._is_fitted = True

        total = sum(self._category_counts.values())
        print(f"[Baseline] Fitted on {total:,} transactions, "
              f"{len(self._category_counts)} unique categories.")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """Return top-K categories by global frequency.

        Returns:
            List of (category, probability) tuples, sorted descending by prob.
            Always returns the same list — no personalisation.
        """
        self._check_fitted()

        if k in self._top_k_cache:
            cats = self._top_k_cache[k]
        else:
            total = sum(self._category_counts.values())
            top   = self._category_counts.most_common(k)
            cats  = [cat for cat, _ in top]
            self._top_k_cache[k] = cats

        total = sum(self._category_counts.values())
        return [(cat, self._category_counts[cat] / total) for cat in cats]

    def predict_categories(self, current_state: str = "", k: int = 3) -> list[str]:
        """Return just the category names (no probabilities).

        Args:
            current_state: IGNORED. The baseline never uses sequence context.
                           Accepted so this method matches the MarkovModel API,
                           allowing both models to be passed to evaluate_model().
            k:             Number of categories to return.
        """
        return [cat for cat, _ in self.predict_top_k(k=k)]

    # ── Evaluation ────────────────────────────────────────────────────────────

    def hit_rate_at_k(
        self,
        test_sequences: list[tuple[str, str]],
        k: int = 3,
    ) -> dict[str, Any]:
        """Compute Hit Rate @ K for the baseline model.

        Args:
            test_sequences: List of (current_category, true_next_category) pairs.
            k:              Number of predictions to consider.

        Returns:
            dict with keys: hit_rate, hits, total, k, top_k_cats
        """
        self._check_fitted()

        top_k = set(self.predict_categories(k=k))
        hits  = sum(1 for _, true_next in test_sequences if true_next in top_k)
        total = len(test_sequences)
        hr    = hits / total if total > 0 else 0.0

        return {
            "hit_rate":   round(hr, 4),
            "hits":       hits,
            "total":      total,
            "k":          k,
            "top_k_cats": list(top_k),
        }

    def get_category_distribution(self) -> pd.DataFrame:
        """Return a DataFrame of all categories with their counts and percentages."""
        self._check_fitted()
        total = sum(self._category_counts.values())
        rows  = [
            {"category": cat, "count": cnt, "pct": round(cnt / total * 100, 2)}
            for cat, cnt in self._category_counts.most_common()
        ]
        return pd.DataFrame(rows)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit(df_train) before predicting.")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"BaselineModel({status}, categories={len(self._category_counts)})"
