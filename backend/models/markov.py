"""
models/markov.py — Markov Chain for purchase prediction (order 1 & 2).

build_markov(sequences, order=1|2)
  Input : list of sequences, each sequence = list of category strings
  Output: MarkovModel (order=1) — state is a single category string
          The model can also be used via Order2Adapter for a uniform API.

Order-1: state = current category   (15 states, 15x15 matrix)
Order-2: state = (prev, current)     (up to 225 states, 225x15 matrix)
         Higher expressiveness at the cost of sparser states.

Both orders share the same MarkovModel dataclass. For order=2 the
'current_state' passed to predict_top_k must be a tuple (prev, current).
The Order2Adapter wraps a MarkovModel trained at order=2 so that it can
be called with a plain string (it looks up the customer's previous purchase
from a context dict and builds the tuple internally).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MarkovModel:
    """Trained first-order Markov chain model.

    Attributes:
        order:          Markov order (1 = depends on last 1 state).
        transitions:    Dict[state → Dict[next_state → probability]].
        state_counts:   Raw observation counts per state.
        global_dist:    Fallback unigram distribution (for unseen states).
        categories:     Sorted list of all known category names.
        total_sequences: Number of training sequences.
        total_transitions: Total state transitions observed.
    """
    order: int
    transitions: dict[str, dict[str, float]] = field(default_factory=dict)
    state_counts: dict[str, dict[str, int]]  = field(default_factory=dict)
    global_dist:  dict[str, float]           = field(default_factory=dict)
    categories:   list[str]                  = field(default_factory=list)
    total_sequences: int   = 0
    total_transitions: int = 0

    def predict_top_k(self, current_state: str, k: int = 3) -> list[tuple[str, float]]:
        """Return top-K predicted next categories given current category.

        Args:
            current_state: Current product category.
            k:             Number of predictions.

        Returns:
            List of (category, probability), sorted by probability descending.
            Falls back to global distribution if state not seen in training.
        """
        if current_state in self.transitions:
            dist = self.transitions[current_state]
        else:
            # Unseen state → use global category distribution
            dist = self.global_dist

        # Sort by probability, descending
        ranked = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def predict_categories(self, current_state: str, k: int = 3) -> list[str]:
        """Return just the category names (no probabilities)."""
        return [cat for cat, _ in self.predict_top_k(current_state, k)]

    def coverage(self) -> float:
        """Fraction of all possible states that have observed transitions."""
        n = len(self.categories)
        if n == 0:
            return 0.0
        # order=1: N states; order=2: up to N^2 states
        possible_states = n if self.order == 1 else n * n
        return min(1.0, len(self.transitions) / possible_states)

    def sparsity(self) -> float:
        """Fraction of transition matrix cells that are zero (or missing)."""
        n = len(self.categories)
        if n == 0:
            return 1.0
        possible_states = n if self.order == 1 else n * n
        filled_cells = sum(len(v) for v in self.transitions.values())
        total_cells  = possible_states * n
        return max(0.0, 1.0 - filled_cells / total_cells)

    def to_dataframe(self) -> pd.DataFrame:
        """Return transition matrix as a DataFrame (rows=from, cols=to)."""
        cats = self.categories
        matrix = pd.DataFrame(0.0, index=cats, columns=cats)
        for from_cat, row in self.transitions.items():
            for to_cat, prob in row.items():
                if from_cat in matrix.index and to_cat in matrix.columns:
                    matrix.loc[from_cat, to_cat] = prob
        return matrix

    def summary(self) -> dict[str, Any]:
        return {
            "order":             self.order,
            "categories":        len(self.categories),
            "total_sequences":   self.total_sequences,
            "total_transitions": self.total_transitions,
            "coverage":          round(self.coverage(), 4),
            "sparsity":          round(self.sparsity(), 4),
        }


# ── Builder ───────────────────────────────────────────────────────────────────

def build_markov(
    sequences: list[list[str]],
    order: int = 1,
) -> MarkovModel:
    """Build a normalised Markov transition model from sequences of categories.

    Args:
        sequences: List of sequences. Each sequence is a list of category strings
                   in chronological order, e.g. ["KITCHEN_DINING", "HOME_DECOR", …].
        order:     Markov order. 1 = depends on last 1 state (classic Markov).

    Returns:
        Fitted MarkovModel.

    Raises:
        ValueError: If sequences is empty or order < 1.
    """
    if not sequences:
        raise ValueError("sequences must be non-empty.")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}.")
    if order > 2:
        raise NotImplementedError("Only order=1 and order=2 are supported.")

    # ── Count transitions ──────────────────────────────────────────────────────
    raw_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    global_counts: defaultdict[str, int] = defaultdict(int)
    total_transitions = 0
    all_categories: set[str] = set()

    for seq in sequences:
        # Skip sequences shorter than order + 1 (no transitions possible)
        if len(seq) < order + 1:
            continue

        for i in range(len(seq) - order):
            if order == 1:
                from_state: str | tuple = seq[i]
            else:  # order == 2
                from_state = (seq[i], seq[i + 1])
            to_state = seq[i + order]

            raw_counts[from_state][to_state] += 1
            global_counts[to_state] += 1
            if order == 1:
                all_categories.add(from_state)      # type: ignore[arg-type]
            else:
                all_categories.add(from_state[0])   # type: ignore[index]
                all_categories.add(from_state[1])   # type: ignore[index]
            all_categories.add(to_state)
            total_transitions += 1

    if total_transitions == 0:
        raise ValueError("No transitions found. Check that sequences have length >= 2.")

    # ── Normalise rows to probabilities ───────────────────────────────────────
    transitions: dict[str, dict[str, float]] = {}
    state_counts: dict[str, dict[str, int]]  = {}

    for from_state, counts in raw_counts.items():
        row_total = sum(counts.values())
        transitions[from_state] = {
            to_state: cnt / row_total
            for to_state, cnt in counts.items()
        }
        state_counts[from_state] = dict(counts)

    # ── Global unigram distribution (fallback for unseen states) ──────────────
    global_total = sum(global_counts.values())
    global_dist  = {cat: cnt / global_total for cat, cnt in global_counts.items()}

    categories = sorted(all_categories)

    model = MarkovModel(
        order=order,
        transitions=transitions,
        state_counts=state_counts,
        global_dist=global_dist,
        categories=categories,
        total_sequences=len(sequences),
        total_transitions=total_transitions,
    )

    print(f"[Markov] Built order-{order} model:")
    print(f"  Sequences:   {len(sequences):,}")
    print(f"  Transitions: {total_transitions:,}")
    print(f"  Categories:  {len(categories)}")
    print(f"  Coverage:    {model.coverage():.1%}")
    print(f"  Sparsity:    {model.sparsity():.1%}")

    return model


# ── Sequence extraction helpers ───────────────────────────────────────────────

def df_to_sequences(
    df: pd.DataFrame,
    customer_col: str  = "customer_id",
    category_col: str  = "category",
    timestamp_col: str = "timestamp",
) -> list[list[str]]:
    """Convert a long-format DataFrame to a list of per-customer category sequences.

    The DataFrame must be sorted by timestamp within each customer, or this
    function will sort it for you.

    Args:
        df:            Long-format DataFrame (one row per transaction).
        customer_col:  Column identifying the customer.
        category_col:  Column with the product category.
        timestamp_col: Column with the timestamp (used for sorting).

    Returns:
        List of sequences (list of category strings), one per customer.
        Sequences with fewer than 2 events are excluded.
    """
    # Ensure sorted order
    df = df.sort_values([customer_col, timestamp_col])

    sequences = []
    for _, grp in df.groupby(customer_col):
        cats = grp[category_col].dropna().tolist()
        if len(cats) >= 2:
            sequences.append(cats)

    print(f"[Markov] Extracted {len(sequences):,} sequences from DataFrame.")
    return sequences


# ── Order-2 adapter (uniform API for evaluator) ──────────────────────────────

class Order2Adapter:
    """Wraps an order-2 MarkovModel so it can be called with a single string state.

    The evaluator calls model.predict_categories(current_state, k=k) where
    current_state is a plain category string. For order-2 we need (prev, current).
    This adapter maintains a per-customer 'previous category' context that is
    populated externally before evaluation.

    Usage (in evaluator):
        adapter = Order2Adapter(markov2_model)
        adapter.set_context(prev_cat)            # set the previous category
        preds = adapter.predict_categories(current_cat, k=3)
    """

    def __init__(self, model: MarkovModel) -> None:
        assert model.order == 2, "Order2Adapter requires an order=2 MarkovModel."
        self._model   = model
        self._prev:   str | None = None

    def set_context(self, prev_category: str) -> None:
        self._prev = prev_category

    def predict_categories(self, current_state: str, k: int = 3) -> list[str]:
        if self._prev is not None:
            state: str | tuple = (self._prev, current_state)
        else:
            state = current_state   # fall back to order-1 key (uses global dist)
        return [cat for cat, _ in self._model.predict_top_k(state, k=k)]

    def predict_top_k(self, current_state: str, k: int = 3) -> list[tuple[str, float]]:
        if self._prev is not None:
            state: str | tuple = (self._prev, current_state)
        else:
            state = current_state
        return self._model.predict_top_k(state, k=k)

    @property
    def model(self) -> MarkovModel:
        return self._model


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    toy = [
        ["A", "B", "A", "C", "B"],
        ["A", "C", "B", "A", "C"],
        ["B", "A", "B", "C", "A"],
        ["C", "A", "B", "C", "B"],
    ]
    m1 = build_markov(toy, order=1)
    m2 = build_markov(toy, order=2)
    print("Order-1 predict from A:", m1.predict_top_k("A", k=2))
    print("Order-2 predict from (A,B):", m2.predict_top_k(("A", "B"), k=2))
    adapter = Order2Adapter(m2)
    adapter.set_context("A")
    print("Adapter predict (prev=A, cur=B):", adapter.predict_categories("B", k=2))
    print("Summaries:")
    print(" order=1:", json.dumps(m1.summary(), indent=2))
    print(" order=2:", json.dumps(m2.summary(), indent=2))
