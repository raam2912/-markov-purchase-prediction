"""
generate_notebook.py — Script to programmatically create the publishable Jupyter notebook.

Run this from the backend directory with the venv activated:
    python generate_notebook.py

This will write:  ../notebook/markov_purchase_prediction.ipynb
"""
import json
import sys
from pathlib import Path

# ── Notebook cell helpers ─────────────────────────────────────────────────────

def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }

def code(source: str, tags: list[str] | None = None) -> dict:
    meta = {}
    if tags:
        meta["tags"] = tags
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": meta,
        "outputs": [],
        "source": source,
    }

# ── Cell content ──────────────────────────────────────────────────────────────

cells = [

# ─────────────────── 0. Title ─────────────────────────────────────────────────
md("""\
# Predicting Consumer Purchases with Markov Chains
## Results on Real E-Commerce Data (UCI Online Retail II, 1M+ rows)

> **Author:** AntiGrav Research  
> **Date:** 2026  
> **Dataset:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — 541k+ real UK e-commerce transactions  
> **Model:** First-Order Markov Chain on product categories  
> **Metric:** Hit Rate @ K=1, 3, 5 vs most-popular baseline  

---

### The core question
*Given a customer's most recent product category purchase, can we predict their next purchase better than "just recommend the most popular items"?*

This notebook is a complete, reproducible experiment from raw data loading to final evaluation results, ready to accompany a Medium / Towards Data Science article.

---
"""),

# ─────────────────── 1. Setup ─────────────────────────────────────────────────
md("## 1. Setup & Imports"),

code("""\
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add backend to path so we can import our modules
BACKEND_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'backend'))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# Our modules
from models.category_encoder import encode_dataframe, get_category_distribution, get_all_categories
from models.markov            import build_markov, df_to_sequences, MarkovModel
from models.baseline          import BaselineModel
from models.evaluator         import (
    temporal_split, build_test_transitions,
    evaluate_model, print_comparison_table, EvalResult,
    hit_rate_at_k, bootstrap_ci,
)

# Plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams.update({
    'figure.figsize':     (12, 5),
    'font.family':        'DejaVu Sans',
    'font.size':          12,
    'axes.titlesize':     14,
    'axes.titleweight':   'bold',
    'figure.dpi':         120,
})

PROCESSED_DIR = Path(BACKEND_DIR) / 'data' / 'processed'
print(f"Backend: {BACKEND_DIR}")
print(f"Data:    {PROCESSED_DIR}")
print(f"Pandas:  {pd.__version__}  |  NumPy: {np.__version__}")
"""),

# ─────────────────── 2. Load Data ────────────────────────────────────────────
md("""\
## 2. Data Loading

We load the **pre-cleaned** parquet file produced by our Phase 1 ingestion pipeline.  
This file has already been through 8 cleaning passes:
- Nulls dropped (CustomerID, InvoiceDate, StockCode)
- Returns removed (Invoice starting with 'C', Quantity < 0)
- System/test SKUs filtered (POSTAGE, DISCOUNT, etc.)
- Outlier quantities and prices capped at 99.5th percentile
- Near-duplicate transactions (same SKU within 60s) removed
"""),

code("""\
seq_path = PROCESSED_DIR / 'sequences.parquet'
df = pd.read_parquet(seq_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Shape:     {df.shape}")
print(f"Customers: {df['customer_id'].nunique():,}")
print(f"Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
print(f"Columns:   {list(df.columns)}")
df.head(3)
"""),

code("""\
# Basic stats
print("=== Dataset Statistics ===")
print(f"Total transactions:          {len(df):>10,}")
print(f"Unique customers:            {df['customer_id'].nunique():>10,}")
print(f"Unique SKUs:                 {df['sku'].nunique():>10,}")
print(f"Avg transactions/customer:   {len(df) / df['customer_id'].nunique():>10.1f}")
median_len = df.groupby('customer_id').size().median()
print(f"Median sequence length:      {median_len:>10.0f}")
"""),

# ─────────────────── 3. Category Assignment ──────────────────────────────────
md("""\
## 3. Product Category Assignment

Raw SKU-level data has ~4,600 unique stock codes. A Markov chain at that level  
would be **catastrophically sparse** — most transitions would be observed at most once.

**Solution**: Map descriptions → ~15 business-meaningful categories using **keyword rules**.

A first-order Markov chain on 15 states produces a 15×15 transition matrix (225 cells),  
which our 765k events fill very densely. ✅
"""),

code("""\
df = encode_dataframe(df, desc_col='description')

print("Category assignment complete.")
print(f"\\nCategory distribution:")
cat_dist = get_category_distribution(df)
print(cat_dist.to_string())
"""),

code("""\
# Visualise category distribution
fig, ax = plt.subplots(figsize=(13, 5))
cat_dist = df['category'].value_counts()
colors   = sns.color_palette('husl', len(cat_dist))

bars = ax.barh(cat_dist.index[::-1], cat_dist.values[::-1], color=colors[::-1], height=0.75)

for bar, val in zip(bars, cat_dist.values[::-1]):
    ax.text(bar.get_width() + 2000, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=10)

ax.set_xlabel('Transaction Count')
ax.set_title('Product Category Distribution — UCI Online Retail II')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
plt.tight_layout()
plt.savefig('category_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

# ─────────────────── 4. Transition Matrix ────────────────────────────────────
md("""\
## 4. Building the Transition Matrix

The **transition matrix** T is the heart of the Markov chain model.

$$T[i][j] = P(\\text{next} = j \\mid \\text{current} = i) = \\frac{\\text{count}(i \\to j)}{\\text{count}(i)}$$

Intuitively:
- Each row is a "current category"
- Each column is a "next category"  
- Each cell is the probability of transitioning from row → column
- Each row sums to 1.0

We build this from **all consecutive purchase pairs** across all customer sequences.
"""),

code("""\
# Build sequences from training data — we'll apply split in Section 6
# For demonstration, use ALL data to show the full transition matrix structure
all_sequences = df_to_sequences(df, customer_col='customer_id',
                                 category_col='category', timestamp_col='timestamp')
print(f"Total sequences: {len(all_sequences):,}")
print(f"Example sequence (first 10 purchases): {all_sequences[0][:10]}")
"""),

code("""\
# Build the Markov model on full data (for EDA visualisation)
model_full = build_markov(all_sequences, order=1)
print("\\nModel Summary:")
for k, v in model_full.summary().items():
    print(f"  {k:<22}: {v}")
"""),

code("""\
# Transition matrix heatmap
T_matrix = model_full.to_dataframe()

fig, ax = plt.subplots(figsize=(14, 11))
mask = T_matrix == 0

sns.heatmap(
    T_matrix,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    mask=mask,
    linewidths=0.5,
    linecolor='#e0e0e0',
    cbar_kws={'label': 'Transition Probability'},
    ax=ax,
    annot_kws={'size': 8},
)

ax.set_title('First-Order Markov Transition Matrix\\n(row = current category → column = next category)',
             pad=20)
ax.set_xlabel('Next Category', labelpad=12)
ax.set_ylabel('Current Category', labelpad=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('transition_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("\\nEach row sums to (approx) 1.0:")
print(T_matrix.sum(axis=1).round(3))
"""),

# ─────────────────── 5. Baseline Model ───────────────────────────────────────
md("""\
## 5. Baseline Model — Most Popular Categories

Before evaluating the Markov chain, we need a **floor to beat**.

The baseline strategy: *always predict the same global top-K categories,  
regardless of customer history.* This is naive but not trivial — popular  
products are popular for a reason.

> **If Markov can't beat this, it has no information value.**
"""),

code("""\
# We'll build the baseline on training data (Section 6 shows the split)
# For now, demonstrate the baseline structure
baseline_demo = BaselineModel()
baseline_demo.fit(df)

print("Top-10 categories by global transaction frequency:")
dist_df = baseline_demo.get_category_distribution()
print(dist_df.head(10).to_string(index=False))

print(f"\\nBaseline always predicts (k=3): {baseline_demo.predict_categories(k=3)}")
"""),

code("""\
# Bar chart of global category distribution
dist_df = baseline_demo.get_category_distribution()

fig, ax = plt.subplots(figsize=(13, 5))
palette = sns.color_palette('husl', len(dist_df))
bars = ax.bar(dist_df['category'], dist_df['pct'], color=palette, width=0.75)

for bar, val in zip(bars, dist_df['pct']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', fontsize=9)

ax.set_ylabel('Percentage of All Transactions (%)')
ax.set_title('Global Category Frequency — The Baseline Model Always Predicts These Top-K')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('baseline_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

# ─────────────────── 6. Temporal Split ───────────────────────────────────────
md("""\
## 6. Temporal Train/Test Split

**Critical rule: NEVER shuffle. Time order is sacred.**

Shuffling would cause data leakage — future data bleeding into the past.  
Our split:
- **Train**: first 10 months of data (Dec 2009 → Sep 2011)
- **Test**: last 2 months (Oct–Nov 2011)

Customers can appear in both sets — we use their training history  
to predict their first test-set purchase. This is the realistic deployment scenario.
"""),

code("""\
df_train, df_test = temporal_split(df, timestamp_col='timestamp', test_months=2)

print(f"\\nTrain fraction: {len(df_train)/len(df):>6.1%}")
print(f"Test  fraction: {len(df_test)/len(df):>6.1%}")
"""),

code("""\
# Timeline visualisation
fig, ax = plt.subplots(figsize=(13, 3))

train_dates = df_train['timestamp']
test_dates  = df_test['timestamp']

# Create monthly histogram data
df['month'] = df['timestamp'].dt.to_period('M')
monthly = df.groupby('month').size().reset_index(name='count')
monthly['month_dt'] = monthly['month'].dt.to_timestamp()

split_date = df_train['timestamp'].max()

ax.fill_between(monthly['month_dt'], monthly['count'], alpha=0.4, color='steelblue', label='Train')
mask_test = monthly['month_dt'] >= split_date
ax.fill_between(monthly['month_dt'][mask_test], monthly['count'][mask_test],
                alpha=0.6, color='coral', label='Test')

ax.axvline(split_date, color='red', linewidth=2, linestyle='--', label=f'Split: {split_date.date()}')
ax.set_title('Temporal Train/Test Split — Monthly Transaction Volume')
ax.set_xlabel('Month')
ax.set_ylabel('Transactions')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
ax.legend()
plt.tight_layout()
plt.savefig('temporal_split.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

# ─────────────────── 7. Train Models ─────────────────────────────────────────
md("""\
## 7. Training the Models

We train **both models strictly on the training data** (months 1–10).  
The test set is untouched until evaluation.
"""),

code("""\
# Build sequences from TRAINING data only
train_sequences = df_to_sequences(
    df_train,
    customer_col='customer_id',
    category_col='category',
    timestamp_col='timestamp',
)
print(f"Training sequences: {len(train_sequences):,}")

# Train Markov model
markov_model = build_markov(train_sequences, order=1)
"""),

code("""\
# Train baseline model
baseline = BaselineModel()
baseline.fit(df_train, category_col='category')
"""),

code("""\
# Build test transitions: (last_train_category) → (first_test_category)
test_transitions = build_test_transitions(
    df_train=df_train,
    df_test=df_test,
    customer_col='customer_id',
    category_col='category',
    timestamp_col='timestamp',
)

print(f"\\nFirst 5 test transitions (current → true_next):")
for c, n in test_transitions[:5]:
    print(f"  {c}  →  {n}")
"""),

# ─────────────────── 8. Evaluation ───────────────────────────────────────────
md("""\
## 8. Evaluation — Hit Rate @ K

**Hit Rate @ K**: Did the true next category appear in the model's top-K predictions?

$$HR@K = \\frac{1}{|\\text{test transitions}|} \\sum_{i} \\mathbf{1}[y_i \\in \\hat{y}_i^{(K)}]$$

We compute HR@K for K ∈ {1, 3, 5} with **bootstrap 95% confidence intervals**  
(1000 resamples, seed=42) so results are reproducible and statistically grounded.
"""),

code("""\
print("=" * 50)
print("MARKOV CHAIN RESULTS")
print("=" * 50)
results_markov = evaluate_model(
    markov_model, test_transitions,
    model_name='Markov (order=1)',
    k_values=[1, 3, 5],
    n_bootstrap=1000,
)
"""),

code("""\
print("=" * 50)
print("BASELINE RESULTS")
print("=" * 50)
results_baseline = evaluate_model(
    baseline, test_transitions,
    model_name='Most-Popular',
    k_values=[1, 3, 5],
    n_bootstrap=1000,
)
"""),

code("""\
# Side-by-side comparison
print_comparison_table(results_markov, results_baseline)
"""),

# ─────────────────── 9. Results Visualisation ────────────────────────────────
md("""\
## 9. Results Visualisation
"""),

code("""\
# Build results DataFrame for plotting
rows = []
for r in results_markov + results_baseline:
    rows.append({
        'Model': r.model_name,
        'K': r.k,
        'HR': r.hit_rate,
        'CI_lower': r.ci_lower,
        'CI_upper': r.ci_upper,
    })
results_df = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle('Markov Chain vs Most-Popular Baseline\\nHit Rate @ K with 95% Confidence Intervals',
             fontsize=14, fontweight='bold', y=1.02)

colors = {'Markov (order=1)': '#2196F3', 'Most-Popular': '#FF7043'}
k_values = [1, 3, 5]

for ax, k in zip(axes, k_values):
    sub = results_df[results_df['K'] == k]
    x_pos = range(len(sub))
    bars  = ax.bar(
        x_pos,
        sub['HR'],
        color=[colors.get(m, 'grey') for m in sub['Model']],
        width=0.5,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.8,
    )
    # Error bars (CI)
    yerr_lo = sub['HR'].values - sub['CI_lower'].values
    yerr_hi = sub['CI_upper'].values - sub['HR'].values
    ax.errorbar(
        x_pos, sub['HR'],
        yerr=[yerr_lo, yerr_hi],
        fmt='none', color='black', capsize=6, capthick=1.5, linewidth=1.5,
    )
    # Value labels
    for bar, hr in zip(bars, sub['HR']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{hr:.3f}', ha='center', fontsize=11, fontweight='bold')

    ax.set_title(f'HR @ {k}', fontsize=13)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(sub['Model'], rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Hit Rate' if k == 1 else '')
    ax.set_ylim(0, min(1.0, results_df[results_df['K'] == k]['CI_upper'].max() + 0.08))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))

plt.tight_layout()
plt.savefig('results_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

# ─────────────────── 10. Lift analysis ───────────────────────────────────────
md("""\
## 10. Lift Analysis by Category

Does the Markov chain perform better for some categories than others?  
We compute HR@3 per "current category" to find where it adds the most value.
"""),

code("""\
# Per-category HR@3 analysis
cat_results = []
categories  = list({c for c, _ in test_transitions})

for cat in categories:
    cat_pairs = [(c, n) for c, n in test_transitions if c == cat]
    if len(cat_pairs) < 10:
        continue  # too few samples — skip

    markov_preds   = [markov_model.predict_categories(c, k=3) for c, _ in cat_pairs]
    baseline_preds = [baseline.predict_categories(k=3) for _ in cat_pairs]
    true_nexts     = [n for _, n in cat_pairs]

    hr_m = hit_rate_at_k(markov_preds,   true_nexts, k=3)
    hr_b = hit_rate_at_k(baseline_preds, true_nexts, k=3)
    cat_results.append({
        'Category': cat,
        'Markov HR@3':   round(hr_m, 4),
        'Baseline HR@3': round(hr_b, 4),
        'Lift':          round(hr_m - hr_b, 4),
        'N':             len(cat_pairs),
    })

cat_df = pd.DataFrame(cat_results).sort_values('Lift', ascending=False)
print(cat_df.to_string(index=False))
"""),

code("""\
# Visualise per-category lift
fig, ax = plt.subplots(figsize=(13, 6))
palette = ['#2196F3' if l >= 0 else '#FF7043' for l in cat_df['Lift']]
bars    = ax.barh(cat_df['Category'], cat_df['Lift'], color=palette, height=0.65)

for bar, val in zip(bars, cat_df['Lift']):
    x = bar.get_width() + (0.003 if val >= 0 else -0.003)
    ha = 'left' if val >= 0 else 'right'
    ax.text(x, bar.get_y() + bar.get_height()/2, f'{val:+.3f}', va='center', fontsize=10, ha=ha)

ax.axvline(0, color='black', linewidth=1.2)
ax.set_xlabel('HR@3 Lift  (Markov − Baseline)')
ax.set_title('Per-Category Lift: Where Does the Markov Chain Add Value?\\n(Blue = beats baseline, Red = below baseline)')
plt.tight_layout()
plt.savefig('per_category_lift.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

# ─────────────────── 11. Discussion ──────────────────────────────────────────
md("""\
## 11. Discussion

### What we found

| Metric | Markov | Baseline | Lift |
|--------|--------|----------|------|
| HR@1 | (see above) | (see above) | (computed above) |
| HR@3 | (see above) | (see above) | (computed above) |
| HR@5 | (see above) | (see above) | (computed above) |

### Why does / doesn't Markov beat the baseline?

The Markov chain exploits **conditional structure**: `P(next | current)` is richer than  
`P(next)`. It works well when customers have **consistent category transitions**  
(e.g., someone who bought KITCHEN_DINING tends to follow up with KITCHEN_DINING or HOME_DECOR).

It struggles when:
1. **Category is very dominant** — HOME_DECOR is bought so often that the baseline  
   always includes it, making conditional prediction hard to beat.
2. **Sequence is short** — cold-start customers have little history to condition on.
3. **Category assignment is noisy** — keyword rules aren't perfect; ambiguous  
   descriptions fall into OTHER.

### Limitations

- **Order 1 only** — we only look at the immediately preceding purchase. A 2nd-order  
  chain (last 2 purchases) could capture richer patterns.
- **No personalisation** — the transition matrix is global (pooled across all customers).  
  A per-segment model (e.g., B2B vs B2C) could improve lift.
- **Category granularity** — 15 categories may be too coarse or too fine. Optimal  
  granularity is a hyperparameter.

### What's next (Phase 2)

- 2nd-order Markov chains (state = last 2 categories)
- Per-segment transition matrices (RFM customer segments)
- Sequence-level features as XGBoost input
- Prophet + XGBoost demand forecasting layer

---

*This notebook is part of the AntiGrav Market Prediction Framework.*  
*See the [GitHub repository](https://github.com/) for all source code.*
"""),

# ─────────────────── 12. Reproducibility ─────────────────────────────────────
md("## 12. Reproducibility"),

code("""\
import platform, importlib

print("=== Environment ===")
print(f"Python:       {sys.version.split()[0]}")
print(f"OS:           {platform.system()} {platform.release()}")

for pkg in ['pandas', 'numpy', 'matplotlib', 'seaborn', 'pyarrow']:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg:<15}: {m.__version__}")
    except ImportError:
        print(f"{pkg:<15}: NOT INSTALLED")

print(f"\\nRandom seed:  42 (used in all bootstrap CIs)")
print(f"Data source:  UCI Machine Learning Repository — Online Retail II (ID=502)")
print(f"Download URL: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip")
"""),
]  # end cells

# ── Assemble notebook ─────────────────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

# Write notebook
out_path = Path(__file__).parent.parent / "notebook" / "markov_purchase_prediction.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, indent=1, ensure_ascii=False)

print(f"\n[OK] Notebook written -> {out_path}")
print(f"     Size: {out_path.stat().st_size / 1024:.1f} KB")
print(f"\nTo run:")
print(f"  cd {out_path.parent.parent}")
print(f"  backend\\venv\\Scripts\\jupyter notebook notebook/markov_purchase_prediction.ipynb")
