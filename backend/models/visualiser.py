"""
models/visualiser.py — All publication-quality charts for the Markov purchase prediction project.

Generates 6 charts and saves them to backend/data/charts/:
  1. category_distribution.png   — bar chart of all 15 category frequencies
  2. transition_matrix.png       — 15x15 heatmap of P(next | current)
  3. hr_comparison.png           — grouped bar: Markov vs Baseline @ K=1,3,5
  4. per_category_lift.png       — horizontal bar: HR@3 lift per current category
  5. sequence_lengths.png        — histogram of customer sequence lengths
  6. temporal_split.png          — timeline showing train / test volumes

Run standalone:
    python models/visualiser.py
Or import and call generate_all_charts(df, model, baseline, results_m, results_b).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")          # headless — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE_MAIN   = "#2563EB"   # deep blue  — Markov
PALETTE_BASE   = "#F97316"   # amber      — Baseline
PALETTE_POS    = "#16A34A"   # green      — positive lift
PALETTE_NEG    = "#DC2626"   # red        — negative lift
BG_COLOR       = "#0F172A"   # dark navy
GRID_COLOR     = "#1E293B"
TEXT_COLOR      = "#F1F5F9"
SUBTEXT_COLOR   = "#94A3B8"
ACCENT_COLORS  = [
    "#2563EB","#7C3AED","#DB2777","#EA580C","#CA8A04",
    "#16A34A","#0891B2","#4F46E5","#BE123C","#15803D",
    "#0369A1","#9333EA","#C2410C","#0D9488","#6B7280",
]

def _dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    BG_COLOR,
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "axes.grid":         True,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.8,
        "text.color":        TEXT_COLOR,
        "xtick.color":       SUBTEXT_COLOR,
        "ytick.color":       SUBTEXT_COLOR,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.facecolor":  GRID_COLOR,
        "legend.edgecolor":  GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "figure.dpi":        140,
        "savefig.dpi":       180,
        "savefig.facecolor": BG_COLOR,
        "savefig.bbox":      "tight",
        "font.family":       "DejaVu Sans",
    })


CHARTS_DIR = Path(__file__).parent.parent / "data" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Chart 1: Category Distribution ───────────────────────────────────────────

def chart_category_distribution(df: pd.DataFrame, category_col: str = "category") -> Path:
    """Horizontal bar chart of transaction count per category."""
    _dark_style()
    cat_dist = df[category_col].value_counts()
    total    = len(df)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Product Category Distribution", fontsize=16, fontweight="bold",
                 color=TEXT_COLOR, y=0.98)

    y_pos   = range(len(cat_dist))
    colors  = ACCENT_COLORS[:len(cat_dist)]
    bars    = ax.barh(list(y_pos), cat_dist.values, color=colors[::-1], height=0.72,
                      edgecolor="none")

    for bar, val in zip(bars, cat_dist.values):
        pct = val / total * 100
        ax.text(bar.get_width() + total * 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)",
                va="center", fontsize=8.5, color=SUBTEXT_COLOR)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(cat_dist.index[::-1] if False else cat_dist.index,
                       fontsize=9.5, color=TEXT_COLOR)
    ax.invert_yaxis()
    ax.set_xlabel("Transaction Count", color=TEXT_COLOR)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.set_xlim(0, cat_dist.max() * 1.22)
    ax.set_title("15 categories assigned via keyword rules  |  UCI Online Retail II  |  765k transactions",
                 fontsize=10, color=SUBTEXT_COLOR, pad=6)

    plt.tight_layout()
    out = CHARTS_DIR / "category_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Chart 1] Saved -> {out.name}")
    return out


# ── Chart 2: Transition Matrix Heatmap ───────────────────────────────────────

def chart_transition_matrix(model: Any) -> Path:
    """Annotated heatmap of the Markov transition probability matrix."""
    _dark_style()
    T = model.to_dataframe()

    # Shorten labels for readability
    short = {
        "BAGS_STORAGE": "BAGS", "CANDLES_LIGHTS": "CANDLES", "CHILDREN_TOYS": "CHILDREN",
        "CHRISTMAS": "XMAS", "CLOCKS_TIMEPIECE": "CLOCKS", "FRAMES_MIRRORS": "FRAMES",
        "GARDEN_OUTDOOR": "GARDEN", "HOME_DECOR": "HOME", "JEWELLERY_ACCESS": "JEWELLERY",
        "KITCHEN_DINING": "KITCHEN", "OTHER": "OTHER", "SEASONAL_GIFT": "SEASONAL",
        "SIGNS_PLAQUES": "SIGNS", "STATIONERY": "STATIONERY", "TEXTILES_SOFT": "TEXTILES",
    }
    T.index   = [short.get(c, c) for c in T.index]
    T.columns = [short.get(c, c) for c in T.columns]

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.suptitle("First-Order Markov Transition Matrix\nP(next category | current category)",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, y=0.99)

    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        T, annot=True, fmt=".2f", cmap=cmap,
        linewidths=0.4, linecolor=GRID_COLOR,
        cbar_kws={"label": "Transition Probability", "shrink": 0.8},
        ax=ax, annot_kws={"size": 7.5, "color": TEXT_COLOR},
        vmin=0, vmax=T.values.max(),
    )
    ax.set_xlabel("Next Category", fontsize=11, color=TEXT_COLOR, labelpad=10)
    ax.set_ylabel("Current Category", fontsize=11, color=TEXT_COLOR, labelpad=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    plt.xticks(rotation=45, ha="right", color=TEXT_COLOR)
    plt.yticks(rotation=0, color=TEXT_COLOR)

    # Colorbar text
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    ax.set_title(f"Trained on 634,540 transitions  |  Coverage: 100%  |  Sparsity: 0%",
                 fontsize=9, color=SUBTEXT_COLOR, pad=8)

    plt.tight_layout()
    out = CHARTS_DIR / "transition_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Chart 2] Saved -> {out.name}")
    return out


# ── Chart 3: HR@K Comparison ─────────────────────────────────────────────────

def chart_hr_comparison(
    results_markov:   list[Any],
    results_baseline: list[Any],
    results_markov2:  list[Any] | None = None,
) -> Path:
    """Grouped bar chart: Markov (order=1) vs Baseline vs Markov (order=2) at K=1,3,5."""
    _dark_style()
    k_vals = [r.k for r in results_markov]

    hr_m  = [r.hit_rate  for r in results_markov]
    ci_lo = [r.hit_rate - r.ci_lower for r in results_markov]
    ci_hi = [r.ci_upper  - r.hit_rate for r in results_markov]
    hr_b  = [r.hit_rate  for r in results_baseline]
    b_lo  = [r.hit_rate - r.ci_lower for r in results_baseline]
    b_hi  = [r.ci_upper  - r.hit_rate for r in results_baseline]

    n_groups = len(k_vals)
    has_m2   = results_markov2 is not None
    n_bars   = 3 if has_m2 else 2
    width    = 0.25
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Hit Rate @ K  —  Markov Chain vs Most-Popular Baseline",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.0)

    offsets = [-width, 0] if not has_m2 else [-width, 0, width]

    # Markov order=1
    b1 = ax.bar(x + offsets[0], hr_m, width, label="Markov (order=1)",
                color=PALETTE_MAIN, alpha=0.9, zorder=3, edgecolor="none")
    ax.errorbar(x + offsets[0], hr_m, yerr=[ci_lo, ci_hi],
                fmt="none", color="white", capsize=5, capthick=1.5, linewidth=1.5, zorder=4)

    # Baseline
    b2 = ax.bar(x + offsets[1], hr_b, width, label="Most-Popular Baseline",
                color=PALETTE_BASE, alpha=0.9, zorder=3, edgecolor="none")
    ax.errorbar(x + offsets[1], hr_b, yerr=[b_lo, b_hi],
                fmt="none", color="white", capsize=5, capthick=1.5, linewidth=1.5, zorder=4)

    # Markov order=2 (optional)
    if has_m2:
        hr_m2  = [r.hit_rate for r in results_markov2]
        m2_lo  = [r.hit_rate - r.ci_lower for r in results_markov2]
        m2_hi  = [r.ci_upper - r.hit_rate for r in results_markov2]
        b3 = ax.bar(x + offsets[2], hr_m2, width, label="Markov (order=2)",
                    color="#7C3AED", alpha=0.9, zorder=3, edgecolor="none")
        ax.errorbar(x + offsets[2], hr_m2, yerr=[m2_lo, m2_hi],
                    fmt="none", color="white", capsize=5, capthick=1.5, linewidth=1.5, zorder=4)
        for bar, val in zip(b3, hr_m2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold", color=TEXT_COLOR)

    # Value labels
    for bar, val in zip(b1, hr_m):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold", color=TEXT_COLOR)
    for bar, val in zip(b2, hr_b):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold", color=TEXT_COLOR)

    # Lift annotations
    for i, (m, b) in enumerate(zip(hr_m, hr_b)):
        lift  = m - b
        color = PALETTE_POS if lift >= 0 else PALETTE_NEG
        ax.annotate(f"{lift:+.1%} lift",
                    xy=(x[i], max(m, b) + 0.04),
                    ha="center", fontsize=8.5, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"HR @ {k}" for k in k_vals], fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel("Hit Rate", color=TEXT_COLOR)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, min(1.0, max(hr_m + hr_b) + 0.15))
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title("Error bars = 95% bootstrap CI  (1000 samples, seed=42)  |  2,115 test transitions",
                 fontsize=9, color=SUBTEXT_COLOR, pad=6)

    plt.tight_layout()
    out = CHARTS_DIR / "hr_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Chart 3] Saved -> {out.name}")
    return out


# ── Chart 4: Per-Category Lift ────────────────────────────────────────────────

def chart_per_category_lift(
    test_transitions: list[tuple[str, str]],
    markov_model:     Any,
    baseline_model:   Any,
    k: int = 3,
    min_samples: int = 20,
) -> Path:
    """Horizontal bar: HR@K lift (Markov - Baseline) per current category."""
    from models.evaluator import hit_rate_at_k

    _dark_style()
    cats = sorted({c for c, _ in test_transitions})
    rows = []

    for cat in cats:
        pairs = [(c, n) for c, n in test_transitions if c == cat]
        if len(pairs) < min_samples:
            continue
        true_nexts = [n for _, n in pairs]
        m_preds = [markov_model.predict_categories(c, k=k) for c, _ in pairs]
        b_preds = [baseline_model.predict_categories(c, k=k) for c, _ in pairs]
        hr_m = hit_rate_at_k(m_preds, true_nexts, k=k)
        hr_b = hit_rate_at_k(b_preds, true_nexts, k=k)
        rows.append({
            "category": cat,
            "markov":   hr_m,
            "baseline": hr_b,
            "lift":     hr_m - hr_b,
            "n":        len(pairs),
        })

    df_lift = pd.DataFrame(rows).sort_values("lift", ascending=True)

    fig, ax = plt.subplots(figsize=(13, max(6, len(df_lift) * 0.55)))
    fig.suptitle(f"Per-Category HR@{k} Lift  (Markov − Baseline)",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.0)

    colors = [PALETTE_POS if v >= 0 else PALETTE_NEG for v in df_lift["lift"]]
    bars   = ax.barh(df_lift["category"], df_lift["lift"],
                     color=colors, height=0.65, edgecolor="none", zorder=3)

    for bar, row in zip(bars, df_lift.itertuples()):
        x_pos = bar.get_width() + (0.003 if row.lift >= 0 else -0.003)
        ha    = "left" if row.lift >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{row.lift:+.3f}  (n={row.n})",
                va="center", fontsize=8.5, color=SUBTEXT_COLOR, ha=ha)

    ax.axvline(0, color=TEXT_COLOR, linewidth=1.2, zorder=2)
    ax.set_xlabel(f"HR@{k} Lift  (Markov − Baseline)", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    pos_patch = mpatches.Patch(color=PALETTE_POS, label="Markov beats baseline")
    neg_patch = mpatches.Patch(color=PALETTE_NEG, label="Baseline beats Markov")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc="lower right")
    ax.set_title(f"Categories with < {min_samples} test transitions excluded",
                 fontsize=9, color=SUBTEXT_COLOR, pad=6)

    plt.tight_layout()
    out = CHARTS_DIR / "per_category_lift.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Chart 4] Saved -> {out.name}")
    return out


# ── Chart 5: Sequence Length Distribution ─────────────────────────────────────

def chart_sequence_lengths(df: pd.DataFrame,
                            customer_col: str = "customer_id") -> Path:
    """Histogram of purchases-per-customer with annotated statistics."""
    _dark_style()
    seq_lens = df.groupby(customer_col).size()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Customer Sequence Length Distribution",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR)

    ax.hist(seq_lens.clip(upper=500), bins=60, color=PALETTE_MAIN, alpha=0.85,
            edgecolor=BG_COLOR, linewidth=0.4, zorder=3)

    med  = seq_lens.median()
    mean = seq_lens.mean()
    ax.axvline(med,  color="#FBBF24", linewidth=1.8, linestyle="--", label=f"Median: {med:.0f}")
    ax.axvline(mean, color="#F87171", linewidth=1.8, linestyle=":",  label=f"Mean:   {mean:.0f}")

    ax.set_xlabel("Transactions per Customer  (capped at 500 for display)", color=TEXT_COLOR)
    ax.set_ylabel("Number of Customers", color=TEXT_COLOR)
    ax.legend(fontsize=10)

    stats_text = (f"Customers: {len(seq_lens):,}\n"
                  f"Min: {seq_lens.min()}    Max: {seq_lens.max():,}\n"
                  f"Std: {seq_lens.std():.0f}")
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            va="top", ha="right", fontsize=9, color=SUBTEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=GRID_COLOR, alpha=0.8))

    plt.tight_layout()
    out = CHARTS_DIR / "sequence_lengths.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Chart 5] Saved -> {out.name}")
    return out


# ── Chart 6: Temporal Split Timeline ─────────────────────────────────────────

def chart_temporal_split(df: pd.DataFrame,
                          timestamp_col: str = "timestamp",
                          test_months: int   = 2) -> Path:
    """Area chart showing monthly transaction volume with train/test split."""
    _dark_style()
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    monthly = df.groupby(df[timestamp_col].dt.to_period("M")).size().reset_index()
    monthly.columns = ["month", "count"]
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()

    max_date   = df[timestamp_col].max()
    split_date = (max_date - pd.DateOffset(months=test_months)).replace(day=1)

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle("Temporal Train / Test Split  —  Monthly Transaction Volume",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR)

    train_mask = monthly["month_dt"] < split_date
    test_mask  = monthly["month_dt"] >= split_date

    ax.fill_between(monthly["month_dt"],  monthly["count"],
                    where=train_mask.values, alpha=0.55, color=PALETTE_MAIN,
                    label=f"Train  ({train_mask.sum()} months)")
    ax.fill_between(monthly["month_dt"], monthly["count"],
                    where=test_mask.values,  alpha=0.75, color=PALETTE_BASE,
                    label=f"Test  ({test_mask.sum()} months)")
    ax.plot(monthly["month_dt"], monthly["count"],
            color=TEXT_COLOR, linewidth=1.0, alpha=0.6)

    ax.axvline(split_date, color="#FBBF24", linewidth=2, linestyle="--",
               label=f"Split: {split_date.strftime('%b %Y')}")
    ax.set_ylabel("Transactions / Month", color=TEXT_COLOR)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.set_xlim(monthly["month_dt"].min(), monthly["month_dt"].max())
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title("No shuffle — time order preserved to prevent data leakage",
                 fontsize=9, color=SUBTEXT_COLOR, pad=6)

    plt.tight_layout()
    out = CHARTS_DIR / "temporal_split.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Chart 6] Saved -> {out.name}")
    return out


# ── Chart 7: Results Summary Card ────────────────────────────────────────────

def chart_results_summary(
    results_markov:   list[Any],
    results_baseline: list[Any],
    results_markov2:  list[Any] | None = None,
) -> Path:
    """Clean results table rendered as a publication-ready figure."""
    _dark_style()

    m1 = {r.k: r for r in results_markov}
    b  = {r.k: r for r in results_baseline}
    m2 = {r.k: r for r in results_markov2} if results_markov2 else {}

    k_list    = sorted(m1.keys())
    has_order2 = bool(m2)

    col_headers = ["K", "Markov HR@K", "95% CI", "Baseline HR@K", "Lift (pp)"]
    if has_order2:
        col_headers.insert(3, "Markov (ord=2)")

    rows = []
    for k in k_list:
        mr  = m1[k]
        br  = b[k]
        lift = mr.hit_rate - br.hit_rate
        row = [
            f"@{k}",
            f"{mr.hit_rate:.4f}",
            f"[{mr.ci_lower:.4f}, {mr.ci_upper:.4f}]",
        ]
        if has_order2:
            row.append(f"{m2[k].hit_rate:.4f}" if k in m2 else "—")
        row += [f"{br.hit_rate:.4f}", f"{lift:+.4f}"]
        rows.append(row)

    n_cols = len(col_headers)
    n_rows = len(rows) + 1  # +1 for header

    fig, ax = plt.subplots(figsize=(13, 1.2 + n_rows * 0.7))
    ax.axis("off")
    fig.suptitle("Evaluation Results — Markov Chain vs Most-Popular Baseline",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)

    table_data = [col_headers] + rows
    col_widths = [0.06, 0.14, 0.26, 0.14, 0.14, 0.12] if not has_order2 else \
                 [0.06, 0.13, 0.24, 0.13, 0.12, 0.12, 0.10]

    tbl = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=col_widths[:n_cols],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_facecolor(BG_COLOR)
        cell.set_edgecolor(GRID_COLOR)
        cell.set_text_props(color=TEXT_COLOR)

        if row_idx == 0:
            cell.set_facecolor(GRID_COLOR)
            cell.set_text_props(fontweight="bold", color=TEXT_COLOR)

        if row_idx > 0 and col_idx == n_cols - 1:
            val_text = table_data[row_idx][col_idx]
            try:
                val = float(val_text)
                cell.set_text_props(
                    color=PALETTE_POS if val >= 0 else PALETTE_NEG,
                    fontweight="bold"
                )
            except ValueError:
                pass

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = CHARTS_DIR / "results_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[Chart 7] Saved -> {out.name}")
    return out


# ── Master runner ─────────────────────────────────────────────────────────────

def generate_all_charts(
    df:               pd.DataFrame,
    markov_model:     Any,
    baseline_model:   Any,
    results_markov:   list[Any],
    results_baseline: list[Any],
    test_transitions: list[tuple[str, str]],
    results_markov2:  list[Any] | None = None,
) -> list[Path]:
    """Generate all 7 charts and return their paths."""
    paths = []
    paths.append(chart_category_distribution(df))
    paths.append(chart_transition_matrix(markov_model))
    paths.append(chart_hr_comparison(results_markov, results_baseline, results_markov2))
    paths.append(chart_per_category_lift(test_transitions, markov_model, baseline_model))
    paths.append(chart_sequence_lengths(df))
    paths.append(chart_temporal_split(df))
    paths.append(chart_results_summary(results_markov, results_baseline, results_markov2))
    print(f"\n[Visualiser] All {len(paths)} charts saved to: {CHARTS_DIR}")
    return paths
