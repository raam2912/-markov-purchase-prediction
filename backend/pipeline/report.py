"""
pipeline/report.py — Generate quality reports (HTML + JSON) for the pipeline run.

Combines outputs from all 4 stages into one comprehensive quality document:
  - Raw data stats (from ValidationReport)
  - Cleaning funnel (from CleaningReport)
  - Sequence stats (from SequenceReport)
  - Overall pipeline health score
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from config import REPORTS_DIR
from pipeline.validate import ValidationReport
from pipeline.clean import CleaningReport
from pipeline.sequence_builder import SequenceReport


# ── JSON report ───────────────────────────────────────────────────────────────

def _build_json_report(
    validation: ValidationReport,
    cleaning: CleaningReport,
    sequences: SequenceReport,
) -> dict:
    """Assemble all metrics into a machine-readable dict."""
    # health_score: 0–100 based on key data quality metrics
    health_factors = {
        "low_null_customer_pct": max(0, 100 - validation.null_customer_pct * 2),
        "low_duplicate_pct":     max(0, 100 - validation.exact_duplicate_pct * 5),
        "retention_rate":        max(0, 100 - cleaning.total_removed_pct),
        "customer_coverage":     min(100, sequences.total_customers_kept / max(1, validation.unique_customers) * 100),
        "sequence_richness":     min(100, sequences.avg_sequence_length / 2),
    }
    health_score = round(sum(health_factors.values()) / len(health_factors), 1)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "health_score": health_score,
        "health_factors": {k: round(v, 2) for k, v in health_factors.items()},
        "validation": {
            "total_rows_raw": validation.total_rows,
            "null_customer_pct": validation.null_customer_pct,
            "cancellation_pct": validation.cancellation_pct,
            "negative_quantity_pct": validation.negative_quantity_pct,
            "zero_price_pct": validation.zero_price_pct,
            "exact_duplicate_pct": validation.exact_duplicate_pct,
            "unique_customers": validation.unique_customers,
            "unique_skus": validation.unique_skus,
            "unique_countries": validation.unique_countries,
            "date_range": f"{validation.date_min} → {validation.date_max}",
            "distribution": validation.distribution_summary,
            "warnings": validation.warnings,
            "errors": validation.errors,
        },
        "cleaning": {
            "rows_raw": cleaning.rows_raw,
            "rows_final": cleaning.rows_final,
            "total_removed": cleaning.total_removed,
            "total_removed_pct": cleaning.total_removed_pct,
            "passes": [
                {
                    "id": p.pass_id,
                    "description": p.description,
                    "rows_before": p.rows_before,
                    "rows_removed": p.rows_removed,
                    "rows_after": p.rows_after,
                    "removal_pct": p.removal_pct,
                    "details": p.details,
                }
                for p in cleaning.passes
            ],
            "outlier_caps": cleaning.outlier_cap_summary,
        },
        "sequences": {
            "total_customers_raw": sequences.total_customers_raw,
            "total_customers_kept": sequences.total_customers_kept,
            "customers_excluded_short": sequences.customers_excluded_short,
            "total_events": sequences.total_events,
            "avg_sequence_length": sequences.avg_sequence_length,
            "median_sequence_length": sequences.median_sequence_length,
            "avg_unique_skus_per_customer": sequences.avg_unique_skus_per_customer,
            "avg_sessions_per_customer": sequences.avg_sessions_per_customer,
            "total_unique_skus": sequences.total_unique_skus,
            "date_range": f"{sequences.date_range_min} → {sequences.date_range_max}",
        },
    }


# ── HTML report ───────────────────────────────────────────────────────────────

def _build_html_report(data: dict) -> str:
    """Generate a self-contained HTML quality dashboard from the JSON report data."""
    score = data["health_score"]
    score_color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"

    passes_rows = "".join(
        f"""
        <tr>
            <td><code>{p['id']}</code></td>
            <td>{p['description']}</td>
            <td>{p['rows_before']:,}</td>
            <td style="color:#ef4444">-{p['rows_removed']:,}</td>
            <td>{p['rows_after']:,}</td>
            <td>{p['removal_pct']:.2f}%</td>
        </tr>"""
        for p in data["cleaning"]["passes"]
    )

    health_bars = "".join(
        f"""
        <div class="factor">
            <span class="factor-name">{k.replace('_', ' ').title()}</span>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{min(100,v):.0f}%;
                     background:{'#22c55e' if v>=80 else '#f59e0b' if v>=60 else '#ef4444'}">
                </div>
            </div>
            <span class="factor-val">{v:.1f}</span>
        </div>"""
        for k, v in data["health_factors"].items()
    )

    val = data["validation"]
    seq = data["sequences"]
    cln = data["cleaning"]

    dist_rows = "".join(
        f"""<tr><td>{col}</td>
            <td>{stats['min']}</td><td>{stats['max']}</td>
            <td>{stats['mean']}</td><td>{stats['p95']}</td><td>{stats['p99_5']}</td>
        </tr>"""
        for col, stats in val.get("distribution", {}).items()
    )

    warnings_html = "".join(f"<li class='warn'>⚠ {w}</li>" for w in val.get("warnings", []))
    errors_html   = "".join(f"<li class='error'>✗ {e}</li>" for e in val.get("errors", []))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AntiGrav — Data Quality Report</title>
<style>
  :root {{
    --bg: #0f0f14; --surface: #1a1a24; --surface2: #22222f;
    --border: #2a2a3a; --text: #e2e2f0; --muted: #8888aa;
    --green: #22c55e; --amber: #f59e0b; --red: #ef4444; --blue: #60a5fa;
    --accent: #7c3aed;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
          font-size: 14px; line-height: 1.6; }}
  .header {{ background: linear-gradient(135deg, #1e1b4b 0%, #0f0f14 100%);
             padding: 32px 48px; border-bottom: 1px solid var(--border); }}
  .header h1 {{ font-size: 1.8rem; color: white; font-weight: 700; }}
  .header .sub {{ color: var(--muted); margin-top: 4px; font-size: 0.85rem; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
  .score-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
                 padding: 32px; margin-bottom: 32px; display: flex; gap: 48px; align-items: center; }}
  .score-circle {{ width: 120px; height: 120px; border-radius: 50%;
                   border: 6px solid {score_color}; display: flex; flex-direction: column;
                   align-items: center; justify-content: center; flex-shrink: 0; }}
  .score-num {{ font-size: 2.4rem; font-weight: 800; color: {score_color}; }}
  .score-label {{ font-size: 0.75rem; color: var(--muted); }}
  .health-factors {{ flex: 1; }}
  .health-factors h3 {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase;
                        letter-spacing: .08em; margin-bottom: 12px; }}
  .factor {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
  .factor-name {{ width: 180px; font-size: 0.8rem; color: var(--muted); flex-shrink: 0; }}
  .bar-bg {{ flex: 1; height: 8px; background: var(--surface2); border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .factor-val {{ width: 40px; text-align: right; font-size: 0.8rem; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
                padding: 20px; }}
  .stat-card .label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase;
                       letter-spacing: .06em; margin-bottom: 6px; }}
  .stat-card .value {{ font-size: 1.6rem; font-weight: 700; color: white; }}
  .stat-card .sub {{ font-size: 0.75rem; color: var(--muted); margin-top: 2px; }}
  section {{ margin-bottom: 36px; }}
  section h2 {{ font-size: 1rem; font-weight: 600; color: var(--text); margin-bottom: 16px;
                padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ text-align: left; padding: 10px 12px; background: var(--surface2); color: var(--muted);
        font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .05em; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}
  code {{ background: var(--surface2); padding: 2px 6px; border-radius: 4px; font-family: monospace;
          font-size: 0.8rem; color: var(--blue); }}
  ul.issues {{ list-style: none; padding: 0; }}
  li.warn {{ color: var(--amber); padding: 4px 0; }}
  li.error {{ color: var(--red); padding: 4px 0; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem;
            font-weight: 600; }}
  .badge-green {{ background: rgba(34,197,94,.15); color: var(--green); }}
  .badge-amber {{ background: rgba(245,158,11,.15); color: var(--amber); }}
  .tag-gen {{ color: var(--muted); font-size: 0.75rem; margin-top: 8px; }}
</style>
</head>
<body>
<div class="header">
  <h1>🧠 AntiGrav — Data Quality Report</h1>
  <div class="sub">UCI Online Retail II · Generated {data['generated_at']}</div>
</div>
<div class="container">

  <!-- Health Score -->
  <div class="score-card">
    <div class="score-circle">
      <span class="score-num">{score:.0f}</span>
      <span class="score-label">/ 100</span>
    </div>
    <div class="health-factors">
      <h3>Health Factors</h3>
      {health_bars}
    </div>
  </div>

  <!-- Key Stats -->
  <div class="grid">
    <div class="stat-card">
      <div class="label">Raw Rows</div>
      <div class="value">{val['total_rows_raw']:,}</div>
      <div class="sub">both sheets combined</div>
    </div>
    <div class="stat-card">
      <div class="label">Clean Rows</div>
      <div class="value">{cln['rows_final']:,}</div>
      <div class="sub">{100-cln['total_removed_pct']:.1f}% retained</div>
    </div>
    <div class="stat-card">
      <div class="label">Customers</div>
      <div class="value">{seq['total_customers_kept']:,}</div>
      <div class="sub">with ≥3 transactions</div>
    </div>
    <div class="stat-card">
      <div class="label">Unique SKUs</div>
      <div class="value">{seq['total_unique_skus']:,}</div>
      <div class="sub">in clean sequences</div>
    </div>
    <div class="stat-card">
      <div class="label">Avg Seq Length</div>
      <div class="value">{seq['avg_sequence_length']:.0f}</div>
      <div class="sub">median {seq['median_sequence_length']:.0f}</div>
    </div>
    <div class="stat-card">
      <div class="label">Countries</div>
      <div class="value">{val['unique_countries']}</div>
      <div class="sub">date range: {seq['date_range']}</div>
    </div>
  </div>

  <!-- Raw Data Validation -->
  <section>
    <h2>Raw Data Validation</h2>
    <table>
      <thead><tr><th>Check</th><th>Value</th><th>Status</th></tr></thead>
      <tbody>
        <tr><td>Null CustomerID</td><td>{val['null_customer_pct']:.2f}%</td>
            <td><span class="badge {'badge-green' if val['null_customer_pct']<30 else 'badge-amber'}">
            {'OK' if val['null_customer_pct']<30 else 'HIGH'}</span></td></tr>
        <tr><td>Cancellations</td><td>{val['cancellation_pct']:.2f}%</td>
            <td><span class="badge badge-green">Expected</span></td></tr>
        <tr><td>Negative Quantity</td><td>{val['negative_quantity_pct']:.2f}%</td>
            <td><span class="badge badge-green">Returns</span></td></tr>
        <tr><td>Zero/Null Price</td><td>{val['zero_price_pct']:.2f}%</td>
            <td><span class="badge badge-green">OK</span></td></tr>
        <tr><td>Exact Duplicates</td><td>{val['exact_duplicate_pct']:.2f}%</td>
            <td><span class="badge badge-green">Low</span></td></tr>
      </tbody>
    </table>
    {'<ul class="issues">' + warnings_html + errors_html + '</ul>' if warnings_html or errors_html else ''}
  </section>

  <!-- Distribution -->
  <section>
    <h2>Distribution Summary (Raw)</h2>
    <table>
      <thead><tr><th>Column</th><th>Min</th><th>Max</th><th>Mean</th><th>P95</th><th>P99.5</th></tr></thead>
      <tbody>{dist_rows}</tbody>
    </table>
  </section>

  <!-- Cleaning Funnel -->
  <section>
    <h2>Cleaning Pipeline — {len(data['cleaning']['passes'])} Passes</h2>
    <table>
      <thead><tr><th>Pass</th><th>Description</th><th>Before</th><th>Removed</th><th>After</th><th>%</th></tr></thead>
      <tbody>{passes_rows}</tbody>
    </table>
  </section>

  <!-- Sequence Summary -->
  <section>
    <h2>Sequence Build Summary</h2>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Total customers (raw)</td><td>{seq['total_customers_raw']:,}</td></tr>
        <tr><td>Customers kept</td><td>{seq['total_customers_kept']:,}</td></tr>
        <tr><td>Excluded (too short)</td><td>{seq['customers_excluded_short']:,}</td></tr>
        <tr><td>Total events in sequences</td><td>{seq['total_events']:,}</td></tr>
        <tr><td>Avg session length</td><td>{seq['avg_sequence_length']}</td></tr>
        <tr><td>Avg unique SKUs / customer</td><td>{seq['avg_unique_skus_per_customer']}</td></tr>
        <tr><td>Avg sessions / customer</td><td>{seq['avg_sessions_per_customer']}</td></tr>
      </tbody>
    </table>
  </section>

</div>
</body>
</html>"""


# ── Public entry point ────────────────────────────────────────────────────────

def run_report(
    validation: ValidationReport,
    cleaning: CleaningReport,
    sequences: SequenceReport,
) -> Path:
    """Generate and save HTML + JSON quality reports.

    Returns the path to the HTML report.
    """
    logger.info("=" * 60)
    logger.info("GENERATING QUALITY REPORT")
    logger.info("=" * 60)

    data = _build_json_report(validation, cleaning, sequences)

    json_path = REPORTS_DIR / "quality_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.success(f"JSON report saved: {json_path}")

    html_path = REPORTS_DIR / "quality_report.html"
    html = _build_html_report(data)
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.success(f"HTML report saved: {html_path}")
    logger.info(f"Open in browser: file:///{html_path.as_posix()}")

    return html_path
