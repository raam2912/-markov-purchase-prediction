"""
main.py — CLI entry point for the AntiGrav data pipeline.

Usage:
    python main.py --all              # Run full pipeline (stages 1–4 + report)
    python main.py --ingest           # Stage 1 only (download + load)
    python main.py --validate         # Stages 1–2
    python main.py --clean            # Stages 1–3
    python main.py --sequences        # Stages 1–4 (no report)
    python main.py --report-only      # Re-generate report from existing parquet (not impl yet)
    python main.py --log-level DEBUG  # Verbose logging

Environment variable overrides (see config.py for full list):
    ANTIGRAV_LOG_LEVEL=DEBUG python main.py --all
    ANTIGRAV_CAP_OUTLIERS=false python main.py --all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Configure loguru before any pipeline imports
def _configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    # Always write full DEBUG log to file
    log_file = Path(__file__).parent / "data" / "pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        rotation="10 MB",
        retention="7 days",
    )


_configure_logging()  # default level; overridden after args parse below

from config import get_config_from_env, PROCESSED_DIR, REPORTS_DIR
from pipeline.ingest import run_ingest
from pipeline.validate import run_validate
from pipeline.clean import run_clean
from pipeline.sequence_builder import run_sequence_builder
from pipeline.report import run_report


console = Console()


def _banner() -> None:
    console.print(Panel(
        Text("AntiGrav — Market Prediction Pipeline\nPhase 1: Data Ingestion", justify="center"),
        style="bold cyan",
        border_style="dim blue",
    ))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AntiGrav data pipeline — UCI Online Retail II ingest & clean",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all",         action="store_true", help="Run full pipeline (stages 1–4 + report)")
    group.add_argument("--ingest",      action="store_true", help="Stage 1 only")
    group.add_argument("--validate",    action="store_true", help="Stages 1–2")
    group.add_argument("--clean",       action="store_true", help="Stages 1–3")
    group.add_argument("--sequences",   action="store_true", help="Stages 1–4 (no report)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate DataFrames after each cleaning pass",
    )

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.all, args.ingest, args.validate, args.clean, args.sequences]):
        args.all = True

    # Re-configure logging with user-specified level
    _configure_logging(args.log_level)

    _banner()
    start = time.perf_counter()

    cfg = get_config_from_env()
    if args.save_intermediates:
        cfg = cfg.model_copy(update={"save_intermediates": True})

    validation_report = None
    cleaning_report   = None
    sequence_report   = None

    try:
        # ── Stage 1: Ingest ───────────────────────────────────────────────────
        ingest_result = run_ingest(cfg)

        if args.ingest:
            logger.success("Stage 1 complete. Stopping as requested.")
            return 0

        # ── Stage 2: Validate ─────────────────────────────────────────────────
        validation_report = run_validate(ingest_result.df, cfg)

        if args.validate:
            logger.success("Stages 1–2 complete. Stopping as requested.")
            return 0

        # ── Stage 3: Clean ────────────────────────────────────────────────────
        df_clean, cleaning_report = run_clean(ingest_result.df, cfg)

        # Save clean transactions immediately (belt-and-braces)
        clean_path = PROCESSED_DIR / "transactions_clean.parquet"
        df_clean.to_parquet(clean_path, index=False, engine="pyarrow")
        logger.success(f"Clean data persisted: {clean_path}")

        if args.clean:
            logger.success("Stages 1–3 complete. Stopping as requested.")
            return 0

        # ── Stage 4: Sequences ────────────────────────────────────────────────
        df_seq, sequences, sequence_report = run_sequence_builder(df_clean, cfg)

        # ── Report ────────────────────────────────────────────────────────────
        if args.all:
            html_path = run_report(validation_report, cleaning_report, sequence_report)
            console.print(f"\n[bold green]✓ Report:[/] file:///{html_path.as_posix()}")

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        return 130

    except Exception as exc:
        logger.critical(f"Pipeline FAILED: {exc}")
        logger.exception(exc)
        return 1

    elapsed = time.perf_counter() - start
    console.print(f"\n[bold green]>> Pipeline complete in {elapsed:.1f}s[/]")
    console.print(f"  Clean data : [cyan]{PROCESSED_DIR / 'transactions_clean.parquet'}[/]")
    console.print(f"  Sequences  : [cyan]{PROCESSED_DIR / 'sequences.parquet'}[/]")
    if args.all:
        console.print(f"  Report     : [cyan]{REPORTS_DIR / 'quality_report.html'}[/]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
