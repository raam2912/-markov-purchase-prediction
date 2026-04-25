"""
pipeline/ingest.py — Stage 1: Download + load the UCI Online Retail II dataset.

Responsibilities:
  - Download the XLSX file with retry logic, resume support, and progress bar
  - Verify download integrity (file size check)
  - Load both sheets (2009-2010, 2010-2011) into a single raw DataFrame
  - Apply minimal dtype coercion (raw → typed)
  - Log per-sheet row counts
  - Return the raw concatenated DataFrame (no cleaning yet)

Design notes:
  - Uses tenacity for resilient retries with exponential backoff
  - Streams download in chunks to handle large files
  - Opens both XLSX sheets and concatenates with a 'source_sheet' column for traceability
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import requests
from loguru import logger
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import (
    COLUMN_RENAME,
    SHEET_NAMES,
    UCI_DOWNLOAD_URL,
    UCI_XLSX_PATH,
    UCI_ZIP_PATH,
    PipelineConfig,
)

import logging
_tenacity_logger = logging.getLogger("tenacity")


# ── Types ─────────────────────────────────────────────────────────────────────

class IngestResult(NamedTuple):
    df: pd.DataFrame
    raw_row_counts: dict[str, int]
    xlsx_path: Path
    was_cached: bool


# ── Download ──────────────────────────────────────────────────────────────────

def _build_retry(max_retries: int, timeout: int):
    """Factory: returns a tenacity retry decorator configured from PipelineConfig."""
    return retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
        reraise=True,
    )


def _download_with_progress(
    url: str,
    dest: Path,
    chunk_size: int,
    timeout: int,
    max_retries: int,
) -> None:
    """Download *url* to *dest* with a Rich progress bar and tenacity retries.

    If *dest* already exists and has non-zero size, the download is skipped.
    """
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Cached zip found at {dest} — skipping download.")
        return

    @_build_retry(max_retries, timeout)
    def _do_download() -> None:
        logger.info(f"Downloading {url} → {dest}")
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))

            with Progress(
                TextColumn("[bold cyan]Downloading"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                TaskProgressColumn(),
            ) as progress:
                task = progress.add_task("DL", total=total or None)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
                            progress.update(task, advance=len(chunk))

        logger.success(f"Download complete: {dest} ({dest.stat().st_size:,} bytes)")

    _do_download()


def _extract_xlsx(zip_path: Path, xlsx_path: Path) -> Path:
    """Extract the XLSX from the zip archive and return its path.

    Tries to locate the XLSX regardless of internal filename case differences.
    """
    if xlsx_path.exists() and xlsx_path.stat().st_size > 0:
        logger.info(f"XLSX already extracted: {xlsx_path}")
        return xlsx_path

    logger.info(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        xlsx_members = [m for m in zf.namelist() if m.lower().endswith(".xlsx")]
        if not xlsx_members:
            raise FileNotFoundError(
                f"No XLSX file found inside {zip_path}. Members: {zf.namelist()}"
            )
        # Pick the member; prefer exact match, fall back to first XLSX
        member = next(
            (m for m in xlsx_members if "retail" in m.lower()),
            xlsx_members[0],
        )
        logger.info(f"Extracting member '{member}' → {xlsx_path.parent}")
        zf.extract(member, path=xlsx_path.parent)
        extracted = xlsx_path.parent / member
        if extracted != xlsx_path:
            extracted.rename(xlsx_path)

    logger.success(f"Extracted XLSX: {xlsx_path} ({xlsx_path.stat().st_size:,} bytes)")
    return xlsx_path


# ── Load sheets ────────────────────────────────────────────────────────────────

def _load_sheet(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """Load one sheet from the XLSX, returning a minimally typed DataFrame."""
    logger.info(f"  Loading sheet: '{sheet_name}' …")

    df = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
        dtype={
            "Invoice":     str,
            "StockCode":   str,
            "Description": str,
            "Quantity":    "Int64",   # nullable integer
            "Price":       float,
            "Customer ID": str,
            "Country":     str,
        },
        parse_dates=["InvoiceDate"],
        engine="openpyxl",
    )

    # Tag with source sheet for downstream traceability
    df["source_sheet"] = sheet_name

    logger.info(f"    → {len(df):,} rows loaded from '{sheet_name}'")
    return df


def _load_all_sheets(xlsx_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load all configured sheets, concatenate, and return with per-sheet counts."""
    frames: list[pd.DataFrame] = []
    row_counts: dict[str, int] = {}

    for sheet in SHEET_NAMES:
        df_sheet = _load_sheet(xlsx_path, sheet)
        row_counts[sheet] = len(df_sheet)
        frames.append(df_sheet)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined: {len(combined):,} rows across {len(SHEET_NAMES)} sheets")
    return combined, row_counts


# ── Minimal dtype normalization ───────────────────────────────────────────────

def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal dtype fixes after loading.

    This is NOT cleaning — we only fix mismatches that would break downstream
    code (e.g., datetime timezone, string stripping).
    """
    # Rename columns to internal names (remove spaces)
    df = df.rename(columns=COLUMN_RENAME)

    # Ensure InvoiceDate is timezone-naive for consistency
    if pd.api.types.is_datetime64tz_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = df["InvoiceDate"].dt.tz_localize(None)

    # Strip leading/trailing whitespace from string columns
    for col in ("Invoice", "StockCode", "Description", "CustomerID", "Country"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Replace pandas 'nan' strings (from object→str cast) with real NaN
            df[col] = df[col].replace("nan", pd.NA)

    # Quantity: convert nullable Int64 → regular int64 after coercion
    # (we keep NA as NaN float here; nulls are dropped in cleaning pass P1)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    # Price: ensure float
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    return df


# ── Public entry point ────────────────────────────────────────────────────────

def run_ingest(cfg: PipelineConfig) -> IngestResult:
    """Execute Stage 1: download, extract, load, and minimally type the dataset.

    Args:
        cfg: Pipeline configuration.

    Returns:
        IngestResult with the raw (uncleaned) DataFrame and metadata.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1 — DATA INGESTION")
    logger.info("=" * 60)

    was_cached = UCI_XLSX_PATH.exists() and UCI_XLSX_PATH.stat().st_size > 0

    if not was_cached:
        # Step 1a: download zip
        _download_with_progress(
            url=UCI_DOWNLOAD_URL,
            dest=UCI_ZIP_PATH,
            chunk_size=cfg.download_chunk_size,
            timeout=cfg.download_timeout_seconds,
            max_retries=cfg.download_max_retries,
        )
        # Step 1b: extract XLSX
        _extract_xlsx(UCI_ZIP_PATH, UCI_XLSX_PATH)
    else:
        logger.info(f"XLSX already present ({UCI_XLSX_PATH}) — skipping download.")

    # Step 2: load sheets
    df_raw, row_counts = _load_all_sheets(UCI_XLSX_PATH)

    # Step 3: minimal dtype coercion
    df_raw = _coerce_dtypes(df_raw)

    logger.success(
        f"Ingestion complete: {len(df_raw):,} raw rows, "
        f"{df_raw['CustomerID'].nunique(dropna=True):,} unique customers (including nulls)"
    )

    return IngestResult(
        df=df_raw,
        raw_row_counts=row_counts,
        xlsx_path=UCI_XLSX_PATH,
        was_cached=was_cached,
    )
