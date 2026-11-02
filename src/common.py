"""Logging and file helpers shared by every step."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from .config import OUT_LOGS


def get_logger(name: str, logfile: str) -> logging.Logger:
    """Logger that prints to stdout and writes (overwrites) outputs/logs/<logfile>."""
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(OUT_LOGS / logfile, mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.propagate = False
    return log


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file so an interrupted run never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def log_df(log: logging.Logger, df: pd.DataFrame, title: str | None = None, **kwargs) -> None:
    """Log a DataFrame as an aligned text table."""
    if title:
        log.info(title)
    for line in df.to_string(**kwargs).splitlines():
        log.info("    %s", line)
