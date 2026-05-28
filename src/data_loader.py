"""data_loader – pure I/O layer for the maritime piracy dataset."""

from __future__ import annotations

import pandas as pd


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load the raw maritime piracy dataset from *file_path* into a DataFrame.

    This function is a pure I/O boundary: it performs no transformations,
    filtering, or feature engineering.  The returned DataFrame contains the
    dataset exactly as it exists on disk.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the source data file (CSV, Parquet, etc.).

    Returns
    -------
    pd.DataFrame
        Raw dataset with original column names and dtypes.
    """
    ...
