"""features.transactions – pure, pipe-compatible data-transformation stubs.

Each function in this module is a *transaction*: it receives an immutable
DataFrame, applies a single, well-scoped transformation, and returns a **new**
DataFrame without mutating the input.  The functions are designed to be
composed via ``pd.DataFrame.pipe()``.
"""

from __future__ import annotations

import pandas as pd


def tx_clean_vessels(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise and clean vessel-related columns.

    Normalises vessel-type categories, drops duplicate IMO entries, and
    coerces date/time columns to ``datetime64`` where appropriate.

    Parameters
    ----------
    df:
        Raw or upstream DataFrame containing vessel metadata columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned vessel columns; all other columns are
        passed through unchanged.
    """
    ...


def tx_extract_nlp(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured NLP features from free-text incident description.

    Parses the incident narrative column to derive binary indicator columns
    (e.g. ``has_weapon``, ``crew_taken_hostage``) and a numeric ``severity``
    score.

    Parameters
    ----------
    df:
        DataFrame that must contain an incident description text column.

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with NLP-derived feature columns.
    """
    ...


def tx_spatial_binning(df: pd.DataFrame) -> pd.DataFrame:
    """Discretise latitude/longitude coordinates into spatial grid bins.

    Projects each incident location onto a configurable grid and adds
    ``lat_bin`` and ``lon_bin`` categorical columns that group incidents by
    geographic region.

    Parameters
    ----------
    df:
        DataFrame containing ``latitude`` and ``longitude`` numeric columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional ``lat_bin`` and ``lon_bin``
        columns encoding the spatial bin assignment.
    """
    ...
