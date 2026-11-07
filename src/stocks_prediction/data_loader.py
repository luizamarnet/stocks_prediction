"""Loading and preparing the raw CSV datasets."""
import numpy as np
import pandas as pd

TRAIN_PATH = "data/processed_data/train_open.csv"
VALIDATION_PATH = "data/processed_data/validation_open.csv"
TEST_PATH = "data/processed_data/test_open.csv"


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training (train + validation) and test price tables.

    Returns:
        Tuple of (training DataFrame, test DataFrame), both indexed by ``Date``.
    """
    train_raw = pd.concat([
        pd.read_csv(TRAIN_PATH, index_col="Date", parse_dates=True),
        pd.read_csv(VALIDATION_PATH, index_col="Date", parse_dates=True),
    ])
    test_raw = pd.read_csv(TEST_PATH, index_col="Date", parse_dates=True)
    return train_raw, test_raw

OHLCV_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
def load_ohlcv(fields: tuple[str, ...] = OHLCV_FIELDS) -> dict[str, pd.DataFrame]:
    """Load the requested price/volume tables for every split, combined by date.
    Args:
        fields: Field names to load. Each one must have train, validation and test CSVs.
    Returns:
        Dictionary mapping each field name to its combined table (Date index, one column per stock).
    Raises:
        ValueError: If the fields do not share the same stock columns.
    """
    tables: dict[str, pd.DataFrame] = {}
    for field in fields:
        train = pd.concat([
            pd.read_csv(f"data/processed_data/train_{field}.csv", index_col="Date", parse_dates=True),
            pd.read_csv(f"data/processed_data/validation_{field}.csv", index_col="Date", parse_dates=True),
        ])
        test = pd.read_csv(f"data/processed_data/test_{field}.csv", index_col="Date", parse_dates=True)
        full = pd.concat([train, test]).sort_index()
        tables[field] = full[~full.index.duplicated(keep="first")]
    reference_columns = tables[fields[0]].columns
    for field, table in tables.items():
        if not table.columns.equals(reference_columns):
            raise ValueError(f"Stock columns of '{field}' do not match '{fields[0]}'.")
    return tables



def build_full_dataset(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    """Concatenate training and test data, sorted by date, without duplicated dates.

    Args:
        train_raw: Training price table.
        test_raw: Test price table.

    Returns:
        Combined price table, sorted by index, keeping the first occurrence of each date.
    """
    full = pd.concat([train_raw, test_raw]).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full


def build_week_labels(index: pd.DatetimeIndex) -> np.ndarray:
    """Label each date with its ISO year-week string, e.g. ``2024-W05``.

    Args:
        index: Datetime index of the combined dataset.

    Returns:
        Array of week labels, one per row.
    """
    iso = index.isocalendar()
    return (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)).values
