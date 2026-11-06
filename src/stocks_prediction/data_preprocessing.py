from typing import Optional, Tuple

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_data_to_table(
    file_path: str,
    price_moment: str = "Open"
) -> pd.DataFrame:
    """
    Read price data from CSV file and reshape into a pivot table with timestamps as index and tickers as columns.

    Args:
        file_path: Path to the CSV file containing price data
        price_moment: Which price field to extract ('Open', 'Close', 'High', 'Low', 'Volume')

    Returns:
        DataFrame with timestamps as index and tickers as columns for the specified price moment

    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the specified price_moment is not found in the data
    """
    # Validate file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Validate required columns exist
    price_moment = price_moment.capitalize() 
    required_cols = ['Date', 'ticker', price_moment]
    cols_lower = [col.lower() for col in df.columns]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Pivot the table to have dates as index and tickers as columns
    pivot_table = df.pivot_table(
        index='Date',
        columns='ticker',
        values=price_moment
    )
    
    # Sort index chronologically
    pivot_table = pivot_table.sort_index()
    
    logger.info(f"Successfully loaded {price_moment} data: {pivot_table.shape[0]} dates, {pivot_table.shape[1]} tickers")
    return pivot_table


def clean_price_data(
    price_df: pd.DataFrame,
    min_observations: int = 100,
    max_missing_pct: float = 0.05
) -> pd.DataFrame:
    """
    Clean price data by removing tickers with excessive missing values and forward-filling remaining gaps.

    Args:
        price_df: DataFrame with dates as index and tickers as columns
        min_observations: Minimum number of non-NaN observations required for a ticker to be kept
        max_missing_pct: Maximum percentage of missing values allowed for a ticker

    Returns:
        Cleaned DataFrame with only valid tickers and forward-filled values

    Raises:
        ValueError: If the resulting DataFrame is empty after cleaning
    """
    if price_df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Calculate missing percentage for each ticker
    missing_pct = price_df.isna().mean()
    
    # Filter tickers based on missing percentage and minimum observations
    valid_tickers = missing_pct[
        (missing_pct <= max_missing_pct) & 
        (price_df.count() >= min_observations)
    ].index
    
    if len(valid_tickers) == 0:
        raise ValueError("No tickers meet the minimum data quality requirements")
    
    # Keep only valid tickers
    cleaned_df = price_df[valid_tickers].copy()
    
    # Forward fill remaining gaps (no look-ahead bias as we only use past data)
    cleaned_df = cleaned_df.ffill()
    
    # Backfill any remaining leading NaNs (only for the very beginning of the series)
    cleaned_df = cleaned_df.bfill()
    
    # Drop any remaining rows with NaN values
    cleaned_df = cleaned_df.dropna()
    
    logger.info(f"Cleaned data: {cleaned_df.shape[0]} dates, {cleaned_df.shape[1]} tickers")
    return cleaned_df


def calculate_returns(
    price_df: pd.DataFrame,
    method: str = 'simple'
) -> pd.DataFrame:
    """
    Calculate returns from price data.

    Args:
        price_df: DataFrame with prices (index=dates, columns=tickers)
        method: Return calculation method ('simple' or 'log')

    Returns:
        DataFrame with returns

    Raises:
        ValueError: If an invalid method is specified
    """
    if method == 'simple':
        returns = price_df.pct_change()
    elif method == 'log':
        returns = np.log(price_df / price_df.shift(1))
    else:
        raise ValueError(f"Invalid return method: {method}. Use 'simple' or 'log'")
    
    # Drop the first row (NaN returns)
    returns = returns.dropna(how='all')
    
    return returns


def split_train_test(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
    validation_ratio: Optional[float] = None,
    date_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Split data into train and test sets with strict temporal ordering to prevent look-ahead bias.

    Args:
        data: DataFrame with datetime index or date column
        train_ratio: Proportion of data to use for training (0.0 to 1.0)
        validation_ratio: Optional proportion for validation set (if None, only train/test split)
        date_column: Name of date column if not using index

    Returns:
        Tuple of (train_data, test_data, validation_data) where validation_data is None if not requested

    Raises:
        ValueError: If ratios are invalid or data is empty
    """
    if data.empty:
        raise ValueError("Input data is empty")
    
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    if validation_ratio is not None and not 0 < validation_ratio < 1:
        raise ValueError(f"validation_ratio must be between 0 and 1, got {validation_ratio}")
    
    # Ensure we have a datetime index
    if date_column is not None:
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")
        data = data.set_index(pd.to_datetime(data[date_column])).drop(columns=[date_column])
    else:
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"Index cannot be converted to datetime: {e}")
    
    # Sort by date to ensure temporal ordering
    data = data.sort_index()
    
    n = len(data)
    
    if validation_ratio is not None:
        # Split into train, validation, and test
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * validation_ratio)
        
        train_data = data.iloc[:train_end]
        validation_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        logger.info(f"Split data: train={len(train_data)}, validation={len(validation_data)}, test={len(test_data)}")
        return train_data, test_data, validation_data
    else:
        # Simple train/test split
        train_end = int(n * train_ratio)
        
        train_data = data.iloc[:train_end]
        test_data = data.iloc[train_end:]
        
        logger.info(f"Split data: train={len(train_data)}, test={len(test_data)}")
        return train_data, test_data, None


def preprocess_and_split(
    file_path: str,
    price_moment: str = "Close",
    train_ratio: float = 0.8,
    validation_ratio: Optional[float] = None,
    min_observations: int = 100,
    max_missing_pct: float = 0.2,
    save_data: bool = True,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Complete preprocessing pipeline: read data, clean it, and split into train/test sets.

    Args:
        file_path: Path to the raw data CSV file
        price_moment: Which price field to use ('Open', 'Close', 'High', 'Low', 'Volume')
        train_ratio: Proportion of data for training
        validation_ratio: Optional proportion for validation
        min_observations: Minimum observations required per ticker
        max_missing_pct: Maximum allowed missing percentage per ticker
        save_data: Whether to save the processed datasets to CSV
        output_dir: Directory to save processed data. If None, it will be derived from file_path
                   as <directory_of_file>/<filename_stem>_processed_data

    Returns:
        Tuple of (train_data, test_data, validation_data)

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If data processing fails
    """
    

    # Read raw data
    logger.info(f"Reading data from {file_path}")
    raw_data = read_data_to_table(file_path, price_moment)
    
    # Clean the data
    logger.info("Cleaning data...")
    cleaned_data = clean_price_data(
        raw_data,
        min_observations=min_observations,
        max_missing_pct=max_missing_pct
    )
    
    # Split into train/test
    logger.info("Splitting data into train/test sets...")
    train_data, test_data, validation_data = split_train_test(
        cleaned_data,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio
    )

    # Save processed data if requested
    if save_data:
        
        # If output_dir is not provided, derive it from file_path
        if output_dir is None:
            file_path_obj = Path(file_path)
            output_dir = "data" / file_path_obj.parent / f"{file_path_obj.stem}_processed_data"
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_path = "data" / output_dir / f"train_{price_moment.lower()}.csv"
        test_path = "data" / output_dir / f"test_{price_moment.lower()}.csv"
        
        train_data.to_csv(train_path)
        test_data.to_csv(test_path)

        logger.info(f"Saved train data to {train_path}")
        logger.info(f"Saved test data to {test_path}")
        
        if validation_data is not None:
            val_path = "data" / output_dir / f"validation_{price_moment.lower()}.csv"
            validation_data.to_csv(val_path)
            logger.info(f"Saved validation data to {val_path}")
        
    
    return train_data, test_data, validation_data


def load_processed_data(
    train_path: str,
    test_path: str,
    validation_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load pre-processed train/test/validation data from CSV files.

    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
        validation_path: Optional path to validation data CSV

    Returns:
        Tuple of (train_data, test_data, validation_data)
    """
    train_data = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test_data = pd.read_csv(test_path, index_col=0, parse_dates=True)
    
    validation_data = None
    if validation_path is not None:
        validation_data = pd.read_csv(validation_path, index_col=0, parse_dates=True)
    
    logger.info(f"Loaded train data: {train_data.shape}")
    logger.info(f"Loaded test data: {test_data.shape}")
    if validation_data is not None:
        logger.info(f"Loaded validation data: {validation_data.shape}")
    
    return train_data, test_data, validation_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess market data.")
    parser.add_argument("--file", type=str, required=True, help="Path to raw CSV file")
    parser.add_argument("--price_moment", type=str, default="Open", help="Price field to use (Open, Close, High, Low, Volume)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion for training")
    parser.add_argument("--validation_ratio", type=float, default=None, help="Optional validation proportion")
    parser.add_argument("--min_observations", type=int, default=100, help="Minimum observations per ticker")
    parser.add_argument("--max_missing_pct", type=float, default=0.05, help="Maximum missing percentage")
    parser.add_argument("--save", action="store_true", help="Save processed data to CSV")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Directory for output files")

    args = parser.parse_args()

    train, test, val = preprocess_and_split(
        file_path=args.file,
        price_moment=args.price_moment,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        min_observations=args.min_observations,
        max_missing_pct=args.max_missing_pct,
        save_data=args.save,
        output_dir=args.output_dir
    )

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    if val is not None:
        print(f"Validation shape: {val.shape}")
