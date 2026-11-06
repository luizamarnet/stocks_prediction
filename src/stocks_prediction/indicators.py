"""Technical indicator helpers."""
import pandas as pd


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) of a price series.

    Args:
        series: Price series.
        period: Rolling window length.

    Returns:
        RSI values in the range 0-100 (NaN where the window is incomplete).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
