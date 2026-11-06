"""Return computation shared by the feature and PCA builders."""
import pandas as pd


def compute_net_return_series(
    prices: pd.DataFrame,
    budget: float | None,
    buy_sell_fee: float,
    commission_bps: float,
    label_type: str = "percentage",
) -> pd.DataFrame:
    """Compute per-day returns.

    If a budget is given, returns are net of trading fees on a fixed-budget
    position. Otherwise, simple percentage returns are used.

    Args:
        prices: Price table (one column per stock), indexed by date.
        budget: Capital allocated per position, or None for plain returns.
        buy_sell_fee: Fixed fee charged per buy and per sell.
        commission_bps: Proportional commission in basis points.

    Returns:
        Table of returns with the same shape as ``prices``.
    """
    if (label_type != "percentage"):
        n_stocks = budget // prices.shift(1)
        entry_fee = buy_sell_fee + (commission_bps / 10000.0) * (n_stocks * prices.shift(1))
        exit_fee = buy_sell_fee + (commission_bps / 10000.0) * (n_stocks * prices)
        invested_capital = n_stocks * prices.shift(1)
        df_return = (n_stocks * (prices - prices.shift(1)) - exit_fee - entry_fee) / invested_capital
        df_return[invested_capital==0] = 0
        return df_return
    return prices.pct_change()
