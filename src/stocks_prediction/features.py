"""Feature matrix and label construction for a single stock."""
import pandas as pd
import numpy as np

from src.stocks_prediction.indicators import calculate_rsi
from src.stocks_prediction.returns import compute_net_return_series


def build_feature_matrix(aux,
    df: pd.DataFrame,
    df_pca: pd.DataFrame,
    target_stock: str,
    n_pca_components: int,
    n_past_days: int,
    ohlcv: dict[str, pd.DataFrame] | None = None,
    test_flag: bool = False,
    budget: float | None = None,
    commission_bps: float = 0.08,
    buy_sell_fee: float = 2.00,
    profit: float = 30.00,
    label_type: str = "percentage",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build the feature matrix and labels for one stock.

    Features include lagged returns, distances to moving averages and Bollinger
    bands, MACD histogram, rolling volatility, RSI and PCA components (with lags).

    Args:
        df: Price table (one column per stock), indexed by date.
        df_pca: PCA components aligned with ``df``.
        target_stock: Column of ``df`` to model.
        n_pca_components: Number of PCA components used.
        n_past_days: Maximum lag (in days) for the lagged features.
        test_flag: Reserved flag (not used in the computation).
        budget: Capital per position for net returns, or None for plain returns.
        commission_bps: Proportional commission in basis points.
        buy_sell_fee: Fixed fee per buy and sell.
        profit: Minimum net profit for the logical (classification) label.
        label_type: Label mode. Any value other than "percentage" requires a budget.

    Returns:
        Tuple of (features, regression target, classification target, price).

    Raises:
        ValueError: If a non-percentual label type is requested without a budget.
    """
    if label_type not in ("profit", "percentage"):
        raise ValueError(
            f"Invalid label_type: {label_type!r}. "
            "Please, label_type must be either 'profit' or 'percentage'."
        )

    df = df.sort_index()

    if (label_type != "percentage") and (budget is None):
            raise ValueError("label_type='profit' requires budget, buy_sell_fee and commission_bps.")

    x_train_return = compute_net_return_series(df, budget, buy_sell_fee, commission_bps)

    x_train = pd.DataFrame(index=x_train_return.index)

    # Lagged returns (every second lag up to n_past_days)
    features: dict[str, pd.Series] = {}
    for i in range(0, n_past_days, 1):
        features[f"return_lag{i}"] = x_train_return[target_stock].shift(i)
    x_train = pd.concat(
        [x_train, pd.DataFrame(features, index=x_train.index)],
        axis=1,
    )

    # Moving averages (trend filters)
    sma_20 = df[target_stock].rolling(window=20).mean()
    sma_50 = df[target_stock].rolling(window=50).mean()
    sma_200 = df[target_stock].rolling(window=200).mean()

    # Bollinger bands (percentage distances to the bands)
    std_20 = df[target_stock].rolling(window=20).std()
    bolling_upper = sma_20 + (std_20 * 2)
    bolling_lower = sma_20 - (std_20 * 2)
    dist_upper = (bolling_upper - df[target_stock]) / df[target_stock]
    dist_lower = (df[target_stock] - bolling_lower) / df[target_stock]

    # MACD (line and histogram)
    ema_12 = df[target_stock].ewm(span=12, adjust=False).mean()
    ema_26 = df[target_stock].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # Moving-average, Bollinger and MACD features with lags
    features = {}
    for i in range(0, n_past_days*2, 2):
        features[f"dist_sma_20_lag{i}"] = ((df[target_stock] - sma_20) / sma_20).shift(i)
        features[f"dist_sma_50_lag{i}"] = ((df[target_stock] - sma_50) / sma_50).shift(i)
        features[f"dist_sma_200_lag{i}"] = ((df[target_stock] - sma_200) / sma_200).shift(i)
        features[f"dist_bollinger_upper_lag{i}"] = dist_upper.shift(i)
        features[f"dist_bollinger_lower_lag{i}"] = dist_lower.shift(i)
        features[f"macd_hist_lag{i}"] = macd_hist.shift(i)
    '''for i in range(5, n_past_days*4+1, 5):
        features[f"dist_sma_20_lag{i}"] = ((df[target_stock] - sma_20) / sma_20).shift(i)'''
    x_train = pd.concat(
        [x_train, pd.DataFrame(features, index=x_train.index)],
        axis=1,
    )

    # Rolling volatility (std of net returns over the last 5 days)
    for i in range(0, n_past_days*2, 2):
        x_train[f"rolling_vol_lag{i}"] = x_train_return[target_stock].rolling(window=5).std().shift(i)

    # RSI (14-day period)
    for i in range(0, n_past_days*2, 2):
        x_train[f"rsi_lag{i}"] = calculate_rsi(df[target_stock], period=14).shift(i)

    # PCA components (current values)
    features = {}
    for i in range(1, df_pca.shape[1] + 1):
        features[f"PC{i}"] = df_pca[f"PC{i}"]
    x_train = pd.concat(
        [x_train, pd.DataFrame(features, index=x_train.index)],
        axis=1,
    )

    # PCA components with lags
    features = {}
    for i in range(2, n_past_days, 1):
        for j in range(1, n_pca_components + 1):
            features[f"PC{j}_lag{i}"] = df_pca[f"PC{j}"].shift(i)
    x_train = pd.concat(
        [x_train, pd.DataFrame(features, index=x_train.index)],
        axis=1,
    )

    # Right before "# Targets and price":
    if ohlcv is not None:
        # Align each field to the rows of df, then build the leak-free OHLCV features
        aligned = {name: table[target_stock].reindex(df.index) for name, table in ohlcv.items()}
        ohlcv_features = build_ohlcv_features(
            aligned["open"], aligned["high"], aligned["low"], aligned["close"], aligned["volume"],
        )
        x_train = pd.concat([x_train, ohlcv_features.reindex(x_train.index)], axis=1)

    # Targets and price
    x_train["price"] = df[target_stock]
    
    if (label_type != "percentage"):
        n_stocks = ((budget - buy_sell_fee) // (df[target_stock] * (1 + (commission_bps / 10000.0)))).astype(("Int64"))
        entry_fee = buy_sell_fee + (commission_bps / 10000.0) * (n_stocks * df[target_stock])
        exit_fee = buy_sell_fee + (commission_bps / 10000.0) * (n_stocks * df[target_stock].shift(-1))
        x_net_return = n_stocks * (df[target_stock].shift(-1) - df[target_stock]) - exit_fee - entry_fee
        invested_capital = n_stocks*df[target_stock]
        
        assert budget >= (invested_capital+entry_fee).all().all()
        
        x_train["target"] = x_net_return/invested_capital
        x_train.loc[(invested_capital == 0) & (x_net_return.notna()), "target"] = 0

        x_train["target_logical"] = (x_net_return >= profit).astype(("Int64"))
        
    else:
        
        x_train["target"] = df[target_stock].pct_change().shift(-1)
        x_train["target_logical"] = (x_train["target"] > 0).astype(int)
        
    
    
    
    assert x_train.iloc[-1].drop(
        labels=["target", "target_logical"],
        ).notna().all(), (
        "NaNs in the last row are only allowed in 'target' and 'target_logic'. "
        "Unexpected NaNs may indicate data leakage or a feature engineering issue."
    )

    
    x_eval = x_train.loc[x_train.index >= aux]

    # Identify columns with at least one NaN in the evaluation period
    nan_columns = x_eval.columns[x_eval.isna().any(axis=0)]

    # Keep only those columns and rows containing NaNs
    x_train_nans = x_eval.loc[
        x_eval[nan_columns].isna().any(axis=1),
        nan_columns
    ]

    
    
    x_train = x_train.dropna()
    
    price = x_train["price"]
    y = x_train["target"]
    y_logical = x_train["target_logical"]

    x_train = x_train.drop(columns=["target"])
    x_train = x_train.drop(columns=["price"])
    x_train = x_train.drop(columns=["target_logical"])
    
    return x_train, y, y_logical, price


'''def build_relevant_features(prices: pd.DataFrame, target_stock: str) -> pd.DataFrame:
    """Build features that use only information available up to each date.
    Args:
        prices: Price table (one column per stock), indexed by date, sorted ascending.
        target_stock: Column of ``prices`` to model.
    Returns:
        DataFrame indexed like ``prices`` with one column per feature.
        Early rows contain NaN until each rolling window is complete.
    """
    price = prices[target_stock]
    returns = price.pct_change()
    market_returns = prices.pct_change().mean(axis=1)  # equal-weight market proxy
    features: dict[str, pd.Series] = {}
    # Momentum at several horizons
    for horizon in (1, 5, 10, 20, 60):
        features[f"mom_{horizon}d"] = price.pct_change(horizon)
    # Volatility and volatility regime
    features["vol_5d"] = returns.rolling(5).std()
    features["vol_20d"] = returns.rolling(20).std()
    features["vol_60d"] = returns.rolling(60).std()
    features["vol_ratio_5_60"] = features["vol_5d"] / features["vol_60d"]
    # Position within the recent range
    features["drawdown_60d"] = price / price.rolling(60).max() - 1
    rolling_mean_20 = price.rolling(20).mean()
    rolling_std_20 = price.rolling(20).std()
    features["zscore_20d"] = (price - rolling_mean_20) / rolling_std_20
    # Shape of the return distribution
    features["skew_20d"] = returns.rolling(20).skew()
    # Market context and relative strength
    features["mkt_mom_5d"] = market_returns.rolling(5).sum()
    features["mkt_mom_20d"] = market_returns.rolling(20).sum()
    features["rel_strength_20d"] = features["mom_20d"] - prices.pct_change(20).mean(axis=1)
    # Calendar effect (day of week as integer 0-4)
    # Calendar effect: one-hot encoding of the day of the week (Mon..Fri)
    day_of_week = pd.Series(prices.index.dayofweek, index=prices.index)
    day_of_week_cat = pd.Categorical(day_of_week, categories=range(5))
    day_dummies = pd.get_dummies(day_of_week_cat, prefix="dow", dtype=float)
    day_dummies.index = prices.index  # Garante o alinhamento das datas

    print(day_dummies)
    features_df = pd.DataFrame(features, index=prices.index)
    result = pd.concat([features_df, day_dummies], axis=1)


    result = result.dropna()
    return result'''

def build_ohlcv_features(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """Build leak-free features from the OHLCV series of one stock.
    Args:
        open_: Opening prices, indexed by date.
        high: Daily high prices.
        low: Daily low prices.
        close: Daily closing prices.
        volume: Daily traded volume.
    Returns:
        DataFrame indexed like the inputs, one column per feature.
        Rows are NaN until each rolling window is complete.
    """
    prev_close = close.shift(1)
    day_range = (high - low)#.replace(0, np.nan)
    log_volume = np.log(volume.where(volume != 0, 1)) #np.log(volume.replace(0, np.nan))
    # Known at the open of day t
    gap_open = (open_ / (prev_close) - 1).where(prev_close!=0, 0)
    # Known after the close of day t-1 (shifted by 1)
    range_pct = ((high - low) / (close)).where(close!=0, 0).shift(1)
    close_location = ((close - low) / (day_range)).where(day_range!=0, 0).shift(1)
    # Parkinson volatility: uses only the high-low range
    parkinson_daily = np.log(high / low) ** 2 / (4 * np.log(2))
    park_vol_5 = np.sqrt(parkinson_daily.rolling(5).mean()).shift(1)
    park_vol_20 = np.sqrt(parkinson_daily.rolling(20).mean()).shift(1)
    # Average True Range (14 days), normalised by price
    prev_close_safe = prev_close
    true_range = pd.concat([
        high - low,
        (high - prev_close_safe).abs(),
        (low - prev_close_safe).abs(),
    ], axis=1).max(axis=1)
    atr_pct = (true_range.rolling(14).mean() / close).shift(1)
    # Volume features (relative to the last 20 days)
    volume_ratio = (log_volume - log_volume.rolling(20).mean()).shift(1)
    volume_z_20 = (
        (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(20).std()
    ).shift(1)



    volume_mean = log_volume.rolling(20).mean()
    volume_std = log_volume.rolling(20).std()

    

    volume_z_20 = volume_z_20.replace([np.inf, -np.inf], 0)
    volume_z_20 = volume_z_20.mask(((log_volume - log_volume.rolling(20).mean()).shift(1) == 0) & (log_volume.rolling(20).std().shift(1) == 0), 0)

    prev_return = close.pct_change().shift(1)
    signed_volume = np.sign(prev_return) * volume_ratio
    # Smoothed close location: persistent buying or selling pressure
    close_location_ma_5 = close_location.rolling(5).mean()

    

    return pd.DataFrame({
        "gap_open": gap_open,
        "range_pct": range_pct,
        "close_location": close_location,
        "close_location_ma_5": close_location_ma_5,
        "park_vol_5": park_vol_5,
        "park_vol_20": park_vol_20,
        "atr_pct_14": atr_pct,
        "volume_ratio_20": volume_ratio,
        "volume_z_20": volume_z_20,
        "signed_volume": signed_volume,
    }, index=close.index)