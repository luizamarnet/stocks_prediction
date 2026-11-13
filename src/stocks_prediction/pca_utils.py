"""PCA fitting (on the training window) and transformation (on the evaluation window)."""
import pandas as pd
from sklearn.decomposition import PCA

from src.stocks_prediction.returns import compute_net_return_series
from sklearn.preprocessing import StandardScaler

def prepare_pca_data_train(
    df: pd.DataFrame,
    n_pca_components: int,
    budget: float | None = None,
    buy_sell_fee: float = 2.00,
    commission_bps: float = 0.08,
    label_type: str = "percentage",
) -> tuple[pd.DataFrame, PCA, StandardScaler]:
    """Fit PCA on the training returns.

    Args:
        df: Price table for the training window.
        n_pca_components: Number of principal components to keep.
        budget: Capital per position for net returns, or None for plain returns.
        buy_sell_fee: Fixed fee per buy and sell.
        commission_bps: Proportional commission in basis points.

    Returns:
        Tuple of (PCA scores as a DataFrame with columns PC1..PCn, fitted PCA object).
    """
    if label_type not in ("profit", "percentage"):
        raise ValueError(
            f"Invalid label_type: {label_type!r}. "
            "Please, label_type must be either 'profit' or 'percentage'."
        )
    df = df.sort_index()



    scaler = StandardScaler()
    
    df_return = compute_net_return_series(df, budget, buy_sell_fee, commission_bps, label_type)
    df_return = df_return.dropna()

    pca = PCA(n_components=n_pca_components)

    #df_return_scaled = scaler.fit_transform(df_return)
    x_train_pca = pca.fit_transform(df_return)

    x_train_pca = pd.DataFrame(
        x_train_pca,
        index=df_return.index,
        columns=[f"PC{i + 1}" for i in range(x_train_pca.shape[1])],
    )

    return x_train_pca, pca, scaler


def prepare_pca_data_test(
    pca: PCA,
    pca_scaler: StandardScaler, 
    df: pd.DataFrame,
    budget: float | None = None,
    buy_sell_fee: float = 2.00,
    commission_bps: float = 0.08,
    label_type: str = "percentage",
) -> pd.DataFrame:
    """Project the evaluation returns onto an already fitted PCA.

    Args:
        pca: PCA object fitted on the training window.
        df: Price table for the evaluation window.
        budget: Capital per position for net returns, or None for plain returns.
        buy_sell_fee: Fixed fee per buy and sell.
        commission_bps: Proportional commission in basis points.

    Returns:
        PCA scores as a DataFrame with columns PC1..PCn.
    """
    if label_type not in ("profit", "percentage"):
        raise ValueError(
            f"Invalid label_type: {label_type!r}. "
            "Please, label_type must be either 'profit' or 'percentage'."
        )
    
    df = df.sort_index()

    df_return = compute_net_return_series(df, budget, buy_sell_fee, commission_bps, label_type)
    df_return = df_return.dropna()

    #df_return_scaled = pca_scaler.transform(df_return)
    x_test_pca = pca.transform(df_return)

    x_test_pca = pd.DataFrame(
        x_test_pca,
        index=df_return.index,
        columns=[f"PC{i + 1}" for i in range(x_test_pca.shape[1])],
    )

    return x_test_pca
