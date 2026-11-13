"""Week-by-week walk-forward evaluation for a single stock."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.stocks_prediction.features import build_feature_matrix
from src.stocks_prediction.models import classifier_train, regressor_train
from src.stocks_prediction.pca_utils import prepare_pca_data_test, prepare_pca_data_train


def run_walk_forward(
    full: pd.DataFrame,
    week_labels: np.ndarray,
    test_weeks: np.ndarray,
    target_stock: str,
    n_pca_components: int,
    n_past_days: int,
    train_size: int,
    warmup: int,
    ohlcv: dict[str, pd.DataFrame] | None = None,
    budget: float | None = None,
    buy_sell_fee: float = 0.08, 
    commission_bps:float = 2.00,
    profit: float = 30.00,
    label_type: str = "percentage",
    val_frac: float = 0.1,
    regressor_flag: bool = False,
    walk_forward_flag: bool = False,
) -> pd.DataFrame:
    """Train and predict week by week for one stock.

    For each test week, models are trained only on data before that week
    (a sliding window of ``train_size`` rows) and predictions are made for the week.

    Args:
        full: Combined open-price table sorted by date.
        week_labels: ISO week label for each row of ``full``.
        test_weeks: Test week labels, in chronological order.
        target_stock: Column of ``full`` to model.
        n_pca_components: Number of PCA components.
        n_past_days: Maximum lag (in days) for the lagged features.
        train_size: Training window length in rows.
        warmup: Extra history rows needed for the longest indicator.
        ohlcv: Optional dict with the ``high``, ``low``, ``close`` and ``volume``
            tables. When given, OHLCV features are added to the feature matrix.

    Returns:
        Predictions indexed by date with columns ``classe_predict_proba``,
        ``retorno_previsto`` and ``price_test``.
    """

    if label_type not in ("profit", "percentage"):
        raise ValueError(
            f"Invalid label_type: {label_type!r}. "
            "Please, label_type must be either 'profit' or 'percentage'."
        )

    weekly_results: list[pd.DataFrame] = []
    week_counter = 0

    for week_label in test_weeks:
        
        week_positions = np.where(week_labels == week_label)[0]
        week_start, week_end = week_positions[0], week_positions[-1] + 1
        week_dates = full.index[week_start:week_end]

        # Training: sliding window, only data BEFORE the week
        train_df = full.iloc[max(0, week_start - train_size):week_start]
        # Evaluation: history (for features) + the week itself
        if walk_forward_flag:
            eval_df = full.iloc[max(0, week_start - warmup):week_end + 1]
        else:
            eval_df = full.iloc[max(0, week_start - warmup):]

        #print(train_df.tail())
        assert train_df.index.max() < week_dates[0]
        expected_last_train_day = train_df.index[-2]

        n_val = int(len(train_df) * val_frac)
        train_df_aux = train_df.iloc[:-n_val]
        val_df = train_df.iloc[-n_val-1:]

        # PCA is fitted on the training window only and applied to the evaluation window
        X_train_pca, pca, pca_scaler = prepare_pca_data_train(train_df_aux, n_pca_components, 
                                                  budget=budget, buy_sell_fee = buy_sell_fee, commission_bps = commission_bps, label_type=label_type)
        val_df_pca = prepare_pca_data_test(pca, pca_scaler, val_df, 
                                                    budget=budget, buy_sell_fee = buy_sell_fee, commission_bps = commission_bps, label_type=label_type)
        
        X_train_pca = pd.concat([X_train_pca,val_df_pca], axis=0)
        X_test_pca = prepare_pca_data_test(pca, pca_scaler, eval_df, 
                                            budget=budget, buy_sell_fee = buy_sell_fee, commission_bps = commission_bps, label_type=label_type)

        X_train, y_train, y_logical_train, _ = build_feature_matrix(week_dates[0],
            train_df, X_train_pca, target_stock, n_pca_components, n_past_days, ohlcv=ohlcv, 
            budget=budget, buy_sell_fee = buy_sell_fee, commission_bps = commission_bps, profit=profit,  label_type=label_type,
        )
        
        assert X_train.index.max() <= expected_last_train_day
        
        X_test, y_test, y_logical_test, price_test = build_feature_matrix(week_dates[0],
            eval_df, X_test_pca, target_stock, n_pca_components, n_past_days, ohlcv=ohlcv,
            budget=budget, buy_sell_fee = buy_sell_fee, commission_bps = commission_bps, profit=profit, label_type=label_type, 
        )

        if walk_forward_flag:
            week_mask = X_test.index.isin(week_dates)
        else:
            week_mask = X_test.index>=week_dates[0]
        X_test = X_test[week_mask]
        y_test = y_test[week_mask]
        y_logical_test = y_logical_test[week_mask]
        price_test = price_test[week_mask]

        #print(X_train.shape)
        #print(X_test.shape)
        #print(X_train.tail())
        #print(X_test.head())
        #print(X_test.tail())
        

        if X_test.empty:
            continue

        
        assert X_train.index.max() < week_dates[0]
        if walk_forward_flag:
            assert X_test.index.min() >= week_dates[0] and X_test.index.max() <= week_dates[-1]
        else:
            assert X_test.index.min() >= week_dates[0] and X_test.index.max() <= full.index[-2]

    
        if walk_forward_flag:
            print(f"{target_stock} - count down: {week_counter}/{len(test_weeks)}")
        
        week_counter += 1

        
        if (label_type == "percentage") and (regressor_flag):

            print("\n--- Training Report (regressor) ---")

            # Regressor
            regressor, _, best_loss_regr, _ = regressor_train(X_train, y_train, val_frac=val_frac)

            y_pred_train = regressor.predict(X_train)
            print("MAE:", mean_absolute_error(y_train, y_pred_train))
            print("RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
            print("R²:", r2_score(y_train, y_pred_train))

        # Classifier
        classifier, best_epoch, best_val_loss, _ = classifier_train(X_train, y_logical_train, val_frac=val_frac)
        print(f"Best epoch: {best_epoch} | validation log loss: {best_val_loss:.4f}")

        # Training evaluation
        train_preds = classifier.predict(X_train)
        print("number 1s: ", np.sum(y_logical_train == 1))
        print("number 0s: ", np.sum(y_logical_train == 0))
        print(f"Accuracy: {accuracy_score(y_logical_train, train_preds):.4f}")
        print(classification_report(y_logical_train, train_preds, zero_division=0))

        # Test evaluation
        test_preds = classifier.predict(X_test)
        print("\n--- Test Report (classifier) ---")
        print("number 1s: ", np.sum(y_logical_test == 1))
        print("number 0s: ", np.sum(y_logical_test == 0))
        print(f"Accuracy: {accuracy_score(y_logical_test, test_preds):.4f}")
        print(classification_report(y_logical_test, test_preds, zero_division=0))

        if (label_type == "percentage") and (regressor_flag):
            y_pred_test = regressor.predict(X_test)
            print("\n--- Test Report (regressor) ---")
            print("MAE:", mean_absolute_error(y_test, y_pred_test))
            #test_error = np.sqrt(mean_squared_error(y_test, y_pred_test))
            print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
            print("R²:", r2_score(y_test, y_pred_test))

        if (label_type == "percentage") and (regressor_flag):
            # Store this week's predictions
            weekly_results.append(pd.DataFrame({
                "classe_predict_proba": classifier.predict_proba(X_test)[:, 1],
                "retorno_previsto": regressor.predict(X_test),#-best_loss_regr,
                "price_test": price_test,
                "val_error": best_loss_regr,
                "test_error": np.sqrt((y_test - y_pred_test)**2),
                "test_returns": y_test,
            }, index=X_test.index))
        else:
            # Store this week's predictions
            weekly_results.append(pd.DataFrame({
                "classe_predict_proba": classifier.predict_proba(X_test)[:, 1],
                "price_test": price_test,
            }, index=X_test.index))

        if not walk_forward_flag:
            break

    predictions = pd.concat(weekly_results).sort_index()
    
    return predictions
