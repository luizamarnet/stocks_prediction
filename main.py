"""Entry point: week-by-week walk-forward evaluation of the MLP strategy for every stock."""
import pandas as pd
import numpy as np

from src.stocks_prediction.backtest import backtest
from src.stocks_prediction.data_loader import (
    OHLCV_FIELDS,
    build_full_dataset,
    build_week_labels,
    load_ohlcv,
    load_raw_data,
)
from src.stocks_prediction.plotting import plot_graphs, plot_graphs_diff, plot_graphs_profit, plot_profit_boxplots
from src.stocks_prediction.walk_forward import run_walk_forward
import time 

# Output locations
RESULTS_PLOTS_DIR = "results/plots"
RESULTS_CODE_CSV = "./resultados_all_stocks_pnl_code.csv"
RESULTS_NO_CODE_CSV = "./resultados_all_stocks_pnl_no_code.csv"
RESULTS_CODE_PERCENT_CSV = "./resultados_all_stocks_pnl_code_percent.csv"
RESULTS_NO_CODE_PERCENT_CSV = "./resultados_all_stocks_pnl_no_code_percent.csv"


def main() -> None:
    """Run the full pipeline: walk-forward per stock, backtest, save results and plots."""
    # Strategy and trading parameters
    budget: float = 700
    commission_bps: float = 0.08
    buy_sell_fee: float = 2
    n_pca_components: int = 2
    n_past_days: int = 5

    profit: float =5.00
    label_type: str = "percentage"
    regressor_flag: bool = True

    if label_type not in ("profit", "percentage"):
        raise ValueError(
            f"Invalid label_type: {label_type!r}. "
            "Please, label_type must be either 'profit' or 'percentage'."
        )
    if label_type=="profit":
        regressor_flag=False

    

    # 1. Load the datasets
    train_raw, test_raw = load_raw_data()            # used for dates and stock names
    ohlcv = load_ohlcv(OHLCV_FIELDS)                 # open, high, low, close, volume
    full = build_full_dataset(train_raw, test_raw)   # close prices: the target series

    # Walk-forward parameters
    train_size: int = len(train_raw) #1500                   # training window length (trading days)
    warmup: int = n_past_days + 200 + 50      # extra history for SMA200, MACD, etc.

    # 2. Label each row with its ISO year-week
    week_labels = build_week_labels(full.index)

    # 3. Test weeks, in chronological order
    test_start = test_raw.index.min()
    test_weeks = pd.unique(week_labels[full.index >= test_start])

    total_profit_code = 0.0
    total_profit_no_code = 0.0

    results_all_pnl_code = pd.DataFrame()
    results_all_pnl_no_code = pd.DataFrame()
    results_all_pnl_code_percent = pd.DataFrame()
    results_all_pnl_no_code_percent = pd.DataFrame()

    for n_stocks_analyzed, target_stock in enumerate(train_raw.columns, start=1):
        
        

        # Walk-forward predictions for this stock
        predictions = run_walk_forward(
            full=full,
            week_labels=week_labels,
            test_weeks=test_weeks,
            target_stock=target_stock,
            n_pca_components=n_pca_components,
            n_past_days=n_past_days,
            train_size=train_size,
            warmup=warmup,
            ohlcv=ohlcv,

            budget=budget,
            buy_sell_fee=buy_sell_fee, 
            commission_bps=commission_bps,
            profit=profit,
            label_type=label_type,
            regressor_flag=regressor_flag,
        )
        
        week_positions = np.where(week_labels == test_weeks[0])[0]
        week_start, week_end = week_positions[0], week_positions[-1] + 1
        week_dates = full.index[week_start:week_end]
        print(len(predictions))
        print(len(full.loc[full.index>=week_dates[0]])-1)
        assert (len(predictions))==(len(full.loc[full.index>=week_dates[0]])-1)
        # Trade simulation and benchmark
        pnl_test, pnl_test_percent, pnl_test_no_code, pnl_test_no_code_percent = backtest(
            predictions, target_stock, budget, buy_sell_fee, commission_bps, profit=profit, label_type=label_type,
        )

        # Accumulate results across stocks (one column per stock)
        results_all_pnl_code = pd.concat(
            [results_all_pnl_code, pd.Series(pnl_test, index=predictions.index, name=target_stock)],
            axis=1,
        )
        results_all_pnl_no_code = pd.concat(
            [results_all_pnl_no_code, pd.Series(pnl_test_no_code, index=predictions.index, name=target_stock)],
            axis=1,
        )
        results_all_pnl_code_percent = pd.concat(
            [results_all_pnl_code_percent, pd.Series(pnl_test_percent, index=predictions.index, name=target_stock)],
            axis=1,
        )
        results_all_pnl_no_code_percent = pd.concat(
            [results_all_pnl_no_code_percent,
             pd.Series(pnl_test_no_code_percent, index=predictions.index, name=target_stock)],
            axis=1,
        )

        results_all_pnl_code.index.name = "Date"
        results_all_pnl_no_code.index.name = "Date"
        results_all_pnl_code_percent.index.name = "Date"
        results_all_pnl_no_code_percent.index.name = "Date"

        # Persist the accumulated results after every stock
        results_all_pnl_code.to_csv(RESULTS_CODE_CSV, index=True)
        results_all_pnl_no_code.to_csv(RESULTS_NO_CODE_CSV, index=True)
        results_all_pnl_code_percent.to_csv(RESULTS_CODE_PERCENT_CSV, index=True)
        results_all_pnl_no_code_percent.to_csv(RESULTS_NO_CODE_PERCENT_CSV, index=True)

        # Per-stock accumulated PnL plots (absolute and percentage)
        plot_graphs(target_stock, pnl_test, pnl_test_no_code,
                    percent=False, save_path=RESULTS_PLOTS_DIR, dates=predictions.index)
        plot_graphs(target_stock, pnl_test_percent, pnl_test_no_code_percent,
                    percent=True, save_path=RESULTS_PLOTS_DIR, dates=predictions.index)

        # Running totals
        total_profit_code += pnl_test[-1]
        total_profit_no_code += pnl_test_no_code[-1]
        print(f"Total profit code: {total_profit_code}")
        print(f"Total profit no code: {total_profit_no_code}")
        print(f"Total profit difference (code-no_code): {total_profit_code - total_profit_no_code}")
        print(f"Total profit code (percent): {total_profit_code / (budget * n_stocks_analyzed)}")
        print(f"Total profit no code (percent): {total_profit_no_code / (budget * n_stocks_analyzed)}")
        print(f"Total profit difference (code-no_code) (percent): "
              f"{(total_profit_code - total_profit_no_code) / (budget * n_stocks_analyzed)}")

        # 4. Reload the saved results and plot the cross-stock comparison
        df_pnl_code = pd.read_csv(RESULTS_CODE_CSV, index_col="Date", parse_dates=True)
        df_pnl_no_code = pd.read_csv(RESULTS_NO_CODE_CSV, index_col="Date", parse_dates=True)

        df_pnl_code_percent = pd.read_csv(RESULTS_CODE_PERCENT_CSV, index_col="Date", parse_dates=True)
        df_pnl_no_code_percent = pd.read_csv(RESULTS_NO_CODE_PERCENT_CSV, index_col="Date", parse_dates=True)

        plot_graphs_diff(df_pnl_code, df_pnl_no_code, percent=False, save_path=RESULTS_PLOTS_DIR)
        plot_graphs_diff(df_pnl_code_percent, df_pnl_no_code_percent, percent=True, save_path=RESULTS_PLOTS_DIR)

        plot_profit_boxplots(df_pnl_code, df_pnl_no_code, percent=False, save_path=RESULTS_PLOTS_DIR)
        plot_profit_boxplots(df_pnl_code_percent, df_pnl_no_code_percent, percent=True, save_path=RESULTS_PLOTS_DIR)

        plot_graphs_profit(df_pnl_code, df_pnl_no_code, percent=False, save_path=RESULTS_PLOTS_DIR)
        plot_graphs_profit(df_pnl_code_percent, df_pnl_no_code_percent, percent=True, save_path=RESULTS_PLOTS_DIR)


if __name__ == "__main__":
    main()
